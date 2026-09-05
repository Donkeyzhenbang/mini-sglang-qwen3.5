"""Minimal native Qwen3.5 MTP-1/MTP-3 greedy draft model.

The top-k=1 chain follows SGLang's EAGLE worker: confirmed target hidden states
are paired with one-token-shifted embeddings, then the shared MTP layer is
recurrently applied for each additional draft step.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class GemmaRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from flashinfer import gemma_rmsnorm

            flat = x.reshape(-1, x.shape[-1])
            return gemma_rmsnorm(flat, self.weight, self.eps).view_as(x)
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(dtype)


def _apply_partial_rope(
    x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, theta: float
) -> torch.Tensor:
    if rotary_dim == 0:
        return x
    inv = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32)
            / rotary_dim
        )
    )
    angles = positions.float()[..., None] * inv
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    rotated, tail = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotated.chunk(2, dim=-1)
    rotated = torch.cat(
        [
            first * cos[:, None] - second * sin[:, None],
            second * cos[:, None] + first * sin[:, None],
        ],
        dim=-1,
    )
    return torch.cat([rotated, tail], dim=-1)


class MTPAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.q_size = 2 * self.heads * self.head_dim
        self.kv_size = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            hidden, self.q_size + 2 * self.kv_size, bias=False
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, config["rms_norm_eps"])
        self.k_norm = GemmaRMSNorm(self.head_dim, config["rms_norm_eps"])
        self.o_proj = nn.Linear(
            self.heads * self.head_dim, hidden, bias=False
        )
        rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
        self.theta = rope.get("rope_theta", config.get("rope_theta", 10000.0))
        factor = rope.get(
            "partial_rotary_factor", config.get("partial_rotary_factor", 1.0)
        )
        self.rotary_dim = int(self.head_dim * factor)
        self.rotary_dim -= self.rotary_dim % 2
        self.rope_cache: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        old_k: torch.Tensor,
        old_v: torch.Tensor,
        old_valid: torch.Tensor,
        new_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, width, _ = x.shape
        qkv = self.qkv_proj(x)
        qg, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        qg = qg.view(batch, width, self.heads, 2 * self.head_dim)
        q, gate = qg.chunk(2, dim=-1)
        q = self.q_norm(q)
        gate = gate.reshape(batch, width, -1)
        k = self.k_norm(k.view(batch, width, self.kv_heads, self.head_dim))
        v = v.view(batch, width, self.kv_heads, self.head_dim).transpose(1, 2)
        if q.is_cuda and self.rope_cache is not None:
            from flashinfer import apply_rope_with_cos_sin_cache_inplace

            flat_q = q.reshape(batch * width, -1).contiguous()
            flat_k = k.reshape(batch * width, -1).contiguous()
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions.reshape(-1),
                query=flat_q,
                key=flat_k,
                head_size=self.head_dim,
                cos_sin_cache=self.rope_cache,
            )
            q = flat_q.view(batch, width, self.heads, self.head_dim).transpose(1, 2)
            k = flat_k.view(batch, width, self.kv_heads, self.head_dim).transpose(1, 2)
        else:
            q = _apply_partial_rope(
                q.transpose(1, 2), positions, self.rotary_dim, self.theta
            )
            k = _apply_partial_rope(
                k.transpose(1, 2), positions, self.rotary_dim, self.theta
            )

        old_width = old_k.shape[-2]
        keys = torch.cat([old_k, k], dim=-2)
        values = torch.cat([old_v, v], dim=-2)
        old_offsets = torch.arange(old_width, device=x.device)
        key_positions = torch.cat(
            [old_offsets.expand(batch, -1), positions],
            dim=1,
        )
        valid = torch.cat([old_valid, new_valid], dim=1)
        mask = valid[:, None, :] & (
            positions[:, :, None] >= key_positions[:, None, :]
        )
        out = F.scaled_dot_product_attention(
            q, keys, values, attn_mask=mask[:, None], enable_gqa=True
        )
        out = out.transpose(1, 2).reshape(batch, width, -1)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out), k, v


class MTPMLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        self.intermediate_size = intermediate
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        if gate_up.is_cuda:
            from flashinfer import silu_and_mul

            flat = gate_up.reshape(-1, gate_up.shape[-1])
            activated = silu_and_mul(flat).view(*gate_up.shape[:-1], -1)
        else:
            gate, up = gate_up.split(self.intermediate_size, dim=-1)
            activated = F.silu(gate) * up
        return self.down_proj(activated)


class MTPLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.self_attn = MTPAttention(config)
        self.mlp = MTPMLP(config)
        self.input_layernorm = GemmaRMSNorm(hidden, eps)
        self.post_attention_layernorm = GemmaRMSNorm(hidden, eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        old_k: torch.Tensor,
        old_v: torch.Tensor,
        old_valid: torch.Tensor,
        new_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        attn, new_k, new_v = self.self_attn(
            self.input_layernorm(x),
            positions,
            old_k,
            old_v,
            old_valid,
            new_valid,
        )
        residual = residual + attn
        x = residual + self.mlp(self.post_attention_layernorm(residual))
        return x, new_k, new_v


class Qwen3_5MTPDraft(nn.Module):
    draft_type = "mtp"

    def __init__(self, config: dict, max_steps: int = 3):
        super().__init__()
        if config.get("mtp_num_hidden_layers", 0) != 1:
            raise ValueError("Only one-layer Qwen3.5 MTP checkpoints are supported")
        if config.get("num_experts", 0):
            raise ValueError("Minimal native MTP supports dense Qwen3.5 only")
        if config.get("mtp_use_dedicated_embeddings", False):
            raise ValueError("Dedicated MTP embeddings are not supported")
        if config.get("attention_bias", False):
            raise ValueError("MTP attention bias is not supported")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("Only SiLU Qwen3.5 MTP is supported")
        rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Only default Qwen3.5 MTP RoPE is supported")
        if max_steps not in (1, 3):
            raise ValueError("Minimal native MTP supports one or three steps")
        self.config, self.max_steps = config, max_steps
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.pre_fc_norm_embedding = GemmaRMSNorm(hidden, eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(hidden, eps)
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.layer = MTPLayer(config)
        self.norm = GemmaRMSNorm(hidden, eps)
        self.cached_k = self.cached_v = None
        self.context_length = 0

    def reset(self):
        self.cached_k = self.cached_v = None
        self.context_length = 0

    def fork_context(self):
        with torch.device("meta"):
            model = type(self)(self.config, self.max_steps)
        model.load_state_dict(self.state_dict(), strict=True, assign=True)
        model.layer.self_attn.rope_cache = self.layer.self_attn.rope_cache
        return model.eval()

    @classmethod
    def from_directory(
        cls,
        folder,
        device,
        dtype,
        *,
        max_steps: int = 3,
        max_position: int | None = None,
    ) -> "Qwen3_5MTPDraft":
        from safetensors import safe_open

        folder = Path(folder)
        raw_config = json.loads((folder / "config.json").read_text())
        config = raw_config.get("text_config", raw_config)
        with torch.device("meta"):
            model = cls(config, max_steps=max_steps)
        weights = {}
        for file in sorted(folder.glob("*.safetensors")):
            with safe_open(file, framework="pt", device=str(device)) as source:
                for key in source.keys():
                    if not key.startswith("mtp."):
                        continue
                    name = key.removeprefix("mtp.")
                    name = name.replace("layers.0.", "layer.", 1)
                    if name in weights:
                        raise ValueError(f"Duplicate MTP checkpoint key: {key}")
                    weights[name] = source.get_tensor(key).to(dtype=dtype)
        attn = "layer.self_attn."
        weights[attn + "qkv_proj.weight"] = torch.cat(
            [
                weights.pop(attn + f"{name}_proj.weight")
                for name in ("q", "k", "v")
            ],
            dim=0,
        )
        mlp = "layer.mlp."
        weights[mlp + "gate_up_proj.weight"] = torch.cat(
            [
                weights.pop(mlp + f"{name}_proj.weight")
                for name in ("gate", "up")
            ],
            dim=0,
        )
        model.load_state_dict(weights, strict=True, assign=True)
        attn = model.layer.self_attn
        if torch.device(device).type == "cuda":
            positions = torch.arange(
                max_position or config.get("max_position_embeddings", 4096),
                device=device,
                dtype=torch.float32,
            )
            inv = 1.0 / (
                attn.theta
                ** (
                    torch.arange(
                        0, attn.rotary_dim, 2, device=device, dtype=torch.float32
                    )
                    / attn.rotary_dim
                )
            )
            angles = positions[:, None] * inv
            attn.rope_cache = torch.cat([angles.cos(), angles.sin()], dim=-1)
        return model.eval()

    def _forward_batch(
        self,
        drafts: list["Qwen3_5MTPDraft"],
        token_rows: list[list[int]],
        hidden_rows: list[torch.Tensor],
        embedding: torch.Tensor,
        head: torch.Tensor,
        *,
        caches: list[
            tuple[torch.Tensor | None, torch.Tensor | None]
        ] | None = None,
        commit: bool,
    ):
        device, dtype = embedding.device, embedding.dtype
        counts = [len(tokens) for tokens in token_rows]
        if any(
            n == 0 or hidden.shape[0] != n
            for n, hidden in zip(counts, hidden_rows)
        ):
            raise ValueError("MTP tokens and target hidden states must align")
        width, batch = max(counts), len(drafts)
        ids = torch.zeros(batch, width, dtype=torch.long, device=device)
        hidden = torch.zeros(
            batch,
            width,
            self.config["hidden_size"],
            dtype=dtype,
            device=device,
        )
        new_valid = torch.zeros(
            batch, width, dtype=torch.bool, device=device
        )
        for index, (tokens, states) in enumerate(zip(token_rows, hidden_rows)):
            count = len(tokens)
            ids[index, :count] = torch.tensor(tokens, device=device)
            hidden[index, :count] = states.to(device=device, dtype=dtype)
            new_valid[index, :count] = True

        cache_rows = caches or [
            (draft.cached_k, draft.cached_v) for draft in drafts
        ]
        old_counts = [
            0 if key is None else key.shape[-2] for key, _ in cache_rows
        ]
        old_width = max(old_counts)
        attn = self.layer.self_attn
        old_k = torch.zeros(
            batch,
            attn.kv_heads,
            old_width,
            attn.head_dim,
            dtype=dtype,
            device=device,
        )
        old_v = torch.zeros_like(old_k)
        old_valid = torch.zeros(
            batch, old_width, dtype=torch.bool, device=device
        )
        for index, ((key, value), count) in enumerate(
            zip(cache_rows, old_counts)
        ):
            if count:
                old_k[index, :, :count] = key[0]
                old_v[index, :, :count] = value[0]
                old_valid[index, :count] = True
        offsets = torch.arange(width, device=device)
        positions = torch.tensor(old_counts, device=device)[:, None] + offsets

        embedded = F.embedding(ids, embedding)
        x = self.fc(
            torch.cat(
                [
                    self.pre_fc_norm_embedding(embedded),
                    self.pre_fc_norm_hidden(hidden),
                ],
                dim=-1,
            )
        )
        x, new_k, new_v = self.layer(
            x, positions, old_k, old_v, old_valid, new_valid
        )
        x = self.norm(x)
        last = torch.stack(
            [x[i, count - 1] for i, count in enumerate(counts)]
        )
        logits = F.linear(last, head).float()
        next_caches = []
        for index, (old_count, count) in enumerate(zip(old_counts, counts)):
            key = torch.cat(
                [
                    old_k[index : index + 1, :, :old_count],
                    new_k[index : index + 1, :, :count],
                ],
                dim=-2,
            ).contiguous()
            value = torch.cat(
                [
                    old_v[index : index + 1, :, :old_count],
                    new_v[index : index + 1, :, :count],
                ],
                dim=-2,
            ).contiguous()
            next_caches.append((key, value))
            if commit:
                drafts[index].cached_k, drafts[index].cached_v = key, value
                drafts[index].context_length += count
        return logits, last, next_caches

    @torch.inference_mode()
    def propose(
        self,
        target_hidden: torch.Tensor,
        next_tokens: list[int],
        block: int,
        target_length: int,
        embedding: torch.Tensor,
        head: torch.Tensor,
    ) -> list[int]:
        return propose_mtp_batch(
            [(self, target_hidden, next_tokens, block, target_length)],
            embedding,
            head,
        )[0]


@torch.inference_mode()
def propose_mtp_batch(items, embedding: torch.Tensor, head: torch.Tensor):
    if not items:
        return []
    model = items[0][0]
    if len({id(item[0]) for item in items}) != len(items):
        raise ValueError("MTP batch needs distinct request contexts")
    drafts, hidden_rows, token_rows, blocks = [], [], [], []
    for draft, hidden, tokens, block, target_length in items:
        if (
            draft.config != model.config
            or draft.fc.weight.data_ptr() != model.fc.weight.data_ptr()
        ):
            raise ValueError("MTP requests must share model weights")
        if not 2 <= block <= draft.max_steps + 1:
            raise ValueError("MTP block exceeds configured step count")
        if draft.context_length + len(tokens) != target_length:
            raise ValueError(
                "MTP context must contain newly confirmed target states"
            )
        drafts.append(draft)
        hidden_rows.append(hidden.squeeze(0) if hidden.ndim == 3 else hidden)
        token_rows.append(list(tokens))
        blocks.append(block)

    logits, last_hidden, caches = model._forward_batch(
        drafts,
        token_rows,
        hidden_rows,
        embedding,
        head,
        commit=True,
    )
    proposals = [
        [tokens[-1], int(token)]
        for tokens, token in zip(token_rows, logits.argmax(-1).tolist())
    ]
    for step in range(1, max(blocks) - 1):
        active = [
            index for index, block in enumerate(blocks)
            if step < block - 1
        ]
        active_drafts = [drafts[index] for index in active]
        active_tokens = [[proposals[index][-1]] for index in active]
        active_hidden = [last_hidden[index : index + 1] for index in active]
        active_caches = [caches[index] for index in active]
        logits, hidden, new_caches = model._forward_batch(
            active_drafts,
            active_tokens,
            active_hidden,
            embedding,
            head,
            caches=active_caches,
            commit=False,
        )
        predictions = logits.argmax(-1).tolist()
        for row, index in enumerate(active):
            proposals[index].append(predictions[row])
            last_hidden[index] = hidden[row]
            caches[index] = new_caches[row]
    return proposals
