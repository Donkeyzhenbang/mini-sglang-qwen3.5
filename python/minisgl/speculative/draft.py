"""Native PyTorch DFlash v1 inference adapter, without remote Python execution.

Algorithm/config reference: https://github.com/z-lab/dflash (Apache-2.0).
Only default RoPE and Qwen3-style DFlashDraftModel checkpoints are supported.
This deliberately readable SDPA implementation is a correctness baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight


def rotary(x, positions, theta):
    dim = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = positions.float()[:, None] * inv[None, :]
    angles = torch.cat([angles, angles], -1)[None, None]
    half = dim // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)


def visibility(query_positions, key_positions, causal, window):
    distance = query_positions[:, None] - key_positions[None, :]
    mask = torch.ones_like(distance, dtype=torch.bool)
    if causal:
        mask &= distance >= 0
    if window is not None:
        mask &= distance < window
        if not causal:
            mask &= distance > -window
    return mask[None, None]


class DraftAttention(nn.Module):
    def __init__(self, c, layer_id):
        super().__init__()
        self.head_dim = c["head_dim"]
        self.heads, self.kv_heads = c["num_attention_heads"], c["num_key_value_heads"]
        hidden, dim = c["hidden_size"], self.head_dim
        bias = c.get("attention_bias", False)
        self.q_proj = nn.Linear(hidden, self.heads * dim, bias=bias)
        self.k_proj = nn.Linear(hidden, self.kv_heads * dim, bias=bias)
        self.v_proj = nn.Linear(hidden, self.kv_heads * dim, bias=bias)
        self.o_proj = nn.Linear(self.heads * dim, hidden, bias=bias)
        self.q_norm = RMSNorm(dim, c["rms_norm_eps"])
        self.k_norm = RMSNorm(dim, c["rms_norm_eps"])
        sliding = (
            c.get("layer_types", ["full_attention"] * c["num_hidden_layers"])[layer_id]
            == "sliding_attention"
        )
        self.window = c.get("sliding_window") if sliding else None
        self.causal = c.get("is_causal", sliding)
        self.theta = c.get("rope_parameters", {}).get("rope_theta", c.get("rope_theta", 10000.0))
        self.cached_k = self.cached_v = None

    def reset(self):
        self.cached_k = self.cached_v = None

    def forward(self, x, context, previous_context, target_length):
        batch, count, _ = x.shape

        def split(t, heads):
            return t.view(batch, -1, heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(split(self.q_proj(x), self.heads))
        new = torch.cat([context, x], dim=1)
        k = self.k_norm(split(self.k_proj(new), self.kv_heads))
        v = split(self.v_proj(new), self.kv_heads)
        positions = torch.arange(previous_context, target_length + count, device=x.device)
        q = rotary(q, positions[-count:], self.theta)
        k = rotary(k, positions, self.theta)
        context_count = context.shape[1]
        new_context_k, new_context_v = k[..., :context_count, :], v[..., :context_count, :]
        if self.cached_k is not None:
            k = torch.cat([self.cached_k, k], dim=-2)
            v = torch.cat([self.cached_v, v], dim=-2)
            new_context_k = torch.cat([self.cached_k, new_context_k], dim=-2)
            new_context_v = torch.cat([self.cached_v, new_context_v], dim=-2)
        key_start = target_length + count - k.shape[-2]
        key_positions = torch.arange(key_start, target_length + count, device=x.device)
        mask = visibility(positions[-count:], key_positions, self.causal, self.window)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        # Retain confirmed context only, never the speculative noise block.
        if self.window:
            new_context_k = new_context_k[..., -self.window :, :]
            new_context_v = new_context_v[..., -self.window :, :]
        self.cached_k = new_context_k.contiguous().clone()
        self.cached_v = new_context_v.contiguous().clone()
        return self.o_proj(y.transpose(1, 2).reshape(batch, count, -1))


class DraftMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        h, i = c["hidden_size"], c["intermediate_size"]
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DraftLayer(nn.Module):
    def __init__(self, c, layer_id):
        super().__init__()
        self.self_attn = DraftAttention(c, layer_id)
        self.mlp = DraftMLP(c)
        self.input_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])

    def forward(self, x, context, previous, length):
        x = x + self.self_attn(self.input_layernorm(x), context, previous, length)
        return x + self.mlp(self.post_attention_layernorm(x))


class DFlashDraft(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if (
            config.get("architectures") != ["DFlashDraftModel"]
            or config.get("model_type") != "qwen3"
        ):
            raise ValueError("Only Qwen3-style DFlash v1 checkpoints are supported")
        rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Non-default draft RoPE is not implemented")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("Unsupported draft activation")
        d = config["dflash_config"]
        self.target_layer_ids = tuple(d["target_layer_ids"])
        self.block_size, self.mask_token_id = d["block_size"], d["mask_token_id"]
        self.layers = nn.ModuleList(
            [DraftLayer(config, i) for i in range(config["num_hidden_layers"])]
        )
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.fc = nn.Linear(len(self.target_layer_ids) * hidden, hidden, bias=False)
        self.hidden_norm = RMSNorm(hidden, eps)
        self.norm = RMSNorm(hidden, eps)
        self.context_length = 0

    @classmethod
    def from_directory(cls, folder, device, dtype):
        from safetensors import safe_open

        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        with torch.device("meta"):
            model = cls(config)
        weights = {}
        for file in sorted(folder.glob("*.safetensors")):
            with safe_open(file, framework="pt", device=str(device)) as f:
                for key in f.keys():
                    if key in weights:
                        raise ValueError(f"Duplicate checkpoint key: {key}")
                    weights[key] = f.get_tensor(key).to(dtype=dtype)
        model.load_state_dict(weights, strict=True, assign=True)
        return model.eval()

    def reset(self):
        self.context_length = 0
        for layer in self.layers:
            layer.self_attn.reset()

    def fork_context(self):
        """Share immutable weights, but give each request its own draft KV state."""
        with torch.device("meta"):
            model = type(self)(self.config)
        model.load_state_dict(self.state_dict(), strict=True, assign=True)
        return model.eval()

    def forward(self, context_features, noise_embeddings, target_length):
        if self.context_length + context_features.shape[1] != target_length:
            raise ValueError("Draft context must contain exactly the newly confirmed target states")
        context = self.hidden_norm(self.fc(context_features))
        x = noise_embeddings
        for layer in self.layers:
            x = layer(x, context, self.context_length, target_length)
        self.context_length = target_length
        return self.norm(x)

    @torch.inference_mode()
    def propose(self, features, anchor, block, target_length, embedding, head):
        if not 2 <= block <= self.block_size:
            raise ValueError("Requested block exceeds the checkpoint training block size")
        ids = torch.full((1, block), self.mask_token_id, dtype=torch.long, device=embedding.device)
        ids[0, 0] = anchor
        opts = self.config["dflash_config"]
        noise = F.embedding(ids, embedding) * opts.get("input_embedding_scale", 1.0)
        hidden = self(features, noise, target_length)
        logits = F.linear(hidden[:, 1:], head) * opts.get("output_multiplier", 1.0)
        cap = opts.get("final_logit_softcapping")
        if cap:
            logits = logits.tanh() if cap == 1 else (logits / cap).tanh() * cap
        return [int(anchor)] + logits.argmax(-1)[0].tolist()
