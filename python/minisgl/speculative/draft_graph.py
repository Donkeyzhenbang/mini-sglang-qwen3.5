"""Batched DFlash CUDA graphs backed by persistent confirmed-context KV storage."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from .draft import rotary


@triton.jit
def _store_context(
    K,
    V,
    KC,
    VC,
    SLOTS,
    PREVIOUS,
    COUNTS,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    CAPACITY: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(1)
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    dim = offset % D
    token = offset // D % N
    head = offset // (D * N)
    count, previous = tl.load(COUNTS + row), tl.load(PREVIOUS + row)
    slot = tl.load(SLOTS + row)
    valid = (head < H) & (token < count)
    src = row * H * N * D + offset
    dst = ((slot * H + head) * CAPACITY + previous + token) * D + dim
    tl.store(KC + dst, tl.load(K + src, mask=valid, other=0), mask=valid)
    tl.store(VC + dst, tl.load(V + src, mask=valid, other=0), mask=valid)


def store_context(k, v, kc, vc, slots, previous, counts):
    batch, heads, count, dim = k.shape
    _store_context[(triton.cdiv(heads * count * dim, 256), batch)](
        k.contiguous(),
        v.contiguous(),
        kc,
        vc,
        slots,
        previous,
        counts,
        H=heads,
        D=dim,
        N=count,
        CAPACITY=kc.shape[-2],
        TILE=256,
    )


class DFlashGraphPool:
    def __init__(self, model, embedding, head, max_batch, max_context, stream):
        if stream == torch.cuda.default_stream(embedding.device):
            raise ValueError("DFlash graph capture requires the executor's non-default CUDA stream")
        self.model, self.embedding, self.head = model, embedding, head
        self.max_context, self.stream = max_context, stream
        attn = model.layers[0].self_attn
        shape = (len(model.layers), max_batch, attn.kv_heads, max_context, attn.head_dim)
        self.keys = torch.zeros(shape, device=embedding.device, dtype=embedding.dtype)
        self.values = torch.zeros_like(self.keys)
        self.graphs = {}
        self.replays = 0
        self.fallbacks = 0

    def propose(self, items, slots):
        model = self.model
        for draft, features, _, block, length in items:
            if draft.fc.weight.data_ptr() != model.fc.weight.data_ptr():
                raise ValueError("DFlash graph contexts must share weights")
            if features.ndim != 3 or features.shape[0] != 1:
                raise ValueError("DFlash graph requires one request per feature tensor")
            if draft.context_length + features.shape[1] != length:
                raise ValueError("DFlash graph context length mismatch")
            if not 2 <= block <= model.block_size or length + block > self.max_context:
                raise ValueError("DFlash graph exceeds checkpoint or cache capacity")
        blocks = {row[3] for row in items}
        new_count = max(row[1].shape[1] for row in items)
        if len(blocks) != 1 or new_count > 16:
            self.fallbacks += 1
            return None
        width = next(iter(blocks))
        new_width = max(width, 1 << (new_count - 1).bit_length())
        old_width = min(
            self.max_context,
            max(256, math.ceil(max(row[0].context_length for row in items) / 256) * 256),
        )
        key = (len(items), width, new_width, old_width)
        if key not in self.graphs and len(self.graphs) >= 16:
            self.fallbacks += 1
            return None

        # Eager prefill/fallback may have materialized an independent cache. Import
        # it once; graph requests thereafter retain views into our stable storage.
        for row, slot in zip(items, slots):
            draft = row[0]
            for lid, layer in enumerate(draft.layers):
                attn = layer.self_attn
                for source, dest in ((attn.cached_k, self.keys), (attn.cached_v, self.values)):
                    if source is None:
                        if draft.context_length:
                            raise ValueError("Missing confirmed DFlash cache")
                        continue
                    if source.untyped_storage().data_ptr() != dest.untyped_storage().data_ptr():
                        count = source.shape[-2]
                        dest[
                            lid, slot, :, draft.context_length - count : draft.context_length
                        ].copy_(source[0])
        if key not in self.graphs:
            self.graphs[key] = _DFlashGraph(self, key, items, slots)
        tokens = self.graphs[key].replay(items, slots)
        for row, slot in zip(items, slots):
            draft, _, _, _, length = row
            draft.context_length = length
            for lid, layer in enumerate(draft.layers):
                attn = layer.self_attn
                start = max(0, length - attn.window) if attn.window else 0
                attn.cached_k = self.keys[lid, slot : slot + 1, :, start:length]
                attn.cached_v = self.values[lid, slot : slot + 1, :, start:length]
        self.replays += 1
        return [[int(row[2])] + token for row, token in zip(items, tokens)]


class _DFlashGraph:
    def __init__(self, pool, shape, items, slots):
        self.pool = pool
        self.batch, self.width, self.new_width, self.old_width = shape
        model, device = pool.model, pool.embedding.device
        self.features = torch.zeros(
            self.batch,
            self.new_width,
            model.fc.in_features,
            device=device,
            dtype=pool.embedding.dtype,
        )
        self.metadata = torch.zeros(5, self.batch, dtype=torch.int64, device=device)
        self.slots, self.previous, self.counts, self.lengths, self.anchors = self.metadata.unbind()
        self._upload(items, slots)
        # All writes are to newly confirmed positions; repeating the warmup stores
        # identical values. No speculative noise or target state is persisted.
        for _ in range(2):
            self._body()
        pool.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=pool.stream):
            self._body()

    def _upload(self, items, slots):
        self.features.zero_()
        for i, row in enumerate(items):
            self.features[i, : row[1].shape[1]].copy_(row[1][0])
        self.metadata.copy_(
            torch.tensor(
                [
                    slots,
                    [row[0].context_length for row in items],
                    [row[1].shape[1] for row in items],
                    [row[4] for row in items],
                    [row[2] for row in items],
                ],
                device=self.metadata.device,
                dtype=torch.int64,
            )
        )

    def _body(self):
        pool, batch, width = self.pool, self.batch, self.width
        model, device = pool.model, pool.embedding.device
        context = model.hidden_norm(model.fc(self.features))
        context_kv = model.project_context_kv(context)
        ids = torch.full((batch, width), model.mask_token_id, device=device, dtype=torch.int64)
        ids[:, 0] = self.anchors
        opts = model.config["dflash_config"]
        x = F.embedding(ids, pool.embedding) * opts.get("input_embedding_scale", 1.0)
        offsets = torch.arange(self.new_width, device=device)
        old_offsets = torch.arange(self.old_width, device=device)
        q_positions = self.lengths[:, None] + torch.arange(width, device=device)
        new_positions = torch.cat([self.previous[:, None] + offsets, q_positions], dim=1)
        new_valid = torch.cat(
            [
                offsets < self.counts[:, None],
                torch.ones(batch, width, device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        for lid, layer in enumerate(model.layers):
            attn = layer.self_attn

            def split(t, heads):
                return t.view(batch, -1, heads, attn.head_dim).transpose(1, 2)

            residual = x
            normalized = layer.input_layernorm(x)
            q, query_k, query_v = attn.qkv_proj(normalized).split(
                [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
            )
            layer_context_kv = (
                context_kv[..., lid, :]
                if context_kv is not None
                else F.linear(
                    context,
                    attn.qkv_proj.weight[attn.q_size :],
                    attn.qkv_proj.bias[attn.q_size :] if attn.qkv_proj.bias is not None else None,
                )
            )
            context_k, context_v = layer_context_kv.split(attn.kv_size, dim=-1)
            q = rotary(attn.q_norm(split(q, attn.heads)), q_positions, attn.theta, attn.rope_cache)
            k = rotary(
                attn.k_norm(split(torch.cat([context_k, query_k], 1), attn.kv_heads)),
                new_positions,
                attn.theta,
                attn.rope_cache,
            )
            v = split(torch.cat([context_v, query_v], 1), attn.kv_heads)
            old_k = pool.keys[lid, :, :, : self.old_width].index_select(0, self.slots)
            old_v = pool.values[lid, :, :, : self.old_width].index_select(0, self.slots)
            keys, values = torch.cat([old_k, k], -2), torch.cat([old_v, v], -2)
            old_valid = old_offsets < self.previous[:, None]
            if attn.window:
                old_valid = old_valid & (old_offsets >= self.previous[:, None] - attn.window)
            key_positions = torch.cat([old_offsets.expand(batch, -1), new_positions], dim=1)
            valid = torch.cat([old_valid, new_valid], dim=1)
            distance = q_positions[:, :, None] - key_positions[:, None, :]
            mask = valid[:, None, :].expand(-1, width, -1)
            if attn.causal:
                mask = mask & (distance >= 0)
            if attn.window is not None:
                mask = mask & (distance < attn.window)
                if not attn.causal:
                    mask = mask & (distance > -attn.window)
            y = F.scaled_dot_product_attention(
                q, keys, values, attn_mask=mask[:, None], enable_gqa=True
            )
            x = residual + attn.o_proj(y.transpose(1, 2).reshape(batch, width, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            store_context(
                k[:, :, : self.new_width],
                v[:, :, : self.new_width],
                pool.keys[lid],
                pool.values[lid],
                self.slots,
                self.previous,
                self.counts,
            )
        hidden = model.norm(x)
        logits = F.linear(hidden[:, 1:], pool.head) * opts.get("output_multiplier", 1.0)
        cap = opts.get("final_logit_softcapping")
        if cap:
            logits = (logits / cap).tanh() * cap
        self.tokens = logits.argmax(-1)

    def replay(self, items, slots):
        self._upload(items, slots)
        self.graph.replay()
        return self.tokens.tolist()
