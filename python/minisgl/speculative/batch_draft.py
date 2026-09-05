"""One native DFlash forward for ragged requests with independent context caches.

Only padding/packing and cache ownership use Python loops. FC, QKV, attention,
MLP and the vocabulary projection run once per layer for the whole batch.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .draft import rotary


@torch.inference_mode()
def propose_batch(items, embedding, head):
    """Items are (draft context, new features [1,N,H], anchor, block, target length)."""
    if not items or len({id(row[0]) for row in items}) != len(items):
        raise ValueError("A draft batch needs distinct request contexts")
    model = items[0][0]
    for draft, features, _, block, length in items:
        if draft.config != model.config or draft.fc.weight.data_ptr() != model.fc.weight.data_ptr():
            raise ValueError("Draft requests must share model weights")
        if not 2 <= block <= model.block_size:
            raise ValueError("Requested block exceeds checkpoint training block size")
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError("Expected one request per feature tensor")
        if draft.context_length + features.shape[1] != length:
            raise ValueError("Draft context must contain exactly newly confirmed states")
    if len(items) == 1:
        draft, features, anchor, block, length = items[0]
        return [draft.propose(features, anchor, block, length, embedding, head)]
    device = embedding.device
    count = len(items)
    blocks = [row[3] for row in items]
    width = max(blocks)
    new_counts = [row[1].shape[1] for row in items]
    context = pad_sequence([row[1][0] for row in items], batch_first=True)
    context = model.hidden_norm(model.fc(context))
    context_kv = model.project_context_kv(context)
    new_width = context.shape[1]
    ids = torch.full((count, width), model.mask_token_id, dtype=torch.long, device=device)
    ids[:, 0] = torch.tensor([row[2] for row in items], device=device)
    opts = model.config["dflash_config"]
    x = F.embedding(ids, embedding) * opts.get("input_embedding_scale", 1.0)
    lengths = torch.tensor([row[4] for row in items], device=device)
    previous = torch.tensor([row[0].context_length for row in items], device=device)
    new_n = torch.tensor(new_counts, device=device)
    block_n = torch.tensor(blocks, device=device)
    ctx_offsets = torch.arange(new_width, device=device)
    noise_offsets = torch.arange(width, device=device)
    q_positions = lengths[:, None] + noise_offsets
    new_positions = torch.cat([previous[:, None] + ctx_offsets, q_positions], dim=1)
    new_valid = torch.cat([ctx_offsets < new_n[:, None], noise_offsets < block_n[:, None]], dim=1)
    for layer_id, layer in enumerate(model.layers):
        attn = layer.self_attn
        states = [row[0].layers[layer_id].self_attn for row in items]
        residual = x
        normalized = layer.input_layernorm(x)

        def split(t, heads):
            return t.view(count, -1, heads, attn.head_dim).transpose(1, 2)

        query_qkv = attn.qkv_proj(normalized)
        q, query_k, query_v = query_qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
        )
        layer_context_kv = (
            context_kv[..., layer_id, :] if context_kv is not None else None
        )
        if layer_context_kv is None:
            layer_context_kv = F.linear(
                context,
                attn.qkv_proj.weight[attn.q_size :],
                attn.qkv_proj.bias[attn.q_size :]
                if attn.qkv_proj.bias is not None
                else None,
            )
        context_k, context_v = layer_context_kv.split(
            [attn.kv_size, attn.kv_size], dim=-1
        )
        q = attn.q_norm(split(q, attn.heads))
        k = attn.k_norm(
            split(torch.cat([context_k, query_k], dim=1), attn.kv_heads)
        )
        v = split(torch.cat([context_v, query_v], dim=1), attn.kv_heads)
        q = rotary(q, q_positions, attn.theta, attn.rope_cache)
        k = rotary(k, new_positions, attn.theta, attn.rope_cache)
        old_counts = [s.cached_k.shape[-2] if s.cached_k is not None else 0 for s in states]
        old_width = max(old_counts)
        old_n = torch.tensor(old_counts, device=device)
        old_offsets = torch.arange(old_width, device=device)
        old_k = k.new_zeros(count, attn.kv_heads, old_width, attn.head_dim)
        old_v = torch.zeros_like(old_k)
        for i, (state, n) in enumerate(zip(states, old_counts)):
            if n:
                old_k[i, :, :n] = state.cached_k[0]
                old_v[i, :, :n] = state.cached_v[0]
        keys, values = torch.cat([old_k, k], dim=-2), torch.cat([old_v, v], dim=-2)
        key_positions = torch.cat([(previous - old_n)[:, None] + old_offsets, new_positions], dim=1)
        valid = torch.cat([old_offsets < old_n[:, None], new_valid], dim=1)
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
        x = residual + attn.o_proj(y.transpose(1, 2).reshape(count, width, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        # Confirmed context only; padding and speculative noise must never persist.
        for i, (state, old_count, new_count) in enumerate(zip(states, old_counts, new_counts)):
            ck = torch.cat([old_k[i : i + 1, :, :old_count], k[i : i + 1, :, :new_count]], dim=-2)
            cv = torch.cat([old_v[i : i + 1, :, :old_count], v[i : i + 1, :, :new_count]], dim=-2)
            if attn.window:
                ck, cv = ck[..., -attn.window :, :], cv[..., -attn.window :, :]
            state.cached_k, state.cached_v = ck.contiguous(), cv.contiguous()
    hidden = model.norm(x)
    logits = F.linear(hidden[:, 1:], head) * opts.get("output_multiplier", 1.0)
    cap = opts.get("final_logit_softcapping")
    if cap:
        logits = (logits / cap).tanh() * cap
    tokens = logits.argmax(-1).tolist()  # One host transfer for all requests.
    for draft, _, _, _, length in items:
        draft.context_length = length
    return [[int(row[2])] + tokens[i][: row[3] - 1] for i, row in enumerate(items)]
