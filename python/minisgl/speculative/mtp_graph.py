"""CUDA graphs for the complete greedy MTP chain, with confirmed-only KV.

The first forward consumes confirmed target states. Recursive draft forwards
write only graph-local scratch KV; rejected proposals never enter persistent KV.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .draft_graph import store_context


class MTPGraphPool:
    def __init__(self, model, embedding, head, max_batch, max_context, stream):
        if stream == torch.cuda.default_stream(embedding.device):
            raise ValueError("MTP capture requires a non-default executor stream")
        self.model, self.embedding, self.head = model, embedding, head
        self.max_context, self.stream = max_context, stream
        attn = model.layer.self_attn
        shape = (max_batch, attn.kv_heads, max_context, attn.head_dim)
        self.keys = torch.zeros(shape, device=embedding.device, dtype=embedding.dtype)
        self.values = torch.zeros_like(self.keys)
        self.graphs = {}
        self.replays = self.fallbacks = 0

    def propose(self, items, slots):
        if not items or len(items) != len(slots):
            raise ValueError("MTP graph requires one slot per request")
        if len(set(slots)) != len(slots) or len({id(r[0]) for r in items}) != len(items):
            raise ValueError("MTP graph requires distinct slots and request contexts")
        if any(s < 0 or s >= self.keys.shape[0] for s in slots):
            raise ValueError("MTP graph slot out of range")
        for draft, hidden, tokens, block, length in items:
            if draft.fc.weight.data_ptr() != self.model.fc.weight.data_ptr():
                raise ValueError("MTP graph requests must share weights")
            if hidden.ndim == 3:
                if hidden.shape[0] != 1:
                    raise ValueError("MTP graph expects one request per hidden tensor")
                hidden = hidden[0]
            if not tokens or hidden.shape != (len(tokens), self.embedding.shape[1]):
                raise ValueError("MTP graph tokens and hidden states must align")
            if draft.context_length + len(tokens) != length:
                raise ValueError("MTP graph context length mismatch")
            if not 2 <= block <= draft.max_steps + 1 or length + block > self.max_context:
                raise ValueError("MTP graph exceeds step count or context capacity")
        blocks = {r[3] for r in items}
        count = max(len(r[2]) for r in items)
        # Long initial prefill and nonuniform tail blocks use the eager oracle.
        if len(blocks) != 1 or count > 16:
            self.fallbacks += 1
            return None
        width = 1 << (count - 1).bit_length()
        block = next(iter(blocks))
        old_width = min(
            self.max_context,
            max(256, math.ceil(max(r[4] + block for r in items) / 256) * 256),
        )
        shape = (len(items), block, width, old_width)
        if shape not in self.graphs and (
            len(self.graphs) >= 32 or torch.cuda.mem_get_info()[0] < 2 * 2**30
        ):
            self.fallbacks += 1
            return None
        for row, slot in zip(items, slots):
            draft = row[0]
            for source, dest in ((draft.cached_k, self.keys), (draft.cached_v, self.values)):
                if source is None:
                    if draft.context_length:
                        raise ValueError("Missing confirmed MTP cache")
                elif source.untyped_storage().data_ptr() != dest.untyped_storage().data_ptr():
                    if source.shape[-2] != draft.context_length:
                        raise ValueError("MTP cache length mismatch")
                    dest[slot, :, :draft.context_length].copy_(source[0])
        if shape not in self.graphs:
            self.graphs[shape] = _MTPGraph(self, shape, items, slots)
        predictions = self.graphs[shape].replay(items, slots)
        for row, slot in zip(items, slots):
            draft, _, _, _, length = row
            draft.cached_k = self.keys[slot:slot + 1, :, :length]
            draft.cached_v = self.values[slot:slot + 1, :, :length]
            draft.context_length = length
        self.replays += 1
        return [[row[2][-1]] + tokens for row, tokens in zip(items, predictions)]


class _MTPGraph:
    def __init__(self, pool, shape, items, slots):
        self.pool = pool
        self.batch, self.block, self.width, self.old_width = shape
        device = pool.embedding.device
        self.ids = torch.zeros(self.batch, self.width, device=device, dtype=torch.long)
        self.hidden = torch.zeros(
            self.batch, self.width, pool.embedding.shape[1],
            device=device, dtype=pool.embedding.dtype,
        )
        self.metadata = torch.zeros(4, self.batch, device=device, dtype=torch.long)
        self.slots, self.previous, self.counts, self.lengths = self.metadata.unbind()
        self._upload(items, slots)
        # Repeating capture writes the same confirmed KV. The previous-length
        # mask excludes those positions from its inputs, including after reset.
        for _ in range(2):
            self._body()
        pool.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=pool.stream):
            self._body()

    def _upload(self, items, slots):
        self.hidden.zero_()
        padded_ids = []
        for i, (_, hidden, tokens, _, _) in enumerate(items):
            self.hidden[i, :len(tokens)].copy_(hidden.reshape(-1, self.hidden.shape[-1]))
            padded_ids.append(list(tokens) + [0] * (self.width - len(tokens)))
        self.ids.copy_(torch.tensor(padded_ids, device=self.ids.device))
        self.metadata.copy_(torch.tensor(
            [slots, [r[0].context_length for r in items],
             [len(r[2]) for r in items], [r[4] for r in items]],
            device=self.metadata.device,
        ))

    def _forward(self, ids, hidden, positions, old_k, old_v, valid, new_valid):
        model = self.pool.model
        x = model.fc(torch.cat([
            model.pre_fc_norm_embedding(F.embedding(ids, self.pool.embedding)),
            model.pre_fc_norm_hidden(hidden),
        ], dim=-1))
        x, k, v = model.layer(x, positions, old_k, old_v, valid, new_valid)
        return model.norm(x), k, v

    def _body(self):
        pool, device = self.pool, self.ids.device
        rows = torch.arange(self.batch, device=device)
        offsets = torch.arange(self.width, device=device)
        old_offsets = torch.arange(self.old_width, device=device)
        valid = offsets < self.counts[:, None]
        positions = (self.previous[:, None] + offsets).clamp(max=pool.max_context - 1)
        old_k = pool.keys[:, :, :self.old_width].index_select(0, self.slots)
        old_v = pool.values[:, :, :self.old_width].index_select(0, self.slots)
        x, k, v = self._forward(
            self.ids, self.hidden, positions, old_k, old_v,
            old_offsets < self.previous[:, None], valid,
        )
        last = x[rows, self.counts - 1]
        tokens = [F.linear(last, pool.head).argmax(-1)]
        store_context(k, v, pool.keys, pool.values, self.slots, self.previous, self.counts)
        # These tensors are private graph scratch. Only confirmed KV was written
        # above; speculative recursion below must never modify pool.keys/values.
        scratch_k = pool.keys[:, :, :self.old_width].index_select(0, self.slots)
        scratch_v = pool.values[:, :, :self.old_width].index_select(0, self.slots)
        ones = torch.ones(self.batch, device=device, dtype=torch.long)
        for step in range(self.block - 2):
            length = self.lengths + step
            x, k, v = self._forward(
                tokens[-1][:, None], last[:, None], length[:, None],
                scratch_k, scratch_v, old_offsets < length[:, None],
                ones[:, None].bool(),
            )
            last = x[:, 0]
            tokens.append(F.linear(last, pool.head).argmax(-1))
            store_context(k, v, scratch_k, scratch_v, rows, length, ones)
        self.tokens = torch.stack(tokens, dim=1)

    def replay(self, items, slots):
        self._upload(items, slots)
        self.graph.replay()
        return self.tokens.tolist()
