"""Paged causal attention with a fixed per-query reduction, including graphs."""

from dataclasses import dataclass

import torch
from minisgl.core import get_global_ctx
from minisgl.kernel.triton.invariant import invariant_attention

from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class InvariantMetadata(BaseAttnMetadata):
    slots: torch.Tensor
    cu_seqlens_q_gpu: torch.Tensor

    def get_last_indices(self, bs):
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class InvariantAttentionBackend(BaseAttnBackend):
    def __init__(self):
        self.kvcache = get_global_ctx().kv_cache
        self.page_table = get_global_ctx().page_table
        self.capture = {}

    def prepare_metadata(self, batch):
        reqs = batch.padded_reqs
        cu, slots = [0], []
        for req in reqs:
            slots.extend([req.table_idx] * req.extend_len)
            cu.append(cu[-1] + req.extend_len)
        kw = dict(device=self.kvcache.device, dtype=torch.int32)
        batch.attn_metadata = InvariantMetadata(torch.tensor(slots, **kw), torch.tensor(cu, **kw))

    def forward(
        self, q=None, k=None, v=None, layer=None, forward_batch=None, save_kv_cache=True, **kwargs
    ):
        if q is None or k is None or v is None or layer is None or forward_batch is None:
            raise ValueError("Queries, KV, layer and batch are required")
        batch = forward_batch.batch
        if save_kv_cache:
            self.kvcache.store_kv(k, v, batch.out_loc, layer.layer_id)
        return invariant_attention(
            q,
            self.kvcache.k_cache(layer.layer_id),
            self.kvcache.v_cache(layer.layer_id),
            self.page_table,
            batch.attn_metadata.slots,
            batch.positions,
        )

    def init_capture_graph(self, max_seq_len, bs_list):
        self.capture = {}

    def prepare_for_capture(self, batch):
        self.prepare_metadata(batch)
        self.capture[batch.size] = batch.attn_metadata

    def prepare_for_replay(self, batch):
        saved = self.capture[batch.padded_size]
        saved.slots.copy_(batch.attn_metadata.slots)
        # Graphs have exactly one query per slot, so cu_seqlens is constant.
