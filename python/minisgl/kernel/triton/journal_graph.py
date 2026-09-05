"""Replay a persistent verify graph's GDN journal with dynamic accepted ranges."""

import torch

from .conv_extend import packed_conv_extend
from .gdn_extend import packed_extend
from .state_copy import LayerStateCopier


def replay_journal(journal, runtime, metadata):
    slots, starts, ends = metadata.unbind()
    for lid, record in journal.items():
        state = runtime[lid]
        conv_qkv = packed_conv_extend(
            record.mixed_qkv, record.conv_weight, state.conv_cache, slots, starts, end_offsets=ends
        )
        packed_extend(
            conv_qkv,
            record.a,
            record.b,
            record.A_log,
            record.dt_bias,
            state.ssm_cache,
            slots,
            starts,
            record.num_q_heads,
            record.num_v_heads,
            record.head_k_dim,
            record.head_v_dim,
            end_offsets=ends,
        )


class JournalReplayGraph:
    def __init__(self, journal, runtime, metadata, slots):
        self.journal, self.runtime = journal, runtime
        self.metadata = metadata.clone()
        self.copier = LayerStateCopier(runtime)
        saved = self.copier.checkpoint(slots)
        stream = torch.cuda.current_stream()
        self.graph = torch.cuda.CUDAGraph()
        rows = list(range(len(slots)))
        try:
            for _ in range(2):
                replay_journal(journal, runtime, self.metadata)
                self.copier.restore(saved, slots, rows)
            stream.synchronize()
            with torch.cuda.graph(self.graph, stream=stream):
                replay_journal(journal, runtime, self.metadata)
        finally:
            self.copier.restore(saved, slots, rows)

    def replay(self, metadata):
        if not self.copier.matches(self.runtime):
            raise RuntimeError("GDN state addresses changed after journal graph capture")
        self.metadata.copy_(metadata)
        self.graph.replay()
