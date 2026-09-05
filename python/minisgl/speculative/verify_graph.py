"""CUDA graphs for bounded, uniform verify blocks with stable target numerics.

The fixed-shape graph exports the GDN journal as persistent tensors. Request
slots and token positions remain dynamic. Capture/warmup restores the initial
recurrent state, and only writes speculative KV locations beyond the prefix.
"""

from __future__ import annotations

import torch
from minisgl.compilation import set_forward_context
from minisgl.kernel.triton.state_copy import LayerStateCopier
from minisgl.model_executor import ForwardBatch


class VerifyGraph:
    def __init__(self, executor, batch):
        self.executor, self.batch = executor, batch
        self.engine = executor.engine
        self.gdn = self.engine.attn_backend.gdn_backend
        self.forward_batch = ForwardBatch.from_batch(batch, attn_backend=self.engine.attn_backend)
        slots = [req.table_idx for req in batch.reqs]
        lengths = [req.extend_len for req in batch.reqs]
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        self.forward_batch.gdn_extend_metadata = (
            torch.tensor(slots, device=self.engine.device, dtype=torch.int32),
            torch.tensor(cu, device=self.engine.device, dtype=torch.int32),
        )
        # Never replace live state allocations: decode graphs already refer to them.
        self.copier = LayerStateCopier(self.gdn._runtime)
        saved = self.copier.checkpoint(slots)
        self.graph = torch.cuda.CUDAGraph()
        try:
            for _ in range(2):
                self._body()
                self.copier.restore(saved, slots, list(range(len(slots))))
            self.engine.stream.synchronize()
            with torch.cuda.graph(self.graph, stream=self.engine.stream):
                self._body()
        finally:
            self.copier.restore(saved, slots, list(range(len(slots))))
            self.gdn.cancel_verify_journal()

    def _body(self):
        self.gdn.begin_verify_journal()
        with (
            set_forward_context(
                forward_batch=self.forward_batch,
                attention_layers=self.engine.attention_layers,
            ),
        ):
            hidden = self.engine.model.model.forward(self.batch.input_ids, self.executor.taps)
            self.features = (
                hidden
                if self.executor.decode_model.capture_final_hidden
                else self.engine.model.model._last_aux_hidden
            )
            self.engine.model.model._last_aux_hidden = None
            self.logits = self.executor.decode_model.project(hidden)
        self.journal = self.gdn._verify_journal

    def replay(self, batch):
        if not self.copier.matches(self.gdn._runtime):
            raise RuntimeError("GDN allocations changed after verify graph capture")
        slots = [req.table_idx for req in batch.reqs]
        if batch is not self.batch:
            self.batch.input_ids.copy_(batch.input_ids)
            self.batch.positions.copy_(batch.positions)
            self.batch.out_loc.copy_(batch.out_loc)
            self.batch.attn_metadata.slots.copy_(batch.attn_metadata.slots)
            self.forward_batch.gdn_extend_metadata[0].copy_(
                torch.tensor(slots, device=self.engine.device, dtype=torch.int32)
            )
        for record in self.journal.values():
            record.request_slots = slots
        self.graph.replay()
        self.gdn._verify_journal = self.journal
        # Request features and predictions may outlive the next replay.
        return self.logits.clone(), None if self.features is None else self.features.clone()
