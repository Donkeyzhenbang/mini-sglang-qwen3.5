"""Wave batching for the experimental target; draft proposals remain per-request.

All live requests share a target forward. Prefills are serialized to support
complete hybrid prefix restoration. CUDA graphs cover decode (one token per
request), including sequential verification; multi-token verification is eager.
"""

from __future__ import annotations

from collections import Counter

import torch
from minisgl.compilation import set_forward_context
from minisgl.core import Batch, Req, SamplingParams, get_global_ctx
from minisgl.engine.graph import GraphRunner
from minisgl.model_executor import ForwardBatch
from torch.nn import functional as F


class _DecodeModel:
    def __init__(self, engine, taps, max_batch):
        self.engine, self.taps = engine, taps
        head = engine.model.lm_head
        self.head = (head.tied_embedding or head).weight
        self.features = (
            torch.empty(
                max_batch,
                len(taps) * self.head.shape[1],
                device=engine.device,
                dtype=engine.dtype,
            )
            if taps
            else None
        )

    def forward(self):
        batch = get_global_ctx().batch
        hidden = self.engine.model.model.forward(batch.input_ids, self.taps)
        if self.features is not None:
            self.features[: batch.padded_size].copy_(self.engine.model.model._last_aux_hidden)
        self.engine.model.model._last_aux_hidden = None
        return F.linear(hidden, self.head).float()


class BatchedTargetExecutor:
    def __init__(self, engine, taps, max_batch, *, cuda_graph=False):
        self.engine, self.taps = engine, taps
        self.max_batch = max_batch
        self.decode_model = _DecodeModel(engine, taps, max_batch)
        self.graph_enabled = cuda_graph
        self.reset_stats()
        if cuda_graph:
            # The service graph cannot export draft features. Capture a decode
            # model with stable feature buffers as well as stable logits buffers.
            engine.graph_runner.destroy_cuda_graphs()
            engine.graph_runner = GraphRunner(
                stream=engine.stream,
                device=engine.device,
                model=self.decode_model,
                attn_backend=engine.attn_backend,
                cuda_graph_bs=list(range(1, max_batch + 1)),
                cuda_graph_max_bs=max_batch,
                free_memory=torch.cuda.mem_get_info()[0],
                max_seq_len=engine.page_table.shape[1],
                vocab_size=self.decode_model.head.shape[0],
                dummy_req=engine.dummy_req,
                attention_layers=engine.attention_layers,
            )
            # Capture only touched dummy state. Subsequent eager execution must
            # not accidentally select the dummy capture state-index tensor.
            engine.attn_backend.gdn_backend._capture_active_bs = None

    def reset_stats(self):
        self.graph_replays = 0
        self.eager_decode_calls = 0
        self.eager_verify_calls = 0
        self.prefill_calls = 0
        self.decode_batch_sizes = Counter()
        self.verify_batch_sizes = Counter()

    def stats(self):
        return dict(
            batch_size=self.max_batch,
            batching="wave",
            prefill="serial",
            draft="per-request eager",
            cuda_graph_enabled=self.graph_enabled,
            graph_scope="target one-token decode/sequential verify",
            captured_batch_sizes=self.engine.graph_runner.graph_bs_list,
            graph_replays=self.graph_replays,
            eager_decode_calls=self.eager_decode_calls,
            eager_verify_calls=self.eager_verify_calls,
            prefill_calls=self.prefill_calls,
            decode_batch_sizes=dict(self.decode_batch_sizes),
            verify_batch_sizes=dict(self.verify_batch_sizes),
        )

    @torch.inference_mode()
    def forward(self, items, *, prefill=False):
        if not items or len(items) > self.max_batch:
            raise ValueError("Invalid target batch size")
        if prefill and len(items) != 1:
            raise ValueError("Hybrid prefix restoration uses serial prefills")
        if len({t.slot for t, _ in items}) != len(items):
            raise ValueError("A batch cannot write the same request state twice")
        decode = not prefill and all(len(tokens) == 1 for _, tokens in items)
        phase = "prefill" if prefill else ("decode" if decode else "verify")
        reqs, histories, positions, locations, flat_tokens = [], [], [], [], []
        for target, tokens in items:
            if not tokens or target.length + len(tokens) > self.engine.max_seq_len:
                raise ValueError("Input exceeds reserved target context")
            history = target.history + list(tokens)
            reqs.append(
                Req(
                    torch.tensor(history, dtype=torch.int32, device="cpu"),
                    target.slot,
                    target.length,
                    self.engine.max_seq_len - len(history),
                    target.slot,
                    SamplingParams(),
                    None,
                )
            )
            histories.append(history)
            positions.extend(range(target.length, len(history)))
            locations.extend(range(target.kv_start + target.length, target.kv_start + len(history)))
            flat_tokens.extend(tokens)
        batch = Batch(reqs, phase)
        batch.padded_reqs = reqs  # Exact graph sizes: no padding into live slots.
        kw = dict(dtype=torch.int32, device=self.engine.device)
        with torch.cuda.stream(self.engine.stream):
            batch.input_ids = torch.tensor(flat_tokens, **kw)
            batch.positions = torch.tensor(positions, **kw)
            batch.out_loc = torch.tensor(locations, **kw)
            self.engine.attn_backend.prepare_state_slots()
            self.engine.attn_backend.prepare_metadata(batch)
            with (
                self.engine.ctx.forward_batch(batch),
                set_forward_context(
                    forward_batch=ForwardBatch.from_batch(
                        batch, attn_backend=self.engine.attn_backend
                    ),
                    attention_layers=self.engine.attention_layers,
                ),
            ):
                if decode and self.graph_enabled:
                    logits = self.engine.graph_runner.replay(batch).clone()
                    # Graph buffers are overwritten on the next replay. Features
                    # can outlive this call through draft context or rollback.
                    features = (
                        self.decode_model.features[: len(items)].clone()
                        if self.decode_model.features is not None
                        else None
                    )
                    self.graph_replays += 1
                else:
                    hidden = self.engine.model.model.forward(batch.input_ids, self.taps)
                    features = self.engine.model.model._last_aux_hidden
                    self.engine.model.model._last_aux_hidden = None
                    # Prefills are serial and need only their last logits.
                    logits = F.linear(
                        hidden[-1:] if prefill else hidden, self.decode_model.head
                    ).float()
                    if prefill:
                        self.prefill_calls += 1
                    elif decode:
                        self.eager_decode_calls += 1
                    else:
                        self.eager_verify_calls += 1
            if decode:
                self.decode_batch_sizes[len(items)] += 1
            elif not prefill:
                self.verify_batch_sizes[len(items)] += 1
            outputs, offset = [], 0
            for (target, tokens), history in zip(items, histories):
                target.history = history
                count = len(tokens)
                outputs.append(
                    (
                        logits if prefill else logits[offset : offset + count],
                        features[offset : offset + count] if features is not None else None,
                    )
                )
                offset += count
        return outputs

    def verify(self, items, *, sequential):
        if not sequential:
            return [
                (logits.argmax(-1).tolist(), features) for logits, features in self.forward(items)
            ]
        outputs = [[] for _ in items]
        for position in range(max(len(tokens) for _, tokens in items)):
            indices = [i for i, (_, tokens) in enumerate(items) if position < len(tokens)]
            steps = [(items[i][0], [items[i][1][position]]) for i in indices]
            for i, result in zip(indices, self.forward(steps)):
                outputs[i].append(result)
        return [
            (
                torch.cat([x[0] for x in row]).argmax(-1).tolist(),
                torch.cat([x[1] for x in row]) if row[0][1] is not None else None,
            )
            for row in outputs
        ]
