"""Batched prefill, draft and target verification for isolated request slots.

All live requests share model forwards. CUDA graphs cover decode (one token per
request), including sequential verification; multi-token verification is eager.
"""

from __future__ import annotations

import time
from collections import Counter

import torch
from minisgl.compilation import set_forward_context
from minisgl.core import Batch, Req, SamplingParams, get_global_ctx
from minisgl.engine.graph import GraphRunner
from minisgl.model_executor import ForwardBatch
from torch.nn import functional as F


class _DecodeModel:
    def __init__(
        self, engine, taps, max_batch, target_numerics, capture_final_hidden
    ):
        self.engine, self.taps = engine, taps
        self.target_numerics = target_numerics
        self.capture_final_hidden = capture_final_hidden
        head = engine.model.lm_head
        self.head = (head.tied_embedding or head).weight
        self.features = (
            torch.empty(
                max_batch,
                (1 if capture_final_hidden else len(taps)) * self.head.shape[1],
                device=engine.device,
                dtype=engine.dtype,
            )
            if taps or capture_final_hidden
            else None
        )

    def forward(self):
        batch = get_global_ctx().batch
        hidden = self.engine.model.model.forward(batch.input_ids, self.taps)
        if self.features is not None:
            captured = (
                hidden
                if self.capture_final_hidden
                else self.engine.model.model._last_aux_hidden
            )
            self.features[: batch.padded_size].copy_(captured)
        self.engine.model.model._last_aux_hidden = None
        return self.project(hidden)

    def project(self, hidden):
        if self.target_numerics == "stable":
            from minisgl.kernel.triton.invariant import invariant_linear

            return invariant_linear(hidden, self.head, fp32_output=True)
        return F.linear(hidden, self.head).float()


class BatchedTargetExecutor:
    def __init__(
        self,
        engine,
        taps,
        max_batch,
        *,
        cuda_graph=False,
        target_numerics="fast",
        capture_final_hidden=False,
        verify_cuda_graph=True,
    ):
        from .numerics import configure_target_numerics

        configure_target_numerics(engine, target_numerics)
        self.engine, self.taps = engine, taps
        self.max_batch = max_batch
        self.target_numerics = target_numerics
        self.decode_model = _DecodeModel(
            engine, taps, max_batch, target_numerics, capture_final_hidden
        )
        self.verify_graphs = {}
        self.verify_graph_enabled = cuda_graph and verify_cuda_graph and target_numerics == "stable"
        self.graph_enabled = cuda_graph
        self.batching = "wave"
        engine.attn_backend.gdn_backend.packed_verify_conv = True
        engine.attn_backend.gdn_backend.verify_state_journal = True
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
        self.verify_graph_replays = 0
        self.graph_replays = 0
        self.eager_decode_calls = 0
        self.eager_verify_calls = 0
        self.prefill_calls = 0
        self.draft_calls = 0
        self.state_journal_commits = 0
        self.state_journal_requests = 0
        self.prefill_batch_sizes = Counter()
        self.draft_batch_sizes = Counter()
        self.decode_batch_sizes = Counter()
        self.verify_batch_sizes = Counter()

    def stats(self):
        return dict(
            batch_size=self.max_batch,
            target_numerics=self.target_numerics,
            batching=self.batching,
            prefill="batched ragged suffixes; per-request cache restoration",
            draft="batched padded ragged contexts",
            cuda_graph_enabled=self.graph_enabled,
            graph_scope=(
                "target decode; uniform parallel verify (stable numerics)"
                if self.verify_graph_enabled else "target one-token decode/sequential verify"
            ),
            verify_cuda_graph_enabled=self.verify_graph_enabled,
            verify_graph_replays=self.verify_graph_replays,
            captured_verify_shapes=[list(shape) for shape in self.verify_graphs],
            packed_gdn_verify_convolution=self.engine.attn_backend.gdn_backend.extend_backend
            == "packed",
            gdn_verify_state_journal=self.engine.attn_backend.gdn_backend.verify_state_journal,
            captured_batch_sizes=self.engine.graph_runner.graph_bs_list,
            graph_replays=self.graph_replays,
            eager_decode_calls=self.eager_decode_calls,
            eager_verify_calls=self.eager_verify_calls,
            prefill_calls=self.prefill_calls,
            draft_calls=self.draft_calls,
            state_journal_commits=self.state_journal_commits,
            state_journal_requests=self.state_journal_requests,
            prefill_batch_sizes=dict(self.prefill_batch_sizes),
            draft_batch_sizes=dict(self.draft_batch_sizes),
            decode_batch_sizes=dict(self.decode_batch_sizes),
            verify_batch_sizes=dict(self.verify_batch_sizes),
        )

    @torch.inference_mode()
    def forward(self, items, *, prefill=False):
        if not items or len(items) > self.max_batch:
            raise ValueError("Invalid target batch size")
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
                uniform_verify = (
                    not decode and not prefill and self.verify_graph_enabled
                    and len({len(tokens) for _, tokens in items}) == 1
                    and len(items[0][1]) in (2, 4, 8, 16)
                )
                if uniform_verify:
                    from .verify_graph import VerifyGraph

                    key = (len(items), len(items[0][1]))
                    if key not in self.verify_graphs:
                        self.verify_graphs[key] = VerifyGraph(self, batch)
                    logits, features = self.verify_graphs[key].replay(batch)
                    self.verify_graph_replays += 1
                elif decode and self.graph_enabled:
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
                    hidden = self.engine.model.model.forward(
                        batch.input_ids, self.taps
                    )
                    features = (
                        hidden
                        if self.decode_model.capture_final_hidden
                        else self.engine.model.model._last_aux_hidden
                    )
                    self.engine.model.model._last_aux_hidden = None
                    # Project only the last prompt token of each request.
                    if prefill:
                        ends = (
                            torch.tensor(
                                [len(tokens) for _, tokens in items], device=hidden.device
                            ).cumsum(0)
                            - 1
                        )
                        hidden = hidden.index_select(0, ends)
                    logits = self.decode_model.project(hidden)
                    if prefill:
                        self.prefill_calls += 1
                        self.prefill_batch_sizes[len(items)] += 1
                    elif decode:
                        self.eager_decode_calls += 1
                    else:
                        self.eager_verify_calls += 1
            if decode:
                self.decode_batch_sizes[len(items)] += 1
            elif not prefill:
                self.verify_batch_sizes[len(items)] += 1
            outputs, offset = [], 0
            for i, ((target, tokens), history) in enumerate(zip(items, histories)):
                target.history = history
                count = len(tokens)
                outputs.append(
                    (
                        logits[i : i + 1] if prefill else logits[offset : offset + count],
                        features[offset : offset + count] if features is not None else None,
                    )
                )
                offset += count
        return outputs

    @torch.inference_mode()
    def prefill(self, targets, prompts):
        states = [t.prepare_prefill(p) for t, p in zip(targets, prompts)]
        pending = [
            (i, t, p[t.length :])
            for i, (t, p) in enumerate(zip(targets, prompts))
            if t.length < len(p)
        ]
        start = time.perf_counter()
        computed = (
            self.forward([(t, suffix) for _, t, suffix in pending], prefill=True) if pending else []
        )
        targets[0].synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        total_tokens = sum(len(suffix) for _, _, suffix in pending)
        results = {
            i: (result, elapsed * len(suffix) / total_tokens)
            for (i, _, suffix), result in zip(pending, computed)
        }
        with torch.cuda.stream(self.engine.stream):
            tokens = [
                t.finish_prefill(p, state, *results.get(i, (None, 0)))
                for i, (t, p, state) in enumerate(zip(targets, prompts, states))
            ]
            return torch.stack(tokens).tolist()

    @torch.inference_mode()
    def propose(self, items):
        with torch.cuda.stream(self.engine.stream):
            if getattr(items[0][1], "draft_type", "dflash") == "mtp":
                from .mtp import propose_mtp_batch

                rows = [
                    (
                        d,
                        torch.cat(t.pending_features).unsqueeze(0),
                        t.pending_next_tokens,
                        size,
                        t.length,
                    )
                    for t, d, _, size in items
                ]
                result = propose_mtp_batch(
                    rows, items[0][0].embedding, items[0][0].head
                )
            else:
                from .batch_draft import propose_batch

                rows = [
                    (
                        d,
                        torch.cat(t.pending_features).unsqueeze(0),
                        anchor,
                        size,
                        t.length,
                    )
                    for t, d, anchor, size in items
                ]
                result = propose_batch(
                    rows, items[0][0].embedding, items[0][0].head
                )
            for t, _, _, _ in items:
                t.pending_features.clear()
                t.pending_next_tokens.clear()
        self.draft_calls += 1
        self.draft_batch_sizes[len(items)] += 1
        return result

    @torch.inference_mode()
    def checkpoint(self, targets):
        from minisgl.kernel.triton.state_copy import LayerStateCopier

        from .target import TargetCheckpoint

        with torch.cuda.stream(self.engine.stream):
            runtime = self.engine.attn_backend.gdn_backend._runtime
            copier = getattr(self, "_state_copier", None)
            if runtime and (copier is None or not copier.matches(runtime)):
                try:
                    copier = self._state_copier = LayerStateCopier(runtime)
                except ValueError:
                    copier = self._state_copier = None
            if copier is not None and runtime:
                snapshot = copier.checkpoint([t.slot for t in targets])
                return [
                    TargetCheckpoint(list(t.history), {}, (snapshot, i))
                    for i, t in enumerate(targets)
                ]
            slots = torch.tensor([t.slot for t in targets], device=self.engine.device)
            states = {
                lid: (rt.conv_cache.index_select(0, slots), rt.ssm_cache.index_select(0, slots))
                for lid, rt in self.engine.attn_backend.gdn_backend._runtime.items()
            }
            return [
                TargetCheckpoint(
                    list(t.history), {lid: (conv[i], ssm[i]) for lid, (conv, ssm) in states.items()}
                )
                for i, t in enumerate(targets)
            ]

    @torch.inference_mode()
    def restore(self, items):
        with torch.cuda.stream(self.engine.stream):
            groups = {}
            legacy = []
            for target, saved in items:
                if saved.packed is None:
                    legacy.append((target, saved))
                    continue
                snapshot, row = saved.packed
                group = groups.setdefault(id(snapshot), (snapshot, [], []))
                group[1].append(target.slot)
                group[2].append(row)
            for snapshot, slots, rows in groups.values():
                if not snapshot.copier.matches(self.engine.attn_backend.gdn_backend._runtime):
                    raise ValueError("GDN state allocations changed since checkpoint")
                snapshot.copier.restore(snapshot, slots, rows)
            for target, saved in items:
                target.history = list(saved.history)
            if not legacy:
                return
            items = legacy
            slots = torch.tensor([t.slot for t, _ in items], device=self.engine.device)
            for lid, rt in self.engine.attn_backend.gdn_backend._runtime.items():
                rt.conv_cache.index_copy_(
                    0, slots, torch.stack([saved.states[lid][0] for _, saved in items])
                )
                rt.ssm_cache.index_copy_(
                    0, slots, torch.stack([saved.states[lid][1] for _, saved in items])
                )
            for target, saved in items:
                target.history = list(saved.history)

    def feasible_blocks(self, targets, candidates):
        # All slots share one allocator and cache budget. Use the longest
        # context conservatively instead of querying CUDA memory per request.
        target = max(targets, key=lambda t: t.length)
        return target.feasible_blocks(target.length, candidates, batch_size=len(targets))

    def verify(self, items, *, sequential):
        if not sequential:
            with torch.cuda.stream(self.engine.stream):
                gdn = self.engine.attn_backend.gdn_backend
                gdn.begin_verify_journal()
                try:
                    result = self.forward(items)
                except BaseException:
                    gdn.cancel_verify_journal()
                    raise
                predictions = torch.cat([logits.argmax(-1) for logits, _ in result]).tolist()
                outputs, offset = [], 0
                for logits, features in result:
                    n = len(logits)
                    outputs.append((predictions[offset : offset + n], features))
                    offset += n
                return outputs
        outputs = [[] for _ in items]
        for position in range(max(len(tokens) for _, tokens in items)):
            indices = [i for i, (_, tokens) in enumerate(items) if position < len(tokens)]
            steps = [(items[i][0], [items[i][1][position]]) for i in indices]
            for i, result in zip(indices, self.forward(steps)):
                outputs[i].append(result)
        with torch.cuda.stream(self.engine.stream):
            predictions = torch.cat(
                [torch.cat([x[0] for x in row]).argmax(-1) for row in outputs]
            ).tolist()
            result, offset = [], 0
            for row in outputs:
                n = len(row)
                result.append(
                    (
                        predictions[offset : offset + n],
                        torch.cat([x[1] for x in row]) if row[0][1] is not None else None,
                    )
                )
                offset += n
            return result

    @torch.inference_mode()
    def commit_verify_states(self, items):
        with torch.cuda.stream(self.engine.stream):
            self.engine.attn_backend.gdn_backend.commit_verify_journal(items)
        self.state_journal_commits += 1
        self.state_journal_requests += len(items)

    def cancel_verify_states(self):
        self.engine.attn_backend.gdn_backend.cancel_verify_journal()
