"""Single-request adapter using the actual mini-SGLang Qwen3.5 target engine.

Paged KV locations are reserved contiguously for one request. Rollback restores
GDN state and logical length; speculative KV beyond that length is overwritten
on replay. This path intentionally does not use the overlap scheduler or graphs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from minisgl.compilation import set_forward_context
from minisgl.core import Batch, Req, SamplingParams
from minisgl.model_executor import ForwardBatch
from minisgl.runtime.adaptive import JointMemoryBudget
from minisgl.runtime.hybrid_cache import HybridPrefixCache
from torch.nn import functional as F


@dataclass
class TargetCheckpoint:
    history: list[int]
    states: dict[int, tuple[torch.Tensor, torch.Tensor]]


class MiniSGLTarget:
    def __init__(
        self,
        engine,
        *,
        capture_layer_ids=(),
        capture_final_hidden=False,
        cache: HybridPrefixCache | None = None,
        budget_bytes=24 << 30,
        safety_bytes=256 << 20,
        verify_mode="parallel",
        slot=0,
        executor=None,
    ):
        if verify_mode not in ("parallel", "sequential"):
            raise ValueError("Unknown verification mode")
        if not 0 <= slot < engine.page_table.shape[0] - 1:
            raise ValueError("Invalid request slot (dummy slot is reserved)")
        self.verify_mode = verify_mode
        self.slot, self.executor = slot, executor
        self.kv_start = slot * engine.max_seq_len
        self.last_cache_event = {}
        self.engine, self.device = engine, engine.device
        self.capture_layer_ids = tuple(capture_layer_ids)
        self.capture_final_hidden = capture_final_hidden
        if self.capture_layer_ids and self.capture_final_hidden:
            raise ValueError("Draft cannot request target taps and final hidden together")
        if any(
            i < 0 or i >= len(engine.model.model.layers.op_list) - 1 for i in self.capture_layer_ids
        ):
            raise ValueError("Target taps must be non-final decoder layer outputs")
        self.gdn = engine.attn_backend.gdn_backend
        self.cache = cache
        self.history: list[int] = []
        self.pending_features = []
        self.pending_next_tokens: list[int] = []
        self.budget_bytes, self.safety_bytes = budget_bytes, safety_bytes
        self.memory_events = []
        self.embedding = engine.model.model.embed_tokens.weight
        head = engine.model.lm_head
        self.head = (head.tied_embedding or head).weight
        engine.page_table[slot, : engine.max_seq_len] = torch.arange(
            self.kv_start, self.kv_start + engine.max_seq_len, dtype=torch.int32, device=self.device
        )

    @property
    def length(self):
        return len(self.history)

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def _forward(self, tokens: list[int], *, prefill=False):
        if self.executor is not None:
            return self.executor.forward([(self, tokens)], prefill=prefill)[0]
        if not tokens or self.length + len(tokens) > self.engine.max_seq_len:
            raise ValueError("Input exceeds the reserved target context")
        new_history = self.history + list(tokens)
        req = Req(
            torch.tensor(new_history, dtype=torch.int32),
            self.slot,
            self.length,
            self.engine.max_seq_len - len(new_history),
            0,
            SamplingParams(),
            None,
        )
        batch = Batch([req], "prefill" if prefill else "verify")
        batch.padded_reqs = batch.reqs
        batch.input_ids = torch.tensor(tokens, dtype=torch.int32, device=self.device)
        batch.positions = torch.arange(
            self.length, len(new_history), dtype=torch.int32, device=self.device
        )
        batch.out_loc = batch.positions + self.kv_start
        with torch.cuda.stream(self.engine.stream):
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
                hidden = self.engine.model.model.forward(
                    batch.input_ids, self.capture_layer_ids
                )
                features = (
                    hidden
                    if self.capture_final_hidden
                    else self.engine.model.model._last_aux_hidden
                )
                self.engine.model.model._last_aux_hidden = None
                logits = F.linear(hidden[-1:] if prefill else hidden, self.head).float()
        self.history = new_history
        return logits, features

    def _kv_view(self, layer, kind):
        t = (
            self.engine.kv_cache.k_cache(layer)
            if kind == "k"
            else self.engine.kv_cache.v_cache(layer)
        )
        return t.view(-1, t.shape[-2], t.shape[-1])[
            self.kv_start : self.kv_start + self.engine.max_seq_len
        ]

    def _prefix_payload(self, logits, features):
        payload = {"last_logits": logits[-1:].detach()}
        if features is not None:
            payload["features"] = features
        for layer in self.engine.kv_cache._layer_mapping:
            for kind in ("k", "v"):
                payload[f"{kind}.{layer}"] = self._kv_view(layer, kind)[: self.length]
        for layer, rt in self.gdn._runtime.items():
            payload[f"conv.{layer}"] = rt.conv_cache[self.slot]
            payload[f"ssm.{layer}"] = rt.ssm_cache[self.slot]
        return payload

    def _restore_prefix(self, entry):
        for name, source in entry.tensors.items():
            if "." not in name:
                continue
            kind, layer = name.split(".")
            layer = int(layer)
            if kind in ("k", "v"):
                self._kv_view(layer, kind)[: len(entry.tokens)].copy_(source)
            elif kind == "conv":
                self.gdn._runtime[layer].conv_cache[self.slot].copy_(source)
            elif kind == "ssm":
                self.gdn._runtime[layer].ssm_cache[self.slot].copy_(source)
        self.history = list(entry.tokens)

    @torch.inference_mode()
    def prepare_prefill(self, prompt):
        """Restore any complete prefix; leave suffix computation to the executor."""
        if not prompt or len(prompt) > self.engine.max_seq_len:
            raise ValueError("Invalid prompt length")
        self.history = []
        self.pending_features = []
        self.pending_next_tokens = []
        with torch.cuda.stream(self.engine.stream):
            self.gdn.on_table_slot_allocated(self.slot)
            self.gdn.prepare_state_slots()
            entry = self.cache.lookup(prompt) if self.cache else None
            self.last_cache_event = dict(
                status="hit" if entry else ("miss" if self.cache else "disabled"),
                matched_tokens=len(entry.tokens) if entry else 0,
                prompt_tokens=len(prompt),
                tier=entry.tier if entry else None,
                stored=False,
            )
            all_features = logits = None
            if entry is not None:
                restore_start = time.perf_counter()
                self._restore_prefix(entry)
                all_features = entry.tensors.get("features")
                if all_features is not None:
                    all_features = all_features.to(self.device)
                logits = entry.tensors["last_logits"].to(self.device)
                self.synchronize()
                self.cache.record_restore(entry, (time.perf_counter() - restore_start) * 1000)
            return dict(entry=entry, features=all_features, logits=logits)

    @torch.inference_mode()
    def finish_prefill(self, prompt, state, result, cost_ms):
        entry, all_features, logits = state["entry"], state["features"], state["logits"]
        with torch.cuda.stream(self.engine.stream):
            if result is not None:
                logits, new_features = result
                if new_features is not None:
                    all_features = (
                        new_features
                        if all_features is None
                        else torch.cat([all_features, new_features])
                    )
            token = logits[-1].argmax()
            if all_features is not None:
                self.pending_features.append(all_features)
            if self.capture_final_hidden:
                self.pending_next_tokens.extend(prompt[1:])
                self.pending_next_tokens.append(int(token))
            if self.cache and (entry is None or len(entry.tokens) != len(prompt)):
                # Reused-prefix compute cost is an estimate; suffix is measured.
                recompute_ms = cost_ms + (entry.recompute_ms if entry else 0)
                self.last_cache_event["stored"] = self.cache.put(
                    prompt, self._prefix_payload(logits, all_features), recompute_ms
                )
            return token

    @torch.inference_mode()
    def prefill(self, prompt):
        state = self.prepare_prefill(prompt)
        start = time.perf_counter()
        result = (
            self._forward(prompt[self.length :], prefill=True)
            if self.length < len(prompt)
            else None
        )
        self.synchronize()
        token = self.finish_prefill(prompt, state, result, (time.perf_counter() - start) * 1000)
        with torch.cuda.stream(self.engine.stream):
            return int(token.item())

    @torch.inference_mode()
    def checkpoint(self):
        with torch.cuda.stream(self.engine.stream):
            return TargetCheckpoint(
                list(self.history),
                {
                    lid: (rt.conv_cache[self.slot].clone(), rt.ssm_cache[self.slot].clone())
                    for lid, rt in self.gdn._runtime.items()
                },
            )

    @torch.inference_mode()
    def restore(self, snapshot):
        if snapshot is None:
            raise ValueError("Rollback requires a checkpoint")
        with torch.cuda.stream(self.engine.stream):
            for lid, (conv, ssm) in snapshot.states.items():
                self.gdn._runtime[lid].conv_cache[self.slot].copy_(conv)
                self.gdn._runtime[lid].ssm_cache[self.slot].copy_(ssm)
        self.history = list(snapshot.history)

    def verify(self, block):
        if self.verify_mode == "sequential" and len(block) > 1:
            # Diagnostic oracle: keep the same GEMM shapes as ordinary decode.
            # This deliberately gives up parallel target verification speed.
            outputs = [self._forward([token]) for token in block]
            logits = torch.cat([item[0] for item in outputs])
            features = (
                torch.cat([item[1] for item in outputs]) if outputs[0][1] is not None else None
            )
            return logits.argmax(-1).tolist(), features
        logits, features = self._forward(block)
        return logits.argmax(-1).tolist(), features

    def commit_features(self, features, count):
        if features is not None:
            self.pending_features.append(features[:count])

    def commit_next_tokens(self, tokens):
        if self.capture_final_hidden:
            self.pending_next_tokens.extend(map(int, tokens))

    @torch.inference_mode()
    def propose(self, draft, anchor, block):
        with torch.cuda.stream(self.engine.stream):
            features = torch.cat(self.pending_features, dim=0).unsqueeze(0)
            if getattr(draft, "draft_type", "dflash") == "mtp":
                result = draft.propose(
                    features,
                    self.pending_next_tokens,
                    block,
                    self.length,
                    self.embedding,
                    self.head,
                )
            else:
                result = draft.propose(
                    features,
                    anchor,
                    block,
                    self.length,
                    self.embedding,
                    self.head,
                )
        self.pending_features.clear()
        self.pending_next_tokens.clear()
        return result

    def feasible_blocks(self, context, blocks, batch_size=1):
        def admission():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            free, _ = torch.cuda.mem_get_info(self.device)
            available = min(free + reserved - allocated, self.budget_bytes - allocated)
            state_bytes = sum(
                t[self.slot].numel() * t.element_size()
                for rt in self.gdn._runtime.values()
                for t in (rt.conv_cache, rt.ssm_cache)
            )
            vocab, hidden = self.embedding.shape
            # Conservative workspace proxy; peak measurements must calibrate it.
            per_token = 8 * vocab + 32 * hidden * self.embedding.element_size() + 384 * context
            return JointMemoryBudget(self.budget_bytes, 0, self.safety_bytes).feasible_blocks(
                blocks,
                live_bytes=max(0, self.budget_bytes - available),
                bytes_per_block_token=per_token * batch_size,
                checkpoint_bytes=state_bytes * batch_size * (2 if self.executor else 1),
            )

        result = admission()
        if self.cache and self.cache.used("gpu") and (not result or max(result) < max(blocks)):
            released = self.cache.used("gpu")
            self.cache.resize_gpu_budget(0)
            self.synchronize()
            self.memory_events.append({"context": context, "gpu_cache_released_bytes": released})
            result = admission()
        return result
