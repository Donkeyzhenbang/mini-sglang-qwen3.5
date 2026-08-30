"""Bounded two-tier prefix cache for a complete KV + recurrent-state bundle.

This is an experimental mini-SGLang implementation, not SGLang's HiCache API.
Transfers are synchronous initially: correctness and measurable transfer cost
take precedence over an unvalidated asynchronous offload pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class CacheEntry:
    tokens: tuple[int, ...]
    tensors: dict[str, torch.Tensor]
    tier: str
    nbytes: int
    recompute_ms: float
    last_access: float
    hits: int = 0


class HybridPrefixCache:
    def __init__(
        self,
        gpu_bytes: int,
        host_bytes: int,
        *,
        policy: str = "cost",
        host_bandwidth_bytes_per_ms: float = 12e6,
    ):
        if min(gpu_bytes, host_bytes) < 0 or policy not in ("cost", "lru"):
            raise ValueError("Invalid cache budget or eviction policy")
        if host_bandwidth_bytes_per_ms <= 0:
            raise ValueError("Transfer bandwidth must be positive")
        self.budgets = {"gpu": gpu_bytes, "cpu": host_bytes}
        self.entries: dict[tuple[int, ...], CacheEntry] = {}
        self.policy = policy
        self.bandwidth = host_bandwidth_bytes_per_ms
        self.stats = dict(
            hits=0, misses=0, offloads=0, evictions=0, recompute_choices=0, transfer_ms=0.0
        )

    def used(self, tier: str) -> int:
        return sum(e.nbytes for e in self.entries.values() if e.tier == tier)

    def _value(self, entry: CacheEntry) -> float:
        if self.policy == "lru":
            return entry.last_access
        age = max(time.monotonic() - entry.last_access, 0)
        return (entry.hits + 1) * entry.recompute_ms / (max(1, entry.nbytes) * (1 + age))

    def _make_room(self, tier: str, needed: int) -> bool:
        if needed > self.budgets[tier]:
            return False
        while self.used(tier) + needed > self.budgets[tier]:
            candidates = [e for e in self.entries.values() if e.tier == tier]
            victim = min(candidates, key=self._value)
            del self.entries[victim.tokens]
            transfer_ms = victim.nbytes / self.bandwidth
            if (
                tier == "gpu"
                and victim.recompute_ms > transfer_ms
                and self._make_room("cpu", victim.nbytes)
            ):
                start = time.perf_counter()
                victim.tensors = {k: v.to("cpu", copy=True) for k, v in victim.tensors.items()}
                self.stats["transfer_ms"] += (time.perf_counter() - start) * 1000
                victim.tier = "cpu"
                self.entries[victim.tokens] = victim
                self.stats["offloads"] += 1
            else:
                self.stats["evictions"] += 1
        return True

    def put(self, tokens, tensors: dict[str, torch.Tensor], recompute_ms: float) -> bool:
        key = tuple(int(t) for t in tokens)
        if not key or not tensors or recompute_ms < 0:
            raise ValueError("Cache requires a nonempty prefix, payload and valid cost")
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        on_cuda = all(t.is_cuda for t in tensors.values())
        tier = "gpu" if on_cuda and size <= self.budgets["gpu"] else "cpu"
        if size > self.budgets[tier]:
            return False
        self.entries.pop(key, None)
        if not self._make_room(tier, size):
            return False
        device = next(iter(tensors.values())).device if tier == "gpu" else torch.device("cpu")
        start = time.perf_counter()
        payload = {k: t.detach().to(device, copy=True) for k, t in tensors.items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.stats["transfer_ms"] += (time.perf_counter() - start) * 1000
        self.entries[key] = CacheEntry(key, payload, tier, size, recompute_ms, time.monotonic())
        return True

    def lookup(self, tokens) -> CacheEntry | None:
        key = tuple(int(t) for t in tokens)
        candidates = [
            e for p, e in self.entries.items() if len(p) <= len(key) and key[: len(p)] == p
        ]
        # A host entry is read directly into live KV/SSM buffers by the caller;
        # there is no unbudgeted duplicate promotion to GPU.
        for entry in sorted(candidates, key=lambda e: len(e.tokens), reverse=True):
            if entry.tier == "cpu" and entry.nbytes / self.bandwidth > entry.recompute_ms:
                self.stats["recompute_choices"] += 1
                continue
            entry.hits += 1
            entry.last_access = time.monotonic()
            self.stats["hits"] += 1
            return entry
        self.stats["misses"] += 1
        return None

    def resize_gpu_budget(self, budget: int) -> None:
        if budget < 0:
            raise ValueError("Budget cannot be negative")
        self.budgets["gpu"] = budget
        self._make_room("gpu", 0)

    def clear(self) -> None:
        self.entries.clear()

    def record_restore(self, entry: CacheEntry, milliseconds: float) -> None:
        if entry.tier == "cpu" and milliseconds > 0:
            observed = entry.nbytes / milliseconds
            self.bandwidth = 0.8 * self.bandwidth + 0.2 * observed
            self.stats["transfer_ms"] += milliseconds
