"""Contextual, measured-cost controller for greedy DFlash block selection."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Observation:
    progress: float
    cost_ms: float
    samples: int = 1


class AdaptiveBlockController:
    def __init__(self, blocks=(1, 2, 4, 8, 16), *, alpha=0.25, exploration_interval=8):
        self.blocks = tuple(sorted(set(blocks)))
        if not self.blocks or self.blocks[0] != 1 or any(b < 1 for b in self.blocks):
            raise ValueError('Candidate blocks must include one-token fallback')
        if not 0 < alpha <= 1 or exploration_interval < 1:
            raise ValueError('Invalid controller parameters')
        self.alpha, self.exploration_interval = alpha, exploration_interval
        self.observations = {}
        self.rounds = 0

    @staticmethod
    def bucket(batch_size: int, context_len: int):
        if min(batch_size, context_len) < 1:
            raise ValueError('Batch and context must be positive')
        return int(math.log2(batch_size)), int(math.log2(context_len))

    def observe(self, block: int, *, batch_size: int, context_len: int,
                progress: int, draft_ms: float, verify_ms: float, restore_ms: float = 0):
        values = (draft_ms, verify_ms, restore_ms)
        if block not in self.blocks or not 1 <= progress <= block:
            raise ValueError('Invalid block progress')
        if any(v < 0 or not math.isfinite(v) for v in values):
            raise ValueError('Latency must be finite and nonnegative')
        cost = max(sum(values), 1e-6)
        key = (*self.bucket(batch_size, context_len), block)
        old = self.observations.get(key)
        if old is None:
            self.observations[key] = Observation(progress, cost)
        else:
            old.progress += self.alpha * (progress - old.progress)
            old.cost_ms += self.alpha * (cost - old.cost_ms)
            old.samples += 1

    def choose(self, *, batch_size: int, context_len: int, feasible_blocks=None,
               remaining: int = 1 << 30) -> int:
        allowed = set(self.blocks if feasible_blocks is None else feasible_blocks)
        blocks = [b for b in self.blocks if b in allowed and b <= remaining]
        if not blocks:
            raise MemoryError('No block fits the current runtime budget')
        bucket = self.bucket(batch_size, context_len)
        unseen = [b for b in blocks if (*bucket, b) not in self.observations]
        self.rounds += 1
        if unseen:
            return unseen[0]  # bounded cold-start exploration, smallest first
        if self.rounds % self.exploration_interval == 0:
            return min(blocks, key=lambda b: self.observations[(*bucket, b)].samples)
        return max(blocks, key=lambda b: self.observations[(*bucket, b)].progress /
                   self.observations[(*bucket, b)].cost_ms)


@dataclass(frozen=True)
class JointMemoryBudget:
    capacity_bytes: int
    fixed_bytes: int
    safety_bytes: int = 256 << 20

    def feasible_blocks(self, blocks, *, live_bytes: int, bytes_per_block_token: int,
                        checkpoint_bytes: int = 0) -> list[int]:
        if min(self.capacity_bytes, self.fixed_bytes, self.safety_bytes, live_bytes,
               bytes_per_block_token, checkpoint_bytes) < 0:
            raise ValueError('Memory sizes must be nonnegative')
        free = self.capacity_bytes - self.fixed_bytes - live_bytes - self.safety_bytes
        # Block=1 does not speculate and needs no rollback checkpoint.
        return [b for b in blocks if b * bytes_per_block_token +
                (checkpoint_bytes if b > 1 else 0) <= free]
