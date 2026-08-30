"""Backend-independent transactional greedy speculative decoding."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from minisgl.runtime.adaptive import AdaptiveBlockController


@dataclass
class GenerationResult:
    token_ids: list[int]
    ttft_ms: float
    decode_ms: float
    rounds: list[dict] = field(default_factory=list)


def greedy_accept(block: list[int], predictions: list[int]) -> tuple[int, int]:
    if not block or len(block) != len(predictions):
        raise ValueError("Verification must return one next-token decision per input")
    accepted = 0
    while accepted + 1 < len(block) and block[accepted + 1] == predictions[accepted]:
        accepted += 1
    return accepted, predictions[accepted]


def generate(
    target,
    draft,
    prompt: list[int],
    max_new_tokens: int,
    *,
    block_size=16,
    adaptive: AdaptiveBlockController | None = None,
    eos_token_id=None,
    feasible=None,
) -> GenerationResult:
    if not prompt or max_new_tokens < 0 or block_size < 1:
        raise ValueError("Invalid generation request")
    if max_new_tokens == 0:
        return GenerationResult([], 0, 0)
    if draft is None and block_size != 1:
        raise ValueError("Speculative blocks require a draft")
    if draft is not None:
        draft.reset()
    begin = time.perf_counter()
    anchor = target.prefill(prompt)
    target.synchronize()
    ttft = (time.perf_counter() - begin) * 1000
    output, rounds = [int(anchor)], []
    decode_begin = time.perf_counter()
    while len(output) < max_new_tokens and output[-1] != eos_token_id:
        remaining = max_new_tokens - len(output)
        context = target.length
        allowed = feasible(context) if feasible else list(range(1, block_size + 1))
        if adaptive:
            block_len = adaptive.choose(
                batch_size=1,
                context_len=context,
                feasible_blocks=[b for b in allowed if b <= block_size],
                remaining=remaining,
            )
        else:
            candidates = [b for b in allowed if b <= block_size and b <= remaining]
            if not candidates:
                raise MemoryError("No decode block fits the memory budget")
            block_len = max(candidates)
        start = time.perf_counter()
        block = target.propose(draft, output[-1], block_len) if block_len > 1 else [output[-1]]
        target.synchronize()
        draft_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        snapshot = target.checkpoint() if block_len > 1 else None
        target.synchronize()
        checkpoint_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        predictions, features = target.verify(block)
        target.synchronize()
        verify_ms = (time.perf_counter() - start) * 1000
        accepted, bonus = greedy_accept(block, predictions)
        newly_emitted = block[1 : accepted + 1] + [bonus]
        if eos_token_id in newly_emitted:
            newly_emitted = newly_emitted[: newly_emitted.index(eos_token_id) + 1]
        # Target cache contains all output except its final emitted token.
        # The old anchor plus accepted draft tokens are committed; bonus is not.
        commit_count = len(newly_emitted)
        start = time.perf_counter()
        if commit_count != block_len:
            target.restore(snapshot)
            _, features = target.verify(block[:commit_count])
        target.commit_features(features, commit_count)
        del snapshot
        target.synchronize()
        restore_ms = (time.perf_counter() - start) * 1000 + checkpoint_ms
        output.extend(newly_emitted)
        rounds.append(
            dict(
                context=context,
                block=block_len,
                accepted_draft=accepted,
                progress=commit_count,
                draft_ms=draft_ms,
                verify_ms=verify_ms,
                restore_ms=restore_ms,
            )
        )
        if adaptive:
            adaptive.observe(
                block_len,
                batch_size=1,
                context_len=context,
                progress=commit_count,
                draft_ms=draft_ms,
                verify_ms=verify_ms,
                restore_ms=restore_ms,
            )
    return GenerationResult(output, ttft, (time.perf_counter() - decode_begin) * 1000, rounds)
