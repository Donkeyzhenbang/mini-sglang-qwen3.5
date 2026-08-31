"""Greedy DFlash wave loop with batched target verification and isolated slots."""

from __future__ import annotations

import time

from .loop import GenerationResult, greedy_accept


def generate_batch(
    targets,
    drafts,
    prompts,
    limits,
    executor,
    *,
    block_size,
    adaptive=None,
    eos_token_id=None,
    sequential=False,
):
    count = len(prompts)
    if not count or not (len(targets) == len(drafts) == len(limits) == count):
        raise ValueError("Mismatched wave inputs")
    if any(n < 1 for n in limits):
        raise ValueError("Batched generation requires max_new_tokens >= 1")
    for draft in drafts:
        if draft is not None:
            draft.reset()
    begin = time.perf_counter()
    output, first_ms, rounds = [], [], [[] for _ in prompts]
    for target, prompt in zip(targets, prompts):
        output.append([target.prefill(prompt)])
        target.synchronize()
        first_ms.append((time.perf_counter() - begin) * 1000)
    decode_begin = time.perf_counter()
    completed_ms = list(first_ms)
    candidates = [b for b in (1, 2, 4, 8, 16) if b <= block_size]
    while True:
        active = [
            i for i in range(count) if len(output[i]) < limits[i] and output[i][-1] != eos_token_id
        ]
        if not active:
            break
        blocks, snapshots, observations = {}, {}, {}
        for i in active:
            target = targets[i]
            context, remaining = target.length, limits[i] - len(output[i])
            allowed = target.feasible_blocks(context, candidates, batch_size=len(active))
            if adaptive:
                size = adaptive.choose(
                    batch_size=len(active),
                    context_len=context,
                    feasible_blocks=allowed,
                    remaining=remaining,
                )
            else:
                feasible = [b for b in allowed if b <= remaining]
                if not feasible:
                    raise MemoryError("No decode block fits the joint wave budget")
                size = max(feasible)
            start = time.perf_counter()
            blocks[i] = (
                target.propose(drafts[i], output[i][-1], size) if size > 1 else [output[i][-1]]
            )
            target.synchronize()
            draft_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            snapshots[i] = target.checkpoint() if size > 1 else None
            target.synchronize()
            observations[i] = dict(
                context=context,
                block=size,
                draft_ms=draft_ms,
                restore_ms=(time.perf_counter() - start) * 1000,
            )
        start = time.perf_counter()
        verified = executor.verify([(targets[i], blocks[i]) for i in active], sequential=sequential)
        targets[0].synchronize()
        verify_ms = (time.perf_counter() - start) * 1000
        replays, emitted, accepted, features = [], {}, {}, {}
        for i, (predictions, hidden) in zip(active, verified):
            accepted[i], bonus = greedy_accept(blocks[i], predictions)
            emitted[i] = blocks[i][1 : accepted[i] + 1] + [bonus]
            if eos_token_id in emitted[i]:
                emitted[i] = emitted[i][: emitted[i].index(eos_token_id) + 1]
            features[i] = hidden
            if len(emitted[i]) != len(blocks[i]):
                replays.append(i)
        start = time.perf_counter()
        for i in replays:
            targets[i].restore(snapshots[i])
        if replays:
            recovered = executor.verify(
                [(targets[i], blocks[i][: len(emitted[i])]) for i in replays],
                sequential=sequential,
            )
            for i, (_, hidden) in zip(replays, recovered):
                features[i] = hidden
        targets[0].synchronize()
        replay_ms = (time.perf_counter() - start) * 1000
        for i in active:
            progress = len(emitted[i])
            targets[i].commit_features(features[i], progress)
            output[i].extend(emitted[i])
            row = observations[i]
            # Shared target costs are amortized; actual throughput uses wave
            # wall time, never a sum of overlapping request latencies.
            row.update(
                accepted_draft=accepted[i], progress=progress, verify_ms=verify_ms / len(active)
            )
            if i in replays:
                row["restore_ms"] += replay_ms / len(replays)
            rounds[i].append(row)
            completed_ms[i] = (time.perf_counter() - begin) * 1000
            if adaptive:
                adaptive.observe(
                    row["block"],
                    batch_size=len(active),
                    context_len=row["context"],
                    progress=progress,
                    draft_ms=row["draft_ms"],
                    verify_ms=row["verify_ms"],
                    restore_ms=row["restore_ms"],
                )
        del snapshots, verified, features
    targets[0].synchronize()
    end = time.perf_counter()
    return [
        GenerationResult(tokens, first, max(0, done - first), rs)
        for tokens, first, done, rs in zip(output, first_ms, completed_ms, rounds)
    ], dict(
        requests=count,
        total_ms=(end - begin) * 1000,
        decode_ms=(end - decode_begin) * 1000,
        output_tokens=sum(map(len, output)),
        decoded_tokens=sum(len(x) - 1 for x in output),
        round_cost_accounting="shared target verify/replay time amortized across participating requests",
    )
