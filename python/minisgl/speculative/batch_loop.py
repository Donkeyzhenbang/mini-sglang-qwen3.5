"""Greedy DFlash with ragged batching and optional continuous slot refill."""

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
    if not count or not targets or len(targets) != len(drafts) or len(limits) != count:
        raise ValueError("Mismatched wave inputs")
    if any(n < 1 for n in limits):
        raise ValueError("Batched generation requires max_new_tokens >= 1")
    begin = time.perf_counter()
    output = [[] for _ in prompts]
    first_ms, completed_ms = [0.0] * count, [0.0] * count
    rounds = [[] for _ in prompts]
    cache_events = [{} for _ in prompts]
    live, admissions = {}, []
    next_request = 0

    def admit():
        nonlocal next_request
        free = [slot for slot in range(len(targets)) if slot not in live.values()]
        indices = list(range(next_request, min(count, next_request + len(free))))
        if not indices:
            return
        slots = free[: len(indices)]
        admitted_ms = (time.perf_counter() - begin) * 1000
        for i, slot in zip(indices, slots):
            live[i] = slot
            if drafts[slot] is not None:
                drafts[slot].reset()
        tokens = executor.prefill([targets[s] for s in slots], [prompts[i] for i in indices])
        targets[0].synchronize()
        ready_ms = (time.perf_counter() - begin) * 1000
        for i, slot, token in zip(indices, slots, tokens):
            output[i] = [token]
            first_ms[i] = completed_ms[i] = ready_ms
            cache_events[i] = dict(getattr(targets[slot], "last_cache_event", {}))
            admissions.append(
                dict(request=i, slot=slot, admitted_ms=admitted_ms, first_token_ms=ready_ms)
            )
        next_request += len(indices)

    admit()
    decode_begin = time.perf_counter()
    mtp = bool(drafts and getattr(drafts[0], "draft_type", None) == "mtp")
    candidates = (
        list(range(1, block_size + 1)) if mtp else [b for b in (1, 2, 4, 8, 16) if b <= block_size]
    )
    while True:
        for i in list(live):
            if len(output[i]) >= limits[i] or output[i][-1] == eos_token_id:
                del live[i]
        admit()
        active = [i for i in live if len(output[i]) < limits[i] and output[i][-1] != eos_token_id]
        if not active:
            if next_request == count:
                break
            continue
        blocks, snapshots, observations, proposals = {}, {}, {}, []
        allowed = executor.feasible_blocks([targets[live[i]] for i in active], candidates)
        adaptive_block = None
        adaptive_context = max(targets[live[i]].length for i in active)
        if adaptive:
            # One action per GPU wave keeps verification shapes uniform and
            # lets the controller observe the actual shared wave cost.
            adaptive_block = adaptive.choose(
                batch_size=len(active),
                context_len=adaptive_context,
                feasible_blocks=allowed,
                remaining=min(limits[i] - len(output[i]) for i in active),
            )
        for i in active:
            target = targets[live[i]]
            context, remaining = target.length, limits[i] - len(output[i])
            if adaptive:
                size = adaptive_block
            else:
                feasible = [b for b in allowed if b <= remaining]
                if not feasible:
                    raise MemoryError("No decode block fits the joint wave budget")
                size = max(feasible)
            blocks[i] = [output[i][-1]]
            if size > 1:
                proposals.append(i)
            observations[i] = dict(
                context=context,
                block=size,
                draft_ms=0.0,
                restore_ms=0.0,
            )
        setup_before = executor.setup_count() if hasattr(executor, "setup_count") else 0
        draft_ms = checkpoint_ms = 0.0
        if proposals:
            start = time.perf_counter()
            proposed = executor.propose(
                [
                    (targets[live[i]], drafts[live[i]], output[i][-1], observations[i]["block"])
                    for i in proposals
                ]
            )
            targets[0].synchronize()
            draft_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            saved = executor.checkpoint([targets[live[i]] for i in proposals])
            targets[0].synchronize()
            checkpoint_ms = (time.perf_counter() - start) * 1000
            for i, block, snapshot in zip(proposals, proposed, saved):
                blocks[i], snapshots[i] = block, snapshot
                observations[i].update(
                    draft_ms=draft_ms / len(proposals), restore_ms=checkpoint_ms / len(proposals)
                )
            del saved, snapshot
        start = time.perf_counter()
        verified = executor.verify(
            [(targets[live[i]], blocks[i]) for i in active], sequential=sequential
        )
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
        use_state_journal = not sequential and hasattr(executor, "commit_verify_states")
        if replays:
            executor.restore([(targets[live[i]], snapshots[i]) for i in replays])
            if not use_state_journal:
                if hasattr(executor, "cancel_verify_states"):
                    executor.cancel_verify_states()
                recovered = executor.verify(
                    [(targets[live[i]], blocks[i][: len(emitted[i])]) for i in replays],
                    sequential=sequential,
                )
                for i, (_, hidden) in zip(replays, recovered):
                    features[i] = hidden
            else:
                executor.commit_verify_states(
                    [(targets[live[i]], len(emitted[i])) for i in replays]
                )
                for i in replays:
                    targets[live[i]].history.extend(blocks[i][: len(emitted[i])])
        else:
            if hasattr(executor, "cancel_verify_states"):
                executor.cancel_verify_states()
        targets[0].synchronize()
        replay_ms = (time.perf_counter() - start) * 1000
        setup_after = executor.setup_count() if hasattr(executor, "setup_count") else 0
        startup = setup_after != setup_before or (
            bool(proposals) and getattr(executor, "last_draft_catchup", False)
        )
        for i in active:
            progress = len(emitted[i])
            targets[live[i]].commit_features(features[i], progress)
            if hasattr(targets[live[i]], "commit_next_tokens"):
                targets[live[i]].commit_next_tokens(emitted[i])
            output[i].extend(emitted[i])
            row = observations[i]
            # Shared target costs are amortized; actual throughput uses wave
            # wall time, never a sum of overlapping request latencies.
            row.update(
                startup=startup,
                accepted_draft=accepted[i],
                progress=progress,
                verify_ms=verify_ms / len(active),
            )
            if i in replays:
                row["restore_ms"] += replay_ms / len(replays)
            rounds[i].append(row)
            completed_ms[i] = (time.perf_counter() - begin) * 1000
        if adaptive:
            adaptive.observe(
                adaptive_block,
                batch_size=len(active),
                context_len=adaptive_context,
                progress=sum(len(emitted[i]) for i in active),
                draft_ms=draft_ms,
                verify_ms=verify_ms,
                restore_ms=checkpoint_ms + replay_ms,
                startup=startup,
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
        request_cache_events=cache_events,
        admissions=admissions,
        completed_ms=completed_ms,
        arrival_time_basis="all requests available at runtime start; TTFT includes queue wait",
        decode_time_basis="wall time after initial prefill, including any refill prefills",
        round_cost_accounting=(
            "shared prefill/draft/checkpoint/verify/state-commit-or-replay "
            "costs amortized across participating requests"
        ),
    )
