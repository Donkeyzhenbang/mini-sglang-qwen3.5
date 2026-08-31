# DFlash acceptance is not a prefix-cache hit rate

`cache.hits/misses` describe hybrid **prompt prefix reuse**. They do not describe
whether draft tokens were accepted. DFlash can speculate on the first request
with prefix caching disabled. Repeated cache hits skip target prefill work, but
do not return a cached answer: decoding still runs again.

The current executor batches prefill suffixes, draft proposals and target work
across requests; see [full batch runtime](full_batch_runtime.md). Ragged draft
requests share a padded forward with independent context caches. In
`--verify-mode sequential`, verification
positions execute serially (each position can batch several requests and use
a target decode CUDA graph). In `parallel`, positions and requests are packed
into a target forward, currently eager. `--continuous-batching` enables offline
slot refill between rounds. The default still groups requests into fixed waves.
The historical measurements below precede the full batch optimization.

## Reading the new counters

Every benchmark now prints a DFlash summary, separate from cache statistics.
With `--show-text`, each question/answer also prints its own summary. JSON has
`speculation` at both run and request levels, including a breakdown by block size.
Existing result files can be analyzed without rerunning inference:

```bash
cd /root/mini-sglang
/root/miniconda3/bin/python -B -m minisgl.runtime.analyze --per-request \
  /root/autodl-tmp/runtime-results/batch-graph-szyRuB/clean-dflash.json
```

Definitions for this implementation:

- A `block=8` contains **one known anchor + seven proposed draft tokens**.
- `draft_tokens_proposed = sum(block - 1)`. One-token fallback proposes nothing.
- `draft_tokens_matched` is the matching prefix returned by the greedy verifier,
  before stopping at EOS. This preserves the meaning of old `accepted_draft` logs.
- `draft_tokens_emitted` counts matched draft tokens actually retained in output,
  including an emitted EOS but excluding any matches beyond EOS. It is recovered
  from old logs as `sum(min(accepted_draft, progress))`.
- `acceptance_rate = draft_tokens_emitted / draft_tokens_proposed`, a token-weighted
  fraction, not an average of per-request or per-round percentages.
- `match_rate_before_eos` reports the alternative pre-EOS convention explicitly.
- `mean_output_tokens_per_round` includes draft tokens plus target-produced
  correction/bonus tokens; it is not itself an acceptance percentage or speedup.
- Runs with no draft proposals report acceptance as `null`/inactive, not 0%.

## What the four-question demonstration actually showed

The recorded fixed-block-8 run used four questions repeated twice. Across both
waves, it proposed 846 draft tokens, matched 244 before EOS truncation, and used
238 in output: **28.13% useful acceptance**, or **28.84% pre-EOS match rate**.
There were 122 speculative request-rounds and 2 one-token fallback rounds, with
mean output progress **2.87 tokens per request-round**. These are request-round
counts, not CUDA graph replay counts or batched target kernel-call counts.

| Question | Output tokens | Draft tokens emitted / proposed | Useful acceptance |
|---|---:|---:|---:|
| Explain KV caching | 63 | 27 / 252 | 10.71% |
| Calculate 17 times 23 | 4 | 2 / 14 | 14.29% |
| Write a Python even-number test | 96 | 77 / 115 | 66.96% |
| Translate one sentence | 19 | 13 / 42 | 30.95% |

The same token sequences and acceptance counts were observed with parallel
verification on this small workload. That does not resolve the larger workload's
known BF16 parity failures.

Measured aggregate decode throughput from the existing single runs:

| Mode, batch 4 with target decode graphs | Decode tokens/s |
|---|---:|
| Target only | 158.83 |
| DFlash fixed 8, sequential verification | 37.14 |
| DFlash fixed 8, parallel verification | 62.09 |
| Adaptive up to 8, sequential verification | 73.27 |

**This demonstration did not speed up decoding.** It established working batching,
cache restore and graph replay. Sequential verification deliberately gives up
parallel verification; even parallel mode here pays for serial request drafting,
low acceptance on several prompts, and GDN checkpoint/rollback/replay. Adaptive's
50.52% useful acceptance is not proof of superiority: it often uses smaller blocks
or falls back, with only 1.68 tokens per request-round in this run. Timings are
single-run observations, not controlled performance estimates.

## Measuring speculative speedup separately

Use `--repeat 1 --gpu-cache-mib 0 --host-cache-mib 0 --verify-mode parallel` to
isolate speculative decode from prefix reuse, with identical baseline/model,
batch size, prompts, output limits and graph settings. Check exact output token
parity before interpreting comparative timings. Increasing `max_new_tokens` is
only a cap: natural EOS can still produce a short answer.

For length experiments, select prompts that elicit substantial code, explanation
or reasoning, and test actual output lengths around 128/256/512/1024 tokens.
Report acceptance, progress per round, draft/verify/restore cost, decode throughput
and end-to-end latency separately. Longer outputs provide more opportunities to
amortize setup/prefill, but do not guarantee speedup: per-round speculative cost
must be below the ordinary decode cost of the tokens actually produced. Batch
size and context length can change both sides of that comparison.

## Fresh cache-disabled check

At `524402c`, a clean run with `--repeat 1 --gpu-cache-mib 0 --host-cache-mib 0 --verify-mode parallel --batch-size 4 --cuda-graph` produced the same four target outputs. It recorded `cache_enabled=false`, zero cache lookups, 119 emitted draft tokens / 423 proposed (28.13%), 2.87 tokens per request-round, and 61 speculative request-rounds plus one fallback. No previous request cache was needed. All 58 CPU/GPU tests passed. [Recorded counters](dflash_acceptance_nocache_2026-08-31.json).
