# Four-request batching, readable answers and CUDA graphs

This is the experimental dense Qwen3.5 BF16 runtime, not the OpenAI service.
The initial four-question GPU checks use Qwen3.5-4B and its matching DFlash v1
checkpoint on RTX 4090. Broader BF16 token-parity limitations still apply; see
[earlier validation](gpu_validation_2026-08-31.md).

## Why the old command reported only misses

Both `--gpu-cache-mib` and `--host-cache-mib` default to zero. Previously, the
runner still performed lookups against an empty cache and counted every request
as a miss. It now reports `cache_enabled: false`, per-request `cache=disabled`,
and does not count disabled-cache lookups as misses.

The cache stores a complete prompt's KV + conv + SSM + draft features + last
logits. A hit needs an earlier *stored complete prompt* that is an exact token
prefix of the new input. Four unrelated chats sharing some template tokens do
not automatically yield a reusable GDN checkpoint at that shared boundary.

Warmup is excluded: its entries and counters are cleared before measurement.
`--repeat 2` repeats the entire input file inside one process, retaining measured
cache entries. Separate CLI invocations never share in-memory prefix caches.

## Copy-paste cloud demo

Run in Bash on the configured server. Do not run multiple benchmark processes
simultaneously: they share one GPU and the default distributed port.

```bash
cd /root/mini-sglang
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
MODEL=/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B
DRAFT=/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B-DFlash
OUT=$(mktemp -d /root/autodl-tmp/runtime-results/chat4-XXXXXX)

/root/miniconda3/bin/python -B -m minisgl.runtime.benchmark \
  --model "$MODEL" --draft "$DRAFT" \
  --mode fixed --block-size 8 --verify-mode sequential \
  --batch-size 4 --cuda-graph \
  --workload benchmark/runtime/workloads/chat4.jsonl --chat-template \
  --repeat 2 --gpu-cache-mib 512 --host-cache-mib 1024 \
  --gdn-extend packed --max-context 1024 --show-text \
  --output "$OUT/dflash.json"
```

Expected cache events: requests 1-4 miss and store their prompts; requests 5-8
hit the same four prompts. Initial validation observed `hits=4, misses=4`, GPU
hits, and 750 target graph replays. Counts can depend on workload, generation
length and memory budget; do not hardcode replay counts as an acceptance gate.

The CLI prints prompt, answer, matched prompt tokens, cache tier, storage result,
and finish reason. JSON additionally preserves prompt/output token IDs. `finish=length`
means `max_new_tokens` in that JSONL row was reached, not EOS; increase that row's
limit and, if necessary, `--max-context` to obtain a longer answer. The code-writing
question deliberately has a 96-token cap and can stop before the end of its example.

Use the same command with `--mode target` for a baseline; `--draft` is ignored in
target mode. Remove `--cuda-graph` for the eager comparison, keeping `--batch-size 4`.
Write a different output filename for each run, then check:

```bash
/root/miniconda3/bin/python -B -m minisgl.runtime.analyze --tokens-only \
  "$OUT/target.json" "$OUT/dflash.json"
```

Use `--mode adaptive --block-size 16` to exercise adaptive block selection.
`--verify-mode parallel` batches multi-token target verification but those calls
remain eager; the graph only covers eligible one-token forwards. Parallel mode
is still experimental and must pass the token check for each compared workload.

To demonstrate host-tier reuse, lower `--gpu-cache-mib` while retaining an adequate
host budget. Transfers remain synchronous. Offload is cost-dependent, so a small
budget alone is not a guarantee of a hit or an improvement in latency.

## What is actually batched/captured

- `--batch-size 4` enables wave batching with four distinct request slots. Target
  verification executes shared forwards; it is not four serial calls to `generate`.
- Prefills and draft proposals are per-request. Draft weights are shared without
  duplication; each request owns its draft KV context, target GDN state and KV region.
- Requests can finish at different times. Subsequent decode batches shrink and can
  use noncontiguous slots. New requests enter after the current wave finishes;
  continuous batching and the HTTP scheduler are not integrated.
- CUDA graphs capture target one-token decode including target features required
  by DFlash. Sequential verification invokes these graphs one position at a time;
  this is not a graph of an entire speculative round or parallel verify block.
- JSON `execution` records captured sizes, actual replay count, eager decode/verify
  counts and observed batch-size histograms. `graph_replays=0` means no measured
  graph replay occurred, regardless of the requested flag.
- The old single-request extend path remains available by omitting both new flags.
  Explicit `--batch-size 1` selects the new executor, permitting an eager-versus-graph
  comparison with the same decode backend. Different execution shapes can still
  change BF16 greedy decisions; this is not a global numerical-equivalence claim.
- Concurrent throughput uses actual wave wall time, not summed overlapping request
  latencies. Aggregate time per decoded token is an inverse-throughputput metric,
  not per-user TPOT or an online latency SLO. Per-round shared target costs are
  amortized across participating requests.

## Validation

```bash
/root/miniconda3/bin/python -B -m pytest tests/cpu tests/gpu -q \
  -o addopts='' -p no:cacheprovider

/root/miniconda3/bin/python -B benchmark/runtime/validate_batch_graph.py \
  --model "$MODEL" --draft "$DRAFT" --output "$OUT/graph-state.json"
```

The state probe compares eager and graph execution from identical real-model
histories at slot orders `[0,1,2,3]`, `[3,0,2]`, `[2]`, `[1,3,0,2]`. It requires
exact logits, auxiliary features, written KV and GDN states; checks inactive
slots remain unchanged; and checks graph outputs survive the next replay.

CPU tests additionally cover independent autoregressive-oracle comparisons with
ragged lengths, full/partial acceptance, rejection, EOS, slot reuse, adaptive
blocks, shared draft weights with isolated KV, effective workload hashes, and
wave wall-time accounting. Neither these tests nor one four-chat workload proves
all-context correctness or a DFlash speedup.
