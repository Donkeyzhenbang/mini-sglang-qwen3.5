# Batched Qwen3.5 / DFlash runtime

The experimental CLI now batches **prefill suffixes, draft model forwards and
target verification**. Cache lookup/restoration remains per request, while model
execution shares a batch. Draft padding uses per-request absolute positions and
valid-key masks; speculative noise and padding never enter confirmed draft KV.
Weights are shared, mutable draft state is independent per slot.

The packed GDN path reuses sequence metadata across layers. Short verification
convolutions use a ragged Triton kernel (width 4), with BF16 rounding before SiLU.
Checkpoint gathers and rollback scatters are batched per layer. Verification
predictions transfer to the host once per batch; memory admission uses the longest
active context and accounts for rollback scratch space.

`--continuous-batching` admits waiting requests into completed slots between
rounds. It is an offline experimental scheduler, **not integration with the main
MiniSGLang HTTP/overlap scheduler**. Refill prefill currently pauses decoding;
there is no mixed prefill/decode kernel or chunked-prefill policy. Without this
flag, requests run in fixed waves, still fully batched inside each stage.

CUDA Graph covers target one-token decode, including eligible replay and
sequential verification. Draft, prefill and multi-token verification stay eager.
GDN checkpoint/replay and low draft acceptance can still outweigh speculation
benefits. Neither batching nor longer answers guarantee a speedup.

## Reproduce on the cloud instance

```bash
cd /root/mini-sglang
export PYTHONPATH="$PWD/python"
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B
DRAFT=/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B-DFlash
OUT=$(mktemp -d /root/autodl-tmp/runtime-results/reproduce-full-batch-XXXXXX)

run_bench() {
  "$PY" -B -m minisgl.runtime.benchmark \
    --model "$MODEL" --draft "$DRAFT" \
    --workload benchmark/runtime/workloads/chat4.jsonl \
    --max-context 1024 --batch-size 4 --gdn-extend packed \
    --cuda-graph --chat-template --show-text "$@"
}
run_bench --mode target --output "$OUT/target.json"
run_bench --mode fixed --block-size 8 --output "$OUT/dflash.json"
"$PY" -B -m minisgl.runtime.analyze "$OUT/target.json" "$OUT/dflash.json"

# Eight queued requests, up to four active; exact prompt repetition tests cache.
run_bench --mode fixed --block-size 8 --continuous-batching --repeat 2 \
  --gpu-cache-mib 512 --host-cache-mib 1024 --output "$OUT/continuous.json"

# Longer mixed explanatory/code output; lengths are caps, EOS can stop earlier.
run_bench --mode fixed --block-size 8 \
  --workload benchmark/runtime/workloads/chat-long4.jsonl --output "$OUT/long.json"
```

Inspect `execution.prefill_batch_sizes`, `draft_batch_sizes`,
`verify_batch_sizes` and `graph_replays` for actual execution counts.
`waves[].admissions` records request-to-slot mappings and admission timestamps;
`completed_ms` allows checking that refill precedes the oldest request finishing.
Output order is original input order, not completion order. TTFT includes queue
wait within each submitted wave. Continuous decode wall time includes subsequent
refill prefills. Shared stage costs are amortized and are not per-user latency.

Prefix-cache hit rate is separate from `speculation.acceptance_rate`: the latter
counts useful emitted draft tokens divided by proposed draft tokens. For an
uncached acceptance experiment leave both cache budgets at zero. `--repeat 2`
is unnecessary for speculative decoding itself.

Correctness and performance comparisons must use identical tokenized workloads
and output tokens. BF16 shape changes can change argmax near ties; the existing
parallel/sequential numerical caveat still applies. Do not report speedups for
token-mismatching runs. `--verify-mode sequential` is a diagnostic oracle.

```bash
"$PY" -B -m pytest -o addopts='' tests/cpu tests/gpu -q
"$PY" -B benchmark/runtime/validate_batch_graph.py \
  --model "$MODEL" --draft "$DRAFT" --output "$OUT/state.json"
```
