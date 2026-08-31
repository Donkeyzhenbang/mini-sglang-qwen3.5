# Stable target numerics for Qwen3.5 / DFlash

Measured coverage and remaining performance limits are recorded in
[the validation report](stable_target_results_2026-08-31.md).

The experimental benchmark now defaults to `--target-numerics stable`. It uses
the same target arithmetic for autoregressive decoding, packed verification,
ragged batches and rollback replay. Multiple requests and verify tokens remain
batched; this is not a fallback that serializes all requests. The regular HTTP
engine and native draft model retain their existing operators.

## Why long outputs diverged

Same-history tracing isolated two independent sources after the convolution fix:

1. The target's BF16 library GEMMs selected different arithmetic with different
   row counts. The first GDN layer's `in_proj_ba` already differed in 18/64 values
   when comparing four decode tokens with four eight-token verify blocks.
   Replaying fewer requests also changed the first `in_proj_qkvz` output.
2. After fixing linear projections, the first Full Attention layer still differed
   between decode and packed verification. Disabling FlashInfer split-KV alone
   did not remove the discrepancy in this installed version.

These small differences propagate through recurrent state and can change a
near-tied greedy choice. Sequential verify still changes batch composition during
rollback, so it did not fix the original problem. Cache hits are unrelated to
draft acceptance, and turning cache on cannot repair numerical inconsistencies.

## Implementation and scope

- Fixed Triton linear tiles, independent of the number or position of query rows.
  Each K tile produces a partial dot product, followed by an explicit FP32 add.
  This prevents a long MMA accumulator from degrading precision. Against a
  separate FP64 oracle at K=9216, maximum error fell from 0.006866 to 0.000183;
  tests retain their original tolerance.
- Full attention reduces each query's causal prefix with the same key tile size
  and ordering. It reads the page table and excludes speculative future KV.
  All query/head programs can execute in parallel.
- Target LM-head outputs remain FP32 through argmax, avoiding an extra BF16
  rounding that can collapse distinct logits into a tie. Weights and other
  activations remain BF16.
- Graph capture uses persistent slot metadata and updates it on replay. The same
  attention and linear kernels run in eager and graph execution.
- Model-local configuration leaves the draft model and global PyTorch functions
  unchanged. The experimental executor supports this mode for BF16 hybrid models
  at TP=1; the benchmark restricts loading to dense Qwen3.5 text models.

`stable` defines a new, consistent target numerical baseline. It does **not**
promise the same tokens as the previous `fast` BF16 implementation or Hugging Face,
nor bitwise reproducibility across GPU architectures/compiler versions. Always
compare target-only and DFlash using the same numerical mode, weights and inputs.
`fast` retains the previous library kernels and their shape-dependent behavior;
use `--target-numerics fast` when reproducing historical performance reports.
No hard 24GB cap, INT4, DFlash2 or HTTP scheduler integration is implied.

## Reproduce

```bash
cd /root/mini-sglang
export PYTHONPATH="$PWD/python"
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B
DRAFT=/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B-DFlash
OUT=$(mktemp -d /root/autodl-tmp/runtime-results/stable-repro-XXXXXX)

run_bench() {
  "$PY" -B -m minisgl.runtime.benchmark \
    --model "$MODEL" --draft "$DRAFT" \
    --workload benchmark/runtime/workloads/chat-long4.jsonl \
    --max-context 1024 --batch-size 4 --gdn-extend packed \
    --target-numerics stable --cuda-graph --chat-template --show-text "$@"
}
run_bench --mode target --output "$OUT/target.json"
run_bench --mode fixed --block-size 8 --output "$OUT/fixed.json"
run_bench --mode adaptive --block-size 8 --output "$OUT/adaptive.json"
"$PY" -B -m minisgl.runtime.analyze \
  "$OUT/target.json" "$OUT/fixed.json" "$OUT/adaptive.json" --per-request

# Independent kernel, state, cache and generation regressions.
"$PY" -B -m pytest -o addopts='' tests/cpu tests/gpu -q
"$PY" -B benchmark/runtime/validate_target_numerics.py \
  --model "$MODEL" --draft "$DRAFT" --output "$OUT/numerics-state.json"
"$PY" -B benchmark/runtime/validate_batch_graph.py \
  --model "$MODEL" --draft "$DRAFT" --output "$OUT/cache-state.json"
"$PY" -B benchmark/runtime/validate_stable_suite.py \
  --model "$MODEL" --draft "$DRAFT" --output-dir "$OUT/generation"
```

The last command checks 256/512-token caps, fixed/adaptive DFlash, sequential and
parallel verification, eager/graph, batch 1/4/8, block 8/16, continuous refill and
CPU prefix-cache restoration. It retains every prompt, response, acceptance
counter and process log. It exits with failure on token mismatch or missing CPU
cache coverage. Outputs may stop at EOS before their configured cap.

Use a fresh output directory. These are correctness coverage runs, not repeated
performance estimates. A successful process exit by itself is not evidence of
token parity; the validator explicitly compares the generated token arrays.
