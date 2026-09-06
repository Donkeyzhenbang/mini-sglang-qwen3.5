# DFlash shared vocabulary projection

The post-anchor slice `hidden[:, 1:]` is a noncontiguous 3D tensor. Passing it
straight to `F.linear` selected a strided batched GEMM, rereading the shared
248320 x 2560 BF16 vocabulary weight for each request. CUDA graph capture
preserved this inefficient operation rather than fixing it.

Flattening the batch/token dimensions before projection selects one GEMM. Both
eager batched proposals and the captured draft now use this layout. No target
arithmetic, verification, masks, RoPE or acceptance rules changed.

A warmed 4090 CUDA-graph microbenchmark with checkpoint dimensions (synthetic
weights and activations; 20 samples/shape) measures median projection latency:

| Batch | Strided ms | Flat ms |
| --- | ---: | ---: |
| 1 | 1.480 | 1.367 |
| 2 | 2.862 | 1.386 |
| 3 | 4.295 | 1.440 |
| 4 | 5.696 | 1.447 |

These synthetic logits are bitwise equal; full-model validation is separate.
The one-wave before profile records 83 of the slow GEMM calls taking 443 ms.
Profiling perturbs wall time and is not used as end-to-end throughput evidence.

Real Qwen3.5-4B BF16, stable target numerics, batch=4, block=8, greedy, 256 output
tokens, warmup=2/repeat=5, no prefix cache: DFlash improves 363.87 -> 412.76 tok/s
(+13.44%); target=328.68 tok/s (+25.58%). All 20 requests/5120 output tokens are
identical to target-only. Acceptance stays 27.30%; five-wave draft time drops
4726 -> 3074 ms. All 96 CPU/GPU tests pass, including new MTP graph regressions.

Reproduce the isolated kernel with `benchmark/runtime/probe_draft_head.py` and
the full length/block/concurrent-ragged matrix with
`benchmark/runtime/run_graph_opt_validation.sh`.
Evidence: `/root/autodl-tmp/runtime-results/graph-opt-20260906/`.
