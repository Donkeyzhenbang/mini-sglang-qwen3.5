# GPU validation: 2026-08-31 (in progress)

Hardware: RTX 4090, 24564 MiB; CUDA capability 8.9; driver 595.71.05.
Runtime: Python 3.12.3, torch 2.9.1+cu128, Transformers 4.57.3,
FlashInfer 0.6.14, sgl-kernel 0.3.21. Existing SM89 loader workaround unchanged.
Raw logs/results: /root/autodl-tmp/runtime-results/2026-08-31/.

## First GPU fixes

- Single-token extend convolution now accumulates in FP32 before BF16 rounding;
  previously each tap product rounded in BF16, unlike the multi-token convolution.
- GDN beta preserves the projection-dtype sigmoid boundary used by Qwen3.5.
- Packed extend uses the same q/k normalization order as stepwise decode.
- Real head dimensions (K=V=128, H=16, HV=16/32), BF16, nonzero initial states:
  packed and stepwise outputs and terminal states pass exact equality.
- Single-token convolution matches grouped PyTorch convolution exactly in the
  regression fixture; history update no longer uses overlapping slice assignment.
- 46 CPU/GPU tests pass. Qwen3.5-0.8B recurrent/packed smoke generations match.

## Reference gate remains open

Transformers 5.13.0 from the existing separate environment runs the torch GDN
fallback. Some smoke prompts diverge from mini-SGLang. On request 0, token 1,
mini has a BF16 top-logit tie (20.25/20.25), while HF has 20.25/20.125.
The first-layer conv history is identical; recurrent state and layer activations
have small numerical differences. This is not a passed HF parity gate.
`benchmark/runtime/trace_target.py` records teacher-forced logits and layer taps
without allowing earlier generation differences to contaminate the comparison.

4B and matching DFlash weights are being downloaded from ModelScope, with file
hashes checked against the pinned Hugging Face revisions. No 4B/DFlash performance
claim has been validated yet. 27B GPTQ loading remains unsupported.

## Real 4B and DFlash validation

Both pinned weight sets are now present and SHA256-verified (ModelScope mirror,
identical weight hashes to the locked Hugging Face revisions). The 4B target,
fixed block-8 DFlash, and independent Transformers 5.13 + FLA 0.5.2 reference
produce identical 176 output tokens on the four-request smoke workload.
The reference still uses torch convolution; only GDN and gated norm use FLA.

Single-run exploratory 4B smoke: target 51.37 decode tokens/s, block-8 DFlash
101.52 decode tokens/s; mean progress 4.10 tokens/round. Peak allocated tensors:
8.20 GiB target, 9.46 GiB with draft. These are single-request experimental
adapter measurements without graphs or scheduler overlap, not a production
baseline or a general model speedup claim. Repeated trials follow.

`validate_transactions.py` passes exact KV, conv, SSM, auxiliary-feature and
logit restoration on GPU and CPU, plus rejection rollback/replay, for both
0.8B and 4B. Full-vs-split prefill logits differ by about 1% relative L2 for
the tested 1024-token input. The 0.8B shared-prefix workload has two generation
differences vs full recomputation; GPU/CPU/LRU cache outputs match each other.
Do not relax the token comparison gate or report those as an off-vs-on speedup.

Native draft was compared to reviewed z-lab/dflash model.py at commit
07ebd93db9f472af339b644bb70221ad8428328a (SHA256
f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e).
FP32 math-SDPA comparison with real draft weights passes atol/rtol 1e-4 across
incremental contexts and blocks 2/4/8/16; max absolute error 8.2e-6.
BF16 relative L2 error is 1.09%-1.27%, reported separately, not bitwise parity.
No remote model code is used in the native inference path.

The synthetic workload now stores its shared prefix in row zero, ensuring
short four-request groups actually exercise reuse. Results record git dirty
state; HF results record which optional GDN operators are installed.

## Repeated trials and expanded gates

18 clean-worktree trials at 31c7bdb (6 modes x 3 randomized repetitions),
4 requests / 176 output tokens each. All smoke outputs match the HF reference.
Median decode tokens/s: target 47.50, block2 58.16, block4 81.12,
block8 110.88, block16 83.18, adaptive 102.53. The block8 observed range is
75.43-114.34 and target range 39.58-52.38: pilot variability is substantial.
The 2.33x block8 and 2.16x adaptive ratios apply only to this experimental
single-request baseline. Adaptive does not beat the best fixed block here.
Block8 component times: draft ~23%, verify ~48%, checkpoint/restore/replay ~29%.

The expanded eight-chat workload FAILS strict parallel token parity: fixed8
matches target on 6/8 requests, adaptive on 6/8, and HF+FLA on 5/8. Never turn
these runs into a validated speedup claim. `probe_verify_shapes.py` reproduces
argmax changes from an identical history/state when only the verify block
shape changes: tied BF16 logits (26.0/26.0 or 21.625/21.625) move by 0.125.
`--verify-mode sequential` is now an explicit diagnostic oracle. It uses the
same DFlash acceptance and rollback loop but executes target tokens one at a
time; fixed8 then matches target on all eight chats. It intentionally sacrifices
parallel verification acceleration. Default `parallel` remains experimental;
this does NOT resolve general numerical consistency or HF parity.

4B shared-prefix synthetic cache runs restore correctly and all GPU/CPU/LRU
outputs match each other, but only 11/12 match no-cache full recomputation.
Synchronous whole-bundle caching increases TTFT on this workload. Neither a
cache speedup nor a benefit of cost eviction over LRU has been demonstrated.

Injected software-budget pressure with real target+draft weights releases
57,223,168 bytes of GPU prefix cache to CPU, restricts feasible blocks to [1],
and restores [1,2,4,8,16] when the budget is restored. This does not simulate
physical 24GB exhaustion or establish 27B capacity. `validate_memory_pressure.py`
is the reproducible integration check.

A 128/512/2048/4096-input-token synthetic sweep completes with a peak of
10.29 GiB allocated tensors for target+draft; strict token parity passes 3/4
length cases (512,2048,4096), fails the 128 case. This crosses the draft sliding
window boundary, but does not certify arbitrary long-context quality or maximum
context capacity. Allocation is distinct from reserved CUDA memory/device usage.

Priorities from these measurements: batch-shape numerical consistency and wider
quality gates; reduce full-target replay after speculative rejection; pack and
pin hybrid state transfers before evaluating asynchronous offload; calibrate
adaptive exploration on changing workloads using held-out data. Production
batching/graphs, full HiCache, 27B GPTQ and DFlash2 remain out of scope so far.
