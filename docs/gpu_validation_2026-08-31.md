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
