# Qwen3.5 MTP baseline and DFlash state-journal results

Measured on one RTX 4090 with BF16 Qwen3.5-4B. Prefix caching was disabled,
four prompts were submitted in one batch, CUDA graphs were enabled, and model
loading/graph capture were excluded. Results are retained under
`/root/autodl-tmp/runtime-results/vllm-compare-W7Nlns` on the experiment host.

## SGLang MTP baseline

The target checkpoint contains one trained MTP layer (15 tensors, about 230
MiB), so no separate draft checkpoint is needed. SGLang 0.5.9 was installed in
an isolated `--system-site-packages` environment and reused Torch 2.9.1+cu128,
sgl-kernel 0.3.21, and Transformers 4.57.3. Package downloads used the Tsinghua
mirror. The benchmark runs each mode in a fresh process, warms each exact shape,
and takes the median of three fixed-length runs.

| Mode | Batch | Output length | Output tok/s | vs target | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| target | 1 | 256 | 90.54 | 1.000x | — |
| MTP 1 step | 1 | 256 | 100.53 | 1.110x | 66.7% |
| MTP 3 steps | 1 | 256 | 106.40 | 1.175x | 39.0% |
| target | 1 | 512 | 91.14 | 1.000x | — |
| MTP 1 step | 1 | 512 | 101.01 | 1.108x | 65.9% |
| MTP 3 steps | 1 | 512 | 109.66 | 1.203x | 40.4% |
| target | 4 | 256 | 314.01 | 1.000x | — |
| MTP 1 step | 4 | 256 | 367.60 | 1.171x | 77.7% |
| MTP 3 steps | 4 | 256 | 413.32 | 1.316x | 53.1% |
| target | 4 | 512 | 312.00 | 1.000x | — |
| MTP 1 step | 4 | 512 | 373.54 | 1.197x | 80.4% |
| MTP 3 steps | 4 | 512 | 445.56 | 1.428x | 58.6% |

These SGLang BF16 target and MTP runs are not strictly token-identical. The
first difference occurred at token 53–183 for three of the four batch prompts;
one prompt remained identical. Therefore these numbers demonstrate a useful MTP
throughput baseline, not strict lossless decoding across different execution
shapes.

SGLang 0.5.9 spec-v2 overlap initially failed in two places for text-only
Qwen3.5: MRoPE did not classify `DRAFT_EXTEND_V2` as extend, and the hybrid GDN
backend rejected that forward mode. The guarded 0.5.9 backport in
`patch_sglang_059_mrope.py` follows the newer SGLang control flow. The overlap
smoke test now completes and reports `disable_overlap_schedule=False`; the
controlled table above keeps overlap disabled in both target and MTP modes.

## MiniSGLang DFlash state journal

The old rejection path restored the GDN checkpoint and executed the accepted
target prefix through every model layer a second time. The new packed verify
journal retains each GDN layer's projected inputs and gating values. After a
rejection it restores the checkpoint and runs only convolution and recurrent
state updates for the accepted prefix. Target layers, attention, MLP, LM head,
and feature extraction are not repeated.

| Batch 4 run | Decode tok/s | Acceptance | State restore/commit ms | vs target |
|---|---:|---:|---:|---:|
| target, 256 | 328.98 | — | — | 1.000x |
| old fixed block 8, 256 | 162.22 | 24.48% | 2278.01 | 0.493x |
| journal block 8, 256 | 195.02 | 24.48% | 745.76 | 0.593x |
| journal block 4, 256 | 203.72 | 45.47% | 678.89 | 0.619x |
| journal adaptive, 256 | 276.83 | 37.89% | 168.64 | 0.842x |
| target, 512 | 328.04 | — | — | 1.000x |
| old adaptive, 512 | 183.00 | 49.11% | 2084.99 | 0.558x |
| journal adaptive, 512 | 288.53 | 37.75% | 271.55 | 0.880x |

All four 256-token and all four 512-token journal outputs are exactly identical
to their stable target-only references. Peak allocated memory was 9.77 GiB for
block 8 and 9.80 GiB for adaptive 512, close to the previous DFlash runs.

Adaptive selection now chooses one block for the whole GPU wave and learns from
total accepted progress divided by the actual shared draft, verify, and state
commit cost. This avoids mixed ragged shapes and incorrectly charging one
request for another request's verification. The exploration interval increased
from 8 to 32 rounds to reduce repeated probing after cold start.

The result remains slower than target-only on this workload. The next limiting
costs are the four-layer DFlash draft and multi-token target verification, while
the low long-context block-8 acceptance makes those costs hard to amortize.

## Reproduction

Create the isolated SGLang environment only on the tested base stack:

```bash
source /etc/network_turbo
bash benchmark/runtime/setup_sglang_mtp_env.sh \
  /root/autodl-tmp/runtime-results/sglang-mtp-env
```

Run the target/MTP matrix. The runner places Triton and TorchInductor artifacts
on the executable data disk because this host mounts `/dev/shm` with `noexec`:

```bash
bash benchmark/runtime/run_sglang_mtp.sh \
  /root/autodl-tmp/runtime-results/sglang-mtp-env/bin/python \
  /root/autodl-tmp/runtime-results/mtp-reproduction \
  /root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B
```

The raw JSON includes prompts, generated text, token IDs, wall times, speculative
counters, package versions, GPU name, and engine arguments. `summarize_sglang_mtp.py`
audits prompt equality, output lengths, prefix-cache misses, and token agreement
before calculating speedups.
