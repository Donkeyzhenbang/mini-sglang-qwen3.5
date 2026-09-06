# Native MTP draft CUDA graphs

The complete one/three-step greedy draft chain now runs in a CUDA graph for
uniform blocks and up to 16 newly confirmed states. `--cuda-graph` enables it;
`--no-draft-cuda-graph` retains eager proposals with the same target graphs.

Persistent KV holds only confirmed target-history inputs. Recursive predictions
stay in graph-private scratch. Captures use previous-length masks and idempotent
confirmed writes, with dynamic slots, lengths, counts and shifted token IDs.
Long initial catch-up, mixed tail blocks and bounded graph-pool exhaustion fall
back to eager; subsequent graph execution imports confirmed KV. Persistent KV
bytes are included in admission estimates (this is not a hard allocator cap).
Invalid ragged padding no longer indexes beyond the MTP RoPE table.

RTX 4090, Qwen3.5-4B BF16 stable numerics, greedy, four raw prompts, batch=4,
256 output tokens/request, two warmup waves, five measured waves, prefix cache
disabled. End-to-end output throughput includes prefill:

| Variant | tok/s | Versus target |
| --- | ---: | ---: |
| Target | 328.68 | 1.000x |
| MTP3 eager draft | 422.72 | 1.286x |
| MTP3 graph draft | 481.79 | 1.466x |

All 20 measured requests / 5120 tokens match the stable target exactly. Draft
acceptance remains 58.98%. Draft wall time across the five waves drops from
4505 ms to 3115 ms. These repeated four prompts are not a broad quality suite.
CPU/GPU regression: 7 tests pass, covering confirmed KV, reset, slot reordering,
ragged counts, eager/graph transitions, context boundaries, and MTP1/MTP3.

Evidence: `/root/autodl-tmp/runtime-results/graph-opt-20260906/` (before,
mtp3-graph.json, mtp3-graph-summary.json, mtp-graph-tests.log). Further DFlash
profiling and broader graph regression are recorded separately.
