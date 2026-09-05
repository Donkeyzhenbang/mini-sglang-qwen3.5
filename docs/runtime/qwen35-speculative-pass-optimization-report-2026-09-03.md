> 更新：2026-09-05 原生 MTP/DFlash 已通过 batch=4 精度与加速验收，见 [最新 GPU 报告](qwen35-native-spec-acceptance-2026-09-05.md)。下文保留当时的阶段性结果。

# Qwen3.5 投机解码与编译 Pass 优化技术报告

日期：2026-09-03

> 状态更新：随后已完成 MiniSGLang 最小原生 Qwen3.5 MTP-1/MTP-3 实现，包括嵌入式权重加载、target final hidden 捕获、右移 token 对齐、独立 MTP KV、batch 内三步递推和 benchmark 入口。当前仍需 GPU token parity 与性能复测，MTP draft CUDA Graph 尚未实现。

## 1. 结论

本轮分别检查了三个问题：Qwen3.5 MTP 在成熟框架里是否能加速、MiniSGLang DFlash 为什么低于 target-only，以及哪些 vLLM/SGLang 优化值得移植。

已有 RTX 4090 实测表明，**Qwen3.5 MTP 本身并不慢**。SGLang 0.5.9 在 batch=4、生成 512 token 时，target-only 为 312.00 tok/s，MTP-1 为 373.54 tok/s，MTP-3 为 445.56 tok/s，分别提升 1.20x 和 1.43x。MiniSGLang 当前 DFlash 慢，主要来自 draft 模型更重、draft/多 token verify 仍走 eager，以及细碎 GEMM 和 kernel launch 开销。

本轮已完成无 GPU 条件下可以安全落地的结构优化：DFlash QKV 融合、gate/up 融合、六层 context-KV 跨层融合、target tap 冗余 clone 消除、原生 MTP-1/MTP-3、数值诊断增强，以及完整 GPU 复测矩阵。CPU 测试 65/65 通过；真实 Qwen3.5-4B checkpoint 的 230.03 MiB MTP 权重也已在 CPU 上 strict load 成功。GPU 性能收益尚未复测，因此本文不把结构变化写成已实现的加速数字。

## 2. 已有实测基线

测试模型为 Qwen3.5-4B BF16，单卡 RTX 4090。投机对照均关闭 prefix cache。

### 2.1 SGLang Qwen3.5 MTP

| 模式 | batch | 生成长度 | 输出 tok/s | 相对 target | draft 接受率 |
|---|---:|---:|---:|---:|---:|
| target-only | 1 | 512 | 91.14 | 1.00x | - |
| MTP-1 | 1 | 512 | 101.01 | 1.11x | 65.9% |
| MTP-3 | 1 | 512 | 109.66 | 1.20x | 40.4% |
| target-only | 4 | 512 | 312.00 | 1.00x | - |
| MTP-1 | 4 | 512 | 373.54 | 1.20x | 80.4% |
| MTP-3 | 4 | 512 | 445.56 | 1.43x | 58.6% |

这排除了“Qwen3.5 MTP 必然低于 target-only”。如果 MiniSGLang 原生 MTP 后续仍慢，应优先检查执行图、KV/GDN 状态、batch 调度和 kernel 数量。

MiniSGLang 已增加最小原生 MTP executor。原有 target loader 仍跳过 `mtp.*`，由独立 MTP loader 只读取嵌入式 MTP 参数，避免把约 230 MiB 权重并入 target 模型。SGLang 数据仍是性能参照；MiniSGLang 新路径已于 2026-09-03 在 RTX 4090 上完成首轮功能与性能复测，详见文末 GPU 复测补充。

### 2.2 MiniSGLang DFlash

| 模式 | 生成长度 | 输出 tok/s | 相对 target | 接受率 |
|---|---:|---:|---:|---:|
| target-only, batch=4 | 512 | 328.04 | 1.00x | - |
| fixed block=8, batch=4 | 256 | 195.02 | 0.59x | 24.48% |
| fixed block=4, batch=4 | 256 | 203.72 | 0.62x | 45.47% |
| adaptive, batch=4 | 512 | 288.53 | 0.88x | 37.75% |

GDN state journal 已把 block=8 从约 162 提升到 195 tok/s，把 adaptive-512 从约 183 提升到 289 tok/s。当前主要矛盾已经从 rollback/state copy 转向 draft 与 verify 的执行开销。

## 3. 严格无损与 batch 数值差异

同一 SGLang 模式、同一 batch shape 连续运行三次，token IDs 完全一致。但同一个首条 prompt 在 target-only batch=1 与 batch=4 之间从第 86 个生成 token 开始不同；MTP-1 和 MTP-3 改变 batch shape 后分别从第 53、51 个 token 开始不同。

开启 SGLang `enable_deterministic_inference` 后，target batch=1 与 batch=4 在 256 token 诊断中完全一致，但性能明显下降：

| target 数值模式 | batch=1 tok/s | batch=4 tok/s | batch shape 一致性 |
|---|---:|---:|---|
| 普通 BF16 | 90.54 | 314.01 | 否，第 86 token 首次不同 |
| deterministic | 30.39 | 118.56 | 是 |

batch shape 会改变 GEMM、Attention 和 GDN reduction 的实现路径。微小 BF16 误差在 recurrent state 中累积，最终可能跨过 argmax 边界。投机解码的“无损”是结果等同于**同一 target 数值执行路径**，并不保证不同 reduction layout 的 token 永远一致。

后续应分别报告同 shape 重复稳定性、speculative 与同数值策略 target 的一致性、batch shape 一致性，以及 stable/deterministic 策略的性能成本。

## 4. DFlash 慢于 target-only 的原因

1. 当前 DFlash checkpoint 有 6 个 Transformer layer；一次 MTP draft step 只执行 1 个训练层，两者 draft 成本不同。
2. MiniSGLang CUDA Graph 当前只覆盖 target 单 token decode；DFlash draft 和 target 多 token verify 仍为 eager。
3. 原实现每层分别运行 Q/K/V 和 gate/up projection。小 batch、小 token block 时，kernel 启动开销占比很高。
4. 接受率依赖 prompt。已有四条 workload 中，代码生成 prompt 的 block=8 接受率约 67%，短 KV-cache 解释 prompt 只有约 11%。
5. speculative round 只有在接受进度足以摊薄 draft、verify 和 state commit 时才获利。adaptive controller 可以减少无效投机，但不能消除 eager kernel 开销。
6. Qwen3.5 词表约 248K。draft 和 verify 若物化完整 FP32 logits，会产生较大的显存流量和临时张量。

## 5. 本轮代码改动

### 5.1 DFlash projection 融合

运行时继续兼容官方 checkpoint 中分离的权重名，在加载时一次性打包：

- `q_proj/k_proj/v_proj` → `qkv_proj`；
- `gate_proj/up_proj` → `gate_up_proj`。

每个 draft layer 的主要 projection GEMM 从 5 个降到 3 个，不增加数学 FLOPs，也不保留两份参数。

### 5.2 六层 context-KV 跨层融合

六个 draft layer 使用相同的 target context。现在把每层 QKV 参数中的 K/V slice 合成一个只读、约 60 MiB 的权重缓冲区，用一次大 GEMM 代替六次小 context projection。fork 后的 request context 共享只读缓冲区，各自保留独立 draft KV state。显存准入预算已计入 60 MiB，可用 `--no-draft-context-kv-fusion` 做消融。

该结构与 vLLM 0.22 `qwen3_dflash.py::precompute_and_store_context_kv` 以及当前 SGLang DFlash fused context-KV 路径一致。CPU 数值等价测试已通过，GPU 是否加速必须由矩阵实验确认。

### 5.3 target tap 拷贝消除

原代码保存 target hidden tap 时执行 `(x + residual).clone()`。加法本身已经生成新 tensor，额外 clone 没有所有权作用。本轮移除该 clone；当前 checkpoint 采集八个 target layer tap，每轮减少八次冗余拷贝。

### 5.4 诊断与复测工具

SGLang MTP benchmark 新增 attention backend、deterministic inference 和 FP32 LM-head 开关。summarizer 新增同 shape 重复一致性、batch shape 一致性和首个不同 token 位置。

新增 `run_qwen35_spec_matrix.sh`，使用独立进程测试 target、fixed block=4/8、adaptive、stable/fast 数值策略、CUDA Graph/eager 和 context-KV fused/unfused。统一使用 batch=4、warmup=1、repeat=3，并保存 prompt、输出、token IDs、接受率、分阶段耗时、图 replay 和显存峰值。

## 6. vLLM/SGLang Pass 优化候选

| 优先级 | 优化 | 预期作用 | 风险或前置条件 | 估算 |
|---|---|---|---|---:|
| P0 | 固定 block 2/4/8/16 的 draft CUDA Graph | 消除六层 draft 的 Python 与 launch 开销 | ragged padding、KV 地址和输出 buffer 必须稳定 | 3–5 人日 |
| P0 | packed multi-token verify piecewise graph | 消除当前最大的 eager target 路径 | GDN journal 输出和可变 token 数需静态 buffer | 4–7 人日 |
| P0 | fused LM-head + argmax | 避免物化 `[token, 248K]` FP32 logits | 保持 stable/fast 数值策略与 TP 语义 | 3–5 人日 |
| P1 | Q/K RMSNorm + RoPE 融合 | 减少 norm、reshape、RoPE kernel | 验证 head_dim 128/256 与 BF16 误差 | 2–4 人日 |
| P1 | GDN QKVZ + BA projection 融合 | 两次输入 GEMM 合并为一次 | checkpoint layout 与 TP 分片 | 2–4 人日 |
| P1 | target tap capture sink | 八个 hidden state 直接写稳定 buffer | graph replay 下的生命周期 | 2–3 人日 |
| P1 | DFlash concat+FC 下推或分块 | 尝试避免 20480 维临时 target feature | 八个 partial GEMM 可能更慢，必须 profile | 2–4 人日 |
| P2 | 小型 FX/Inductor pass manager | 自动做 clone/no-op 消除和注册融合 | compile cache key、自定义 op graph break | 5–8 人日 |
| P2 | Int4 target + BF16 draft | 为 24GB 单卡释放 cache/graph 空间 | Mini target 量化尚未实现 | 1–2 周 |

vLLM 0.22 可直接参考 `vllm/compilation/passes/` 下的 `qk_norm_rope_fusion.py`、`rope_kvcache_fusion.py`、`clone_elimination.py`、`noop_elimination.py` 和 `pass_manager.py`。当前 SGLang 更偏显式模块和专用 kernel：DFlash 已使用融合 QKV/gate-up、fused context-KV、`torch.compile`，并把 draft sampling 纳入 CUDA Graph。

建议先做 P0 图覆盖，再做 fused LM-head/argmax，最后依据 Nsight 数据决定 QK-RoPE 或 GDN projection。现在直接搭通用 FX pass manager，可能只优化到非热点图，并增加 graph break 调试成本。

## 7. MiniSGLang 原生 MTP 工作量

本轮已经完成最小原生 MTP-1/MTP-3 的下列工作：

1. 加载目前被跳过的 `mtp.*` 权重，checkpoint 内约有一层、约 230 MiB MTP 参数；
2. 实现 input embedding norm、target hidden norm、concat+FC、训练 full-attention layer和共享 target LM head；
3. 为每个请求维护独立 MTP KV state 与 rollback length；
4. 接入现有 greedy verifier，支持 1-step 和重复 3-step proposal；
5. benchmark 支持 `--mode mtp --mtp-steps 1|3`，并纳入 target/MTP/DFlash 统一矩阵；
6. CPU 测试覆盖官方权重键重打包、batch/serial 一致性、增量 KV/全量重建一致性，以及投机态不会污染确认态。

当前属于可读 eager 功能版本。剩余工作是为固定三步 draft 与 verify shape 捕获 CUDA Graph，并在同一 deterministic/stable 策略下对齐 SGLang token IDs，再比较 acceptance、draft、verify、state 和总吞吐。验收目标参考已测 SGLang：长输出 batch=1 至少约 1.1x，batch=4 约 1.2–1.4x。达到前可以称为“原生功能支持”，还不能称为“性能支持”。

## 8. GPU 恢复后的复现与验收

```bash
cd /root/mini-sglang
bash benchmark/runtime/run_qwen35_spec_matrix.sh \
  /root/miniconda3/bin/python \
  /root/autodl-tmp/runtime-results/qwen35-spec-matrix
```

验收条件：

- fused/unfused context-KV 的 greedy token 完全一致；
- 融合不改变接受率，三轮重复中 draft time 稳定下降；
- 峰值显存增加接近声明的约 60 MiB；
- stable DFlash 与 stable target-only token 完全一致；
- fast 模式若出现 token 差异，必须单独报告；
- target/draft/verify graph replay counter 与预期相符。

当前实例 GPU 已恢复，MiniSGLang target/MTP-1/MTP-3 首轮矩阵已完成。SGLang MTP 已有独立实测对照；SGLang main 和 vLLM 0.22 的 DFlash 代码已经静态 review，其 DFlash 性能仍需在隔离环境复测，避免污染当前 Torch/kernel 组合。


## 9. 2026-09-03 GPU 复测补充

环境为单张 RTX 4090、Qwen3.5-4B BF16、四条 256-token 长回答、batch=4、parallel verify、packed GDN。CUDA Graph 当前只覆盖 target 单 token decode。

| 路径 | 数值模式 | 输出 tok/s | 相对 target | 接受率 |
|---|---|---:|---:|---:|
| target-only | stable | 332.85 | 1.00x | - |
| MTP-1 优化前 | stable | 260.61 | 0.78x | 79.37% |
| MTP-3 优化前 | stable | 272.17 | 0.82x | 58.24% |
| MTP-3 融合版 | stable | 299.56 aggregate / 302.66 median | 0.90x / 0.91x | 58.98% |
| target-only | fast | 327.49 | 1.00x | - |
| MTP-3 优化前 | fast | 299.64 | 0.91x | 56.40% |
| MTP-3 融合版 | fast | 313.37 | 0.96x | 56.40% |

融合版使用 FlashInfer Gemma RMSNorm、SiLU×Mul 和预计算 RoPE。stable MTP-3 五轮为 285.25–305.62 tok/s，中位数 302.66 tok/s；20/20 个请求与 stable target 的 token IDs 完全一致。stable draft 阶段每 wave 均值从约 1.18 s 降到 0.94 s。

GPU 复测修复了嵌套 text_config 的 MTP 识别、final-hidden CUDA Graph buffer 宽度为零、graph capture 失败后清理二次异常，并把误标为 DFlash 的输出改为通用 Speculation 标签。

当前 MTP-3 是 MiniSGLang 的最佳 MTP 设置，但还未超过 target-only。剩余差距集中在 eager 三步 draft、eager 多 token verify 和 GDN snapshot/journal commit。SGLang 的 MTP-3 已达到 1.43x，说明接受率不是主要瓶颈；下一阶段应实现 MTP draft graph、piecewise verify graph 和 fused LM-head+argmax。
