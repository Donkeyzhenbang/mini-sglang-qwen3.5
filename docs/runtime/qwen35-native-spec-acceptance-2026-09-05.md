# Qwen3.5-4B 原生 MTP / DFlash：batch=4 GPU 验收（2026-09-05）

本轮在 MiniSGLang 原生引擎中完成 MTP-1、MTP-3 和 DFlash 的图执行优化。
固定 MTP-3、DFlash block=8 已在相同 stable target-only 基线上获得加速，并逐 token 对齐。
这份报告更新了 2026-09-03 报告中“原生投机仍慢于 target-only”的阶段性结论。

## 环境与统计口径

- 单卡 NVIDIA RTX 4090，Qwen3.5-4B BF16；Torch 2.9.1+cu128，Python 3.12.3。
- target 权重：`/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B`。
- DFlash 权重：`/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B-DFlash`，六层 draft，训练 block=16。
- batch=4，parallel verify，packed GDN，`--target-numerics stable --cuda-graph`。
- 最大上下文 4096；GPU/CPU prefix cache 均为 0。性能收益与 HiCache 命中无关。
- 使用同一四条中文解释/代码 prompt。256-token 实验 warmup=2、repeat=5；
  512-token 实验 warmup=2、repeat=3。
- decode tok/s = 所有 wave 的 decoded_tokens / 所有 wave 的 decode wall time。
  排除每请求的第一个 prefill token；不是把四个重叠请求的时间相加。
- E2E tok/s 包含测量 wave 内的 prefill。模型加载和预热建图不计入稳态测量。
  测量期间新建图的耗时仍包含在 wall time 中。
- 原始 JSON 保留 prompt、输入/输出 token IDs、完整回答、接受率、分阶段时间、
  graph 计数、显存峰值、参数和 Git revision。

## 性能与精度

### 四请求各输出 256 token，五轮

| 模式 | decode tok/s | 相对 target | E2E tok/s | draft 接受率 | token 对齐 |
|---|---:|---:|---:|---:|---|
| target-only | 330.23 | 1.000× | 328.53 | — | 基准 |
| MTP-1 | 415.94 | 1.260× | 412.87 | 79.37% | 20/20，5120 token |
| MTP-3 | 427.35 | 1.294× | 423.72 | 58.98% | 20/20，5120 token |
| DFlash block=4 | 344.66 | 1.044× | 342.63 | 47.49% | 20/20，5120 token |
| DFlash block=8 | 370.53 | 1.122× | 368.22 | 27.30% | 20/20，5120 token |

MTP-3 每 wave 为 415.02–433.11 tok/s；DFlash block=8 为 369.05–371.75 tok/s。
表中报告全轮聚合吞吐，没有挑最高的一轮。
MTP 的三个 draft token 与 DFlash block=8 的七个 draft token 分母不同；
接受率不能直接当作加速比。

### 四请求各输出 512 token，三轮

| 模式 | decode tok/s | 相对 target | E2E tok/s | draft 接受率 | token 对齐 |
|---|---:|---:|---:|---:|---|
| target-only | 327.86 | 1.000× | 326.90 | — | 基准 |
| MTP-3 | 437.55 | 1.335× | 435.80 | 61.52% | 12/12，6144 token |
| DFlash block=8 | 412.41 | 1.258× | 410.87 | 31.70% | 12/12，6144 token |

MTP-3 峰值 allocated/reserved 约 9.18/9.30 GiB；
DFlash block=8 约 10.65/12.01 GiB，均在本机显存预算内。

### 混合长度与槽位变化

额外将四条 prompt 重新排序组成 12 个请求，输出限制轮换为 1、17、73、129。
MTP-3 和 DFlash block=8 各 12/12 请求、660 token 与 target 完全一致。
聚合 decode 吞吐分别为 210.60、184.71 tok/s，target 为 144.33 tok/s。
这组用于检查请求提前退出、剩余 batch 缩小和 graph/eager 切换；
单个短 wave 可能不加速，不能据此宣称任意请求都更快。

主验收中，MTP-3 与 DFlash block=8 各覆盖 44 个请求、11924 个输出 token，
全部逐 token 对齐。另测 MTP-1、DFlash block=4 各 20 个请求。

## 代码改动与瓶颈解释

### 1. 直接重放 journal 中的接受区间

GDN 卷积及递推 kernel 支持独立 start/end。
拒绝后的提交直接访问验证 journal 的原始投影张量，
不再每层 `cat` 接受片段，也不再每层创建 CPU→GPU 元数据。
保持原递推顺序；未参与回滚的请求状态不受影响。

### 2. 跨层状态备份与恢复

用 GPU 指针表将各层 conv/SSM 状态 gather/scatter 合并成两次 kernel launch。
没有移动现有状态分配，decode CUDA graph 中的地址仍有效。
快照独立持有数据，支持多个未释放快照和乱序子集恢复。

### 3. target 验证与 GDN 提交 CUDA graph

stable 数值模式下，统一 block=2/4/8/16 的验证可以捕获 CUDA graph。
位置和请求槽是动态 GPU 输入，GDN journal 张量由图持有。
捕获和预热会恢复初始 recurrent state，避免额外推进状态。
ragged 尾部使用 eager 回退。拒绝后的 GDN journal 重放也捕获图，
动态更新接受区间，而不重新执行整个 target。

此前 CUDA graph 只覆盖单 token decode，投机的多 token verify 仍走 eager；
再叠加逐层状态处理和 draft Python/kernel 启动开销，
足以抵消接受 token 带来的收益。本轮没有修改 greedy verifier 来换速度。

### 4. DFlash draft 图及融合

- 为四个请求分配固定地址的六层 confirmed-context KV 存储。
- 在图中执行 FC、六层 attention/MLP 和 argmax。
- 只写已确认 context 的 KV，speculative noise 不进入持久缓存。
- 首次长 prefill、不同 block 混合和超过图缓存上限的形状安全回退到 eager。
- 允许 eager/graph 之间切换；必要时导入独立 KV，再保持固定存储视图。
- 融合 RMSNorm、SiLU×Mul、RoPE，保留原有 BF16 中间舍入语义；
  cosine/sine 表在所有层和 request context 间共享。
- padding 超出实际上下文时，RoPE 表读取有边界保护。
- 384 MiB 的 draft KV 存储（本配置）和共享 RoPE 表计入内存准入预算。
- MTP 递归 proposal 的 argmax 改为每步一次批量 CPU 传输。

### 5. Adaptive 成本反馈与图形状修复

旧策略把首次建图和恢复 draft 历史的补算耗时当作常态成本，导致错误选择 block。
现在把这些回合标记为 startup，仍完整计入吞吐和请求时间，
但不把它们当作稳态成本样本；在采到稳态样本前继续完成候选动作的校准。
Adaptive 的 context padding 统一为 16，减少切换 block 产生的冗余图；
draft 图数量上限为 32，创建新图前保留至少 2 GiB 可用显存余量。

同一 256-token、五轮实验从 267.25 提升到 356.50 tok/s，
相对 target 为 1.080×，20/20 请求、5120 token 全部对齐。
峰值 allocated/reserved 约 11.03/13.93 GiB。
本负载固定 DFlash block=8 的 370.53 tok/s 仍更好，
因此默认复现选固定模式，不把 adaptive 说成自动达到全局最优。

### 6. 回归范围

最终执行 `python -m pytest -o addopts='' -q tests/cpu tests/gpu`：
**94 passed**。

GPU 测试检查数值 kernel、非连续/乱序槽、部分前缀恢复、未参与请求不被修改、
多个快照生命周期、draft reset 后槽复用、eager→graph→eager 缓存一致性、
长短请求混合以及最大上下文附近的 padding。
journal graph 测试还检查捕获不污染状态、动态接受长度和重写 journal 后正确读取新数据。

## 复现

云端当前分支：`feat/hybrid-memory-runtime`。
无需重装 Torch、SGLang 或 vLLM 环境。

```bash
cd /root/mini-sglang
bash benchmark/runtime/run_native_spec_acceptance.sh \
  /root/miniconda3/bin/python \
  /root/autodl-tmp/runtime-results/reproduce-native-spec-$(date +%Y%m%d-%H%M%S)
```

脚本依次运行 target、MTP-3、DFlash block=8；每个模式都是四请求批量执行，
不是把四句完整生成串行运行。模式之间使用独立进程，避免同时占用 GPU 干扰成绩。
输出 `summary.json`；任一 token 不一致或默认速度门槛未满足，脚本返回非零。
添加 `SHOW_TEXT=1` 环境变量会把 prompt 和回答同时打印进各模式日志；
完整回答无论是否设置该变量都会保存在 JSON 中。

现有结果可以单独审计：

```bash
python benchmark/runtime/compare_native_spec.py \
  TARGET.json MTP.json DFLASH.json --summary summary.json
```

消融开关：
`--no-draft-cuda-graph` 关闭 DFlash draft 图；
`--no-verify-cuda-graph` 关闭多 token 验证图及其持久 journal 图，
但保留 target 单 token decode 图。

原始实验目录：
`/root/autodl-tmp/runtime-results/mtp-20260905`。
主要结果文件：
`target-final.json`、`mtp3-accepted.json`、`mtp1-accepted.json`、
`dflash-b4-accepted.json`、`dflash-b8-journalgraph.json`，
以及 `acceptance-summary.json`、`long-summary.json`、`ragged-summary.json`。

## 使用边界

这里的“对齐”是同一个 stable target 数值策略下的 greedy token 一致性，
不是声称默认 fast BF16、不同框架、随机采样也一定逐 token 相同。
本次实测为 dense 4B BF16、TP=1 和离线 batch=4；不代表 27B Int4、
多卡、HTTP 在线服务负载或任意输入都已完成验收。
测试能证明这些路径通过回归，不能证明不存在任何潜在 bug。
默认复现选择已经重复验收的 MTP-3 / DFlash block=8。


## 2026-09-06 一键脚本独立复跑

在干净的 `20b06d1` 上直接运行本报告中的脚本，完成三个独立进程、各五轮测试，
自动验收 `passed: true`。没有手工改动输出 token 或计时结果。

| 模式 | decode tok/s | 相对 target | E2E tok/s | token 对齐 |
|---|---:|---:|---:|---|
| target-only | 328.06 | 1.000× | 326.08 | 基准 |
| MTP-3 | 427.14 | 1.302× | 423.71 | 20/20，5120 token |
| DFlash block=8 | 368.26 | 1.123× | 365.74 | 20/20，5120 token |

独立复跑结果位于
`/root/autodl-tmp/runtime-results/reproduce-native-spec-20260906`。
这是对已提交工具和代码的再次验证；上文 2026-09-05 表格保留原始数据。
