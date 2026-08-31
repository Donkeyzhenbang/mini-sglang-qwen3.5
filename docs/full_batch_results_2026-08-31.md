# Qwen3.5 + DFlash：批处理优化实测记录

这是旧 `fast` 数值路径的历史记录。后续修复已通过长回答对照，见
[stable 数值路径验证](stable_target_results_2026-08-31.md)；复现本页数据需显式指定 `--target-numerics fast`。

日期：2026-08-31。GPU：RTX 4090 24GB；target：Qwen3.5-4B BF16；draft：对应 DFlash v1。
性能与模型验证的代码提交为 `9f872a99b7735ffcea11f38dbc48d2ec1811e918`。
原始实验保存在云端 `/root/autodl-tmp/runtime-results/full-batch-C3sUfk`。
[复现命令](full_batch_runtime.md) · [机器可读结果](full_batch_results_2026-08-31.json)

## 本轮代码变化

| 改动 | 实际行为 |
|---|---|
| Batched prefill | 缓存恢复后，把不同长度的 prompt suffix 合并前向；仅计算各请求最后位置的 logits |
| Batched draft | 每层 FC/QKV/attention/MLP 及输出投影合批执行；使用各请求的绝对位置、padding mask 和独立 draft KV |
| 连续补位 | 新增 `--continuous-batching`，请求完成后在轮次间复用空闲 slot，不再等整组最长请求结束 |
| GDN 卷积优化 | 新增短序列 ragged Triton kernel；verify 卷积不再逐请求调用 PyTorch conv1d |
| 元数据与 checkpoint | GDN 各层共用 sequence metadata；checkpoint/restore 按层合批；按最长活动上下文统一查询显存预算 |
| 同步与观测 | draft 和 verify 各自一次批量 token 回传；去掉逐请求同步；记录真实 prefill/draft/verify batch size 和补位时间 |
| 数值 bugfix | decode 卷积改为 FP32 乘加、输入 dtype 舍入、再 SiLU，与 BF16 prefill/verify 的步骤一致 |

CPU 卷积回退路径也统一为 FP32 乘加，并补充独立 CPU BF16 回归测试；这项后续修复不改变已测 GPU 路径。

核心提交：`c1bb9e4`（批处理、补位、GDN 优化），`9f872a9`（卷积数值修复）。
这是实验 CLI 路径，尚未把 DFlash 接入 MiniSGLang 的 HTTP/overlap scheduler。

## 可以成立的性能结论

下面两组比较都通过相同 workload hash、模型配置、GPU 和完整输出 token 检查。
均为三次测量的中位数，保留所有单次结果；不代表置信区间或生产环境性能保证。

| 对比 | 修改前 / 固定波次 | 修改后 / 连续补位 | 提升 |
|---|---:|---:|---:|
| 4 请求 DFlash，block=8，无 prefix cache，交替运行旧版/新版；decode tokens/s | 96.71 | 120.57 | 24.7% |
| 8 请求、最多并发 4、GPU cache=512MiB、host=1024MiB；端到端 output tokens/s | 123.48 | 196.59 | 59.2% |

第一组旧版为 `f8410e8`，从独立 detached worktree 运行，主工作目录不切换分支。
旧版三次为 96.85 / 64.33 / 96.71，新版为 120.57 / 117.40 / 122.70 tokens/s。
第二组固定波次为 123.48 / 114.11 / 125.06，连续补位为 204.07 / 196.59 / 156.86。
单次波动较大，因此没有只选最快一次报告。

连续补位与固定波次的 TTFT 含不同的排队语义，不能直接用两者平均 TTFT 评价延迟优劣。
第二组只比较整个相同任务集合的端到端吞吐。两组实验不能混在一起计算一个总加速比。

**这不是“DFlash 已经超过普通 decode”的结论。** 最终 target-only batch=4 的三次
decode 吞吐中位数为 161.85 tokens/s，仍高于此次 DFlash。无缓存、8 个请求连续补位的
单次对照中，target-only 为 245.71，DFlash 为 198.42；batch=8 同样仍以 target-only 更快。

## 功能与状态验证

- CPU/GPU 测试共 **65 项通过**。卷积修复前的独立 fixture 有 14,332 / 32,768 个元素不一致，修复后逐元素一致。
- 同一历史下，eager/CUDA Graph 的 logits、target features、KV、conv、SSM 逐元素一致；不活动 slot 不变，graph 返回值不会被下一次 replay 覆盖。
- GPU 和 CPU prefix bundle 恢复，以及合批 checkpoint/restore，均通过逐元素检查。
- 4 请求短问答、8 请求连续补位、batch=8、eager、adaptive 和 host-cache 的已比较输出全部一致。
- batch=8 确实执行一次 8 请求 prefill；draft batch sizes 为 8/6/4/2，CUDA Graph 捕获大小为 1–8。该次 DFlash 峰值 allocated 为约 **10.44 GiB**。
- GPU cache=512MiB 的重复负载为 4 miss + 4 hit；小 cache=64MiB 的一次验证为 3 hit + 5 miss、3 offload、1 次选择重算。小缓存不保证所有 prefix 都保留，输出仍为 8/8 一致。
- 固定 block=8 的短问答有效 draft 接受率仍为 **119 / 423 = 28.13%**，平均 **2.87 tokens / request-round**。它与 prefix hit rate 无关。

真实 draft 的 ragged batch 对照中，BF16 有 17/18 个候选 token 与串行相同；同权重 FP32
对照的候选序列一致、draft KV 在严格容差内一致。不能宣称 BF16 下改变 GEMM 形状总是保持候选 token 不变。

## 尚未解决的问题

1. **长回答严格 token 一致性尚未通过。** 新增混合解释/代码负载，4 条各上限 256 token。
   修复后 parallel DFlash 与相同提交的 target-only 仅 1/4 完整一致；sequential 也仅 1/4。
   这些运行即使成功输出，也不具备用来宣称加速的条件，原始数据完整保留。
2. 单独让 target 输出层直接产生 FP32 logits 的探针，其自身对照为 2/4 一致，未合入。
   卷积修复消除了一个真实的算子差异，但 GEMM 形状、活动 batch 和 replay 路径的 BF16 数值差异仍需继续定位。
3. CUDA Graph 仍只覆盖 target 单 token decode、相应 replay 和 sequential verify。
   draft、prefill、多 token verify 仍 eager。低接受率、GDN 状态回滚和重新执行仍有明显成本。
4. 连续补位时的 prefill 会暂停 decode；尚无混合 prefill/decode、chunked prefill 或主 HTTP scheduler 集成。
5. prefix cache 仍是本项目的轻量 KV+GDN bundle cache，不是完整 SGLang HiCache；CPU 传输同步执行。
   未新增 27B INT4 支持、DFlash2、SSD 层级缓存或生产级显存硬隔离。

下一阶段应优先定位长回答首次分歧位置，并在相同历史、相同 batch 组成下逐层比较
decode、verify、rollback/replay；在正确性边界清楚之后，再推进多 token graph 和减少 GDN replay。
