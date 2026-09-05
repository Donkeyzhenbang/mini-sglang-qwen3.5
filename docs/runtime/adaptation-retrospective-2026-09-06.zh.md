# Qwen3.5 Hybrid Runtime：MTP / DFlash 适配与优化复盘

版本：2026-09-06。项目仓库：`Donkeyzhenbang/mini-sglang-qwen3.5`，分支 `feat/hybrid-memory-runtime`。本文整理 2026-08-30 至 2026-09-06 的代码审查、CPU/GPU 实验与提交记录；阶段性失败保留，不用最终结果覆盖历史问题。

项目目前已实现 MiniSGLang 原生 Qwen3.5-4B BF16 的 MTP-1/MTP-3、DFlash v1、真实 batch 执行、混合状态缓存和多条 CUDA Graph 路径。在已测单卡 4090、greedy、离线 batch=4 工作负载中，MTP-3 与 DFlash block=8 均已超过相同数值策略的 target-only，输出 token 完全一致。最新 SGLang 横向数据和本次重跑结果见同目录《SGLang横向对比-20260906.md》及机器可读证据。

这是一项推理系统工程与实验研究项目。当前证据不支持“完整移植 HiCache”“27B Int4 已部署”“DFlash2 已支持”“任意输入保证加速”或“生产服务零 bug”。

## 1. 项目起点与实际范围

起点是 MiniSGLang Qwen3.5 PR #133 对应代码，最初审查提交 `f5e606c`。已有模型结构、权重加载、GDN、普通生成和请求槽管理，因此工作不是从零实现 Qwen3.5；新增工作主要集中在混合状态正确性、投机执行闭环和性能优化。

| 组成 | 实际完成的工作 | 尚未覆盖 |
|---|---|---|
| Qwen3.5 target | 修复共享 GDN 状态池、前缀长度/快照生命周期、稀疏 KV 映射、显存预留；统一 decode/verify 数值路径 | 多模态、多卡与所有 attention 后端的完整验收 |
| 混合缓存 | KV + conv + SSM + target features 等组成一致的前缀包；GPU/CPU 预算、同步 offload、恢复、LRU/cost 策略 | 完整 SGLang HiCache、异步预取、SSD、活跃序列分页卸载 |
| 原生 DFlash | checkpoint 加载、八层 target taps、六层 draft、greedy verify、拒绝回滚、批量 draft、固定/自适应 block | DFlash2、随机采样分布验证 |
| 原生 MTP | 加载 checkpoint 内训练好的 MTP 层；独立 KV、右移 token/hidden 对齐、递归三步 proposal | MTP draft 自身的完整 CUDA Graph |
| 执行优化 | ragged prefill、批量前向、连续补槽实验入口；decode/verify/DFlash draft/GDN replay 图；Triton 融合 | 主 HTTP/overlap scheduler 的投机集成、线上 TPOT p99/SLO |
| 评测 | token 对照、首次分歧、真实 wave wall time、分阶段成本、接受率、graph 计数、显存和版本证据 | 大规模质量榜单、统计置信区间、跨架构逐位保证 |

技术栈以 Python、PyTorch、Triton、CUDA Graph、FlashInfer 为主。当前完成的是显式执行图和算子优化，不是开发了通用 FX/Inductor 编译 pass manager；简历不能把这些内容写成已有 C++/CUDA 编译器成果。

## 2. 为什么接一个 draft model 并不简单

DFlash checkpoint 是训练好的 draft 模型，无需为本次适配重新训练。但推理框架需要实现完整闭环：

```text
target 已确认历史与 hidden features
        ↓
MTP / DFlash 生成候选
        ↓
target 同时验证多个位置
        ↓
接受连续匹配前缀，产生纠正或 bonus token
        ↓
提交 KV / conv / SSM / draft KV / features 的同一逻辑边界
        ↓
下一轮
```

Qwen3.5-4B 有 32 层，其中 24 层 GDN、8 层 Full Attention。Full Attention 的拒绝后回退可以主要依靠有效 KV 长度管理；GDN 的 recurrent state 已被候选 token 原地更新，不能只缩短一个长度字段。实现难点是让不同状态对应同一历史，并使维护这些状态的成本低于省下的 target decode。

MTP 使用 target checkpoint 内的一层训练权重，约 230 MiB，复用 target embedding/LM head；三步 draft 是递归调用该层，不是三层独立权重。DFlash 使用六层独立 draft，结合八个 target hidden taps，前五层 sliding attention，最后一层 full attention；mask、RoPE、特征层编号必须按配置和参考实现处理，不能直接当成普通小号自回归 Qwen。

## 3. 第一类问题：混合状态的逻辑长度与生命周期

### 3.1 页对齐 KV 与非对齐 GDN 快照错位

最初 CPU 复现发现：300-token prompt、page size=256 时，Radix KV handle 向下对齐到 256，而保存的 GDN 状态已处理 300 token。恢复后再计算尾部，相当于重复处理部分历史。哨兵状态实测为 `cached_len=256, restored_state_marker=300`，页对齐控制用例为 `256/256`。

修复不只是补一个长度字段。快照记录并校验真实 `state_len`，只发布 KV 与 GDN 完全同边界的状态；没有对应页边界快照时安全 miss。不能把 300-token 状态重新标成 256-token 状态。实验缓存进一步采用完整前缀包，将 KV、conv、SSM、features、末位置 logits 绑定同一逻辑长度。

### 3.2 混合 attention 后端创建了两套 recurrent state

`fa,fi` 分别构造 prefill/decode 后端时，原实现产生两个独立 GDN runtime。CPU 工厂测试给 prefill 状态写入哨兵值，decode 一侧仍为空。修复为组合 Full Attention 后端外包一层共享 Hybrid/GDN 状态管理，保证 prefill 和 decode 看到同一份 recurrent state。

这条路径与 4090 默认 `fi` 不同，不能用该问题解释所有旧 GPU 结果。发现手段是对象所有权和状态传递检查，而不是仅看模型能否输出自然语言。

### 3.3 快照必须绑定计算边界，不能随取随用

overlap 场景下，调度线程处理上一轮结果时，下一轮 GPU forward 可能已经修改 live state。等待某次输出拷贝结束不等于“拿到上一轮逻辑长度的 state”。修复在 engine stream 的确定边界生成独立快照，再发布给调度器；请求槽清零、恢复也遵循计算流所有权。

最初这一项是静态竞态分析，不是已捕获所有 overlap 竞态。最终 native spec 验收仍为离线 runner；不能把基础快照修复扩大成完整线上 overlap 验收。

### 3.4 输出长度与已进入模型的长度不同

每轮最后输出的纠正/bonus token 通常尚未送入 target，下轮才作为 anchor。因此状态长度对应“prompt + 输出中除最后一个之外的 token”。拒绝时应恢复块前状态，提交 anchor + 接受前缀，丢弃 rejected suffix；EOS 和输出上限截断同样遵循这一约定。

测试覆盖全接受、零候选接受、部分接受、连续拒绝、EOS、请求提前结束和槽复用。只比较最终文本会漏掉下一轮才暴露的 off-by-one，因此还检查 KV、conv、SSM、features 和未参与请求是否被修改。

## 4. 第二类问题：不是每一次 token 分歧都是 rollback bug

### 4.1 排查方法：先固定历史，再找最早偏离的算子

长回答首次出现差异后，继续比较两条各自生成的序列已经不能定位原因：历史不同，后面所有状态自然都会不同。实际排查采用以下顺序：

1. 找到第一个不同 token 的位置，记录两边候选 logits 及 top-1/top-2 间隔。
2. 强制两条路径使用相同 token 历史、初始 KV/GDN state 和请求组成。
3. 单独比较 decode、parallel verify、sequential verify、rollback/replay。
4. 逐层检查投影、卷积、GDN recurrence、Full Attention、hidden taps、LM head。
5. 将首个差异缩成独立小张量 fixture；使用 PyTorch/FP64 oracle，避免两个错误实现互相验证。
6. 算子修复通过后，回到真实权重长生成、不同 batch 排列和缓存恢复做回归。

本地历史探针保留在 `work/stable-target-results/raw/` 与 `work/full-batch-results/`；原始同历史追踪和失败生成结果也保留在历史归档中。

### 4.2 真实算子差异：BF16 舍入发生在不同位置

GDN decode 卷积与 BF16 prefill/verify 的乘加、cast、SiLU 顺序不同。独立 fixture 修复前有 14,332 / 32,768 个元素不一致。统一为 FP32 乘加、按输入 dtype 舍入、再执行激活后，fixture 逐元素一致。CPU fallback 也跟随相同语义，避免 CPU 测试掩盖 GPU 差异。

关键教训：数学公式相同不等于有限精度实现相同。融合 kernel 如果删除一次原来存在的 BF16 cast，可能改变后续状态，即使理想实数运算等价。

### 4.3 GEMM 形状改变了 target 的算术路径

修复卷积后，四个 decode token 与四个八-token verify 块对比，第一层 GDN 的 `in_proj_ba` 已有 18/64 个值不同。回滚时活动请求减少，又会改变 `in_proj_qkvz` 的结果。这表明不同 row count 触发的 BF16 GEMM 算术选择本身就会带来差异。

曾经只把 LM head 改成 FP32，长回答从 1/4 完全一致变成 2/4，仍失败；只改 sequential verify 也不能保证成功，因为回滚后的 batch 组成仍可能变化。这些尝试没有被包装成已解决。

最终采用 target 局部的固定 Triton linear tile；K 分块产生 partial dot，再显式 FP32 累加，规约过程不依赖 query 行数和位置。独立 K=9216、FP64 oracle 对照中，最大误差从 0.006866 降到 0.000183，没有放宽测试容差。

### 4.4 Full Attention 与 LM head 仍需单独对齐

线性层固定后，第一个 Full Attention 层依然存在 decode/packed-verify 差异。仅关闭 FlashInfer split-KV 在当时安装版本中不够。实现按每个 query 的真实因果终点、相同 key tile 和规约顺序计算 attention，排除 speculative future KV；多个 query/head 仍并行执行，并非逐请求串行。

target LM head 将 FP32 logits 保留到 argmax，避免输出层再转 BF16 将邻近 logits 压成相同值。早期探针见过两个候选从 `26.0/26.0` 变成 `26.125/26.0`，足以改变 greedy 选择。

### 4.5 stable 的准确含义

`stable` 定义新的、统一的 target 数值执行基线：target-only 与 speculative 在同一策略下比较。它不保证等于历史 fast BF16、Hugging Face、SGLang 默认路径，也不保证跨 GPU/编译器逐位相同。

因此这里的精度结论是“已测 greedy 输出与同策略 target 完全一致”，不是“语言任务正确率 100%”。SGLang 同 shape 重复、跨 batch、target/MTP、跨框架比较应分别报告；轻微算术变化可能改变完整续写，但不能仅凭 token 不同认定模型语义质量下降。

历史 stable 验收 `c5eb97b` 覆盖 15 组生成运行、80 条请求，含 batch 1/4/8、256/512 输出上限、固定/自适应、并行/逐 token、eager/graph、连续补槽和 CPU 缓存。其吞吐仍低于 target，说明正确性完成与性能完成是两个里程碑。

## 5. 第三类问题：接受率高，为何仍然慢

是否加速取决于一轮实际推进量与全部成本：

```text
spec 吞吐 ≈ 本轮已提交 token 数 /
            (draft + verify + snapshot/restore/commit + 调度同步开销)

只有每个提交 token 的总成本低于 target decode，投机才有收益。
```

MTP-3 有三个候选 token，DFlash block=8 通常包含 anchor + 七个候选。接受率分母不同，不能直接比较；更不能把 prefix cache hit rate 当成候选接受率。第一次 miss、第二次 hit 描述历史前缀复用，关闭全部 prefix cache 后仍能测投机接受率与加速。

最初 benchmark 将四条请求依次完整生成，虽然外部看见四句输入，但 GPU 并未每步批量计算。之后实现 ragged prefill、batched draft/verify 和连续补槽，记录真实 batch size。四个请求应共享一次 wave 的墙钟时间，不能把重叠请求耗时累加再计算吞吐。

短输出容易被 prefill、建图、draft context 初始化淹没；长输出更容易摊薄这些开销，但低接受率、过重 draft 或 eager verify 仍可能更慢。“开启投机一定更快”不是有效验收假设。

## 6. 性能关键一：用 GDN journal 替代整模型重放

旧实现拒绝后恢复 recurrent state，再将接受前缀经过整个 target，重复 attention、MLP、投影、LM head 和 features 提取。其逻辑容易验证，但重复计算会吞掉投机收益。

优化在 verify 时保留每层 GDN 所需的 projected mixed-QKV、a、b。拒绝后从块前 checkpoint 恢复，仅对接受输入重跑卷积与 recurrent 更新；已全接受请求直接保留验证后状态。它利用“同一历史下投影输入已计算完成”的事实，不需要再次执行 Transformer。

进一步给 Triton conv/GDN extend 加 start/end，让 replay 直接读取 journal 原始区间；避免每层拼接接受片段、创建索引和 CPU→GPU 元数据。所有层共享动态区间，未参与回滚的 slot 不变。

为什么不保留块中每个 token 的完整 SSM？4B 每个请求的全部 GDN SSM 边界约 48 MiB，16 个边界约 768 MiB/请求；batch=4 仅这些边界就约 3 GiB，尚不含 conv、KV、draft 和 graph。journal 选择保存紧凑投影输入并局部重算，以少量递推换取状态内存。

历史同 workload：block=8、256-token DFlash 从 162.22 到 195.02 decode tok/s，状态 restore/commit 从 2278ms 降到约 746ms，但仍慢于 target。这是瓶颈迁移的证据，不是优化失败：后续成本已转向 eager draft/verify。

## 7. 性能关键二：图覆盖与地址生命周期

### 7.1 只打开 decode graph 不等于投机主要路径已入图

早期日志显示 CUDA Graph 已开启，但只覆盖 target 单 token decode。DFlash draft、多个 token 的 target verify、拒绝后的 state commit 仍是 eager，包含大量 Python 调度与细碎 kernel launch。

后续分别实现 uniform block=2/4/8/16 的 stable verify graph、DFlash draft graph、GDN journal replay graph。真实 ragged 尾部和不支持形状安全回退 eager，不能为 graph 随意改接受长度或输出内容。

### 7.2 图捕获本身会执行计算，必须恢复状态

warmup/capture 会推进 live recurrent state。如果捕获后直接进入正式推理，相当于额外消费了一段 token。verify 与 journal 图在捕获前保存初始状态，完成后恢复；测试不仅检查 replay 输出，还检查“捕获没有污染状态”。

位置、slot、接受区间作为稳定 GPU buffer 的动态内容更新；journal 和输出的生命周期由图持有。输出若直接返回 graph 内部复用 buffer，下次 replay 会改写调用者仍持有的结果，因此必须在需要跨 replay 保存的位置建立独立所有权。

### 7.3 跨层状态拷贝不能移动已有 live state

将 24 层 conv/SSM 分别 index_select、stack、index_copy，CPU 与 launch 成本较高。实现 GPU 指针表，分别用 conv 和 SSM 两次 kernel gather/scatter。保留原有 live 分配地址，避免已经捕获的 decode graph 继续访问旧地址。

测试覆盖多个同时存在的快照、乱序子集恢复、非连续 slot 和不活动请求不变。单一快照、固定 slot 的 happy path 不足以验证这些生命周期问题。

### 7.4 DFlash 持久 KV 只允许写已确认历史

为六层 draft 建立固定地址 confirmed-context KV 池；speculative noise block 不写入持久历史。长 prefill/积累 features 超过图支持范围时 eager 补齐，并支持 graph→eager→graph 转换、旧缓存导入、slot reset 后复用。

这部分难点是容量、有效长度和所有权同时正确。测试覆盖乱序 slot、不同接受长度、边界 padding、上下文补齐，以及 replay 是否读取新数据。共享 RoPE 表对越界 padding 有显式保护。

## 8. 性能关键三：融合需要保留有限精度语义

| 优化 | 节省的工作 | 验证重点 |
|---|---|---|
| DFlash QKV、gate/up 打包 | 每层多个小 GEMM 合并，不保留重复原权重 | 官方 checkpoint key 转换、shape 和真实权重输出 |
| 六层 context-KV 一次投影 | 用较大 GEMM 代替六次 context 投影 | 约 60 MiB 只读融合权重计入预算，逐层 slice 正确 |
| target tap clone 消除 | `x + residual` 已产生新 tensor，移除多余 clone | 不误删跨 replay 必需的所有权拷贝 |
| RMSNorm、SiLU×Mul、RoPE Triton kernel | 减少细碎 launch 和中间张量 | 保留原 BF16 中间舍入；RoPE 的 BF16 products/cos 表及 padding 边界 |
| MTP 批量 argmax 回传 | 从逐请求 `.item()` 改为每步一次 batch 回传 | 接受 token 不变，避免隐藏设备同步 |

未实现的 fused LM-head+argmax、完整 MTP draft graph、通用 FX pass manager 属于后续候选，不能写成已落地。也没有以 Nsight 图作为本次全部结论的证据；现有归因主要基于分阶段计时、图计数、受控开关和独立 kernel/状态探针。

一次 MTP batched recursive-KV packing 尝试虽然通过 5 个 CPU 测试，GPU 吞吐却降到约 255–270 tok/s，而当时 fast 基线约 313 tok/s，最终撤回。减少 Python 对象或看似更整齐的张量布局不必然降低真实设备成本，应保留负结果。

## 9. Adaptive 为什么会被建图成本误导

策略按 batch/context bucket 观察真实接受进度和 draft+verify+state 成本，候选 block 为 1/2/4/8/16。早期模型把首次建图、draft 历史补齐当作 steady-state 成本，导致某候选刚尝试一次便被错误排除。

修复将 startup 回合显式标记，仍完整计入用户墙钟时间，但不更新稳态成本预测；继续完成候选的稳态校准。同时统一 adaptive context padding 宽度为 16，减少切换 block 导致的冗余图，图缓存上限 32，创建前保留至少 2 GiB 显存余量。

同一 256-token 五轮阶段实验由 267.25 提升到 356.50 decode tok/s，超过约 330.23 target；但固定 DFlash block=8 仍为 370.53，更快。这里证明了成本观测和图形状管理的重要性，没有证明 adaptive 优于最佳固定策略，也没有证明全局最优调度。

## 10. 性能演进与不能混算的结果

| 阶段 | 当时的实测结论 | 解读 |
|---|---|---|
| 初始单请求 eager pilot | target 47.50，DFlash8 110.88 tok/s | 小规模受限 smoke；不是相对成熟框架的 2.33× |
| 长生成 stable 正确性完成 | target 328.98，DFlash8 162.22 | 统一数值路径后，状态重放/eager 成本暴露 |
| journal 提交 | DFlash8 195.02 | 状态成本显著降低，仍未加速 |
| MTP 原生融合，9月3日 | stable MTP3 约 300，target 332.85 | 接受率接近 59%，主要仍卡执行成本 |
| verify / draft / replay 图，9月5日 | MTP3 427.35，DFlash8 370.53，target 330.23 | 相同 stable、batch4、256token、五轮，token 一致 |
| 长输出，9月5日 | MTP3 437.55，DFlash8 412.41，target 327.86 | 512token、三轮，分别约 1.335× / 1.258× |
| 干净提交独立复跑，9月6日 | MTP3 427.14，DFlash8 368.26，target 328.06 | 一键验收再次通过，不依赖未提交 runtime 修改 |

阶段间 workload、数值模式、repeat 或 baseline 有变化，不能串乘为一个总加速比，不能当成严格单变量消融。resume 应引用最终同轮 target 对照，不引用早期弱基线的 2.33×。

9月5日主验收：MTP3 和 DFlash8 各覆盖 44 请求、11,924 输出 token 完全一致，包含 256/512 输出及 1/17/73/129 混合长度；这些是四条基础 prompt 的重复与重排，不是 44 个独立问题。最终 CPU/GPU 回归 94 项通过。MTP 峰值 allocated/reserved 约 9.18/9.30 GiB，DFlash8 约 10.65/12.01 GiB；张量 allocated、allocator reserved 和整卡占用是不同指标。

## 11. 显存与 Hybrid Cache：已做什么，尚未证明什么

原模型按所有 32 层分配 KV，而只有 8 层 Full Attention 使用 KV。通过 layer-id 映射只为这些层分配，消除该维度 75% 的冗余 KV 层分配；不能表述为总显存下降 75%。GDN 活跃状态在初始化预算中预留，按 max-running slots 计算，不能用 KV token 数代替槽数。

4B 每请求槽约 48 MiB SSM + 1.125 MiB conv，默认 257 槽约 12.33 GiB；实际 batch=4 不会自动让默认槽池只分配四份。这是显存预算需要同时考虑模型架构和并发上限的例子。

GPU/CPU cache 按完整前缀包迁移与驱逐，恢复成本大于重算时可选 recompute；已经验证真实卸载和缩块链路。软件预算注入测试曾卸载约 54.6 MiB，将可选 block 限制为 `[1]`，预算恢复后再开放候选。这不是物理 24GB 耗尽实验。

早期共享前缀实验中，CPU offload 未体现 TTFT 收益，且当时还有分段数值差异；后续 stable 修复使状态恢复正确，但没有因此证明 cost policy 优于 LRU。当前同步传输、整包策略需要更多公共前缀/访问频率/PCIe 成本消融。27B Int4、硬显存隔离和任意 context 容量保证均未完成。

## 12. 版本、复现与后续验收

| 提交 | 可核对的工作 |
|---|---|
| `4058b07` | 初始混合状态、快照、KV 映射、显存预算修复 |
| `07115d7` / `aba1c78` | Hybrid Cache 与 adaptive 原型；原生 DFlash 闭环 |
| `c1bb9e4` / `9f872a9` | 真批处理、连续补槽、卷积舍入修复 |
| `c5eb97b` | stable target 数值路径与长输出对齐 |
| `98b4801` / `8502077` | GDN journal 与 DFlash 投影融合 |
| `7585738` / `142c605` | 原生 MTP-3 与真实 GPU 修复/优化 |
| `aab05f4` / `40dfce7` | 接受区间直接 replay，跨层状态拷贝 |
| `c8ac8e6` / `aaf1ce6` | verify、DFlash draft、journal CUDA Graph |
| `67f0afb` | adaptive startup 计费与图形状上限 |
| `20b06d1` / `47f5147` | 验收工具、报告与干净提交独立复跑 |

云端基本复现：

```bash
cd /root/mini-sglang
bash benchmark/runtime/run_native_spec_acceptance.sh \
  /root/miniconda3/bin/python \
  /root/autodl-tmp/runtime-results/repro-$(date +%Y%m%d-%H%M%S)

PYTHONPATH=$PWD/python OMP_NUM_THREADS=4 \
  python -m pytest -o addopts='' -q tests/cpu tests/gpu
```

脚本在独立进程依次测 target、MTP3、DFlash8，每个模式内部是真 batch=4；缓存关闭。`SHOW_TEXT=1` 可在日志显示 prompt/回答，JSON 始终保存 token 和文本。比较器检查输入、配置、模式、版本、wave 计数与逐 token 输出，不仅检查进程返回码。

后续优先级：扩大独立 prompt 与上下文长度、明确 EOS/随机采样质量边界；补充 profiler timeline 和稳定单变量消融；在隔离兼容环境测 SGLang DFlash；再推进 MTP draft graph、LM-head/argmax 融合与 HTTP 调度集成。只有完成相应实验，才能扩展当前离线 greedy 性能结论。

历史来源包括本地 `mini-sglang-qwen35-dflash-review.md`、`gpu-experiments-2026-08-31.md`、`full-batch-optimization-2026-08-31.md`、`stable_target_numerics.md`、`stable_target_results_2026-08-31.md`、`mtp-and-state-journal-results-2026-09-01.md`、`qwen35-speculative-pass-optimization-report-2026-09-03.md` 和 `native-spec-20260906-artifacts.tar.gz`。早期报告中的“未完成”是当时状态；本报告据后续验收更新，不改写原始记录。
