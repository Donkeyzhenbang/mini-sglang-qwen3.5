# MiniSGLang Qwen3.5：DFlash 算子清单与 CPU 审查

日期：2026-09-18。云端仓库：`/root/mini-sglang`；分支：`feat/hybrid-memory-runtime`。审查基线：`da4e840`；本轮代码修复：`e18e375`。本次没有 GPU，未安装或重建环境。

## 1. 适配过程中实际新增了哪些算子

下表按 Triton kernel 入口计数，共 **9 个**，分为 draft 运算、hybrid 状态推进/回退辅助、target 数值稳定三类。这里的“新增”指在本项目适配过程中新增代码，不代表发明了新的数学算法；部分能力也服务于 MTP 和普通 prefill。

文件均相对于仓库根目录。

| 算子 / kernel | 文件与首次提交 | 作用及实现要点 |
|---|---|---|
| RMSNorm / `_rms` | `python/minisgl/kernel/triton/draft_ops.py`，`aaf1ce6` | 将归一化与权重乘法放入一个 kernel；FP32 归约，归一化结果按输入 dtype 舍入后再乘权重，保留 BF16 中间舍入语义。 |
| SiLU × Up / `_silu_mul` | 同上 | 从 packed gate/up 输入读两半，计算 SiLU(gate) × up；SiLU 结果先舍入，减少逐元素 kernel launch。 |
| Cached RoPE / `_rope` | 同上 | 位置查表、半维旋转及乘加融合；两个乘积分别舍入再相加，避免融合改写 BF16 运算语义。 |
| Packed causal convolution / `_conv_extend` | `python/minisgl/kernel/triton/conv_extend.py`，`c1bb9e4` | 一次 launch 处理多个请求各自的 token 区间；读取旧卷积历史、推进短序列、写回最终历史。当前针对 kernel width=4/history=3。支持用结束偏移限定 accepted prefix。 |
| Packed GDN extend / `_extend_kernel` | `python/minisgl/kernel/triton/gdn_extend.py`，`aba1c78` | 把多 token recurrence 从 Python 循环移到设备端，同一请求内部仍按时序递推；不同请求/head/tile 并行。保持与单 token decode 相同的运算顺序，支持 accepted prefix 重放。**不是 chunk-parallel WY 实现。** |
| 跨层状态 gather/scatter / `_copy_slots` | `python/minisgl/kernel/triton/state_copy.py`，`40dfce7` | 通过每层状态指针表批量保存/恢复请求状态；conv、SSM 各一次 launch。恢复时写回原 allocation，不更换 graph 已捕获的状态地址。 |
| Confirmed KV scatter / `_store_context` | `python/minisgl/speculative/draft_graph.py`，`aaf1ce6` | 根据 slot、旧长度、新增数量，将已确认 context KV 写入固定容量缓存。DFlash 与 MTP graph 共用。MTP 递归草稿另用 scratch KV，不把未确认草稿写入长期缓存。 |
| Stable linear / `_linear` | `python/minisgl/kernel/triton/invariant.py`，`c5eb97b` | target 侧固定 tile/K 归约方式，使用显式 FP32 `add.rn` 约束部分和累加，降低 decode/verify 行数变化带来的数值差异。 |
| Stable attention / `_attention` | 同上 | target full attention 按每个 query 的因果长度做固定分块归约，避免验收 block 的整体形状改变单个 query 的计算边界。属于数值稳定路径，不是 draft 的 SDPA 替代品。 |

可以把调用关系概括为：

```text
DFlash propose
  → RMSNorm / SiLU×Up / cached RoPE
  → PyTorch SDPA、F.linear
  → confirmed KV scatter

Target verify
  → full attention：stable linear / stable attention（stable 配置）
  → hybrid GDN：packed convolution / packed recurrence

接受部分草稿
  → 状态快照恢复到 verify 前
  → 只重放 accepted prefix 的卷积与 GDN journal
  → 不重跑整个 target 的 attention / MLP
```

### 不应算成我们新写的算子

- `gdn_fused_proj.py`、`gdn_decode.py`、`causal_conv1d.py` 首次加入均来自底座 `f5e606c`（Qwen3.5 support）。其中 projection 算子主要做布局拆分/整理，不能说成我们写了新的 GEMM。
- DFlash FC/QKV、gate/up、跨层 context KV projection 的合并主要是权重打包和 `F.linear` 调用重组；没有因此增加一套自研 CUDA GEMM。
- DFlash attention 使用 PyTorch SDPA，没有自研 DFlash FlashAttention。
- `journal_graph.py`、`verify_graph.py`、draft/MTP graph pool 主要负责 capture/replay、缓冲区和状态生命周期；CUDA Graph 编排本身不是新的数学算子。
- 之前 vocabulary head 的 flatten 优化改变了输入布局，使 `F.linear` 走更合适的 GEMM 路径；也不应算成新增 kernel。

## 2. 本轮修复及证据

### 2.1 非连续 tensor 与算子输入契约

原 SiLU kernel 使用扁平指针算术，但 wrapper 没有打包非连续输入；RMSNorm 没有处理非连续 weight，RoPE 没有处理非连续 cache。这些输入在 shape 正确时仍可能读取错误的物理位置。

修复：统一在 wrapper 校验形状、dtype、device，并对必要输入执行 `contiguous()`；原本连续的输入不会因此产生新数据副本。拒绝奇数 gate/up 宽度、不匹配的位置长度及 RoPE 表宽度，空 batch 直接返回，避免零 grid launch。新增 CPU 打包/拒绝输入测试和 GPU 数值对照测试。

这属于边界正确性缺口，不证明过去的标准模型路径已经触发：标准路径的张量通常本来连续。

### 2.2 RoPE 负位置越界

原 kernel 的 mask 只检查 `position < capacity`，负数仍可能读取表前地址。本次同时检查非负下界。

超出表范围的位置使用 identity rotation，延续原上界 padding 约定；负位置也作为 padding sentinel。**这不是长上下文 RoPE 外推能力**，调用方必须保证真实 token 在合法范围、padding 不参与 attention。新增 GPU graph replay 的 padding 测试，当前因无 GPU 跳过。

### 2.3 状态快照/恢复的非法 slot 和重复写

原状态复制可接受负数、超出容量的 slot 或 snapshot row；重复目标 slot 会造成多个程序并发写同一份状态。

修复：在任何 CUDA 分配/launch 前检查 host 整数、slot 范围、目标 slot 唯一性、snapshot row 范围；允许多个不同 slot 从同一个 snapshot row 恢复。构造 copier 时检查 conv/SSM slot 容量一致并位于同一设备。journal accepted-prefix 提交也拒绝重复目标 slot、非法 slot 和非正整数接受长度。

这些检查不读取 GPU tensor 内容，不引入 `.item()` 或 GPU→CPU 同步。

### 2.4 Graph 缓存只比较 allocation，漏掉 slot/view 身份

原 MTP/DFlash 用底层 storage 指针判断缓存是否已位于 pool。同一 allocation 的不同请求 slot 具有相同 storage 指针，因此换槽或错误视图可能被当成“已经导入”。

修复：检查 shape、dtype、device；共享 allocation 时还必须具有相同 storage offset 和 stride。不同 slot/view 会在 replay 前报错。独立 eager cache 校验通过后可以正常导入；reset 后空缓存合法。

同时改为先验证整个 batch 的所有 K/V，再执行导入，避免后一条请求失败时前一条已写入缓存。本次**没有实现任意 slot 迁移**，尤其不能用逐条原地复制处理 slot 交换，否则会覆盖尚未复制的源。正常的 batch 行重排仍然允许，只要每个请求保留它的物理 slot。

证据：加载原提交 `da4e840` 的两个 `propose` 方法，用 CPU tensor 和占位 replay 运行实际主机侧路径；MTP、DFlash 均让错槽缓存到达 replay。相同用例在修复后于 replay 前抛出 `ValueError`。这是 CPU 主机逻辑的前后对照，不是真实 GPU 推理结果。

### 2.5 Verify graph 新形状准入

draft/MTP graph pool 原本已经限制 32 个形状并检查空闲显存；journal graph 原本也有数量上限。缺口在 verify graph 缓存没有同类准入限制。

本次给 verify graph 新形状增加最多 32 个以及至少 2 GiB 空闲显存门槛；未获准的新形状走现有 eager 路径，已捕获形状继续复用。**这不是全局显存硬预算，也不保证新一次 capture 一定不会 OOM**：单图需求可能超过门槛，多个池还会累积占用。

## 3. 验证结果

| 项目 | 结果 |
|---|---|
| 修改前 CPU/GPU 测试收集 | 66 passed，30 skipped |
| 修改后 CPU/GPU 测试收集 | **101 passed，32 skipped**；新增 35 个 CPU 参数化用例和 2 个 GPU 用例 |
| 原提交错槽缓存复现 | MTP、DFlash 均到达不应进入的 replay；修复后均提前拒绝 |
| Ruff `E9,F,I` | 本次修改文件通过 |
| Python compileall、git diff --check | 通过 |
| 本次 GPU kernel / graph 执行 | **未执行，无 GPU** |
| 本次完整模型 token 精度、吞吐和显存峰值 | **未测量**，不能沿用旧数据作为新提交验收结论 |

云端原始记录：`/root/autodl-tmp/runtime-results/cpu-review-20260918/`，包括 `before-tests.log`、`after-tests.log`、`baseline-cache-repro.log`、`lint.log` 和修改前源码归档。使用 `-o addopts=''` 是因为当前环境没有 pytest-cov，不为此次审查重装依赖。

## 4. 仍需完成的事项

| 优先级 | 未完成项 | 完成标准 |
|---|---|---|
| P0 | 新提交 GPU 回归 | 32 个 GPU 用例实际通过；非连续输入、padding、状态快照恢复与 graph 动态重放没有 illegal access，必要时配合 Compute Sanitizer。 |
| P0 | 完整模型精度/性能复测 | 同权重、同 prompt、同 greedy/stable 数值配置，batch=4 比较 target-only/MTP3/DFlash8；逐 token 对齐；256/512 输出、不同接受长度、连续补槽、reset/reorder 均通过。计时同时保留 graph 开关和预热口径。 |
| P1 | 全局显存预算 | 联合统计 target KV、draft KV、GDN 状态、journal、graph private pools 和 scratch；目前 graph 数量阈值与预留显存检查不能替代按字节准入及安全回收。 |
| P1 | 更高并发与长上下文 | 真实测试 batch 8/16、长短请求混合和较长上下文；当前 pool 仍按 max_batch × max_context 预分配，没有证明高并发显存/性能可扩展性。 |
| P1 | 采样与跨后端等价性 | greedy/stable 的逐 token 相同不能推出所有采样策略分布无损，也不能推出跨 SGLang、不同 GPU/编译器严格相同；需要单独验收。 |
| P2 | 进一步性能实验 | 当前 packed GDN 仍为时序 recurrence，stable target kernel 以吞吐换数值约束；是否引入 WY、替换归约路径或调整 block，需要 GPU profiler 与精度门禁。此次未改变这些计算公式。 |

本轮没有给所有低层算子增加 GPU 数据内容的同步校验；内部 GPU 元数据仍依赖合法调用链。恢复 GPU 后需要覆盖边界并使用设备端工具排查，不能把这些 CPU 检查称为完整内存安全证明。

## 5. GPU 恢复后的执行入口

```bash
cd /root/mini-sglang
export PYTHONPATH="$PWD/python"
export OMP_NUM_THREADS=4
python -m pytest -o addopts='' tests/cpu tests/gpu -q

# 此脚本拒绝已存在的输出目录，请每次使用新名字。
bash benchmark/runtime/run_graph_opt_validation.sh python \
  /root/autodl-tmp/runtime-results/post-cpu-review-gpu-$(date +%Y%m%d-%H%M%S)
```

脚本默认读取 `/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B` 和对应 `Qwen3.5-4B-DFlash`；可通过 `MODEL`、`DRAFT` 环境变量覆盖。它包含 batch=4、256/512 输出、DFlash block 4/8/16、ragged continuous batching 和独立 prompt 精度比较。GPU 测试通过之前，不应把本轮提交标记为已完成 GPU 精度/性能验收。
