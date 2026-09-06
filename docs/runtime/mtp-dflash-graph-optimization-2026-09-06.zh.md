# MTP draft CUDA Graph 与 DFlash 256 性能优化

本轮在云端 `/root/mini-sglang` 完成，复用已有 Torch/FlashInfer 环境，没有重新安装推理环境。代码提交：`5b9ced6`（MTP graph）、`76e054e`（DFlash 词表投影），分支 `feat/hybrid-memory-runtime`。本报告覆盖前述收尾之后的追加工作，替代旧报告中“MTP draft graph 尚未实现”的状态描述。

## 1. 最终验收结果

Qwen3.5-4B BF16，RTX 4090，stable target 数值策略，greedy；四条原始 prompt，不套聊天模板，batch=4 同时执行。每种配置预热两轮、测量五轮，关闭 prefix cache。吞吐为输出 token 总数 / wave 总墙钟，包含 prefill，排除加载和预热。最终矩阵来自干净提交 `76e054e`，不混用历史最好成绩。

| 输出长度 | Target-only | MTP-3 + draft graph | 相对 target | DFlash-8 + graph + 投影优化 | 相对 target |
|---|---:|---:|---:|---:|---:|
| 256 | 328.75 tok/s | 474.89 tok/s | **+44.5%** | 408.43 tok/s | **+24.2%** |
| 512 | 326.07 tok/s | 487.61 tok/s | **+49.5%** | 461.35 tok/s | **+41.5%** |

- 上述每项 20 次请求；MTP-3、DFlash-8 在两种长度下均与对应 target 输出 token 完全一致。
- 另测 12 次请求、输出上限 1/17/73/129 的 batch=4 连续补槽；MTP-1、MTP-3、DFlash-8 均与 target 对齐，共 660 token/模式。该项用于状态与调度正确性，不以单轮混合长度吞吐替代稳态性能。
- 再增加 8 条独立中英文 prompt，覆盖代码、算术、翻译、系统概念和长资料总结；输入长度 16～897 token，每条输出 128 token。MTP-3、DFlash-8 均精确对齐。
- 因而本轮 MTP-3、DFlash-8 各有 **60 次请求、17,044 个输出 token** 的精确一致性证据。基础矩阵和补槽来自四条 prompt 的重复/长度组合，另有八条独立 prompt；不是 60 道独立质量题，也不是大规模质量榜单。
- 完整 `tests/cpu`、`tests/gpu`：**96 passed**。
- 主矩阵峰值 allocated 显存约：target 8.69 GiB，MTP 9.24 GiB，DFlash 10.67 GiB；不是显卡全部占用，也不是硬显存上限。

## 2. MTP 为什么原来没有 draft graph 也能加速

此前已经有 target decode、parallel verify 和 GDN journal replay graph。MTP 又使用 checkpoint 内嵌的单层权重递归生成，而不是执行完整 target。在这组输入上，MTP-3 draft token 接受率为 58.98%/61.52%，已经足以覆盖部分 proposal 成本。

但旧 proposal 每一步都需要：整理变长 KV、构造输入、执行 eager 前向、argmax、将 token 返回 CPU，再启动下一步。三步递归意味着重复的 Python 调度和设备同步。

新增 `MTPGraphPool` 把下面的链条放进同一张图：

```text
已确认 target hidden + 右移 token
         │
         ├─ MTP 第一步 ── 写 confirmed KV 池
         │       ↓ prediction / hidden
         ├─ MTP 第二步 ── 仅写图内临时 KV
         │       ↓ prediction / hidden
         └─ MTP 第三步 ── 仅写图内临时 KV
                         ↓
                  一次取回整条 proposal
```

工程关键点：

1. **持久状态与临时候选分离。** 第一步只消费已经确认的 target 历史；后续递归的 KV 只存在图内 scratch。拒绝后不需要清理错误候选写入的持久缓存。
2. **动态内容、固定地址。** slot、previous length、confirmed count、target length 和 token IDs 更新到固定 GPU buffer；batch/block/context bucket 选择对应图。
3. **捕获不能推进逻辑状态。** warmup/capture 只重复写相同 confirmed KV，读取由 previous-length mask 限定；递归不写持久池。
4. **回退与恢复可互操作。** 初始长 prefill、非统一尾部 block 或图池资源不足时回退 eager；下一次 graph 从 eager 的 confirmed KV 导入。请求 reset 不会把旧 slot 尾部视为有效历史。
5. **资源有界。** 图数量上限及空闲显存保护避免无限捕获；confirmed KV 池计入 admission 预估。它仍不是 allocator 的硬上限。

初次控制实验的 256 结果：eager draft 422.72 → graph draft 481.79 tok/s，增加约 14.0%；五轮 draft 墙钟 4505 → 3115 ms。接受率保持 58.98%。这是“完整 graph proposal 路径”的收益，包含 KV 整理与 CPU 往返减少，不能全部归因于单独减少 kernel launch。

随后在同一干净提交上只切换 `--no-draft-cuda-graph`，保持 target/verify/state graph 开启，获得以下独立消融；所有输出仍与 target 一致：

| 输出长度 | MTP-3 eager draft | MTP-3 graph draft | graph 路径增加吞吐 |
|---|---:|---:|---:|
| 256 | 423.88 tok/s | 474.89 tok/s | +12.0% |
| 512 | 428.08 tok/s | 487.61 tok/s | +13.9% |

这两轮不是同进程同时测试；均独立进程、相同参数和五轮聚合，保留原始记录，不将初次 481.79 与不同轮次的 target 拼接成最终表格。

测试覆盖 MTP-1/MTP-3、不同确认长度、请求重排、slot reset、graph→eager→graph、长 prefill 导入和上下文边界。边界测试还发现旧 eager 路径的无效 ragged padding 可能越过 RoPE 表；现在仅将无效位置置零，真实位置不变。

最终计数中，256 的 MTP draft 有 505 次 graph replay、15 次 fallback；512 有 1010 次 replay、15 次 fallback，分别捕获 8/15 种形状。覆盖率不是 100%，回退仍是正确性与资源边界的一部分。

## 3. DFlash 256 的具体瓶颈

### 3.1 先重新建立稳定基线

修改 DFlash 前，本轮五次测试为约 363/360/363/366/367 tok/s，聚合 **363.87 tok/s**；对应 target 为 328.68 tok/s。此前横向测试的 336.35 tok/s 未复现。

因此不能把“256 只有 +5.9%”解释成固定性能上限，也不能用模型加载或未预热解释。此前那次逐轮降速的具体系统原因仍未确认；本轮保留 GPU 温度、时钟、功耗和利用率日志，不倒推不存在的历史遥测。

### 3.2 用 profile 锁定词表投影

原代码：

```python
logits = F.linear(hidden[:, 1:], head)
```

`hidden` 为 `[batch, block, hidden_size]`。去掉每条请求的 anchor 后，`hidden[:, 1:]` 是非连续三维视图。该路径选择了 strided batched GEMM，对每条请求重复读取共享的大词表权重。

在该 checkpoint 中，词表投影权重为 `248320 × 2560`、BF16，约 1.18 GiB。一个 batch 并行并不自动保证这些权重由一次有效 GEMM 共享读取。

一轮 profile 中，慢 GEMM 共 83 次、累计约 443 ms；次数也与实际活跃 batch>1 的 draft 调用对应。`.tolist()` 的 CPU 时间包含等待 GPU 完成，不能把它的累计时间全当成 Python 转换开销。

### 3.3 修复布局，保留语义

```python
selected = hidden[:, 1:].reshape(-1, hidden.shape[-1])
logits = F.linear(selected, head).view(batch, block - 1, -1)
```

先合并 batch/token 维度，再进行一次共享矩阵乘法。修改同时覆盖 eager batched draft 和 CUDA Graph draft；target 数值、RoPE、attention mask、验证和接受规则均未改动。

已有 graph 只会重复捕获时选择的运算，**不会自动修复非连续布局导致的低效 GEMM**。

### 3.4 三层证据

**算子探针：** checkpoint 同尺寸的合成权重/输入，CUDA Graph 开启，每个形状测量 20 次：

| Batch | 原投影中位数 | 合并后中位数 |
|---|---:|---:|
| 1 | 1.480 ms | 1.367 ms |
| 2 | 2.862 ms | 1.386 ms |
| 3 | 4.295 ms | 1.440 ms |
| 4 | 5.696 ms | 1.447 ms |

本探针 logits 完全一致。batch=1 原本就没有跨请求重复，较小差异不作为结构性收益；主要证据是 batch=2/3/4 随请求数增长的重复成本被消除。合成算子结果不等于模型质量结果。

**完整 profile：** slow GEMM 消失，单轮累计 kernel 时间由约 2488 ms 降至 2165 ms；target `_linear` 总时间为 1344.16 → 1344.38 ms，基本不变。Profiler 会扰动运行，profile 墙钟不用于正式吞吐表。

**实际生成：** 首次前后对照 363.87 → 412.76 tok/s，增加 13.4%；五轮 draft 时间 4726 → 3074 ms。接受率仍为 27.30%，20 次请求输出与 target 完全相同。最终干净提交再次得到 408.43 tok/s，见开头统一矩阵。

## 4. Block size 并非越大越好

优化后在同一组 256 输出、batch=4、五轮测试中：

| DFlash block | 接受率 | E2E 吞吐 | 相对 target |
|---|---:|---:|---:|
| 4 | 47.49% | 400.62 tok/s | +21.9% |
| 8 | 27.30% | **408.43 tok/s** | **+24.2%** |
| 16 | 13.51% | 387.02 tok/s | +17.7% |

三种 block 均精确对齐。block=4 接受率最高但每轮推进有限；block=16 验证更多候选，实际接受增长不足以抵消成本。这组输入仍优先选 block=8，不把此结果推广为所有 prompt/context/batch 的最优配置。

## 5. 复现

在云端仓库执行，输出目录必须不存在：

```bash
cd /root/mini-sglang
git log -3 --oneline
bash benchmark/runtime/run_graph_opt_validation.sh \
  /root/miniconda3/bin/python \
  /root/autodl-tmp/runtime-results/graph-repro-$(date +%Y%m%d-%H%M%S)
```

脚本依次测试 256/512 的 target、MTP-3、DFlash-8；256 的 DFlash-4/16；混合长度连续补槽的 target、MTP-1/3、DFlash-8；最后运行八条独立 prompt 的正确性回归。保存所有输入、输出 token、接受率、graph 计数、显存和比较摘要。各 GPU 测试串行启动，避免相互竞争。独立 prompt 回归只检查精度，不要求含新增图捕获的单轮结果满足稳态加速门槛。

`--cuda-graph` 现在同时启用支持形状的 MTP draft graph；在同一 benchmark 命令上增加 `--no-draft-cuda-graph`，可仅关闭 draft graph，保留 target/verify/state graph 做消融。可单独用 `benchmark/runtime/probe_draft_head.py --model ... --output ...` 重现词表投影微基准。

## 6. 边界与剩余工作

- 仍有不支持的形状安全回退 eager，不能声称全路径 CUDA Graph。
- 尚未实现 fused target LM-head+argmax、通用 FX/Inductor pass manager、HTTP/overlap 投机服务集成或随机采样分布验收。
- 当前一致性针对 stable target greedy；不是所有框架的 BF16 输出逐位相同，也不保证跨 GPU/编译器一致。
- 本轮没有重跑 SGLang，不拿旧的 SGLang 数字宣称新代码已全面超过它。
- 原始遥测、比较 JSON、测试日志和 profiler 摘要位于 `/root/autodl-tmp/runtime-results/graph-opt-20260906/`；完整 trace 保留云端，附大小与 SHA256，避免把大型 trace 或模型权重混入 Git。
