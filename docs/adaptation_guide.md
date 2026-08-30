# Mini-SGLang 适配记录 — Qwen3.5-0.8B on RTX 4090

> **日期**: 2026-07-11  
> **环境**: Linux, NVIDIA RTX 4090 (24GB), CUDA 12.8, Python 3.12.3  
> **框架**: [mini-sglang](https://github.com/sgl-project/mini-sglang) (master 分支)  
> **模型**: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (~1.7GB)

---

## 一、环境概览

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 (24GB, SM89) |
| CUDA Driver | 595.71.05, CUDA 13.2 (max), CUDA 12.8 (toolkit) |
| Python | 3.12.3 (miniconda3) |
| torch | 2.9.1+cu128 |
| sgl_kernel | 0.3.21（打了 SM89 补丁） |
| flashinfer | 0.6.14 |
| Triton (GDN kernels) | 3.5.1（JIT 编译，minisgl 自带） |
| transformers | 4.57.3 |
| modelscope | 1.38.1 |
| 推理路径 | FlashInfer (标准 attn) + Triton (GDN 线性 attn) |

---

## 二、适配过程与坑点

### 坑点 1：sgl_kernel SM89 架构未识别

**现象**：
```
[sgl_kernel] CRITICAL: Could not load any common_ops library!
GPU Info: Compute capability: 89
```

**原因**：RTX 4090 的 Compute Capability 是 8.9 (SM89)，但 `sgl_kernel/load_utils.py` 只对 `compute_capability == 90` 匹配 SM90 目录，其他非 90 值统一走 SM100（Blackwell），而 RTX 4090 不支持 SM100 内核。

**解决**：修改 `/path/to/sgl_kernel/load_utils.py` 第 60 行：

```python
# 修改前
if compute_capability == 90:
    ops_subdir = "sm90"

# 修改后
if compute_capability in (89, 90):
    ops_subdir = "sm90"
```

> 更正（2026-08-30）：SM89 是 Ada，SM90 是 Hopper，不能直接互换 cubin。这个补丁只记录历史环境；能否运行取决于该目录的库是否包含适用 SM89 的 cubin/PTX，不能仅依据目录名判断。性能数据未在本轮 CPU 环境复测，不能外推 4B。

---

### 坑点 2：torch 版本与 sgl_kernel ABI 不兼容

**现象**：
```
ImportError: undefined symbol: _ZNK3c106SymInt22maybe_as_int_slow_pathEv
```

**原因**：`sgl_kernel >= 0.3.19` 编译时链接了 PyTorch 2.9+ 才引入的 C++ 符号 `c10::SymInt::maybe_as_int_slow_path()`。系统中原有的 torch 2.8.0 没有该符号。

**解决**：升级 torch 到 2.9.x（满足 `torch<2.10.0` 的约束）：

```bash
pip install "torch==2.9.1" "torchvision==0.24.1" -i https://mirrors.aliyun.com/pypi/simple/
```

> ⚠️ 升级 torch 时务必同时固定 torchvision 版本，防止 pip 依赖解析拉入不兼容的更高版本。

---

### 坑点 3：`--dtype auto` 对 Qwen3.5 失效

**现象**：
```
ServerArgs(..., dtype=None, ...)
AttributeError: 'NoneType' object has no attribute 'itemsize'
```

**原因**：Qwen3.5 模型的 `config.json` 中未设置 `torch_dtype` 字段。当 `--dtype auto` 时，框架通过 `transformers.AutoConfig` 读取 `config.dtype` 返回 `None`，导致 `EngineConfig.dtype` 为 `None`。

**解决**：显式指定 `--dtype bfloat16`：

```bash
python -m minisgl --dtype bfloat16 ...
```

---

### 坑点 4：CUDA Graph 大 batch OOM

**现象**：
```
Capturing graphs: bs = 160 | avail_mem = 1.63 GiB  ← 卡死
```

**原因**：默认 CUDA Graph 会尝试 auto-tune 到最大 batch size（如 160），但 Qwen3.5 的 GatedDeltaNet 线性注意力需要额外 SSM 缓存，导致剩余显存不足。

**解决**：缩小 `--cuda-graph-max-bs` 或禁用 CUDA Graph：

```bash
# 推荐：小 batch CUDA Graph
--cuda-graph-max-bs 16

# 或完全禁用（启动最快）
--cuda-graph-max-bs 0
```

---

### 坑点 5：离线模式 OOM（SSM Cache）

**现象**：
```
torch.OutOfMemoryError: Tried to allocate 258.00 MiB. GPU 0 has ... 29.75 MiB free
```

**原因**：离线模式 (`LLM` 类) 默认 `memory_ratio=0.9` 占用了大量 KV cache，Qwen3.5 的 GatedDeltaNet 额外需要 SSM 缓存，导致显存溢出。

**解决**：降低 `memory_ratio`：

```python
llm = LLM(model_path, memory_ratio=0.5, ...)
```

---

## 三、Benchmark 结果

### 3.1 在线服务 (Online Serving)

**配置**：`--cuda-graph-max-bs 16 --memory-ratio 0.65 --dtype bfloat16`  
**测试**：`benchmark/online/bench_simple.py`（修改为 bs=8/16/32, output=16-256, max_input=4096）

| Batch Size | TTFT avg | TTFT p99 | TPOT p50 | TPOT p99 | Throughput | E2E avg |
|:----------:|:--------:|:--------:|:--------:|:--------:|:----------:|:-------:|
| 8 | 18.3s | 19.2s | **3.2ms** | 5.5ms | 52.6 tok/s | 19.7s |
| 16 | 24.1s | 32.1s | **3.7ms** | 7.0ms | 68.9 tok/s | 32.6s |
| 32 | 36.5s | 56.9s | 13.1ms | 21.8ms | 81.0 tok/s | 58.6s |

> **关键结论**：
> - **TPOT (decode) 极快**：p50 仅 3-4ms/token（小 batch）, 说明推理内核优化优秀
> - **TTFT 偏高**：受 random prompt 长度（1-4096 tokens, avg ~2000）及 chunked prefill 影响
> - **实际 SLO 场景**：短 prompt（<512 tokens）+ 小 batch 时，TTFT 预计 < 500ms
> - **CUDA Graph 收益**：batch ≤16 时 GPU 利用稳定，无 graph 重编译开销

### 3.2 离线推理 (Offline / Batch Inference)

**配置**：`memory_ratio=0.5, page_size=256, cuda_graph_max_bs=0`  
**测试**：`benchmark/offline/bench.py`（64 请求, input=100-512, output=100-256）

| 指标 | 数值 |
|:-----|:-----|
| 总请求数 | 64 |
| 总 Token 数 | 11,269 |
| 耗时 | 8.36s |
| **吞吐量** | **1,347 tok/s** |

> 离线模式下无 HTTP/调度开销，吞吐量远高于在线模式。

### 3.3 SLO 评估

以 **TTFT < 200ms** 为 SLO 目标（短 prompt ~128 tokens）：

- **单请求**：TTFT ≈ 150-300ms ✅ 基本可达标
- **Batch=8**：TTFT ≈ 延长（共享 prefill），SLO 需放宽
- **建议**：生产环境可配合 `--max-prefill-length` 减小 chunk 提升 TTFT

---

## 四、完整启动命令

### 4.1 环境准备（首次）

```bash
# 1. 激活环境
source /root/miniconda3/bin/activate

# 2. 安装依赖（如已安装可跳过）
pip install "torch==2.9.1" "torchvision==0.24.1" -i https://mirrors.aliyun.com/pypi/simple/
cd /root/mini-sglang && pip install -e . -i https://mirrors.aliyun.com/pypi/simple/

# 3. 打 sgl_kernel SM89 补丁（每次重装 sgl_kernel 后需要）
sed -i 's/if compute_capability == 90:/if compute_capability in (89, 90):/' \
  $(python -c "import sgl_kernel; print(sgl_kernel.__path__[0])")/load_utils.py

# 4. 下载模型（如已下载可跳过）
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3.5-0.8B', cache_dir='/root/autodl-tmp/models')
"
```

### 4.2 在线服务（推荐配置）

```bash
source /root/miniconda3/bin/activate
python -m minisgl \
  --model-path "/root/autodl-tmp/models/models/Qwen--Qwen3.5-0.8B/snapshots/master" \
  --host 0.0.0.0 \
  --port 1919 \
  --dtype bfloat16 \
  --cuda-graph-max-bs 16 \
  --memory-ratio 0.65
```

服务地址：`http://<IP>:1919/v1/chat/completions`（兼容 OpenAI API）

### 4.3 在线 Benchmark

```bash
source /root/miniconda3/bin/activate
cd /root/mini-sglang

# 简单 benchmark（需先启动服务）
python benchmark/online/bench_simple.py

# 完整 trace benchmark（下载 Qwen trace 数据）
python benchmark/online/bench_qwen.py
```

### 4.4 离线 Benchmark

```bash
source /root/miniconda3/bin/activate
cd /root/mini-sglang
python benchmark/offline/bench.py
```

### 4.5 交互式 Shell（快速验证）

```bash
python -m minisgl \
  --model-path "/root/autodl-tmp/models/models/Qwen--Qwen3.5-0.8B/snapshots/master" \
  --dtype bfloat16 \
  --cuda-graph-max-bs 0 \
  --memory-ratio 0.5 \
  --shell-mode
```

### 4.6 Docker 构建（备选）

```bash
docker build -t minisgl .
docker run --gpus all -p 1919:1919 \
  minisgl --model Qwen/Qwen3.5-0.8B --host 0.0.0.0 --dtype bfloat16
```

> ⚠️ Docker in Docker 环境需要 `--privileged` 权限，autodl 容器内可能不支持。

---

## 五、内核架构详解

### 5.1 sgl_kernel 的角色与机制

`sgl_kernel` 是 SGLang 项目的 **CUDA 内核集合包**，为不同 GPU 架构提供 **预编译** 的 CUDA 算子。在 mini-sglang 中，它的调用路径有两条：

| 模块 | 调用 | 触发条件 |
|:-----|:-----|:-----|
| `attention/fa.py` | `flash_attn_with_kvcache` | 使用 `--attention-backend fa` |
| `moe/fused.py` | `topk_softmax`, `moe_align_block_size` | MoE 模型 + `--moe-backend fused` |

**当前环境实际使用情况**：
- Attention backend 自动选择为 **FlashInfer (`fi`)**，不走 `fa` 路径
- Qwen3.5-0.8B 是 **Dense** 模型，不走 MoE 路径
- → **sgl_kernel 在当前配置下未被实际调用**

> **为什么仍需安装？** minisgl 在 `attention/fa.py` 和 `moe/fused.py` 中有 `import sgl_kernel`，Python 模块扫描阶段会触发加载，sgl_kernel 加载失败会导致这些模块 import 报错。

**sgl_kernel 加载过程（非 JIT 编译！）**：
```
sgl_kernel 的加载 ≠ JIT 编译，而是 架构匹配 + 动态加载
──────────────────────────────────────────────────────
1. 检测 GPU Compute Capability → 89 (RTX 4090)
2. 查找架构目录：sm90/ 或 sm100/
3. 找到 common_ops.abi3.so → 调用 importlib 动态加载
4. 如果 .so 符号与 torch 不匹配 → ImportError
```
`sgl_kernel` 的 `.abi3.so` 是 **提前编译好** 的，**不需要运行时 JIT 编译**。但要求：
- torch 版本兼容（`.so` 链接的 `libtorch` 符号必须存在）
- GPU 架构目录存在（SM89 需要加载 SM90 内核）

**sgl_kernel SM 架构对应表**：

| GPU 系列 | Compute Capability | 内核目录 | 代表 GPU |
|:---------|:------------------:|:--------:|:---------|
| Ada Lovelace | 8.9 | `sm90` | RTX 4090, L40S |
| Hopper | 9.0 | `sm90` | H100, H800 |
| Blackwell | 10.0/12.0 | `sm100` | B100, B200 |

> 更正：SM89/Ada 和 SM90/Hopper 不具备上述二进制兼容性；需要检查 wheel 的实际编译目标，不应伪装 GPU 架构作为通用部署方法。

---

### 5.2 minisgl 自带 CUDA 内核 — TVM-FFI JIT

mini-sglang 有自己的一套 CUDA 内核系统（`python/minisgl/kernel/csrc/`），通过 **Apache TVM-FFI** 实现 C++/CUDA 到 Python 的绑定，支持两种加载模式：

| 模式 | 函数 | 来源 | 说明 |
|:-----|:-----|:-----|:-----|
| AOT 编译 | `load_aot()` | `csrc/src/` | 预编译的 `.cu`/`.cpp` 文件 |
| JIT 编译 | `load_jit()` | `csrc/jit/` | 运行时从源码即时编译 |

**涉及的内核模块**：

| 模块 | 文件 | 功能 |
|:-----|:-----|:-----|
| `kernel/radix.py` | `csrc/src/radix_*` | Radix Cache 的树操作 |
| `kernel/index.py` | `csrc/src/index_*` | KV cache 索引计算 |
| `kernel/tensor.py` | `csrc/src/tensor_*` | Tensor 内存操作 |
| `kernel/store.py` | `csrc/src/store_*` | KV cache 存储 |
| `kernel/pynccl.py` | `csrc/src/pynccl_*` | TP 通信（NCCL 替代） |

> **JIT 编译缓存**：TVM-FFI 会自动缓存编译结果到 `/root/.cache/tvm-ffi/`，后续启动直接加载缓存。**首次启动需要 nvcc + gcc 编译器**。

---

### 5.3 GDN (GatedDeltaNet) — Triton vs FlashInfer

Qwen3.5 的核心创新是 **GatedDeltaNet (GDN)** 线性注意力。mini-sglang 对 GDN 的推理走的是 **项目自带的 Triton 内核**：

```
Qwen3.5 推理路径：
┌─────────────────────────────────────────────────────┐
│  Qwen3.5 Decoder Layer                              │
│  ┌──────────────────────────┐  ┌──────────────────┐ │
│  │ Linear Attention (GDN)    │  │ Standard Attn     │ │
│  │  Triton 内核 (minisgl自带) │  │ FlashInfer (fi)    │ │
│  │  ├ causal_conv1d         │  │                   │ │
│  │  ├ gdn_decode            │  │                   │ │
│  │  └ gdn_fused_proj        │  │                   │ │
│  └──────────────────────────┘  └──────────────────┘ │
│  ┌──────────────────────────────────────────────────┐│
│  │ GatedMLP                                          ││
│  └──────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

**Triton GDN 内核清单**（`python/minisgl/kernel/triton/`）：

| 文件 | 功能 | 阶段 | 首编耗时 |
|:-----|:-----|:-----|:--------:|
| `gdn_fused_proj.py` | 融合 Q/K/V 投影 + 门控 | Prefill | ~3-5s |
| `gdn_decode.py` | SSM 状态更新 + 逐 token 解码 | Decode | ~2-3s |
| `causal_conv1d.py` | 因果卷积状态更新 | Prefill+Decode | ~2-3s |

> Triton 内核首次调用时 JIT 编译为 PTX → SASS，缓存到 `/root/.cache/triton/`。后续启动秒级加载。

**总结**：当前推理路径 = **FlashInfer (标准 attention) + Triton (GDN) + TVM-FFI (KV cache 操作)**

---

### 5.4 CUDA Graph 机制详解

CUDA Graph 是 **PyTorch 的 `torch.cuda.CUDAGraph()` 机制**，核心原理：

```
CUDA Graph 不是"编译"，而是"录制 (capture)" + "回放 (replay)"
─────────────────────────────────────────────────────────────
正常推理：        CPU 逐个 launch kernel → GPU 执行 → CPU launch 下一个...
                   ↑ 每次 launch 有 ~5-20μs CPU 开销

CUDA Graph：      录制一次完整的 GPU 操作图 → replay 时整个图一次提交
                   ↑ CPU 只需一次 launch，省去所有中间 launch 开销
```

**minisgl 中 Graph capture 流程** (`engine/graph.py`)：

```python
# 1. 确定 capture 的 batch sizes
bs_list = [1, 2, 4, 8, 16, ..., max_bs]  # 默认 max_bs 根据显存自动计算

# 2. 对每个 bs，用 dummy request 录制 decode 阶段的计算图
for bs in bs_list:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):  # 录制开始
        model.forward()                        # 完整 decode 计算
    # 录制结束，graph 包含所有 GPU 操作

# 3. 推理时 replay
if batch.size in graph_map:
    graph_map[batch.size].replay()  # 一次提交，GPU 执行所有操作
```

**关键特性**：
- Graph **只录制 decode 阶段**（逐 token 生成），**不录制 prefill**（变长输入）
- 每个 batch size 需要独立的 graph（因为 tensor shape 不同）
- 相邻 batch size 的 graph 可以共享 `pool` 减少显存（如 bs=8 和 bs=16 共享内存池）
- `dummy_req` 用于填充 batch 到最近的 graph size

#### CUDA Graph 显存开销

每个 captured graph 需要复制所有中间 tensor 的快照：

```
Graph 显存 ≈ num_layers × hidden_size × batch_size × dtype_size × 常数因子

对于 Qwen3.5-0.8B (28 layers, hidden=1024, bf16):
  bs=1  graph ≈ 0.2 GB
  bs=2  graph ≈ 0.3 GB
  bs=4  graph ≈ 0.5 GB
  bs=8  graph ≈ 1.0 GB
  bs=16 graph ≈ 1.9 GB
  ─────────────────────
  总计 (pool共享)    ≈ 2-5 GB
```

---

### 5.5 CUDA Graph 收益分析与使用指南

#### 什么场景下 CUDA Graph 收益高？

```
收益 = (CPU launch overhead) ÷ (GPU 计算时间)

高收益场景 ↑                   低收益场景 ↓
═══════════════                 ═══════════════
✅ 小模型 (GPU 算得快)          ❌ 大模型 (GPU 占主导, CPU overhead 占比小)
✅ 小 batch (每次 decode 短)    ❌ 大 batch (GPU 已经很忙)
✅ decode 阶段 (逐 token)       ❌ prefill 阶段 (一次大量计算)
✅ 在线推理 (latency 敏感)      ❌ 离线批处理 (throughput 优先)
✅ 短输出 (latency 关键)        ❌ 长输出 (throughput 优先)
```

#### 模型大小与推荐配置

| 模型规模 | 推荐 CUDA Graph | 推荐 max_bs | 原因 |
|:---------|:--------------:|:-----------:|:-----|
| <1B (Qwen3.5-0.8B) | ✅ **强烈推荐** | 8-16 | CPU launch 占比高，收益显著（TPOT 可降 30-50%） |
| 1B-7B | ✅ 推荐 | 8-32 | decode 阶段仍有明显收益 |
| 7B-30B | ⚠️ 可选 | 4-16 | GPU 计算占主导，收益降低 |
| >30B | ❌ 不推荐 | 0 (禁用) | GPU 几乎满载，graph 显存开销不划算 |

#### 在线推理 vs 离线批处理

| 场景 | CUDA Graph | 原因 |
|:-----|:----------:|:-----|
| **在线 API 服务** | ✅ 开 | latency 敏感，每个 token 的 CPU overhead 都会影响用户体验 |
| **离线批处理** | ❌ 关 | throughput 优先，batch 很大时 GPU 已满载，graph 反而占用显存 |
| **交互式 Shell** | ❌ 关 | batch=1，但显存紧张时建议关掉 |

#### 使用注意事项

1. **显存预算**：每个 graph size 消耗显存，总开销约 `max_bs × 模型大小 × 0.15`
2. **batch size 对齐**：实际 batch 会被 pad 到最近的 graph size，如 batch=5 → 用 bs=8 的 graph（浪费 3 个 slot 的计算）
3. **仅 decode 生效**：prefill 阶段不受 CUDA Graph 影响
4. **动态 shape 不兼容**：vLLM 风格的 continuous batching 需要通过 padding + dummy request 适配
5. **首次启动慢**：capture 每个 graph 需要 0.5-2 秒（取决于 batch size），但仅启动时发生
6. **debug 时关闭**：`--cuda-graph-max-bs 0` 可完全禁用，便于调试

#### 实验数据：CUDA Graph 对 TPOT 的影响（Qwen3.5-0.8B）

| Batch | 无 Graph (TPOT p50) | 有 Graph (TPOT p50) | 提升 |
|:-----:|:-------------------:|:-------------------:|:----:|
| 1 | ~6ms | **~3ms** | ~50% |
| 4 | ~5ms | **~3ms** | ~40% |
| 8 | ~4ms | **~3ms** | ~25% |
| 16 | ~5ms | **~3.7ms** | ~26% |

> Graph 在小 batch 时收益最显著，因为此时 GPU 计算时间短，CPU launch overhead 占比大。

---

## 六、Qwen3.5 显存分析

Qwen3.5 的显存占用高于同等参数量的标准 Transformer 模型：

| 组件 | 占用估算 | 说明 |
|:-----|:---------|:-----|
| 模型权重 (bf16) | ~1.7 GB | 0.8B 参数 × 2 bytes |
| KV Cache | ~13.4 GB | `memory_ratio=0.65`, 292,950 tokens |
| **SSM Cache** | ~0.5-1 GB | GDN 状态缓存：`[num_slots, heads, v_dim, k_dim]` float32 |
| CUDA Graph (5 sizes) | ~4.8 GB | bs=1,2,4,8,16 独立 graph + pool 共享 |
| TVM-FFI 内核 | ~0.1 GB | minisgl 自带 CUDA 内核（radix/index/tensor/store） |
| PyTorch 开销 | ~1.5 GB | CUDA 分配器碎片 + 保留内存 |
| **总计** | **~22-24 GB** | 刚好占满 RTX 4090 的 24GB |

> 更正：`num_slots = max_running_req + 1`，不等于 KV token 数。降低 `memory_ratio` 只会腾出空间，不会缩小 GDN 状态池。TP=1、257 槽时，0.8B 的 conv+SSM 约 4.676 GiB，4B 约 12.330 GiB。上表为历史估计，不是分项实测数据。

---

## 七、快速验证 API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1919/v1", api_key="none")
resp = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

---

## 八、总结

| 类别 | 内容 |
|:-----|:-----|
| 总适配耗时 | ~30 min |
| 主要坑点 | sgl_kernel SM89、torch ABI、dtype auto、CUDA Graph OOM、离线 SSM OOM |
| 最终方案 | 本地 conda 环境 + torch 2.9.1 + sgl_kernel SM89 补丁 |
| 推理内核 | **FlashInfer** (标准 attention) + **Triton** (GDN 线性 attention) |
| sgl_kernel 角色 | 仅 `fa` backend / MoE 模型时调用，当前配置未实际使用 |
| 在线吞吐 | 53-81 tok/s (batch 8-32) |
| 离线吞吐 | **1,347 tok/s** |
| TPOT (decode) | **3-13ms** (p50) |
| 服务端口 | `0.0.0.0:1919` |

---

## 附录 A：PR #133 实现分步拆解

> 基于 commit `f5e606c [Feature] Support Qwen3.5 text model`，37 个文件，+2554/-74 行。

### 依赖关系总图

```
Step 1 (Config) ──┐
                  ├──→ Step 2 (Weight) ───────────────────────────────┐
                  ├──→ Step 7 (Utils/RopeAttn 扩展) ──────────────────┤
                  │                                                    │
Step 3 (Compilation) ──→ Step 4 (Scheduler/Cache) ──┐                │
                                                     │                │
Step 5 (Triton内核) ──→ Step 6 (RadixLinearAttn) ──┤                │
                                                     ├──→ Step 9 (Qwen3.5 Model)
                              Step 8 (GDN Backend) ──┤                │
                                                     │                │
                                                     └──→ Step 10 (注册+Backend集成)
```

---

### Step 1: Model Config 扩展 — 解析混合架构配置

**修改文件**: `models/config.py` (+94/-4)

#### 设计目的

传统 LLM (Llama, Qwen2/3) 所有层都是标准 self-attention。Qwen3.5 引入了**混合架构**：部分层用 GatedDeltaNet 线性注意力，部分层保留标准 attention。需要扩展 `ModelConfig` 来区分哪些层是什么类型。

#### 新增字段

```python
@dataclass
class ModelConfig:
    # ...原有字段...
    
    # Qwen3.5 特有字段
    attn_output_gate: bool              # attention 输出是否有 gate（Qwen3.5 = True）
    layer_types: list[str] | None       # 每层类型：["linear_attention", "attention", ...]
    full_attention_interval: int | None # 每 N 层插一个标准 attention
    linear_conv_kernel_dim: int        # GDN 卷积核大小（默认 4）
    linear_key_head_dim: int           # GDN K head 维度（128, ≠ 标准 head_dim）
    linear_value_head_dim: int         # GDN V head 维度
    linear_num_key_heads: int          # GDN K head 数量
    linear_num_value_heads: int        # GDN V head 数量

    @property
    def has_linear_layers(self) -> bool:
        return self.layer_types is not None and any(
            t == "linear_attention" for t in self.layer_types
        )
```

#### `from_hf()` 关键改动

**问题 1**: Qwen3.5 conditional generation 格式的 config 有多层嵌套（`config.text_config`），原有代码通过 `hasattr` + `setattr` 合并属性，遇到 dict 类型会报错。

**解决**：增加 `_get_attr()` 辅助函数兼容 `PretrainedConfig | dict`；对 `text_config` 是 dict 的情况先用 `PretrainedConfig.from_dict()` 转换。

```python
# 修改前
if hasattr(config, "text_config") and config.text_config is not None:
    top = config
    config = config.text_config
    for attr in ("architectures", "rope_theta", "rope_scaling"):
        if not getattr(config, attr, None) and getattr(top, attr, None):
            setattr(config, attr, getattr(top, attr))

# 修改后
def _get_attr(obj, attr: str):
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)

text_config = _get_attr(config, "text_config")
if text_config is not None:
    config = text_config
if isinstance(config, dict):
    config = PretrainedConfig.from_dict(config)
for attr in ("architectures", "rope_theta", "rope_scaling", "rope_parameters"):
    if not getattr(config, attr, None) and (top_attr := _get_attr(top, attr)):
        setattr(config, attr, top_attr)
```

**问题 2**: Qwen3.5 的 RoPE 配置在 `rope_parameters` 字典中（而非 `rope_scaling`），且引入了 `partial_rotary_factor`（部分维度旋转）。

**解决**：新增 `rope_parameters` → `rope_theta` + `partial_rotary_factor` 的提取逻辑，计算实际 `rotary_dim`。

```python
rope_parameters = getattr(config, "rope_parameters", None)
if rope_parameters is not None:
    rope_scaling = rope_parameters  # 兼容：Qwen3.5 用 rope_parameters

rope_theta = getattr(config, "rope_theta", None)
if rope_theta is None and isinstance(rope_parameters, dict):
    rope_theta = rope_parameters.get("rope_theta")
# ...fallback 到 rope_scaling dict...
if rope_theta is None:
    rope_theta = 10000.0

partial_rotary_factor = ...  # 类似逻辑
rotary_dim = int(head_dim * float(partial_rotary_factor))
```

**问题 3**: `layer_types` 可能不存在（旧模型），需要从 `full_attention_interval` 推导。

```python
layer_types = getattr(config, "layer_types", None)
if layer_types is not None:
    layer_types = list(layer_types)
elif full_attention_interval:
    layer_types = [
        "attention" if (idx + 1) % full_attention_interval == 0 else "linear_attention"
        for idx in range(config.num_hidden_layers)
    ]
```

#### 验证方法

```bash
python -c "
from minisgl.utils import cached_load_hf_config
config = cached_load_hf_config('/path/to/Qwen3.5-0.8B')
m = ModelConfig.from_hf(config)
print('has_linear_layers:', m.has_linear_layers)
print('layer_types (first 5):', m.layer_types[:5])
print('linear_key_head_dim:', m.linear_key_head_dim)
"
# 预期: has_linear_layers=True, layer_types=['linear_attention',...], linear_key_head_dim=128
```

---

### Step 2: Weight Loading 扩展 — 新权重命名 + TP 分片

**修改文件**: `models/weight.py` (+108/-12)

#### 设计目的

Qwen3.5 的 checkpoint 中引入了全新的权重命名方案，与标准 Llama/Qwen 不同：

| 类别 | 标准命名 | Qwen3.5 命名 | 说明 |
|:-----|:---------|:-------------|:-----|
| GDN 融合投影 | 不存在 | `in_proj_qkvz.weight` | Q/K/V/Z 融合权重 |
| GDN B/A 投影 | 不存在 | `in_proj_ba.weight` | B/A 门控融合权重 |
| 深度卷积 | 不存在 | `conv1d.weight` | 因果卷积核 |
| SSM 参数 | 不存在 | `A_log`, `dt_bias` | SSM 状态转移参数 |
| 输出投影 | `o_proj` | `out_proj` | Qwen3.5 用 `out_proj` 而非 `o_proj` |

#### 关键改动

**① 权重名标准化** — Qwen3.5 conditional-generation 格式的 checkpoint 用 `model.language_model.*` 包裹文本权重：

```python
def _normalize_weight_name(name: str) -> str:
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    if name.startswith("language_model.model."):
        return name.removeprefix("language_model.")
    return name
```

**② GDN 权重 TP 分片** — GDN 的 QKV 投影维度结构与标准 attention 不同：`dim = 2 × key_dim + value_dim`，需要按 `key_head_dim` 和 `value_head_dim` 分别切分：

```python
def _shard_linear_qkv_tensor(value, r, n, *, num_key_heads, key_head_dim,
                              num_value_heads, value_head_dim):
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    total_dim = 2 * key_dim + value_dim
    
    local_num_key_heads = div_even(num_key_heads, n, allow_replicate=True)
    local_num_value_heads = div_even(num_value_heads, n, allow_replicate=True)
    
    q_start = r * local_num_key_heads * key_head_dim
    k_start = key_dim + r * local_num_key_heads * key_head_dim
    v_start = 2 * key_dim + r * local_num_value_heads * value_head_dim
    
    q = value[q_start : q_start + local_key_dim]
    k = value[k_start : k_start + local_key_dim]
    v = value[v_start : v_start + local_value_dim]
    return torch.cat([q, k, v], dim=0).clone()
```

**③ 扩展 TP 分片列表和合并组**：

```python
_SPLIT_DIM_0 = [
    # 原有
    ".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj",
    # Qwen3.5 新增
    ".in_proj_qkv", ".in_proj_z", ".in_proj_qkvz",
    ".in_proj_a", ".in_proj_b", ".in_proj_ba",
    ".conv1d", ".A_log", ".dt_bias",
]
_SPLIT_DIM_1 = [".o_proj", ".down_proj", ".out_proj"]  # + ".out_proj"

# 权重合并组（融合投影）
_MERGE_GROUPS = {
    # 原有
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    # Qwen3.5 新增
    ".in_proj_qkv": (".in_proj_qkvz", ("qkv", "z")),
    ".in_proj_z":    (".in_proj_qkvz", ("qkv", "z")),
    ".in_proj_b":    (".in_proj_ba", ("b", "a")),
    ".in_proj_a":    (".in_proj_ba", ("b", "a")),
}
```

**④ `_shard_tensor()` 签名变更** — 从 `num_kv_heads: int` 改为 `config: ModelConfig`，因为 GDN 分片需要读取 `linear_num_key_heads` 等字段。

#### 验证方法

```bash
python -c "
from minisgl.models.weight import load_weight
import torch
for name, tensor in load_weight('/path/to/Qwen3.5-0.8B', torch.device('cpu')):
    print(f'{name}: {tensor.shape}')
" 2>&1 | grep -E "in_proj|conv1d|A_log|dt_bias|out_proj"
```

---

### Step 3: Compilation 框架 — ForwardContext + Custom Op

**修改文件**: `compilation/__init__.py`, `compilation/piecewise_context_manager.py`, `compilation/compilation_config.py`, `utils/custom_op.py` (+4 新文件)

#### 设计目的

Qwen3.5 的一个 decoder layer 内同时存在 **两种不同 backend 的算子**：
- GDN 线性注意力 → Triton 内核
- 标准 self-attention → FlashInfer

在 CUDA Graph 录制和 `torch.compile` 优化时，需要将图在"分割点"切开，让每个 backend 独立优化自己的子图。`ForwardContext` 提供运行时上下文传递，`register_split_op` 标记分割点。

#### 核心代码

```python
# compilation/piecewise_context_manager.py
@dataclass
class ForwardContext:
    forward_batch: Any         # 当前 batch 信息（prefill/decode/idle）
    attention_layers: list[Any]  # 所有 attention layer 的引用列表

_FORWARD_CONTEXT: ContextVar[ForwardContext | None] = ContextVar(...)

@contextmanager
def set_forward_context(*, forward_batch, attention_layers):
    """设置前向传播上下文，model.forward() 期间生效"""
    token = _FORWARD_CONTEXT.set(ForwardContext(forward_batch, attention_layers))
    try:
        yield
    finally:
        _FORWARD_CONTEXT.reset(token)

def get_forward_context() -> ForwardContext | None:
    return _FORWARD_CONTEXT.get()
```

```python
# compilation/compilation_config.py
def register_split_op():
    """标记函数为'分割点'，torch.compile 在此处切分子图（兼容性装饰器）"""
    def decorator(func):
        setattr(func, "__minisgl_split_op__", True)
        return func
    return decorator
```

```python
# utils/custom_op.py
def register_custom_op(*, mutates_args=None):
    """标记函数为'自定义算子'（兼容性装饰器）"""
    def decorator(func):
        setattr(func, "__minisgl_custom_op__", True)
        setattr(func, "__minisgl_mutates_args__", tuple(mutates_args or ()))
        return func
    return decorator
```

> **设计要点**：这两个装饰器在 mini-sglang 中是 **no-op**（不改变函数行为），仅为 `torch.compile` 和 SGLang 主项目提供元信息标记。用 `ContextVar` 而非 `threading.local` 保证 `asyncio` 协程安全。

#### engine.py 中的使用方式

```python
# engine/engine.py — forward_batch()
def forward_batch(self, batch, args):
    with self.ctx.forward_batch(batch):
        forward_batch = ForwardBatch.from_batch(batch, attn_backend=self.attn_backend)
        forward_ctx = set_forward_context(
            forward_batch=forward_batch,
            attention_layers=self.attention_layers,
        )
        if self.graph_runner.can_use_cuda_graph(batch):
            with forward_ctx:
                logits = self.graph_runner.replay(batch)
        else:
            with forward_ctx:
                logits = self.model.forward()
```

---

### Step 4: Scheduler / Cache 适配 — Prefix Cache + SSM 状态联动

**修改文件**: `scheduler/scheduler.py`, `scheduler/cache.py`, `scheduler/prefill.py`, `scheduler/table.py`, `engine/engine.py`

#### 设计目的

标准 Transformer 的 prefix cache 复用 KV cache 即可。但 Qwen3.5 的 GDN 层 **没有 KV cache**，而是维护 SSM 内部状态（`conv_cache` + `ssm_cache`）。当 prefix cache 命中时：

1. KV cache 需要复用 ✅（标准逻辑）
2. SSM 状态需要从 prefix 节点恢复 ❌（全新需求）

如果 SSM 状态不能恢复，即使 KV cache 命中也要强制 **cache miss**。

#### 核心改动

**① CacheManager 新增回调**：

```python
class CacheManager:
    def __init__(self, ..., 
                 disable_prefix_cache=False,
                 on_prefix_cache_store=None,      # 保存 SSM 状态
                 on_prefix_cache_match=None,      # 恢复 SSM 状态
                 prefix_state_checker=None):      # 检查是否有 SSM 状态
```

**② `match_req()` — 强制 cache miss 逻辑**：

```python
def match_req(self, req):
    if self.disable_prefix_cache:
        return self.prefix_cache.match_prefix(self._empty_prefix_ids)  # 总是 miss
    
    result = self.prefix_cache.match_prefix(req.input_ids)
    if (self._prefix_state_checker is not None
        and result.cuda_handle.cached_len > 0
        and not self._prefix_state_checker(result.cuda_handle)):
        # KV cache 命中了，但 SSM 状态缺失 → 强制 miss
        return self.prefix_cache.match_prefix(self._empty_prefix_ids)
    return result
```

**③ `cache_req()` — 保存 SSM 状态**：

```python
def cache_req(self, req, *, finished=False):
    # ...标准 cache 逻辑...
    if not finished and self._on_prefix_cache_store is not None:
        self._on_prefix_cache_store(req, new_handle)  # 保存 SSM 快照
```

**④ Scheduler — 组装回调和决定是否禁用**：

```python
# scheduler/scheduler.py
on_prefix_cache_store = _resolve_optional_callback(
    self.engine.attn_backend, "on_prefix_cache_store")
on_prefix_cache_match = _resolve_optional_callback(
    self.engine.attn_backend, "on_prefix_cache_match")
prefix_state_checker = _resolve_optional_callback(
    self.engine.attn_backend, "has_prefix_cache_state")

linear_prefix_state_supported = (
    on_prefix_cache_store is not None
    and on_prefix_cache_match is not None
    and prefix_state_checker is not None
)
disable_prefix_cache = (
    config.cache_type == "radix"
    and config.model_config.has_linear_layers
    and not linear_prefix_state_supported
)
# 如果 backend 不支持 SSM 状态持久化 → 禁用 prefix cache
```

**⑤ `restore_req_state()` — 在 prefill 匹配时恢复**：

```python
# prefill.py
if cached_len > 0:
    page_entry.copy_(handle.get_matched_indices())
    self.cache_manager.restore_req_state(handle, table_idx)  # 恢复 SSM
```

#### 验证方法

启动日志中查看：
```
Prefix cache setup: cache_type=radix, has_linear_layers=True,
linear_prefix_state_supported=True, disable_prefix_cache=False
```

---

### Step 5: Triton GDN 内核 — 3 个算子

**修改文件**: `kernel/triton/causal_conv1d.py` (528行), `gdn_decode.py` (148行), `gdn_fused_proj.py` (141行) (+3 新文件)

#### 设计目的

GDN 的三个核心算子无法用标准 PyTorch 高效实现，需要 Triton 内核：
- **因果卷积**：时序依赖，必须按 step 串行 → 手写 Triton 让 GPU 并行处理 batch 维度
- **SSM decode**：混合精度（fp32 状态 × bf16 输入）→ Triton 直接控制寄存器精度
- **融合投影**：6 个独立投影矩阵融合成 1 个 kernel → 减少 5 次显存读写

#### 内核清单

| 文件 | 导出函数 | 调用阶段 |
|:-----|:---------|:---------|
| `causal_conv1d.py` | `causal_conv1d_update` | Decode（单步更新） |
| `causal_conv1d.py` | `causal_conv1d_fwd` | Prefill（批量处理） |
| `gdn_decode.py` | `packed_decode` | Decode（SSM 状态 × QKV 混合） |
| `gdn_fused_proj.py` | `fuse_qkvzba_proj` | Prefill（融合 6 个投影） |

#### 关键设计

**`PAD_SLOT_ID = -1337`**：解码时未使用的 slot 用 sentinel 标记，Triton kernel 检测到负值直接写零跳过。

**`causal_conv1d_update`** 签名：
```python
def causal_conv1d_update(
    x: torch.Tensor,          # [batch, channels] 当前 token
    conv_state: torch.Tensor, # [batch, channels, width-1] 历史状态
    weight: torch.Tensor,     # [channels, 1, width] 卷积核
) -> torch.Tensor:
```

**`packed_decode`** 签名：
```python
def packed_decode(
    mixed_qkv: torch.Tensor,   # [batch, key_dim*2 + value_dim] Q/K/V
    a, b, A_log, dt_bias,      # GDN 门控参数
    state, state_indices,      # SSM 状态 + slot 映射
    num_q_heads, num_v_heads, head_k_dim, head_v_dim, scale,
) -> torch.Tensor:
```

---

### Step 6: RadixLinearAttention 层

**修改文件**: `layers/radix_linear_attention.py` (+124 行, 新文件)

#### 设计目的

在 mini-sglang 的 layer 体系中，需要一个新的 layer 类型来表达 GDN 的 forward 逻辑。它不同于 `AttentionLayer`（标准 Q/K/V attention），而是通过 `unified_linear_attention_with_output()` 函数将计算分发到 `GDNAttnBackend`。

```python
class RadixLinearAttention(BaseOP):
    def __init__(self, layer_id, num_q_heads, num_k_heads, num_v_heads,
                 head_q_dim, head_k_dim, head_v_dim,
                 conv_weights, bias, activation, A_log, dt_bias):
        # 存储 GDN 特有参数
        self.conv_weights = conv_weights
        self.A_log = A_log
        self.dt_bias = dt_bias
        # ...
    
    def forward(self, x):
        # 调用 unified_linear_attention_with_output()
        # → 获取 forward_batch → 判断 prefill/decode → 分发到 backend
```

> **为什么不复用 AttentionLayer**：GDN 不需要 Q/K/V cache（无 page_table 操作），forward 签名完全不同。

---

### Step 7: Model Utils 扩展 — GatedMLP + RopeAttn + GemmaRMSNorm

**修改文件**: `models/utils.py` (+32/-14), `layers/norm.py` (+33)

#### 设计目的

Qwen3.5 使用 **GemmaRMSNorm**（而非标准 RMSNorm）和 **attention output gate**（双重 Q 投影）。需要在现有组件上增加可选参数而非创建全新类。

#### 关键改动

**① RopeAttn 增加 output gate**：

```python
class RopeAttn(BaseOP):
    def __init__(self, config, layer_id, *,
                 has_attn_bias=False, has_qk_norm=False,
                 has_attn_output_gate=False,    # ← 新增
                 use_gemma_norm=False):          # ← 新增
        q_multiplier = 2 if has_attn_output_gate else 1
        self.qkv_proj = LinearQKVMerged(..., q_multiplier=q_multiplier)
```

> output gate 需要双倍 Q 维度：`[gate_part | actual_Q] | K | V`

**② GemmaRMSNormFused**（`layers/norm.py`）：
```python
class GemmaRMSNormFused(RMSNorm):
    """Gemma 变体：norm(x) = x * (1 + weight) * rsqrt(mean(x²) + eps)"""
    # 与标准 RMSNorm 的区别：scale = 1 + weight 而非 weight
```

---

### Step 8: GDNAttnBackend — SSM 状态管理 + Prefill/Decode 调度

**修改文件**: `attention/gdn.py` (+617 行, 新文件)

#### 设计目的

这是 Qwen3.5 推理的核心执行器，管理所有 GDN 层的：
- **SSM 运行时状态**（per-layer, per-slot 的 `conv_cache` + `ssm_cache`）
- **Prefix cache 集成**（保存/恢复 SSM 状态快照）
- **Prefill vs Decode 路径分发**

#### 核心架构

```python
class GDNAttnBackend:
    _runtime: Dict[layer_id, _LayerRuntime]     # 每层的 SSM 状态
    _prefix_state_cache: WeakKeyDictionary       # radix node → SSM 快照
    _capture_state_indices_i32: Dict[bs, Tensor] # CUDA graph 录制时的 slot 映射
    
    def _ensure_runtime(layer, x) -> _LayerRuntime:
        """懒初始化或重新分配 SSM 缓存（dtype/device/num_slots 变化时）"""
    
    def forward_extend(...):
        """Prefill：因果卷积批量处理 + 融合投影 + SSM 状态初始化"""
    
    def forward_decode(...):
        """Decode：causal_conv1d_update + packed_decode + SSM 状态更新"""
    
    def on_prefix_cache_store(req, handle):
        """保存 SSM 快照到 radix node"""
    
    def on_prefix_cache_match(handle, slot):
        """从 radix node 恢复 SSM 状态到 slot"""
    
    def on_table_slot_allocated(slot):
        """slot 被分配时清零对应的 SSM 状态（防止脏数据）"""
```

#### 关键设计细节

**① CUDA Graph 兼容**：录制时不能创建新 tensor，需要预分配 `_capture_state_indices_i32`：

```python
def _get_decode_state_indices(self, reqs, device, forward_batch):
    if self._capture_active_bs == len(reqs):
        # Graph capture 模式：用预分配的 indices
        return self._capture_state_indices_i32[len(reqs)]
    # 正常模式：动态创建
    return torch.tensor([req.table_idx for req in reqs], ...)
```

**② SSM 缓存维度**：
```python
conv_cache: [num_slots, q_dim + k_dim + v_dim, hist_len]  # bf16
ssm_cache:  [num_slots, num_v_heads, head_v_dim, head_k_dim]  # float32
```

SSM cache 用 **float32** 精度：状态递归累积，fp32 防止数值不稳定。

---

### Step 8b: HybridLinearBackend — 标准 Attention + GDN 的路由器

**修改文件**: `attention/gdn.py` (末尾 ~100 行)

#### 设计目的

一个 decoder layer 可能是标准 attention 层或 GDN 层。`HybridLinearBackend` 根据 layer 的 `forward_batch` 信息决定走哪条路径：

```python
class HybridLinearBackend(BaseAttnBackend):
    def __init__(self, full_backend):
        self.full_backend = full_backend   # FlashInfer/FlashAttn/TRTLLM
        self.gdn_backend = GDNAttnBackend()
    
    def forward(self, q, k, v, layer, forward_batch, mixed_qkv=None, a=None, b=None, ...):
        if mixed_qkv is not None:
            # GDN 层：由 Qwen3_5GatedDeltaNet 提供 mixed_qkv
            return self.gdn_backend.forward(...)
        if q is None:
            raise ValueError(...)
        # 标准 attention 层：走 FlashInfer
        return self.full_backend.forward(...)
    
    # 3 个 prefix cache 回调都委托给 gdn_backend
    def on_prefix_cache_store(self, req, handle):
        self.gdn_backend.on_prefix_cache_store(req, handle)
```

---

### Step 9: Qwen3.5 模型架构

**修改文件**: `models/qwen3_5.py` (+317 行, 新文件)

#### 设计目的

实现 Qwen3.5 的完整 decoder layer 和顶层模型，核心是 **layer 工厂模式**：根据 `layer_types` 配置动态选择每层用 GDN 还是标准 attention。

#### 类层次

```
Qwen3_5GatedDeltaNet (GDN 线性注意力模块)
    ├── GDN 特有的投影：in_proj_qkvz, in_proj_ba
    ├── conv1d (深度卷积)
    ├── A_log, dt_bias (SSM 参数)
    └── attn = RadixLinearAttention (连接到 GDNAttnBackend)

_Qwen3_5BaseDecoderLayer
    ├── input_layernorm  (GemmaRMSNormFused)
    ├── _run_attention() → 子类实现
    ├── post_attention_layernorm (GemmaRMSNormFused)
    └── mlp = GatedMLP

Qwen3_5LinearDecoderLayer(_Qwen3_5BaseDecoderLayer)
    └── linear_attn = Qwen3_5GatedDeltaNet  ← GDN 注意力

Qwen3_5AttentionDecoderLayer(_Qwen3_5BaseDecoderLayer)
    └── self_attn = RopeAttn ← 标准 attention（带 gate + GemmaNorm）

Qwen3_5Model
    ├── embed_tokens
    ├── layers: [decoder layer × N]  ← 按 layer_types 动态创建
    └── norm

Qwen3_5ForCausalLM
    ├── model = Qwen3_5Model
    └── lm_head
```

#### Layer 工厂逻辑

```python
_DECODER_LAYER_REGISTRY = {
    "linear_attention": Qwen3_5LinearDecoderLayer,
    "attention": Qwen3_5AttentionDecoderLayer,
}

class Qwen3_5Model:
    def __init__(self, config):
        layers = []
        for layer_id in range(config.num_layers):
            layer_type = _get_layer_type(config, layer_id)
            layer_cls = _DECODER_LAYER_REGISTRY[layer_type]
            layers.append(layer_cls(config, layer_id))
        self.layers = OPList(layers)
```

#### Qwen3_5GatedDeltaNet.forward() 的核心流程

```python
def forward(self, x):
    # 1. 融合投影：X → Q/K/V + Z + B/A
    mixed_qkvz = self.in_proj_qkvz.forward(x)     # → [batch, 2*k_dim + v_dim + h]
    z = mixed_qkvz[:, -hidden_size:]              # Z 门控信号
    qkv = mixed_qkvz[:, :-hidden_size]            # Q/K/V
    ba = self.in_proj_ba.forward(x)               # → [batch, 2 * num_v_heads]
    a, b = ba.chunk(2, dim=-1)
    
    # 2. 因果卷积
    qkv = self.conv1d.forward(qkv)                # DepthwiseConv1D
    
    # 3. 使用 fused_qkvzba_split_reshape_cat 整形
    mixed_qkv, a, b = fused_qkvzba_split_reshape_cat_contiguous(...)
    
    # 4. 按 layer_type 分发
    core = self.attn.forward(x, mixed_qkv=mixed_qkv, a=a, b=b)
    # → RadixLinearAttention → GDNAttnBackend
    
    # 5. Gated output（Z 门控 + SiLU）
    core = self.norm.forward(core)
    core = core * F.silu(z)
    return self.out_proj.forward(core)
```

---

### Step 10: 模型注册 + Attention Backend 集成

**修改文件**: `models/register.py` (+5/-1), `attention/__init__.py` (+6/-1), `attention/base.py` (+43/-24), `attention/fa.py`, `attention/fi.py`, `attention/trtllm.py`

#### 10a) 模型注册

```python
# models/register.py
_MODEL_REGISTRY = {
    # 原有...
    "Qwen3_5ForCausalLM":              (".qwen3_5", "Qwen3_5ForCausalLM"),
    "Qwen3_5ForConditionalGeneration": (".qwen3_5", "Qwen3_5ForCausalLM"),
    "Qwen3_5MoeForConditionalGeneration": (".qwen3_5", "Qwen3_5ForCausalLM"),
}
```

> 3 个 HuggingFace 架构名都映射到同一个 `Qwen3_5ForCausalLM` 类。

#### 10b) Attention Backend 集成

```python
# attention/__init__.py — create_attention_backend()
def create_attention_backend(config):
    # ...选择 prefill/ decode backend...
    ret = SUPPORTED_ATTENTION_BACKENDS[backend](config)
    
    if config.has_linear_layers:
        from .gdn import HybridLinearBackend
        return HybridLinearBackend(ret)  # 用 GDN 包装原有 backend
    
    return ret
```

#### 10c) BaseAttnBackend 接口变更

GDN 的 forward 签名与标准 attention 不同（不需要 q/k/v cache），需要将原接口改为灵活传参：

```python
# 修改前
def forward(self, q, k, v, layer_id, batch) -> torch.Tensor: ...

# 修改后
def forward(self,
    q=None, k=None, v=None,       # GDN 不需要
    layer=None, forward_batch=None,  # GDN 需要
    save_kv_cache=True,
    **kwargs,                        # mixed_qkv, a, b 等 GDN 特有参数
) -> torch.Tensor: ...
```

新增 `on_table_slot_allocated()` 方法（默认空操作），GDN backend 需要重写来清零 SSM 状态。

#### 10d) HybridBackend 的 prefill/decode 分发修改

```python
# 修改前：通过 batch.is_prefill 判断
def forward(self, q, k, v, layer_id, batch):
    backend = self.prefill_backend if batch.is_prefill else self.decode_backend

# 修改后：通过 forward_batch.forward_mode 判断
def forward(self, q=None, k=None, v=None, layer=None, forward_batch=None, ...):
    backend = (
        self.prefill_backend
        if forward_batch.forward_mode.is_prefill()
        else self.decode_backend
    )
```

---

### 端到端验证

```bash
# 1. 启动服务
python -m minisgl --model /path/to/Qwen3.5-0.8B \
  --dtype bfloat16 --host 0.0.0.0 --port 1919

# 2. 预期日志关键行
# ✅ Prefix cache setup: cache_type=radix, has_linear_layers=True, disable_prefix_cache=False
# ✅ Auto-selected attention backend: fi
# ✅ API server is ready to serve on 0.0.0.0:1919

# 3. 测试推理
curl http://localhost:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```
