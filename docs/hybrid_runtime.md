Latest runtime: [batched prefill/draft, GDN optimization and continuous slot refill](full_batch_runtime.md). This is an experimental CLI path; HTTP scheduler integration remains separate work.

GPU validation update: see [2026-08-31 experiments](gpu_validation_2026-08-31.md). The 4B smoke gate passes, but expanded parallel token parity remains open. Use `--verify-mode sequential` only as a diagnostic oracle.

Batch/Graph update: [four-request demo](batch_graph_demo.md) documents opt-in wave batching, readable answers, repeat-cache hits and target decode CUDA graphs. The CPU-only implementation notes below describe the earlier stage.

# Memory-aware Qwen3.5 runtime: experimental implementation

本分支把 MiniSGLang 作为实验平台，研究混合状态缓存与基于实测成本的投机块长选择。实现日期 2026-08-30。当前云端只有 CPU，没有任何本项目的新 GPU 吞吐、加速比或 27B 部署结果。

## 已实现的范围

- 修复普通引擎的 GDN 前缀状态长度校验、快照时间边界、请求槽清零/恢复 stream 所有权，以及 `fa,fi` 独立状态池问题。
- 仅为 full-attention 层分配 KV，初始化预算包含活跃 conv/SSM、前缀快照、workspace 和外部 draft 权重。
- 独立实验入口直接调用真实 MiniSGLang Qwen3.5 target，增加全位置 verify、正确 residual hidden taps、原生 PyTorch DFlash v1 draft。
- greedy 接受、拒绝时恢复 conv/SSM 并重放已确认输入；KV 通过逻辑长度截断，拒绝位置后续覆盖。
- 完整前缀状态包包含 KV、conv、SSM、target features、最后位置 logits；GPU/CPU 两级预算、LRU/cost 策略、同步 offload 和恢复、恢复成本大于重算时选择重算。
- 自适应候选块长 1/2/4/8/16，按 batch/context bucket 记录进度和 draft+verify+恢复成本，EWMA、受限探索、显存不足时缩块或卸载 GPU 前缀缓存。
- 固定策略和自适应策略共用 workload/engine；记录逐轮成本、token 输出、TTFT、显存峰值、缓存事件、代码 revision 和配置/工作负载哈希。
- 可选 packed GDN extend kernel 将 token 循环放入 GPU，减少 Python launch。默认未开启；它不是 chunk-parallel WY 算法。

## 不能宣称已完成的部分

1. GPU logits 对齐、kernel 数值验证、4090 性能/显存实验尚未运行。
2. 实验 runner 是单卡、单请求、BF16、greedy、无 CUDA Graph/overlap。普通服务的修复不等于 DFlash 已接入 OpenAI 多请求服务。
3. 此处是自有轻量分层缓存，不是移植了 SGLang HiCache 的全部实现。当前 offload 对象是可复用前缀快照；没有将活跃序列的全部 KV 在每轮 GPU/CPU 间分页换入换出。
4. 27B INT4/AWQ/GPTQ loader 和量化 GEMM 尚未实现，实验入口会明确拒绝 quantization_config；也未验证可匹配的 27B draft。权重按 4 bit 计算只能得到存储下界，不能证明 24GB 能运行整个系统。
5. DFlash2、随机采样、continuous batching、多 GPU、异步预取、SSD 层、全局最优内存分配不在首版范围。
6. 内存规划包含保守 workspace 估计与实时 admission，不是对 PyTorch/CUDA 所有内部申请的硬限额。不得承诺永不 OOM，必须用 GPU 峰值校准。

## 缓存一致性

普通 Radix 路径只发布与 KV handle 长度精确一致的 GDN 快照。非页对齐 prefill 没有相应边界快照时会安全 miss；没有伪造较短状态。快照在 engine stream 上生成，后续 decode 不会修改它。请求槽的清零与恢复也在 engine stream 上执行。

实验 runner 的 cache key 是完整 token 前缀，payload 的所有状态对应同一个逻辑长度，不依赖 Radix 页对齐。KV/GDN 是一个原子一致性包，不能只命中其中一半。第一版按整包驱逐/offload，尚未实现 KV 与 GDN 各自独立迁移、再以重放补齐边界的复杂策略。每个 cache 实例绑定一个固定 target/draft 组合，不跨模型复用。

Cost 策略以访问次数、保存的重算时间、大小和最近访问时间排序；这是可与 LRU 对照的启发式，不宣称理论最优。CPU 恢复带宽从保守初值开始，再根据同步恢复的观测更新。大于预算的 entry 不缓存，不会突破 host/GPU cache 限额强行放入。

## DFlash 约定

Target：Qwen3.5 dense 文本模型。Draft：与 target 匹配的 Qwen3-style `DFlashDraftModel` checkpoint，默认 RoPE，SiLU。4B 可使用 `z-lab/Qwen3.5-4B-DFlash`，需固定 revision；不执行 Hugging Face remote Python。

参考来源：[DFlash](https://github.com/z-lab/dflash)、[4B draft 配置](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash/blob/main/config.json)。读取 checkpoint 的 block size、target taps、head dims、滑窗和 mask 语义，不把 draft 当成普通自回归小模型。

每轮 block 包含已知 anchor 和若干 draft token。Target logits 验证下一 token，接受连续匹配部分；拒绝后恢复到本轮前状态，并重放 anchor + 已接受部分。最后新输出的 token 尚未进入 target cache，下轮作为 anchor。因此缓存长度始终对应“prompt + 输出中除最后一个外的 token”。EOS 在块中出现时同样遵守这个约定。

Draft KV 只保留已确认 target context，不保留 noise block；滑窗层按 checkpoint 窗口裁剪，full-attention 层保留全部 context。连续 block=1 的轮次会积累尚未送入 draft 的特征，恢复 speculation 时一次补齐，不能漏掉这些轮次。

## 现在能运行的 CPU 验证

从 `/root/mini-sglang` 执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m pytest tests/cpu tests/gpu -q -o addopts='' -p no:cacheprovider
python -m minisgl.runtime.benchmark --help
python benchmark/runtime/inspect_memory.py \
  --config /root/autodl-tmp/models/models/Qwen--Qwen3.5-0.8B/snapshots/master/config.json \
  --slots 2 --context 4096
```

GPU tests 在无 GPU 时显式 skip，benchmark 在无 GPU 时退出 `GPU_UNAVAILABLE`，不生成伪造性能结果。CPU tests 包括独立自回归 oracle 的全接受/部分接受/连续拒绝/EOS/输出上限测试，以及真实小型 draft 的增量上下文与从头计算对照。

## GPU 可用后的实验命令

首先在单独环境用支持 Qwen3.5 的 Transformers 运行 `benchmark/runtime/hf_reference.py`。不要直接升级主环境的 transformers 4.57.3。参考环境必须能导入 `Qwen3_5ForConditionalGeneration`，并记录完整依赖版本。将 target-only 的 token 输出与 HF 对齐后再比较 DFlash。

当前服务器只有 4B target/draft 的小配置和 revision 清单，没有下载完整权重。模型存数据盘，不放只有约 3GB 剩余空间的系统盘。云端直连 Hugging Face 在本轮超时，小配置通过本机获取后同步；完整权重下载仍需可用的网络通路。

已锁定 target `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，draft `1905b5a05f974e82798be3ea0fd6baf2e4805a1c`。配置副本位于 `benchmark/runtime/configs/`，清单位于 `benchmark/runtime/models.lock.json`。

先运行 `python benchmark/runtime/prepare_models.py --directory /root/autodl-tmp/models/hybrid-runtime`，仅下载小配置并记录不可变 revision；加 `--weights` 才下载完整权重。再次运行会沿用 `models.lock.json` 的 revision。下面命令中的路径应替换为 manifest 里的实际路径。

```bash
python benchmark/runtime/make_workload.py \
  --output /root/autodl-tmp/runtime-results/shared.jsonl

python -m minisgl.runtime.benchmark \
  --model /root/autodl-tmp/models/Qwen3.5-4B \
  --mode target --workload benchmark/runtime/smoke.jsonl \
  --output /root/autodl-tmp/runtime-results/target.json

python -m minisgl.runtime.benchmark \
  --model /root/autodl-tmp/models/Qwen3.5-4B \
  --draft /root/autodl-tmp/models/Qwen3.5-4B-DFlash \
  --mode fixed --block-size 16 --workload benchmark/runtime/smoke.jsonl \
  --output /root/autodl-tmp/runtime-results/fixed.json

python -m minisgl.runtime.benchmark \
  --model /root/autodl-tmp/models/Qwen3.5-4B \
  --draft /root/autodl-tmp/models/Qwen3.5-4B-DFlash \
  --mode adaptive --workload benchmark/runtime/smoke.jsonl \
  --output /root/autodl-tmp/runtime-results/adaptive.json

python -m minisgl.runtime.analyze \
  /root/autodl-tmp/runtime-results/target.json \
  /root/autodl-tmp/runtime-results/fixed.json \
  /root/autodl-tmp/runtime-results/adaptive.json
```

缓存实验在上述固定 workload 上增加 `--host-cache-mib 512 --gpu-cache-mib 128 --cache-policy lru`，再仅把 policy 改为 `cost` 对照。按实际主机 RAM 调整 host budget。packed kernel 的对照仅改变 `--gdn-extend packed`，先通过 GPU kernel test，再比较 target-only token parity 与 TTFT。

与 HF 参考结果比较使用 `python -m minisgl.runtime.analyze --tokens-only <hf.json> <target.json>`。不能将 HF whole-generation 时间冒充 TTFT；不能在输出 token 不一致时报告有效加速比。

## 实验矩阵与验收

1. 正确性：0.8B/4B target 对 HF；target 对 fixed/adaptive；prefix cache off/on；GDN recurrent/packed；重复请求、prefix+suffix、EOS、页边界。
2. 缓存：no cache、GPU only、GPU+CPU LRU、GPU+CPU cost；控制公共前缀比例、长度、重算成本、host/GPU 限额。
3. 投机：block=1/2/4/8/16、adaptive；控制 context、工作负载难度和内存压力。controller 的 batch bucket 接口已存在，当前真实 runner 只传 batch=1。
4. 内存：例如 12/16/20/24GiB budget，记录成功的 context、峰值、cache offload/recompute/缩块事件。不能仅凭静态估算宣称可支持的最大 context。
5. 报告：单请求 TTFT、decode tokens/s、聚合 TPOT、每轮进度与成本、显存峰值；warmup 与实测分开。当前 runner 不提供并发服务 TPOT p99 或线上 SLO 结论。

GPU 验证发现浮点差异时，先检查 logits/hidden 与首次分歧位置，不能简单放宽 token 一致性要求。只有通过这组基线，再考虑批处理、Graph、异步 offload 和量化模型。

## 研究结论的边界

目前可以写成“设计并实现了实验原型与可复现实验管线”；不能写“27B INT4 在 4090 上达到某吞吐”“比 HiCache/DFlash 快多少”或“所有五个阶段已验收”。创新性是否成立，需要对照实验、消融和瓶颈解释支持。
