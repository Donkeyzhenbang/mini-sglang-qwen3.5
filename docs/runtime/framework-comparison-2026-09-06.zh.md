# MiniSGLang 与 SGLang：Qwen3.5-4B 横向复测

日期：2026-09-06。结论：MiniSGLang 原生 MTP-3、DFlash8 在本轮 batch=4 中均快于同框架 stable target；MTP-3 达到已安装 SGLang 0.5.9 MTP-3 吞吐的约 84%–91%，没有全面超过 SGLang。SGLang DFlash 尚未测到，不能给出该组合的速度或精度结论。

## 测试口径

- 同一张 RTX 4090，Qwen3.5-4B BF16、TP=1；同一份 target 权重和 tokenizer。
- 四个实际并行请求，输入 token 长度为 38 / 42 / 38 / 37，直接传相同 `input_ids`。不加 chat template，避免旧 SGLang chat-template 数据与 Mini 原始 prompt 混比。
- 每个请求输出 256 或 512 token；每种 shape 预热两轮、测量五轮，每项 20 请求。SGLang 设置 ignore_eos；Mini 的实际输出均达到指定上限，长度审计通过。属于固定长度吞吐实验，不是自然 EOS 任务延迟评测。
- max context=4096，greedy、seed=42、CUDA Graph 开启，prefix cache 关闭。模式之间独立进程且顺序运行，避免同时占用 GPU。
- SGLang 关闭 overlap，使用默认 BF16 数值路径；Mini 使用 stable 数值路径。两者 attention/linear 算术、图覆盖、KV 池配置与软件版本不完全相同，因此这是配置明确的实际运行对比，不是纯粹的同算子调度器消融。
- 表格统一为 **output tokens / 全部测量 wave 墙钟时间之和**，包含 prefill。SGLang 还包含 `Engine.generate` IPC 开销，Mini 为进程内调用。模型加载与预热不计入稳态，测量中发生的建图或抖动不剔除。
- 每个模式只有四个独立 prompt；五轮重复用来观察吞吐和重复稳定性，不能说成 20 个独立质量题。

SGLang 复用已有隔离环境：Torch 2.9.1、SGLang 0.5.9、sgl-kernel 0.3.21、Transformers 4.57.3、FlashInfer 0.6.3。Mini 主环境仍为 Torch 2.9.1+cu128、FlashInfer 0.6.14。没有升级主环境依赖。

## 性能：五轮聚合，不挑最快结果

| 输出长度/请求 | 框架与模式 | E2E output tok/s | 相对各自 target | draft 接受率 | 五轮范围 |
|---|---|---:|---:|---:|---:|
| 256 | SGLang target | 318.23 | 1.000× | — | 317.11–318.71 |
| 256 | SGLang MTP-3 | 457.16 | 1.437× | 61.39% | 455.82–459.00 |
| 256 | Mini target stable | 317.50 | 1.000× | — | 304.68–328.38 |
| 256 | Mini MTP-3 | 414.66 | 1.306× | 58.98% | 410.35–417.46 |
| 256 | Mini DFlash8 | 336.35 | 1.059× | 27.30% | 306.61–359.79 |
| 512 | SGLang target | 319.57 | 1.000× | — | 318.98–320.11 |
| 512 | SGLang MTP-3 | 472.33 | 1.478× | 62.41% | 464.71–474.46 |
| 512 | Mini target stable | 318.29 | 1.000× | — | 305.99–325.72 |
| 512 | Mini MTP-3 | 396.91 | 1.247× | 61.52% | 358.19–426.79 |
| 512 | Mini DFlash8 | 401.75 | 1.262× | 31.70% | 384.61–410.52 |

Mini MTP/SGLang MTP 比值为 90.70%（256）和 84.03%（512），对应低约 9.30% / 15.97%。SGLang target 与 Mini target 在本轮接近。DFlash 接受率低于 MTP 不等于一定更慢，候选数、每轮进度与全部执行成本共同决定吞吐；本轮 512-token DFlash 的聚合吞吐略高于 Mini MTP。

Mini MTP-512 五轮为 426.79 / 403.90 / 390.24 / 358.19 / 412.55 tok/s，DFlash-256 为 359.79 / 340.65 / 349.32 / 330.41 / 306.61。本轮不是低方差实验，没有运行中完整 CPU/GPU timeline 或频率遥测，不能把抖动武断归因于温度、GPU 降频、系统噪声或某个代码 bug。报告保留原始值；历史 9月5日更高的结果不用于替代本轮横向对照。

当前可证实的结构差距是：SGLang 日志显示捕获 MTP draft CUDA Graph，Mini MTP proposal 自身仍主要 eager；Mini 已覆盖 target verify 与 GDN commit 图。接受率和 target 算术路径也不同。上述因素是后续受控消融方向，不能仅凭总体时间给每个因素分配加速贡献。

## 精度：token 一致性与任务质量分开

| 对照 | 256 token：完整一致请求 | 512 token：完整一致请求 | 说明 |
|---|---:|---:|---|
| Mini MTP-3 vs Mini stable target | 20/20 | 20/20 | 同数值策略 greedy 验收通过 |
| Mini DFlash8 vs Mini stable target | 20/20 | 20/20 | 同数值策略 greedy 验收通过 |
| SGLang MTP-3 vs SGLang target | 5/20 | 0/20 | 分别相当于 1/4、0/4 独立 prompt |
| Mini stable target vs SGLang target | 0/20 | 0/20 | 跨框架算术路径不同，不能直接当质量分数 |

Mini MTP 和 DFlash 各在本轮覆盖 40 请求、15,360 输出 token，与对应 stable target 全部一致。每个框架、每个模式、每个输出长度的五轮输出均可重复：后四轮对首轮的 16/16 请求一致。

首次不同 token 的位置按 **从 1 开始** 计数：

| 对照 | prompt 1 | prompt 2 | prompt 3 | prompt 4 |
|---|---:|---:|---:|---:|
| SGLang MTP vs target，256 | 54 | 8 | 无差异 | 24 |
| SGLang MTP vs target，512 | 54 | 8 | 263 | 24 |
| Mini target vs SGLang target，256/512 | 142 | 52 | 115 | 143 |

这里不能说“SGLang 精度错误”或“Mini 更准确”。BF16 GEMM、attention、GDN reduction 随 shape/实现发生细微变化，可能越过 greedy argmax 边界；不同续写会进一步改变后续接受率。之前已用同历史探针对 Mini 定位并修复这类问题，但本轮没有给 SGLang 每个首次分歧逐层追踪，因此不能认定它们全部是相同原因。

本轮没有 GSM8K/MMLU/HumanEval 等任务评分，也没有 perplexity、随机采样分布检验或人工盲评。能成立的结论是 Mini 的同策略 token 保真性通过；跨框架任务质量优劣未知。

## SGLang DFlash：明确的未测项

已安装 SGLang 0.5.9 的 speculative enum 中没有 DFlash，0.5.9/0.5.10 源码中也未找到 DFlash 文件。磁盘上的 main 源码 `771e613d96de0ee89631bc308a2525aaeae9f13e` 包含 DFlash worker/model，但其 `pyproject.toml` 声明 Torch 2.13.0、Transformers 5.12.1；当前环境是 Torch 2.9.1、Transformers 4.57.3。

实际用独立 `PYTHONPATH` 指向 main 并复用既有 SGLang 环境，`Engine` 导入失败于 `PreTrainedConfig` / `PretrainedConfig` API 差异，原始 traceback 保存在 `sglang-main-import.log`。没有修改主环境或伪造 DFlash 数字。这说明当前环境不兼容，不是 DFlash 算法不可运行。

下一步需要在独立兼容环境运行相同 input IDs 的 SGLang target/MTP/DFlash；必须重新测其 target，不能把新框架 DFlash 与旧 0.5.9 target 直接相除。该项尚未完成，不能把本报告说成所有框架/算法组合已全部验收。

## 复现与证据

云端结果目录：`/root/autodl-tmp/runtime-results/framework-compare-20260906`。本地原始归档为 `framework-compare-20260906.tar.gz`，解包目录为 `framework-compare-20260906/`。汇总 `framework-summary.json` 保存全部统计、首次分歧和输入结果文件 SHA256；所有 prompt、完整回答、token IDs 均在原始 JSON 中。

新工具 `benchmark/runtime/compare_frameworks.py` 审计四请求输入 token、输出长度、重复轮数、缓存关闭、上下文、GPU、Torch 和 native token 一致性。机器字段 `audit_passed=true` 表示输入/统计审计与 native 验收通过，不表示跨框架 token 全同或语言质量评分通过。

本轮 runtime 基于 `47f5147`，测量期间只有 benchmark 工具的未提交变动，native JSON 如实记录 `git_dirty=true`；不是新一次干净 checkout 验收。干净 `20b06d1` 的独立验收是之前单独保留的数据。新增一键 runner 将本轮参数固化，脚本的组合入口通过 shell 语法检查，实际执行证据来自本轮已完成的同参数分段脚本，不声称该组合入口另行完整跑过。

```bash
cd /root/mini-sglang
bash benchmark/runtime/run_framework_comparison.sh \
  /root/autodl-tmp/runtime-results/vllm-compare-W7Nlns/sglang-env/.venv/bin/python \
  /root/autodl-tmp/runtime-results/framework-repro-$(date +%Y%m%d-%H%M%S) \
  /root/miniconda3/bin/python
```

必须使用新输出目录。runner 不安装依赖，先跑 SGLang target/MTP，再跑 Mini target/MTP/DFlash，最后审计汇总；它不包含尚不可用的 SGLang DFlash。所有 JIT 临时文件放可执行的数据盘目录，避开本机 `/dev/shm` 的 `noexec` 限制。
