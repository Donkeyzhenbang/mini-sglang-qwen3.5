# SGLang DFlash 补充对照与项目收尾

2026-09-06。本文补充前一版报告中未测的 SGLang DFlash 路径。实际运行对象是**已关闭、未合并的 PR #19952**，固定源码 `5106aa523aee7a77471fdb50e0cb1e8b21da9427`，不是 SGLang 当前正式发布版。正式新版因依赖跨度较大仍未完成测试，不混用二者结论。[上游 PR](https://github.com/sgl-project/sglang/pull/19952)

## 最终状态

- MiniSGLang 原生 MTP-1/MTP-3、DFlash v1、stable target、批量 verify、GDN journal、状态拷贝和多条 CUDA Graph 路径已经实现，并在既定单卡、BF16、greedy、离线 batch=4 范围内通过验收。
- 已完成 SGLang 0.5.9 target/MTP 对照，以及本次 SGLang PR target/MTP/DFlash 对照；保留输入、完整输出、接受率、五轮时间和失败日志。
- 已定位并修正隔离 PR 源码对当前 draft checkpoint 的 RoPE 与分层 attention 配置读取问题，完成单项消融。没有修改 MiniSGLang target、SGLang target 权重或接受/拒绝规则。
- 收尾时恢复隔离 PR 源码为完整配置兼容版本，保留原始文件备份和补丁 SHA256；实验 GPU 进程已结束。不再扩展新优化，不改主环境、不关闭云服务器。

## 环境与启动问题

PR 的依赖声明使用 Torch 2.9.1、sgl-kernel 0.3.21，与现有环境接近。新建独立 `--system-site-packages` 环境，复用原 SGLang 环境的其他依赖，仅从清华源安装 FlashInfer Python 0.6.4 与 cubin 0.6.4（下载约 242 MiB）。旧 0.5.9 环境和 MiniSGLang 主环境均保留。

最初复用 FlashInfer 0.6.3 被启动版本检查拒绝；通过安装声明的兼容版本解决，没有删除或绕过检查。随后 PR target 的 piecewise warmup 缺少 MRoPE positions，出现 `NoneType.ndim` 异常。基准显式关闭该未验证的 piecewise 路径，保留完整 decode/verify/draft CUDA Graph；所有模式使用同样设置。该处理是明确的运行范围选择，不等同于修复了通用 piecewise MRoPE。

基准保持同四条原始输入 token、batch=4、256/512 输出、warmup=2、repeat=5、context=4096、greedy、无 prefix cache、关闭 overlap。吞吐为总输出 token / 五轮墙钟时间总和，包含 prefill 和 Engine IPC，不含模型加载与预热。所有模式实际输出均达到指定长度。

## 发现的 draft 契约问题

**RoPE 配置键不兼容。** checkpoint 将 theta=10,000,000 写在 `rope_parameters` 中，旧 Qwen3Config 的 `rope_theta` 默认值却为 10,000。实际配置探针确认两字段同时存在，原 DFlash 模型只读后者。补丁优先读取 checkpoint 的新字段，保留旧格式兼容；对本次未验证的非 default 新 RoPE 类型明确拒绝。

**各层 mask 被统一为非因果。** 当前 checkpoint 前五层为 sliding attention，最后一层 full attention；审阅的参考实现默认前五层 causal、最后一层 non-causal。PR 原实现将六层全部设为 `ENCODER_ONLY`。补丁按层设置 causal/full，并将参考条件 `query_position - key_position < 4096` 映射为 FlashInfer `window_left=4095`。

补丁仅支持这次验证的 causal sliding/default-RoPE 配置，使用文件 SHA256 保护目标版本，保留备份，重复应用无修改。CPU 检查覆盖真实配置、旧字段 fallback、窗口边界可见集合和重入校验。GPU 已测上下文小于滑窗长度；**超过 4096 的真实滑窗边界尚未做 GPU 验收**，不能扩大为所有上下文支持。

## 五轮聚合结果与单项消融

| 模式 | 256 output tok/s | 相对 PR target | 256 接受率 | 512 output tok/s | 相对 PR target | 512 接受率 |
|---|---:|---:|---:|---:|---:|---:|
| PR target | 319.25 | 1.000× | — | 320.39 | 1.000× | — |
| PR MTP-3 | 406.54 | 1.273× | 55.79% | 448.94 | 1.401× | 59.85% |
| PR DFlash8 原始 | 319.51 | 1.001× | 14.17% | 353.44 | 1.103× | 17.08% |
| PR DFlash8 只修 RoPE | 390.14 | 1.222× | 26.19% | 458.07 | 1.430× | 30.71% |
| PR DFlash8 RoPE + 分层 mask | 405.01 | 1.269× | 27.11% | 441.77 | 1.379× | 30.36% |

完整兼容修正相对原始 PR DFlash 的吞吐约提升 26.8%（256）和 25.0%（512）。只修 RoPE 已恢复大部分接受率，说明本次较大的损失确实与配置解析有关，不能全部归因于 GPU 执行效率。

分层 mask 修正并不在所有长度都进一步提升吞吐：512-token 的 RoPE-only 更快。恢复模型配置契约与选择最快候选行为是不同问题；没有为了取最高速度保留错误配置。该实验是 RoPE-only 与组合修正的顺序消融，没有 mask-only，也不是完整二因子实验，不能把两个因素贡献视为相互独立。

## 精度边界

每个 PR 模式在同 shape 的五轮输出均重复一致。但 PR MTP、原始 DFlash、RoPE-only 与完整兼容 DFlash，各自对 PR target 的完整请求匹配均为 0/20（两个长度均如此）。原始 DFlash 与完整兼容 DFlash 之间也仅 5/20 完整相同。

因此这些数字用于**默认 BF16、不同执行形状的性能与接受率诊断**，不能宣称 SGLang PR 已通过严格 token 无损验收。此前 MiniSGLang 的同历史诊断证明过 BF16 shape/reduction 可以改变 greedy 结果，但本轮没有逐层定位 PR 的每次分歧，不能直接断言它们全是浮点差异而没有状态问题。没有做任务质量评分，不能据此评价两框架谁的回答更准确。

与之区分，MiniSGLang 最新 stable 复测中，MTP/DFlash 各 40 请求、15,360 token 与同策略 target 全部一致。此前的 94 项 CPU/GPU 回归仍是 native runtime 的回归证据；本轮只改 benchmark/兼容补丁工具，没有把这 94 项说成新增 PR GPU 回归。

Mini 的当前 MTP 吞吐约为正式旧版 SGLang 0.5.9 的 84%–91%；这项结论保留。不能改用这个 PR 的较低 MTP 数字，宣称全面超过 SGLang。不同版本的 target、投机、数值路径必须各自成组比较。

## 交付与可复现性

本地补充证据归档：`sglang-dflash-20260906-evidence.tar.gz`，不包含模型权重、虚拟环境或凭据。云端原始目录：`/root/autodl-tmp/runtime-results/sglang-dflash-20260906`。

工具：`bench_sglang_mtp.py` 新增 DFlash 模式及实际导入源码/revision 记录；`patch_sglang_pr19952_dflash.py` 为显式、版本受保护的兼容补丁；`summarize_sglang_dflash.py` 核对输入、模式设置、版本、缓存、长度和五轮数据。原始、RoPE-only、完整兼容三份 DFlash 源码及执行脚本保存在证据中。

在现有云端环境重跑时，先确认选定源码变体，始终使用新的输出文件：

```bash
cd /root/mini-sglang
BASE=/root/autodl-tmp/runtime-results/sglang-dflash-20260906
export PYTHONPATH=$BASE/sglang-5106aa523aee7a77471fdb50e0cb1e8b21da9427/python
export OMP_NUM_THREADS=4
export TMPDIR=/root/autodl-tmp/runtime-results/vllm-compare-W7Nlns/runtime-tmp

$BASE/env/bin/python benchmark/runtime/bench_sglang_mtp.py \
  --model /root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B \
  --draft /root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B-DFlash \
  --mode dflash --block-size 8 \
  --source-revision 5106aa523aee7a77471fdb50e0cb1e8b21da9427+draft-a23935c8 \
  --workload /root/autodl-tmp/runtime-results/framework-compare-20260906/inputs-256.jsonl \
  --batches 4 --lengths 256 512 --warmup 2 --repeats 5 --context-length 4096 \
  --output "$BASE/repro-dflash-$(date +%Y%m%d-%H%M%S).json"
```

当前收尾不继续做：正式新版 SGLang DFlash 环境、PR 严格 token 分歧定位、MTP draft graph、HTTP/overlap 投机接入、随机采样质量、27B Int4、DFlash2 或更大上下文验收。这些是后续任务，不混入本次完成项。
