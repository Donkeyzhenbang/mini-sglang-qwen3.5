# 简历项目经历：推理系统方向

以下是可放入简历的项目模块，不虚构学历、公司、职级或个人任职信息。技术贡献基于既有 Qwen3.5 PR 继续实现，不表述为从零开发完整推理框架。

## 可直接使用的版本

**面向 Qwen3.5 混合架构的推理运行时优化**｜个人系统项目｜2026.08–2026.09

技术栈：Python / PyTorch / Triton / CUDA Graph / FlashInfer / MiniSGLang

- 基于 MiniSGLang Qwen3.5 实现原生 MTP-1/MTP-3 与 DFlash v1，打通 target hidden 提取、批量候选生成、多 token 验证及 KV/conv/SSM 状态提交，支持离线 batch=4 并发与请求槽复用。
- 定位并修复前缀 KV/GDN 长度错位、快照生命周期及 BF16 decode/verify 数值分歧；通过同历史逐层追踪、独立 FP64 oracle 和固定规约算子建立 stable target 基线，最新复测中 MTP-3/DFlash-8 各 60 次请求、17,044 输出 token 与 target 完全一致。
- 设计 GDN verify journal 和 confirmed/scratch KV 分离的 MTP 三步 CUDA Graph；结合跨层状态 gather/scatter 及投影融合，在 RTX 4090、4B BF16、batch=4 的 256/512 固定输出测试中，MTP 相对自身 target 加速约 45%–50%，DFlash 加速约 24%–41%（五轮聚合）。
- 通过 CPU/CUDA profiler 定位 DFlash 非连续 hidden 导致的低效 batched GEMM，将词表投影改为共享二维 GEMM；batch=4 算子探针从约 5.70 ms 降至 1.45 ms，真实 256 输出对照吞吐额外提升约 13%，接受率与最终输出保持不变。
- 构建 KV+GDN GPU/CPU 预算与成本策略、自适应 speculative block 实验及可复现评测工具；完成 96 项 CPU/GPU 回归，并完成早期版本与 SGLang 0.5.9/实验 PR 的对照，明确数值路径与测量边界。

项目仓库：https://github.com/Donkeyzhenbang/mini-sglang-qwen3.5 ，开发分支 `feat/hybrid-memory-runtime`。

## 面试时可展开的三个深入点

**混合状态事务。** Qwen3.5 的 GDN 不能像普通 KV 一样仅截断长度。解释 anchor、已输出但尚未入 cache 的最后一个 token、拒绝后接受前缀提交，以及 journal 为什么比保留每个 token 的完整 FP32 SSM 更节省内存。

**数值一致性诊断。** 讲清如何固定 token 历史和 recurrent state 找首个差异，为什么只改 FP32 LM head 或关闭 split-KV 不够，为什么不同 GEMM 行数可能改变 BF16 greedy 输出。说明 stable 是同一 target 数值策略，不宣称等同所有框架或语言任务正确率 100%。

**CUDA Graph 与真实成本。** 讲清捕获会实际推进状态、live state 地址不可随意迁移、返回 buffer 会被下一次 replay 改写；解释 MTP 三步递归为何只能把 confirmed KV 写入持久池，以及 DFlash 即使开启 graph，非连续布局仍可能选择低效 GEMM。结合 profiler、独立算子探针和完整生成做三层验证。

## 数字使用说明

上述简历采用 2026-09-06 追加优化后干净提交 `76e054e` 的 E2E output throughput：Mini target 328.75/326.07，MTP-3 474.89/487.61，DFlash-8 408.43/461.35 tok/s（256/512）。各模式五轮，包含 prefill；四条独立 prompt 重复测试，不是大规模质量基准。另有 12 次混合长度连续补槽作为正确性回归，未用其单轮吞吐宣传稳态性能。本轮没有重跑 SGLang，不能把早期 84%–91% 的相对吞吐当成优化后的结论。

9月5日受控验收曾测到 MTP 1.335×、DFlash 1.258× 的 decode 加速，9月6日干净脚本复跑在 256 输出下为 1.302× / 1.123×。可在面试补充说明，不能混用不同轮次/口径只挑最高值。

收尾新增案例：在隔离的 SGLang DFlash PR 中，通过配置探针定位 RoPE theta 被旧加载器从 1e7 误读为默认 1e4；补齐分层 attention 语义并做 RoPE-only 消融，组合修正将接受率从约 14%/17% 提升到 27%/30%，吞吐相对原始 PR draft 提升约 27%/25%。该 PR 默认 BF16 未通过对 target 的严格 token 一致性，不能把这项写成无损加速成果。

不要写：完整 HiCache 移植、27B Int4 在 24GB 部署、DFlash2、正式新版 SGLang DFlash 已完成验收、全场景优于 SGLang、随机采样严格无损、HTTP 服务生产验收、通用编译 pass manager，或没有实际证据的 Nsight/C++ CUDA kernel 成果。MTP 三步 draft graph 现已落地，可以写，但须保留不支持形状回退 eager 的边界。
