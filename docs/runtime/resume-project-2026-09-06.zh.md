# 简历项目经历：推理系统方向

以下是可放入简历的项目模块，不虚构学历、公司、职级或个人任职信息。技术贡献基于既有 Qwen3.5 PR 继续实现，不表述为从零开发完整推理框架。

## 可直接使用的版本

**面向 Qwen3.5 混合架构的推理运行时优化**｜个人系统项目｜2026.08–2026.09

技术栈：Python / PyTorch / Triton / CUDA Graph / FlashInfer / MiniSGLang

- 基于 MiniSGLang Qwen3.5 实现原生 MTP-1/MTP-3 与 DFlash v1，打通 target hidden 提取、批量候选生成、多 token 验证及 KV/conv/SSM 状态提交，支持离线 batch=4 并发与请求槽复用。
- 定位并修复前缀 KV/GDN 长度错位、快照生命周期及 BF16 decode/verify 数值分歧；通过同历史逐层追踪、独立 FP64 oracle 和固定规约算子建立 stable target 基线，最新复测中 MTP/DFlash 各 40 请求、15,360 输出 token 与 target 完全一致。
- 设计 GDN verify journal，仅重放接受前缀的卷积与递推，替代完整 target 重算；结合跨层状态 gather/scatter、投影融合及 verify/draft/state-replay CUDA Graph，单卡 RTX 4090、4B BF16、batch=4 实测 MTP 相对自身 target 提升约 25%–31%，DFlash 提升约 6%–26%（256/512 固定输出、五轮聚合）。
- 构建 KV+GDN 前缀包的 GPU/CPU 预算与成本策略、自适应 speculative block 实验框架及可复现评测工具；完成 94 项 CPU/GPU 回归，并与 SGLang 0.5.9 对照，MTP 达到其约 84%–91% 吞吐，明确数值路径与测量波动边界。

项目仓库：https://github.com/Donkeyzhenbang/mini-sglang-qwen3.5 ，开发分支 `feat/hybrid-memory-runtime`。

## 面试时可展开的三个深入点

**混合状态事务。** Qwen3.5 的 GDN 不能像普通 KV 一样仅截断长度。解释 anchor、已输出但尚未入 cache 的最后一个 token、拒绝后接受前缀提交，以及 journal 为什么比保留每个 token 的完整 FP32 SSM 更节省内存。

**数值一致性诊断。** 讲清如何固定 token 历史和 recurrent state 找首个差异，为什么只改 FP32 LM head 或关闭 split-KV 不够，为什么不同 GEMM 行数可能改变 BF16 greedy 输出。说明 stable 是同一 target 数值策略，不宣称等同所有框架或语言任务正确率 100%。

**CUDA Graph 与真实成本。** 讲清捕获会实际推进状态、live state 地址不可随意迁移、返回 buffer 会被下一次 replay 改写，以及 adaptive 如何将 startup 与 steady-state 分开学习但不从用户耗时中删除 startup。

## 数字使用说明

上述简历采用最新 2026-09-06 横向测试的 E2E output throughput：Mini target 317.50/318.29，MTP 414.66/396.91，DFlash 336.35/401.75 tok/s（256/512）。各模式五轮，包含 prefill；四条独立 prompt 重复测试，不是大规模质量基准。MTP-512 与 DFlash-256 波动较大，范围和原始数据见对比报告。

9月5日受控验收曾测到 MTP 1.335×、DFlash 1.258× 的 decode 加速，9月6日干净脚本复跑在 256 输出下为 1.302× / 1.123×。可在面试补充说明，不能混用不同轮次/口径只挑最高值。

不要写：完整 HiCache 移植、27B Int4 在 24GB 部署、DFlash2、SGLang DFlash 性能已测、全场景优于 SGLang、随机采样严格无损、HTTP 服务生产验收、通用编译 pass manager、已实现完整 MTP draft graph，或没有实际证据的 Nsight/C++ CUDA kernel 成果。
