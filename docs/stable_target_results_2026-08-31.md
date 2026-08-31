# Qwen3.5 / DFlash 长回答一致性修复验证（2026-08-31）

代码版本：`c5eb97b43b4cb1a701d78dba31012bd0e3ea732a`；所有生成测量均在干净工作区运行。
GPU：NVIDIA GeForce RTX 4090；PyTorch 2.9.1+cu128；BF16 Qwen3.5-4B + 原生 DFlash v1。

## 结果

15 组生成运行、共 80 条请求输出完成严格 token 对照；每组均以相同 stable 数值模式的 target-only 为基线。
覆盖 256/512-token 上限、batch 1/4/8、block 8/16、parallel/sequential、eager/CUDA Graph、连续补请求与 CPU 前缀缓存恢复。
输出可能提前遇到 EOS；实际长度见下表。

- 74 项 CPU/GPU 测试通过，包含独立 FP64 算子对照与 batch 排列、分块不变性测试。
- 真实权重：同历史下 logits、DFlash 特征、KV、GDN 状态逐位一致；拒绝后的回滚重放一致。
- CUDA Graph：更新 slot 和 context 后与 eager 一致；无关请求状态不变，输出不被后续 replay 覆盖。
- 完整前缀的 GPU/CPU 恢复通过；接受率与前缀缓存命中率分别统计。

## 单次覆盖测量

以下数据说明测试覆盖与当前开销，不作为稳定加速比。不同 batch、调度配置之间不能直接归因于 DFlash。

| 运行 | 各请求实际输出长度 | Decode token/s | 接受率 | 峰值 allocated GiB |
|---|---|---:|---:|---:|
| long256-target | 256,256,256,256 | 328.98 | — | 8.32 |
| long256-fixed | 256,256,256,256 | 162.22 | 24.48% | 9.75 |
| long256-adaptive | 256,256,256,256 | 180.91 | 45.11% | 9.74 |
| long256-sequential | 256,256,256,256 | 57.85 | 24.48% | 9.76 |
| long256-eager | 256,256,256,256 | 152.66 | 24.48% | 9.75 |
| long256-batch1 | 256,256,256,256 | 50.71 | 24.38% | 9.33 |
| long256-block16 | 256,256,256,256 | 146.39 | 12.46% | 9.78 |
| long512-target | 512,512,512,512 | 328.04 | — | 8.32 |
| long512-fixed | 512,512,512,512 | 165.69 | 27.46% | 9.77 |
| long512-adaptive | 512,512,512,512 | 183.00 | 49.11% | 9.80 |
| chat8-target | 63,4,96,19,63,4,96,19 | 158.57 | — | 8.31 |
| chat8-batch8-target | 63,4,96,19,63,4,96,19 | 308.82 | — | 8.64 |
| chat8-batch8-fixed | 63,4,96,19,63,4,96,19 | 195.76 | 28.13% | 10.28 |
| chat8-continuous | 63,4,96,19,63,4,96,19 | 106.59 | 28.13% | 9.74 |
| chat8-host-cache | 63,4,96,19,63,4,96,19 | 166.90 | 28.13% | 9.79 |

## 根因与修复

此前同一历史下，第一层 GDN 的线性投影就会随 batch/verify 形状改变 BF16 结果；固定线性规约后，
第一个 Full Attention 层仍有 decode/verify 差异。仅禁用 split-KV 或仅改 LM head 为 FP32 均不足以解决。

修复采用固定矩阵乘法 tile、分段点积后的显式 FP32 累加、按每个 query 的因果终点做固定顺序 attention 规约，
以及 FP32 target logits。多个请求和 verify token 仍并行，draft 算子和全局 PyTorch 函数未被替换。
独立 K=9216 探针：最大误差 0.006866455 → 0.000183105；未放宽测试容差。

## 当前性能边界

这次修复解决长输出一致性。当前这组 batch 4 长回答中，DFlash 仍慢于 target-only，不能宣称已获得投机加速。

| long256 模式 | draft ms | verify ms | restore/replay ms |
|---|---:|---:|---:|
| long256-target | 0.00 | 2964.89 | 0.00 |
| long256-fixed | 1732.26 | 2199.22 | 2278.01 |
| long256-adaptive | 1096.77 | 3375.07 | 1031.77 |

以上阶段耗时按共享 batch 成本分摊后汇总，不是每条请求独占的 GPU 时间。
后续性能优化的实测重点是 GDN 拒绝后的恢复/重放和 draft 开销。

## 复现与边界

见 [完整复现命令](stable_target_numerics.md) 与 [机器可读结果](stable_target_results_2026-08-31.json)。
实验入口默认 `--target-numerics stable`；复现历史实现需要显式 `--target-numerics fast`。

**stable 是新的、统一的 target 数值基线，不保证与历史 fast BF16 或 Hugging Face 输出相同，
也不承诺跨 GPU、编译器逐位复现。** 本轮没有实现 INT4、DFlash2、HTTP 调度器集成或 24GB 硬内存上限。

云端原始 JSON、日志、诊断脚本：`/root/autodl-tmp/runtime-results/parity-fix-s35SwV`。
