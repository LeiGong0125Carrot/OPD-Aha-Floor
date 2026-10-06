# ZoomBench 接入与"压缩组件"方向（2026-10-06 上午，滚动更新）

## 1. 为什么换基准
- TreeBench 天花板已到：9B + GT crop 也只有约 53，4B 的 A 在 51；剩余题目多为感知蒸馏无法解决的推理题（Perspective 类所有臂 14%）。
- V\* 贴顶：A 的 94.76 = 181/191。
- 同配置重跑噪声：A 四次 TB 49.9–51.6、V\* 93.2–94.8；405 / 191 题一题值 0.25 / 0.52 分。本季 25 个臂没有一个在噪声带外超过 A。
- **决定（用户 10-06）**："超过 A"不再是目标；新目标是**压缩 OPD-Aha 的组件，在近似效果下做减法**。判读基准改为大基准：ZoomBench（845 题）为主、HR-Bench 4K/8K（各 800）为辅，逐题池化 + 配对 bootstrap；TB/V\* 只作参照。"近似"= 配对 95% 区间含 0 且点估计差 ≤ 1 点。

## 2. ZoomBench 接入
- 数据：`inclusionAI/ZoomBench`，845 题（621 道 MCQ + 224 道开放题），每题带 GT 框与 crop 图；OPD-Aha 原代码自带 `eval/prepare_data.py --benchmark zoombench` 与 `eval/run_eval.sh BENCHMARK=zoombench`（原 parquet 4 GB 含图，需在计算节点解包）。
- 评测：`Vision-OPD-setup/eval_zoom.sh`（独立于 TB/V\* 流水线；模型清单驱动、幂等），vLLM 服务 + 原 `infer.py`，judge = `judge_qwenlm.py` rule-first 协议 + gpt-oss-120b（与我们的 V\* 口径一致；ZoomBench 不在 MCQ 列表，开放题与规则判错的 MCQ 都送 LLM）。
- 只能评各臂保留的峰值 ckpt（TB 峰 / V\* 峰）；这是"按别的基准选 ckpt"的口径，判读时注明。卡型：RTX Pro 6000（B200 与环境不兼容）。

## 3. 第一批结果（10-06 09:30，judge 口径；judge 未完成的模型给 MCQ 规则下界）

| 模型（峰值 ckpt） | ZoomBench judge（845） | MCQ 规则下界（621） |
|---|---|---|
| base Qwen3.5-4B | **49.59** | 53.95 |
| A ahaAn2-50 / rr1-30 / rr2-30 | **60.00 / 58.70 / 60.24** | 64.57 / 64.41 / 65.06 |
| Ahf-50 / Ahfr2-50 | **60.12** / 待 | 65.38 / 64.90 |
| Ahm-40 / Ahm-50 / Ahmr2-50 | 待 | 64.25 / 66.34 / 63.61 |
| supn2-30 | 待 | 60.39 |
| Amis-50 / Amisr2-30 / Amisr2-50 | 待 | 62.64 / 60.71 / 65.54 |

读法（规则下界与 judge 一致的部分）：
- A 相对 base **+10 点**（TB 上只有 +5）：ZoomBench 正是 crop 蒸馏该起作用的题。
- A 三次相差 1.5 点（judge）/ 0.65（MCQ 规则），噪声带比 TB 小。
- Ahf ≈ A；sup-only 的 MCQ 下界低 4 点（V\* 上曾与 A 同分，这里能分开）；Amis 的 TB/V\* 峰 ckpt 低 2–4 点，但 r2 的 step50 反而 65.5——ckpt 选取口径的问题，等全步补评。

## 4. 压缩方向的梯子（哪些组件能去）

| 组件 | 能否去掉 | 证据 / 检验 |
|---|---|---|
| null 前向 | **不能** | S / S2 / IR / 层探针 / 镜像探针 / Amis 全部判负（`directions_summary_2026-10-05.md`）；只能工程压缩（前缀共享，全图编码一次，约 1.1× 前向） |
| 翻转集外的倾斜（u 的 88% 偏移） | **训练中** | CFD-u：与 A 唯一差别就是这 88%；14:30 出分 |
| 翻转位的 u 数值 vs 硬标签 | 训练中 | CFD（γ=50 硬化 + 2 倍权重）vs CFD-u |
| 正半边 | 可能可去 | sup-only：V\* 终点同 A，ZoomBench MCQ 下界低 4 → 等 judge |
| 软分布 | 基本可去 | γ=50 硬标签：TB −1.4、V\* 无损 |
| 首 token 倾斜（A-t0） | 诊断项 | 只占 A 信号 1.5%，不作贡献 |
| top-K 大小、JSD α | **不做** | 超参消融，无方法论含义（用户 10-06） |

若 CFD-u ≈ A，再叠 sup（翻转位、只压不提）得到最小形式：**目标 = p⁺，只在反事实翻转 argmax 的位置压掉 p⁰ 偏好的 token**——无 β、无软倾斜、无正半边。

## 5. CFD 的翻转判定与权重（实现口径，sup `1b05979`）
- 位置 t ≥ 1，学生前缀 y_{<t} 下，教师两个分布压缩到学生 top-100 + tail：p⁺（[全图, crop]）、p⁰（[全图, 均值色块]）。a⁺ = argmax p⁺，a⁰ = argmax p⁰。**翻转 ⟺ a⁺ ≠ a⁰**，且 a⁺ 不在 tail 桶（不可判定，0.3%）；t=0 不算（无学生前缀）；padding 按 loss_mask 排除。不用 u 的数值，只用其符号在 argmax 上的体现。
- 所有位置都算损失：非翻转位目标 = p⁺（Vision-OPD），权重 1；翻转位目标 = A 的 softmax(log p⁺ + 4u)（CFD-u，权重 1）或 γ=50 锐化的 p⁺（CFD，权重 1+λ=2）。损失 = Σ w·JSD / Σ w，分母也加权，λ 只在序列内重新分配。
- 实测：翻转率 6.1–6.4%，前重后轻；翻转位学生与教师 argmax 一致率从 0.2 升到 0.45；常见翻转 appears→is、on→in、is→has、visible→in（"看清后更肯定、更精确"），颜色/方位词只占翻转后 a⁺ 的 7%。

## 6. 下一步
1. ZoomBench judge 补齐（Ahm ×3、sup、Amis ×3）；HR-Bench 4K/8K 加进同一批评测脚本。
2. CFD-u / CFD（r1 全步评测 14:30 出；r2 等新 hold）按 06 §4 判读树；通过则 CFD-u+sup 一臂 ×2。
3. 全部臂的峰值 ckpt 在三个大基准上重排，形成"同效果、少组件"的总表。
