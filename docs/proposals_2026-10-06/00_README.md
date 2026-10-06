# 三条候选方法（2026-10-06，v0.1，待讨论）

> 来源：10-05 晚讨论（用户 + 助手），在 Amis 终审判负、null-free 线、IR 线、层探针、镜像探针全部收线之后形成。
> 目标口径沿用 `negative_history/summary_history_future_on_full_aha_2026-10-03.md` §0：在 OPD-Aha 基础上提出**不同的方法论**，要么性能超过 A，要么**减少组件、与 A 持平**。不接受纯工程项（前缀共享、vLLM 教师等）作为贡献。
> 固定配置与判读口径不变：2459 题 6karmA pair、seed 42、batch 48、n=2、lr 2e-6、51 步、3 卡/SP1、JSD α=0.5、学生 top-100 + tail、冻结教师；gpt-oss-120b judge；TB（TreeVGR 官方 HF greedy）+ V\*（`eval/infer.py`）。
> 参照：A（4 次）TB 峰 49.88 / 51.11 / 51.11 / 51.60，V\* 峰 94.76 / 93.19 / 94.24 / 91.62，池化 49.22 / 92.84；V0 TB 49.38 / 48.64，V\* 87.4–88.5；同配置重跑峰差 1.23。

## 0. 本季已经排除的自由度（这三条方法的边界）

| 自由度 | 已试臂 | 结论 | 出处 |
|---|---|---|---|
| 目标形状（exp 倾斜的函数形式、β 的剂量/调度/分半边） | floor / tanh / γ / sup / C / D / supb63 / Ahm / Ahf / X1 / floor+pair，约 15 臂 | 无一两次峰值超过 51.60；任何让目标更尖的改动都伤 OCR 与关系推理，无论动哪一半 | `results_comparison_2026-10-02.md`、`summary_history_future_on_full_aha_2026-10-03.md` §3 |
| 去掉 null 前向（内部读出 / 学生参照 / 学生底座） | S / St / S2 / IRf / IRfl2 / IRfl4 / 层探针 / 镜像探针 | $p^0$ 只存在于"同模型、同前缀、换输入"的那次计算里；线性读出 R² 0.008；学生参照回到 V0，学生底座塌缩 | `null_free_line_summary_2026-10-04.md`、`ir_results_2026-10-05.md`、`full_to_priv_probe_2026-10-05.md` |
| 跨样本统计量替代 null（donor crop / token 基线） | Amis r1/r2、$b(v)$ 预检 | $u$ 题目特异（held-out 基线解释 −6% 能量）；donor crop 是假证据，目标 TV 不降反升，TB 49.14 / 47.90 | `amis_results_2026-10-05.md`、`opd_aha_mechanism_conclusion_2026-10-05.md` §5 |
| 注意力引导裁剪 | — | 用户判定：OPRD 等已做；vLLM 不暴露注意力 | 10-05 讨论 |

本季最硬的两条正面证据，也是三条方法的共同依据：

1. **增益主体是 argmax**：γ=50 硬标签均值 49.55 vs A 50.93，V\* 无损——软分布只贡献约 1.4 点。
2. **A 学的是"压制没看清关键区域时的冲动"**，$u$ 的质量集中在推理正文的感知描述 token，答案位几乎为零；同一个 β 同时放大与证据无关的风格差异，这是格式漂移与 OCR 丢分的来源（机制结论 §3 限定 2）。

## 1. 三条候选一览

| # | 名称 | 针对的痛点 | 去掉的组件 | 新增前向 | 先验 | 文档 |
|---|---|---|---|---|---|---|
| P1 | 反事实翻转蒸馏（Counterfactual-Flip Distillation, CFD） | 目标重构组件堆叠；增益可能来自分布整形（OPSA / Mirage 类质疑）；对比信号里的风格泄漏 | β、指数倾斜、tail 病理；预算分配依据改为"反事实是否翻转决策" | 0（仍需 null） | 中高 | `01_P1_counterfactual_flip.md` |
| P2 | 组内相对的特权似然加权（Group-Relative Privileged Likelihood, GRPL） | OPD 缺轨迹级判别（V-Zero 诊断）；null 覆盖不了"看对了地方但推理走偏" | **null 前向、人工 null 视图** | 0（减一次） | 中，由离线预检决定 | `02_P2_group_relative_weighting.md` |
| P3 | 可实现性门控的非对称倾斜（Realizability-Gated Asymmetric Tilt, RGT） | 教师修正学生看不见（FP-OPD）；探针"看了但没看清" | 无 | 0 | 低（目标形状家族） | `03_P3_realizability_gated_tilt.md` |

## 2. 推荐执行顺序

1. **P1 的离线预检**（不训练，用 `probe_layer_prior/full_1005_1234/` 的 400 题 dump）：翻转集占比、翻转位置的类型分布、翻转位置之外 A 的倾斜带来的 TV 份额。一张卡 <1 小时。
2. **P2 的离线预检**（`group_relative_teacher_signal_plan_2026-10-05.md` §3 已写好）：G1–G4，一张卡 3–4 小时。
3. P1 训练 ×2（loss 层改动，与 A 同成本）；P2 若预检过门槛再训练 ×2（比 A 省一次前向）。
4. P3 只在 P1 结果出来后决定是否做。

## 3. 两条与方法无关、但决定判读的前置项

- **评测协议**：峰值 + ×2 同侧规则对噪声友好（峰值有向上偏置，噪声带 1.23 与多数臂差距同量级）。建议所有新臂同时报告逐题池化 + 配对 bootstrap（A 与 X 同题对错差的 95% CI），并把**格式合规率**（缺 `<answer>` 数、裸选项数）列为必报指标。
- **top-k 索引来源**：`dp_actor.py` 中 real / null 两次教师前向都用学生 top-100 索引。后段严重偏离时正确 token 可能只在 tail 桶，$u$ 对它无方向。P1 的翻转判定在 top-100 + tail 上做，若 argmax 落在 tail 桶则该位置视为"不可判定"，不进翻转集（见 P1 §4.3）。
