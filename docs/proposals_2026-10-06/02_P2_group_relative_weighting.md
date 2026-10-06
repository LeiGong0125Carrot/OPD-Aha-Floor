# P2：组内相对的特权似然加权（Group-Relative Privileged Likelihood, GRPL）（2026-10-06，v0.1）

> 一句话：不做 null 前向。用同一 prompt 的 n 条 sibling rollout 在**特权教师（pair 视图）**下的序列似然做组内归一化，得到每条 rollout 的相对可信度，以此加权 Vision-OPD 的蒸馏损失。两次前向（学生 + 教师 real），零人工视图，所有输入在分布内。
> 这是 OPD-Aha 作者 10-05 建议（"OPD 都是 batch 训练，类似 GRPO，可以从本 batch 的别的样本拿信号"）的另一种读法，与 Amis（拿别的样本的**图**）互补：Amis 已判负，本线拿的是别的 rollout 的**教师评分**。
> 预检设计已在 `negative_history/group_relative_teacher_signal_plan_2026-10-05.md` §3，本文补齐训练目标、落点与判读，不重复预检细节。

## 1. 痛点

- **OPD 缺轨迹级判别**（V-Zero 的诊断）：token 级对齐不知道一条回答作为整体是否在偏离；学生进入错误路径后，教师只能给"局部合理"的续写修正。
- **null 对比覆盖不了"看对了地方但推理走偏"**：这类位置 $u\approx0$（机制结论 §1：$u$ 标的是"哪些 token 依赖证据"），而 TB 的关系推理类（Ordering / Comparison / Spatial Containment）恰是 sup 家族与 Ahm 丢分最多的地方。
- **null 视图是分布外输入**（均值色块 + "Zoomed-in view" 措辞），已观察到措辞经 $u$ 泄漏（机制结论 §5）。

## 2. 为什么它不受"null 不可替代"结论的约束

本季关闭去 null 线的依据是：$p^0$ 的信息只在"同模型同前缀换输入"的计算里。GRPL **不试图近似 $p^0$**，它提供的是 null 没有的另一种信息（轨迹级相对可信度）。所以它的成败取决于一个独立的问题：**特权教师在序列级有没有足够的分辨力**。探针线历史给出 $R_C\approx0.62$（教师信号区分对错轨迹），这是本线最大的风险，也是预检 G2 要直接测的量。

## 3. 方法

同一 prompt 的 $n$ 条 rollout $y^{(1..n)}$，教师 pair 视图 $p^+$ 已对每条算过（Vision-OPD 本来就要算）。

**序列分数**（两种，预检决定取哪种）
$$s_k=\frac{1}{|T_k|}\sum_{t\in T_k}\log p^+_t\big(y^{(k)}_t\big),\qquad T_k=\text{全部位置（S1）或前 30\% 位置（S2，}u\text{ 质量集中处）}$$

**组内相对量**
$$A_k=\frac{s_k-\mathrm{mean}_j s_j}{\mathrm{std}_j s_j+\epsilon}\quad(n\ge3)\qquad\text{或}\qquad A_k=s_k-\mathrm{mean}_j s_j\quad(n=2)$$

**权重与损失（T1，主臂）**
$$w_k=n\cdot\mathrm{softmax}_j(A_j/\tau)_k,\qquad \mathcal L=\frac{\sum_k w_k\sum_t \mathrm{JSD}_\alpha(p^+_{k,t}\,\|\,p^S_{k,t})}{\sum_k w_k\,|T_k|}$$

$\tau$ 控制组内权重的集中度：$\tau\to\infty$ 退化为 Vision-OPD（V0），$\tau\to0$ 变成"只学组内最可信的一条"。默认 $\tau=1$，扫 $\{0.5,1,2\}$。$w_k$ 的均值恒为 1，总剂量与 V0 相同——这是刻意的：本季证据表明调总剂量换不来 TB，GRPL 只改分配。

**不做的变体**：T2（$A_k<0$ 的 rollout 做反向压制）。EVT-neg 历史 0 胜 4 负、S2 塌缩，advantage 形式的负权重在本设定下有系统性失败史。

### 3.1 可选：与 P1 叠加

GRPL 决定"学哪条 rollout"（轨迹级），P1 决定"学哪些位置"（token 级），两者正交，可以在 P1 的目标上乘 $w_k$。但叠加后需要 null 前向，失去 P2 的成本优势；只在两者各自成立后再做。

## 4. 实现落点

- **数据流**：`ray_trainer.py` 已把同 prompt 的 rollout 用 `uid` 标记（Amis 的 `pick_donor_indices` 按 `uid` 取过）。`s_k` 需要教师 realized-token log-prob：`dp_actor.py` 的 `teacher_outputs["log_probs"]`（real 视图）已传入 loss（`compute_self_distillation_loss(teacher_log_probs=...)`）。
- **组内归一化的边界**：verl 的 micro-batch / 动态 batch 切分可能把同一 `uid` 的 rollout 分到不同 micro-batch。两种处理：(a) 在 `ray_trainer` 的 batch 级先算 $s_k$ 与 $w_k$，作为 `tensor_batch["grpl_weight"]`（标量/样本）传下去，loss 层只乘权重——推荐；(b) loss 层按 `uid` 分组，要求同组在同一 micro-batch，需改 batch 切分。
- **配置**：`self_distillation.grpl_enable`、`grpl_score: all|front30`、`grpl_tau`、`grpl_norm: z|center`。守卫：`grpl_enable` 要求 `counterfactual_null_mode=None` 且 `rollout.n>=2`；同组样本数 <2 时该组权重置 1。
- **指标**：`grpl/w_min`、`w_max`、`w_std`（组内权重分布）；`grpl/score_spread`（组内 $s$ 极差）；`grpl/frac_groups_degenerate`（极差 < ε 的组比例，n=2 时预计不小）。
- **测试**：`grpl_tau→∞` 时与 V0 bit-identical；权重均值恒为 1；同 uid 跨 micro-batch 时权重与单 micro-batch 一致（采用 (a) 后自动成立）；padding 不进 $s_k$。

## 5. 离线预检（go/no-go，`group_relative_teacher_signal_plan_2026-10-05.md` §3）

| 门槛 | 含义 |
|---|---|
| G1 ≥ 1/3（n=8）含对错混合的组 | 否则组内信号稀疏 |
| G2：S1 或 S2 的组内配对 AUC ≥ 0.70 | 教师序列级分辨力足够 |
| G2 全部 < 0.65 | **本线关闭**，与 $R_C$ 0.62 一致 |

若 G2 只在 S2（前 30%）上过门槛，主臂用 S2；若 n=2 的 G1 太低，训练需改 n=4（lr 随有效 batch 缩放，见 `progress_report` §1.2 第 4 条：n8+lr8e-6 ≈ n2+lr2e-6）。

## 6. 预测与 kill 条件

- **成功判据**：TB 与 V\* 与 A 持平（×2，池化 CI 覆盖 A），且比 A 少一次前向、无 null 视图。这是"减少组件、效果相近"口径下的成功。
- **分类别预测**：相对 V0，关系推理类（Ordering / Comparison / Spatial Containment）上升；相对 A，Attributes 可能略低（没有 token 级证据方向）。
- **Kill**：预检 G2 < 0.65；或训练后 V\* 落在 V0 区间且 TB 不高于 V0——说明轨迹级加权在本设定下没有 token 级证据不可替代。
- **风险预登记**：n=2 时组内只有一对，$w$ 的方差大；教师序列似然偏好"短而安全"的回答，可能诱导回答变短（Amis r2 出现过的裸选项漂移），格式合规率必报。

## 7. 成本

2 次前向（学生 + 教师 real），比 A 少 null 前向及其输入构建开销（实测 S 臂 145.6 s vs Ahf 209.9 s）；组内统计量为标量运算。
