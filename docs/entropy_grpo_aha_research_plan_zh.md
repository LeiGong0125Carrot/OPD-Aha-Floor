# 研究计划：Entropy-GRPO + 视觉收益约束下的最大熵 OPD-Aha

> **状态：研究方案，尚未接入训练、尚无模型训练结果。**  
> 记录日期：2026-11-10。  
> 目标仓库：`LeiGong0125Carrot/OPD-Aha-Floor`。  
> 编写时检查的仓库基线：`0d25f6f7d47f497a33e7a47e826bce8e61e285bf`。  
> 本文记录讨论形成的方案，并补充推导、实现约定、实验判据和风险。新增推导与建议均明确标识，不代表原论文已经提出或验证。此次变更只涉及文档，不修改默认训练配方。

## 目录

1. [方案摘要与边界](#s01)
2. [已有发现、证据与研究假设](#s02)
3. [可借鉴的已发表工作](#s03)
4. [符号与原始 OPD-Aha](#s04)
5. [修改 Aha：保留视觉收益的最大熵重构](#s05)
6. [数学性质、对偶推导与数值求解](#s06)
7. [用熵构造 GRPO 奖励](#s07)
8. [联合损失与梯度边界](#s08)
9. [可复核的四候选数值例子](#s09)
10. [完整训练流程与数据接口](#s10)
11. [top-k、tail bucket 与采样分布](#s11)
12. [仓库接入建议与配置草案](#s12)
13. [实验矩阵与执行阶段](#s13)
14. [指标、日志与成功判据](#s14)
15. [风险与停止条件](#s15)
16. [单元测试与验收清单](#s16)
17. [尚未解决的问题与后续方向](#s17)
18. [参考资料与来源定位](#s18)

---

<a id="s01"></a>
## 1. 方案摘要与边界

### 1.1 要记录的核心思路

截图中的建议可具体化为两个互补的修改：

- **Entropy-GRPO：** 对同一输入采样多条学生 rollout，用学生 predictive entropy 构造各自的奖励，再做组内相对优势更新；首版不使用最终答案 correctness 作为训练奖励。
- **最大熵 Aha 重构：** 不再只使用固定强度的视觉指数倾斜目标，而是在保留指定比例的 Aha 视觉偏好增益、且不扩大相对 privileged teacher 的 KL 偏移的条件下，求最大熵目标。

两者的分工是：**Aha 提供局部视觉方向；GRPO 调整哪些推理路径更容易出现。**

候选总体目标为：

$$
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{Aha\text{-}ME}}
+\lambda_{\mathrm{RL}}\mathcal L_{\mathrm{H\text{-}GRPO}}.
$$

`Aha-ME` 在本文中只是 maximum-entropy target 的工作标签，不是已经确立的新方法名称。

### 1.2 本方案不做什么

首版保持 real/null 两个 teacher 条件和原始 student rollout，不新增 teacher 视角、答案奖励、reflection 词表奖励、attention 奖励、可训练 critic 或分叉采样。保留 JSD 蒸馏，不把它改称为某种等价的 policy-gradient loss。

最终答案 correctness 仍用于独立评估、模型选择和机制诊断；“不作为训练奖励”不等于“不评估任务正确性”。

本方案的最大熵 target、reward clipping 上限，以及其他实验中可能存在的 entropy floor / probability floor 是不同概念，不能因为仓库名包含 `Floor` 就混为一谈。

### 1.3 当前完成程度

目前完成的是数学方案记录、参考文献页面核验，以及一个四候选分布上的数值复核。尚未完成真实模型 logits 审计、分布式求解器、GRPO 接入、GPU 训练或 benchmark 验证。本文的伪代码、字段和配置均为拟议接口，不是现成可运行的仓库命令。

---

<a id="s02"></a>
## 2. 已有发现、证据与研究假设

### 2.1 原论文支持的机制

以讨论中提供的 `opd_aha_arxiv.pdf` 为主要方法依据，公开入口见 [R0]。

1. **共享错误 prefix 可掩盖 privileged supervision。** 学生的错误视觉解释积累后，teacher 和 student 可能一起偏向错误 continuation；同一 teacher 的 real–null contrast 仍可保留视觉偏好。来源：[R0] §2、§3.1。
2. **reflection 的相对优势主要来自更强的错误延续抑制。** 不应简化为“teacher 直接奖励 wait”；反思 token 的绝对概率也可能下降。来源：[R0] §4.2，印刷页 6–7。
3. **反思后出现从 prefix support 向 visual-evidence support 的转移。** 视觉支持恢复存在时间延迟。来源：[R0] §4.3，印刷页 7。
4. **证据特定方向不可被泛化为任意扰动。** token-shuffled contrast 和无关区域对照提示，恢复依赖具体视觉信号，而非仅仅改变目标幅度或不确定性。来源：[R0] Appendix B.2，印刷页 17–18。

### 2.2 用户在讨论中报告的新增观察

用户报告：相对基础模型，OPD-Aha 产生更多 reflection tokens；反思之后 predictive entropy 明显上升，对 visual tokens 的 attention 也上升。

**证据状态：**这些是本轮讨论报告的实验观察。本文不附带其原始日志、图表、统计检验或效应量，因此不编造具体增幅、显著性、模型配置或因果结论。后续应保存对应 checkpoint、数据划分、窗口定义、层/头聚合方式与原始分析产物的可追溯位置。

新增 entropy / attention 观察不能直接替代原论文的 continuation support / visual-evidence support 指标。

### 2.3 本方案要检验的假设

> Aha 已经提供离开错误 continuation 的方向，但在相同视觉偏好预算下，可能存在更有利于后续重新探索的 target shape；Entropy-GRPO 可能把这种局部空间转化为更有价值的推理路径。

这是**待验证假设**，不是“原始 Aha 已发生熵坍缩”的既定诊断。也不意味着更多 reflection、更高 entropy 或更多 visual attention 本身就是更好的推理。

一个优先诊断问题是：**成功反思和失败反思是否都伴随 entropy / attention 上升？**若二者都上升，熵更可能是重新探索发生的标志，而不是重新探索有价值的判据。

---

<a id="s03"></a>
## 3. 可借鉴的已发表工作

以下出版状态与论文要点于 2026-11-10 根据官方论文集页面核验。这里记录与方案相关的思想，不把这些论文当作当前组合有效性的证据。

| 文献 | 出版状态 | 借鉴点 | 与本方案的区别 |
| --- | --- | --- | --- |
| [R1] The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning | NeurIPS 2025 | EM-RL 把负熵当作奖励；区分 reward-driven RL、直接 entropy finetuning 与 inference adjustment | 本方案测试正熵奖励，符号与目标不同；不能用该论文证明正熵有效 |
| [R2] Demystifying Entropy Control in LLM RL Training: Theoretical Analysis and Dynamic Scheduling | ICML 2026 | 熵控制效果依赖正负样本的分布差异，讨论动态切换增熵与减熵 | 其反馈条件不同，不能直接转移到无 correctness reward 的视觉蒸馏 |
| [R3] Mind Your Entropy: From Maximum Entropy to Trajectory Entropy-Constrained RL | ICML 2026 | 区分当前动作熵和未来累计熵；显式约束探索预算 | 原文是含 value/critic 估计的 off-policy 框架，不是 GRPO |
| [R4] Entropy-preserving reinforcement learning | ICLR 2026 | REPO 修改 advantage，ADAPO 调整 clipping；熵控制可进入策略更新 | 保持探索不等于获得视觉纠错；本文首版不同时引入其全部修改 |
| [R5] GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy | ICML 2026 | 用熵重分配已有 reward、构造更细粒度更新 | 熵重分配任务奖励与熵本身作为 reward 是两件事 |
| [R6] Relative Entropy Policy Search | AAAI 2010 | 在相对熵约束下推导分布更新、通过对偶求解 | 本文具体的视觉收益约束来自 Aha，不是 REPS 的直接复现 |

GRPO 的基础接口参考 DeepSeekMath [R7] §4.1；这里将它作为方法原始来源，不额外声称其会议录用状态。OPD-Aha [R0] 是本项目基线，不属于本轮外部非 OPD 文献筛选。

---

<a id="s04"></a>
## 4. 符号与原始 OPD-Aha

### 4.1 统一符号

| 符号 | 定义 |
| --- | --- |
| $x,I$ | 文本问题与学生原始图像 |
| $I^+,I^0$ | teacher privileged evidence 与 visual null |
| $h_{i,t}=(x,y_{i,<t})$ | 第 $i$ 条 rollout 在位置 $t$ 的文本 prefix |
| $\pi_\theta$ | 可训练学生策略 |
| $\pi_b$ | 本轮冻结的 behavior / rollout policy |
| $p^+_{i,t},p^0_{i,t}$ | 同一个冻结 teacher 在 real/null 条件下的分布 |
| $u_{i,t}$ | real–null log-probability contrast |
| $q^A_{i,t}$ | 原始 Aha target |
| $q^*_{i,t}$ | 本方案约束最大熵 target |
| $\beta$ | 原始 Aha 视觉倾斜强度 |
| $\kappa$ | 保留的平均视觉偏好增益比例 |
| $\epsilon^A_{i,t}$ | 原始 Aha 相对 $p^+_{i,t}$ 的 KL，作为 target KL 预算 |
| $\varepsilon_{\mathrm{clip}}$ | GRPO ratio clipping 范围，与 target KL 预算无关 |
| $\lambda_{\mathrm{RL}}$ | RL loss 相对 distillation loss 的系数 |
| $G$ | 同一输入的 rollout 数量 |
| $M_{i,t}$ | 有效学生 response token mask |
| $T_i=\sum_tM_{i,t}$ | 有效 response 长度 |

所有对数使用自然对数，熵和 KL 的单位是 nats。理论先在固定有限 support 上推导；全词表与 top-k + tail 的区别见第 11 节。

### 4.2 原始目标

省略 $i,t$ 下标：

$$
u(v)=\log p^+(v)-\log p^0(v).
$$

原始 Aha 求解：

$$
q^A=\arg\max_{q\in\Delta(\mathcal V)}
\left\{\beta\mathbb E_q[u]-D_{\mathrm{KL}}(q\Vert p^+)\right\},
$$

得到：

$$
q^A(v)=\frac{p^+(v)e^{\beta u(v)}}{\sum_wp^+(w)e^{\beta u(w)}}
=\operatorname{softmax}\left((1+\beta)\log p^+-\beta\log p^0\right)_v.
$$

上述定义来自 [R0] §3.1–3.2，Eq. (4)–(7)，印刷页 4–5。

**术语限制：**本文“视觉收益”只表示 $\mathbb E_q[u]$ 的增加，不等于任务正确率、校准度或因果视觉依赖。$u(v)<0$ 也不等于该 token 必然错误。

### 4.3 论文配置与当前仓库配置不可混淆

[R0] Appendix C.2 / Table 4（印刷页 19）列出 JSD、student top-100 + one tail bucket、每题 8 条 rollout、随机种子 42、学习率 $2\times10^{-6}$、global batch size 96、最大 prompt/response 长度 8192/1024。

这些是**上传论文版本的配置**，不是本文对当前每个训练脚本的运行配置保证。正式实验前应导出现有 launcher 的 resolved config，并记录实际的 checkpoint、数据子集、$\beta$、batch 语义和所有已有 floor/center 开关。

---

<a id="s05"></a>
## 5. 修改 Aha：保留视觉收益的最大熵重构

### 5.1 简单 entropy bonus 为什么只是基线

给原始 target objective 加一个 $\eta H(q)$：

$$
\max_q\left\{\beta\mathbb E_q[u]-D_{\mathrm{KL}}(q\Vert p^+)+\eta H(q)\right\}
$$

会得到：

$$
q^{\mathrm{temp}}(v)=\operatorname{softmax}
\left(\frac{\log p^+(v)+\beta u(v)}{1+\eta}\right).
$$

这等价于对 Aha logits 做统一温度平滑。它没有显式限制为了增熵可以损失多少视觉偏好，因此列为必要对照，不作为主方案。

### 5.2 用原始 Aha 定义收益与预算

定义：

$$
\mu^+=\mathbb E_{p^+}[u],\qquad
\mu^A=\mathbb E_{q^A}[u],\qquad
\Delta^A=\mu^A-\mu^+,
$$

$$
\epsilon^A=D_{\mathrm{KL}}(q^A\Vert p^+).
$$

若 $q_\beta\propto p^+e^{\beta u}$，则：

$$
\frac{d}{d\beta}\mathbb E_{q_\beta}[u]=\operatorname{Var}_{q_\beta}(u)\geq0.
$$

所以 $\beta\geq0$ 时 $\Delta^A\geq0$。该指数倾斜族的单调性亦见 [R0] Appendix A，Eq. (15)。

引入 $\kappa\in[0,1]$，设置：

$$
m=\mu^++\kappa\Delta^A.
$$

例如 $\kappa=0.9$ 表示保留原 Aha 相对 teacher 的至少 90% **平均视觉偏好增益**，不表示保留 90% accuracy。

### 5.3 新的优化问题【本方案提出】

$$
\boxed{
\begin{aligned}
q^*=\arg\max_{q\in\Delta(\mathcal V)}\quad&H(q)\\
\mathrm{s.t.}\quad&\mathbb E_q[u]\geq m,\\
&D_{\mathrm{KL}}(q\Vert p^+)\leq\epsilon^A.
\end{aligned}}
\tag{P}
$$

第一条约束保留视觉偏好增益，第二条约束不允许比原始 Aha 更远地离开 privileged teacher。剩余自由度用于最大化候选不确定性。

注意：这约束的是 target，而不是直接对学生策略施加硬约束。JSD 蒸馏只提供软匹配压力。

---

<a id="s06"></a>
## 6. 数学性质、对偶推导与数值求解

### 6.1 假设与保证

假设 support 有限，$p^+(v),p^0(v)>0$，$u(v)$ 有限，$\beta$ 有限。可行集合是 simplex 与线性约束、KL sublevel set 的交集，是非空紧凸集，因为 $q^A$ 总是可行。

$H(q)$ 严格凹，因此原问题存在唯一最优分布。精确求解时：

$$
H(q^*)\geq H(q^A),
$$

$$
\mathbb E_{q^*}[u]-\mu^+\geq\kappa\Delta^A,
$$

$$
D_{\mathrm{KL}}(q^*\Vert p^+)\leq D_{\mathrm{KL}}(q^A\Vert p^+).
$$

这些保证不包括 $H(q^*)\geq H(p^+)$、$H(q^*)\geq H(\pi_\theta)$、逐 token 正确性、学生 accuracy、reflection 频率、训练稳定性或全局最优性。

### 6.2 为什么 $\kappa=1$ 必须退化回原始 Aha

设 $Z=\sum_vp^+(v)e^{\beta u(v)}$。由 $\log q^A=\log p^++\beta u-\log Z$ 可得：

$$
D_{\mathrm{KL}}(q\Vert q^A)
=D_{\mathrm{KL}}(q\Vert p^+)-\epsilon^A
+\beta\left(\mu^A-\mathbb E_q[u]\right).
\tag{I}
$$

当 $\kappa=1$ 时，两条约束使右侧不大于零。KL 非负，所以只能 $q=q^A$。

因此：**不扩大 KL 邻域且保留全部视觉增益时，不存在额外的增熵自由度。**必须允许 $\kappa<1$，或者另行研究扩大 KL 预算；首版只考虑前者。

### 6.3 对原始 Aha 的偏移上界【记录时补充的推导】

由恒等式 (I)，任何满足 (P) 的分布都满足：

$$
D_{\mathrm{KL}}(q\Vert q^A)
\leq\beta(1-\kappa)\Delta^A.
$$

这为 $\kappa$ 提供了额外解释：它同时限制新 target 离开原 Aha 的程度。但它仍然只是分布层面的约束，不保证某个被抑制的错误 token 概率不会回升。

### 6.4 拉格朗日对偶

对视觉下限与 KL 上限分别引入 $a,b\geq0$：

$$
\mathcal J(q,a,b)
=H(q)+a(\mathbb E_q[u]-m)
+b(\epsilon^A-D_{\mathrm{KL}}(q\Vert p^+)).
$$

固定 $a,b$，对 $q$ 求极大得到：

$$
\boxed{q_{a,b}(v)=\operatorname{softmax}
\left(\frac{b\log p^+(v)+a u(v)}{1+b}\right).}
\tag{D1}
$$

对偶目标为：

$$
\boxed{
g(a,b)=(1+b)\log\sum_v\exp\left(\frac{b\log p^+(v)+a u(v)}{1+b}\right)-am+b\epsilon^A,
}
$$

求 $\min_{a,b\geq0}g(a,b)$。其梯度为：

$$
\partial_a g=\mathbb E_{q_{a,b}}[u]-m,
\qquad
\partial_b g=\epsilon^A-D_{\mathrm{KL}}(q_{a,b}\Vert p^+).
\tag{D2}
$$

投影下降可写成：

$$
a\leftarrow[a+\eta_a(m-\mathbb E_q[u])]_+,
$$

$$
b\leftarrow[b+\eta_b(D_{\mathrm{KL}}(q\Vert p^+)-\epsilon^A)]_+.
$$

视觉收益不足时提高 $a$；偏离 teacher 过远时提高 $b$。这解释了更新方向，但固定步长投影迭代本身不构成有限步收敛保证。

若 $\beta>0$、$u$ 非常量、$\kappa<1$，可用适当比例的 $q^A$ 与 $p^+$ 混合构造严格可行点，因此满足 Slater 条件。$\kappa=1$、零预算等退化情况单独处理，不依赖对偶乘子保持有限。

### 6.5 KKT 与数值验收

应检查：

$$
a(\mathbb E_q[u]-m)\approx0,
\qquad
b(\epsilon^A-D_{\mathrm{KL}}(q\Vert p^+))\approx0.
$$

对可行候选 $q$，$g(a,b)-H(q)$ 是有意义的 primal–dual gap；**不可行候选的 gap 不能直接当作最优性证书。**

建议先用 CPU float64、二维有界优化验证，再实现 GPU batched solver。GPU 版按 token chunk 计算，只返回 detached target 与诊断，避免保存求解图。不得声称少量固定迭代天然等价于精确最优解。

接受 target 时至少检查：归一化、finite 值、视觉约束、KL 约束、熵不低于 $q^A$，并记录 gap / KKT 残差。记录 best feasible iterate；若无法达到容差，回退到 $q^A$。

### 6.6 边界情况

| 情形 | 处理 |
| --- | --- |
| $\kappa=1$ | 直接返回 $q^A$ |
| $\beta=0$ | $q^A=p^+$、KL 预算为零，返回 $p^+$ |
| $u$ 为常量或预算近零 | 返回原 target；记录退化率 |
| 增益 $\Delta^A$ 很小 | 不使用不稳定的增益比例诊断；用绝对残差 |
| 求解出现 NaN/Inf、约束违背或熵反而下降 | 回退原 target，并计入 fallback rate |
| probability smoothing / clipping | 必须重新归一化；所有预算用同一处理后的分布计算；保证只针对处理后的问题 |

---

<a id="s07"></a>
## 7. 用熵构造 GRPO 奖励

### 7.1 学生 predictive entropy

对冻结 behavior policy $\pi_b$：

$$
H^b_{i,t}=-\sum_v\pi_b(v\mid I,h_{i,t})\log\pi_b(v\mid I,h_{i,t}).
$$

不是 sampled-token surprisal $-\log\pi_b(y_{i,t}\mid h_{i,t})$，也不是 attention entropy、teacher entropy、最终答案的经验频率熵。

每轮缓存 behavior entropy。即使 learner 对同一批数据更新多次，也不在更新中不断重算奖励和 advantage。

### 7.2 R0：最直接的平均熵奖励

$$
R_i^{\mathrm{raw}}=\frac1{T_i}\sum_tM_{i,t}H^b_{i,t}.
\tag{R0}
$$

它是截图想法的最直接测试。使用平均而非总和，减少“增加长度就直接累加奖励”的激励，但不消除长度选择偏差。

### 7.3 R1：由新 target 给出局部上限的熵奖励

定义 $H^*_{i,t}=H(q^*_{i,t})$，使用：

$$
\boxed{
R_i^{\mathrm{cap}}=\frac1{T_i}\sum_tM_{i,t}\min\{H^b_{i,t},H^*_{i,t}\}.
}
\tag{R1}
$$

学生熵上升可增加奖励，但超过当前 target 的最大可行熵后，不获得额外收益。

**准确解释：**$H^*$ 是满足 (P) 的分布中的最大熵，并不是所有学生策略必须遵守的硬熵上限。一个学生分布即使熵低于 $H^*$，也可能完全不满足视觉约束；视觉方向仍由 JSD 提供。

R1 也不惩罚超过上限的熵，只使这部分奖励饱和。不能把它叫作“熵超限惩罚”或 entropy floor。

### 7.4 R1 的路径依赖风险【实现审计重点】

不同 rollout 访问不同 prefix，$H^*$ 随之改变。因而高 reward 可能来自“更有益的重新探索”，也可能来自“走进更容易保持不确定的状态”。

需要同时记录 $H^b$、$H^*$、二者最小值和 saturation fraction，并在相同或匹配 prefix 上比较。R1 不是已经消除 reward hacking 的保证。

对于 saturation 很高的组，奖励差异可能主要反映 $H^*$ 的路径差异，而不是学生进一步提高了熵。应把这一点作为可证伪分析，而非事后解释。

### 7.5 组内 advantage

同一 $(I,x)$ 的 $G$ 条 rollout 构成一个组：

$$
\bar R=\frac1G\sum_iR_i,
\qquad
s_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},
$$

$$
A_i^H=\frac{R_i-\bar R}{\max(s_R,\sigma_{\min})}.
\tag{A}
$$

这里明确使用 population standard deviation。组内差异低于预先设置的容差时置零；不要用很小的数值噪声制造满幅度优势。

首版每题 $G=8$ 与论文设置保持一致。可增加 mean-only centering 作为后续消融，但不要与 target/reward 改动同时混入首个比较。

**关键约定：**这是每条 rollout 的 reward。不能先算一个 group entropy，再把同一个标量给组内所有 rollout，否则所有 group-relative advantages 都为零。

### 7.6 熵奖励不直接区分同一状态的当前 action

固定同一个 prefix 时，$H(\pi_b(\cdot\mid h))$ 对所有下一词候选相同。reward-driven 更新的辨别力来自后续访问状态的差异，而不是“当前 action 自带一个熵”。

极端的单步、同状态任务中，当前分布熵作为每条样本 reward 会给出零 group advantage。若目标是直接改变这个状态的熵，需要直接 entropy regularization 或其他 action-dependent 信号；这与本方案的路径优化不同。

---

<a id="s08"></a>
## 8. 联合损失与梯度边界

### 8.1 GRPO surrogate

令：

$$
\rho_{i,t}(\theta)=\exp\big(\log\pi_\theta(y_{i,t}\mid I,h_{i,t})-\log\pi_b(y_{i,t}\mid I,h_{i,t})\big).
$$

使用：

$$
\mathcal L_{\mathrm{H\text{-}GRPO}}
=-\frac1G\sum_i\frac1{T_i}\sum_tM_{i,t}
\min\left\{\rho_{i,t}A_i^H,
\operatorname{clip}(\rho_{i,t},1-\varepsilon_{\mathrm{clip}},1+\varepsilon_{\mathrm{clip}})A_i^H\right\}.
\tag{PG}
$$

此处保留 GRPO 的组内比较和 clipped update 思路 [R7]，不引入 critic。是否使用 reference-policy KL 必须作为独立配置记录；公式 (PG) 本身不包含这一项，teacher-target KL 约束不能替代 student-update trust region。

### 8.2 Aha-ME 蒸馏

$$
\mathcal L_{\mathrm{Aha\text{-}ME}}
=\frac{\sum_{i,t}M_{i,t}\operatorname{JSD}(\operatorname{sg}[q^*_{i,t}],\pi_\theta(\cdot\mid I,h_{i,t}))}
{\sum_{i,t}M_{i,t}}.
\tag{KD}
$$

对称 JSD 的定义为：

$$
\operatorname{JSD}(q,p)=\tfrac12D_{\mathrm{KL}}(q\Vert m)+\tfrac12D_{\mathrm{KL}}(p\Vert m),\quad m=\tfrac12(q+p).
$$

teacher target detached，但 $m$ 中的学生分量不能错误地整体 detach；应沿用经过验证的现有 JSD 实现。

### 8.3 总损失与归一化

$$
\boxed{\mathcal L=\mathcal L_{\mathrm{Aha\text{-}ME}}+\lambda_{\mathrm{RL}}\mathcal L_{\mathrm{H\text{-}GRPO}}.}
$$

(KD) 是 valid-token mean，(PG) 是 sequence-mean/token-mean。两种归一化会给长短轨迹不同的相对权重，需显式记录。不要悄悄改变原基线的 reduction，然后把差异归因于方法。

首版可另做归一化匹配的控制实验。分布式聚合必须确认 denominator 是本 rank、global batch 还是 gradient-accumulation batch 的有效 token 数。

### 8.4 Stop-gradient 契约

| 量 | 本轮 learner update 是否求梯度 |
| --- | --- |
| real/null teacher logits、$u$、$q^A$ | 否 |
| $q^*$、dual variables、$m$、$\epsilon^A$ | 否 |
| behavior entropy、target entropy、reward、advantage | 否 |
| behavior sampled-token log-probability | 否 |
| 当前学生 sampled-token log-probability | 是，进入 GRPO |
| 当前学生分布 | 是，进入 JSD |

### 8.5 为什么这不是直接最大化学生 entropy 的精确梯度

若目标写成 $J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[R_\theta(\tau)]$，一般有：

$$
\nabla J=\mathbb E\left[R_\theta(\tau)\nabla\log\pi_\theta(\tau)+\nabla R_\theta(\tau)\right].
$$

本文冻结 $R_b$，只以 reward-driven surrogate 改变路径分布；没有直接反传 $\nabla R_\theta$。此外，group normalization、token-level ratio、clipping 和长度归一化都限制了无偏梯度解释。

因此不能称其为“完整轨迹 Shannon entropy 的精确无偏优化”。特别是平均条件熵不是完整 sequence distribution entropy；固定 horizon 下，后者对应条件熵之和的期望，而不是简单除以随机长度。

---

<a id="s09"></a>
## 9. 可复核的四候选数值例子

> **仅为人工构造的分布例子，不是模型训练结果。** 本文记录时使用 float64 二维对偶优化重新计算，并检查约束、归一化和 KL 恒等式。

取：

$$
p^+=[0.65,0.15,0.18,0.02],\quad
p^0=[0.78,0.12,0.06,0.04],\quad\beta=2,\quad\kappa=0.9.
$$

| 候选（仅作直觉标签） | $p^+$ | $q^A$ | $q^*$ |
| --- | ---: | ---: | ---: |
| 当前解释延续 | 0.650000 | 0.195342 | 0.130306 |
| 重新检查 | 0.150000 | 0.101427 | 0.189231 |
| 视觉支持较强的候选 | 0.180000 | 0.701067 | 0.638046 |
| 其他候选 | 0.020000 | 0.002164 | 0.042417 |

| 指标 | 原始 Aha | 最大熵 target |
| --- | ---: | ---: |
| $H(q)$ | 0.813363937 | 1.001325475 |
| $\mathbb E_q[u]$ | 0.755718715 | 0.690031822 |
| $D_{\mathrm{KL}}(q\Vert p^+)$ | 0.673859272 | 0.673859272 |

$\mu^+=0.098849789$，增益保留比例约为 0.900000。对应乘子约为 $a=1.552943554$、$b=0.139490584$。

将原始 Aha 直接升温到相同 entropy，需要温度约 1.473666415；其平均视觉偏好约为 0.625058646，低于约束下限 0.690031822。

这个例子说明：在同一 entropy 水平下，协同调整视觉倾斜与平滑可以比统一升温保留更多平均视觉偏好。**它不证明真实 VLM 上更好。**另外，“其他候选”的概率升高，提醒我们平均收益约束不是逐 token 的反错误保证。

### 9.1 数值复核伪代码

```text
logp = log(normalize(p_plus))
u = logp - log(normalize(p_null))
logqa = log_softmax(logp + beta * u)
qa = exp(logqa)
mu_plus = dot(p_plus, u)
mu_aha = dot(qa, u)
eps = dot(qa, logqa - logp)
m = mu_plus + kappa * (mu_aha - mu_plus)

objective(a, b):
    logits = (b * logp + a * u) / (1 + b)
    logq = log_softmax(logits)
    q = exp(logq)
    value = (1 + b) * logsumexp(logits) - a * m + b * eps
    grad_a = dot(q, u) - m
    grad_b = eps - dot(q, logq - logp)
    return value, [grad_a, grad_b]

(a, b) = bounded_minimize(objective, lower_bounds=[0, 0])
q_star = softmax((b * logp + a * u) / (1 + b))
assert visual_constraint(q_star)
assert kl_constraint(q_star)
assert H(q_star) >= H(qa) - tolerance
```

同一恒等式 (I) 在 100 个随机四维 simplex 分布上进行数值检查，通过。未据此宣称完成通用求解器测试。

---

<a id="s10"></a>
## 10. 完整训练流程与数据接口

### 10.1 每个 iteration 的流程

```text
1. 固定 behavior snapshot pi_b。
2. 对每个输入采样 G 条完整学生 response；保存 prompt/group/rollout ID。
3. 保存每个 action 的 behavior log-probability、有效 mask、结束原因。
4. 在同一学生 prefix 上获取 behavior predictive entropy 和固定 support。
5. 同一个冻结 teacher 分别在 real/null 图像上评分。
6. 计算 u、原始 q_A、视觉收益下限 m、KL 预算 eps_A。
7. 求解 q_star；执行约束检查与 fallback。
8. 在同一 support 上计算 H_b 和 H_star；生成 raw/capped reward。
9. 按 prompt group 计算冻结的 advantage。
10. learner 前向：当前 sampled-token log-probability + 当前分布。
11. 计算 JSD 与 clipped GRPO loss；加权后更新学生。
12. 记录 target 残差、reward、entropy、梯度、长度及重复指标。
13. 进入下一轮时刷新 behavior snapshot，重新采样。
```

整个 iteration 中不以正确性过滤 rollout，不把正确答案输入 reward/target 构造。teacher privilege 沿用基线合法训练信息。

### 10.2 建议缓存字段（拟议，不表示仓库已存在）

| 字段 | 典型形状/单位 | 目的 |
| --- | --- | --- |
| `prompt_group_id`, `rollout_id`, `behavior_version` | 每序列标识 | 防止跨题、跨 snapshot 混组 |
| `response_mask`, `termination_reason` | `[B,T]` / 每序列 | EOS、padding、truncation 语义 |
| `behavior_action_logprob` | `[B,T]`，nats | GRPO denominator，始终为原始 token action |
| `behavior_entropy_bucketed` | `[B,T]`，nats | 冻结熵奖励 |
| `support_token_ids` | `[B,T,K]` | 同一前缀下的固定 top-k identity |
| `target_aha`, `target_maxent` | `[B,T,K+1]` | 蒸馏与比较 |
| `target_entropy`, `visual_floor`, `kl_budget` | `[B,T]` | reward cap 与约束检查 |
| `solver_status`, `fallback_mask` | `[B,T]` | 数值质量审计 |
| `entropy_reward`, `entropy_advantage` | `[B]` | 组内 RL 更新 |

teacher 分布无需长期全量存盘，可按 chunk 计算并缓存必要的 detached target。模型权重、训练样本、logs 和 outputs 不提交到 Git，遵守 `AGENTS.md`。

### 10.3 冻结与同步

reward 计算依赖本轮 behavior policy，不能用已更新 learner 的 entropy 替代。分布式 shuffle、packing、microbatch 和重排必须保持 group ID。若异步策略年龄不一致，先采用同步版本隔离研究问题，不在首版引入复杂 off-policy correction。

---

<a id="s11"></a>
## 11. top-k、tail bucket 与采样分布

### 11.1 同一个 support 是硬约定

以 behavior snapshot 在每个 prefix 的 top-$K$ token 集 $S$ 为显式 support，剩余词表合并为 tail：

$$
p_{\mathrm{tail}}=1-\sum_{v\in S}p(v).
$$

student、real teacher、null teacher、$q^A$ 和 $q^*$ 必须使用相同的 token identities 与同一个 tail partition。一次 learner update 内固定 $S$，不能各自重新 top-k 后直接逐坐标比较。

### 11.2 合并后的熵不是全词表熵

对真实分布：

$$
H_{\mathrm{full}}=H_{\mathrm{bucketed}}+p_{\mathrm{tail}}H(p(\cdot\mid\mathrm{tail})).
$$

因此不能拿 full-vocabulary student entropy 与 bucketed target entropy 做 `min`。首版 R0/R1 都使用相同 bucket entropy；另记录 full entropy 或可用近似作为诊断。

还应记录 tail mass。表面上的增熵可能主要是向 tail 移动质量，而非保留有意义的显式候选。

### 11.3 先倾斜再合并，与先合并再倾斜不等价

一般有：

$$
\sum_{v\in\mathrm{tail}}p^+(v)\left(\frac{p^+(v)}{p^0(v)}\right)^\beta
\neq
p^+_{\mathrm{tail}}\left(\frac{p^+_{\mathrm{tail}}}{p^0_{\mathrm{tail}}}\right)^\beta.
$$

**实施前必须审计现有 Aha 的 target 构造顺序。**若在 bucket 空间直接使用 $u_{\mathrm{tail}}=\log p^+_{\mathrm{tail}}-\log p^0_{\mathrm{tail}}$，所有定理只针对这一 coarse problem。

若已有基线是 full-vocabulary tilt 后再聚合，则不能不加说明地换成 coarse tilt，再声称保持原 Aha。解决方式应先选定并记录：全词表审计版本，或一个单独声明的 coarse-Aha baseline。不能直接套用需要指数倾斜结构的 $\kappa=1$ 退化证明，而忽略原 target 已不满足该结构。

### 11.4 tail 不是 RL 的 action

GRPO ratio 必须基于实际采样 token 的 log-probability。即使该 token 不在 top-$K$，也不能用整个 tail bucket 的概率代替它。tail 只服务于分布蒸馏和粗化熵计算。

### 11.5 采样温度与 top-p 的约定

需要记录 entropy 用的是原始 softmax、温度变换后分布，还是经过 top-p/top-k 截断的实际采样分布。行为策略概率与 learner ratio 的解释必须一致。

首个数学审计可使用 temperature=1、无截断以排除混淆；正式模型实验应与基线保持相同 decoding 配置。若既有实现使用未变换 log-probability 为经过变换的采样做近似更新，必须记录这一近似，不能默认为精确 importance sampling。

---

<a id="s12"></a>
## 12. 仓库接入建议与配置草案

### 12.1 接入范围

已确认仓库存在 `scripts/`、`verl/trainer/ppo/core_algos.py`、`verl/trainer/ppo/ray_trainer.py`、`verl/trainer/ppo/reward.py`、`verl/trainer/ppo/metric_utils.py` 和 `eval/`。以下是拟议职责划分，**不是逐函数代码审计或已完成实现**。

| 区域 | 拟议职责 |
| --- | --- |
| 新增独立 target solver 模块（具体路径实施时确定） | 原始 target、约束构造、对偶求解、fallback、数值诊断 |
| `verl/trainer/ppo/ray_trainer.py` | group ID、behavior snapshot、reward/target 数据传递 |
| `verl/trainer/ppo/core_algos.py` | 新 entropy advantage 与现有 policy surrogate 的适配 |
| `verl/trainer/ppo/reward.py` 或独立 reward 模块 | 与 correctness reward 严格分离的熵评分 |
| 实际 actor loss 计算处（待定位） | JSD + GRPO 的加权与反传 |
| `verl/trainer/ppo/metric_utils.py` | 约束、entropy、reward、梯度与稳定性日志 |
| `scripts/` | 单独的新实验入口；旧 `train_opd_aha.sh` 默认行为不变 |

已有 `hidden_probe/hidden_plan.md` 属于其他研究记录，本次不修改、不合并其机制或结论。

### 12.2 配置草案

下列 YAML 是方案规格，不是当前仓库可直接识别的配置：

```yaml
experiment:
  name: entropy_grpo_aha_maxent_v1
  enabled: false                 # 后续实现后显式启用；默认不改变旧配方
  seed: 42
  group_size: 8

target:
  mode: constrained_max_entropy
  beta: null                     # 从所选 Aha 基线 resolved config 继承并记录
  kappa: 0.90                    # 起始测试点，不是验证过的最优值
  support: behavior_topk_plus_tail
  topk: 100
  freeze_support_per_rollout: true
  solver:
    audit_dtype: float64
    production_dtype: float32
    method: bounded_dual          # 算法选择仍须通过测试
    max_iterations: null          # 由真实 logits 审计决定
    constraint_atol: null         # 由精度与真实量级决定
    fallback: original_aha

entropy_reward:
  kind: capped                   # raw 是必要对照
  entropy_space: bucketed
  reduction: valid_response_mean
  detach: true
  use_correctness: false
  use_reflection_count: false
  use_visual_attention: false

group_advantage:
  population_std: true
  std_floor: null                 # 根据离线 reward 分布预先确定
  zero_group_tolerance: null

gradient_update:
  lambda_rl: null                 # 先检查梯度规模，再做小范围对照
  ratio_clip: null                # 继承/显式记录现有 GRPO 设置
  distillation: jsd
  reward_frozen_during_inner_updates: true
  direct_entropy_bonus: 0.0       # 如测试，必须作为独立消融
```

未填写项是实施前的必要决策，不应被程序默默赋予任意值。null 配置不构成可运行配方。

### 12.3 联合权重的起点

先测 $g_{\mathrm{KD}}$ 与 $g_{\mathrm{RL}}$ 的范数及 cosine similarity，再选择使 RL 初始梯度贡献小于 KD 的 $\lambda_{\mathrm{RL}}$。不要根据 loss 标量大小直接配权。

梯度审计可按固定间隔进行，避免每步双重 backward 的显著开销；若只在参数子集上测量，必须标明是代理指标。首版固定权重，不同时加入自动梯度平衡、动态 clipping 和多个新 schedule。

---

<a id="s13"></a>
## 13. 实验矩阵与执行阶段

### 13.1 必要的因子分解

原讨论的五项对照保留，同时增加 E0，以免完整方案同时改变 target 和 reward cap 后无法解释收益来源。

| ID | Target | RL / regularization | 要回答的问题 |
| --- | --- | --- | --- |
| A | 原始 Aha | 无 | 基线 |
| B | 原始 Aha | raw entropy GRPO | 截图最直接的思路是否有效 |
| C | 温度平滑 Aha | raw entropy GRPO | 是否统一平滑已足够 |
| D | 约束最大熵 target | 无 | target 改动的独立价值 |
| E0 | 约束最大熵 target | raw entropy GRPO | target 与 RL 的交互；与 B 比较 |
| E | 约束最大熵 target | capped entropy GRPO | reward cap 的增量；与 E0 比较 |
| F（诊断） | 原始 Aha | 直接学生 entropy bonus | group 路径优化是否优于直接局部增熵 |

温度对照 C 应尽量与约束 target 做 per-state entropy matching；需记录温度搜索失败率及达到的熵误差。可增加 temperature-only/no-RL 对照，与 D 做更严格比较。

### 13.2 Phase 0：数学与真实 logits 审计

先不训练模型，完成：

- 复现第 9 节算例和边界条件；比较 primal 与 dual。
- 在现有 checkpoint 的固定 rollout 上统计 $\Delta^A$、$\epsilon^A$、可获得的 entropy gain、fallback 与约束残差。
- 比较成功/失败反思、非反思匹配位置的 entropy 和 visual-support dynamics。
- 检查 raw/capped reward 的组内方差是否具有足够信号，是否主要由长度、tail mass、模板文字或 target cap 决定。
- 审计 target 构造顺序、采样分布、mask、old log-probability 和 group ID。

**进入下一阶段的前提：**真实 logits 上求解稳定，并且 group reward 不是几乎全零或主要由无意义混淆驱动。

### 13.3 Phase 1：最小训练对照

先做 A、B、D，分别判断 entropy-RL 与 target 改动是否值得继续；再做 C、E0、E、F。所有试验从同一个明确的 Aha checkpoint 或同一个基础 checkpoint 开始，不混用两种起点。

为延续讨论，首轮可从相同 Aha checkpoint 开始，并冻结同一个原始 teacher；不得把 teacher 也悄悄替换为新的学生快照。

### 13.4 Phase 2：稳健性与机制

若初步有效，再检查 $\kappa\in\{0.8,0.9,0.95,1.0\}$ 等少量点；$\kappa=1$ 必须退化原 target。$\lambda_{\mathrm{RL}}$ 做小范围 sweep，不在测试集上调参。

seed 42 用于首轮可比性；随后增加多个预先登记的种子。保持训练数据、评估 prompt、teacher privilege、rollout budget、decoding 参数和 checkpoint selection 规则一致。

对成本同时报告 optimizer steps、生成 token、训练 token、teacher forward 次数、solver 时间和 GPU-hours；只比较 step 数可能掩盖求解与长度开销。

### 13.5 数据与评估协议

优先沿用当前项目已固定的训练/验证/测试划分，不把论文中的完整数据量直接当作当前运行量。对新增数据子集保存 manifest/hash。

论文 benchmark 可作为候选评估范围，但具体运行集合以项目现行协议为准。全局 accuracy、长回复稳定性与 OOD 能力共同决定是否保留方案；不能只挑选 reflection 增多的例子。

---

<a id="s14"></a>
## 14. 指标、日志与成功判据

### 14.1 局部分布优化指标

| 指标 | 含义 |
| --- | --- |
| `target_entropy_gain` | $H(q^*)-H(q^A)$ |
| `visual_constraint_slack` | $\mathbb E_{q^*}[u]-m$ |
| `kl_constraint_slack` | $\epsilon^A-D_{\mathrm{KL}}(q^*\Vert p^+)$ |
| `visual_gain_retention` | $(\mathbb E_{q^*}[u]-\mu^+)/\Delta^A$，仅在分母可靠时 |
| `target_kl_to_aha` | $D_{\mathrm{KL}}(q^*\Vert q^A)$ |
| `dual_gap`, `kkt_residual` | 可行性与最优性诊断 |
| `solver_iterations`, `fallback_fraction` | 成本及失败率 |
| `negative_u_mass`, `tail_mass` | 平均约束是否掩盖个别负方向或 tail 扩张 |

### 14.2 GRPO 与训练稳定性

记录 reward 的组内均值/标准差、zero-advantage group fraction、advantage 分位数、ratio 分位数、clip fraction、学生相对 behavior 的更新 KL、两分支梯度范数与夹角。

KL clipping 不是硬 trust-region 保证；即使 target 求解严格满足约束，student 更新仍可能过大。

### 14.3 行为与机制指标

保留原论文的 continuation support、visual-evidence support，并增加用户报告的 entropy/attention 诊断。分析时固定反思事件定义和前后窗口，记录多次反思、非反思对照、长度与 token 类型匹配方式。

attention 需要预先规定层、头、视觉 token 数、归一化方式和聚合函数。attention mass 上升仅是行为相关证据，不等于因果视觉依赖增强。

反思率按“含反思的 response 比例”和“每百 token 事件数”同时记录，防止长度增加造成伪提升。进一步记录错误到正确、正确到错误、无效反复改口等情况。

### 14.4 任务与效率指标

最终评价包括：独立测试 accuracy、paired 差异及置信区间、多种子波动、OOD 表现、平均/p95 长度、重复 n-gram/句子比例、达到生成上限比例、单位预算的有效答案产出。

### 14.5 成功与失败如何判定

支持研究假设需要同时看到：约束成立；相对原 Aha 的任务改善；相对熵匹配温度对照的额外价值；相对 direct entropy bonus 的可解释差异；没有以显著重复、长度触顶或 OOD 回退换取收益。

若只有 entropy、reflection 或 attention 上升，而任务不改善，应报告“改变了行为分布，但没有支持有效推理改善”。若温度对照同样好，应收缩贡献到平滑效应，而不是坚持复杂约束设计必要。

---

<a id="s15"></a>
## 15. 风险与停止条件

| 风险 | 可能症状 | 检查与处理 |
| --- | --- | --- |
| entropy reward 鼓励犹豫或无意义路径 | reward 上升、accuracy 下降、解释反复 | 保留原 Aha 对照，检查 reward 与行为的条件关系；必要时停止该 reward 分支 |
| cap 随 prefix 变化导致状态选择投机 | 高 reward 主要来自高 $H^*$ | 分别记录学生熵和 cap；匹配 prefix 诊断 |
| 平均视觉约束掩盖错误 token 回升 | 负 $u$ 候选质量增多 | 检查逐 token/集合质量，不能只看平均收益 |
| teacher visual preference 不等于真实正确性 | teacher score 良好但答错 | 独立任务与视觉干预评估；不把 score 当 ground truth |
| 极小组内差异被归一化放大 | advantages 很大但 reward 差异极小 | std floor、零差异容差及 mean-only 对照 |
| 长度和 EOS 投机 | 拖延结束、触顶率上升 | 固定 EOS/mask 规则；同时看平均奖励与有效完成率 |
| 整条轨迹奖励强化错误前缀 | 先犯错再反思变得频繁 | 首版记录风险；局部 credit 只能作为单独后续方案 |
| JSD 与 GRPO 梯度冲突 | 梯度夹角长期为负、性能震荡 | 调低 RL 权重，先做独立分支消融 |
| top-k / tail 定义错配 | 熵提升集中在 tail 或约束无意义 | 固定 support、审计重构顺序和 full entropy |
| solver 数值失败 | 不可行 target、NaN、fallback 激增 | 回退、记录、修复后重跑；不宣称精确解 |
| mixed precision / decoder mismatch | ratio 异常、熵计算不一致 | float64 oracle、float32 分布计算、采样协议审计 |

在运行前根据基线确定性能/重复/触顶容忍带与检查频率。若出现 NaN、明显退化性重复、生成上限频繁触发、约束大量违背或验证性能持续越界，停止当前配置并保留诊断。不能在看到结果后临时修改停止标准。

---

<a id="s16"></a>
## 16. 单元测试与验收清单

### 16.1 数学模块

- [ ] 输出 finite、非负且归一化。
- [ ] $\kappa=1$ 精确返回原始 $q^A$。
- [ ] $\beta=0$、real=null、零 KL 预算返回原 teacher。
- [ ] 对偶梯度与有限差分一致。
- [ ] 与小词表 primal constrained optimizer 比较。
- [ ] 验证视觉下限、KL 上限、entropy ordering 与恒等式 (I)。
- [ ] 测试一条约束不活跃、两条均活跃、退化预算等情形。
- [ ] 对数极端值、近零 tail、不同 vocab/support 大小稳定。
- [ ] 验证 batch/chunk 结果与逐 token 计算一致。
- [ ] fallback 保持原基线 target，失败率可追溯。

### 16.2 Reward / advantage 模块

- [ ] 只有真实 response token 进入统计，padding 不计入。
- [ ] 相同 reward 的组输出零 advantage。
- [ ] shuffle/microbatch 不破坏 prompt grouping。
- [ ] raw/capped 使用相同 entropy space。
- [ ] cap 饱和行为符合定义，没有被错误实现成 floor。
- [ ] reward、advantage、behavior log-probability 全部 detached。
- [ ] 不读取 ground-truth correctness、reflection count 或 attention 作为 reward。
- [ ] 实际 action log-probability 不被 tail probability 替代。

### 16.3 集成

- [ ] 新开关关闭时恢复原训练路径和默认配方。
- [ ] $\lambda_{\mathrm{RL}}=0$ 隔离 target-only 实验。
- [ ] 原 target + RL 只改变 reward 分支，不隐藏增加额外正则。
- [ ] teacher 和 behavior snapshot 没有梯度。
- [ ] JSD mixture 中的学生梯度正确。
- [ ] 单卡/分布式 reduction 与 mask 语义一致。
- [ ] resolved config、基线 commit、checkpoint hash、dataset manifest 完整记录。
- [ ] 默认 scripts 不被实验逻辑覆盖。

以上清单除第 9 节明确报告的数值检查外，均为待完成项目。

---

<a id="s17"></a>
## 17. 尚未解决的问题与后续方向

1. **entropy 上升是有效反思的原因，还是反思后的伴随现象？** 当前观察不足以直接回答，需要匹配状态或受控续写分析。
2. **平均视觉收益是否是最合适的约束？** 它易求解但不保证具体纠错候选；逐集合约束会改变可行域和理论，应独立评估。
3. **何时需要增熵、何时应收敛？** 当前方案使用有上限的正熵奖励，不代表它适合整段推理的所有位置。
4. **group 带来的收益是什么？** 若直接 entropy bonus 同样有效，不能声称 group 结构提供独立价值。
5. **动态 support 会不会主导结果？** 固定词表算例不能代替真实 top-k 流程审计。
6. **局部 target 改善如何转化为时序 credit？** 首版采用 sequence advantage，尚未解决精细的因果归因。

后续可研究视觉冲突门控、反思后 entropy-change reward、视觉恢复 reward、局部 credit 或成功/失败条件下的动态调节。但这些不属于首版，不能与主方案一起加入后再无法归因。

### 当前决策记录

| 决策 | 状态 |
| --- | --- |
| 不以 answer correctness 作为首版训练奖励 | 已确定的方案边界 |
| 保留 real/null teacher 和 JSD | 已确定的方案边界 |
| 用 $\kappa$ 控制视觉增益保留比例 | 本方案提出 |
| KL 预算沿用原 Aha 偏移 | 本方案提出 |
| raw 与 capped entropy reward 都做对照 | 已规划 |
| $\kappa=0.9$ | 起始测试建议，未验证 |
| $\lambda_{\mathrm{RL}}$、solver 容差和迭代数 | 待真实 logits/梯度审计 |
| GPU 训练效果、novelty 和论文定位 | 尚无结论 |

---

<a id="s18"></a>
## 18. 参考资料与来源定位

### R0. OPD-Aha：项目基线

Chenhao Qiu, Dawei Li, Yechao Zhang, Lei Gong, Zhen Tan. **OPD-Aha: From Linguistic Momentum to Visual Reflection in Multimodal On-Policy Distillation.** arXiv:2609.16459, 2026.

- 公开入口：https://arxiv.org/abs/2609.16459
- 主要方法依据：讨论中提供的 `opd_aha_arxiv.pdf`。不假设所有公开修订版与该文件完全一致。
- real/null contrast：§3.1，印刷页 4。
- 原始 KL-regularized target：§3.2，Eq. (5)–(10)，印刷页 5。
- reflection 相对优势与恢复：§4.2–4.3，印刷页 6–7。
- 原始 target 的理论：Appendix A，印刷页 16。
- evidence-specific controls：Appendix B.2，印刷页 17–18。
- JSD、top-100 + tail、8 rollouts 等配置：Appendix C.2 / Table 4，印刷页 19。

### R1. 熵本身作为 RL 奖励

Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, Hao Peng. **The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning.** NeurIPS 2025, Main Conference.

官方论文集：https://proceedings.neurips.cc/paper_files/paper/2025/hash/9a14cf2ee36cf2524f54d8f196a6734c-Abstract-Conference.html

借鉴 EM-RL 的负熵奖励接口，不把负熵结果当作正熵有效性的证据。

### R2. 熵控制的条件依赖

Jingchu Gai et al. **Demystifying Entropy Control in LLM RL Training: Theoretical Analysis and Dynamic Scheduling.** ICML 2026, PMLR 306.

官方论文集：https://proceedings.mlr.press/v306/gai26b.html

### R3. 轨迹熵约束

Guojian Zhan et al. **Mind Your Entropy: From Maximum Entropy to Trajectory Entropy-Constrained RL.** ICML 2026, PMLR 306.

官方论文集：https://proceedings.mlr.press/v306/zhan26d.html

### R4. 在 policy update 中控制熵

Aleksei Petrenko et al. **Entropy-preserving reinforcement learning.** ICLR 2026.

官方论文集：https://proceedings.iclr.cc/paper_files/paper/2026/hash/124dde499d62b58e97e42a45b26d7369-Abstract-Conference.html

### R5. 熵引导的 reward shaping

Hongze Tan et al. **GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy.** ICML 2026, PMLR 306.

官方论文集：https://proceedings.mlr.press/v306/tan26e.html

### R6. 相对熵约束的分布优化

Jan Peters, Katharina Mülling, Yasemin Altun. **Relative Entropy Policy Search.** AAAI 2010, pp. 1607–1612. DOI: 10.1609/aaai.v24i1.7727.

官方入口：https://ojs.aaai.org/index.php/AAAI/article/view/7727

### R7. GRPO 基础

Zhihong Shao et al. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.** arXiv:2402.03300, 2024, §4.1.

原文：https://arxiv.org/abs/2402.03300

---

## 结尾：应当如何表述本计划的贡献候选

可以表述为：**在保持大部分视觉偏好增益和原始 teacher 邻域的前提下，优化蒸馏 target 的不确定性；再用 group-relative entropy reward 研究这种局部自由度能否转化为有效的推理路径探索。**

当前不能表述为：已经证明 entropy 导致 reflection、attention 导致视觉纠错、原始 Aha 存在熵坍缩、该方案提高了 benchmark，或整个联合训练具有全局最优保证。

最终能否形成方法贡献，取决于它能否在任务效果、熵匹配温度对照、direct entropy baseline、成本与稳定性之间提供可复现且可解释的额外价值。
