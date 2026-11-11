# 算法设计：以 GRPO 为主体的熵奖励与 OPD-Aha 细粒度视觉监督

> **状态：独立研究设计，尚未接入训练，尚无模型训练结果。**
>
> 本文记录本轮讨论确定的新主线：**保持 GRPO 的采样、组内比较和 clipped policy update 为主体，用学生预测熵替代答案正确性奖励，再用 OPD-Aha 提供 token 级视觉监督。**
>
> 本文是新增文档，不覆盖或修改 [`entropy_grpo_aha_research_plan_zh.md`](entropy_grpo_aha_research_plan_zh.md)。旧文档中的约束最大熵 target 保留为后续消融，而不是当前算法的前提。
>
> 编写时检查的仓库基线：`dfd487b55fac456070e9f04312ec63e2b1abe642`。本次提交仅新增本文；不修改 README、原研究文档、默认配置或训练代码。

## 目录

1. [设计定位与旧方案的区别](#s01)
2. [已有观察、研究假设与训练边界](#s02)
3. [符号与基本约定](#s03)
4. [用学生熵构造 rollout reward](#s04)
5. [组内优势与 GRPO 更新](#s05)
6. [OPD-Aha 的局部视觉目标](#s06)
7. [版本一：GRPO 主体与独立 JSD 监督](#s07)
8. [版本二：Aha 修正 token advantage](#s08)
9. [数值例子：高熵轨迹中的错误 token 如何被抑制](#s09)
10. [完整训练流程与伪代码](#s10)
11. [数据接口、概率支持与实现注意事项](#s11)
12. [实验矩阵与执行顺序](#s12)
13. [指标与日志](#s13)
14. [风险、失败判据与方法边界](#s14)
15. [测试和验收清单](#s15)
16. [后续设计空间与来源说明](#s16)

---

<a id="s01"></a>
## 1. 设计定位与旧方案的区别

### 1.1 当前方案的主次关系

当前设计不是“先构造最大熵 teacher，再附加一点 RL”，而是：

```text
同一个图像与问题
        │
        ▼
学生行为策略生成 G 条 rollout
        │
        ├── 学生预测熵 → rollout reward → group-relative advantage
        │                                      │
        │                                      ▼
        │                              GRPO clipped policy update
        │                                      ▲
        └── 同一批 prefix → teacher real/null → Aha 局部视觉监督
                                               │
                         版本一：独立 JSD loss ──┘
                         版本二：token advantage 修正
```

两个模块的分工是：

| 模块 | 信号粒度 | 作用 | 不能保证什么 |
| --- | --- | --- | --- |
| Entropy-GRPO | rollout / group | 根据熵奖励改变生成路径的概率 | 高熵不保证正确、有效反思或视觉依赖 |
| OPD-Aha | 每个学生 prefix 的 token distribution | 给局部更新提供视觉偏好方向 | 局部视觉支持不等于长期任务价值 |

**核心研究问题：Aha 的局部视觉方向，能否使 entropy-driven trajectory learning 形成有效探索，而不是仅增加犹豫或随机性？**

### 1.2 与旧文档的关系

| 项目 | 旧文档的重点 | 本文主线 |
| --- | --- | --- |
| 算法组织 | 约束最大熵 target reconstruction | GRPO 采样—奖励—优势—更新循环 |
| 熵的主要角色 | 优化 teacher-derived target 的形状 | 作为每条学生 rollout 的奖励 |
| 首版 target | 约束优化后的 `q*` | 原始 Aha target `q^A` |
| Aha 的职责 | 保留视觉收益并增加 target 熵 | 补充轨迹级奖励缺少的局部视觉监督 |
| 核心实现 | target 求解器与联合 loss | entropy reward 管线与视觉监督融合 |
| `q*` 的地位 | 核心组件 | 后续可选消融，不是必要条件 |

旧方案与新方案不互相覆盖。本文不否定旧方案的局部数学推导，但调整了方法的研究优先级。

### 1.3 “GRPO 为主体”的操作定义

主体性体现在使用 GRPO 的 rollout group、reward、advantage、behavior-policy ratio 和 clipped update，而不是把 GRPO 写在 loss 的第一项。

不预先规定 RL 梯度必须比 Aha 梯度大。两路实际贡献应通过梯度范数、夹角和局部符号冲突诊断，不能仅比较两个 loss 的数值大小。

---

<a id="s02"></a>
## 2. 已有观察、研究假设与训练边界

### 2.1 讨论中的已有观察

研究者报告：相对基础模型，OPD-Aha 产生更多 reflection tokens；在 reflection 后，学生预测熵明显提高，对 visual tokens 的 attention 也提高。

这些是本轮设计的动机。本文未重新读取这些实验的原始日志，因此不新增具体数值、显著性或因果结论。原稿中的 continuation support / visual-evidence support 与新增 entropy / attention 实验不是同一指标，不能混用。

还需区分：

- “有效反思之后出现高熵”与“直接奖励高熵能够诱发有效反思”不是同一命题。
- 不主张原始 OPD-Aha 已发生 target entropy collapse。
- 先前 correctness reward 与 OPD 的负面实验，不等于证明所有 reward–distillation 组合无效。

### 2.2 待验证假设

**H1：**以学生熵构造的组内相对奖励，能改变模型的轨迹选择，但单独使用可能增加无效犹豫。

**H2：**加入 Aha 的局部视觉监督，能够缓解这种无方向探索，提高最终任务表现或反思后的恢复质量。

**H3：**把 Aha 信号接入 token advantage，与简单并列 JSD loss 相比，可能产生不同的信用分配和稳定性效果。

H1–H3 均不是已验证结果。特别是 H3 的两个实现不应提前宣称严格等价或进阶版必然更优。

### 2.3 首版保持不变的边界

- 学生使用原始图像；teacher 仍使用既有 privileged real view 与 visual null。
- 不增加第三个 teacher view，不引入分割、attention 奖励或新的外部 reward model。
- 不使用答案正确性作为训练 reward，也不按正确性筛选训练 rollout。
- 不奖励 `wait`、`actually` 等词的出现次数。
- accuracy 和 correctness 可以用于独立诊断、验证与最终评估，但不得泄漏进入训练 reward、group 筛选或 token 权重。
- 首版不增加 branching、环境回滚或反思事件检测器。
- 使用既有 Aha target；不要求实现旧文档的最大熵求解器。

---

<a id="s03"></a>
## 3. 符号与基本约定

| 符号 | 含义 |
| --- | --- |
| `x, I` | 问题与学生可见的原始图像 |
| `I+`, `I0` | teacher privileged real view 与 visual null |
| `π_b` | 本批 rollout 的冻结行为策略 |
| `π_θ` | 正在更新的学生策略 |
| `y_i` | 同一输入的第 i 条学生 rollout |
| `h_{i,t}=(x,y_{i,<t})` | 第 t 个预测位置的文本 prefix |
| `M_{i,t}` | 有效学生 response token mask |
| `T_i=Σ_t M_{i,t}` | 有效 response 长度 |
| `p^+_{i,t}, p^0_{i,t}` | 同一冻结 teacher 在两种视觉条件下的分布 |
| `u_{i,t}` | real–null log-probability difference |
| `q^A_{i,t}` | 原始 Aha 重构分布 |
| `H_{i,t}` | 行为策略在该 prefix 的预测熵 |
| `R_i^H, A_i^H` | entropy rollout reward 与组内相对优势 |
| `B_{i,t}^{JSD}` | 从局部 JSD 推导的蒸馏更新权重 |
| `λ_Aha` | Aha 相对于 RL 的权重 |
| `ε_clip` | PPO/GRPO ratio clipping 范围 |

全文使用自然对数，熵、KL、JSD 的单位均为 nats。JSD 采用等权的标准定义：

$$
\operatorname{JSD}(p,q)
=\frac12D_{\mathrm{KL}}(p\Vert m)
+\frac12D_{\mathrm{KL}}(q\Vert m),
\qquad m=\frac{p+q}{2}.
$$

若实现使用加权 JSD、不同系数或其他 divergence，版本二的公式必须重新对应，不能直接套用。

为说明 joint objective，正文采用“每条回答内平均有效 token，再平均回答”的聚合：

$$
\langle f\rangle
=\frac1G\sum_{i=1}^{G}\frac1{T_i}\sum_tM_{i,t}f_{i,t}.
$$

这是一项明确的设计约定，不意味着当前仓库必然使用此聚合。实现时应记录实际 aggregation，并在消融间保持一致；不同长度归一化会改变更新权重。

---

<a id="s04"></a>
## 4. 用学生熵构造 rollout reward

### 4.1 熵的定义

在学生 rollout 的每个预测位置，计算：

$$
H_{i,t}
=-\sum_{v\in\mathcal V}
\pi_b(v\mid I,h_{i,t})\log\pi_b(v\mid I,h_{i,t}).
$$

这是整个下一词分布的 Shannon entropy，不是：

- sampled token 的 surprisal `−log π_b(y_{i,t})`；
- reflection 词频；
- visual attention entropy；
- 同组最终答案类别的多样性熵。

在完全相同的 prefix 下，下一词分布熵相同，不会因为这次采到不同 token 而改变。不同 action 的差异主要通过后续访问的状态体现。

### 4.2 原始熵奖励

首个最直接的版本：

$$
\boxed{
R_i^{H,\mathrm{raw}}
=\frac1{T_i}\sum_tM_{i,t}H_{i,t}.
}
$$

它让同一问题下平均预测熵较高的 rollout 获得较高 reward。它并不评价答案是否正确。

使用平均值而非总和，可以移除“多生成一个 token 就直接累加一份正熵”的简单激励，但不保证完全消除长度偏差或重复行为。

### 4.3 有上限的熵奖励

作为并行对照：

$$
\boxed{
R_i^{H,\mathrm{cap}}
=\frac1{T_i}\sum_tM_{i,t}\min(H_{i,t},H_{\max}).
}
$$

`H_max` 在训练前使用训练集上的 calibration rollout 确定，并在对应实验中固定。记录 calibration 数据、熵计算口径和阈值规则，不使用测试集或 correctness 调参。

首版不让 `H_max` 依赖 `q*`，因此不需要最大熵 target 求解器。阈值不是已知最优值，应作为实验因素。

上限只让超过阈值的熵不再额外加分，不能自动区分“有效反思”和“无意义犹豫”。

### 4.4 冻结奖励，而不是直接对熵反向传播

一次 rollout–learner 更新周期内，`H`、`R^H` 和 `A^H` 都由冻结的行为策略计算并 detach。

GRPO 通过这些数值调整生成路径概率，不直接对 reward 内的 entropy 求导。这与直接加 `−αH(π_θ)` 的 entropy regularization 不同；后者作为独立对照实验。

因此，不应将这个 surrogate 描述为完整的参数相关 entropy objective 的精确梯度。

---

<a id="s05"></a>
## 5. 组内优势与 GRPO 更新

### 5.1 Group 定义

一个 group 必须对应同一图像、同一问题和同一输入条件。不能只按相同文本问题、不同图片混组，也不能把相同 token index 当成相同推理状态。

计划沿用讨论中的 `G=8` 作为起点；正式运行应以 resolved config 为准。只有有效 rollout 的 group 才进行 reward 比较。

### 5.2 Advantage

定义组内均值和 population standard deviation：

$$
\bar R=\frac1G\sum_iR_i^H,
\qquad
s_R=\sqrt{\frac1G\sum_i(R_i^H-\bar R)^2}.
$$

稳定化的组内优势为：

$$
\boxed{
A_i^H=\frac{R_i^H-\bar R}{\max(s_R,\sigma_{\min})}.
}
$$

- `σ_min` 防止微小 reward 噪声被归一化放大。
- 组内 reward 完全相同，所有 `A_i^H=0`，但 Aha 分支仍然训练。
- `G<2`、缺失 entropy、NaN 或空 response 必须显式处理并记录，不能静默污染组均值。
- 缺失 entropy 不等于零 entropy。不要把无效记录当作正常低 reward。
- 只减均值、不除标准差可以作为消融，但必须单独命名，不能混淆数值尺度。

### 5.3 GRPO-style policy loss

$$
\rho_{i,t}(\theta)
=\frac{\pi_\theta(y_{i,t}\mid I,h_{i,t})}
{\pi_b(y_{i,t}\mid I,h_{i,t})}.
$$

$$
\mathcal L_{H\text{-}GRPO}
=-\left\langle
\min\left(
\rho_{i,t}A_i^H,
\operatorname{clip}(\rho_{i,t},1-\varepsilon_{clip},1+\varepsilon_{clip})A_i^H
\right)
\right\rangle.
$$

同一 rollout 内的 token 共享 `A_i^H`，这是本设计希望通过 Aha 补充的粗粒度之处。

“GRPO 没有细粒度信号”不是一般性定义；这里特指每条 rollout 使用一个标量 reward、再广播 trajectory advantage 的版本。

reference-policy KL penalty、直接 entropy coefficient 等额外项不在上述简化公式内。若代码本来启用它们，必须明确记录并在对照中固定；不能悄悄增加第二个 entropy 优化通道。

---

<a id="s06"></a>
## 6. OPD-Aha 的局部视觉目标

### 6.1 提取 real–null 视觉偏好

在同一条学生 rollout 的同一个 prefix：

$$
p^+_{i,t}(v)=p_\phi(v\mid I^+,h_{i,t}),
\qquad
p^0_{i,t}(v)=p_\phi(v\mid I^0,h_{i,t}),
$$

$$
u_{i,t}(v)=\log p^+_{i,t}(v)-\log p^0_{i,t}(v).
$$

`u` 是有正有负的 log-ratio，不是概率分布，也不是答案正确性标签。

### 6.2 原始 target reconstruction

$$
\boxed{
q^A_{i,t}(v)
=\frac{p^+_{i,t}(v)e^{\beta u_{i,t}(v)}}
{\sum_wp^+_{i,t}(w)e^{\beta u_{i,t}(w)}}.
}
$$

实现使用 log-space softmax，避免直接指数导致溢出。teacher 参数冻结，target 在当前 learner update 内固定。

这一部分来自原始 OPD-Aha 的 real–null contrast 与 target reconstruction。首版不改变 `β` 的含义，也不需要引入旧文档的 `κ`、视觉收益保留比例或最大熵对偶乘子。

### 6.3 为什么首版不修改成最大熵 target

当前分工是“熵决定轨迹级探索压力，视觉偏好决定局部方向”。让两路都最大化熵可能重复施压，也会使实验难以定位收益来源。

因此先保留 `q^A`；需要改善的是 Aha 与 GRPO 的结合方式。`q^A → q*` 放在算法主线有效之后再消融。

---

<a id="s07"></a>
## 7. 版本一：GRPO 主体与独立 JSD 监督

### 7.1 完整目标

$$
\mathcal L_{Aha}
=\left\langle
\operatorname{JSD}\left(
\pi_\theta(\cdot\mid I,h_{i,t}),
\operatorname{sg}[q^A_{i,t}]
\right)
\right\rangle.
$$

$$
\boxed{
\mathcal L_1
=\mathcal L_{H\text{-}GRPO}
+\lambda_{Aha}\mathcal L_{Aha}.
}
$$

这是首个应实现的完整算法，不是先训练 Aha 再训练 GRPO。两路在同一批学生 rollout 上联合更新同一个学生。

### 7.2 两路信号如何配合

GRPO 对一条高熵轨迹提供正 advantage；Aha 在这条轨迹的某个 prefix 上仍可以压低视觉上不受支持的 token。

反过来，一条低熵轨迹整体受负 advantage，也不代表其中每个视觉上合理的局部行为都必须被净压低。

实际净梯度取决于两路梯度的大小和方向，不保证 Aha 自动战胜所有错误强化。`λ_Aha` 应依据梯度诊断和验证集结果选择，不能只依据 loss 标量大小。

### 7.3 梯度边界

- `q^A` detach，teacher 不更新。
- reward、组内统计量与 advantage detach。
- `π_b` 的概率和采样轨迹固定。
- 当前学生 `π_θ` 保留梯度。
- JSD 中 `m=(π_θ+q^A)/2` 依赖当前学生，**不能为了冻结 target 而把整个 m 也 detach**。
- 不对离散采样操作或固定 prefix 直接反向传播。

### 7.4 优点与限制

优点是保留原有分布级蒸馏信号，便于与原始 Aha 对齐；无须把所有监督都转换为 sampled-token reward。

限制是两路更新可能冲突，且 trajectory reward 仍然广播到整条回答。版本二研究更紧密的融合方式，而不是预先假设版本一不足。

---

<a id="s08"></a>
## 8. 版本二：Aha 修正 token advantage

### 8.1 目标

让每个 token 的最终更新权重包含：

$$
\boxed{
\widetilde A_{i,t}
=A_i^H+\lambda_{Aha}B_{i,t}^{JSD}.
}
$$

其中 `A_i^H` 是整条轨迹共享的相对熵奖励，`B_{i,t}^{JSD}` 是当前位置的视觉蒸馏修正。

`B` 是局部更新权重，不是已验证的 action-value estimate 或真实长期 advantage。

### 8.2 JSD-compatible 的局部权重

在固定 prefix，简写行为分布为 `p=π_b`，固定 Aha target 为 `q=q^A`。定义：

$$
\boxed{
B^{JSD}(v)
=\frac12\log\frac{p(v)+q(v)}{2p(v)}.
}
$$

对采样出的 token 取值：

$$
B_{i,t}^{JSD}
=\frac12\log
\frac{\pi_b(y_{i,t}\mid I,h_{i,t})+q^A_{i,t}(y_{i,t})}
{2\pi_b(y_{i,t}\mid I,h_{i,t})}.
$$

解释：

| 条件 | B 的符号 | 局部倾向 |
| --- | --- | --- |
| `q(v)>p(v)` | 正 | 增加该 token 的概率 |
| `q(v)<p(v)` | 负 | 减少该 token 的概率 |
| `q(v)=p(v)` | 零 | 不增加蒸馏修正 |

不能直接用 `log q−log p` 替换后仍称其等价于 JSD。后者与局部 reverse-KL 蒸馏梯度的联系不同。

### 8.3 推导与准确的等价范围

标准 JSD 的概率导数为：

$$
\frac{\partial\operatorname{JSD}(p,q)}{\partial p(v)}
=\frac12\log\frac{2p(v)}{p(v)+q(v)}.
$$

结合 `∇p(v)=p(v)∇log p(v)`，在当前策略 `θ=b`：

$$
\boxed{
-\nabla_\theta\operatorname{JSD}(\pi_\theta,q)
\big|_{\theta=b}
=
\mathbb E_{v\sim\pi_b}
\left[B^{JSD}(v)\nabla_\theta\log\pi_\theta(v)\big|_{\theta=b}\right].
}
$$

这个等式只覆盖：

1. 固定 prefix，固定且 detach 的 q；
2. 标准等权 JSD；
3. 相同的概率支持与一致的采样分布；
4. 在行为策略处比较梯度；
5. 不含 clipping、额外归一化或权重裁剪的局部期望更新。

它**不**证明：完整 rollout 目标的精确梯度等价、多次更新后等价、实际有限样本梯度相同，或两路 clipped loss 与一次融合 clipping 等价。

特别是 prefix 的访问分布在这里固定，没有对未来状态访问求导。局部蒸馏权重不能自动变成长期因果信用。

### 8.4 统一 clipped objective

$$
\boxed{
\mathcal L_2
=-\left\langle
\min\left(
\rho_{i,t}\widetilde A_{i,t},
\operatorname{clip}(\rho_{i,t},1-\varepsilon_{clip},1+\varepsilon_{clip})
\widetilde A_{i,t}
\right)
\right\rangle.
}
$$

在这个版本中，先通过 `B` 体现蒸馏，不再默认额外加入同一份完整 JSD，以免重复计算。若测试 `L_2 + η L_Aha`，必须作为单独的三通道变体记录。

融合后正负号可能改变，clipping 的激活区域也会改变。因而版本二本身是一个新的 surrogate，不应描述成对版本一完全等价的实现优化。

### 8.5 数值稳定性与方差

使用 log-space 计算：

$$
B=\tfrac12\left[
\operatorname{logaddexp}(\log p,\log q)-\log 2-\log p
\right].
$$

在正概率下，`B ≥ −(log 2)/2`，但正方向不具有固定上界：若学生对 teacher-preferred token 的概率极小，正 B 可以很大。

裁剪 B、重新标准化 B 或做 token gate 可能有助于稳定，但会改变上述精确梯度关系，应作为显式变体。首版至少记录 B 分位数、极值和采样方差。

分布级 JSD 会直接监督可用 support 上的许多候选；采样式 B 在单个位置只有有限 action 样本，可能很少覆盖 student 低概率的纠错 token。这是版本二相对版本一的关键取舍。

---

<a id="s09"></a>
## 9. 数值例子：高熵轨迹中的错误 token 如何被抑制

以下全部是教学数值，不是实验结果，也不表示观察到了特定模型行为。

同一问题下有三条轨迹：

| 轨迹 | 行为示意 | 熵 reward |
| --- | --- | ---: |
| A | 很快确定答案，不再检查 | 0.4 |
| B | 重新检查后形成结论 | 0.8 |
| C | 较多犹豫，也包含错误判断 | 0.9 |

均值为 0.7。为展示信号相加，**本小例子只减均值，暂不除标准差**：

$$
A_A^H=-0.3,\qquad A_B^H=0.1,\qquad A_C^H=0.2.
$$

现在看 C 内的三个不同 prefix：

| 局部行为 | 学生对已采样 token 的 p | Aha target 对它的 q | B |
| --- | ---: | ---: | ---: |
| 继续错误判断 | 0.70 | 0.10 | −0.279808 |
| 开始重新检查 | 0.10 | 0.30 | +0.346574 |
| 给出视觉支持的判断 | 0.20 | 0.60 | +0.346574 |

每一行是不同位置，不是同一概率向量，因此各行的 p 或 q 无须相加为 1。

取教学用 `λ_Aha=1`，最终权重为：

| 局部行为 | 轨迹基础优势 | Aha 修正 | 最终权重 |
| --- | ---: | ---: | ---: |
| 错误延续 | +0.200000 | −0.279808 | **−0.079808** |
| 重新检查 | +0.200000 | +0.346574 | **+0.546574** |
| 视觉支持的判断 | +0.200000 | +0.346574 | **+0.546574** |

所以，整条 C 因高熵获正反馈，不意味着其中所有 token 都必须被强化。Aha 可以对局部行为改变净更新方向。

**不要把这个教学权重当作标准化 GRPO 的直接输出。**若使用第 5 节的 population 标准差且未触发 `σ_min`，三条 advantage 约为：

$$
[-1.388730,\ 0.462910,\ 0.925820].
$$

此时 C 的基础值不再是 0.2；使用同样 `λ_Aha=1` 并不足以翻转示例错误 token 的正号。`λ_Aha` 必须根据真实尺度校准，不能用上表暗示错误 token 总能被压制。

另一个限制是 teacher 可能给无效反思较高概率，因此 B 的符号也不能直接当作 ground-truth token quality。

---

<a id="s10"></a>
## 10. 完整训练流程与伪代码

### 10.1 一个训练周期

1. 保存当前学生为冻结的 behavior snapshot，按相同输入采样 G 条 rollout。
2. 记录每个有效 response token 的行为 log-probability、预测熵、prefix 和 group ID。
3. 用 raw 或 capped entropy 计算 rollout reward，在同一输入组内计算 advantage。
4. teacher 在相同学生 prefix 上分别处理 real/null visual view，得到 q^A。
5. 版本一计算 GRPO loss 与独立 JSD；版本二构造 B 和融合 advantage。
6. 使用当前学生重新前向，联合反向传播并更新学生；teacher 不更新。
7. 当前周期内冻结 behavior、reward、advantage、target 和概率 support。
8. 下一个周期用更新后的学生重新采样；记录训练指标，并定期做独立评估。

### 10.2 版本一伪代码

以下是接口层伪代码，不是仓库已有可运行函数或配置名称。

```python
# Outer loop: freeze the policy that actually generates this batch.
behavior = frozen_snapshot(student)
rollouts = sample_grouped(behavior, inputs, group_size=G)

with no_grad():
    # Must match the documented behavior distribution and token mask.
    old_logp = behavior_log_probs(behavior, rollouts)
    entropy = behavior_predictive_entropy(behavior, rollouts)

    entropy_score = entropy
    if reward_mode == "capped":
        entropy_score = minimum(entropy_score, H_max)

    reward = masked_response_mean(entropy_score, rollouts.response_mask)
    advantage = group_relative_advantage(
        reward, rollouts.group_id,
        std_floor=sigma_min, population_std=True,
    )

    logp_real = teacher_log_probs(teacher, rollouts, real_view)
    logp_null = teacher_log_probs(teacher, rollouts, null_view)
    u = logp_real - logp_null
    q_aha = softmax(logp_real + beta * u)

for minibatch in learner_batches(rollouts):
    current_logp, current_distribution = student_forward(student, minibatch)
    ratio = exp(current_logp - old_logp[minibatch])

    loss_grpo = clipped_surrogate(
        ratio, advantage[minibatch], response_mask[minibatch], clip_epsilon,
    )
    loss_aha = masked_jsd(
        current_distribution, q_aha[minibatch].detach(),
        response_mask[minibatch],
    )

    loss = loss_grpo + lambda_aha * loss_aha
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 10.3 版本二的差异

在无梯度区间内，针对每个已采样 token 构造：

```python
# p_b and q_aha must have the same mathematically defined support.
B = 0.5 * (
    logaddexp(old_logp, sampled_log_q_aha) - log(2.0) - old_logp
)
combined_advantage = advantage[:, None] + lambda_aha * B
```

然后用 `combined_advantage` 替换 GRPO surrogate 的 trajectory advantage。默认不再附加同一份完整 JSD。

“先冻结 B、再多次更新”是一个 policy surrogate；越偏离行为策略，越不能依赖第 8 节的局部梯度等式。需要记录 inner epochs、ratio 和 clip fraction。

---

<a id="s11"></a>
## 11. 数据接口、概率支持与实现注意事项

### 11.1 建议保存的字段

| 字段 | 典型维度 | 用途 |
| --- | --- | --- |
| `group_id` | `[N]` | 稳定的图像–问题分组 |
| `response_ids` | `[N,T]` | 学生采样 token |
| `response_mask` | `[N,T]` | 去除 padding、非学生生成内容 |
| `old_logp` | `[N,T]` | 实际 behavior 的 sampled-token log-probability |
| `behavior_entropy` | `[N,T]` | 冻结的逐位置学生熵 |
| `entropy_reward` | `[N]` | raw/capped rollout reward |
| `entropy_advantage` | `[N]` | 组内相对优势 |
| `support_ids` | `[N,T,K]` 或全词表 | 对齐分布的 token 身份 |
| `logp_teacher_real/null` | `[N,T,K(+tail)]` | real–null 对比与 target |
| `q_aha` | 同上 | 当前更新固定的 target |
| `aha_local_weight` | `[N,T]` | 版本二 B，仅在定义清楚后启用 |
| `behavior_version` | batch / rollout 元数据 | 防止 stale policy 混用 |

这些是建议接口，不表示已加入仓库。

### 11.2 Full-vocabulary entropy 与 top-k + tail

上传稿的配置使用 student top-100 + one tail bucket。该配置不是对当前任意 launcher 的保证；运行前需导出 resolved config。

粗化分布与完整分布的熵满足：

$$
H_{full}
=H_{bucketed}+p_{tail}H(p(\cdot\mid tail)).
$$

因此两种 entropy 不是同一指标。首版优先记录 full-vocabulary behavior entropy；若为资源原因使用 bucket entropy，必须明确命名、固定支持规则、报告 tail mass，并单独校准 `H_max`。

不应把 top-k 的已保留概率重新归一化后，当成完整词表熵。不同表示下的 entropy threshold 不能直接互换。

### 11.3 版本二的 tail 风险

不能把 tail bucket 的总概率当作某个具体 sampled token 的概率，塞入第 8 节的单 token 公式。

版本二需二选一：

- 使用一致全词表下的学生概率与 Aha target 概率；或
- 明确在 coarse categories 上定义 JSD，重新对应 bucket 与 sampled-token 梯度，并通过专门单元测试验证。

第二种不是原则上不可能，但不能跳过推导直接声称与全词表版本等价。版本一可以先保留现有粗化 JSD，版本二等待支持一致性验证。

在 inner updates 中固定支持集合；重新选 top-k 会引入额外目标变化，破坏简单的局部比较。

### 11.4 Sampling temperature / top-p

数学推导假设采样分布、`old_logp`、当前 policy ratio 的定义一致。若 rollout 使用 temperature、top-p 或其他截断，必须明确行为策略到底是哪一个分布。

不能用变换后的分布采样，却无说明地用未变换概率宣称 on-policy 无偏恒等式。最小数学测试使用同一个正概率 categorical 分布；真实训练的 correction 和支持问题单独审计。

### 11.5 Mask、长度与分布式分组

- response mask 是否包含 EOS、是否排除格式 token，应预先固定；不要根据 correctness 决定 mask。
- 截断回答和无效回答要单独统计。不能将其处理成一种容易获取额外 reward 的漏洞。
- 分布式 minibatch 可能打散 group；组内统计应在完整 group 上完成再切分，不可误用当前 microbatch 的均值。
- teacher target、学生分布和 sampled token 必须共享正确的 causal shift，避免 next-token 对齐差一位。

---

<a id="s12"></a>
## 12. 实验矩阵与执行顺序

### 12.1 核心对照

| 编号 | 方法 | 检验问题 |
| --- | --- | --- |
| E0 | 原始 OPD-Aha | 视觉蒸馏基线 |
| E1 | Entropy-GRPO，raw reward，无 Aha | 单独熵奖励产生什么行为？ |
| E2 | Entropy-GRPO，capped reward，无 Aha | 上限是否减少无效增熵？ |
| E3 | Entropy-GRPO + 原始 Aha JSD（版本一） | 轨迹奖励与局部视觉监督是否互补？ |
| E4 | 融合 JSD-derived token weight（版本二） | 优势层融合是否优于并列 loss？ |
| E5 | 原始 Aha + 直接学生熵正则 | 是否只需局部分布调节，而非 group RL？ |
| E6 | reward 为常数的 GRPO 分支 + Aha | group signal 消失后是否正确退化？ |

E3、E4 至少与相同 raw/capped reward 设定比较；不能同时改奖励和融合方式后把收益全归因于一个模块。

### 12.2 实验匹配

匹配学生起点、数据、G、rollout token budget、teacher view、β、长度上限、优化步数、有效 token 聚合与评估方式。

保留原始基线的全部配置快照。若先用 Aha checkpoint 做小规模 continuation pilot，再从 base 训练，二者分别报告，不混写成同一实验。

RL-only 方法没有 teacher 成本，因此除 matched rollout budget 外，还需报告实际 teacher FLOPs / 调用量和 wall-clock，避免虚构等算力比较。

随机种子、评价样本和判断标准在实验开始前固定。单个 seed 的上升只能作为 pilot 依据，正式判断需重复运行。

### 12.3 执行阶段

**阶段 A：离线信号审计。**在固定 rollout 上检查 entropy 口径、组内方差、长度相关性、Aha target 方向及两路局部符号冲突。correctness 可用于分析，但不参与训练信号。

**阶段 B：实现与 smoke test。**先实现版本一，并检查关闭 Aha / 关闭 RL 时的退化行为。无需最大熵 target solver。

**阶段 C：小预算训练。**运行 E0–E3 与 E5，判断是不是只有 entropy reward 在上升，而真实表现不升或重复恶化。

**阶段 D：统一 advantage。**完成梯度和 support 测试后，再运行 E4，比较有限采样方差、clipping 和纠错覆盖。

**阶段 E：target 消融。**只有主体框架值得保留时，再比较原始 q^A、temperature smoothing 和旧方案 q*。

---

<a id="s13"></a>
## 13. 指标与日志

### 13.1 训练信号

- raw/capped reward 均值、标准差、分位数；达到上限的 token 比例。
- 组内 reward 标准差、零方差 group 比例、`σ_min` 激活比例。
- full/bucket entropy、tail mass、按位置与长度分层的 entropy。
- JSD、Aha real–null signal 大小，以及 B 的分位数和极值。
- token advantage 的正负比例、Aha 修正前后的符号翻转比例。
- ratio、clip fraction、inner update 数量和策略偏移。

### 13.2 两路梯度的关系

版本一在固定频率的诊断 batch 上分别记录：

$$
\|g_{RL}\|,\quad\|\lambda_{Aha}g_{Aha}\|,\quad
\frac{g_{RL}^{\top}g_{Aha}}
{\|g_{RL}\|\|g_{Aha}\|}.
$$

这是诊断，不要求每个训练 step 都做昂贵的独立 backward。梯度长期冲突、某一路始终占绝对主导，均需单独解释。

### 13.3 行为与任务结果

- 最终 accuracy / 原有任务指标；固定采样预算下的表现。
- 反思频率、反思前后纠错率、先对后错率。
- 重复率、平均长度、长度分布与触顶率。
- 相同 prefix 条件下的策略熵，与整条轨迹平均熵分开记录。
- 同组轨迹的实际多样性，不用一个平均 entropy 数字替代。
- 视觉干预敏感性、已有 visual support 指标；attention 仅作辅助诊断。

反思词命中只是粗粒度事件标记，不是“有效反思”的 ground truth。已有词汇压制后仍出现替代表达的观察，意味着不能只数固定单词。

---

<a id="s14"></a>
## 14. 风险、失败判据与方法边界

### 14.1 Reward hacking

模型可能通过持续不确定、冗长表达、罕见词或固定套路通向高熵状态来提高 reward。capping 和长度平均不能彻底解决。

如果 reward / entropy 上升，而 accuracy、恢复率下降，或重复与触顶明显恶化，应记录为代理目标失配，而不是方法已改善探索。

### 14.2 局部方向不等于长期价值

Aha 的 u 来自 teacher real–null contrast，并不保证每个正 u token 都正确。JSD-derived B 也不是经过环境验证的 Q-value。

因此，不应声称本算法已经解决 long-horizon credit assignment；它提供的是相对于统一 trajectory advantage 的局部修正。

### 14.3 融合尺度与错误强化

标准化后的 A_i^H 可能远大于 B。此时即使 teacher 不支持某个 token，净权重仍可为正。

反之，Aha 过强可能使 entropy-GRPO 分支几乎不起作用。必须报告 λ 与真实梯度尺度，而不是用公式结构代替证据。

### 14.4 采样式蒸馏的覆盖不足

版本二只能使用有限采样 action；低概率但重要的修正 token 可能很少被直接监督。版本一的分布级 JSD 是必要对照，不应为统一形式而放弃它。

### 14.5 停止或回退判据

以下应触发审计、降权或回退，而非继续仅追求 reward：

- reward 非有限、概率错位、group 混组或 entropy 缺失；
- entropy reward 提升伴随持续任务退化；
- 重复率、触顶率或长度持续异常上升；
- B 极端值和 clipping 占比异常，有限样本梯度非常不稳定；
- 版本二无法通过固定 prefix 的梯度测试；
- 额外正确性信息意外进入 reward 或选择流程。

具体数值阈值应依据原始基线波动预先设定，不在本文伪造通用阈值。

---

<a id="s15"></a>
## 15. 测试和验收清单

### 15.1 数学与单元测试

- 均匀 K 类分布：entropy 等于 log K；尖锐分布：entropy 接近零。
- 相同 reward 的 group：RL advantage 全零；Aha 更新仍存在。
- group permutation 不改变各 rollout 对应 advantage；padding 不影响 reward。
- p=q 时，JSD 和 B 均为零。
- q(v)>p(v) 时 B(v)>0，反之为负。
- 枚举小词表，比较完整 JSD 梯度与 B 的期望 policy-gradient surrogate。
- 固定 support 下测试 full / bucket 的熵关系。
- teacher、q、reward、advantage、behavior probability 无参数梯度；当前学生有梯度。
- λ_Aha=0 退化为 entropy-GRPO；关闭 RL 分支后，版本一退化为既有 Aha loss。

### 15.2 已完成的局部公式检查

编写本文时，在一个四类分布上用 float64 自动微分检查：

```text
p = [0.70, 0.10, 0.18, 0.02]
q = [0.1953, 0.1014, 0.7011, 0.0022]
```

比较 `∇JSD(softmax(z),q)` 与 `∇[−Σ_v p_b(v) B(v) log π_θ(v)]`，在 `θ=b` 的最大绝对梯度差约为 `8.67e-18`。

这只验证一个小型、枚举全词表、固定 prefix 的数学实现检查；不是已接入仓库的测试，不是采样方差实验，也不是模型训练结果。

### 15.3 实现验收

- 所有新增功能通过独立实验开关启用，默认训练行为保持不变。
- 明确保存 behavior snapshot 和 policy version，不能 inner step 后重算 reward 冒充旧策略 reward。
- 记录真实采样分布、temperature、top-p 与 old log-probability 的语义。
- 不在 reward、mask、group 选择或 B 中读取 correctness。
- 不提交模型权重、数据、日志、缓存或机器私有路径。
- 配置项名称必须在真正实现后才作为可运行命令发布；本文不声称存在新增 CLI。

---

<a id="s16"></a>
## 16. 后续设计空间与来源说明

### 16.1 当前建议的开发优先级

**首先实现版本一：**Entropy-GRPO + 原始 Aha JSD。

**其次研究版本二：**entropy trajectory advantage + JSD-derived local weight，经统一 clipping 更新。

**最后研究 target 变化：**temperature smoothing 与旧文档的约束最大熵 q*。

不在首版同时引入动态 β、reflection gating、额外奖励模型和多种归一化，否则难以判断主假设是否成立。

### 16.2 仍然开放的问题

1. 正熵奖励是否能给当前学生带来超出直接熵正则的价值？
2. raw 与 capped reward 哪个更稳健？阈值如何随训练变化是否需要另行研究？
3. Aha 对 high-entropy but unproductive continuation 的监督是否足够可靠？
4. sampled-token 融合带来的方差，是否抵消细粒度 clipping 的潜在收益？
5. outcome correctness 不参与训练时，最终任务方向是否足够明确？
6. 什么情况下确有必要修改 q^A，而不只是修改其与 RL 的融合方式？

### 16.3 来源与归属

本文以本轮已经讨论的设计为依据，不把新提案写成已有论文结论。

- **用户报告的观察：**reflection 增多、反思后 entropy 与 visual attention 上升；本文仅作为研究动机记录，没有新增实验证据。
- **原始 OPD-Aha：**real–null log-ratio、指数倾斜 target、蒸馏流程来自上传稿 §3.1–3.2，Eq. (4)–(10)；JSD 与 top-100 + tail、每题 8 条 rollout 来自上传稿 Appendix C.2 / Table 4。论文入口：[OPD-Aha](https://arxiv.org/abs/2609.16459)。
- **GRPO 参考入口：**[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)。本文借用 group-relative advantage 和 clipped surrogate，不声称提出 GRPO。
- **熵奖励参考入口：**[The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning](https://arxiv.org/abs/2505.15134)。此前讨论用于区分 entropy-as-reward 与直接熵正则；其中负熵方向不能直接作为本文正熵奖励有效性的证据。
- **本轮提出的研究设计：**正熵 reward 与 Aha 的分工、版本一联合训练方案，以及版本二的 JSD-derived token 权重融合。
- **旧方案独立保留：**[`entropy_grpo_aha_research_plan_zh.md`](entropy_grpo_aha_research_plan_zh.md)。其最大熵 target 不作为本文主算法的必备步骤。

## 结语

当前主线可以归纳为：

$$
\boxed{
\text{GRPO 用熵提供轨迹级探索压力}
\; + \;
\text{Aha 用视觉偏好提供 token 级纠错方向}.
}
$$

首个完整版本的目标是验证这两路信号是否真正互补；进阶版本研究它们在 advantage 层怎样融合。评价标准是任务表现、恢复质量与训练稳定性，而不是 entropy、reflection 词数或 reward 单独变大。
