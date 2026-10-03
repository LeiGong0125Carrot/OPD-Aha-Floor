# 保留完整 OPD-Aha 的历史—未来协同残余抑制

## 设计文档 v0.1

**整理日期：** 2026-10-03  
**方法状态：** 候选研究设计；未宣称已实现、完成 VLM 训练或优于 OPD-Aha。  
**对应方案：** Pure OPD / JSD；保留完整 Aha target，通过历史与已观察未来调节额外负向修正。  
**建议仓库路径：** `docs/negative_history/aha_history_future_residual_design.md`  
**结果依据：** `results_comparison_2026-10-02.md` 的固定提交快照 `ddd5b6e`。本文没有将该快照中的运行状态当作当前实时状态。

> **核心思路：完整 Aha 始终提供基础监督；history 提出额外抑制需求，future 只调节这份额外抑制。**
>
> 不先删除 Aha 的正向重构，不引入 PPO，不把 history 与 future 分别作为两个无条件放大器。新增模块无证据可用时，回退到完整 Aha，而不是 suppression-only 或普通 OPD。

本文区分三种内容：**已有记录**来自固定版本的结果和代码；**候选设计**来自本轮讨论；**数学推论**由给定公式推出。公式性质不等同于训练收益或因果机制的证明。

---

## 阅读目录

| 章节 | 内容 |
|---|---|
| 01 | 已有实验对设计的约束 |
| 02 | 方法范围与固定部分 |
| 03 | 符号、数据来源与局部视觉信号 |
| 04 | 历史模块：从累积量改为平均强度 |
| 05 | 未来模块：已观察后缀的冲突密度 |
| 06 | 主候选：history–future 调节 Aha 的残余修正 |
| 07 | Pairwise odds 与逐步数值例子 |
| 08 | 数学性质及其适用边界 |
| 09 | 独立 future-weighting 对照与最终 JSD |
| 10 | 训练流程、代码接入与边界处理 |
| 11 | 最小训练对照、日志与验收 |
| 12 | 风险、创新性与待决定事项 |
| 13 | 方法总结与来源登记 |

---

## 01｜已有实验对设计的约束

### 1.1 为什么从完整 Aha 出发

以下数值来自结果快照 [S1][s1]，均为文档中的 judge-based 指标；池化均值不是新的独立训练实验。

| 方法 | 改动 | TB 池化均值 | V* 池化均值 |
|---|---|---:|---:|
| A：完整 OPD-Aha | 完整正、负 visual tilt，β=4 | 49.22 | 92.84 |
| B：suppression-only | 删除正半边的额外放大 | 47.74 | 89.79 |
| C：B + 历史 | 累积历史提高负向强度 | 47.53 | 90.58 |
| D：B + 未来 | 后续冲突对 JSD 位置加权 | 47.04 | 88.83 |

按该快照预设的两次峰值规则，B 属于不可判读，C、D 判负。池化比较则显示 A 高于 B/C/D。两种口径应分别报告，不能将池化下降改写成 B 已通过峰值规则被明确判负。[S1][s1]

**本设计采用的判断：** 在当前数据、训练预算和已报告结果下，应保留 A 作为工作基础。C、D 的结果混合了共享的 suppression-only 改动与各自新增模块，不能直接等同于“历史/未来信息无用”。同一 run 的 checkpoints 存在关联，池化差异本身也不构成严格的统计显著性或因果归因。

B 仍然使用 real-evidence teacher 的基础分布，因此不是“完全没有正向视觉信息”。它删除的是：对正向 real/null preference 的**额外重构增幅**。这与“只用负向证据构造时间统计”是两个不同选择。

### 1.2 C 和 D 提醒了什么

C 使用：

$$
N_t=\sum_{k<t}c_k,\qquad
\beta_t=4\left(1+\frac{N_t}{1+N_t}\right).
$$

原始累计量持续增加时，映射逐渐接近上限 8。结果快照报告两次运行的平均 β_t 为 6.31 / 6.19，并将其解释为历史饱和后主要表现为剂量变化。[S1][s1] **6.3 是报告的平均值，不是理论饱和值。** 仅有平均值不能完整说明轨迹内变化，仍需要相应分布读数和剂量对照。

D 使用后缀均值及映射 `Fbar/(1+Fbar)`，位置权重平均为 1.037。[S1][s1] 其 loss 的分子和分母同时带权，因此如果所有权重相同，常数会完全抵消。关键应是权重的相对差异，而不只是平均值偏离 1 多少。[S2][s2]

**本轮设计响应：** 不再以 B 为基础；历史采用均值；future 的工作尺度与历史对齐；组合时只调节一份残余修正，不默认同时增强 target 和 JSD 权重。

### 1.3 Ahm 的地位

Ahm 已有代码支持完整 Aha 基础、prefix-mean history，并只增强负半边。[S3][s3] 在本设计依据的结果快照中，Ahm 和 supb63 尚未列出最终结果。[S1][s1]

因此：Ahm 是可复用的 history-only 实现；下面的 HF 是新增候选，不是 Ahm 的既有验证结论。本文也不假设两项运行现在仍未完成。

---

## 02｜方法范围与固定部分

### 2.1 固定不动的内容

保留当前已验证 Aha 实验的学生初始化、冻结 teacher、privileged/null 两视图、同一 student prefix、完整原始 rollout、词表支持近似及 JSD 聚合口径。Student 仍在 full image 上生成；teacher real/null 只作为训练评分条件。

“两个视图”指当前项目已采用的两种 teacher 输入条件；不擅自修改 pair 模式、图像顺序、null scope 或 privileged crop/hide 构造。

本设计不引入 PPO、advantage、LOO、critic、额外 verifier、GU loss、attention 提取、候选枚举/重排、teacher takeover 或句段重写。Future 来自同一条 rollout 已生成的后缀，不额外采样未来。

### 2.2 与此前 PPO implementation plan 的关系

此前方案是“视觉代价 → 历史风险 Φ → 后续 return → advantage → PPO + 可选 anchor”。本文件是另一条 **pure OPD/JSD** 路线，不继承上述 Φ、return、advantage、PPO clipping 或额外 JSD anchor。

本方案的唯一训练损失仍为 target 与 student 的 JSD。时间信息进入 target 的残余重构；独立 F 对照才进入位置权重。

### 2.3 当前重构方向不交给时间统计决定

完整 Aha target 记为：

$$
q_t^A(v)=\frac{p_t^+(v)\exp(\beta u_t(v))}{\sum_{w\in V}p_t^+(w)\exp(\beta u_t(w))}.
$$

当前工作值为 β=4，与结果快照中的 A 一致。[S1][s1]

History/future 不负责确定哪个替代答案正确，也不产生新的词表方向。当前 real/null 的 u_t(v) 继续决定词表级视觉偏好；时间统计只控制 Aha 之外的额外修正量。

---

## 03｜符号、数据来源与局部视觉信号

为避免把“prefix”和“历史 gate”都记作 h_t，本文用 ξ_t 表示文本 prefix，h_t 专指历史 gate。

| 符号 | 定义与来源 |
|---|---|
| $i$ | 一条实际采样的 rollout；主推导固定 i 后省略该下标 |
| $t$ | response 的文本预测位置，不是 image token 位置 |
| $y_t$ | student 在位置 t 实际生成的 token |
| $\xi_t=(x,y_{<t})$ | 问题与当前位置之前的 student 文本 |
| $p_t^+(v)$ | 冻结 teacher 在 privileged 条件和 ξ_t 下的概率 |
| $p_t^0(v)$ | 同一 teacher 在 matched null 条件和同一 ξ_t 下的概率 |
| $p_t^S(v)$ | student 在 full image 和同一文本 prefix 下的概率 |
| $M_t$ | 有效 response token mask |
| $u_t(v)$ | real/null log-probability difference |
| $n_t(v)$ | 非负的词表级视觉反对强度 |
| $c_t$ | 在实际生成 token 上取得的视觉反对强度 |
| $h_t, f_t$ | 分别由历史/未来均值得到的有界 gate |
| $\Delta\beta_t$ | 相对于完整 Aha 增加的负半边系数，不是总系数 |

### 3.1 先计算 log-ratio，再保留负向部分

$$
u_t(v)=\log p_t^+(v)-\log p_t^0(v),\qquad
n_t(v)=\max(-u_t(v),0),\qquad
c_t=n_t(y_t).
$$

也就是：

$$
c_t=\max\left(\log p_t^0(y_t)-\log p_t^+(y_t),0\right).
$$

若 real/null 对已生成 token 的概率分别为 0.60 和 0.80，则 c_t=−log(0.75)≈0.287682。若分别为 0.30 和 0.10，则 u_t>0，c_t=0。

**这里不是 probability gap，不是 ranking，也不是先 exponentiate 后再做累计。** n_t、c_t 非负，但不是概率；以自然对数计算时可用 nats 表述数值尺度。

### 3.2 同一视觉差分的两种用途

词表级 n_t(v) 用于修改当前 target；realized-token c_t 用于描述已采样路径。两者不能互相替代。

已有实施计划直接使用 teacher 两侧完整 softmax 下的 realized-token log-probability 计算 c_t，避免实际采样 token 不在 student top-k 中时，用 tail bucket 代替它的概率。[S2][s2]

Real/null 差异是在问题、prefix 和 teacher 固定下的条件视觉效应；不是去除所有语言作用之后的纯视觉真值。c_t>0 也不是事实错误标签。

---

## 04｜历史模块：平均冲突提出额外修正需求

### 4.1 Exclusive prefix mean

定义过去的有效位置数：

$$
L_t^-=\sum_{k<t}M_k.
$$

历史均值：

$$
\bar H_t=\frac{\sum_{k<t}M_kc_k}{\max(L_t^-,1)}.
$$

第一位置没有历史，规定 Hbar_1=0。当前 c_t 不进入自己的历史均值。

例如，过去四步 c=[0,0.10,0.10,0.20]，累计为 0.40，均值为 0.10。

这一统计量回答的是“过去平均每 token 受到多强反对”，不是完整的累计距离。选择均值可减弱原始累积量随长度增长导致的饱和，但会稀释早期局部错误，并对近期恢复反应较慢。这是明确的语义取舍，而非无损归一化。

### 4.2 历史 gate

沿用 Ahm 的形式：

$$
h_t=\frac{\bar H_t}{\bar H_t+\kappa},\qquad\kappa>0.
$$

第一版复用 κ=0.10。[S3][s3] κ 是算法尺度，不是仅用于防除零的数值 epsilon。

| Hbar_t | h_t（κ=0.10） | 解释 |
|---:|---:|---|
| 0 | 0 | 未观察到历史负向代价 |
| 0.02 | 0.166667 | 较弱历史冲突 |
| 0.10 | 0.50 | 达到当前工作尺度 |
| 0.30 | 0.75 | 较强历史冲突 |

### 4.3 History-only / Ahm 对照

$$
\Delta\beta_t^H=\beta h_t,\qquad
q_t^H(v)\propto q_t^A(v)\exp[-\Delta\beta_t^H n_t(v)].
$$

当 β=4、h_t=0.5 时，额外负向系数为 2，负半边总系数为 6；正半边仍使用 4。这与已有 Ahm 的“完整 A + 只调负半边”定义一致。[S3][s3]

**局限：** 仅凭历史均值较大，不能判断当前状态后面的路径是否仍在积累冲突。HF 让 future 对这份额外需求进一步调节。

---

## 05｜未来模块：已观察后缀的冲突密度

### 5.1 Strict suffix mean

定义：

$$
L_t^+=\sum_{k>t}M_k.
$$

有已观察后缀时：

$$
\bar F_t=\frac{\sum_{k>t}M_kc_k}{L_t^+}.
$$

没有后缀时，可在数组中将 Fbar_t 存成 0，但必须同时保存“无可用后缀”状态。**零占位不意味着未来被证明为零风险。**

该定义采用已存在 Section 05 的严格后缀均值，而非后缀总和。[S4][s4] 它不包含当前 c_t，也不需要窗口长度或折扣 γ。

Future 是 rollout 完成后才能计算的训练统计；teacher 在每个位置评分时仍只以此前 prefix 为条件。它不是让 teacher 前向偷看未来，也不是推理时增加一次 lookahead。

### 5.2 与历史共用工作尺度

$$
f_t=\frac{\bar F_t}{\bar F_t+\kappa},\qquad\kappa=0.10.
$$

| Fbar_t | 旧 D 的 Fbar/(1+Fbar) | 本候选的 Fbar/(κ+Fbar) |
|---:|---:|---:|
| 0.02 | 0.019608 | 0.166667 |
| 0.15 | 0.130435 | 0.600000 |

这个变化只重新设定尺度，不创造新信息。共用 κ 是少参数的第一版选择，不意味着历史与未来的经验分布相同。

### 5.3 可用性约束

定义 η_t∈{0,1}：只有该 rollout 按既定规则正常完成，并且当前位置有至少一个有效后续 token 时，η_t=1。

本版建议：长度截断、finish_reason 未知或无有效后缀时，η_t=0，使 future 不触发额外修正。对截断样本仍保留 Aha 监督，不删掉整个样本。

这是一项**新增候选工程决策**。已有 negative-history 实施计划明确记录 v1 未区分 finish_reason，不能把本建议写成旧 D 已有功能。[S2][s2]

后续公式使用 η_t f_t；它不是“未来置信概率”，只表示该统计在本版规则下是否可用。

---

## 06｜主候选：history–future 协同调节残余修正

### 6.1 一份残余修正，而不是两个放大器

定义：

$$
\boxed{\Delta\beta_t^{HF}=\beta h_tf_t\eta_t.}
$$

完整 target：

$$
\boxed{
q_t^{HF}(v)=
\frac{q_t^A(v)\exp[-\Delta\beta_t^{HF}n_t(v)]}
{\sum_{w\in V}q_t^A(w)\exp[-\Delta\beta_t^{HF}n_t(w)]}.
}
$$

等价地：

$$
q_t^{HF}(v)\propto p_t^+(v)\exp\left[\beta u_t(v)-\beta h_tf_t\eta_t[-u_t(v)]_+\right].
$$

逐半边展开：

$$
\boxed{
q_t^{HF}(v)\propto p_t^+(v)
\exp\left[
\beta\max(u_t(v),0)
+\beta(1+h_tf_t\eta_t)\min(u_t(v),0)
\right].
}
$$

注意第二项使用的是 min(u,0)≤0。不能把定义为正数的 n_t=[−u_t]_+ 直接带正号加入指数。

### 6.2 为什么使用乘积

乘积对应本轮候选假设：**需要超出 Aha 的额外纠偏时，既应有历史冲突背景，也应在实际观察的后缀中看到冲突持续。**

| 历史 h_t | 未来 f_t | HF 的处理 |
|---|---|---|
| 低 | 低 | 基本保持完整 Aha |
| 低 | 高 | 额外修正仍较小，新出现的冲突主要交给 Aha |
| 高 | 低 | 收回较多 history-only 额外修正 |
| 高 | 高 | 施加较明显的额外负向修正 |

这不是逻辑上被证明的唯一规则。尤其是“历史低、未来高”的早期错误，可能也值得增强，但本版没有单独处理它。选择乘积是把新增模块聚焦在持续冲突，而不是声称覆盖所有错误状态。

### 6.3 为什么不同时增加 future loss weight

HF 只修改 Δβ_t，最终每个有效位置仍按原 token-mean 聚合 JSD。不要默认再乘 1+f_t。

如果同时修改 target 和 loss 权重，就会对同一类状态进行两层增强，也增加解释难度。未来加权单独作为 F 对照，见第 09 节。

### 6.4 Future 不提供词表级正确答案

f_t 是一个 realized-path 的标量，不能告诉我们未采样候选的反事实后果。HF 保留当前 n_t(v) 的方向，只让 f_t 缩放额外抑制。它是 future-informed target modulation，不是 PPO 式正负 action credit，也不是因果归因。

---

## 07｜Pairwise odds 与数值例子

### 7.1 从熟悉的 Aha pairwise 公式开始

原 Aha：

$$
\log\frac{q_t^A(v)}{q_t^A(w)}
=
\log\frac{p_t^+(v)}{p_t^+(w)}
+\beta[u_t(v)-u_t(w)].
$$

HF 只在其上多加一项：

$$
\boxed{
\log\frac{q_t^{HF}(v)}{q_t^{HF}(w)}
=
\log\frac{q_t^A(v)}{q_t^A(w)}
+\Delta\beta_t^{HF}[n_t(w)-n_t(v)].
}
$$

因此，基础竞争关系仍由 Aha 决定；history/future 只调整反对强度的相对差。

### 7.2 两个 token 的完整计算

以下是教学数据，只列出两个 token，其余词表概率省略；不据此断言最终 top-1。数值计算使用未舍入的 log-ratio，表中小数为展示时舍入。

| Token | Real teacher p+ | Null teacher p0 |
|---|---:|---:|
| diamond | 0.60 | 0.80 |
| striped | 0.15 | 0.10 |

于是：

$$
u(\texttt{diamond})=\log0.75\approx-0.287682,
\qquad
u(\texttt{striped})=\log1.5\approx0.405465.
$$

β=4 时：

$$
\frac{q^A(\texttt{striped})}{q^A(\texttt{diamond})}
=
\frac{0.15}{0.60}
\left(\frac{1.5}{0.75}\right)^4
=0.25\times16=4.
$$

设历史均值 Hbar=0.10，所以 h=0.50。比较三种情况：

**仅用历史：** Δβ=4×0.50=2，odds=4×exp(2×0.287682)=7.111111。

**后续仍有较强冲突：** Fbar=0.15，f=0.60，Δβ=4×0.50×0.60=1.20，odds≈5.649194。

**后续冲突较少：** Fbar=0.02，f=1/6，Δβ=1/3，odds≈4.402570。

| 设置 | 额外 Δβ | striped / diamond odds |
|---|---:|---:|
| 原 Aha | 0 | 4.000000 |
| H / Ahm | 2.000000 | 7.111111 |
| HF，后续持续冲突 | 1.200000 | 5.649194 |
| HF，后续冲突较少 | 0.333333 | 4.402570 |
| 无可用 future，回退 Aha | 0 | 4.000000 |

**读法：** Future 没有决定 striped 是正确答案。它决定历史提出的额外抑制保留多少；完整 Aha 的 4 倍基础 odds 一直存在。

### 7.3 同一条 trajectory 如何计算时间统计

假设一条正常结束的 response，在七个有效位置上的 realized 代价是：

$$
c=[0,\ 0.10,\ 0.20,\ 0.40,\ 0.20,\ 0.10,\ 0].
$$

关注第 4 个位置（1-based）：

$$
\bar H_4=(0+0.10+0.20)/3=0.10,
$$

$$
\bar F_4=(0.20+0.10+0)/3=0.10.
$$

当前 c_4=0.40 不进入 Hbar_4 或 Fbar_4。它在相邻其他位置的历史/未来统计中可以出现，这是各位置上下文不同的结果。

κ=0.10、β=4 时，h_4=f_4=0.50，Δβ_4=1。当前词表级 n_4(v) 再决定各 token 的额外修正。以上代价数组只是时间统计示例，不要求实际采样 token 必须是上一小节的 diamond。

---

## 08｜数学性质及其适用边界

下面都是给定公式在一致概率支持上的数学性质，不是模型性能保证。

### 8.1 精确回退

h_t=0、f_t=0 或 η_t=0 时，Δβ_t=0，因此 q_t^{HF}=q_t^A。若 n_t(v) 在整个支持上都为 0，残余项也不会改变 target。

实现上“数值相等”与“bit-identical”要区分。门关闭时应直接走原 Aha 路径；避免对已经归一化的 log q_A 无必要地再做一次 log-softmax 后宣称逐比特相同。

### 8.2 保留非负集合内部的相对比例

若 u_t(v)≥0 且 u_t(w)≥0，则 n_t(v)=n_t(w)=0：

$$
\frac{q_t^{HF}(v)}{q_t^{HF}(w)}
=
\frac{q_t^A(v)}{q_t^A(w)}.
$$

这保留的是 Aha 已经建立的正向/中性候选之间的相对比例，不是将其重置成原始 p+ 比例。

### 8.3 额外修正的方向

当 n_t(v)<n_t(w) 时，HF 增加 v 相对 w 的 odds；若两者反对强度相同，其相对比例不变。

归一化常数为：

$$
Z_t^{HF}=\sum_w q_t^A(w)e^{-\Delta\beta_t n_t(w)}\le1.
$$

u≥0 的 token 的未归一化权重不变，归一化后的概率可以增加。对 u<0 的 token，最终绝对概率不保证逐个都下降：反对较弱的 token 可能因共同归一化相对获益。不要把“只做负向 multiplier”写成“每个负 token 的绝对概率都必然降低”。

### 8.4 系数有界不等于目标偏移有界得足够小

对有限的均值和 κ>0：

$$
0\le\Delta\beta_t^{HF}<\beta,
\qquad
\beta\le\beta_t^-<2\beta.
$$

β_t^- 是负半边总系数。β=4 时其范围为 [4,8)。但是 n_t(v) 来自 log-ratio，仍可能很大，因此残余 target 依然可能非常尖锐。必须测量 q^{HF} 相对 q^A 的实际变化，而不只记录 β_t^-。

### 8.5 恒定统计会退化为恒定剂量

如果一段轨迹上的 Hbar 和 Fbar 都近似不变，h_tf_t 也可能近似常数。均值化避免了原始 cumsum 的机械增长，但不保证产生有价值的轨迹内动态。

---

## 09｜独立 future-weighting 对照与最终 JSD

### 9.1 四种模式的精确定义

| 模式 | Target | 未归一化位置权重 w_t |
|---|---|---:|
| A | q_t^A | 1 |
| H / Ahm | Normalize(q_t^A exp[−βh_tn_t]) | 1 |
| F（独立对照） | q_t^A | 1+η_tf_t |
| HF（主组合候选） | Normalize(q_t^A exp[−βh_tf_tη_tn_t]) | 1 |

F 沿用“用未来重分配位置重要性”的思路，但将基础 target 改回完整 A，并使用 κ 尺度。HF 则将 future 放在残余强度上。F 不是 HF 的必需前置组件，也不是另加一个 loss。

### 9.2 JSD 与 loss reduction

对每个有效位置：

$$
\ell_{i,t}=\operatorname{JSD}(q_{i,t},p_{i,t}^S).
$$

A、H、HF：

$$
\boxed{
L=\frac{\sum_{i,t}M_{i,t}\ell_{i,t}}{\sum_{i,t}M_{i,t}}.
}
$$

独立 F：

$$
\boxed{
L_F=\frac{\sum_{i,t}M_{i,t}w_{i,t}\ell_{i,t}}{\sum_{i,t}M_{i,t}w_{i,t}}.
}
$$

分母必须使用与分子相同的权重。它规范化的是权重总量，不保证梯度范数或有效步长不变。若 w_t 在全部有效位置恒定，F 与原 token-mean 在数学上相同。

不要先对每条 rollout 单独归一化再等权平均；那会改变原有的 token-level 聚合口径。分布式/梯度累积的分母需与整个有效聚合批次一致，不能简单平均各 microbatch 的均值来代替。

### 9.3 为什么 HF 不需要 baseline / advantage

未来统计只改变 target 生成规则，而不直接给 sampled token 正负 action credit。训练方向仍由 q_t 与 p_t^S 的差异决定。因此本版不需要 old/new policy ratio、PPO clipping 或同题 leave-one-out baseline。已有每题 rollout 数按 Aha 设置保持，不因该模块增加。

---

## 10｜训练流程、代码接入与边界处理

### 10.1 每批的计算顺序

| 阶段 | 输入 | 输出 | 梯度 |
|---|---|---|---|
| Student rollout | 原全图、问题、既定采样器 | 原始 y、mask、finish_reason | 采样记录固定 |
| Teacher real/null 评分 | 相同 y 与各位置 prefix、既定两种条件 | 词表/支持上的 logp；精确 realized logp | no_grad |
| 时间统计 | realized logp、mask | c、Hbar、Fbar、h、f、η | no_grad |
| Target 构造 | 原 Aha logits、n、Δβ | q^{HF} | no_grad |
| Student 学习 | 当前 student 分布、固定 q^{HF} | JSD 与参数梯度 | 只更新 student |
| 下一批 | 更新后的 student | 新 rollout | 重新评分 |

图像和文本打包的绝对位置可以不同，但 real/null/student 必须以 response 相对位置对齐。不得重新 tokenize 已保存的 response 后假设 token 边界完全一致。

### 10.2 张量级伪代码

下列是设计级伪代码，不是已接入仓库、已运行 GPU 的实现。`logp_plus_support` 与 `logp_null_support` 必须复用当前 Aha 的一致支持定义；`realized_lp_*` 则来自完整 softmax 下对实际 token 的精确评分。

```python
# Shapes: mask [B,T]; realized_lp_* [B,T]; support logp [B,T,K].
# Preconditions: beta >= 0; kappa > 0; all valid logp finite;
# complete [B] is True only for a recognized normal termination.
# Each row is one original rollout, not a concatenation of trajectories.

with torch.no_grad():
    valid = mask.bool()
    m = valid.float()

    # Clear padding BEFORE arithmetic; reject non-finite values on valid tokens.
    lp_r = torch.where(valid, realized_lp_plus.float(), 0.0)
    lp_n = torch.where(valid, realized_lp_null.float(), 0.0)
    c = torch.relu(lp_n - lp_r)                     # no exponentiation

    prefix_sum = c.cumsum(dim=1) - c               # strict k < t
    prefix_count = m.cumsum(dim=1) - m

    suffix_sum = c.flip(1).cumsum(1).flip(1) - c    # strict k > t
    suffix_count = m.flip(1).cumsum(1).flip(1) - m

    Hbar = prefix_sum / prefix_count.clamp_min(1.0)
    Fbar = suffix_sum / suffix_count.clamp_min(1.0)
    h = Hbar / (Hbar + kappa)
    f = Fbar / (Fbar + kappa)
    eligible = valid & (suffix_count > 0) & complete[:, None]

    # The existing implementation must provide finite padded support rows.
    u = logp_plus_support.float() - logp_null_support.float()
    n = torch.relu(-u)
    base_aha_logits = logp_plus_support.float() + beta * u

    if mode == "A":
        target_logits = base_aha_logits            # use existing A path
        weights = m
    elif mode == "H":
        delta_beta = beta * h * m
        target_logits = base_aha_logits - delta_beta[..., None] * n
        weights = m
    elif mode == "F":
        target_logits = base_aha_logits
        weights = m * (1.0 + eligible.float() * f)
    elif mode == "HF":
        delta_beta = beta * h * f * eligible.float()
        target_logits = base_aha_logits - delta_beta[..., None] * n
        weights = m
    else:
        raise ValueError("unknown mode")

    log_q = torch.log_softmax(target_logits, dim=-1)
    q = log_q.exp()

# Student probability retains gradients; use the existing JSD implementation.
per_token_jsd = jsd_per_token(q.detach(), student_probs_same_support)
# Reduce numerator/denominator consistently across the actual update batch.
loss = global_sum(weights * per_token_jsd) / global_sum(weights)
```

代码中的 `global_sum`、support 构造及 `jsd_per_token` 是现有训练框架的接口占位，不能据此声称完整训练已可运行。`global_sum` 还必须与 DDP 的梯度平均约定匹配。

### 10.3 现有代码接入点

已有实施计划把 realized null logp 从 `dp_actor.py` 传给 `compute_self_distillation_loss`；时间统计在 `core_algos.py` 的 no_grad 分支内计算。[S2][s2]

已读取的代码快照允许 history 在完整 A 基础上运行，但仍阻止 `future_weight=True` 与完整 A 同时使用。[S5][s5] 因此不能只修改 launcher。建议新增语义清楚的模式选择，补完整 A 的 F/HF 分支，并保留旧 B/C/D 路径用于复现。

伪代码中的 `mode` 是建议接口，不是当前仓库已经存在的配置名。不得通过直接关闭旧守卫，误把尚未验证的组合伪装成旧分支。

### 10.4 必须明确的边界

**Padding：** 先置零再相减/累加，不能依赖 `NaN * 0`。Padding 不进入历史、未来、损失或平均分母。累计必须在每条 rollout 边界重置。

**EOS：** 真实采样的终止 token 保持既定 response mask；其后没有未来，所以 HF 的额外修正为 0，但 Aha 监督保留。

**长度截断：** 本版建议对该 rollout 关闭 future 触发的修改，保留 Aha。不能把截断当成已经低风险结束。H-only 对照保持已有 Ahm 行为，二者的终止处理差异需明示。

**支持集与 tail bucket：** 沿用 Aha 既有 top-k+tail；tail bucket 的 log-ratio 是集合质量的差异，不是其中每个 token 的反对强度。Pairwise 结论在实现所用的一致支持上成立，不能将聚合桶性质误解为桶内各 token 的精确性质。

**非有限值：** 有效位置发现无穷或 NaN，明确报错排查；不暗中 clipping u 或切换概率差，避免同时改变信号几何。

---

## 11｜最小训练对照、日志与验收

### 11.1 训练顺序

首先读取已有 Ahm/supb63 的最终结果，不从本报告快照推断这些运行当前是否结束。随后以完整 A 为基础，逐项训练 H、独立 F 或 HF；不默认同时启动所有组合。

| 对比 | 回答什么 | 不能单独证明什么 |
|---|---|---|
| H vs A | 完整 A 上均值历史加强是否改善性能 | 动态性优于固定剂量 |
| F vs A | 完整 A 上重新分配 JSD 位置权重是否有效 | 真正的 action credit 或因果恢复 |
| HF vs A | 条件化残余修正是否改善 A | 收益一定来自历史—未来配合 |
| HF vs H | 收回部分历史额外抑制是否有帮助 | future 选择了正确位置，而非平均少罚 |
| 相应 fixed-negative-dose vs 动态版本 | 固定加强能否解释收益 | 单靠平均 β 匹配完整重构幅度 |

### 11.2 固定剂量对照必须保留 A 的正半边

定义：

$$
q_t^{\mathrm{const}}(v)\propto p_t^+(v)
\exp\left[\beta\max(u_t(v),0)+(\beta+\delta)\min(u_t(v),0)\right].
$$

δ 固定，不随位置变化；其工作值应明确依据相应动态运行的有效 token 平均 Δβ 来设定，并记录选择过程。若是看完运行后设定，标作描述性剂量匹配，不伪称预注册。

已有 supb63 仍是 B 基础，不等价于这个完整 A 上的固定负半边对照。[S1][s1]

由于 Δβ 与当前 n_t(v) 的相关性也影响 target，平均 Δβ 匹配不等于 target TV、熵或梯度幅度完全匹配。必须同时报告实际 target 偏移。

### 11.3 何时考虑时间错配控制

只有直接训练出现值得解释的收益时，再考虑保留有效位置 Δβ 的边际分布、打乱其时间对应的训练控制。它不新增 teacher 视图或重排候选 token。

打乱时应保持 η=0 的位置仍不施加修正，并明确这保持的是 Δβ 分布，而不是与 n_t 的联合分布或实际 target TV。该控制帮助检验位置对应的价值，不是严格的因果证明。

本文件不要求新增独立科学 probe、多 seed sweep 或先训练预测器。单元测试和训练日志用于验证实现与解释实际训练结果。

### 11.4 保留现有评测口径

结果快照使用 2459 题、seed=42、batch 48、51 步、lr=2e−6、n=2；每次取 {30,40,50} checkpoint 峰值，并采用两次同侧的经验判据。[S1][s1]

实施时继承当前已验证 A 的完整 manifest；不要只复制这些标量而漏掉 processor、pair/null scope、采样器或并行设置。结论表继续使用 judge-based 指标。池化 checkpoint 均值作为辅助描述，不把 checkpoints 当作独立 runs。

### 11.5 最小日志

| 目的 | 记录内容 |
|---|---|
| 局部代价尺度 | c 均值、分位数、c>0 比例 |
| 是否真正有时间动态 | Hbar/Fbar、h/f 的分位数和轨迹内标准差 |
| 是否只是剂量变化 | Δβ、负半边总 β 的有效位置统计 |
| 实际改变了 target 多少 | q^{HF} 或 q^H 相对 q^A 的 TV/JSD、target entropy |
| F 是否接近常数 | 原始及归一化位置权重的离散程度 |
| 边界规则覆盖 | η=0 比例、正常终止/截断/未知 finish_reason 比例 |
| 是否生成退化 | 重复率、cap 命中率、response 长度、judge accuracy |

### 11.6 数学与工程验收

门关闭应回到原 Aha 路径；手算检查 exclusive prefix/suffix；首位置 h=0；最后位置与截断的 η=0；正半边系数不被时间模块放大；非负集合内部 odds 保持；HF 不同时带 F 位置权重；F 分母同权；valid token 的 teacher 反馈无梯度、student JSD 有梯度。

测试必须能抓住 inclusive history、future 包含当前位、β_t 错乘正半边、padding 进入累计、截断伪装正常终止和重复归一化改变聚合口径等错误。

这些验收只验证数学与实现，不预测模型是否优于 Aha。

---

## 12｜风险、创新性与待决定事项

### 12.1 不能把较低未来冲突称作已证明的恢复

Fbar 小可能来自真正的视觉一致性改善，也可能来自泛化叙述、避开细节、语言先验主导、较短后缀或 teacher 对重复内容同样认可。HF 不直接识别事实正确性或重复循环。

### 12.2 均值与全后缀仍有局限

Prefix mean 会稀释局部错误；suffix mean 可能被离当前位置很远的事件或大量中性文本稀释。样本数很少的后缀方差可能较大。本版不新增 window、EMA、长度阈值或置信度因子，是为了先保持少量变量，不是已经解决这些问题。

### 12.3 乘积 gate 的特定盲区

历史低、未来高的新错误只获得较小额外修正；两个均值都稳定时，乘积仍可能近似常数；乘积可能使平均 Δβ 很小，从而重现“额外作用过弱”。需要根据实际 Δβ 与 target 偏移判断，而不是只看到 h、f 都不为零。

### 12.4 相对更好可能只是少罚

Δβ^{HF}≤Δβ^H。HF 比 H 好，可以仅仅因为平均抑制更弱。比较 A、固定负向剂量与必要的时间错配训练，才能更有把握解释时间信息的作用。

### 12.5 保留 Aha 公式不意味着保留全部原性能

对 target 的附加修正仍可能改变学生的输出风格、校准、关系推理或停止行为。log-ratio 可很大，β 有界不提供性能或稳定性保证。截断回退也可能影响短长 response 之间的相对学习，需报告相关率。

### 12.6 方法层面应如何表述

**可以作为候选命题：** 在完整 Aha 已经提供有效局部视觉重构的条件下，不对所有有冲突历史的状态一律增强抑制，而利用已观察未来是否持续冲突来调节附加纠偏。

**不能提前声称：** 已识别视觉冲突的因果来源；已学到 action credit；已证明正半边必要；已证明 HF 优于 Aha；或它是一个全新的通用优化框架。

当前结构仍属于 Aha-compatible target reconstruction。其潜在贡献在于历史需求与后续持续性共同决定额外修正是否适用；创新性和有效性需要训练及相关工作比较建立，本文件没有完成新的文献查新。

### 12.7 实施前仍需确认的选择

| 项目 | 本版工作建议 | 状态 |
|---|---|---|
| 基础与局部代价 | 完整 A；negative log-ratio | 与最近讨论一致 |
| κ | 历史与未来共用 0.10 | 初始工作尺度，不宣称最优 |
| HF 额外系数 | βh_tf_tη_t | 新候选；尚未验证 |
| HF 的额外 loss weight | 不使用 | 保持单一作用位置 |
| Future 不可用 | 回退 A；不删除正常可评分样本 | 新工程规则，需 adapter 支持 |
| 新增 clipping / rank / discount | 第一版不加 | 保持信号与变量数量稳定 |
| 原包 PPO/GU 模块 | 不继承 | 与 pure JSD 路线明确分开 |

相对于当前 Ahm，HF 可以不增加新的可调标量；但乘积结构、共用 κ、回退规则和系数上限仍然是方法假设，不能将它宣传成“无设计自由度”。

---

## 13｜方法总结与来源登记

### 13.1 最小公式集合

$$
\begin{aligned}
u_t(v)&=\log p_t^+(v)-\log p_t^0(v),\\
n_t(v)&=[-u_t(v)]_+,\qquad c_t=n_t(y_t),\\
\bar H_t&=\frac{\sum_{k<t}M_kc_k}{\max(\sum_{k<t}M_k,1)},\\
\bar F_t&=\frac{\sum_{k>t}M_kc_k}{\max(\sum_{k>t}M_k,1)},\\
h_t&=\frac{\bar H_t}{\bar H_t+\kappa},\qquad
f_t=\frac{\bar F_t}{\bar F_t+\kappa},\\
\Delta\beta_t^{HF}&=\beta h_tf_t\eta_t,\\
q_t^{HF}(v)&\propto p_t^+(v)\exp\left[\beta u_t(v)-\Delta\beta_t^{HF}n_t(v)\right],\\
L_{HF}&=\frac{\sum_{i,t}M_{i,t}\operatorname{JSD}(q_{i,t}^{HF},p_{i,t}^S)}{\sum_{i,t}M_{i,t}}.
\end{aligned}
$$

其中，η_t 只在 rollout 正常完成且当前有已观察有效后缀时为 1；否则本候选关闭额外 future-dependent 修正。Fbar 的零占位不能替代 η 的语义。

### 13.2 一句话总结

> **当前视觉差异决定方向；完整 Aha 提供基础 target；历史平均冲突提出额外抑制需求；已观察未来的持续冲突调节这份需求；最终仍只用 JSD 训练 student。**

### 13.3 来源与版本

| 编号 | 材料 | 本文使用范围 |
|---|---|---|
| P1 | 用户提供的 `revision_opd_arxiv(6).pdf`，第 4–5 页 §3.1–3.2 / Eqs. 4、6、8–10；第 19 页 Table 4 | Aha real/null、指数重构、pairwise odds、JSD 监督框架 |
| [S1][s1] | `results_comparison_2026-10-02.md`，commit `ddd5b6e` | A/B/C/D 结果、运行快照、C/D 机制解释 |
| [S2][s2] | `implementation_plan.md`，commit `ddd5b6e` | realized logp、原 C/D 接入、加权 loss、原截断局限 |
| [S3][s3] | `scripts/train_pair_ahist.sh`，commit `9efb5b6` | Ahm 的完整 A 基础、均值历史、κ=0.10 |
| [S4][s4] | `05_future_conflict_persistence.md`，commit `ddd5b6e` | strict suffix mean、future position weighting 的原设计 |
| [S5][s5] | `verl/trainer/ppo/core_algos.py`，commit `9efb5b6`，1150–1255 行 | 已有 runtime guard、history 支持完整 A、旧 future guard |
| C1 | 本轮关于“保持 Aha，history 提需求、future 调残余”的对话 | HF 乘积规则、F 独立对照、回退规则、训练对照与限制 |

来源中较强的因果表述不被本文件自动当作已证明结论；本文只据其记录制定候选设计。数值例子是合成计算，不是模型输出或训练结果。本文未修改已有 GitHub 文件，也不是对旧 PPO implementation plan 的静默覆盖。

[s1]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/ddd5b6e7fabb0bd2b73e083ffa4d42699dcce693/docs/negative_history/results_comparison_2026-10-02.md
[s2]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/ddd5b6e7fabb0bd2b73e083ffa4d42699dcce693/docs/negative_history/implementation_plan.md
[s3]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/9efb5b6ce460345a49110bc6126a10be9c707efa/scripts/train_pair_ahist.sh
[s4]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/ddd5b6e7fabb0bd2b73e083ffa4d42699dcce693/docs/negative_history/05_future_conflict_persistence.md
[s5]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/9efb5b6ce460345a49110bc6126a10be9c707efa/verl/trainer/ppo/core_algos.py#L1150-L1255
