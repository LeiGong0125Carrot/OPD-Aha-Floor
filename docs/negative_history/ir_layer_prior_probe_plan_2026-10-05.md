# 探针计划：哪些内部读出接近“缺 crop 的先验”，能否替代 visual null？

**版本：** v0.2（完整修订方案）  
**日期：** 2026-10-05  
**文档分支：** `negative_history`  
**状态：** 计划；本次只更新文档，不实现代码、不提交作业、不取消已有运行，也不新增实验结果。

> 保留原计划的研究问题、三种 teacher 评分、逐层／block 两类候选与 M1–M5 指标框架；补全数据隔离、回归定义、tail 构造、判读和实施接口。原始 v0.1 保存在 [e581326][D0]，原文中的实验记录没有在其他文件中被改写。
>
> **正式训练目标不变：** 单次冻结 privileged teacher 前向，无在线 visual-null 前向；teacher 的最终分布始终是 anchor，student 正常更新，不使用 student-anchor。目标表现为 TB > 50、V\* 达到 92–93 的量级，是否达到由真实训练决定。
>
> **成本边界：** 离线诊断会运行 real、null、text 三种评分。若用真 null 拟合全局系数，它属于“离线 null 监督校准、在线单前向”，不能称为研究与训练全过程从未使用 null。此处 probe 是离线测量，不是新增一个神经网络解码头。

## 0. 来源、修订内容与执行边界

### 0.1 来源快照

- 原计划：`negative_history@e58132609fc4a7af9bbb00c6eff174a522a051b6`。[D0]
- 前置设计：`teacher_internal_residual_reconstruction_plan.md`（`cc115a4`）；IR 结果按 `6a6ef86` 快照解释。[D1]、[D2]
- null-free 历史结果：`null_free_line_summary_2026-10-04.md`。本文只引用已报告结果，不推断排队实验已完成。[D3]
- 核对的代码：`sup@d379de1e6c33f71bd6e565cb5f5ac77cec0b26fe`，其 IR 实现来自 `98ee0b9`。本文复核了原 A 的 tail 与 detached target 路径；其他接口按已核对的 IR 实现与前置设计衔接。开发前必须重新读取分支，保留并行修改。[C1]、[C2]、[C3]、[C4]、[C5]
- 下文标为“修订定义”“建议默认值”的内容是本次设计选择，不是 v0.1 原本已明确的设置或既有 API。

### 0.2 从 v0.1 到 v0.2 的关键修订

| 项目 | 原计划或未明确处 | 本版处理 |
|---|---|---|
| prior 的含义 | “语言先验”与缺 crop 分布交替使用 | 严格区分 p⁰ 和 p_text；接近某分布不等于发现独立语言模块 |
| M2 权重 | 未说明是否逐 token 拟合 | 主候选为全局固定系数；逐位置拟合仅是不可部署的 oracle 诊断 |
| 选择与检验 | 同一 400 题上量指标 | 按题目／原图分组隔离 fit、selection、held-out；不随机拆 token |
| 读出与 log-ratio | 直接回归，常数项不明确 | 在显式 top-k 内统一中心化；回归不把 tail 当作独立词表 logit |
| tail | “按 A，tail 的 û=0” | 原 A 是先聚合再计算 tail log-ratio；置零必须单列变体 |
| 逐层 prior 与 IR | 未指出代数关系 | 共享最终 D 的完整词表方案等价于后缀 IR；压缩顺序可能打破等价 |
| student 梯度 | 认为共享 teacher 上游本身改变梯度 | 同一 detached target、student 状态和权重产生同一梯度；近似误差另论 |
| 失败结论 | 线性读出全部不可能／归因于 attention 重算 | 只否定已测候选族；不把未做的因果干预当事实 |
| M2 高但 M3 失败 | 自动进入加权 IR | 不越过主门槛；单列后续探索，须另作决定 |
| 时长与显存 | 约 1 h、只估 hidden 缓存 | 保留为旧估计，重新计入逐层词表投影、精度和评分成本 |

## 1. 动机与可检验问题

### 1.1 A 的实际参照

同一个冻结 teacher 在相同问题和 student prefix 下计算：

$$
p_t^+=p_\phi(\cdot\mid I,C,x,y_{<t}),\qquad
p_t^0=p_\phi(\cdot\mid I,C_{\rm blank},x,y_{<t}).
$$

这里 null_scope=last：全图 I 保留，仅第二张 crop 内容变成匹配的均值色块。差分是这个模型对 crop 内容替换的条件响应，不是将概率机械分解为彼此独立的“语言项＋视觉项”。纯文本分布 p_text 另作诊断参照，不代替 p⁰。[D0]、[C2]

S/St 没有恢复 A 的 V\* 表现；这支持继续寻找不同的参照，但不能证明所有单前向方法都不可能。[D3]

### 1.2 已报告的 IR 动机结果

| 配置 | 平均 target TV | TB 峰 | V\* 峰 | 报告现象 |
|---|---:|---:|---:|---|
| [12,20)，λ=1，full tail | 0.088 | 47.65 | 87.43 | 未恢复 A 的表现 |
| [12,20)，λ=4，full tail | 0.267 | 46.67 | 85.86 | 格式与长度变化，TB 缺 `<answer>` 约 12–15% |
| A 的历史参照 | 约 0.27 | 49.9–51.6 | 91.6–94.8 | 非本次新实验 |

以上按结果文档记录。[D2] 平均 TV 相近不代表相同位置、方向或目标形状都匹配；这些结果反对“只要加大该区间就足够”，不证明该区间完全没有视觉信息。

### 1.3 本轮问题

**从一次 real teacher 前向取得的内部读出，能否在未用于选择的数据上，近似 p⁰ 并重现 A 的 reconstructed target？**

问题分三层：

1. **分布近似：** 是否有 p̂ 接近 p⁰，而非只接近 p_text 或高频输出？
2. **目标近似：** û = log p⁺ − log p̂ 经 β=4 放大后，q̂ 是否仍然接近 q_A？
3. **训练迁移：** 通过前两项的候选，在新 student prefix 和实际训练中是否有效？

探针回答前两层，正式训练回答第三层。不训练新读出头，不以 attention 权重作为视觉真值，不叠 history/future、PPO 或 student-reference。

### 1.4 可能的有效结论

通过意味着“在指定读出族、模型、数据和 prefix 上找到可检验的参照近似”，不是“找到纯语言层”。失败意味着“未在本次有限搜索中找到合格候选”，不是“null 必须由 attention 重算才能得到”的因果证明。

## 2. 探针设计与分布定义

### 2.1 数据、分组和生成

保留 v0.1：从 `train_6karmA_pair.parquet` 的训练数据抽 400 题，seed=42；base Qwen3.5-4B 看全图贪心生成一次，最大 completion=1024 token。路径由运行参数提供，不写机器绝对路径。

**修订定义：** 将这 400 题按共享原图／样本来源分组，约分为 fit 200、selection 100、held-out 100。若一图多问导致不能精确满足题数，以组完整性优先并报告实际数量。相同题目的所有 token、crop 版本及后续 prefix 检查必须属于同一组。不得使用 TB/V\* 测试题或其 judge 结果选择层／系数。

- fit：拟合唯一一组全局 block 系数、确定活动位置阈值；逐层候选无需拟合。
- selection：比较层、归一化口径与已固定的全局系数候选，锁定一个主候选。
- held-out：只确认已锁定主候选及预定基线，不重新选层、调系数或门槛。
- 同时写出抽样／分组 manifest、图像和问题标识哈希。若发现近重复，先修分组再评分。

初始 student 与 teacher **权重相同但输入不同**：全图 student 与 pair teacher 不要求同分布。贪心 prefix 也不是正式 n2 采样分布的完整覆盖，必须单列这一限制。

### 2.2 评分输入与位置

生成的 response token IDs 固定后，所有条件 teacher-force 同一条 response，不重新生成 teacher 回答。

| 条件 | 输入 | 采集 |
|---|---|---|
| Real | [全图, crop]＋原问题／system＋同一 response | z⁺、边界 h⁰…hᴸ、最终 pre-norm 状态及 D_t |
| Null | [全图, 匹配均值色块]＋完全相同文本和 response | z⁰、p⁰；复用 null_scope=last 的现有图像构造 |
| Text | 原问题／system＋同一 response，不提供图像 | z_text、p_text；只作诊断，不加“看不到图”等提示 |

保留既有文本，不添加 GT 答案、caption、hint 或重新组织 reasoning。Text 删除多模态占位及输入张量时，要记录精确模板；文本中原本提及图片的文字不随意改写。

用与正式评分一致的 tokenizer、图像 resize、chat template 和 scoring temperature τ。τ 从配置读取并记录，不能将 greedy 生成的“temperature=0”用于除 logits。每个条件分别计算自己的 response_start_idx。

为避免索引歧义：令 P 为 response 在拼接序列中的零基起点、t=0…T−1，则预测 y_t 的状态位于 P+t−1。先在未 packing 序列验证，再映射 packed 索引；禁止读取 y_t 所在位置去预测 y_t。结束符是否参与 loss 按现有 response_mask；padding 不计入。

主离线后端沿用 HF、BF16、无梯度、只读 hook。v0.1 的 eager 作为兼容验证起点，不把“必须 eager”当理论要求；记录其与训练后端的 anchor 差异，不能让后端变化伪装成 prior 效果。

### 2.3 共享支持集与原 A 目标（主口径）

令 K_t 是 base student 在该完整图像 prefix 上的 top-100 token IDs。支持集不是 teacher 自己的 top-k，也不是把不同模型的各自 top-k 按数组下标相减。base 的全图 teacher-forced 评分若不能从生成过程可靠取得，应单独计入一次评分成本。

对任意完整词表分布 p，定义压缩算子：

$$
\mathcal C_{K_t}(p)=\big((p(v))_{v\in K_t},\ \sum_{v\notin K_t}p(v)\big).
$$

压缩后的最后一列记为 ⊥，仅表示集合，不是 tokenizer token。令 P⁺=C_K(p⁺)、P⁰=C_K(p⁰)、P_text=C_K(p_text)。必须先获得完整词表的归一化，再 gather；不能将显式 top-k 重新归一化成 1 而丢掉 tail 质量。

**主诊断完全复现原 A 的“先聚合、再重构”：**

$$
u_t(k)=\log P_t^+(k)-\log P_t^0(k),\qquad
Q_t^A=\operatorname{softmax}\big(\log P_t^++4u_t\big),\quad k\in K_t\cup\{\bot\}.
$$

候选同样先算完整 p̂、再压缩：

$$
\widehat P_t=\mathcal C_{K_t}(\widehat p_t),\qquad
\widehat u_t=\log P_t^+-\log\widehat P_t,\qquad
\widehat Q_t=\operatorname{softmax}(\log P_t^++4\widehat u_t).
$$

**tail 的差分不置零：** û_t(⊥)=log P⁺_t(⊥)−log P̂_t(⊥)。原代码只有专门开启 tail_u_zero 才置零，不是 A 的默认定义。[C3] tail-zero 与完整词表重构后再聚合，可作为独立敏感性变体，但不得拿一种规则的 M3 通过结果去启动另一种训练规则。

数值上复用现有 add_tail 的稳定 log1mexp 与边界保护，记录 clamp 次数和归一化误差；合成测试另给 FP64 精确参照。零 tail 的有限占位与额外归一化策略须与训练一致，不能隐式加 smoothing。所有主比较使用相同支持集、mask 和数值策略。

### 2.4 候选 A：共享最终归一化的逐层 prior

设 h^ℓ 为第 ℓ 个 block 边界的 residual stream，h⁰ 是第一块之前，hᴸ 是最终 norm 之前。L、hidden size 和模块路径从实际模型读取；当前计划预期 L=32，不能在其他架构上静默沿用。

对 Qwen3.5 当前已核对的 zero-centered RMSNorm：

$$
D_t^{final}=\operatorname{diag}\left(\frac{1+w_{norm}}{\sqrt{\operatorname{mean}_d[(h_t^L)^2]+\epsilon}}\right).
$$

用相同输出头与温度读出：

$$
z_t^+=\tau^{-1}(W D_t^{final}h_t^L+b_{out}),\qquad
z_{t,\ell}^{final}=\tau^{-1}(W D_t^{final}h_t^\ell+b_{out}),\qquad
p_{t,\ell}^{final}=\operatorname{softmax}(z_{t,\ell}^{final}).
$$

模型实际无 output bias 时 b_out=0；若存在，两种分布都使用同一个 bias。实际 BF16 舍入下应以模型原输出为 p⁺，读出重建只在预定容差内核对，不假定 FP32 分解逐比特等于 BF16 logits。[C1]

遍历 ℓ=0…L；ℓ=L 是必要的负控制，应重现 p⁺、使候选重构退回普通 privileged OPD。ℓ=0 使用的 D_t 仍由真实最终状态产生，因此不属于纯文本读出。

### 2.5 候选 B：该层自身缩放的 prior（单独标记）

$$
D_t^\ell=\operatorname{diag}\left(\frac{1+w_{norm}}{\sqrt{\operatorname{mean}_d[(h_t^\ell)^2]+\epsilon}}\right),\qquad
p_{t,\ell}^{own}=\operatorname{softmax}\big(\tau^{-1}(W D_t^\ell h_t^\ell+b_{out})\big).
$$

仍使用原最终 norm 的学习参数，只替换该层输入对应的 RMS 尺度，不引入新头或训练校准矩阵。这只是另一种诊断读出，不保证早层表示已与最终 head 对齐。final/own 两族分别记录，不混合层编号或结果。

### 2.6 候选 C：全局 block 组合及可部署的伪 prior

每块贡献定义为：

$$
r_{t,j}=\tau^{-1}W D_t^{final}(h_t^{j+1}-h_t^j),\qquad j=0,\ldots,L-1.
$$

系数 w 的拟合方式见 §3.2。正式候选只有 **L 个全局标量**，不随题目、token 或未来 prefix 重拟合。由拟合后的 w 构造：

$$
r_t^w=\sum_jw_jr_{t,j},\qquad
\widehat z_t^0=z_t^+-r_t^w,\qquad
\widehat p_t^0=\operatorname{softmax}(\widehat z_t^0).
$$

再按 §2.3 聚合 p̂⁰，计算真正的 tail log-ratio 和 Q̂。不把 Σw_jr_j 的均值或和直接塞成一个“tail logit”。

部署时可先合并隐藏增量，再做一次头部投影：r^w=τ⁻¹WD_final Σw_jΔh_j。全局系数来自离线 null 监督，必须随结果披露。负系数不自动解释成“语言层”，大系数或共线性也不能隐瞒。

### 2.7 数学核对：候选 A 在完整词表上等价于后缀 IR

对共享 D_final、相同 bias 和 τ 的读出：

$$
\log p_t^+(v)-\log p_{t,\ell}^{final}(v)
=\tau^{-1}[WD_t^{final}(h_t^L-h_t^\ell)]_v+c_t,
$$

c_t 与 token 无关，进入 softmax 后消失，因此：

$$
\widehat q_{t,\ell}^{V}
=\operatorname{softmax}\big(z_t^++\beta r_t^{\ell:L}\big).
$$

例如“第 20 层作为 prior”在完整词表上就是“[20,L) 后缀 residual 重构”。候选 C 也有同类完整词表关系。该恒等式不自动延伸到先聚合的 tail，因为 logsumexp 和线性求和不交换；候选 B 使用 D_ℓ，也不具有同一简单后缀等价。

**研究价值来自如何有依据地选择／校准内部参照，而不是仅把已有后缀 IR 改称 internal prior。** 主训练口径始终按 §2.3，完整词表等价用于理论与单测，不能替换它。

## 3. 指标、拟合和判读的完整定义

### 3.1 M1：哪个读出接近哪种参照？

对每个层／归一化族记录 TV(P̂_ℓ,P⁰)、TV(P̂_ℓ,P_text)，另报 TV(P⁺,P⁰)、TV(P⁺,P_text)、TV(P⁰,P_text)。

每个 TV 先在单个回答位置的 K+1 类别上计算，再报告 token 加权均值、中位数、P90、逐题均值与样本数。主表固定用 token 权重，逐题均值作为长度敏感性检查，不用二者挑最优。

保留 v0.1 的初筛：候选相对最终 teacher 的 null-TV 更小，才值得进一步解释。但 M1 不是训练通过条件。大量普通语法位置接近可能掩盖答案位置差异；接近 p_text 也不证明相应层不含图像信息。

### 3.2 M2：全局拟合，而不是用测试位置的真 null 求系数

**本版新增的操作定义：** 在 fit split 上，以每个位置的显式 K 个 token 构造回归；tail 不进入回归，但必须进入 M3 的 target 检验。

为避免 logit 与 log-probability 之间的公共常数影响，定义 teacher 条件权重：

$$
\omega_{t,v}=\frac{p_t^+(v)}{\sum_{k\in K_t}p_t^+(k)},\quad v\in K_t,\qquad
\mathcal Z_t(f)_v=f_v-\sum_{k\in K_t}\omega_{t,k}f_k.
$$

使用 ū_t=Z_t(u_{t,K}) 与 r̄_t,j=Z_t(r_{t,j,K})。p⁺、p⁰ 的显式项仍来自完整词表概率；只在度量权重里作 top-k 条件化，不改变 A target。这个度量是本版的拟合选择，不把它声称为原 JSD 的等价替代。

令 X_t 为 K×L 矩阵，其列为 r̄_t,j，Ω_t=diag(ω_t)，N 为 fit 有效位置数：

$$
H=\frac1N\sum_{t\in fit}X_t^\top\Omega_tX_t,\qquad
b=\frac1N\sum_{t\in fit}X_t^\top\Omega_t\bar u_t.
$$

建议固定的数值正则（不是 GPU 蒸馏参数）：

$$
\eta=10^{-4}\,\frac{\operatorname{tr}(H)}L+10^{-12},\qquad
w=(H+\eta I)^{-1}b.
$$

FP64 累积 H、b 并求解，只存 L×L 与 L 大小的充分统计；不拼接全部词表回归矩阵。记录秩、条件数、特征值范围、η 和 w。H=0／有效能量为零时标为不可拟合，不捏造高分。

**R² 的明确口径：** 在未用于拟合的 split 上，以每个位置零中心纠正为基线：

$$
R^2_{corr}=1-\frac{\sum_t\|\bar u_t-X_tw\|_{\Omega_t}^2}
{\sum_t\|\bar u_t\|_{\Omega_t}^2}.
$$

这不是带全局截距的普通 pooled R²；它度量相对 token preference 的解释比例，可以为负。分母每位置平均小于 10⁻¹² 时标记 undefined 并报告能量，不填 0 或 1。拟合只在 fit 进行；selection／held-out 仅应用固定 w。

保留原阈值 R²≥0.5、R²<0.2 作为方向性标尺，不解释为已找到“crop 块”或所有线性方法不可能。高 R² 若未转化为更好的 M3，不能直接进入主训练。

可选的逐位置 w_t oracle 必须单独标为 `oracle_non_deployable`：它使用该位置真 u，只用于诊断候选 span 的上限，不参与层选择或训练 target，不得将其 R² 当作全局模型的结果。本轮不穷举 2^L 个 block 子集，也不新增位置预测网络。

### 3.3 M3：直接比较最终 reconstructed targets

主指标为 d_t=TV(Q̂_t,Q_A,t)，按 §2.3 相同构造计算，报告 token 均值／中位数／P90、逐题均值和计数。β固定4，不为某候选在 held-out 上调强度。

相关性使用显式 top-k 上按 §3.2 同一 ω 中心化的 û 与 u：

$$
Corr_\omega=\frac{\sum_t\langle\mathcal Z_t(\widehat u_t),\mathcal Z_t(u_t)\rangle_{\omega_t}}
{\sqrt{\sum_t\|\mathcal Z_t(\widehat u_t)\|_{\omega_t}^2\sum_t\|\mathcal Z_t(u_t)\|_{\omega_t}^2}}.
$$

它是预先定义的 pooled、逐位置去公共偏移的加权相关性，不是把所有原始 logits flatten 后任意调用 Pearson。低能量／零方差按 undefined 处理。另报每位置相关性的分布作为辅助，不用平均相关系数代替主定义。tail 另报 log-ratio 误差和目标质量误差，不混入回归的显式坐标。

保留 v0.1 门槛：**d_t 中位数 <0.10 且 Corr_ω>0.7**。这些是筛选启发式，不保证训练性能；还必须通过 §3.4–§3.6 的相对基线检查。

### 3.4 M4：失败 IR 与不纠正基线都要比较

预先固定两个 target 基线，在完全相同位置重算：

- 不纠正：Q_base=P⁺。若它也轻易通过绝对 TV 门槛，说明样本大部分位置本来就不需要明显重构。
- IRfl4：[12,20)、λ=4、**完整词表重构后聚合**，复现已运行 full-tail 的 target，而不是重写成 tail=0 的版本。[D2]、[C1]

候选必须在 token 平均 target-TV 上优于两者，不能只比失败 IR 好。报告逐题配对误差差值及按原图／题组 bootstrap 的95%区间（固定seed，建议1000次）；若区间包含0，写“证据不足”，不称明确优于。token 不能当独立题目计算置信区间。

### 3.5 M5：答案位置与生成控制分层

优先使用现有评测格式识别 `<answer>...</answer>` 内容；标签缺失时，仅按预先规定的终局答案格式提取。不得把推理正文第一次出现的 A/B/C/D 自动当最终答案。将字符 span 可靠映射回生成 token IDs，排除标签本身并记录多 token 答案。

在可定位的答案位置重复 M1/M3，保留原“相同门槛”的要求，同时报告解析覆盖率、无标签、空答、截断、多答案歧义的比例。不能定位则记缺失，不能算零误差。答案样本不足或相关性 undefined 时，结论为待补充，不强行判通过。

另外单列停止 token／结构标签位置与其余回答位置，以免只匹配语法或提前结束模式。不得用 GT 答案内容选择读出层；格式解析不等于正确性标注。答案子集是预定诊断，不预先断言 A 的全部 V\* 收益集中在这些位置。

### 3.6 新增防伪通过检查：真实视觉差分活跃的位置

按 fit 上 a_t=TV(P⁺_t,P⁰_t) 的75%分位数设阈值 a_*，锁定后用于 selection／held-out。除完整数据外，在 a_t≥a_* 的子集报告 M3 与两个基线；零阈值／退化分布时明确说明该分层没有区分力。

该子集只用于离线分析，不成为正式训练的 null-based gate。主候选应在活动子集的 token 平均 target-TV 上也优于两个基线；否则标为“总体接近但关键变化不足”。报告分位数统计、计数，不只报全体中位数。

### 3.7 数值例子：prior 近似与 target 近似不能互相替代

合成三 token 词表：p⁺=(0.30,0.60,0.10)、p⁰=(0.10,0.80,0.10)、p̂=(0.12,0.78,0.10)，β=4。

| 量 | 真 null | 内部候选 |
|---|---:|---:|
| prior 的 striped/diamond 概率比 | 0.125 | 约0.15385 |
| reconstructed target 的 striped/diamond 比 | 128 | 约55.7832 |
| target 完整分布 | (0.988213,0.007720,0.004067) | (0.974222,0.017464,0.008313) |

prior-TV=0.02，target-TV约0.013991；这个例子没有严重失真，但表明小 prior 变化仍能明显改变 pairwise odds。真实是否可接受应直接测 M3，不由 M1或相关性单独推断。三 token 例子中的数字不属于真实实验结果。

## 4. 选择规则、决策与训练衔接

### 4.1 锁定候选的流程

1. 在 fit 完成读取、活动阈值和全局 w 拟合；固定候选族、τ、β、tail 与数值配置。
2. 在 selection 先检验 M1，再按 M3/M4/M5/活动位置要求筛选；在通过者中，按 token 平均 target-TV 从小到大选一个主候选。误差数值持平时优先部署开销更低者，再按预登记名称排序，禁止人工挑测试表现更好的层。
3. 仅将这个主候选和两个固定基线带到 held-out。确认同样的判读要求，不在 held-out 上回头选第二个候选。
4. 通过后输出锁定的 layer／normalization 或 w 文件、其哈希与全部配置。不得在确认后偷偷用完整400题重新拟合 w，除非另设独立确认集并记录新版本。

### 4.2 决策表

| 观察 | 本轮允许的结论 | 后续建议（不自动执行） |
|---|---|---|
| selection 与 held-out 的 M3/M4/M5 及活动位置检查通过 | 找到在初始 prefix 上的可部署近似 | 实现 internal_prior，小规模工程验证后单次训练 |
| M1 好、M3 不好 | prior 的平均接近不足以保住重构 | 不进入主训练；记录放大和 tail 误差 |
| 全局 M2 R²高、M3仍失败 | 显式纠正可拟合，但完整 target 未通过 | 检查 tail、尺度与泛化；加权 IR 须另列探索，不绕过门槛 |
| 逐位置 oracle 高、全局 w 低 | span 有局部表达能力，固定系数不足 | 不能称已找到单前向方法；本轮不新增预测网络 |
| held-out 失败 | 选择结果未在本轮留出数据确认 | 不继续复用该 held-out 调参；扩展研究须新计划 |
| 所有候选均不合格 | 已测读出族缺少足够证据 | 可结束该候选族投入；不宣布所有线性读出或单前向方法不可能 |

v0.1 中“高R²直接跑加权IR”的分支在本版不作为主线自动放行。若确需尝试新 λ，必须先在 selection 锁定，不使用 held-out 或 TB/V\* 调 λ。

### 4.3 正式训练只使用内部参照

候选确认后，拟议新增 `teacher_target_mode=internal_prior`。一次冻结 real teacher 前向得到 p⁺ 与 p̂；按 §2.3 得到 Q̂，再交给原 JSD、mask、IS、token聚合与优化器。

$$
\ell_t=JSD\big(\operatorname{sg}(\widehat Q_t),P^S_t\big).
$$

β=4、JSD α=0.5、top-100＋tail、pair 视图、teacher 冻结和原训练预算先保持不变。不调用 null/text teacher，不把 student 放进 anchor 或 reference；student 只提供 rollout、支持集和待更新预测。固定输入与支持集时，target 不随 student 参数漂移。

开启在线禁 null 守卫并检查计数。离线需要 null 的进程必须与在线训练配置隔离，不全局移除守卫。模型不会训练新 LM head 或隐藏态转换网络。全局 w 固定加载，不根据在线 null 或 student discrepancy 更新。

### 4.4 性能验证与后续 prefix

沿用原计划：首个完整探索运行使用既有训练数据、seed42、n2、lr2e-6、51步等配置，精确值以运行 manifest 与匹配基线为准；评测按现有 TB／V\* judge 流程及 {30,40,50} 规则。TB>50 且 V\*≥92 时再补重复；“92–93量级”不是超过93就失败。

与同配置的普通 pair privileged OPD 和 A 比较。历史 V0 使用 hide 时只能作历史参照，不声称严格匹配消融。先不加入 history／future，也不靠禁止 EOS 或改最小输出长度制造成功。

初始 greedy prefix 上的近似不保证后续 on-policy 泛化。若已有后期 student rollout，可在不改已锁定候选的情况下作额外、明确计费的离线 real/null 复核；没有则在新训练中验证最终表现，不虚构覆盖。该复核不进入训练目标。重复同seed结果与不同seed结果分开命名，多个checkpoint不是独立训练样本。

## 5. 成本、工程接口与执行安排

### 5.1 离线实现文件（拟议，不是已存在功能的承诺）

| 位置 | 修改内容 | 保持不变 |
|---|---|---|
| `scripts/probe_layer_prior.py` | 三视图评分、分组、逐层读出、M1–M5、全局拟合与报告 | 不调用 optimizer、不训练模型 |
| `verl/utils/teacher_residual.py` 或独立 probe helper | 新的多边界只读采集；复用已核对 norm／head 定义 | 原 ResidualCapture 接口和 IR 默认路径 |
| `ray_trainer.py` 的图像工具 | 离线复用 null_scope=last 的均值色逻辑、记录输入结构 | 不更改正式 A 的 null 构造 |
| tests（如 `scripts/test_layer_prior_probe.py`） | 数学、tail、回归隔离、位置、梯度与生命周期测试 | 不替代真实GPU兼容检查 |

现有 ResidualCapture 只保留区间端点和最终状态，且要求四个 hook 各触发一次。[C1] 采集所有 h⁰…hᴸ 时必须新增独立 collector 或显式可选模式，不把“已有四个hook验证过”当作新 collector 已通过。使用每块 pre-hook 加最终 norm pre-state，或等价的边界采集；最后 hᴸ 不能拿归一化后的 hidden state 代替。

### 5.2 vLLM 与训练侧分工

**vLLM rollout 核心不需要修改。** 离线 probe 使用 HF 评分；生成可复用现有 vLLM base rollout，但必须记录 greedy 配置并取得相同 token IDs。全部层数据不通过 rollout RPC 返回。[C4]、[C5]

候选通过后才扩展 verl：

- `workers/config/actor.py`：新增带验证的 internal_prior 配置（kind、layer或固定weights、norm口径、β、tail策略），默认保持 legacy。
- `trainer/ppo/ray_trainer.py`：按 target mode 禁止 online null 图像与 text 分支构造，保持 real pair 输入。
- `workers/actor/dp_actor.py`：teacher 的一次 no_grad 前向取得所需状态和正常logits，输出独立的 target 字段；支持 packing／response 对齐。
- `trainer/ppo/core_algos.py`：单独 internal_prior target 路由，保留 JSD／IS／聚合；不得复用 S/S2 的交换槽位来误改 anchor。
- launcher：实验 tag 编码 prior种类、层／权重hash、norm与tail；不覆盖旧 checkpoint或W&B运行。

新配置名和文件名是本版实施建议，不意味着 API 已存在。FSDP 权重可用期、FlashAttention／融合算子、compile兼容和SP必须逐项验证；复用当前IR的SP1作为第一版范围，其他SP设置未验证则显式报错。只读采集不需要完整 attention 矩阵，不能因此承诺所有内核已经兼容。

### 5.3 隔离计算预算与存储预算

v0.1 的“单卡约1小时”是估计，不是测量。除400题的三种评分，还要计入全图 greedy 生成、必要的支持集评分、每层词表投影及 own-norm 候选；若额外复查prefix，也单独计数。先用8–16个来自fit的样本验证正确性与吞吐，不据此调整方法的held-out阈值。

缓存仅保留每题回答预测位置。以T=1024、L+1=33、d=2560为例，BF16边界状态为173,015,040 bytes（165 MiB）；转为FP32则330 MiB，尚不包括权重、logits与临时张量。现有IR `_select()` 会转FP32，不能直接套用BF16估计。[C1]

逐层读出不能保存 `[T,L,V]` 全词表数组。按回答位置和词表分块，在线累计完整logsumexp、显式K项与tail，再立即释放临时logits。M2只需显式K上的block贡献和L×L统计；可使用gather后的head行分块，避免一次物化 `[T,K,d]` 的大数组。需要确切tail时使用补集logsumexp或经验证的log1mexp，绝不能只在K上归一化。

按 fit→selection→held-out 顺序处理：fit结束后固定w，再在后两份样本的首次real前向中评价它，减少重复前向。若实施选择其他两遍流程，必须把额外成本写进报告。

### 5.4 输出和仓库卫生

建议输出到确认被忽略的 `eval/probe_layer_prior/<run_id>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 模型／tokenizer／代码hash、环境、模板、τ、seed、split、tail、输入与mask规则 |
| `per_layer.csv` | M1/M3/M5的逐层与norm族统计、基线、计数和误差区间 |
| `regression.json` | 固定全局w、η、秩、R²、fit/selection/held-out身份、oracle单列 |
| `metrics.json` | M1–M5、活动子集、无效相关性计数、解析覆盖、成本与数值诊断 |
| `selection.json` | 唯一主候选、锁定时间与hash、门槛及每项通过／失败原因 |

不得提交权重、图像、parquet、完整rollout、hidden states、logits、原始日志或凭证。[G1] 结果完成后只将可追溯的汇总和图表说明写入文档；小型校准系数是否入库另经确认。未确认忽略规则前，不把生成目录纳入git。

### 5.5 验收阶段与权限

| 阶段 | 交付 | 条件 |
|---|---|---|
| P0：本文 | v0.2完整计划 | 仅文档commit，未运行探针 |
| P1：离线实现 | 脚本、单测、code review | §5.6数学与输入对齐通过 |
| P2：小样本工程检查 | 吞吐、显存、hook不改anchor | 明确能否按预算跑400题 |
| P3：400题诊断 | 锁定候选与held-out报告 | 不借测试结果回头选层 |
| P4：训练接入 | 单前向target路由、legacy回归 | 在线null计数0、teacher冻结、梯度正确 |
| P5：训练评测 | 首轮及条件性重复、实际总成本 | 接近A是待验证结果，不预填 |

原计划中的 hold／orch／预计结束时间仅是历史排程。本次不取消、替换或新开任何GPU作业；执行前重新查询实际状态并确认。后续代码拟在 `sup` 开发并review，计划及结果归档仍在 `negative_history`；不在文档分支顺带改训练代码。

### 5.6 必须具备的测试（全部是待实现验收要求）

1. **只读采集：** hook开关不改变p⁺；每个边界只触发预期次数，异常后移除hook，buffer不跨样本复用。
2. **读出数学：** hᴸ重建实际最终logits；偏置不重复、τ只缩放一次、1+w_norm正确；RMSNorm(Δh)作为错误变体必须被检出。
3. **逐层控制：** ℓ=L得到p⁺；共享D的完整词表prior重构与后缀IR相等；own-D不强制该恒等式。
4. **tail：** 显式列加tail约为1；先聚合再重构严格匹配A；非零tail log-ratio案例与置零版本不同；极小tail、K=V和padding有限且计数正确。
5. **β=0与常数：** β=0回到普通privileged OPD；给完整词表纠正加共同常数不改target；不在聚合后只改部分列来假装这个不变性。
6. **回归：** 合成已知共享w在留出样本上可恢复；每位置加常数不改中心化拟合；退化能量标undefined；fit之外的数据不参与H/b或阈值计算。
7. **位置与答案：** 不同prompt长度、左右padding、packing、首回答token、EOS、多token选项及缺标签正确处理；不把正文选项字母误作最终答案。
8. **梯度：** 两条完全不同的teacher构造路径生成相同detached Q时，student JSD梯度相同；target不接收梯度，student预测分支有梯度。
9. **聚合：** 不等长microbatch以sum/count形成token加权值；指标名含max/argmax不触发误聚合；分位数不靠平均局部中位数冒充全局值。
10. **训练接入：** legacy A/IR/S等不被默认改变；internal_prior禁null/text、禁student anchor、拒绝缺失系数或错误归一化；同一prefix下目标可复现。

## 6. 结果与判读记录（2026-10-05 13:05，400 题全量，run `full_1005_1234`）

> 代码：sup `fcdf7e7`（`verl/utils/layer_prior_probe.py`、`scripts/probe_layer_prior.py`、`scripts/test_layer_prior_probe.py`，§5.6 第 1–9 项单测全过，第 10 项训练接入未实现）。汇总文件：`docs/negative_history/probe_layer_prior/full_1005_1234/`（metrics / selection / regression / per_layer / manifest / held-out post-hoc）；positions.npz、完整回答、hidden states 不入库。
> 环境：Qwen3.5-4B（HF snapshot 851bf6e8…）、transformers 5.5.0、torch 2.10、bf16 + sdpa、单张 RTX Pro 6000（hold 20814117，与 IRfl2 评测共卡）；fp32 matmul 精度 `highest`。

| 项目 | 实际 |
|---|---|
| 数据 | `train_6karmA_pair.parquet`（sha1 见 manifest）seed 42 抽 400 题；原图全部唯一，分组退化为按题：fit 200 / selection 100 / held-out 100 |
| 回答 | base 全图 HF 贪心，max 1024；截断 9/400；答案位可定位 384/400（末行"[Correct/Final] answer: **X**"360、行首字母 24），missing 11，unmappable 5 |
| 位置数 | fit 29,395 / selection 13,368 / held-out 12,605；答案位 189 / 97 / 98；活跃位（a\* = fit 的 75% 分位 = 0.079）7,349 / 3,416 / 3,155 |
| 重建核对 | ℓ=L 读出 vs 模型 bf16 logits：mean TV 0.003–0.005（断言阈 0.02），hook 各触发一次 |
| 成本 | 1,834 s（生成 2.5 s + 四次评分 1.3 s + 逐层读出 0.7 s / 题），峰值显存 15.5 GB，0 异常 |
| 参照尺度 | TV(P⁺,P⁰) token 均值 0.065（中位 0.017，75% 0.079，90% 0.19，95% 0.29）；TV(P⁺,P_text) 0.23–0.25；TV(P⁰,P_text) 0.23–0.24。**p⁰ 离 p⁺ 比离语言先验近 3.5 倍** |
| 基线 | d_base = TV(P⁺,Q_A)：all 0.27–0.28（中位 0.12），active 0.56–0.57；d_IRfl4：all 0.42–0.43，active 0.70–0.71（比不纠正还远） |
| **M1** | 无任何候选比 p⁺ 自己更接近 p⁰。final-D：ℓ≤20 TV 0.997（质量几乎全在 tail 桶，词表坐标下未成形），ℓ=24 0.98，28 0.84，30 0.69，31 0.36，32(=p⁺) 0.065；own-D 同形状（31 层 0.34）；活跃位上 31 层 0.52 vs p⁺ 0.21。对 p_text 的距离曲线同形状，没有一层比 p⁺（0.23）更像语言先验 |
| **M2** | 全局 w（fit 29,395 位置，满秩，cond 4.6e3）：in-sample R²_corr = **0.008**，selection 0.007，held-out 0.007；32 个权重全在 ±0.2 内。真 u 与各 block 贡献的任何固定线性组合基本正交 |
| **M3** | selection 上 d 的 token 均值最小者 = final_32 = 不纠正基线（0.274），global_w 0.275；所有逐层候选 ≥0.34，中位 ≥0.12；Corr_ω 全部在 −0.06…+0.09。无候选过 M3 |
| **M4** | 无候选优于不纠正基线；全部优于 IRfl4 仅因 IRfl4 本身比不纠正更差 |
| **M5** | 答案位 TV(P⁺,P⁰) 仅 0.012–0.021，d_base 中位 0.000：对 base 自己的贪心回答，crop 很少改变答案字母的分布；该子集上无候选过门槛（31 层 d 均值 0.64–0.68） |
| 活跃位 | 所有候选 d 均值 0.65–0.85，全部劣于不纠正（0.56） |
| held-out | 无锁定候选，未做确认；其余候选的 held-out 数字只写入 post-hoc 文件，未用于任何选择 |
| **判定** | `no_candidate_passed_selection`：按 §4.2 末行，**结束对"单次 real 前向的线性读出族（逐层 final-D / own-D、全局 block 权重）"的投入**；不宣称所有单前向方法不可能 |

### 6.1 两个样本（selection 集，逐 token）

**样本 1，row 719**（"What color is the first letter on the top left of the sign?"，GT C），回答第 5 个 token，前文 "The sign on the" → 下一词。这是 selection 集里 crop 改变教师分布最大的位置：

| 分布 | top 词（概率） |
|---|---|
| P⁺（看 crop） | left 0.98, far 0.01, top 0.008 |
| P⁰（不看 crop） | **top 0.93**, left 0.04, far 0.03 |
| P_text | tail 0.39, left 0.24, top 0.14 |
| Q_A（β=4） | left 1.00 |
| p_16 / p_24 读出 | tail 1.00 / tail 0.998（未成形） |
| p_28 读出 | tail 0.71, left 0.27 |
| p_30 / p_31 读出 | left 0.69 / **left 0.976**（≈ p⁺，不是 p⁰） |
| global-w 先验 | left 0.98（≈ p⁺） |

TV(P⁺,P⁰)=0.945。"不看 crop 的信念是 top"这件事在 real 前向的残差流里**任何一层都读不出来**：中层读出还没成形，晚层读出已经被 crop 证据覆盖成 left。p⁰ 不是本次计算的一个子部分，而是另一次计算的结果。

**样本 2，row 1706**（"primary color of the building on the far left edge"，GT C），回答第 1 个 token：

| 分布 | top 词（概率） |
|---|---|
| P⁺ | The 0.57, C 0.39 |
| P⁰ | C 0.58, The 0.31 |
| Q_A | **The 0.986**, C 0.012 |
| p_31 读出 | C 0.84, " C" 0.06（top-1 像 p⁰，TV 仍 0.35） |
| Q̂(p_31) | The 1.00（d=0.014，碰巧同向） |
| global-w 先验 / Q̂ | The 0.60, C 0.36 / **C 0.51, The 0.45**（d=0.54，方向反了） |

这一位上 A 的目标把"先解释再作答"从 0.57 推到 0.986——A 的 u 在首 token 上编码的是作答风格而非视觉证据。31 层读出在 top-1 上像 p⁰，但同题同层在其他位置（样本 1）又完全像 p⁺：像不像 p⁰ 不是层的属性，而是逐位置随机的，所以逐层候选的 Corr_ω 在全体位置上为零。

### 6.2 结论

1. 在 Qwen3.5-4B 的单次 pair 前向里，响应位置的残差流不保留"没有 crop 时会怎么想"的信息；晚层读出 ≈ p⁺，中早层读出在词表坐标下未成形，block 贡献的任何固定线性组合与真 u 正交（R² 0.008）。
2. 这也解释了 IRfl4：它的目标离 Q_A 比不纠正更远（0.43 vs 0.27），放大 [12,20) 贡献放大的是与 u 正交的方向。
3. p⁰ 离 p⁺ 很近（0.065）而离语言先验很远（0.23）：A 的 null 不是语言先验；"找内部的语言先验来替代 null"这一表述与 A 实际使用的参照不同，若要检验"语言先验参照经训练后是否也能到 A 的效果"，应直接用精确的语言先验（纯文本教师前向或 `null_scope=all`）做一次训练上界，而不是在内部找近似。
4. 本轮排除的只是线性读出族；用户已明确排除 tuned lens / 新增解码头方向。

## 7. 风险与解释边界

### 7.1 相同detached target意味着相同student梯度

固定student参数、输入／prefix、支持集、mask、IS权重和聚合，若Q̂=Q_A，则：

$$
\nabla_\theta JSD(\operatorname{sg}(\widehat Q),P_\theta)
=\nabla_\theta JSD(\operatorname{sg}(Q_A),P_\theta).
$$

teacher内部共享上游计算不会凭空增加一条student梯度路径。真实风险在于target不完全相同、误差在不同位置分布不同，以及训练后prefix改变。小TV也不是任意参数化下梯度差异小的保证，必须单独验证。

### 7.2 内部prior不是被识别出的纯语言模块

所有h来自real输入，final-D还含最终视觉条件信息。自己的D_ℓ也可在单次前向中计算，只是不同读出口径。接近p⁰只能说明预测近似；对高频token的偏好、层间未校准和输入结构差异都可能产生近似表象，不能通过一个TV值确认“语言层”。

### 7.3 线性拟合不是完整因果归因

共享D下的加性读出是实际计算分解，不是重新删除某个block后的网络输出。相关block造成的大正负系数可能互相抵消；R²高不等于那些block唯一携带crop信息。逐位置oracle通过也不代表能在未知prefix上求出不依赖null的系数。

### 7.4 有限数据结论与部署泛化

400题、base greedy、固定层族的检验不能证明所有单前向或线性方法可行／不可行。未测的非线性读出也不能被预先称为救命方案。近似失败不唯一归因于attention重算；近似成功不保证TB/V\*超过A，也不保证同一参数在其他模型架构上适用。

### 7.5 研究贡献和费用要如实表述

本路线的潜在贡献是“用受控离线检验发现并固定可部署的内部参照，在线保留teacher anchor而省掉null前向”。不是通过重新命名后缀IR就获得全新公式，也不是全程无null监督。正式训练无null的节省、离线null校准成本、额外head投影与存储成本必须分别报告。

## 8. 来源索引

[D0]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/e58132609fc4a7af9bbb00c6eff174a522a051b6/docs/negative_history/ir_layer_prior_probe_plan_2026-10-05.md
[D1]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/cc115a48ccb17c0144ded5a68139f2174e85347e/docs/negative_history/teacher_internal_residual_reconstruction_plan.md
[D2]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/6a6ef866e2ce675d011b33b53d8786756fccf9f6/docs/negative_history/ir_results_2026-10-05.md
[D3]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/6a6ef866e2ce675d011b33b53d8786756fccf9f6/docs/negative_history/null_free_line_summary_2026-10-04.md
[C1]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/d379de1e6c33f71bd6e565cb5f5ac77cec0b26fe/verl/utils/teacher_residual.py
[C2]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/d379de1e6c33f71bd6e565cb5f5ac77cec0b26fe/verl/trainer/ppo/ray_trainer.py
[C3]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/d379de1e6c33f71bd6e565cb5f5ac77cec0b26fe/verl/trainer/ppo/core_algos.py
[C4]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/d379de1e6c33f71bd6e565cb5f5ac77cec0b26fe/verl/workers/actor/dp_actor.py
[C5]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/d379de1e6c33f71bd6e565cb5f5ac77cec0b26fe/scripts/run_visual_counterfactual_unit.sh
[G1]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/e58132609fc4a7af9bbb00c6eff174a522a051b6/AGENTS.md
