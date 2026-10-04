# Candidate S｜无 visual-null 的学生自参照蒸馏：详细设计与实施计划

**版本：** v0.1  
**整理日期：** 2026-10-04  
**文档分支与目录：** `negative_history` / `docs/negative_history/`  
**状态：** 候选方法规格与工程计划；本文不构成已完成 S 训练、性能验证或新颖性验证的报告。

## 0. 范围、来源与状态边界

本文整理了 S 的 motivation、公式、数值例子、梯度路径、支持集近似、既有训练循环及拟议实现步骤。与之前 PPO/history 文档不同，本方案沿用 OPD/JSD，不引入新的策略梯度目标。

- **设计来源：** `negative_history@30e1c3b7dbf80e80c5206c05b93ecf548a73ba4c` 的 summary §7.7。[D1]
- **代码核对基线：** `sup@bea75c6bb399feeb731713f9592f86989618ac48` 的 actor、loss、trainer 和 launcher。[C1–C6]
- **已存在：** Aha 的 rollout、real/null teacher 评分、top-k＋tail、JSD、IS correction、梯度累积与优化器路径。
- **待实现：** `counterfactual_reference=student`、端到端跳过 null 构造与前向、student-reference target 及测试。
- **本文新增的是文档，不修改训练代码。** 示例概率、梯度和 loss 是合成算例，不是实验结果。

若执行时代码已经更新，先记录新 commit 与本基线的差异，再接入；不得将文档中的建议配置视为当前已经存在的开关。

## 1. 研究目标与第一轮固定约束

### 1.1 要回答的问题

在不进行任何 visual-null 前向的条件下，仅利用已有 privileged teacher 与当前 full-image student 分布，能否保留接近 Aha 的有效监督与任务表现？

省去 null 是明确的计算目标；“得到的差值仍然是纯视觉证据”不是已成立结论。S 将参照从条件视觉对照改为学习者当前状态。

### 1.2 第一轮只改变参照

固定 teacher、训练数据、视图、student 可训练参数范围、优化器、采样、JSD、支持集、IS、聚合及评测流程。仅将重构中的 null 分布替换为当前 student 分布。[D1]

不增加以下内容：

- null 预热、null 预测头、偶发 null 校准或为 S 诊断而执行 null 前向；
- 新 LM head、head-only 训练、hidden-state 对齐、冻结 student 主干等额外结构修改；
- PPO/GRPO 新目标、GU、critic、history/future、X1/X2、floor、ST、tanh、target sharpening；
- 候选轨迹枚举、teacher takeover、改写 student prefix；
- 新的超参数扫描、额外训练步数、独立科学 probe 或常数剂量对照。

Student 的 `requires_grad` 范围继承 A，不擅自改成“只训练输出层”或重新定义哪些模块冻结。

## 2. 输入与符号：三个分布不能混用

固定问题 x、全图 I、privileged crop C 与已生成前缀 y_<t。

| 符号 | 定义 | S 是否计算 |
|---|---|---|
| p_t^S | 当前 student θ，看全图 I、问题 x、同一个前缀 | 是，保留预测侧梯度 |
| p_t^+ | 冻结 teacher φ，看 [I,C]、问题 x、同一个前缀 | 是，no_grad |
| p_t^0 | 同一个 teacher，看 [I,blank(C)]、相同文本条件 | 否；只用于解释原 A |
| q_t^S | 使用 p_t^+ 与 detached p_t^S 构造的监督分布 | 是，无 target 梯度 |

当前 pair A 使用 `counterfactual_null_scope=last`：只将最后一个 crop 替换为 mean-color 图，全图仍存在。[D1][C5] 因而原 u=log p^+−log p^0 是 crop 替换的条件增量，不是“全部视觉与纯语言”的差异。

相同初始 teacher/student 权重不保证 p^S=p^0：全图与全图＋空白图块仍是不同输入。只有分布确实相等时，S target 才与 A target 相等。

本文 log 均指自然对数。t 是 response 相对位置，不是多模态输入的绝对 token 索引，也不是 optimizer step。

## 3. 核心 target：学习者状态相关的外推

定义：

$$u_t^S(v)=\log p_t^+(v)-\log\operatorname{sg}(p_t^S(v)).$$

其中 sg 表示 stop-gradient：不改变数值，只切断该参照路径的梯度。

$$q_t^S(v)=\frac{p_t^+(v)\exp(\beta u_t^S(v))}{\sum_w p_t^+(w)\exp(\beta u_t^S(w))}.$$

等价地：

$$q_t^S(v)\propto\frac{[p_t^+(v)]^{1+\beta}}{[\operatorname{sg}(p_t^S(v))]^\beta}.$$

第一轮 β=4，继承 A；β=0 是用于数学测试的普通 teacher target，不是新增训练臂。

生产实现以 `log_softmax(log_p_teacher + beta * (log_p_teacher - log_p_student.detach()))` 为主体，不直接计算幂与概率除法。正式数值路径需复用 A 的 dtype、tail 保护和温度约定。

`u_t^S>0` 表示 teacher 比 student 更支持该 token；不自动表示该 token 正确或其差异完全来自 crop。

## 4. Pairwise 解释与完整概率算例

对两个指定 token v、w，定义概率比 R_T=p^+(v)/p^+(w)，R_S=p^S(v)/p^S(w)。这不是 v 对其余全部词表的 p(v)/(1−p(v))。

$$R_q=R_T(R_T/R_S)^\beta,$$

$$\log R_q=\log R_T+\beta(\log R_T-\log R_S).$$

令 x_T=log R_T，x_S=log R_S，则 x_q=x_S+(1+β)(x_T−x_S)。外推发生在 log-odds 空间，不是概率空间的线性加减。

### 4.1 三 token 完整算例

| Token | Student p^S | Teacher p^+ | 未归一化权重，β=4 | Target q^S |
|---|---:|---:|---:|---:|
| striped | 0.30 | 0.60 | 9.60 | 0.987781350 |
| diamond | 0.60 | 0.30 | 0.01875 | 0.001929260 |
| plain | 0.10 | 0.10 | 0.10 | 0.010289389 |

权重和 Z=9.71875。Teacher 的 striped/diamond 比值为 2，student 为 0.5，target 为 512。512 是概率比，不是概率。

### 4.2 条件性退火与反向外推

固定 teacher odds=2、相同图片/问题/prefix，只比较不同 student 参数状态：

| Student odds | Teacher odds | S target odds，β=4 |
|---:|---:|---:|
| 0.5 | 2 | 512 |
| 1 | 2 | 32 |
| 2 | 2 | 2 |
| 4 | 2 | 0.125 |

师生接近时外推减弱；越过 teacher 后外推方向反转。x_q−x_T=−β(x_S−x_T)，因此“刹车”可能把 target 推到 teacher 的另一侧，不是稳定性保证。

整个分布 p^S=p^+ 时 q^S=p^+。只在一对 token 上 odds 相同，不等于整个分布一致。训练是否收敛到这一状态未知，不能写成“必然退火到 V0”。冻结 teacher 的 target 也只在固定输入/prefix 上固定，换 rollout 后不保证 target 数值相同。

## 5. S 不是单纯锐化 teacher

合成例子：p^+=[0.60,0.30,0.10]，p^S=[0.75,0.20,0.05]，顺序为 striped/diamond/plain。两者排序相同，但 S 得到：

$$q^S\approx[0.073044812,\ 0.451403027,\ 0.475552161].$$

Plain 的 teacher 概率只有 0.10，但 teacher/student 比值为 2；striped 比值为 0.8，所以重构可以翻转 teacher 排序。

直接锐化 q∝(p^+)^a，a>0，不改变 teacher 排序。S 还依赖 p^S，因此两者不能互称。

风险：teacher 的次要候选可能因 student 给得更低而获得极大权重。例如 p^+=0.01、p^S=0.0001 时，β=4 的未归一化权重为 10^6。这不是正确性证据，也不是发生了数值溢出才会出现的问题。

## 6. 支持集：student top-100＋tail

### 6.1 必须沿用的顺序

1. 当前 student 前向选择每个 response 位置的 top-100 token IDs；不足 100 时取词表可用大小。
2. 冻结 teacher 在完全相同的 token IDs 上 gather 概率，禁止各自 top-k 后按列相减。
3. 显式概率来自全词表归一化，保留原质量；分别添加 tail=1−sum(top-k probabilities)。
4. 在 K+1 类分布上构造 S target，再计算 JSD。[C1][C2]

Tail 是“同一个显式集合之外的全部 token”的集合概率，不是 tokenizer 中可输出的新 token。选择集合的离散索引不求导；student 显式概率和 tail 的预测路径仍保留梯度。

### 6.2 聚合顺序改变重构结果

教学 top-3 例子，显式类别 diamond/striped/plain 的 student 为 [.40,.30,.20]，teacher 为 [.20,.40,.20]。剩余 checkered/floral/dotted 的 student 为 [.05,.03,.02]、teacher 为 [.15,.03,.02]。

合并以后 student tail=.10、teacher tail=.20。S 的四类未归一化权重为：

```text
diamond: 0.0125
striped: 1.264197530864
plain:   0.20
tail:    3.20
```

归一化 target 为 [.002672826,.270318429,.042765220,.684243524]。Tail 可成为最大类别，不能把这个现象描述为“模型预测 tail 这个词”。

若先逐 token 重构再合并，三个尾部权重为 12.15、.03、.02，总和 12.20，不等于 3.20。因此该近似不等于全词表重构的精确压缩。Top-k 不为保留类别的概率比提供上界。

S 第一轮不擅自改成 teacher top-k、union support、固定 tail 质量或先全词表重构后截断；这些都是另外的方法变化。

## 7. JSD 与 stop-gradient

单位置使用 α=.5 的 JSD：

$$m_t=(q_t^S+p_t^S)/2,$$

$$\ell_t=\tfrac12\mathrm{KL}(q_t^S\|m_t)+\tfrac12\mathrm{KL}(p_t^S\|m_t).$$

- Teacher 前向 no_grad；target 计算中的 student 分布 detach；最终 q 整体 stop-gradient。
- 同一份 student 输出作为被监督预测时不能 detach。
- 混合分布 m 中的 student 路径要保留梯度，不能连 JSD 一并置于 no_grad。
- 不对采样 response IDs 求导，不执行 through-sampling gradient。

以 §4.1 的完整精度 q 为固定 target：

| Student 预测 | 对该固定 q 的 JSD |
|---|---:|
| [.30,.60,.10] | 0.320023011 |
| [.35,.55,.10] | 0.285181663 |

如果下一次用新的 student [.35,.55,.10] 重新构造 target，则 q_new≈[.976159361,.005002558,.018838081]。固定的是本次反传的 target 路径，不是把最初的 q 永久缓存。

以下只表示 target kernel 的拟议逻辑，不是可直接替代完整训练函数的代码：

```python
# log_s/log_t: same support, normalized, same scoring convention
with torch.no_grad():
    u = log_t.detach() - log_s.detach()
    log_q = torch.log_softmax(log_t.detach() + beta * u, dim=-1)
# Pass log_q and differentiable log_s to the EXISTING JSD implementation.
```

不使用 in-place detach 修改原 student tensor。不自动加入新的 u clipping、probability floor 或 target 温度。

## 8. Rollout、逐位置监督与路径统计

Student 先生成完整 response，例如 `The pattern is diamond.`。在预测 diamond 的位置，prefix 是 `The pattern is`，target 可以主要支持 striped。生成记录不是正确答案标签。

下一个位置仍使用原始 prefix `The pattern is diamond`，不将它人工改成 striped。每个位置独立得到自己的 teacher/student 分布和 q，所有有效位置参与分布蒸馏。参数更新后的重新生成才可能出现不同回答。

可选记录 realized-token 统计：

$$c_t^S=[\log p_t^S(y_t)-\log p_t^+(y_t)]_+.$$

它表示“学生对实际选择的 token 比 teacher 更支持”，不是原来的 crop-opposition。若保留此日志，必须用全词表归一化的精确 realized log-prob，不能用 tail 概率代替某个尾部 token 的概率。第一轮不把 c^S 再变成 history/future 权重或额外损失。[D1][D2]

Response 相对位置、causal shift、teacher response start、图像模板、EOS、stop 与 truncation 处理均继承 A；不在本实验里重新定义 mask。诊断只能说明观察到的后缀，不能把 max-length 截断当成正确结束。

## 9. 已核对的训练框架：原始 JSD 不等于最终 backward loss

### 9.1 现有代码链路

| 文件/函数 | 已存在的行为 |
|---|---|
| ray_trainer.py / fit | Student rollout、repeat/union、response mask、old-logprob 与 rollout correction、构建 teacher batch |
| dp_actor.py / update_policy | Student forward；teacher no_grad forward；teacher 使用 student top-k IDs；调用蒸馏 loss |
| core_algos.py / compute_self_distillation_loss | 支持集与 tail、Aha target、JSD、IS 加权、agg_loss |
| dp_actor.py / _optimizer_step | 梯度裁剪、非有限梯度检查、优化器更新 |

以上来自 [C1–C6]，不是本文重新发明的 trainer。S 需要替换其中 target 参照并关闭 null 路径。

### 9.2 保留既有 IS correction

已核对 launcher 设置 `self_distillation.is_clip=2.0`、`rollout_is=token`、阈值 2.0。[C3] 既有 loss 中，对实际 sampled token：

$$r_t^{SD}=\min\{\exp(\operatorname{clip}(\operatorname{sg}(\log p_t^{current}(y_t)-\log p_t^{old}(y_t)),-20,20)),2\}.$$

随后在字段存在时再乘预先计算的 rollout IS 权重 r_t^rollout。它们是 student 评分之间的 correction，不是 u^S，也不是视觉正确性指标。

$$\ell_t^{weighted}=r_t^{SD}r_t^{rollout}\ell_t.$$

示例：原始 JSD=.32，r_SD=1.5，r_rollout=.8，得到 .384。不要把分母换成 sum(IS weights)；继承当前 agg_loss 的有效 token 分母与外层缩放。

`old_log_probs` 不必等于 rollout engine log-probs。当前 actor 在满足自己的 on_policy 分支条件时会将 old 设为 current.detach()，此时第一项 ratio=1。其他路径可重算或复用旧评分。[C1][C5] S 的词表参照始终是当前可训练 forward 的 detached 分布，不使用只有 [B,T] 的 old_log_probs 冒充词表分布。

### 9.3 聚合与 microbatch 边界

`raw_per_token_loss=kl_loss.sum(-1)` 是对支持集求和。之后加 IS、mask，再调用 agg_loss。无附加权重、单一张量的教学情形才可简写为 sum(M*JSD)/sum(M)。

实际 agg_loss 可接收 global token count；缺省使用局部 mask.sum()；actor 外层还做 microbatch scaling。固定 microbatch 分支逐次 backward 累积梯度，完成一个 mini-batch 后再 optimizer.step。[C1][C2]

因此不能仅凭名称 token-mean 宣称当前多卡执行必然等价于全局所有 token 等权。接入 S 时记录并回归分母、DP/SP、microbatch scaling，不在该臂顺便修订 A 的聚合。

`ppo_*` 参数名不代表此路径新增 PPO clipped-advantage loss。纯 vopd 样本使用蒸馏路径；缺少 teacher 输入的 fallback、其他 entropy/KL 开关是否触发，应记录现有配置，不能偷偷改变。

## 10. 与 Aha 的信息差异，以及应撤回的过强表述

### 10.1 师生一致但共同偏错

合成分布按 striped/diamond/plain 排列：

```text
student:             [.30,.60,.10]
privileged teacher:  [.30,.60,.10]
null teacher (A only):[.10,.80,.10]
```

S 因 p^S=p^+ 而 q^S=p^+，此位置 JSD=0。Aha 的 stripe/diamond odds 为 .5×4^4=128，仍能提供条件视觉纠正。本例只是说明信息不等价，不要求在 S run 中计算 null。

### 10.2 差值的精确分解

仅为解释，加入未在 S 中计算的 p^0：

$$\log p^+-\log p^S=(\log p^+-\log p^0)+(\log p^0-\log p^S).$$

第二项包含模型、校准、学习进展及输入布局差异。省掉 null 并未在代数上消除混杂。

### 10.3 写作与实验解读边界

| 不应写成已知事实 | 应采用的表述 |
|---|---|
| 第 0 步 S 必然等于 A | 只有相同支持集/温度下 p^S=p^0 才相等；相同权重不充分 |
| S 必然随训练退火到 V0 | 当师生分布接近时外推减弱；不保证单调或收敛 |
| S 的刹车保证稳定 | Target 可反向越过 teacher，参数更新稳定性另行验证 |
| Teacher–student gap 是纯 visual evidence | 是 privileged teacher 与 learner 的偏好差距 |
| Top-k 已经控制所有极端比值 | 它压缩分布；不是所有比值的上界 |
| Target TV 更大证明更尖 | TV 衡量距参照的差异，熵/最大概率另测 |
| S 是唯一可能的零 null 方案 | S 是当前选定的最小改动候选，不作排他性声明 |
| Target 更新证明 student 超越 teacher | 它是训练信号，正确性与最终泛化仍由任务评测判断 |

“自参照/自动退火”的命名本身不构成独立 novelty。本文不进行新一轮文献查新；正式写作需核对已有 teacher–student extrapolation 与 Aha 相关消融，不能宣称公式首次提出。

## 11. 拟议工程改动：确保 null 真正不再执行

### 11.1 配置解耦

新增建议字段 `counterfactual_reference: null | student`，默认 null。读取默认字段不得改变既有 A 的数学运算顺序；默认回归要求同输入下 target/loss/gradient 与原实现一致。

当前代码用 `counterfactual_null_mode is not None` 同时控制 null forward 与 target reconstruction。S 不能仅清空 null_mode，否则会错误退回 V0；也不能仅替换 loss 内的 u，却仍在外面计算 null。

内部显式区分：

- 是否进行 target reconstruction；
- reconstruction 的 reference 是 null 还是 student；
- 是否需要构造和执行 null teacher 输入。

若 reference=student，任何 null 构造、null teacher 调用、null 缓存读取及 null calibration 都应被跳过。若存在遗留 null-only 参数，应记录为未使用或以明确守卫拒绝；不能静默回退到 null。

### 11.2 模块任务

| 位置 | S 所需修改 | 不能改变的部分 |
|---|---|---|
| actor config / YAML | 新 reference enum、校验与输出 manifest | A 默认值与现有参数含义 |
| ray_trainer.py | Student 模式跳过 mean-color null 数据构造/校验 | 原始 student rollout 与 teacher-real pair 输入 |
| dp_actor.py | 跳过 null forward；继续 real teacher no_grad；取得当前 student logps | top-k IDs 对齐、IS、optimizer 与梯度累积 |
| core_algos.py | 参照选择：teacher-null 或 current student.detach()；复用 JSD | tail/温度/数值保护及 A 分支 |
| launcher | 独立 S 实验名、继承 A 配置、新字段 | 不改 A 脚本的默认行为、不自动恢复进其他实验目录 |
| 测试 | A 回归、S 数学/梯度、null 禁用、调用链回归 | 不把未执行验收写成已通过 |

拟议第一版对 student reference 与 hist/future、pos/neg/split、sup/floor/ST/tanh/gamma≠1 的组合报错。已有 null 模式的历史实验不受此新守卫破坏。

### 11.3 概念循环，不是新 trainer

```text
student rollout -> 既有 batch / old logprob / IS 准备
for mini-batch:
    zero_grad
    for microbatch:
        current student forward (梯度开启)，取 top-k IDs / logps
        frozen teacher-real forward (no_grad)，在相同 IDs 上评分
        student reference: 跳过 null；null reference: 沿用 A
        复用 add_tail，构造 detached q
        复用 JSD -> IS -> mask -> agg_loss -> outer scaling
        backward（只累积梯度）
    既有 grad clipping / finite check / optimizer step
按既有配置保留 frozen teacher，进入后续 rollout
```

Teacher source、requires_grad、dtype、scoring temperature、更新频率等来自已确认 A manifest；不从变量名推断模型实际被如何更新。

## 12. 数据与数值契约

| 字段 | 典型形状 | 要求 |
|---|---|---|
| response IDs / mask | [B,T] | 使用真实采样记录与原 A mask |
| student top-k IDs | [B,T,K] | 当前训练 forward 选出；teacher 同列 gather |
| student / teacher top-k logps | [B,T,K] | 各自全词表归一化后的概率；相同温度约定 |
| student / teacher distill logps | [B,T,K+1] | add_tail 后的同一支持集 |
| sampled-token current / teacher logps | [B,T] | 精确 gather；不由 tail 近似 |
| old / rollout logps, IS weights | [B,T] | 只沿用已有 correction，不当作 S 词表参照 |
| detached reconstructed target | [B,T,K+1] | finite、归一化、无梯度 |

Student 与 teacher 的图像 token 数可能不同。按 response 相对位置和 causal shift 对齐，不按总输入绝对索引硬对齐。

保留 A 的稳定 tail 计算（logsumexp、expm1 及已有数值界限），不在 S 第一轮改变 tail floor。有效位置出现 NaN/Inf 应报告位置与原因，不悄悄退回 p^+ 或加入新 clipping。Padding 遵守已有安全占位与 mask；先产生 NaN 再乘零不是有效保护。

若增加 FP32 target 计算等数值变动，应独立记录并验证 A 路径不被改动，不能声称只有参照变化却暗改其他数值规则。

## 13. 验收计划：数学、梯度、调用链分别测试

以下是待实现时的验收要求，不是本文提交即代表测试已通过。

| 测试组 | 必须确认 |
|---|---|
| A 默认回归 | reference 缺省/null 时，既有 target、JSD、IS 后 loss 与梯度不变 |
| S 数学 | §4 基本 target/512 odds；β=0 返回 teacher；p^S=p^+ 返回 teacher |
| S 动态 | 固定 teacher 的四种 student odds 得到 512/32/2/.125；不将其当训练收敛测试 |
| 排序与尾部 | §5 排序反转、§6 tail=.684243524；聚合前后重构不等价 |
| 梯度路径 | q 无梯度、teacher 无梯度、student prediction 有梯度；含 mixture 的 JSD 路径完整 |
| 分布有限性 | 有效概率归一、边界有限、极小概率按原数值规则处理；不静默更换目标 |
| Null 禁用 | 测试替身令 null builder/forward 一被调用就抛错；S 必须仍能完成，不只检查计时键为零 |
| 支持集对齐 | Teacher 使用 student IDs；sampled token 在 top-k 外时统计仍使用精确 realized logp |
| IS/聚合回归 | 全 1 权重与已有逻辑一致；非 1 权重相乘、分母与 microbatch scaling 不变 |
| 优化器时序 | 多次 backward 前参数未 step；每个 mini-batch 完成后既有 optimizer 调用一次 |
| 恢复与输出 | 新实验名/目录、配置写入 checkpoint、resume 不串 A/X1/S |

可添加一个有限计算的小规模集成验收来确认实际调用图、梯度与日志可用；它是工程正确性检查，不是额外科学 probe，也不能替代完整任务评测。

## 14. 训练与评测：按现有口径执行 S 两次

[D1] 记录的固定实验条件为：2459 题高清 6karmA、seed=42、batch 48、n=2、51 步、lr=2e-6、3 卡/SP1。运行前从 A 的实际 manifest 核实 checkpoint、数据版本、crop 生成、分辨率、模板、采样温度与截断、warmup、优化器和全部配置；上述摘要不是完整 launch 配置。

拟议运行名为 `pair_S_6karmA` 与 `pair_Sr2_6karmA`。两次沿用同一 seed 与原重复规则，不新增 multi-seed sweep。两次结果均保留，不能第二次只报告最优 checkpoint 而隐藏退化。

评测继承：TB 使用既有 TreeVGR HF greedy，V* 使用既有 infer.py；准确率采用 gpt-oss-120b judge。每次记录 {30,40,50} 的两项分数及各自峰值。[D1]

**初期筛选门槛：** 两次 TB 峰值均≥50.5，且两次 V* 峰值均≥93.5。分别取峰不代表同一个 checkpoint 同时达标，应另列每个 checkpoint 的完整成绩。峰值达标也不是统计等效性证明。

| 对照/运行 | 当前用途 |
|---|---|
| A 已有同流程结果 | 目标性能与计算开销参照 |
| V0 已有同流程结果 | 判断 S 外推是否区别于普通 privileged distillation |
| S 两次 | 本轮新实验；只换参照，不叠时间机制 |

不重启 X2，不把 S 与 hidden-state/head-only 设计混成一个实验，不为拿到峰值延长步数。需要分析 TB 分类别变化，但小类别与关联 checkpoint 不视为独立重复证据。

## 15. 监测与解释：不让诊断重新引入 null

复用现有 raw/weighted JSD、target max probability、target entropy、target TV、actor grad norm、长度与重复行为记录；S 新增指标应注明只是建议，不声称源码已经具备。

优先记录：

- reference 模式、teacher-real/null 调用次数、teacher/student requires_grad；S null 调用必须为零；
- TV(q^S,p^+) 与 JSD(q^S,p^+)：target 离 teacher 多远；它们不是“正确率”或严格“尖锐程度”；
- target entropy、max probability、tail mass、argmax 是否为 tail、student top-k 外的 teacher 质量；
- teacher/student 距离、u^S 的分位数，以及逐步变化；不把全局下降自动解释为视觉纠正完成；
- raw/IS-weighted JSD、IS 分布、有效 token 数与聚合缩放；
- EOS/长度上限比例、重复输出、训练/推理时间、峰值显存、吞吐。

禁止在 S 中额外计算 p^0 以便画 u^S 与 u^A 的相关性。若比较已有 A 日志，只能用已存在的独立基线资料，并说明 rollout/prefix 不同的限制。

文档的耗时估算为 student 28s、teacher-real 26s、null 26s、backward 46s：26/126≈20.6% 是该四项之和中 null 的份额。[D1] 它不等于端到端已测加速；rollout、old scoring、通信、offload、保存和评测都需要单独计时。S 不再显式执行 null，也可能仍有其他 teacher/reference 路径，必须实际审计。

达到性能目标只能支持“这个 null-free 代理在该设置下有用”，不能据此认定提取到了纯视觉因果作用。失败则首先区分实现错误、极端 target/尾部、退化与信号不足，不直接宣判所有 null-free 方法无效。

## 16. 交付顺序与本次文档的边界

1. 锁定 A 的代码与 resolved config；记录本设计基线与实际运行基线的差异。
2. 独立实现 reference 分支、null 禁用和 runtime guards；A 默认回归先通过。
3. 完成数学/梯度/调用链测试及工程验收，保存机器可读运行 manifest 到仓库忽略的输出目录。
4. 执行 S 两次，按固定口径评测并核对实际时间/显存。
5. 写结果文档，再决定是否研究时间机制；不提前将任何达标或 novelty 写入摘要。

本次只新增这份 Markdown，不提交模型、数据、checkpoint、日志、缓存或凭据，不更新已有训练代码、不覆盖旧结果、不合并 main/sup。本文的代码片段是设计说明，不是已集成的 S trainer。

**最后的研究问题：** S 能否把 teacher–student 差距转化为足够有效的 privileged supervision，在完全不执行 visual-null 的条件下保留接近 Aha 的表现？目前目标构造已定义清楚，性能、训练稳定性和独立新颖性仍待验证。

## 17. 固定版本来源

正文中的仓库事实按下列不可变版本记录。相对链接方便阅读，固定链接用于复核当时状态。外部论文的优先权判断不由这些仓库记录替代。

- [D1] [候选 S、现有结果与约束：summary §7.7，30e1c3b](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/30e1c3b7dbf80e80c5206c05b93ecf548a73ba4c/docs/negative_history/summary_history_future_on_full_aha_2026-10-03.md)。[同目录最新文档](summary_history_future_on_full_aha_2026-10-03.md)。
- [D2] [历史实现计划：realized-logprob 与聚合约定，30e1c3b](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/30e1c3b7dbf80e80c5206c05b93ecf548a73ba4c/docs/negative_history/implementation_plan.md)。
- [C1] [dp_actor.py：update_policy、_forward_micro_batch、_optimizer_step，bea75c6](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/bea75c6bb399feeb731713f9592f86989618ac48/verl/workers/actor/dp_actor.py)。
- [C2] [core_algos.py：compute_self_distillation_loss、add_tail、agg_loss，bea75c6](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/bea75c6bb399feeb731713f9592f86989618ac48/verl/trainer/ppo/core_algos.py)。
- [C3] [run_visual_counterfactual_unit.sh：teacher、JSD、top-k 与 IS，bea75c6](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/bea75c6bb399feeb731713f9592f86989618ac48/scripts/run_visual_counterfactual_unit.sh)。
- [C4] [actor.yaml：已有配置，bea75c6](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/bea75c6bb399feeb731713f9592f86989618ac48/verl/trainer/config/actor/actor.yaml)。
- [C5] [ray_trainer.py：fit、teacher/null 构造、old logprob 与 correction，bea75c6](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/bea75c6bb399feeb731713f9592f86989618ac48/verl/trainer/ppo/ray_trainer.py)。
- [C6] [rollout_corr_helper.py：已有 correction，bea75c6](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/bea75c6bb399feeb731713f9592f86989618ac48/verl/trainer/ppo/rollout_corr_helper.py)。
