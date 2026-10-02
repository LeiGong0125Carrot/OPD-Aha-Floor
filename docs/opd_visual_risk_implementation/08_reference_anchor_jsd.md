# 08｜可选 teacher JSD anchor

**目标：** 给偏离冻结 teacher 的常规分布设置代价；不保证正确性或防重复。
**输入：** 同一 response位置的 teacher real分布、当前student分布、mask、lambda。
**输出：** anchor loss；teacher一侧stop-gradient。

## 1. 角色与总损失

$$L_t=L_t^{\rm clip}+\lambda\operatorname{JSD}(p_t^+,p_t^S).$$

p+ 是已计算的 privileged real teacher，不增加第三个teacher view；pS来自更新中的student。它不同于 PPO 的 old student 参照。

这里没有 Aha 重构 q。JSD只是辅助分布匹配。Lambda=0完全关闭；lambda>0是真实新增权衡，不能称为免费稳定化。

## 2. 完整 JSD 定义

$$m_t(v)=\tfrac12(p_t^+(v)+p_t^S(v)),$$
$$\operatorname{JSD}=\tfrac12\operatorname{KL}(p_t^+\|m_t)+\tfrac12\operatorname{KL}(p_t^S\|m_t).$$

m是计算中间值，不是另一个teacher或训练candidate。Teacher target detach；m仍随student变化，**不要把整个m detach后声称实现相同JSD**。

## 3. 数值例子

Teacher `[.60,.30,.10]`，student `[.30,.60,.10]`，m=`[.45,.45,.10]`。

两个KL均为：

$$.6\log(.6/.45)+.3\log(.3/.45)=.050969711.$$

所以JSD=.050969711。若两者相同则JSD=0。

另一个贯穿训练更新的例子：teacher=`[.70,.20,.10]`、student=`[.63,.27,.10]`，JSD=.003537583631。

若教学参数lambda=.10，policy loss=.66317，则总loss=.663523758363。训练数值必须用未舍入的advantage，不直接复制聊天中的三位小数。

## 4. 为什么 anchor 可能阻碍纠错

Teacher在错误prefix下也可能偏向diamond。即使student把概率从diamond移向正确的striped，JSD仍然对差异收费。它不检查哪个更正确。

如果teacher和student都支持重复 “this is red”，anchor可能接近零。不能把p+称为纯语言先验，也不能说其保留的是无噪声推理能力。

## 5. 完整词表与稀疏支持

[P1, Appendix C.2, p.19] 报告top-100 + tail。生产接入可复用既有近似，但要保证两个分布在同一个partition上比较。

**本计划新增可复现选项 [E]：** 用rollout snapshot的student top-100作为固定集合K，本批actor更新期间不变；teacher/student在K的各token上保留真实full-softmax概率，tail=$1-\sum_{v\in K}p(v)$。

这不是宣称与原仓库动态top-k完全相同。若继承的backend使用其他支持策略，须在manifest中记录并保证对照一致。实际token的c_t始终独立从完整log normalizer gather，不使用tail代替。

若teacher全分布不能缓存、只能缓存K+tail，actor更新时不能任意换K后继续使用旧teacher数值。可选择固定K，或显式承担相应teacher重评分成本；不许无记录增加前向。

不要对K内单独renormalize忽略tail。累计浮点误差导致tail轻微负值时，使用已有有测试的数值修正并记录次数；大幅负值视为错误，不自动掩盖。

## 6. 数值与梯度实施

完整logits参考实现使用：

```text
log_s = log_softmax(student_logits)
log_t = log_softmax(teacher_logits.detach())
log_m = logaddexp(log_s, log_t) - log(2)
jsd = .5 * sum(exp(log_s)*(log_s-log_m))
    + .5 * sum(exp(log_t)*(log_t-log_m))
```

fp32计算，按response mask和与policy相同的命名reduction汇总。Actor提供raw/full分布的温度与teacher必须由配置明确；不要与PPO采样策略logp混用。

## 7. 验收与成本

对称性；相同分布为零；三token数值一致；teacher无梯度、student有梯度；tail质量守恒；固定支持不随microbatch漂移。数学参考仅实现完整finite-logits版本，生产稀疏适配器尚未实现。

若原流程只保存sampled logp，加anchor需要额外分布存储或streaming统计。复用teacher forward不代表显存/通信零成本。

**参考实现：** `jsd_from_logits`。**下一步：** [09](09_end_to_end_training_loop.md)。
