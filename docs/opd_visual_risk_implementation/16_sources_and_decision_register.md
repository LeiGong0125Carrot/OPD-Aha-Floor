# 16｜来源、设计状态与决策登记

## 1. 来源层级

### [P1] 用户上传文稿

**OPD-Aha: From Linguistic Momentum to Visual Reflection in Multimodal On-Policy Distillation**，本对话附件 `revision_opd_arxiv(6).pdf`，24页。

| 本计划引用的内容 | 原文位置 |
|---|---|
| student全图、teacher特权视图、共同student prefix | §2.1，p.2–3 |
| real/null同teacher、mean RGB、u=log p+−log p0 | §3.1，p.4–5，Eq.2–4 |
| KL-regularized reconstruction与指数解、pairwise odds | §3.2，p.5，Eq.5–8 |
| q监督student、有效response位置平均 | §3.2，p.5，Eq.9–10 |
| 非零u不必然corrective | §3.1延续，p.5 |
| KL radius等价与beta单调性 | Appendix A，p.16 |
| teacher为初始student冻结副本、JSD、top100+tail及配置 | Appendix C.1–C.2，p.19 |

原文没有验证本包提出的Phi历史风险、visual-risk future return、位置级LOO、PPO主目标或GU视觉迁移。不把这些内容写成OPD-Aha原方法。

### [C] 本轮逐步讨论

用户明确提出保留干净的privileged/null条件、利用历史、negative-only、减少超参数、兼顾性能/创新，不做独立probe；并要求每理解一节再进入下一节。

逐步解释的候选模块：probability-gap局部代价；historical Phi增量；discounted observed future；LOO相对advantage；PPO clipping；可选teacher JSD；固定反馈后只更新student。GU是独立替代路线。

这些是方案讨论，不是用户已经报告过的训练效果。此前援引的外部方法名仅说明讨论来源；本交付没有重新核验那些网页/版本，也没有引用其成绩作为当前方法有效性的证据。

### [E] 本实施计划明确补充的工程决策

Tensor schema、response shift、实际sampler概率契约、EOS/截断、变长LOO、distributed aggregation、稀疏支持冻结、精确数值测试与配置阻断。它们用于把讨论落实为可运行接口，不冒充之前已确认的实验设置。

## 2. 决策登记表

| ID | 项目 | 当前状态 | 实施要求 |
|---|---|---|---|
| D01 | 两teacher视图、student原rollout | 用户约束 | 不增加视图或重写trajectory |
| D02 | probability-gap vs negative-log-ratio | 本轮教学工作版本为prob-gap | log另命名，不静默替换 |
| D03 | Phi=2H-log(1+H) | 候选函数，无性能验证 | linear对照保留；尺度/上限不是无假设 |
| D04 | gamma | 数值未锁定 | .9/1仅演示；正式配置必填 |
| D05 | epsilon | 数值未锁定 | .2仅演示或继承经确认旧配置 |
| D06 | anchor开关与lambda | 可选，数值未锁定 | .1仅演示；关闭则weight=0 |
| D07 | 变长LOO | 新增工程选择 | time-aligned、terminated peer补0、固定n-1 |
| D08 | loss reduction | 工作实现为token_mean | 标注surrogate；另支持trajectory_sum_mean |
| D09 | actual behavior logp | 需要仓库核验 | 采样与重评分一致，不能忽略温度/截断 |
| D10 | sparse JSD支持 | 建议fixed snapshot top100+tail | 非原仓库已确认事实；对照保持一致 |
| D11 | cap截断 | observed-prefix、无bootstrap | 不等于真实未来为零；单独报告cap率 |
| D12 | 已采样EOS | 纳入有效动作 | 不伪造、不默认剔除 |
| D13 | teacher freeze与old记录 | 主线结构 | teacher初始固定，old在每批期间固定 |
| D14 | GU与PPO关系 | 独立对照 | 默认互斥，不相加 |
| D15 | seed/数据/预算/评估 | 保留seed42，其他继承当前有效baseline | 不合并历史不同配置峰值 |
| D16 | 仓库API/路径 | 本次未读取真实repo | 13中的目录与接口均是建议 |
| D17 | 测试与训练结论 | 只有CPU数学测试 | 不声称完成GPU训练或性能验证 |

## 3. 前面对话中需要精确化的地方

1. “历史大就增加惩罚”不等于将过去风险直接加给当前动作。当前使用的是新增风险d。
2. 全后缀return不是反事实分叉搜索，也不证明每个后续成本都由当前token单独造成。
3. 负向证据不意味着所有advantage都负；相对低风险可以得到正更新。
4. PPO clipping不是硬概率界限，不阻止跨batch累计塌缩。
5. JSD anchor不是正确性检查，可能阻碍真实纠正。
6. 参数.9/.2/.1未因用户理解示例而变成最优训练参数。
7. 使用length normalization、截断、归一化advantage会改变目标/尺度，不能当成无关紧要的工程细节。

## 4. 本包的验证范围

数学公式通过合成数值、梯度与mask测试核对。PDF的相关定义已与上传文件对应核读。模型/数据/仓库/分布式采样器未被接入，外部论文的新颖性归属未重新系统检索。

本文件旨在把来源和不确定性留在交付件中，避免实施者把讨论中的候选设想误写为已经证实的研究结论。
