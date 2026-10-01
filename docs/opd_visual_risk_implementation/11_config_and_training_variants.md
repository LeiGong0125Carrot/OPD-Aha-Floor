# 11｜配置模板、最小训练对照与待锁定参数

**目标：** 让实施者能区分已讲清楚的机制、需要沿用的配置和尚未确定的数值。
**配套：** [configs/train_plan.yaml](configs/train_plan.yaml)。`null`表示必须从当前实验配置填入或人工锁定，不是运行默认值。

## 1. 不把教学数字当成最优参数

对话出现gamma=.9/1、epsilon=.20、lambda=.10，是数值教学；当前没有模型训练证明它们最优。

[P1, Appendix C.2, p.19]报告6,241样本、teacher为初始student冻结副本、JSD、top-100+tail、lr=2e-6、global batch96、n8、最大prompt/response8192/1024、seed42。这是PDF版本的报告，不是已经核实的当前项目launcher。

当前实际项目的数据子集、crop/hide版本、每题rollout数和预算必须从现有有效Aha基线manifest继承。不要把历史高清2459样本运行和文稿6241样本配置混合。

## 2. 参数分类

| 项目 | 处理 |
|---|---|
| seed | 保持42，不做多seed sweep |
| model/teacher checkpoint | 从当前baseline显式导出；teacher不能随训练刷新 |
| data、view、processor、prompt | 原样继承并hash |
| sampler temperature/top-p/top-k | 原样继承，完成行为策略logp适配；不静默改成1 |
| optimizer、lr、batch、n、更新预算 | 原样继承，不能只用聊天数字猜 |
| cost_kind | 主工作版本probability_gap；log版本独立实验 |
| risk_transform | linear或bounded_slope，配置名明确 |
| gamma | 一个future折扣参数，需锁定；0为local、1为无折扣 |
| clip_epsilon | PPO更新参数，复用已有实现值或明确新选择 |
| anchor_weight | 可选；0关；启用时需锁定 |
| advantage_normalization | 默认none，不偷偷白化 |
| loss_reduction | 工作版本token_mean；精确目标讨论另见15 |
| EOS/truncation | 按02、05显式继承/记录 |

## 3. 最小、可解释的直接训练对照

这不是要求全参数组合搜索。先沿固定种子和相同预算比较下面的少量分支：

| ID | 代价/风险/credit | 回答什么问题 |
|---|---|---|
| AHA | 当前最强可复现Aha基线，不改公式 | 比较锚点 |
| VR-LOCAL | prob-gap，linear，gamma=0，PPO，固定anchor设置 | 局部直接策略优化是否可用 |
| VR-TEMP | 同LOCAL，仅gamma改为锁定future值 | 时序credit的增量 |
| VR-HIST | 同TEMP，仅风险改为bounded_slope | 显式历史边际风险的增量 |
| GU-LOCAL（可选） | prob-gap GU，匹配anchor设置 | 不同局部负向优化几何 |

非Aha分支在前几项比较中保持lambda一致，避免把anchor开关与history效果混在一起。仅在有需要时再比较anchor=0；不要宣称从一个混合改动实验已识别每个因果因素。

若已有可信的VR-LOCAL结果可复用，则不强制重复训练。任何复用都必须检查相同seed/数据/预算/代码版本，而非只比历史峰值。

## 4. 不新增独立probe

本计划中的synthetic unit tests验证公式、mask和梯度；model integration smoke test验证数据流，不是用预训练dump预测训练收益的独立科学probe。

直接训练的最终结果依据既定judge-based评测；不要用训练proxy、候选概率或者teacher margin代替任务准确率。

## 5. 比较与报告

保持当前TreeBench/V*等已经选定的正式评测、judge版本、解码、checkpoint频率和选择规则。不得新增一个评测框架后把结果称为同设置提升。

至少同时报告任务结果、生成长度、cap命中率、重复率、绝对entropy、EOS/stop行为。Proxy下降而准确率下降、或输出变成无信息短句，不能称为成功。

## 6. 启动前必须填完整

真实repository root、baseline manifest、data revision、teacher checkpoint、sampler adapter、n>=2、gamma、epsilon、lambda（若开anchor）、PPO epochs、optimizer step预算、gradient accumulation、loss reduction、termination规则。

模板包含验证脚本；其缺值报错是有意的，以避免未决定值被当作训练默认。配套 `demo_math.yaml` 只描述CPU例子，不可拿来启动VLM训练。
