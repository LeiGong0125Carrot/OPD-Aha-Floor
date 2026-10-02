# 07｜PPO-Clip：把固定 advantage 转成 student 更新

**目标：** 在同一批 rollout 的更新中限制继续推动极端概率变化的激励。
**输入：** new/old policy logp、固定 advantage、mask、clip epsilon。
**输出：** scalar policy loss、ratio 和 directional clip 日志。

## 1. PPO 概率比不是 real/null 比

$$r_t=\exp(\log\pi_\theta(y_t\mid h_t)-\log\pi_{\rm old}(y_t\mid h_t)).$$

$\pi_{\rm old}$ 是采样这批 response 时的 student 行为策略；不是 null teacher，也不是冻结 initial teacher。Old logp 在整个本批更新期间固定。

New/old 必须使用相同 sampling-policy 定义。Temperature/top-p/top-k/processor 的契约见 [02](02_rollout_data_contract.md)。不要拿 real teacher $p_t^+$ 作为分母。

## 2. Loss 与梯度边界

$$L_t^{\rm clip}=-\min\left(r_t A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right).$$

A 和 old logp 均 detach。New logp 保留计算图。Teacher cost、history、return 不在反向传播路径中。

若每批只有一次整批 forward/backward，开始时 r≈1，clipping 对该次即时梯度可能尚未活跃；optimizer 一步的实际跳变不被公式硬限制。多 microbatch/多轮更新时同一固定 old policy 参照才逐渐体现 ratio 变化。不能宣称“一用了 PPO clip，单步就被硬限幅”。

## 3. 数值例子（教学 epsilon=.2）

负 advantage A=-.737，old probability=.70：

| new probability | ratio | clipped loss |
|---:|---:|---:|
| .70 | 1.00 | .7370 |
| .63 | .90 | .6633 |
| .56 | .80 | .5896 |
| .35 | .50 | .5896 |

正 advantage A=.737，old=.10：

| new probability | ratio | clipped loss |
|---:|---:|---:|
| .10 | 1.00 | -.7370 |
| .11 | 1.10 | -.8107 |
| .12 | 1.20 | -.8844 |
| .20 | 2.00 | -.8844 |

超过区间后 plateau 只针对该样本、朝 advantage 有利方向的继续优化激励。共享参数、其他token和anchor仍可让概率越过这些位置。下一批 old policy 会更新；clipping不能阻止跨批长期熵塌缩。

## 4. 实现步骤

1. 对固定 response、student full image 重新 forward。
2. Gather 当前行为策略的 new logp；校验与 old 的 tokenizer、mask、温度和截断规则一致。
3. fp32 计算 `log_ratio=new-old.detach()`，有效位置外先清零。
4. `ratio=exp(log_ratio)`；非有限值触发数值错误，不静默 clamp 到某个任意值。
5. 计算 unclipped/clipped 两项，取 min 再取负。
6. 按已锁定的 loss reduction 汇总。
7. 加可选 anchor，按既定 optimizer/gradient clipping 更新。

Loss 可为负，不代表实现错误。它的数值不是最终答案准确率，也不是可跨任意 reward scale 直接比较的性能指标。

## 5. Loss reduction 必须显式命名 [E]

参考工作配置保留对话所述的有效 response-token mean：

$$L=\frac{\sum_{j,t}M_{j,t}L_{j,t}}{\sum_{j,t}M_{j,t}}.$$

这是一个 token-level surrogate；变长 sampled batch 的随机长度分母会改变严格的 sequence-risk 梯度解释。

参考代码另支持 `trajectory_sum_mean`：先每条 response 求和，再平均 response。这更适合在 gamma=1、on-policy 未剪裁等条件下讨论 episodic risk 梯度，但数值尺度会变，不能在同一实验名下切换。

未启用 advantage whitening、variance normalization、额外 entropy bonus。若旧训练器自动执行这些步骤，需检查并记录，不能 unknowingly 改写风险尺度。

## 6. 必须记录

ratio均值/分位数；初始new-old最大偏差；按 A正/负分开的主动 clipping 比例；policy loss；A的正负比例与量级；梯度范数；optimizer skip/nonfinite 次数。

`active_clip_fraction` 指实际处于 plateau 方向的样本，不只是 ratio落区间外的总比例。

## 7. 验收

上表数值；A=0策略梯度为零；正/负梯度方向；已clip样本该项梯度为零；old/A无梯度；pad不影响loss；初始ratio≈1；在同一固定batch内old不随new更新。

**参考实现：** `ppo_clip_loss`。PPO只是更新接口，不是当前研究的创新归属。**下一步：** [08](08_reference_anchor_jsd.md)。
