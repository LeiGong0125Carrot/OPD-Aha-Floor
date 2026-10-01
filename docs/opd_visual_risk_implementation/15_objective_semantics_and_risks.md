# 15｜优化目标的严格含义、近似与不可省略的风险

**用途：** 防止“代码能运行”被误写成“精确优化某个已证明正确的目标”。
**说明：** 下列恒等式和梯度边界为对话公式的数学整理/推导，不是新的性能证据。

## 1. 一个可明确定义的完整轨迹目标

令student实际采样策略为 $\pi_\theta$，frozen teacher给定每条已生成trajectory的cost为 $c_t(y_{\le t})$。成本不显式依赖当前student参数，除了通过实际生成的序列影响其取值。

$$J(\theta)=\mathbb E_{y\sim\pi_\theta}\left[\Phi\left(\sum_{t=1}^Tc_t\right)\right].$$

目标是降低期望代理风险；不是保证正确答案。Teacher正误判断没有被加入，baseline和正advantage也不产生GT标签。

## 2. 为什么可以使用风险增量和suffix return

$$d_t=\Phi(H_t+c_t)-\Phi(H_t),\quad H_t=\sum_{k<t}c_k.$$

无折扣且完整定义的终止轨迹下：

$$\sum_td_t=\Phi\left(\sum_tc_t\right),\qquad R_t=\sum_{k\ge t}d_k.$$

在on-policy、共同支持、允许交换微分与期望等常规条件下，score-function梯度为：

$$\nabla J=\mathbb E\left[\sum_t\nabla_\theta\log\pi_\theta(y_t\mid h_t)R_t\right].$$

当前位置之前的风险是 $h_t$ 的函数，其与本步score的期望项为零，因此可移除。独立于当前动作的baseline同样在期望中抵消：

$$\nabla J=\mathbb E\left[\sum_t\nabla\log\pi_\theta(y_t\mid h_t)(R_t-b_t)\right].$$

定义A=b-R后，用 $-A\log\pi$ 的负梯度更新方向对应风险下降。单个sample的未来代价不是因果证明；这是分布层面的信用分配关系。

## 3. 参考工作实现与上述理想目标的差别

| 选择 | 必须承认的差别 |
|---|---|
| gamma<1 | R是折扣的局部surrogate，不再等于Phi首尾差 |
| 仅对R折扣、无外层gamma^(t-1) | 不能直接称为从起点折扣总风险的精确梯度 |
| PPO-Clip，多轮使用固定batch | 是受限的近似policy更新，不是始终on-policy的精确REINFORCE |
| token_mean | sampled batch的随机长度分母可能改变sequence-risk梯度权重 |
| trajectory_sum_mean | 更接近episode目标，但尺度不同，影响lr/anchor相对强度 |
| 长度上限设observed-prefix边界0 | 只评价截断前缀，未估计真正未观测未来 |
| 动态top-p/top-k | 实际行为策略支持/可微性必须审计，不能自动套用完整softmax推导 |
| top-k+tail JSD | 比较的是压缩partition上的分布，不是完整词表JSD |

这些不是禁止使用的近似，但必须写入实验方法与manifest。

## 4. Negative-only evidence与有正负advantage不矛盾

c始终非负且只来自视觉降低支持的部分。然而A=b-R可以为正：比其他实际trajectory少积累风险的路径获得相对强化。它不是positive-u amplification，也不要求全部负PG。

若所有路径均错误，较少风险的路径仍可能得到正A。因此A>0不是事实正确性标签。

## 5. 为什么不是简单重写Aha

固定state下，$\min_\pi KL(\pi\|p^+)+\lambda E_\pi[c]$ 的自由分布最优解仍是指数倾斜，不能仅换写法宣称全新方法。

当前候选主线的区别在于：on-policy完整轨迹的历史风险与future credit进入policy update；不是先求每个state的 $q_t$ 再匹配。PPO/LOO/JSD单独都不是创新点。能否形成足够的方法贡献需要实际训练比较和后续文献核对，不能由本实施包保证。

## 6. 长度与重复风险不能被隐藏

- 累计非负成本：可能鼓励早停，减少有用但存在代价的推理。
- 按长度平均：可能鼓励加入中性填充词稀释成本。
- Gamma折扣：可能更重视近处，弱化较远错误归因。
- Phi历史增长：能加强持续代价，但不能确认它就是错误承诺深度。
- 终止后的peer为零：LOO可能把长回答与已结束短回答作不理想比较。
- 视觉一致但无意义的重复：c可能一直为零，所有风险模块失明。
- 高熵/低熵不是直接的正确性，不能重引入OPSA entropy gate充当标签。

本版只记录这些行为，不无声新增停止奖励、重复惩罚或答案验证奖励。否则方法目标就变了。

## 7. 视图与信号的解释边界

Real/null差值保留question与student prefix，因此是条件视觉效应；不是独立于语言上下文的纯视觉真值。

Signed log-ratio之和具有sequence likelihood ratio含义；negative-only截断后是只累计反对的量；prob-gap之和是绝对概率下降的累计，不是nats、距离或正确性概率。

把累计量称为“走错了多远”只能作为直觉类比，论文方法中应写精确代理指标定义。

## 8. 为何先写清这些边界

用户目标是同时争取性能和方法创新。本计划提供可实现、可区分的候选骨架，但不能用术语变化替代实证贡献，也不能用合理motivation保证优化改善。

优先将协议、mask、behavior logp、risk、baseline、reduction固定，使训练成功或失败能对应明确的算法版本，而不是混杂实现差异。
