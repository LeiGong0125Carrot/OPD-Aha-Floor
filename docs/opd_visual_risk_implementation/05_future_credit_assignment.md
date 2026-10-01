# 05｜已观测 future 的信用分配与终止处理

**目标：** 将 rollout 中已经发生的后续视觉风险用于当前位置的反馈，不增加未来采样。
**输入：** `increment[B,T]`、response mask、finish_reason、gamma。**输出：** `returns[B,T]`。

## 1. Future 的范围

Future 指已经完成的 student response 中，当前 token 之后的真实后缀。不是 teacher 预知未来，不修改原始 prefix，不为当前 token 枚举其他分支。

Teacher 在每个 $k$ 仍只以 $y_{<k}$ 为条件评分。所有位置评分结束后，训练程序才沿数组逆序累计。

## 2. 当前及后续，而不只是未来

本版定义：

$$R_t^\Phi=d_t+\gamma d_{t+1}+\gamma^2d_{t+2}+\cdots,$$
$$\boxed{R_t^\Phi=d_t+\gamma R_{t+1}^\Phi.}$$

它包含当前 $d_t$。不要从 $t+1$ 开始却仍然使用本文件的例子和配置名。纯 future-only 是不同方法，本版不启用。

## 3. 数值例子

先采用线性风险，即 d=c。若后缀为 `[.10,.30,0,.20]`：

- gamma=0：returns=`[.10,.30,0,.20]`，退回局部反馈。
- gamma=1：returns=`[.60,.50,.20,.20]`。
- gamma=.9：当前位置 return=$.10+.9(.30)+.9^3(.20)=.5158$。

使用历史 Phi 后，只是把输入从 c 换成 d，后缀求和代码不变。

Gamma=.9、1 是说明数字，不是已锁定或已证明合适的训练值。折扣新增一个控制参数，不再同时增加 lookahead window、额外 rollout 数或 critic。

## 4. History 与 future 的分工

$H_t=\sum_{k<t}c_k$ 描述到达当前状态前的累计量；$R_t^\Phi$ 描述当前及之后实际发生的风险增量。

固定一个 t，两段位置没有重叠，但预测并非统计独立。Risk d 已由历史条件化，因此 R 含有历史影响；不能因此声称当前 token 已被证明造成所有后续错误。

后续同一 d 会进入多个较早位置的 return，这是 policy-gradient 信用分配的结构，不是把 episodic risk 定义成所有 return 再相加。**总风险是 sum(d)，不是 sum(R)。**

## 5. 终止、截断与 mask：新增工程决策 [E]

- 正常 EOS/stop：最后一个实际输出动作参与评分，之后为吸收状态，后续代价为零。
- 长度上限：无 critic、无追加 rollout 的约束下，只能计算已观察部分的 return。本版把未观测 continuation 留空并设边界 accumulator=0，称为 **observed-prefix return**，不能称无偏完整未来成本。
- 日志单独标出 `finish_reason=length`，报告长度上限命中率。不得把这种截断当作模型主动正确终止。
- 基础设施中断不是环境终止：不能给这条不完整结果零未来风险后继续当正常样本训练。
- Pad 不产生代价，不参与 actor/anchor loss，不能改变本行之前的 return。

若继承框架有 time-limit bootstrap，不能默认复用，因为本版没有 value head。必须显式关停对应机制或另作设计，而不是用任意旧 critic 值填入。

## 6. 逆序实现

```text
acc = zeros(B)
for t in reversed(range(T)):
    acc = where(mask[:,t], increment[:,t] + gamma * acc, 0)
    returns[:,t] = acc
```

参考代码为 O(BT) 时间、O(BT) 存储输出；没有新增模型 forward。生产环境可以向量化，但必须与该参考逐项一致。

## 7. 折扣改变了什么目标

Gamma=1 时有明确 telescoping：$R_t^\Phi=\Phi(H_{T+1})-\Phi(H_t)$。

Gamma<1 时它是 discounted local surrogate；不能仍用首尾差公式。当前实现也没有额外的外层 $\gamma^{t-1}$ 权重，因此不要无条件称其为从起点折扣总风险的精确梯度。详见 [15](15_objective_semantics_and_risks.md)。

## 8. 验收与风险

测试 gamma两端、.5158例子、末端 d、pad/EOS、Phi后缀和。日志记录当前 d 与 future contribution=`R-d` 的尺度；它是训练内日志，不是独立 probe。

Future 不会修复缺失的代价：若循环 token 的所有 c=0，则所有 R=0。也不能把低 R 自动称为正确或恢复。

**参考实现：** `discounted_cost_to_go`。**下一步：** [06](06_group_baseline_advantage.md)。
