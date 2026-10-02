# 04｜历史累计、风险函数与新增代价

**目标：** 让历史改变“新冲突的边际代价”，而不是把过去的错误再罚到当前 token。
**状态：** $\Phi$ 为对话中的候选设计 [C]，并非上传论文验证过的模块。
**输入：** `cost[B,T]`、mask。**输出：** `history[B,T]`、`increment[B,T]`。

## 1. 历史是 exclusive sum

$$H_t=\sum_{k<t}c_k,\qquad H_1=0.$$

例如代价 `[.20,.30,.30,.10,.30,.20]` 对应历史：

`[0,.20,.50,.80,.90,1.20]`。

不要使用 inclusive cumsum 直接作为 $H_t$；否则当前 token 在风险增量里会被重复算入。每条 rollout、每个 batch 重新计算，不跨 rollout 累计。

## 2. 两个命名清楚的风险版本

线性对照：

$$\Phi_{\rm linear}(H)=H,\qquad d_t=c_t.$$

历史适应候选：

$$\Phi(H)=2H-\log(1+H),$$
$$d_t=\Phi(H_t+c_t)-\Phi(H_t).$$

$\Phi$ 是 scalar 函数，不是模型、不输出词表分布，也不生成新 token。它只把累计冲突映射成风险。

## 3. 稳定计算与例子

不要对两个很接近的大 $\Phi$ 值直接作差；用等价形式：

$$\boxed{d_t=2c_t-\log\left(1+\frac{c_t}{1+H_t}\right).}$$

实现：`d = 2*c - log1p(c/(1+H))`。

同样 $c_t=.20$：

| 历史 | 新增风险 |
|---:|---:|
| $H=0$ | $.4-\log1.2=0.217678443$ |
| $H=2$ | $.4-\log(3.2/3)=0.335461479$ |

历史较重时，新冲突的成本更大。但若 $c_t=0$，则任何历史下 $d_t=0$，不会凭空产生局部处罚。

## 4. 代数性质与解释边界

$$\Phi'(H)=1+\frac{H}{1+H},\quad \Phi''(H)=\frac1{(1+H)^2}>0.$$

因此对 $H,c\ge0$：

$$c\le d\le2c.$$

有界的是“每步相对于当前局部代价的额外历史放大”，不是整个 return、梯度或训练过程。若 c 使用未截断的大 negative log-ratio，d 仍可很大。

另外常数 1 和上限 2 都是设计选择。没有额外可调参数，不代表没有隐含尺度。当 `probability_gap` 换成 `negative_log_ratio` 时，$H$ 的尺度改变，Phi 的有效强度也随之改变。

## 5. 为什么不是重复计算

设 $H_{t+1}=H_t+c_t$。无折扣累加满足：

$$\sum_{k=t}^Td_k=\Phi(H_{T+1})-\Phi(H_t).$$

中间风险水平在相邻项之间抵消。整条序列 $\sum_td_t=\Phi(H_{T+1})$，而不是把 $\Phi(H_t)$ 每步再相加。

不能将 $d_t$ 再乘 $1+H_t$ 或额外 $\beta_t(H_t)$，除非明确引入新的实验分支；这不是当前已整理方案。

## 6. 与 length-wise normalization 的关系

前面对话曾讨论 $H_t/(t-1)$、正负占比和方向分布。最终逐步讲解的主线使用 **未按长度归一化的负累计量**，不能在实现时默默替换为均值。

原因是本版表达“已积累的量影响边际风险”。改成均值会改变状态语义、telescoping 目标及风险函数尺度。可以在日志记录均值以理解长度影响，但默认不能将它额外乘回 loss。

同样，$H_t$ 单调增加，不因后面的正视觉信号减少。修复后的路径只有不再增加或增加较慢；本版不是可衰减、可撤销的错误债务状态。

## 7. 接口与检查

```text
history_and_risk(cost, response_mask, risk='linear'|'bounded_slope')
  -> history, increment
```

全程 no_grad。测试：第一位置 H=0；重复/变长 batch 各行独立；零 cost 对应零 d；有界增量；整条和/后缀和 telescoping；FP32 与 FP64 数值接近。

**参考实现：** `history_and_risk`。**下一步：** [05](05_future_credit_assignment.md)。
