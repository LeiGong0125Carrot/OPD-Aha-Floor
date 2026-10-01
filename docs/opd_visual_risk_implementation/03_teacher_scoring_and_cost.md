# 03｜Teacher 评分与局部视觉代价

**目标：** 从两次 teacher 评分提取可计算、可追踪的 realized-token 代价。
**输入：** response_ids、response_mask、privileged/null 输入。**输出：** `u[B,T]`、`cost[B,T]`。

## 1. 两个 teacher 分布是什么

$$
p_t^+(v)=p_\phi(v\mid I^+,x,y_{<t}),\quad
p_t^0(v)=p_\phi(v\mid I^0,x,y_{<t}).
$$

Teacher 同权重、同 tokenizer、同 scoring 温度、同文本 prefix。只改变既定 visual evidence。Question 始终保留，不移除问题、不增加解释句、不加入答案。

解释应是“在 question/prefix 固定下，改变视觉输入造成的条件预测差异”，而非去掉所有语言因素的纯视觉真值。[P1, §3.1, p.4–5]

## 2. 原始信号与本轮工作版本

原 Aha 信号：

$$u_t(v)=\log p_t^+(v)-\log p_t^0(v).$$

本轮逐步讲解采用：

$$c_t=[p_t^0(y_t)-p_t^+(y_t)]_+.$$

这里取的是 **student 实际生成的 $y_t$**，不是找词表里最负的 token，也不是 sampled-token bottom-20% 排名。

| 示例 | $p^0(y)$ | $p^+(y)$ | $u(y)$ | 概率差代价 $c$ |
|---|---:|---:|---:|---:|
| 视觉降低支持 | .80 | .60 | $\log .75=-.287682$ | .20 |
| 视觉提高支持 | .10 | .30 | $\log3=1.098612$ | 0 |
| 无变化 | .10 | .10 | 0 | 0 |

`[x]_+=max(x,0)` 的输入在主工作版本中是 **普通概率差**，在 log 版本中才是负 log-ratio。不能混为一个表达。

## 3. 可选 log 版本不得悄悄替换主版本

$$c_t^{\log}=[-u_t(y_t)]_+.$$

同样的 .80→.60：概率差=.20，log代价=.287682。前者累积的是绝对概率下降量；后者累积的是负 log-ratio 幅度。指数 $e^u=p^+/p^0$ 保留次序且一一对应，不是“只剩排名”。

只有不截正负的 signed sum 满足完整 sequence identity：

$$\sum_{k<t}u_k(y_k)=\log\frac{P_\phi(y_{<t}\mid I^+,x)}{P_\phi(y_{<t}\mid I^0,x)}.$$

Negative-only sum 已不是完整 sequence log-likelihood ratio；概率差之和更没有这个概率链式含义。不能把所有 history 都统一称为 nats。

## 4. 计算步骤

1. `teacher.eval()` 并冻结参数，在 `no_grad` 下 teacher-force 既定 response。
2. 在正确 next-token 行计算完整 softmax 的 log normalizer。
3. Gather `response_ids`，得到 `logp_plus_y`、`logp_null_y`。
4. 转 fp32；测试用 fp64。Pad 先掩掉，不能依赖 `NaN * 0`。
5. 计算 `u = logp_plus_y - logp_null_y`。
6. 概率差可用稳定式：
   `cost = exp(logp_null_y) * (-expm1(min(u, 0)))`。
7. 所有输出 detach，pad=0，验证 cost 非负且有限。

稳定式等于 $p^0(1-e^u)$ 的负半边。对于极小概率，两次 `exp` 后直接相减可能损失精度；fp32仍可能低估极端小质量，这是概率差指标自身的敏感性限制。

## 5. Top-100 + tail 不等于 sampled-token 概率

[P1, Appendix C.2, p.19] 提到 top-100+tail 用于分布支持。这个压缩可供 anchor/蒸馏使用，但不能用 tail bucket 的总概率冒充一个实际 token 的概率。

即使 $y_t$ 不在 top-100，也必须单独从完整 logits 与 log normalizer gather 它的真实 logp。若现有接口只返回截断候选内的重归一化 logp，需要扩展接口后才可使用。

## 6. 不额外添加的设计

本版不加正负阈值、rank selection、entropy weighting、question-free view、attention gate。近零小噪声默认仍按公式处理，其累计影响在日志中观察；不能无记录地加 eps 阈值去修改符号。

不把 `u<0` 写成“事实错误”。例如真实属性被反复生成时，real teacher 可能仍然支持它，代价为零。

## 7. 单元测试与验收

- 三个表中样例数值一致。
- real=null 时全部 cost=0。
- 同比例缩小概率时 log代价不变、概率差按尺度缩小。
- Teacher 的参数及 gathered logp 不接收 actor 梯度。
- padding 中的任意数值不改变有效位置。
- 实际 token gather 与手工 log-softmax 一致。

**参考实现：** `reference/math_core.py::visual_cost`。**下一步：** [04](04_history_risk_transform.md)。
