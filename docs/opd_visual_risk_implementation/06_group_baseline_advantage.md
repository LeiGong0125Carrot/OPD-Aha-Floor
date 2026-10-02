# 06｜同题 leave-one-out baseline 与 advantage

**目标：** 把“非负的风险代价”变成相对参照的正/负更新权重。
**输入：** returns[B,T]、mask、prompt_group_id。**输出：** baseline、advantage，同形状、均 detach。

## 1. 定义与符号方向

同题固定组内有 n 条独立 rollout。第 j 条位置 t 的 baseline：

$$b_{i,j,t}=\frac{1}{n-1}\sum_{\ell\ne j}\widetilde R_{i,\ell,t}.$$

$$\boxed{A_{i,j,t}=b_{i,j,t}-R_{i,j,t}.}$$

这里 R 是代价，不是越大越好的 reward。A>0 只表示这次代价低于参照；A<0 表示较高。不得写成 `R-baseline` 然后仍使用相同 PPO 符号。

## 2. 两条 rollout 的完整例子

已有两条同题结果：R1=.60、R2=.12。排除自己：b1=.12、b2=.60，所以 A1=-.48、A2=+.48。

历史风险版使用 R1=.9123179275、R2=.1754614789，得到 A1=-.7368564487、A2=+.7368564487。

参照也必须用同一种 Phi/gamma/cost 尺度。不能用原始 R 的均值减去风险变换后的 R。

## 3. 不是候选枚举，也不只训练胜者

这些是现有采样预算中实际生成的多条 response。所有有效样本参与更新；不按风险选 top-k、不重排输出、不额外从某个位置分叉。

同题不同 response 的 prefix 一般不同。LOO 是统计参照，不是固定相同 prefix 的精确 state value，也不是 critic。

## 4. 变长序列必须规定如何比较 [新增工程决策 E]

主参考实现：将每条 response 按自然终止后的吸收状态补零到组内公共 T：

$$\widetilde R_{j,t}=0\quad\text{当第j条已结束且不存在位置t。}$$

Baseline 分母始终为 n-1；已经结束的其他 rollout 仍以零 return 计入，而不是仅平均“仍然活着的 peers”。Actor 自己无效的位置仍完全 mask 掉。

例：一条在第2个位置仍有 R=.50，另一条第1个位置已终止，则前者位置2 baseline=0，advantage=-.50。短回答会影响这个参照，因此不能说这种补零规则长度中立；它需要在训练中结合提前结束、截断和最终准确率监控。

本规则是为了使实现可复现的工程选择，不是用户此前逐项确定过的事实。若改为 prompt-level total-risk baseline 或其他变长处理，必须新命名配置并更新测试，不能混用。

## 5. 独立性与 baseline 的限定

固定组大小、给定同一问题独立采样的其他 rollout，不依赖当前采样动作，这给出 state/action-independent baseline 的标准条件。组内 beam 搜索、按结果筛选样本、把当前 return 自己也放进均值、动态只保留部分样本，都会改变这个论证。

即使条件成立，也不证明该参照在不同 prefix 下有低方差，更不证明它能识别事实正确性。参见 [15](15_objective_semantics_and_risks.md)。

## 6. 工程实施顺序

1. Teacher评分及 returns 全部完成。
2. 跨卡按 `prompt_group_id` 组装完整固定组，核对 n 条是否都存在。
3. 在统一 response 相对 t 上补零；只对真正终止/已观测尾部按既定规则补，不把失败请求当零。
4. 求 group sum，减去自身 return，再除以 n-1。
5. Advantage=`baseline-return`，mask 后 detach。
6. **之后**才切 actor microbatch。不能在每个 microbatch 中用不完整组计算 baseline。

若 n<2，主参考实现报错，不能偷偷用自己的均值或新增 rollout。基础设施导致组不完整则重试完整组或按既定一致规则跳过完整组。

## 7. 不默认加入的操作

不将 A 截成负数；不强制让所有视觉反对 token 都只接收负 advantage；不自动 batch z-score/白化；不将 denominator 换成全 batch 大小。

Negative-only evidence 可以生成正 advantage：因为较少的负风险值得相对强化。所有候选都错时“较少错的那条”仍会被相对强化，这是目标限制，不是数学错误。

## 8. 验收

n=2结果反号；相同风险全部A=0；不同group绝不互相影响；n=1报错；跨卡/单卡结果一致；row permutation不改变回映结果；terminated peer不从分母消失；baseline无梯度。

**参考实现：** `leave_one_out`。**下一步：** [07](07_ppo_clip_update.md)。
