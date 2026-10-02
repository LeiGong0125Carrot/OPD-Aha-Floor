# 12｜数值测试、集成验收与训练内失败监测

**目标：** 排除实现错误，记录已知退化风险；不把工程测试当作性能预测probe。
**配套：** [tests/test_math_core.py](tests/test_math_core.py)、[validation](validation)。

## 1. 已提供的CPU数学单元测试

运行：

```bash
cd opd_visual_risk_implementation
python -m pytest -q
```

覆盖：prob-gap/log信号、signed log-ratio序列恒等式、exclusive history、Phi telescoping与边界、gamma=0/1/.9、变长pad/EOS、LOO组隔离与n=1错误、PPO正负clip数值与梯度、teacher/old/A detach、完整词表JSD数值与梯度、GU方向和零gate、不同reduction确实不同。

实际测试数量和通过情况以 `validation/test_results.txt` 为准。没有GPU/VLM/分布式训练已运行的暗示。

## 2. 当前参考实现未覆盖、接入后必须完成

| 测试 | 目的 |
|---|---|
| Teacher real/null response索引手工核对 | 排除next-token shift与图像token偏移 |
| Sampling/replay初始ratio≈1 | 确认实际behavior policy而非混合temperature/logp |
| Actor/rollout权重版本一致 | 排除stale weight与异步错位 |
| 单卡与跨卡完整LOO组 | 排除microbatch局部baseline |
| Packed/unpacked response一致 | history/return不越过样本边界 |
| EOS/stop/cap的真实token记录 | 不伪造终止，不把cap当正常完成 |
| 稀疏top100+tail mass和梯度 | 不丢tail，不用tail替实际token概率 |
| Gradient accumulation全局分母 | 不因变长microbatch改变loss权重 |
| Resume重放 | teacher、sampler、RNG、optimizer版本正确 |

对这些测试的结果必须单独记录，不能把CPU数学测试的“通过”复制为集成验收结论。

## 3. 每次训练更新的最小日志

**信号：** sampled u的正/负比例；cost均值/分位数；real-null近零比例；H终值；d/c在c>0位置的范围；return和future contribution量级。

**优化：** A正负比例、baseline幅度、初始new-old logp差、ratio分位数、active clipping fraction（正/负拆开）、policy/anchor/total loss、梯度范数、非有限更新次数。

**生成：** 有效response长度分布、正常终止比例、cap命中率、明确标注的重复指标、绝对token entropy及末段entropy、EOS概率统计（注明词表/策略尺度）。

**任务：** 既定judge准确率和checkpoint。不得把proxy优化当作judge改善。

## 4. 重复指标只用于监测，不改变训练目标

用户已报告类似“this is red”反复直到cap的失败。可以继承现有token n-gram重复率和生成文本日志。若新设n-gram长度/重复阈值，要写进日志规格；它们不是自动加入reward的隐形参数。

不要通过增加repetition penalty、缩短max length或强制EOS来隐藏训练循环，再宣称优化有效。安全暂停和保存checkpoint可以执行，但方法配置不能无记录更改。

## 5. 对失败的判读边界

- entropy下降+重复/cap上升：与过度集中一致，不能单独锁定因果模块。
- cost/return下降+答案准确率下降：可能是proxy利用，而非成功纠错。
- 很短回答+零cost：可能提前逃避视觉判断；不能只按风险低夸奖。
- 长中性文本+均值cost下降：若启用了长度平均，可能是稀释代价。
- 所有u≈0：无视觉差异信号，任何history/future模块都不能凭空提供正确监督。
- teacher与student同样循环：anchor可能失效。

## 6. 安全与数值中断

非有限有效logp、cost、return、ratio、loss、gradient；组成员缺失；old/new策略定义不一致；teacher误入optimizer；response shift不一致：应阻止该更新并保存诊断。

其他训练行为阈值继承当前工程安全策略或在运行前显式设定。本计划不伪造一组已验证的“停止阈值”。单纯策略loss为负不是中断条件。

## 7. 完成标准

数学测试通过只是第一层。必须进一步确认数据契约、采样分母、分布式baseline和终止语义，才可进入正式训练。只有相同实验条件下的judge评测和退化监测，才能支持性能主张。
