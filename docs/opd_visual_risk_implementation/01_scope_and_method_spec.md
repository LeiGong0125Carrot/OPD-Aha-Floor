# 01｜实施范围、方法定义与固定约束

**用途：** 在开发前锁定“做什么、不做什么、哪些只是候选设计”。
**依据：** 本轮对话 [C]；上传 OPD-Aha 文稿 [P1]。来源编号见 [16](16_sources_and_decision_register.md)。
**状态：** 候选算法的实施规格，不是已完成模型训练或性能结论。

## 1. 最终主线

保留 OPD-Aha 的 evidence estimator 来源：student 看全图，冻结 teacher 在相同 student prefix 上，分别看 privileged evidence 与尺寸匹配的 mean-RGB null。改变的是信号进入 student 更新的方式：

$$
(p_t^+,p_t^0) \rightarrow c_t \rightarrow H_t,d_t
\rightarrow R_t^\Phi \rightarrow A_t^\Phi
\rightarrow L_{\mathrm{PPO}}+\lambda L_{\mathrm{anchor}}.
$$

这里不构造 Aha 的指数重加权 target $q_t$。普通 teacher JSD 可以作为辅助 anchor，但不能把它描述成“完全不蒸馏 teacher”。PPO、JSD、LOO 本身不是方法创新；研究差别在视觉负证据、历史相关风险和时序信用分配的组合目标。

## 2. 主线与独立对照必须分开

| 分支 | 信号进入更新的方式 | 默认是否叠加其他分支 |
|---|---|---|
| 原 OPD-Aha | $q\propto p^+e^{\beta u}$ 后蒸馏 | 独立基线 |
| Temporal visual-risk PPO | realized-token 代价→风险→return→advantage→PPO | 主候选 |
| Visual GU | $G=[p^0(y)-p^+(y)]_+$，直接 $G/(2-p^S(y))$ | 独立局部对照，不默认加到 PPO |
| Suppression-only Aha | 负半边指数重构 | 已讨论的对照，不作为本主线的新骨架 |

主候选不再设 Aha 的动态 $\beta_t$。历史 adaptiveness 由 $d_t=\Phi(H_t+c_t)-\Phi(H_t)$ 表达；不要再额外乘同一个历史 gate，否则会重复增强并新增难以解释的自由度。

## 3. 本轮详细讲解所对应的工作版本

- 局部代价：`probability_gap`，$c_t=[p_t^0(y_t)-p_t^+(y_t)]_+$。
- 可替换信号：`negative_log_ratio`，$c_t=[-u_t(y_t)]_+$，需单独命名实验。
- 历史：exclusive cumulative sum，$H_t=\sum_{k<t}c_k$，不是历史均值。
- 风险：$\Phi(H)=2H-\log(1+H)$；`linear` 版本用 $\Phi(H)=H$。
- 当前及未来：$R_t=d_t+\gamma R_{t+1}$，包含当前步。
- baseline：同题独立 rollout 的 leave-one-out；变长序列规则是新增工程决策，见 06。
- advantage：$A_t=b_t-R_t$，正值表示相对较少风险，不等于事实正确。
- actor：PPO-Clip；可选 $\lambda\operatorname{JSD}(p_t^+,p_t^S)$。

这些是对话方案的实现对象。用户理解各模块，不代表已经锁定所有训练超参数或确认该方案有效。

## 4. 必须保留的约束

1. 只用既定两种 teacher 视图；不增加 teacher full-image 第三次视图，不改 null 定义。
2. 不提取或干预 attention；不使用 teacher takeover、句段剪接、prefix 重写、候选分支枚举/重排。
3. Student 按现有采样器生成完整原始 rollout。future 来自已经生成的后缀，不新增 lookahead 采样。
4. 不增加 verifier、critic、reward model、GT token 正确性标注。
5. 保持 seed=42；不安排多 seed sweep，不安排独立科学 probe。单元测试和工程 smoke test 只验证实现，不预测训练收益。
6. 复用当前已验证的 Aha 数据、初始化、采样预算、更新预算、评估规则和 checkpoint 规则。不能把 PDF 的配置无条件视为当前仓库配置。
7. 不把低 student probability 当作视觉错误代理；不复刻 OPSA bottom-20% sampled-token 惩罚。

## 5. 已知失败与未确认解释

用户报告：OPSA 迁移后会重复类似 “this is red” 的短句直到长度上限；正常 spatial reasoning 回答只有几百 token。短轨迹、熵结构和负更新造成过度自信，是讨论中的解释，不是已分离确认的唯一原因。

本候选方法没有自动防重复保证。若 real/null 均支持重复 token，视觉代价可能接近零；future、Phi、baseline、PPO clipping 和 anchor 都不能凭空创造缺失的错误标签。

## 6. 开发完成的定义

完成本计划指：数据契约与 sampler 一致、数学模块通过测试、分布式组装和 mask 正确、能在现有训练框架完成受控更新与日志记录。**不意味着达到 Aha 性能。**

代码接入路线见 [13](13_repository_integration_tasks.md)，风险与目标语义见 [15](15_objective_semantics_and_risks.md)。
