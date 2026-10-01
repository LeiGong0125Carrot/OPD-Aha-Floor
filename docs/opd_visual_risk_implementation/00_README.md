# OPD 视觉冲突时序风险训练｜Implementation Plan

**版本：** v0.1，候选实施规格。  
**交付：** 每个section单独一个Markdown文件；另附CPU数学参考、测试、配置模板和精确例子。  
**范围：** 整理本轮已逐步讲解的算法，不宣称已完成VLM训练或优于OPD-Aha。

## 1. 主线一览

$$
\begin{aligned}
c_t&=[p_t^0(y_t)-p_t^+(y_t)]_+,\\
H_t&=\sum_{k<t}c_k,\\
d_t&=\Phi(H_t+c_t)-\Phi(H_t),\quad\Phi(H)=2H-\log(1+H),\\
R_t^\Phi&=d_t+\gamma R_{t+1}^\Phi,\\
A_t^\Phi&=b_t^\Phi-R_t^\Phi,\\
L&=L_{\rm PPO\text{-}Clip}(A^\Phi)+\lambda\operatorname{JSD}(p^+,p^S).
\end{aligned}
$$

**保留的部分：** privileged/null两视图、同一student prefix、冻结teacher、student全图原始on-policy rollout。  
**改变的部分：** 不先构造Aha的q；以历史相关的后续视觉风险形成policy advantage。  
**独立分支：** GU只作为局部负向学习对照，不默认叠加主线。

本轮数值讲解使用probability-gap，negative-log-ratio是单独可切换信号。Phi是候选设计；gamma/epsilon/lambda均没有被教学例子锁定为最优值。

## 2. 按section阅读或交给实施者

| 文件 | 内容 |
|---|---|
| [01｜范围与方法规格](01_scope_and_method_spec.md) | 约束、主线、GU互斥、已知失败与未验证部分 |
| [02｜Rollout数据契约](02_rollout_data_contract.md) | tensor字段、sampled logp、token shift、EOS、采样器策略 |
| [03｜Teacher信号与代价](03_teacher_scoring_and_cost.md) | real/null评分、prob-gap/log两种尺度、实际token gather |
| [04｜历史风险变换](04_history_risk_transform.md) | exclusive H、Phi、增量d、稳定计算与不重复计费 |
| [05｜Future信用分配](05_future_credit_assignment.md) | 已观测后缀、折扣return、mask、EOS/cap语义 |
| [06｜LOO与advantage](06_group_baseline_advantage.md) | 同题组、变长对齐、b-R、分布式与梯度边界 |
| [07｜PPO更新](07_ppo_clip_update.md) | old/new比值、clipping数值、loss reduction与日志 |
| [08｜可选JSD anchor](08_reference_anchor_jsd.md) | 常规teacher约束、完整/稀疏支持、成本与限制 |
| [09｜端到端循环](09_end_to_end_training_loop.md) | 采样→评分→冻结反馈→actor更新→重新采样 |
| [10｜GU独立对照](10_gu_alternative.md) | gate、G/(2-pS)、梯度性质、与主线互斥 |
| [11｜配置与训练对照](11_config_and_training_variants.md) | 必填参数、继承manifest、少量直接训练分支 |
| [12｜测试与失败监测](12_tests_and_failure_monitoring.md) | 数学/集成测试边界、重复/熵/cap/准确率 |
| [13｜仓库接入任务](13_repository_integration_tasks.md) | 建议模块接口、PR顺序、交付与未交付 |
| [14｜完整数值例子](14_worked_numeric_example.md) | 两条toy轨迹逐token表、精确R/A/PPO/JSD值 |
| [15｜目标语义与风险](15_objective_semantics_and_risks.md) | exact vs surrogate、长度/截断/代理风险限制 |
| [16｜来源与决策登记](16_sources_and_decision_register.md) | 原文、对话候选和新增工程选择分层 |

建议先读01和02，再按03→09实现；开发前必须检查11、15、16。GU可单独交给一个实施任务，不影响主线。

## 3. 附带的可执行内容

| 路径 | 作用 |
|---|---|
| `reference/math_core.py` | cost、history/Phi、future、LOO、PPO、JSD、GU纯tensor参考 |
| `reference/generate_example.py` | 生成精确的完整教学例子JSON |
| `reference/check_config.py` | 检查未锁定参数和互斥/固定约束；不启动训练 |
| `tests/` | CPU数值/梯度/配置测试 |
| `configs/train_plan.yaml` | 真实训练接入模板，缺值有意阻止启动 |
| `configs/demo_math.yaml` | 仅CPU例子参数，不是训练推荐 |
| `examples/worked_example.json` | 完整精度的示例中间量 |
| `validation/` | 本次测试与文档完整性检查结果 |

## 4. 本地运行数学参考

在现有兼容的Python/PyTorch环境中：

```bash
cd opd_visual_risk_implementation
python -m pytest -q
python -m reference.generate_example
python -m reference.check_config configs/train_plan.yaml
```

最后一个命令在未填写真实manifest和未锁定参数时**应当失败**。它不是一个ready-to-launch训练脚本。`requirements.txt`列出参考测试依赖，不替用户的模型训练栈锁版本。

## 5. 不应发生的“自动补全”

不改采样器以绕过old/new logp不一致；不删除EOS或把截断伪造为正常结束；不增加第三视图、attention、critic、候选重排、teacher takeover；不把Phi和未来成本再叠一个未解释的history gate；不自动将prob-gap换成log；不把测试通过写成模型性能提升。

## 6. 最关键的实施提醒

所有teacher-derived反馈先no_grad计算并冻结；本批更新只改变student。LOO在完整同题组上算好后才切microbatch。PPO分母来自真实采样策略而不是real/null teacher。JSD teacher一侧固定，但其混合分布m中的student路径保持梯度。

本包已给出可核验的数学与接口边界；真实仓库/VLM/GPU接入仍需按13完成。
