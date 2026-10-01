# 09｜端到端训练循环与 stop-gradient 边界

**目标：** 把所有已解释模块接进一次训练更新，不再新增算法机制。
**路径：** 主候选 temporal-risk PPO；GU独立分支见10。

## 1. 六个阶段

| 阶段 | 操作 | 固定/可训练 |
|---|---|---|
| A | 用当前student、原始全图采样既定数量完整response | 保存采样策略版本与old logp |
| B | 同一冻结teacher在real/null下teacher-force评分 | teacher固定，response固定 |
| C | cost→exclusive history→increment→discounted returns | no_grad |
| D | 跨卡组装同题组，LOO baseline→advantage | no_grad，先于microbatch |
| E | 新student重放固定response，PPO+可选JSD | 只student有梯度 |
| F | 结束本批更新，刷新rollout actor，重新采样 | 下一批重新评分，不原地编辑旧response |

## 2. 建议接口伪代码

```python
# Framework adapters below are proposed interfaces, not existing repository APIs.
rollouts = sampler.generate(prompts, policy_version=current_version,
                            sampling_spec=inherited_sampling)
validate_rollouts(rollouts)

with no_grad():
    teacher = scorer.score_real_and_null(rollouts)
    signal = visual_cost(teacher.logp_plus_y, teacher.logp_null_y,
                         rollouts.mask, kind=cfg.cost_kind)
    risk = history_and_risk(signal['cost'], rollouts.mask, cfg.risk_transform)
    local_returns = discounted_cost_to_go(risk['increment'], rollouts.mask, cfg.gamma)
    grouped = gather_complete_prompt_groups(local_returns, rollouts)
    feedback = leave_one_out(grouped.returns, grouped.mask, grouped.group_ids)
    frozen_feedback = scatter_back_with_ids(feedback)

for epoch in range(cfg.ppo_epochs):
    for microbatch in fixed_rollout_batches:
        actor = student_adapter.score_for_update(microbatch)
        policy = ppo_clip_loss(actor.policy_logp, microbatch.old_policy_logp,
                               microbatch.advantage, microbatch.mask,
                               cfg.clip_epsilon, cfg.loss_reduction)
        anchor = optional_anchor(actor, microbatch.teacher_cache)
        total_loss = policy['loss'] + cfg.anchor_weight * anchor
        scaled_backward_with_global_denominators(total_loss)
    optimizer_step_at_inherited_accumulation_boundary()

synchronize_rollout_weights_after_update()
```

注意：optimizer的真实step/microbatch边界必须继承现有框架；伪代码不是建议“每个epoch只更新一次”。反馈和old logp在本批所有实际optimizer step期间保持固定。

## 3. 每次更新中哪些数不动

固定：response_ids、old_policy_logp、teacher scores/cache、c/H/d/R/b/A、sampling/processor元信息。

变化：当前student权重、new policy logp、student raw full分布、PPO ratio、anchor、各项loss。

不对采样token离散序列反传；不对teacher反传；不每个actor microbatch重算history/LOO。历史与未来不是“没有梯度就没用”，它们通过固定A决定actor更新。

## 4. 贯穿示例

某位置old probability=.70。Real teacher=.70、null=.80→c=.10。过去H=.80；本步及后续c=`[.10,.30,.20]`。

Gamma=1、候选Phi得到R^Phi=.9123179275；其他rollout参照=.1754614789→A=-.7368564487。

更新中new probability=.63→ratio=.90。教学epsilon=.20，不触发clipping→policy loss=.6631708038。

Teacher full三token分布 `[.70,.20,.10]`，student=`[.63,.27,.10]`→JSD=.0035375836。教学lambda=.10→总loss=.6635245622。

若用聊天舍入A=-.737，会得到policy loss=.6633；两者不能混成一个精确测试常数。完整精确结果由 examples JSON 提供。

## 5. 分布式与gradient accumulation

LOO需要完整组，不能在每GPU或microbatch独立用缺失同题样本算baseline。通信只需分组scalar序列和IDs，不必all-gather完整词表。

Token mean要使用整个有效训练batch的有效token分母。DDP默认平均各rank梯度时，应按框架约定把本rank numerator缩放到global denominator；直接对每个变长microbatch求mean再平均会改变权重。

`trajectory_sum_mean` 则用全局有效response数。两种reduction不可无记录切换；GPU接入需验证单卡/多卡结果在可接受数值容差内一致。

## 6. 缓存与资源

Teacher只需一次real和一次null的teacher-forced评分过程，内部可分microbatch。缓存gathered logp与可选anchor支持，不默认存完整[B,T,V]。

Old logp可保存而不常驻old模型副本，但必须与实际sampler策略一致。Rollout推理actor与训练actor不同后端时，更新前验证policy版本和初始ratio。

## 7. 异常与恢复

非有限概率/ratio/梯度：停止该更新，保存样本ID、配置hash和必要诊断；不静默删除高风险token继续优化。

Incomplete rollout group：不能拿残余样本计算LOO。Resume要恢复student、optimizer、scheduler、RNG、global step、sampler版本。冻结teacher始终来自初始teacher checkpoint，不能从最新student重建。

不自动改变lambda/gamma/repetition penalty/输出长度来救火；这会变成新的方法配置。安全暂停属于工程保护，不是新训练奖励。

## 8. 验收

固定batch反馈hash在所有更新epoch一致；student产生梯度teacher不产生；old logits记录不更新；样本/组边界不串；下一批才使用新rollout。真实GPU模型接入与训练尚未执行。
