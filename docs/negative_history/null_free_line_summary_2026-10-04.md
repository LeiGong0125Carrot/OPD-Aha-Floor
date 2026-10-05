# 去 visual-null 线总结：S / S-tail / S2 的设定与发现（2026-10-04 晚）

> 口径：准确率均为 **gpt-oss-120b LLM-as-judge**；TB 用 TreeVGR 官方 HF greedy，V\* 用 OPD-Aha 官方 `infer.py`；每次跑取 `{30,40,50}` 峰值，每个方法跑两次。
> 初期达标 = 两次 TB 峰值 ≥ 50.5 **且** V\* 峰值 ≥ 93.5。固定配置：2459 题高清 6karmA、seed 42、3 卡/SP1、n2、lr 2e-6、batch 48、51 步、β=4、JSD α=0.5、学生 top-100 + tail 支持集、冻结教师。
> 详细记录：`candidate_s_results_2026-10-04.md`（含用户 §7 复核）、`candidate_s_null_free_implementation_plan.md`、`summary_history_future_on_full_aha_2026-10-03.md` §7。
> 参照：A（原版 OPD-Aha + pair + n2）TB 峰 49.88 / 51.11 / 51.11 / 51.60，V\* 峰 94.76 / 93.19 / 94.24 / 91.62，池化 49.22 / 92.84；V0（StdOPD，无 tilt，hide 视图）TB 49.38 / 48.64，V\* 87.4–88.5。

---

## 1. 问题与动机

OPD-Aha 的目标 q ∝ p⁺·exp(β·u)，u = log p⁺ − log p⁰，p⁰ 来自一次额外的 teacher 前向：pair 视图 + `null_scope=last`，即**同一冻结教师看 [全图, 空白块]**（全图保留、只去掉 crop）。所以 u 是 crop 的增量，不是语言先验。
OPD-Aha 作者建议探索**完全不做 null 前向**也能接近 Aha 效果的方法（novelty 与独立性）。本线测试了三种零 null 的替代。

## 2. 三种设定

| 臂 | 底座 | 参照 | u | 前向数 | 代码 |
|---|---|---|---|---|---|
| **A**（参照） | 冻结教师 p⁺（全图+crop） | 冻结教师 p⁰（全图+空白块） | log p⁺ − log p⁰ | 3 | — |
| **S** | p⁺ | **当前学生** p^S（全图，detach） | log p⁺ − log sg(p^S) | 2 | `counterfactual_reference=student`，sup `df44663` |
| **S-tail（St）** | 同 S | 同 S，但 tail 桶的 u 置 0 | 同 S | 2 | `counterfactual_reference_tail_u_zero=True`，`b71a3d8` |
| **S2** | **sg(p^S)**（学生自己） | 单次冻结教师看 **hidebox**（全图上 2.42× GT 框区域均值色遮挡） | log sg(p^S) − log p_T(hidebox) | 2 | `counterfactual_reference=teacher`，`187a952`；数据 `train_6karmA_hidebox.parquet` |

共同点：每步省掉一次教师前向（实测 S 每步 145.6 s，对照 Ahf 209.9 s，其中 null 前向本身 25.3 s）；`VOPD_FORBID_NULL=1` 全程开启，任何 null 路径一进入就 raise；`reference=null` 默认路径与改动前 `bea75c6` 在 12 配置 × 2 聚合下 loss / 梯度 / 指标 bit-identical。

## 3. 结果

| 臂 | TB 30/40/50 | TB 峰 | V\* 30/40/50 | V\* 峰 | 判定 |
|---|---|---|---|---|---|
| S r1 | 47.90 / 47.41 / 48.40 | 48.40 | 87.96 / 86.91 / 87.96 | 87.96 | ✗ |
| Sr2 | 50.12 / 48.15 / 47.16 | 50.12 | 85.86 / 87.43 / 87.96 | 87.96 | ✗ |
| St r1 | 46.42 / 48.64 / 48.15 | 48.64 | 87.43 / 87.96 / 87.43 | 87.96 | ✗ |
| Str2 | 48.40 / 48.15 / 48.40 | 48.40 | 87.43 / 85.86 / 87.43 | 87.43 | ✗ |
| S2 r1 | 训练中（约 23:30 训完），**第 6 步起生成塌缩** | — | — | — | 见 §4.3 |

池化（6 ckpt）：S TB 48.19 / V\* 87.35；St 48.02 / 87.26；A 49.22 / 92.84。
**S 与 St 的全部 12 个 V\* 点都落在 85.9–88.0，与 V0 区间重合。** TB 分类别（相对 A）形态与 sup 家族 / Ahm / X1 一致：Attributes +10 左右，Ordering / Spatial Containment −5 到 −7，OCR −3。

## 4. 发现

### 4.1 学生参照不携带 crop 增量；V\* 增益完全依赖 null 对比
去掉 null 后 V\* 回到 V0 水平，说明 A 相对 V0 的约 5 点 V\* 增益全部来自"同一教师、有/无 crop"的对比。学生分布替代不了它：第 0 步之后 u_S 混入了学生已学到的部分，而且语义变成"学生还没学会的部分"。

### 4.2 尾部病理是真实现象但不是失败原因
S 的目标在约 7%（最坏 micro-batch 包络，非全局）的位置把质量推到学生 top-100 之外的 tail 桶，源于 (p⁺/p^S)^β 在学生低概率 token 上爆炸。St 把 tail 的 u 置 0 后，全局 tail-argmax 比例降到 0.0002，但 TB / V\* 与 S 几乎相同。**排除了"数值病理是主因"**。

### 4.3 以学生自身为底座会塌缩（S2）
S2 前 4 步正常（回答约 130–170 token），第 5 步回答长度 38，第 6–10 步只剩"答案字母 + 结束符"2 token，第 11 步起第一个 token 就是结束符（回答为空），训练 ppl = 1.0、grad_norm → 1e-6，进入退化不动点。
机制：目标 ∝ sg(p^S)·exp(4·(log p^S − log p_T))，放大"看到完整图的学生比看不到 GT 区域的教师更确定"的方向，而这个方向在所有位置上最一致的实现就是"直接给答案、提前结束"；学到的分布又成为下一步更尖的底座，没有冻结分布拉回。S 有 p⁺ 作锚所以不塌；S2 去掉了锚。
（用户文档 §10.3 曾预告"刹车不是稳定性保证"。）

### 4.4 观测层修正
`reduce_metrics` 按字段名含 `max`/`min` 子串取 np.max/np.min，原 `target_max_prob` / `target_argmax_tail_frac` 是最坏 micro-batch 的包络而非全局值（用户复核发现）。已改为计数式上报（`*_sum` + `target_stat_count`，mean(sum)/mean(count) = token 加权全局值）并新增显式 MAX 的 `target_top1_prob_worst_mb`（sup `ca3245b`）。旧日志（含 A/Ahf/S/Sr2）的这两个字段只能按包络读。

### 4.5 与 OPD-Aha 论文的关系
论文附录 C 的 ablation（Table 5 散度、Table 6 概率 vs log 重构、Table 7 null 构造：噪声 / 不匹配图 / 黑图 / 均值色）全部保留 null 前向；方法节明确写"requires a comparison that does not depend on the student's distribution"、"attributes the change to visual input rather than differences between models"。S/St/S2 是把论文在动机中排除的比较实际跑了一遍，结果与其论点一致；这不是对论文 ablation 的重复，可直接用于写作定位。

## 5. 结论与剩余方向

**结论**：在本配置下，"去掉 visual-null 前向"的三种最简替代（学生参照、学生参照 + 尾部修正、学生底座 + 遮挡教师参照）都不成立。精确的 crop 反事实在数学上就是同一模型的两次评估，用学生分布替代会丢信号，用学生做底座会塌缩。

| 剩余候选 | 性质 | 预期 |
|---|---|---|
| 文本先验参照（教师只看文字，约 1/10 null 成本） | 仍是一次反事实前向，不是零 null | V\* 可能守住，TB 有稀释风险 |
| 前缀共享（全图 KV 复用，null 分支只算 crop/文本/回答） | 量与 A 完全相同，工程改进 | 应复现 A，省约 90% null 成本 |
| "去 zoom 保对比"（冻结教师全图为底座 + hidebox 参照，3 前向） | 回答"A 的增益来自 zoom 还是对比方向" | 不省算力；若 V\* 守住说明对比方向是关键 |

**待决定**：S2 r1 是否停训（评测只会给出空回答的分数）；S2r2 是否原样重跑（预期同点塌缩）。

## 6. 流程记录（10-04）
- checkpoint 保留规则：评测结束前保留全部中间 checkpoint（45/51 也可能是峰值），仅磁盘 <500G 时删最老的非 30/40/50；`merged/` 每臂只留 TB 峰值步（137→43 目录，释放约 975G）。
- `pair_ahaan8sp1` / `pair_supn2` 的 TB judge 文件曾被重复评测追加（468–810 行），已按首次出现去重（备份 `_dup_backup/`），峰值不变。
- 编排器 bug：脚本顶部 unset 了全部 `SLURM_*` 变量，`set -u` 下引用 `$SLURM_JOB_ID` 使认领失败，GPU 空 12 分钟；已改用 hostname + pid。规则：编排器内不引用 `SLURM_*`。
- 代码：OPD-Aha-Floor:sup `df44663`（S）、`ca3245b`（指标）、`b71a3d8`（St）、`187a952`（S2）；测试 `scripts/test_sref.py` 12 组。
