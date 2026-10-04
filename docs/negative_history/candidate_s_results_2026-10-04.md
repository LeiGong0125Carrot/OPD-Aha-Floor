# Candidate S（学生参照、零 visual-null 前向）结果与诊断（2026-10-04 中午）

> 口径：准确率均为 **gpt-oss-120b LLM-as-judge**；TB 用 TreeVGR 官方 HF greedy，V\* 用 OPD-Aha 官方 `infer.py`。
> 每次跑取 `{30,40,50}` 峰值；初期达标 = 两次跑 TB 峰值 ≥ 50.5 **且** V\* 峰值 ≥ 93.5。
> 设计与实施计划：`candidate_s_null_free_implementation_plan.md`（v0.1）；代码：OPD-Aha-Floor:sup `df44663`。
> **状态**：S r1 已出齐；Sr2 评测中（约 12:40 出齐），本文判定标注"待两次齐"。

---

## 1. 实现回顾

- 唯一改动：OPD-Aha 的 u = log p⁺ − log p⁰ 中的 p⁰ 换成当前学生分布 p^S（detach），u_S = log p⁺ − log sg(p^S)，q_S ∝ p⁺·exp(β u_S)，β=4；其余（JSD α=0.5、学生 top-100 + tail 支持集、pair 视图、冻结教师、IS、聚合、2459 题 6karmA、seed 42、3 卡/SP1、n2、lr 2e-6、51 步）全部继承 A。
- 配置 `counterfactual_reference=student`；`counterfactual_null_mode` 保持 `mean_color` 作为重构门。student 模式下 ray_trainer 不构造 null 图像、dp_actor 不执行 null 前向；`VOPD_FORBID_NULL=1` 使任何 null 路径 raise，S 两次训练都开着它。
- 验证：`scripts/test_sref.py` 9 组全绿，其中 reference=null 路径与 `bea75c6` 在 12 配置 × 2 聚合下 loss / 梯度 / 指标 **bit-identical**；独立 code review 无阻塞项；2 步冒烟通过（`reference_is_student=1`、`teacher_null_forward_frac=0`、`counterfactual_mean_color_fraction=0`、无 NaN）；反向检查通过（A 配置在禁 null 开关下如期报错）。
- 实测每步 145.6 s（Ahf 同配置 209.9 s）；其中 null 前向本身 25.3 s，其余差异含节点差异，不计入方法收益。

---

## 2. 结果

### 2.1 S r1（已出齐）

| | step30 | step40 | step50 | 峰值 | 初期口径 |
|---|---|---|---|---|---|
| TB | 47.90 | 47.41 | 48.40 | 48.40 | ✗（< 50.5） |
| V\* | 87.96 | 86.91 | 87.96 | 87.96 | ✗（< 93.5） |

### 2.2 Sr2（评测中）

TB step30 规则分 49.38（judge 待出）；其余约 12:40 出齐。本文不据此下判定。

### 2.3 与参照的位置

| 方法 | TB 峰值 | V\* 峰值 | 说明 |
|---|---|---|---|
| **A**（4 次） | 49.88 / 51.11 / 51.11 / 51.60 | 94.76 / 93.19 / 94.24 / 91.62 | 目标参照 |
| V0（原版 StdOPD，无 tilt，hide 视图） | 49.38 / 48.64 | 87.4 – 88.5 | 无 null 对比的下限 |
| 所有 null 底座的改造臂（B/C/D/Ahm/Ahf/X1 等） | 44.9 – 51.6 | **≥ 89.0** | 没有一个 V\* 低于 89 |
| **S r1** | 48.40 | **87.96** | V\* 落到 V0 水平 |

**读法**：S 的 V\* 掉到了 V0 的水平——去掉 null 之后，tilt 给 V\* 带来的约 5 点增益全部消失；S 在 V\* 上等价于普通蒸馏。TB 3 ckpt 池化 47.90（A 49.22）。

### 2.4 TB 分类别（S r1 3 ckpt 池化，相对 A 9 ckpt 池化）

| 类别 (n) | A | S | S − A |
|---|---|---|---|
| Perception/Attributes (29) | 55.2 | 64.4 | **+9.2** |
| Perception/Material (13) | 54.7 | 56.4 | +1.7 |
| Perception/Physical State (23) | 61.4 | 62.3 | +1.0 |
| Reasoning/Perspective Transform (85) | 14.1 | 14.5 | +0.4 |
| Reasoning/Contact and Occlusion (41) | 47.7 | 47.2 | −0.5 |
| Perception/Object Retrieval (16) | 83.3 | 81.2 | −2.1 |
| Perception/OCR (68) | 77.9 | 75.0 | −2.9 |
| Reasoning/Comparison (44) | 50.8 | 47.0 | −3.8 |
| Reasoning/Spatial Containment (29) | 69.3 | 64.4 | −5.0 |
| Reasoning/Ordering (57) | 38.2 | 32.7 | **−5.5** |

稳定翻转：A 稳对（≥7/9）而 S 全错（0/3）11 题；反向 4 题（Ahf 为 1 / 3）。形态与 sup 家族 / Ahm / X1 一致：局部属性涨，关系推理与 OCR 掉。

---

## 3. 机制读数（train_S.log）

| step | teacher_student_kl | target_tv | target_max_prob | target_entropy | 回答长度 |
|---|---|---|---|---|---|
| 1 | 0.192 | 0.429 | 0.946 | 0.554 | 146 |
| 11 | 0.123 | 0.251 | 0.994 | 0.509 | 186 |
| 21 | 0.140 | 0.306 | **0.9997** | 0.445 | 93 |
| 31 | 0.118 | 0.323 | 0.9987 | 0.453 | 112 |
| 51 | 0.119 | 0.304 | 0.9995 | 0.453 | 85 |

51 步均值：`teacher_student_kl` 0.138、`counterfactual_target_tv` 0.324、`target_argmax_tail_frac` **0.074**、`target_tail_mass` 0.008、`counterfactual_u_abs_mean` 1.59（从 0.80 单调上升）。Sr2 的 51 步均值与 r1 一致（KL 0.141、TV 0.302、tail argmax 0.071）。
对照 Ahf（null 底座）：target_max_prob 稳定 0.997，target_tv 0.23–0.29，末期回答长度 119。

---

## 4. 诊断

按设计文档 §15 的失败分流（实现错误 / 极端 target 与尾部 / 退化 / 信号不足）：

1. **不是实现错误**：默认路径 bit-identical、冒烟与反向检查通过、`VOPD_FORBID_NULL` 全程开启且 `teacher_null_forward_frac=0`、u 有限守卫未触发。
2. **退火只发生了一半**：师生 KL 从 0.19 降到 0.12 后停滞，目标没有退回 p⁺（target_tv 维持 0.30），而是停在极尖状态（max_prob ≈ 0.9995）。原因是 (p⁺/p^S)^β 在学生给得很低的 token 上爆炸——u 的支持集均匀均值从 0.80 一路升到 1.59。这正是文档 §5 指出的风险。
3. **尾部病理真实发生**：约 7% 的位置上，目标的最大类别是 **tail 桶**（学生 top-100 之外的集合），学生无法具体跟随。文档 §6.2 的算例预言了这一点；null 底座下该比例接近 0。
4. **crop 增量信号丢失**：V\* 考的是"看没看到小物体"，最依赖 null 对比隔离出的那部分信息。学生参照分不出"全图已知、crop 未知"，所以 V\* 最先崩、且崩到 V0。
5. **输出变短**：回答长度从 146 降到 85（Ahf 末期 119），与目标过尖一致；格式/截断率尚未单独核对。

**结论（待 Sr2 齐后正式化）**：用学生分布直接替代 null 教师的最简形式不成立。它反过来确认了 **null 前向提供的 crop 反事实不可由学生分布替代**，A 在 V\* 上的增益就来自这个对比。

---

## 5. 后续候选（不自动开跑，等决定）

| 候选 | 内容 | 预期 | 成本 |
|---|---|---|---|
| (a) 修尾部病理再试 S | 参照只在学生的显式 top-100 上定义，tail 桶 u=0 | 去掉一个数值病理；但 V\* 崩到 V0 更像信号丢失而非数值问题，救回 V\* 的把握不大 | 小改 + 2 次训练 |
| (b) 文本先验参照 | 教师只看文字（无图）的一次前向作参照，约 1/10 null 成本 | 仍是反事实，V\* 可能守住；TB 有"全图信息被当作视觉支持一起放大"的稀释风险（§7.5） | 需改 null 前向的 prompt 构造 + 2 次训练 |
| (c) 前缀共享 | 复用全图 KV，null 分支只算 crop/文本/回答 | 量与 A 完全相同，应复现 A；省约 90% null 成本；是工程改进不是方法 novelty | 中等工程量 |

若作者的核心要求是"不依赖 visual null"→ (b)；若是"省算力"→ (c)。

---

## 6. 来源

- 代码：OPD-Aha-Floor:sup `df44663`（S 实现 + 测试 + launcher `scripts/train_pair_sref.sh`）。
- 日志：`Vision-OPD-setup/logs/train_S.log`、`train_Sr2.log`、`train_Ssmoke.log`、`train_Anullchk.log`、`eval_S.log`、`vjudge_S.log`。
- 逐题：`eval/judge/treebench/pair_s_6karma-step{30,40,50}-priv-none-nothink_answer.jsonl`、`eval/judge/vstar/S-step{30,40,50}_seed42_answer.jsonl`。
- 前序：`summary_history_future_on_full_aha_2026-10-03.md` §7（去 null 讨论）、`candidate_s_null_free_implementation_plan.md`。
