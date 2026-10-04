# Candidate S（学生参照、零 visual-null 前向）结果与诊断（2026-10-04 中午）

> 口径：准确率均为 **gpt-oss-120b LLM-as-judge**；TB 用 TreeVGR 官方 HF greedy，V\* 用 OPD-Aha 官方 `infer.py`。
> 每次跑取 `{30,40,50}` 峰值；初期达标 = 两次跑 TB 峰值 ≥ 50.5 **且** V\* 峰值 ≥ 93.5。
> 设计与实施计划：`candidate_s_null_free_implementation_plan.md`（v0.1）；代码：OPD-Aha-Floor:sup `df44663`。
> **状态**：S r1、Sr2 均已出齐（Sr2 12:19）。**两次判定：未达标**（见 §2.2）。后续 (a) S-tail 已于 10-04 12:51 开跑（`b71a3d8`，tag `pair_St[r2]_6karmA`，hold 20814117）。

> **2026-10-04 复核提示**：原始实验数值及 §1–§6 记录保留；`target_max_prob` / `target_argmax_tail_frac` 的聚合口径存在源码可定位的问题，不能直接作全局均值／位置比例解释。结果与诊断的限定见 [§7 复核补充](#s-metric-review)。本次仅更新文档，未修复代码或新增实验。


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

### 2.2 Sr2（已出齐）与两次判定

| | step30 | step40 | step50 | 峰值 | 初期口径 |
|---|---|---|---|---|---|
| TB | 50.12 | 48.15 | 47.16 | 50.12 | ✗（< 50.5，差 0.38） |
| V\* | 85.86 | 87.43 | 87.96 | 87.96 | ✗（< 93.5） |

**S 两次判定：未达标。** TB 峰值 48.40 / 50.12，V\* 峰值 87.96 / 87.96。6 ckpt 池化：TB 48.19（A 49.22），V\* 87.35（A 92.84，V0 87.4–88.5）。
两次的 V\* 全部六个 checkpoint 落在 85.9–88.0，与 V0 区间重合；两次 TB 的最高点分别在 step50（r1）和 step30（r2），无一致的时间形态。

6 ckpt 池化分类别（相对 A 9 ckpt）：Attributes **+10.3**、Physical State +1.7、Material +1.7、Perspective +1.2；Ordering **−5.8**、Spatial Containment −5.6、OCR −2.7、Comparison −2.7、Object Retrieval −2.1、Contact −1.4。稳定翻转：A 稳对（≥7/9）而 S 稳错（≤1/6）9 题，反向 4 题。

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

**结论（两次齐，10-04 12:30）**：用学生分布直接替代 null 教师的最简形式在本配置下不成立，两次 V\* 都落在 V0 区间。按 §7 的复核限定，这支持"首轮最简 S 没有保住 A 的效果、尤其是 V\*"，不单独证明"null 反事实不可替代"；尾部 / 极端比值是合理的失败假设，其发生比例须用 `ca3245b` 之后的计数式指标重新测量（Sr2 日志仍是旧字段）。

---

## 4.1 与 OPD-Aha 论文的关系（10-04 核对 `revision_opd_arxiv.pdf`）

论文附录 C 的 ablation（Table 5 监督散度 FKL/RKL/JSD；Table 6 概率空间 vs log 概率重构，对照"标准特权目标" V\* 89.0；Table 7 visual null 的构造：高斯噪声 / 不匹配图 / 黑图 / 均值色，Avg6 79.6–80.4）**全部保留 null 前向**，没有"以学生分布为参照"的一组。
论文方法节（Eq. 2–4 前后）明确排除了这条路："Isolating this surviving visual signal requires a comparison that does not depend on the student's distribution"；"Even when p⁺_t and p^S_t are nearly indistinguishable, u_t can remain nonzero and reveal the visual preference hidden by their agreement"；"Because p⁺_t and p⁰_t use the same teacher and student prefix, u_t attributes the prediction change to visual input rather than differences between models"。
**定位**：S 不是论文已有的 ablation，而是把论文在动机中排除的比较实际跑了一遍；两次 V\* 落回"标准特权目标"量级（论文 89.0；我们 V0 87.4–88.5），在经验上与论文的论点一致。这一定位可直接用于写作，不构成对论文消融的重复。

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

---

<a id="s-metric-review"></a>
## 7. 复核补充：结果判定、指标聚合与诊断边界（2026-10-04）

> **复核范围**：原始结果快照 [ae1d592][R1]；实现快照 `df4466353f8f010f82d84bbdb5eb7b7c353507dc`。本节依据该报告和源码复核整理，区分报告记录、源码确认与分析推论。不新增实验结果，不重跑训练或 judge，不修改任何训练代码。
>
> **阅读约定**：§1–§6 保留原记录，所有实验数值不变；其中对指标和机制的解释，须结合本节的限定阅读。本节没有把 Sr2 的规则分转换为 judge 结果，也不根据原排程推断 Sr2 已完成。

### 7.1 结果仍然未达标，但性能相近不等于机制等价

根据原报告，S r1 的 TB 峰值为 **48.40**、V\* 峰值为 **87.96**，分别低于 50.5 和 93.5；此判定不受下面的诊断指标问题影响。Sr2 在被复核的快照中仍待 judge 结果。[R1]

S 的 V\* 落在 V0 所报 87.4–88.5 的数值范围，只能说明性能水平接近。原表 V0 使用 **hide**，S 使用 **pair**，且 target 定义不同；不能据此认定二者优化机制等价，或把两者作为完全匹配条件的消融。

TB 的描述性变化仍保留：池化 47.90（A 为 49.22）；Attributes +9.2、Ordering −5.5、Spatial Containment −5.0、Comparison −3.8、OCR −2.9；A 稳对而 S 全错 11 题，反向 4 题。这些结果展示类别与逐题行为差异，但一个运行内多个 checkpoint 不是独立训练重复，不能直接转换为因果或显著性结论。[R1]

报告的 145.6 s 与 Ahf 的 209.9 s 不能全部归因于删除 null。保留原文的限定：null 前向本身约 25.3 s，其余差异包含节点差异；本次没有新增速度测量。[R1]

### 7.2 源码确认：两个名称含 max 的指标被再次取最大值

**第一层：loss 函数计算 microbatch 内的有效位置均值。** 令

$$
m_t=\max_v q_t(v),\qquad a_t=\mathbf{1}[\arg\max_v q_t(v)=\mathrm{tail}].
$$

`compute_self_distillation_loss()` 对它们做 `masked_sum / valid_token_count`，随后用 `.detach().item()` 存为普通标量，字段分别是 `self_distillation/target_max_prob` 与 `self_distillation/target_argmax_tail_frac`。[C1]

**第二层：actor 收集各 microbatch 的标量，再调用统一 reducer。** `update_policy()` 通过 `append_to_dict(metrics, micro_batch_metrics)` 收集列表；末尾对这些列表调用 `reduce_metrics(local_metrics_to_reduce)`。这两个字段不是显式指定聚合类型的 `Metric` 对象。[C2]

**第三层：reducer 按字段名包含的子串选择 max/min/mean。** 对普通数值列表，`reduce_metrics()` 的实际分支为：[C3]

```python
elif "max" in key:
    metrics[key] = np.max(val)
elif "min" in key:
    metrics[key] = np.min(val)
else:
    metrics[key] = np.mean(val)
```

| 字段（省略 self_distillation/） | loss 函数输出 | actor 本地列表汇总 |
|---|---|---|
| `target_max_prob` | 当前 microbatch 的平均最大类别概率 | **max**，名称含 `max` |
| `target_argmax_tail_frac` | 当前 microbatch 的 tail-argmax 比例 | **max**，`argmax` 中也含 `max` |
| `target_entropy` | 当前 microbatch 的平均熵 | mean |
| `target_tail_mass` | 当前 microbatch 的平均 tail 质量 | mean |

所以，本地更新结束时，前两个字段至少经过了 **“各 microbatch 内先平均，再在 microbatch 之间取最大值”**。后续跨卡、跨 step 的汇总不会自动补回被丢弃的分子与分母，不能把这种输出直接解释为全局有效位置均值或比例。

**合成例子，不是实测修正值**：四个 microbatch 的有效位置数相同，tail-argmax 比例为 `[0, 0, 0.02, 0.08]`。整体比例应为 0.025（2.5%），当前名称规则输出 0.08（8%）。不能据此推算真实运行的正确比例，只能确认现有字段不具有所声称的全局比例含义。

另一个边界：不含 `max` 的字段在此层取 microbatch 标量的算术平均；若各 microbatch 有效位置数不同，它也不自动等于全局 token 加权平均。这个问题应与上述 max 聚合问题分开处理。

### 7.3 对 §3–§4 的读数和结论作明确限定

| 原报告读数／解释 | 复核后的准确读法 |
|---|---|
| `target_max_prob ≈ 0.9995`，因而整体 target 几乎 one-hot | **不能作整体均值解释**。该值来自带 max 汇总的统计；按当前路径，说明存在很尖的 microbatch，不等于全部有效位置的平均最大概率约为 0.9995。 |
| `target_argmax_tail_frac = 0.074`，因而约 7.4% 的训练位置以 tail 为最大类别 | **不能作全局位置比例解释**。tail 成为最大类别的现象存在，但总体发生率未由这个字段确定；报告中的跨 step 平均仍然是在平均已被 max 汇总的量。 |
| `target_tail_mass = 0.008`、`target_entropy ≈ 0.45` 与上述值放在一起解释 | 指标汇总算子不同，不能把它们当作同一全局位置集合、同一权重下的联合统计来推断尾部占比或整体尖锐程度。数值保留，口径需统一。 |
| “tail 目标学生无法跟随” | Student 可以学习 tail 集合的**总质量**；缺失的是集合内部逐 token 的明确目标，而不是完全没有可优化的梯度。 |
| “不是实现错误” | 原报告的回归、独立参照、禁 null 和 smoke 检查支持已测试路径的正确性，但不能排除全部工程或观测问题；本次确实定位了诊断汇总口径问题。 |

**影响范围**：上述问题发生在指标汇总与解释层，不是本次发现了 S 的 JSD/target 数学路径错误；也不直接改变原有梯度、checkpoint 或 benchmark judge 分数。本节没有修改 reducer，更没有声称修复后性能会提高。

修订前，A/Ahf 的同名指标也应检查是否经过同一条 reducer 路径；若是，不能拿其 `target_max_prob` 当作已经校准的全局均值参照。

### 7.4 为什么 KL 下降、log-ratio 差距仍可能较大？

源码中的 `teacher_student_kl` 是在当前 top-k＋tail 压缩分布上的

$$
D_{\mathrm{KL}}(p_t^+\|p_t^S)
=\sum_v p_t^+(v)\log\frac{p_t^+(v)}{p_t^S(v)},
$$

而 `counterfactual_u_abs_mean` 是在同一支持集上对

$$
\left|\log p_t^+(v)-\log p_t^S(v)\right|
$$

做**类别均匀平均**，不是按 teacher 概率加权。定义来自源码；以下是分析推论。[C1]

两者同时呈现“KL 下降、均匀平均的绝对差距上升”并不矛盾：teacher 主要概率质量上的匹配可以改善，而低概率类别上的 log-ratio 仍较大。支持集本身也随 student 变化；仅凭跨 step 均值，不能定位是哪些固定 token 引起变化。

S 使用指数外推，所以师生 KL 下降并不保证 target 已接近 teacher。报告中的 `target_tv` 从 0.429 到 0.304、末期仍约 0.30，与“target 尚未整体退回 teacher”的判断相容；但 `target_max_prob` 的聚合问题意味着，不能进一步直接声称所有位置都停在近 one-hot 状态。[R1]

**结论**：自动退火是当师生分布接近时的条件性质，不是沿训练步数保证发生的过程。极端概率比是合理的失败假设，尚未由当前汇总读数单独完成因果验证。

### 7.5 null-free 的失败边界与后续候选的含义

原报告支持的直接结论是：**最简 S 在当前配置下的首轮没有保住 A 的效果，尤其是 V\*。** 它不证明任意 null-free 方法都不可能成功，也不把本次下降唯一归因于 crop 增量丢失。参照语义变化、外推强度、尾部近似和优化过程仍可能共同作用。

输出长度从所列 step1 的 146 变成 step51 的 85，是报告的观察；仅凭这两个读数，不能证明生成塌缩或“target 过尖导致缩短”。保留“格式／截断率尚未核对”的原状态。[R1]

原 §5 的候选继续列为**未决、未验证，不自动开跑**：

| 候选 | 可以回答的问题 | 本次复核的限定 |
|---|---|---|
| tail 的 `u=0` | 去掉聚合 tail 的额外指数放大后，S 是否改善？ | 不保证全局归一化后的 tail 概率不变，也不解决显式低概率 token 的极端比值；不是恢复原 null 视觉差分。 |
| 文本先验参照 | 更便宜的文本参照能否提供有效监督？ | 仍需一次无图参照计算，不能等同于完全没有信息移除对照；成本约 1/10 与性能预期都待实测。 |
| 前缀共享 | 保留原对比时，能复用多少计算？ | 仍依赖 visual-null；约 90% 的 null 成本节省是候选预期，不是本次实验结果。 |

在这三条之间作机制驱动的选择之前，应先澄清指标口径。性能未达标已经成立，但“尾部问题占多大比例、是否是主要失败原因”尚未闭合。

### 7.6 建议的观测修订与验收（尚未实施）

本次只记录复核意见，不修改训练代码。后续若修订观测，建议优先修正聚合定义，而不是立即新增算法臂：

1. **以分子／分母计数定义全局指标**。对有效位置 mask 为 M 的原始 target，记录 `sum(M * max(q))`、`sum(M * 1[argmax(q)=tail])`、`sum(M * q_tail)`、`sum(M * H(q))` 及共同的 `sum(M)`。在 microbatch 与数据并行样本维正确汇总后，再除以有效位置总数；不能重复统计并行复制的同一位置。
2. **不要只给字段改名或只用等权 MEAN 替代 max**。这能避开名字匹配，但不同 microbatch 长度下仍可能不是 token 加权均值。需要保留计数；如需最坏 microbatch 指标，单独明确命名并记录，不能与全局平均混用。
3. **端到端验证指标管道**。除 loss 层测试外，覆盖 `append_to_dict → reduce_metrics → 最终记录`：上面等长度例子应得到 2.5% 的全局比例，最坏 microbatch 为 8%；再加有效长度不等和 padding 的例子，核对带计数结果。仅在 loss 层断言字段正确，检测不到本次发现的后处理问题。
4. **区分可恢复与不可恢复的数据**。若已保存逐位置或逐 microbatch 的原始值及计数，可重新汇总；若旧日志只保存 max 汇总后的标量，不能无损恢复历史全局均值／比例。不能补写估计值冒充实测值。
5. **保持实验状态边界**。等待并记录 Sr2 的真实 judge 结果；不把旧预计出分时间当作完成证据，不自动启动 tail 修正、文本参照或前缀共享训练。本节没有执行 GPU、分布式测试或重算原始训练日志。

**复核总结**：S 首轮失败是直接实验事实；max 名称驱动的诊断聚合问题可在源码中定位；“普遍接近 one-hot”“约 7% 位置 tail 最大”及“null 反事实不可替代”的强解释暂不成立。先澄清观测，再决定修尾部、换参照或研究计算复用。

### 7.6.1 观测修订已实施（2026-10-04 12:05，OPD-Aha-Floor:sup `ca3245b`）

按 §7.6 的 1–3 条修改了 `compute_self_distillation_loss()` 的指标输出；不改任何训练数学，`reference=null` 路径的 loss / 梯度 / 既有均值指标与 `bea75c6` 仍 bit-identical（`scripts/test_sref.py` 第 9 组）。

| 原字段 | 现字段 | 聚合含义 |
|---|---|---|
| `target_max_prob`（名含 max → 被 np.max） | `target_top1_prob_sum` + `target_stat_count`；`target_top1_prob`（等权 micro-batch 均值，数值与旧字段的 micro-batch 内均值逐比特相同）；`target_top1_prob_worst_mb`（显式 `Metric(MAX)`） | mean(sum)/mean(count) = token 加权全局均值；worst_mb 单独命名 |
| `target_argmax_tail_frac`（"argmax" 含 max → 被 np.max） | `target_tail_top1_sum` + `target_stat_count`；`target_tail_top1_frac`（等权均值） | 同上 |
| `target_entropy`、`target_tail_mass`（本来就是 mean） | 保留，另加 `_sum` | 可按计数重算 token 加权值 |

测试（`test_sref.py` 第 10 组）：loss 函数输出的所有普通字段名不含 `max`/`min` 子串；`append_to_dict → reduce_metrics` 管道上复现 §7.2 的例子（`[0,0,0.02,0.08]` → 计数聚合 2.5%，旧命名 8%，显式 MAX 8%）；不等长 micro-batch 下计数聚合 = token 加权 0.5%、等权均值 5% 两者区分；生产 sum/count 与逐位置手算一致。

未做的事：跨 dp worker 的汇总仍是 ray_trainer 既有路径（各 worker 等权），worker 间 micro-batch 数相同时 mean(sum)/mean(count) 仍等于全局值；§7.6 第 4 条——旧日志（含 S r1/Sr2、A、Ahf 的 `target_max_prob`）只有 max 后标量，不可恢复，文中一律按"最坏 micro-batch 包络"读。此次修订发生在 Sr2 训练之后，Sr2 日志仍是旧字段。

### 7.7 本节来源（固定提交，不随分支漂移）

- [R1：本次原始结果记录，ae1d592][R1]。测试通过、速度、judge 分数与训练读数均引用这份报告，本次未独立重跑。
- [C1：core_algos.py，df44663][C1]，`compute_self_distillation_loss()`：student-reference、KL/u 诊断定义、逐位置 target 统计及 microbatch 标量输出。
- [C2：dp_actor.py，df44663][C2]，`update_policy()`：收集 microbatch 指标，末尾调用 `reduce_metrics()`；同文件还包含 null 前向守卫。
- [C3：metric/utils.py，df44663][C3]，`reduce_metrics()`：普通列表按名称中 `max`／`min` 子串分派聚合；`Metric` 对象另走显式聚合分支。

[R1]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/ae1d592d0939f7459ba62774a870a45b1722b733/docs/negative_history/candidate_s_results_2026-10-04.md
[C1]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/df4466353f8f010f82d84bbdb5eb7b7c353507dc/verl/trainer/ppo/core_algos.py
[C2]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/df4466353f8f010f82d84bbdb5eb7b7c353507dc/verl/workers/actor/dp_actor.py
[C3]: https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/df4466353f8f010f82d84bbdb5eb7b7c353507dc/verl/utils/metric/utils.py
