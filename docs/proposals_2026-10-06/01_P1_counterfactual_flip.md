# P1：反事实翻转蒸馏（Counterfactual-Flip Distillation, CFD）（2026-10-06，v0.1）

> 一句话：OPD-Aha 的有效监督只发生在 crop 把教师的 argmax 翻转的位置；在这些位置用锐化的 $p^+$ 并加大损失权重，其余位置退化为 Vision-OPD。去掉 β、指数倾斜与 tail 病理，预算分配的依据从"位置 / 熵 / 剂量"改为"反事实是否翻转决策"。

## 1. 痛点

- **目标重构组件堆叠。** OPD-Aha（β、exp 倾斜、top-k+tail）、VAD（投影、支持/反驳预算、clip、弱正则）、Decomposed OPD（归一化、语言保持项）都在往目标里加组件；OPSA 与 "Mirage of Performance Gains" 类工作质疑此类增益可能来自分布整形而非信号本身。领域缺一个**最小充分目标**。
- **对比信号里的风格泄漏。** VAD 残差词云里 zoom / crop / blurry 占比高；本仓库探针（机制结论 §3 限定 2、§5）发现 $u$ 在首 token 上编码"先解释再作答"的风格，教师 prompt 的 "Zoomed-in view" 措辞经 $u$ 泄漏给学生（$|b|$ 最大的词含 zoom +1.07）。β 对所有位置一视同仁地放大，这些位置被一并放大。

## 2. 本仓库证据如何推出该方法

| 证据 | 推论 |
|---|---|
| γ=50 硬标签 49.55 vs A 50.93，V\* 无损 | 增益主体是 argmax；软分布贡献 ≈1.4 |
| $u$ 质量集中在感知描述 token，答案位 TV($p^+$,$p^0$) 仅 0.012–0.021 | 有效监督集中在少数位置 |
| 任何让目标更尖的改动（Ahm / X1 / γ）都伤 OCR 与关系推理 | 副作用来自**非证据位置**被锐化 |
| sup-only V\* 终点持平 A、TB 丢关系推理 | 正半边（往哪个 token 走）在翻转位置是必要的 |
| Amis：donor crop 作 null 失败 | 翻转判定必须用真正的无证据视图（mean_color），不能换 |

合起来：A 的增益 ≈ "在 crop 翻转 argmax 的位置，学 $p^+$ 的 argmax"；A 的副作用 ≈ "在其余位置也做了软倾斜"。CFD 保留前者、去掉后者。

## 3. 方法

记 $a^+_t=\arg\max_v p^+_t(v)$，$a^0_t=\arg\max_v p^0_t(v)$，均在学生 top-100 + tail 支持集上取。

**翻转集**
$$\mathcal F=\{\,t:\ a^+_t\neq a^0_t,\ a^+_t\notin\text{tail}\,\}$$

**目标分布**
$$q_t=\begin{cases}\mathrm{softmax}\big(\gamma\cdot\log p^+_t\big) & t\in\mathcal F\quad(\gamma\ \text{锐化温度，默认}\ \gamma=\gamma_{\mathrm{flip}})\\ p^+_t & t\notin\mathcal F\end{cases}$$

**损失**
$$\mathcal L=\frac{\sum_t w_t\,\mathrm{JSD}_\alpha(q_t\,\|\,p^S_t)}{\sum_t w_t},\qquad w_t=1+\lambda\cdot\mathbf 1[t\in\mathcal F]$$

$\gamma_{\mathrm{flip}}\in\{1, 4, 50\}$（1 = 只做预算分配不锐化；50 = 本季 γ 臂的硬标签；4 = 中点），$\lambda\in\{0, 1, 3\}$。主臂建议 $\gamma=50,\lambda=1$；$\gamma=1,\lambda=1$ 作为"纯预算分配"对照；$\gamma=50,\lambda=0$ 与本季 γ 臂的区别只在非翻转位置是否锐化，直接测"非翻转位置的锐化是否就是副作用来源"。

与 A 的关系：A 在所有位置用 $p^+e^{\beta u}$；CFD 在 $\mathcal F$ 外用 $p^+$（β=0），在 $\mathcal F$ 内用温度锐化而不是 $u$ 倾斜。CFD 不含 $u$ 的数值，只用 $u$ 的符号信息在 argmax 上的体现，因此天然没有 tail 爆炸与风格幅度放大。

### 3.1 可选变体（不进主臂）

- **CFD-u**：翻转位置目标改为 $\mathrm{softmax}(\log p^+_t+\beta u_t)$（即 A 的目标只在 $\mathcal F$ 内使用）。它回答"翻转位置内部，$u$ 倾斜与温度锐化哪个更好"。
- **软翻转**：用 $\mathrm{TV}(p^+_t,p^0_t)$ 或 $p^0_t(a^+_t)$ 的连续量替代硬指示函数。本季教训是连续剂量（Ahm/Ahf）不如离散选择；先做硬版本。

## 4. 实现落点

全部在 `verl/trainer/ppo/core_algos.py::compute_self_distillation_loss`，`counterfactual_null_mode="mean_color"` 分支内，null 前向、数据流、`dp_actor` 不动。

### 4.1 配置（`verl/workers/config/actor.py` + `actor.yaml`）

```
counterfactual_target_mode: "tilt"      # tilt（现有 A）| flip（P1）
flip_gamma: 50.0
flip_lambda: 1.0
flip_tail_policy: "exclude"             # exclude | include（a⁺ 落在 tail 桶时是否计入翻转集）
```

运行时守卫（loss 函数内，因训练路径不执行 dataclass 校验）：`flip` 必须 `counterfactual_null_mode is not None`；与 `hist_*` / `floor` / `tanh` / `gamma` 互斥；`flip_gamma>0`；`flip_lambda>=0`。

### 4.2 loss 层改动（示意）

```python
# 在 L1155 分支内，拿到 teacher_real_distill_log_probs (B,T,K+1), teacher_null_distill_log_probs
if counterfactual_target_mode == "flip":
    a_real = teacher_real_distill_log_probs.argmax(-1)          # (B,T)
    a_null = teacher_null_distill_log_probs.argmax(-1)
    tail_idx = teacher_real_distill_log_probs.shape[-1] - 1
    flip = (a_real != a_null)
    if flip_tail_policy == "exclude":
        flip = flip & (a_real != tail_idx)
    flip = flip & loss_mask.bool()
    sharpened = F.log_softmax(flip_gamma * teacher_real_distill_log_probs, dim=-1)
    teacher_distill_log_probs = torch.where(flip.unsqueeze(-1), sharpened, teacher_real_distill_log_probs)
    position_weight = 1.0 + flip_lambda * flip.float()           # (B,T)
    counterfactual_target_tv_per_token = 0.5 * (teacher_distill_log_probs.exp()
                                                - teacher_real_distill_log_probs.exp()).abs().sum(-1)
else:
    ...  # 现有 tilt 路径逐 bit 不变
```

`position_weight` 乘到 `raw_per_token_loss`，并按样本内归一化：`agg_loss` 的分母改为 `Σ_t w_t·mask`（与 D 臂的"加权分母按比例缩放 `batch_num_tokens`"同一处理，见 `progress_report_2026-10-02.md` §3.4）。

### 4.3 指标（`self_distillation/flip_*`）

- `flip_frac`：翻转位置占有效 token 比例；`flip_frac_tail_excluded`：因 $a^+$ 落 tail 被排除的比例。
- `flip_pos_rel`：翻转位置的相对位置均值（预期集中在前 30–50%）。
- `flip_target_tv`、`nonflip_target_tv`（后者在 flip 模式下恒为 0，作为守卫）。
- `flip_student_agree`：翻转位置上学生 argmax 已等于 $a^+$ 的比例（训练中应上升）。

### 4.4 测试（`scripts/test_flip.py`）

- `counterfactual_target_mode=tilt` 时与当前 A 路径 bit-identical（复用 `test_sref.py` 的对拍框架）。
- 构造 (B,T,K+1) 玩具 log-prob：翻转集判定、tail 排除、$\gamma=1,\lambda=0$ 时等于 Vision-OPD、$\lambda$ 加权分母正确、padding 位不进翻转集。
- 变异体必须被抓：用 $u$ 的符号而非 argmax 判翻转；分母漏加权；tail 桶参与 argmax 但未排除。

## 5. 离线预检（先于训练，用现有 400 题探针 dump）

| 指标 | 预期 | 若不符 |
|---|---|---|
| F1 翻转集占比 | 5–20% | >40% 说明 mean_color null 过于"空"，几乎处处翻转，方法退化为 γ 臂 |
| F2 翻转位置的相对位置分布与词类（实词 / 功能词 / 答案字母） | 集中在推理正文的感知描述实词 | 若答案字母位占比高，与机制结论冲突，需复核 |
| F3 A 的目标 TV 中落在 $\mathcal F$ 外的份额 | ≥ 40%（即 A 把相当比例的偏移花在非翻转位置） | <15% 说明 A 本来就近似 CFD，P1 的"去副作用"空间小 |
| F4 $\mathcal F$ 外 A 倾斜最强的 token 类型 | 风格词（zoom / based / The）、OCR 文字 | 若是证据实词，则硬翻转会漏信号，改用软翻转 |

F3 是 go/no-go：份额 ≥40% 进训练。

## 6. 预测与 kill 条件

- **预测**：V\* 与 A 持平（argmax 信息完整），TB ≥ A（OCR / Spatial Containment 不再被误伤，缺 `<answer>` 率回到 V0 水平），目标 TV 低于 A（0.27）。
- **分类别预测**：相对 A，OCR 与关系推理 ≥ 0，Attributes 持平（不再有 sup 家族的 +5）。
- **Kill**：V\* 落到 V0 区间（<89）→ 非翻转位置的软倾斜承担了实质作用，argmax 假说被推翻；TB 低于 A 且 OCR 仍掉 → 误伤不来自非翻转位置的锐化。两种失败都是有判定力的结论，可进论文的机制节。

## 7. 与本季已有臂的关系

| 臂 | 与 CFD 的差别 |
|---|---|
| γ=50 | 所有位置锐化；CFD 只在 $\mathcal F$ 内 |
| floor | 谷区负 $u$ 截 0，仍是连续倾斜；CFD 是离散选择 |
| sup-only | 去正半边；CFD 保留 argmax 的正向信息 |
| D（未来加权） | $w_t$ 来自后缀冲突均值，均值 1.037；CFD 的 $w_t$ 来自翻转指示，恒有变化 |
| Ahm / X1 | 调 β 剂量；CFD 无 β |

## 8. 成本

与 A 相同（3 次前向）；loss 层增加两次 argmax 与一次 where，可忽略。
