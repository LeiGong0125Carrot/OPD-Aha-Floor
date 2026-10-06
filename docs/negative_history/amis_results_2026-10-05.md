# 臂 Amis：mismatched-crop visual null —— 设置、实现与结果（2026-10-05，滚动更新）

> 口径：准确率一律 gpt-oss-120b judge；TB 用 TreeVGR 官方 HF greedy，V\* 用 OPD-Aha 官方 `infer.py`。**判读规则（用户 10-05 20:45）：峰值取全部已保存 ckpt（5,10,…,50,51），不只 30/40/50**；历史臂（A/V0/…）只评过 30/40/50，与新臂比较时同时给两列。
> 代码：OPD-Aha-Floor `sup` 分支 `4250803`（实现）+ `5db52df`（审查跟进）。编排与评测脚本在仓库外 `Vision-OPD-setup/`（`orch_mis*.sbatch`、`trim_ckpt_peaks.sh`）。

## 1. 动机

- 今天的探针线（`ir_layer_prior_probe_plan_2026-10-05.md` §6、`full_to_priv_probe_2026-10-05.md`、`opd_aha_mechanism_conclusion_2026-10-05.md`）证明 null 前向不可替代，但 null 的**内容**可以换。
- A 的 u = log p⁺ − log p⁰ 混有两条通道：crop **内容**（证据）与"有没有一张真实 zoom 图在场"（风格：zoom/cropped 措辞、先解释再作答）。后者被认为是格式漂移与 OCR 丢分的来源。
- OPD-Aha 作者建议"从本 batch 的别的样本拿信号"；用户选定读法：**null 视图 = [全图 i, 同 batch 另一 prompt j 的 crop]**，教师照常再前向一次，其余完全在 OPD-Aha 框架内、与 A 同成本（3 次前向）。预期：风格通道在两次前向里抵消，证据通道保留。
- 预先记录的风险：crop j 给教师带来假证据（颜色题的 null 里看到别的红色物体），u 变噪。论文 Table 7 里均值色略优于不匹配图可能就是这个原因。

## 2. 设置

与 A（`pair_ahaAn2`：OPD-Aha + pair + n=2）**逐项相同**：Qwen3.5-4B，冻结初始教师，β=4，JSD α=0.5，学生 top-100 + tail，2459 题 6karmA pair 数据，seed 42，batch 48，n=2，lr 2e-6，51 步，每 5 步存 ckpt，3 卡 / SP1（A 的 rr1 配置）。唯一差别：

| | A | Amis |
|---|---|---|
| real 视图 | [全图 i, crop i] | 同 |
| null 视图 | [全图 i, **均值色块**（crop i 尺寸）] | [全图 i, **crop j**，j 为同 batch 另一 prompt，resize 到 crop i 的像素尺寸] |
| j 的选取 | — | 每步对 batch 内 48 个 prompt 做有种子的 Sattolo 错排（单环、无不动点；种子 = data.seed×1000003 + global_step）；同一 prompt 的两条 rollout 共用同一 j |

resize 是硬约束：dp_actor 的 null 前向复用 real 前向的 input_ids / position_ids，只替换图像输入，图像 token 数必须逐个相同（`ray_trainer` 对每个视图断言 prefix 张量相等）。代价：crop j 的长宽比被拉伸（审查指出的设计提醒；可选改进 = 先按 i 的长宽比居中裁剪再 resize）。

可选开关（本轮未开）：`MIS_SUP=1` 只压不提（`counterfactual_u_clip_pos`）；`MIS_VIEWS=2` 两个不同 j 的 null 在 log 空间取平均（u = log p⁺ − mean_j log p⁰_j）。

## 3. 实现（sup `4250803` + `5db52df`）

| 文件 | 改动 |
|---|---|
| `verl/utils/mismatch_null.py`（新） | `sattolo_derangement`、`pick_donor_indices`（按 uid 选 j，确定性，2 视图时第二个 j 用 perm1∘perm1 避免重复）、`mismatch_crop_images`（全图保留，crop 换成 resize 后的 j）、`average_null_log_probs` |
| `verl/trainer/ppo/ray_trainer.py` | `_maybe_build_self_distillation_batch`：`counterfactual_null_mode ∈ {mean_color, mismatch_crop}` 门控；运行时守卫（训练时 actor 配置是 DictConfig，`__post_init__` 不跑）：未知模式、num_views∉{1,2}、2 视图需 mismatch、mismatch 需 null_scope=last、缺 uid、不足 2/3 个 prompt、任一样本缺教师图；新静态方法 `_mismatch_crop_teacher_images`（含 `VOPD_FORBID_NULL` 守卫）；逐视图 prefix 断言；指标 `counterfactual_mismatch_crop_fraction` / `mismatch_unique_prompts` / `null_num_views`；manifest 记 null_mode/scope/views/u_clip_pos。legacy（mean_color/None）路径逐比特不变 |
| `verl/workers/actor/dp_actor.py` | 可选第二 null 前向（`teacher_null2_multi_modal_inputs`），all/topk/realized log-prob 三组取算术平均（= 概率几何平均，top-k 质量 ≤ 1 仍成立）；默认关 |
| `verl/workers/config/actor.py`、`actor.yaml` | 新模式值；`counterfactual_null_num_views: 1` |
| `scripts/train_pair_mis.sh` | A 配置 + `COUNTERFACTUAL_NULL_MODE=mismatch_crop`；tag `pair_Amis[r2]` / `pair_Amiss` / `pair_Amis2v` |
| `scripts/test_mismatch_null.py` | 52 项 CPU 单测（错排/确定性/同 prompt 共用 j/2 视图不重复；图像尺寸与内容；配置守卫；平均；trainer 静态方法；legacy 路径不变；core_algos 对模式无感），全过 |

审查（子 agent，commit `4250803`）：无必改项；两条应改（fallback 分支 donor 缺图、守卫措辞）已在 `5db52df` 修；设计提醒 = 长宽比拉伸。

## 4. 训练期（两条并行：r1 hold 20829689，r2 hold 20888750）

冒烟 2 步：mismatch 比例 1.0、每步 48 个不同 prompt、null 前向正常、prefix 断言未触发、目标 TV 0.29。

| | A | Amis r1 | Amis r2 |
|---|---|---|---|
| 目标 TV(q, p⁺) 均值 | 0.271 | 0.285 | 0.283 |
| 回答长度 均值 / 范围（token） | ~160 / 122–244 | 94 / 49–159 | 89 / 47–177 |
| grad_norm 均值 | — | 5.3 | 6.0 |
| 每步时间 | 130–150 s | 189 s | 171 s |

训练稳定（无 NaN、无塌缩；第 5–8 步和第 24–28 步两次长度下探后恢复）。**目标 TV 不低于 A 反而略高**——与"风格通道被抵消、u 变小"的预期相反，提示 crop j 引入的噪声量 ≥ 被抵消的风格量。

## 5. 结果（滚动）

### 5.1 TB（judge 口径）

| 臂 | step30 | step40 | step50 | 峰（30/40/50） | 全步峰 |
|---|---|---|---|---|---|
| A（4 次） | — | — | — | 49.88 / 51.11 / 51.11 / 51.60 | 未评 |
| V0（2 次） | — | — | — | 49.38 / 48.64 | 未评 |
| Amis r1 | 待 | 待 | 待 | 待 | 待（补评中） |
| Amis r2 | 47.16 | 47.16 | 47.90 | **47.90** | 待（补评中） |

### 5.2 V\*（judge 口径）

| 臂 | step30 | step40 | step50 | 峰（30/40/50） | 全步峰 |
|---|---|---|---|---|---|
| A（4 次） | — | — | — | 94.76 / 93.19 / 94.24 / 93.72 | 未评 |
| V0（2 次） | — | — | — | 87.4–88.5 | 未评 |
| Amis r1 | 待 | 待 | 待 | 待 | 待 |
| Amis r2 | 90.58（attr 89.6 / rel 92.1） | 90.05 | 87.96 | **90.58** | 待 |

r2 的 V\* 介于 V0（88）与 A（93–95）之间，末端（step50）回落到 V0 水平。

### 5.3 推理期格式（TB 输出：缺 `<answer>` 标签数 / 均长词数）

| 臂 | step30 | step40 | step50 |
|---|---|---|---|
| A | 25 / 198 | 21 / 176 | 17 / 165 |
| Amis r1 | 23 / 201 | 待 | 待 |
| Amis r2 | **59** / 160 | **82** / 147 | 待 |

缺标签回答的形态（关键差异）：

| 臂 | 缺标签回答均长 | 主要形态 |
|---|---|---|
| A step30 | 561 词 | 18/25 推理过长被截断；7 条 "Correct answer: **C. …**" |
| Amis r1 step30 | 323 词 | 15/23 **以裸选项收尾**（"C. Both the black shoes…"）；8 条截断 |
| Amis r2 step40 | **88 词** | 51/82 裸选项收尾（"A. The first one"）、24 条只剩一个字母 "C"、7 条截断 |

即 A 的漂移是"想太长写不到结尾"，Amis r2 的漂移方向相反：**回答变短，丢掉评测模板要求的 `<answer>` 包装，用训练 prompt 的作答格式（"X. 选项"）直接结束**。judge 能救回大部分裸选项（r2 规则分 30–37 → judge 47），所以这种漂移对 judge 口径伤害有限，但说明 mismatched null 没有"抵消风格通道"，而是引入了另一种风格偏移，且两条同配置同 seed 的 run 在推理行为上分叉（训练期机制指标几乎相同）。

## 6. 当前读法（待 r1 与 V\* 补齐后终审）

- r2：TB 峰 47.90（与 IRf 47.65、St 48.64 同档，低于 V0 和 A），V\* 峰 90.58（高于 V0 约 2.5，低于 A 约 3，且末端回落到 88）。null 对比的增益只保住了一部分。
- 机制预期（风格通道抵消）在三个可观测量上都没有兑现：目标 TV 不降反升；缺标签不减反增；漂移形态换了方向。最可能的解释是预登记的风险：crop j 的内容是假证据，u 里混入与 i 无关的内容和风格噪声，噪声量大于被抵消的"zoom 图在场"效应。
- 若 r1 同样 ≤ V0、V\* < 92，本线判负；剩余可试的两项（2 视图平均压噪声、按长宽比居中裁剪）只在 V\* 守住 A 而 TB 因格式丢分时才值得投。

## 附录 A：核心模块源码（`verl/utils/mismatch_null.py` @ `5db52df`）

```python
# Copyright 2026 Vision-OPD / OPD-Aha-Floor authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Arm Amis -- "mismatched-crop" visual null for OPD-Aha target reconstruction.

OPD-Aha (arm A) builds the null view as [full image, mean-colour block] (null_scope=last). The
log-ratio u = log p+ - log p0 then mixes two channels: the CONTENT of the crop (evidence) and the
mere PRESENCE of a real zoom image (style: "zoomed/cropped" wording, explain-before-answer, ...).
Here the null view is [full image, crop of ANOTHER prompt in the same batch]: a real zoom image is
present in both forwards, so the style channel cancels while the evidence channel (which only the
own crop carries) is kept. Everything else of the A target is unchanged (same beta, JSD, support).

Hard constraint (dp_actor): the null forward re-uses the real teacher's input_ids / position_ids
and swaps only the image inputs, so the donor crop MUST be resized to the own crop's pixel size
(identical `image_grid_thw`). ray_trainer additionally asserts prefix-tensor equality.

Pure-python helpers live here so they can be unit-tested on CPU without importing ray.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["sattolo_derangement", "pick_donor_indices", "mismatch_crop_images", "average_null_log_probs"]


def sattolo_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a permutation of range(n) with NO fixed point (single cycle; Sattolo's algorithm).
    n must be >= 2."""
    if n < 2:
        raise ValueError(f"mismatch_crop: need at least 2 distinct prompts in the batch, got {n}")
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))          # j in [0, i): never i itself -> single cycle, no fixed point
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def pick_donor_indices(uids, seed: int, step: int, num_views: int = 1) -> list[np.ndarray]:
    """For every sample i (uids[i] = prompt id shared by its n rollouts) choose `num_views` donor
    sample indices whose prompt differs from uids[i]. All rollouts of one prompt get the same donors.
    Deterministic in (seed, step). Returns a list of `num_views` int arrays of shape [batch].
    With num_views == 2 the second donor is also different from the first."""
    uids = list(uids)
    uniq = list(dict.fromkeys(uids))                      # first-occurrence order
    n = len(uniq)
    if num_views not in (1, 2):
        raise ValueError(f"mismatch_crop: num_views must be 1 or 2, got {num_views}")
    if n < 2 or (num_views == 2 and n < 3):
        raise ValueError(
            f"mismatch_crop: need >= {2 if num_views == 1 else 3} distinct prompts for num_views={num_views}, got {n}"
        )
    first_idx = {}
    for i, u in enumerate(uids):
        first_idx.setdefault(u, i)
    pos = {u: k for k, u in enumerate(uniq)}
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(step))
    perm1 = sattolo_derangement(n, rng)
    perms = [perm1]
    if num_views == 2:
        perm2 = sattolo_derangement(n, rng)               # independent draw from the same stream
        # if a prompt got the same donor twice, move the second donor one step along perm1's cycle
        # (perm1[perm1[k]] != k for n >= 3 since perm1 is a single n-cycle, and != perm1[k])
        same = perm2 == perm1
        perm2 = np.where(same, perm1[perm1], perm2)
        perms.append(perm2)
    out = []
    for perm in perms:
        donors = np.array([first_idx[uniq[perm[pos[u]]]] for u in uids], dtype=np.int64)
        out.append(donors)
    for donors in out:
        for i, d in enumerate(donors):
            if uids[int(d)] == uids[i]:
                raise RuntimeError("mismatch_crop: donor has the same prompt as the sample (derangement bug)")
    if num_views == 2:
        for i in range(len(uids)):
            if uids[int(out[0][i])] == uids[int(out[1][i])]:
                raise RuntimeError("mismatch_crop: the two donors share a prompt")
    return out


def mismatch_crop_images(teacher_images: list, donor_crop, normalize: Callable, resample=None) -> list:
    """null_scope=last with a mismatched donor: keep every image but the last (normalised, as the
    mean-colour path does); replace the last by the donor crop resized to the own crop's exact size."""
    if len(teacher_images) < 1:
        raise ValueError("mismatch_crop: teacher_images is empty")
    from PIL import Image
    own = normalize(teacher_images[-1])
    donor = normalize(donor_crop)
    if donor is own or (
        isinstance(donor_crop, dict) and isinstance(teacher_images[-1], dict)
        and donor_crop.get("path") is not None and donor_crop.get("path") == teacher_images[-1].get("path")
    ):
        raise RuntimeError("mismatch_crop: donor crop is the sample's own crop")
    if resample is None:
        resample = Image.BICUBIC
    out = [normalize(img) for img in teacher_images[:-1]]
    out.append(donor.resize(own.size, resample) if donor.size != own.size else donor.copy())
    if out[-1].size != own.size:
        raise RuntimeError("mismatch_crop: resized donor size mismatch")
    return out


def average_null_log_probs(a, b):
    """Two-view null (num_views=2): geometric mean in probability space = arithmetic mean of log-probs,
    consistent with u = log p+ - mean_j log p0_j. For top-k log-probs the mass stays <= 1 (geometric
    mean <= arithmetic mean of two valid sub-distributions). `None` passes through if both are None."""
    if a is None and b is None:
        return None
    if a is None or b is None:
        raise ValueError("average_null_log_probs: both views must be present")
    if a.shape != b.shape:
        raise ValueError(f"average_null_log_probs: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    return 0.5 * (a + b)
```

## 附录 B：ray_trainer 的 null 分支（节选，`5db52df`）

```python
_build_null_view = (counterfactual_null_mode in ("mean_color", "mismatch_crop")
                    and str(self_distillation_cfg.get("counterfactual_reference", "null") or "null") == "null")
_mismatch = _build_null_view and counterfactual_null_mode == "mismatch_crop"
...
if _mismatch:
    donor_idx_views = pick_donor_indices(list(batch.non_tensor_batch["uid"]),
                                         seed=int(self.config.data.get("seed", 0) or 0),
                                         step=int(getattr(self, "global_steps", 0) or 0),
                                         num_views=null_num_views)
...
for view_k in range(null_num_views):
    if _mismatch:
        donor_i = int(donor_idx_views[view_k][i])
        donor_images = batch.non_tensor_batch[teacher_image_key][donor_i]
        teacher_null_images = self._mismatch_crop_teacher_images(teacher_images, donor_images[-1])
    else:
        teacher_null_images = self._mean_color_teacher_images(teacher_images, null_scope=...)
    ...  # same messages / prompt inputs as the real view; assert input_ids/attention_mask/position_ids/response_start_idx equal
```
