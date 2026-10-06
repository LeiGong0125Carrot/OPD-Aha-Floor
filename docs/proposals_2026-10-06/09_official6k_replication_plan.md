# 09 官方 Vision-OPD-6K 全量复现：Ahm / Ahf（2026-10-06 立项；A 不自训，直接评公开 ckpt）

> 10-06 下午更新（用户）：OPD-Aha 在 6K 上的训练原文已经做过，不再自训 A，直接评测公开 ckpt `CewEhao/OPD-Aha-4B`（同一条评测流水线：ZoomBench / HR-Bench 4K / 8K / V\*，rule-first 与 pure-llm 双口径）。A 的 hold 与编排器已取消；下文的 A 配置保留作为 Ahm/Ahf 的配方参照。

## 0. 为什么做

- 08 号文档的 ZoomBench 批次（我们自己的 2459 题 pair 数据、51 步）：A 峰 60.24、Ahm 峰 61.30、Ahf 峰 60.12，而 OPD-Aha 论文在 ZoomBench 上报的数字明显更高（用户 10-06："这对比的结果说实话，低于 opd-aha 文中呈现的结果"）。
- 论文用的是本仓库的评测代码（用户确认），所以差距只能来自**训练数据/配方**（6241 题官方数据 vs 我们 2459 题高清 pair 子集；单张 zoom crop 教师图 + `null_scope=all` vs 我们的 [全图, crop] pair + `null_scope=last`；96×n8×70 步 vs 48×n2×51 步）或**评测口径**（rule-first vs pure-llm；官方 `CewEhao/OPD-Aha-4B` 已排进同一条 ZoomBench 流水线做对照）。
- 目标：在官方数据与配方下各训练一次 A、Ahm、Ahf，拿到（a）我们复现的 A 与论文/官方 ckpt 的差距，（b）Ahm/Ahf 相对 A 在 845 题 ZoomBench 上是否仍有 1–1.5 点的优势。

## 1. 训练配置（`scripts/train_official6k.sh`）

| 项 | 论文/官方 `train_beta4.sh` | 本次（3×RTX Pro 6000，driver 1T） |
|---|---|---|
| 数据 | `data/train.parquet` 6241 题；`images`=带红框学生图，`bbox_images`=单张 zoom crop 教师图；学生提示含 "Only focus on the objects inside the red bounding box" | 同 |
| null | `mean_color`，`null_scope` 默认 `all`（唯一一张教师图换成同尺寸均值色块） | 同 |
| β / α / top-K | 4 / 0.5 / 100+tail | 同 |
| lr / max_prompt / max_resp / seed | 2e-6 / 8192 / 1024 / 42 | 同 |
| batch × n × 卡 | 96 × 8 × 8 卡（2 节点） | **48 × 8 × 3 卡**（96×n8 会撑爆 driver CPU 内存，见 EVT-neg 记录） |
| 步数 | 70（= 6720 样本 ≈ 1.08 epoch） | **130**（= 6240 样本 ≈ 1.0 epoch） |
| 保存 | 每 10 步 | 每 10 步（13 个 ckpt，全部评测取峰） |
| Ahm | — | `hist_adaptive_beta=True, hist_mode=mean, kappa=0.10, half=neg` |
| Ahf | — | 同上但 `hist_mode=hf` |

未改动的部分：冻结初始教师、JSD、学生 top-K+tail、rollout 温度等全部走 `run_visual_counterfactual_unit.sh` 默认值。各臂各跑一次（用户 10-06："训练一次"）。

## 2. 评测（`Vision-OPD-setup/eval_off6k.sh`）

- 基准：**ZoomBench（845）与 V\*（191）**。**TreeBench 不测**（用户 10-06：TB 只留给 CFD / CFD-u 收尾）。
- 代码：全部用 OPD-Aha 的 `eval/infer.py`（vLLM serve，perception chat template，max_tokens 4096，temperature 0）+ `eval/judge_qwenlm.py` + `cal_acc.py`。
- Judge：gpt-oss-120b，两种 protocol 分目录保存：`eval/judge/rule-first/<bench>/` 与 `eval/judge/pure-llm/<bench>/`。注意 ZoomBench 不在 `MCQ_BENCHMARKS` 里，rule-first 对它只做 mathruler 精确匹配后全部交 LLM，所以两口径在 ZoomBench 上预期接近；V\* 的 rule-first 走首字母规则。
- 全部 13 个 ckpt 都测，峰值各报各的；评测完按 V\* 峰 + ZoomBench 峰裁剪（`trim_off6k.sh`）。
- `orch_zoom3` 顺带给 08 号文档里的 14 个旧 ckpt（含 base、官方 OPD-Aha-4B）补 V\* 推理与 pure-llm 判分，使新旧表同口径。

## 3. 执行

- hold：20927453（Ahm）、20927868（Ahf），各 3 卡 / 1T / 1d15h；编排器 `orch_off6k{Ahm,Ahf}.sbatch`（standard 分区 job 20928147/48）；官方 ckpt 与旧臂名单的双口径判卷走 `orch_zoom3.sbatch`（1 卡 hold 20927219）：冒烟 2 步（检查 null 前向、hist 开关）→ 训练（`resume_mode=auto`，hold 到期可重投续跑）→ 评测 → 裁剪 → 释放 hold。
- 预计：训练 ≈ 27–31 h（n8 单步 ≈ 750–850 s），评测 ≈ 3 h/臂。
- 判读：分基准报告，不合并；Ahm/Ahf vs A 用逐题 paired bootstrap（同 08 §2）。
