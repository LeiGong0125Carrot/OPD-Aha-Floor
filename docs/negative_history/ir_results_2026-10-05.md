# 路线一 IR（单次冻结教师前向的内部残差重构）实验记录（2026-10-05，滚动更新）

> 设计：`teacher_internal_residual_reconstruction_plan.md`（v0.1，`cc115a4`）。实现：OPD-Aha-Floor:sup `98ee0b9`（+ `d379de1` tag 编码）。
> 口径：gpt-oss-120b judge；TB 用 TreeVGR 官方 HF greedy，V\* 用 OPD-Aha 官方 `infer.py`；`{30,40,50}` 峰值。
> 用户 10-05 00:45 授权：对本线自主调 λ / 层区间 / tail，目标 **TB > 50 且 V\* 92–93**。先单次探索，有希望的配置再补第二次。
> 固定：教师看 pair 特权视图 [全图, crop]（与 A 同），学生看全图；无 null 前向（`VOPD_FORBID_NULL=1`）；JSD α=0.5、学生 top-100+tail、冻结教师、2459 题 6karmA、seed 42、3 卡/SP1、n2、lr 2e-6、51 步。
> 参照：A TB 峰 49.88–51.60 / V\* 91.6–94.8（池化 49.22 / 92.84）；V0（无 tilt）TB 49.38 / 48.64，V\* 87.4–88.5。

## 0. 工程验收（10-05 01:01，冒烟 2 步，`VOPD_IR_VERIFY=1`）

- 三张卡均报 `capture on/off teacher log_probs identical=True, topk_logps identical=True`，四个 hook 各触发 1 次：只读采集不改变 anchor。
- `teacher_target_mode_internal=1`、`teacher_null_forward_frac=0`；teacher_forward 22.5 s（A 22–26 s，额外的分块头部读出可忽略）；峰值显存 79.6 GB（A 84.5 GB）；首步 150 s 含预热。
- legacy 路径与 `187a952` 在 A/sup/S/普通 OPD 配置下 loss / 学生梯度 / 指标 bit-identical（`scripts/test_internal_residual.py`）。

## 1. 结果表（滚动）

| 臂 | [a,b) | λ | tail | TB 30/40/50 | TB 峰 | V\* 30/40/50 | V\* 峰 | 51 步均值：target_tv / r_abs / top1 / H / tail_mass | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| **IRf** | [12,20) | 1.0 | full | 47.65 / 46.17 / 47.41 | 47.65 | 83.77 / 85.86 / 87.43 | 87.43 | 0.088 / 0.75 / 0.82 / 0.53 / 0.003 | ✗；V\* 在 V0 区间下沿；目标离 p⁺ 太近（A 的 null tilt TV ≈ 0.27） |
| IRfl4 | [12,20) | 4.0 | full | 训练中（05:04 开） | | | | | 预计 09:40 出分 |
| IRfl2 | [12,20) | 2.0 | full | 排队（IRfl4 后） | | | | | |
| IRal4 | [12,20) | 4.0 | aha | 等第三张 hold | | | | | |
| IRa | [12,20) | 1.0 | aha | 等第三张 hold（IRal4 后） | | | | | |

## 2. 读法（随结果更新）

- **IRf λ=1**：训练全程稳定（无塌缩，回答长度 94–160，grad_norm 收敛到 1–2），但目标几乎就是 p⁺（TV 0.09），结果与普通特权 OPD（V0）同一水平。两种可能：(a) λ=1 剂量太小——用 λ=4、2 检验；(b) [12,20) 的贡献方向本身不携带 crop 信息——若 λ↑ 后 V\* 仍不动而 TB 下降，则指向 (b)，下一步换区间（后段 [20,28) 或含更多 full-attention 层的区间）。
- 训练期第 5–7 步的回答长度下探（34–77）与 Ahf 同阶段相同（warmup 瞬态），第 8 步起恢复，不是 S2 式塌缩。

## 3. 决策树（预登记）

| 观察 | 动作 |
|---|---|
| λ=4：V\* ≥ 90 且 TB ≥ 49 | 继续 λ=4 第二次 + IRa λ=4 对照 tail |
| λ=4：V\* 仍 < 89、target_tv 仍 < 0.15 | 剂量不是瓶颈 → 换区间 [20,28)（含 23、27）或 [8,24) |
| λ=4：回答长度 < 20 或 TB < 45 | 过冲 → 回 λ=2 结果定夺 |
| 两种 tail 相近 | 只续 full |
