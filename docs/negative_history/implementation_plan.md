# Negative-History OPD — Implementation Plan

对应设计文档 `docs/negative_history/02–07`（branch `negative_history`, head 33b6f11）。
本文是实现侧的计划：怎么接进现有 OPD-Aha-sup 管线、预登记判据、测试与跑法。

---

## 一、臂 → 现有基础的映射

| 设计臂 | 公式 | 现有对应 | 还缺什么 |
|---|---|---|---|
| A | `q ∝ p⁺·e^{βu}` | **已有 ×4**（49.88/51.11/51.11/51.60） | 无 |
| B | `q ∝ p⁺·e^{β·min(u,0)}` | **= supn2/supn2r2，已有 ×2**（50.37/47.41）<br>（§03 的 `exp(−β[−u]₊)` ≡ 现 `u_clip_pos` 门，逐式相同） | **r3**（两次差 2.96 超噪声带，B 是 2×2 的基角，必须稳住） |
| C | B + `β_t = β(1+s_t)` | 新 | 实现 + ×2 + **misalign 控制**（见 §四） |
| D | B + `w_t = 1+r_t` 加权 JSD | 新 | 实现 + ×2 |
| E | B + 两者 | 新 | 实现 + ×2（建议 C/D 出结果后再跑，见 §六） |

---

## 二、代码改动（全部门控，默认关闭 = 与现行为 bit-identical）

### 1. dp_actor.py —— 一行

`teacher_null_outputs["log_probs"]`（null 视图 realized-token log-prob, [B,T]）已经由
`_forward_micro_batch` 算出，只是没被取用。提取并传给 loss：

```python
teacher_null_log_prob = teacher_null_outputs["log_probs"]   # 新增提取
...
compute_self_distillation_loss(..., teacher_null_log_probs=teacher_null_log_prob)
```

**为什么用 realized 路径而不是在 top-k support 上 gather**：采样 token 不保证落在
student top-100 里（tail 质量中位 1e-4 但 p90≈0.04），support gather 会有缺失位需要
fallback；realized log-prob 是全词表归一化的精确值，无此问题，且与 `teacher_log_probs`
（real 侧 realized，已存在）天然同对齐、同 shift——**02 §9/§10 的对齐契约自动满足**，
不引入任何新的对齐风险。

### 2. core_algos.py —— 轨迹状态分支（全部 `no_grad`）

```python
# c_t = [−u_t(y_t)]₊   (02 §6; 用 realized 量, [B,T])
u_realized = teacher_log_probs - teacher_null_log_probs
c = torch.relu(-u_realized) * loss_mask                    # padding 置 0 (02 §11)

# 历史 (04): N_t = Σ_{k<t} c_k  **exclusive**; s = N/(1+N); β_t = β(1+α_H·s)
N = torch.cumsum(c, dim=1) - c                             # exclusive cumsum
s = N / (1.0 + N)
beta_t = beta * (1.0 + alpha_H * s)                        # [B,T], α_H=1 固定

# 未来 (05): 后缀均值 → r → w;  K_t=0 ⇒ w=1 (05 §6)
K  = (loss_mask.cumsum(1).flip? ...)  # 反向 exclusive 计数
F  = (c 的反向 exclusive cumsum)
Fbar = F / K.clamp(min=1); r = Fbar/(1+Fbar)
w  = 1.0 + alpha_F * r                                     # [1,2), α_F=1 固定
```

应用点：
- **C**：tilt 一行从 `beta * u_term` 变为 `beta_t.unsqueeze(-1) * u_term`（仅当门开）。
- **D**：加权聚合。**不必改 agg_loss**——把 `loss_mask * w` 作为 mask 传入现有
  `agg_loss(token-mean)`，其默认分母 `mask.sum()` 恰好给出 05 §14 的
  `Σ M w ℓ / Σ M w`（分母同权，不改有效 lr）。诊断指标仍用未加权 mask，保持跨臂可比。

### 3. 配置门（workers/config/actor.py + actor.yaml）

```yaml
counterfactual_hist_adaptive_beta: false    # C/E
counterfactual_hist_alpha: 1.0
counterfactual_hist_shuffle: false          # misalign 控制 (只配合 hist 用)
counterfactual_future_weight: false         # D/E
counterfactual_future_alpha: 1.0
```

**运行时守卫写在 loss 函数里**（dataclass 校验在本仓训练路径上是死代码，γ 臂已证）：
三个门都要求 `u_clip_pos=True`（设计以 negative-only 为底座）+ `null_mode` 存在 +
与 `st_enable`/`floor`/`tanh_scale`/`target_gamma≠1` 互斥（单变量纪律）。

### 4. launcher `scripts/train_pair_neghist.sh`

从 `train_pair_sup.sh` 派生，环境变量 `HIST=1/0 FUT=1/0 HIST_SHUFFLE=1/0`，
实验名自动带臂标识（`pair_nhC_6karmA` 等——γ 臂学费：名字必须含参数，否则
resume_mode=auto 会把不同臂续到同一 checkpoint 目录）。默认 **3卡/SP1, n2, lr2e-6**。

---

## 三、新增指标（全部无条件上报，供跨臂对照）

| 指标 | 定义 | 用途 |
|---|---|---|
| `nh/c_mean`, `nh/c_frac_pos` | 实现冲突均值 / c>0 占比 | c_t 的量级与稀疏度（探针曾测 o_t 的 p50=0——若 c 极稀疏,历史状态无信息量,臂的前提直接可读） |
| `nh/N_last_mean` | 轨迹末端累积 N_T | 历史状态动态范围 |
| `nh/s_mean`, `nh/beta_t_mean` | 有效抑制强度 | C 的机制读数（β_t 是否真的分化了） |
| `nh/w_mean`, `nh/w_p90` | 位置权重分布 | D 的机制读数（w 是否真的在重分配） |

机制读数是 γ 臂的成功经验：**"机制是否发生"与"是否有用"分开读**，失败也能定位原因。

---

## 四、misalign 控制（C-temporal 的学费，预登记）

C-temporal（h=16 时间信用）判死的方式：**misalign 控制臂 ≥ 主臂**——增益来自扰动幅度
而非时间对齐。Arm C 的 `s_t` 同样是"历史调制强度"，必须预登记同款检验：

> `hist_shuffle=True`：对每条轨迹把 `s_t` 沿时间维随机重排（保边际分布，毁时间对齐）。
> **若 C 两次超阈值而 C-shuffle 不低于 C，则 C 的增益判为幅度效应，臂判负。**

触发条件：只有 C 的两次峰值**都 >51.60** 才跑 C-shuffle（×1 即可，它是证伪器不是臂）。

---

## 五、预登记判据（口径：{30,40,50} 峰值，×2 两次同侧；锚 = A 四次 49.88–51.60）

| 对比（07 §9） | 读法 |
|---|---|
| C−B / D−B / E−B | 主对比。但 **B 的两次差 2.96 超噪声带**，先补 B-r3；因子对比用 **B 的三次中位数**做基线 |
| 任一臂两次 >51.60 | 有增益 → C 还需过 misalign 控制 |
| 两次 <49.88 | 判负 |
| 区间内 | 不可判读（与本季 8 个改造臂同档） |
| 交互 Δ_H×F (07 §10) | 仅当 C/D 至少一个有信号才值得读 E |

**诚实前置**（写在结果前面）：本季与 C/D 相近的先验不利——TGNVMT 门控判负
（但其 EMA/margin-mask 正是 04 刻意移除的两个部件）、轨迹 reweight 探针测得
ρ(1)=0.151 弱持续、位置富集仅 1.3×。有利证据：R_C≈0.62 是轨迹统计里唯一有判别力的
量，而 c_t 正是该族；答案声明后信号断崖 0.20× 与 w_t 的加权方向一致。

**已知局限**（02 §12）：v1 不区分 finish_reason，截断轨迹的后缀统计照常参与 w_t。
依据：max_response_length=1024 而 T 的 p90=230，截断率低；上报截断率指标，若 >2% 再修。

---

## 六、测试与跑法

**scripts/test_neghist.py**（tanh/γ 的对拍纪律）：独立参照实现（numpy 逐式照抄文档公式）
→ 与生产 loss 数值对拍 <1e-5；**变异体必须被抓住**：① N 用 inclusive cumsum（02 §4 要求
exclusive）② w 的分母漏加权（05 §14）③ c 不乘 mask ④ future 含当前位。另：全门关 ≡ sup
bit-identical；变长 mask 下 N/F 的手算用例；`c_t` 与 02 §7 工作例逐数对拍。

**跑法分两段**（新 3 卡 hold，n2 约 3.3h/次）：

| 段 | 内容 | 预算 |
|---|---|---|
| 1 | **B-r3**（supn2r3）+ **C ×2** + **D ×2** | 5 跑 ≈ 17h + 评测 |
| 2 | 视段 1：C/D 有信号 → **E ×2**（+ C-shuffle ×1 若 C 超阈值）；全无信号 → E 不跑（07 的交互项无意义），线收账 | 0–10h |

实现顺序：dp_actor 一行 → core_algos 轨迹分支 + 门 + 指标 → launcher → 测试全绿 →
**code-review（流程规则）** → 提交推送 → 申 3 卡 hold → sbatch 编排器（单编排器独占全部
tags；cleaner 动态扫描本批臂）。
