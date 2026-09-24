# ST 算法档案：selective suppression + conserved mass transfer

**结论先行：判负。** 双基准同向低于主对照 supn8 约 2.6 点、且**末端单调下滑**（趋势与
supn8 相反）、逐题翻转不对称 2:7。**最大产出不是这个负结果本身，而是它暴露的方法论教训：
TV 标量剂量匹配 ≠ 干预结构匹配**，以及由此得到的机制推论 —— **视觉对比的抑制作用需要
弥散地作用在分布上，不能压缩为少数 token 的精确搬运。**

代码：`verl/trainer/ppo/core_algos.py`，gate `counterfactual_st_enable`（commit `84d51ea`）。
基线 `483d70f`。写作日期 2026-09-24。

---

## 一、算法定义

ST **完全替换**指数 tilt，改为概率空间内的有界扣减 + 组内等比守恒转移。

记 a = p⁺（特权 teacher 分布），s = 学生分布（detach），u = log p⁺ − log p⁰，
全部在学生 top-100 + tail 支持上。定义相对头部与显著负向：

$$\mathcal H(r;\rho) = \{v : r(v) \ge \rho \cdot \max_w r(w)\}, \qquad
\mathcal N = \{v : u(v) < -\tau\}$$

**三个集合**（tail 桶永不入 D/R）：

$$\boxed{\;\mathcal D = \mathcal N \cap \mathcal H(a;\rho) \cap \mathcal H(s;\rho)\;}
\qquad \mathcal R = \{v : u(v) \ge -\tau\} \qquad
\mathcal F = \mathcal V \setminus (\mathcal D \cup \mathcal R)$$

- **𝒟（donor，被扣）**：师生**都**把它放在相对头部，但真实视觉输入相对 null 在降低它 ——
  "共同的错误惯性，且视觉反对"。
- **ℛ（recipient，接收）**：视觉不显著反对的 token（含视觉支持与近中性）。
- **𝓕（不动）**：**视觉反对但低概率**的 token（Wait 类替代入口）+ tail 桶。

**转移量**（两道上限）：

$$M_D = \sum_{v\in\mathcal D} a(v), \quad M_R = \sum_{v\in\mathcal R} a(v), \qquad
\boxed{\;\delta = \min(\epsilon,\ \alpha_{\max} M_D)\;}$$

若 𝒟 空或 M_R=0 则 δ=0、q=a（零转移守卫）。**目标**：

$$q(v) = \begin{cases}
a(v)\left(1 - \dfrac{\delta}{M_D}\right), & v \in \mathcal D \\[8pt]
a(v)\left(1 + \dfrac{\delta}{M_R}\right), & v \in \mathcal R \\[8pt]
a(v), & v \in \mathcal F
\end{cases}$$

**无全局 softmax** —— 扣减与增加已经守恒。

超参：τ=0.10（负向门槛）、ρ=0.10（头部相对阈值）、**α_max=0.75**（剂量校准后，见第五节）、
ε=0.30（绝对上限，实测从未触顶）。

### 可机器验证的性质（全部在单测中断言）

| 性质 | 表达式 |
|---|---|
| 质量守恒 | $\sum_v q(v) = \sum_v a(v) = 1$ |
| donor 确实下降且有 floor | $(1-\alpha_{\max})a(v) \le q(v) \le a(v),\ v\in\mathcal D$ |
| recipient 内比例不变 | $q(v)/q(w) = a(v)/a(w),\ \forall v,w \in \mathcal R$ |
| **干预幅度精确可控** | $\mathrm{TV}(q,a) = \tfrac12\sum_v\lvert q-a\rvert = \delta$ |
| 最小结构注入 | 组内等比缩放是"固定组间转移量下 $\min D_{\mathrm{KL}}(q\Vert a)$"的解 |

最后一条给了"不额外改变组内结构"一个干净的最优性表述。

---

## 二、代码

```python
u_term = teacher_real_distill_log_probs - teacher_null_distill_log_probs
if counterfactual_st_enable:
    with torch.no_grad():
        a_prob = teacher_real_distill_log_probs.exp()
        s_prob = student_distill_log_probs.detach().exp()
        explicit = torch.ones_like(a_prob, dtype=torch.bool)
        if use_topk and self_distillation_config.distillation_add_tail:
            explicit[..., -1] = False                     # tail 永不入 D/R
        a_exp = a_prob.masked_fill(~explicit, 0.0)
        s_exp = s_prob.masked_fill(~explicit, 0.0)
        head_a = a_exp >= rho * a_exp.amax(dim=-1, keepdim=True)
        head_s = s_exp >= rho * s_exp.amax(dim=-1, keepdim=True)
        donors     = (u_term < -tau) & head_a & head_s & explicit
        recipients = (u_term >= -tau) & explicit
        mass_d = (a_prob * donors).sum(-1);  mass_r = (a_prob * recipients).sum(-1)
        valid  = (mass_d > 0) & (mass_r > 0)
        delta  = torch.minimum(full_like(mass_d, eps), alpha_max * mass_d) * valid.float()
        scale_d = (1 - delta / mass_d.clamp_min(1e-12)).unsqueeze(-1)
        scale_r = (1 + delta / mass_r.clamp_min(1e-12)).unsqueeze(-1)
        q_st = a_prob * torch.where(donors, scale_d,
                     torch.where(recipients, scale_r, torch.ones_like(a_prob)))
        q_st = q_st / q_st.sum(-1, keepdim=True).clamp_min(1e-12)   # 浮点卫生
        teacher_distill_log_probs = (q_st + 1e-12).log()
```

**实现要点**

1. **全程 `no_grad`**，且 s 显式 `.detach()` —— selector 用学生分布但梯度不回流，
   student 侧梯度只经 JSD。
2. **头部 max 只在 explicit 列上取**（tail 遮零）—— tail 质量大的位置头部集会偏宽，
   属设计选择，`st_donor_tokens_frac` 可监控。
3. 末尾再归一化一次是浮点卫生（数学上已守恒）。
4. **与 `counterfactual_u_clip_pos` / `counterfactual_floor_alpha` 三者互斥**
   （每个都是单变量消融，叠加会毁掉归因）。

**仪表盘**（5 个）：`st_delta_tv`（实际 δ）、`st_donor_coverage`（δ>0 的位置占比）、
`st_donor_tokens_frac`、`st_mass_donor`、`st_mass_recipient`。单测断言
`st_delta_tv == counterfactual_target_tv`（现有 TV 指标在 ST 下精确等于 δ，跨臂可比）。

**单测**：`scripts/test_st.py`，七组 —— off 逐位一致、空 donor 守卫、**提案数值例作
golden case**（torch 生产路径精确复现 δ=0.30）、守恒 + TV=δ、floor + 比例不变、
仪表盘、config 互斥校验。

---

## 三、详细举例：四臂在同一张表上

同一个人造六 token 例（β=4，s=a）：

| token | a=p⁺ | u | 判定（ST） | A | floor | supn8 | **ST** |
|---|---|---|---|---|---|---|---|
| Therefore | 0.600 | −0.154 | **donor**（头部∧负向） | 0.00089 | 0.00089 | 0.5416 | **0.360** ↓ |
| Thus | 0.150 | −0.288 | **donor** | 0.00013 | 0.00013 | 0.0792 | **0.090** ↓ |
| Wait | 0.040 | −0.118 | 𝓕（负向但**非头部**） | 0.00007 | 0.00011 | 0.0417 | **0.040** = |
| Actually | 0.030 | +0.693 | recipient | 0.00132 | 0.00132 | 0.0501 | **0.075** ↑ |
| Indeed | 0.010 | −0.405 | 𝓕（非头部） | 0.00001 | 0.00003 | 0.0033 | **0.010** = |
| However | 0.170 | +1.917 | recipient | **0.99759** | 0.99753 | 0.2841 | **0.425** ↑ |
| | | TV(·,a) | | 0.828 | 0.828 | 0.136 | **0.300** |

计算：头部阈值 = 0.1×0.600 = 0.06 → 头部 = {Therefore, Thus, However}；
𝒟 = {Therefore, Thus}（However 的 u>0 不是负向），M_D = 0.750；
ℛ = {Actually, However}，M_R = 0.200；δ = min(0.30, 0.5×0.750) = 0.300
（此例用 α_max=0.5 复现提案原表；生产用 0.75）。
donor 统一乘 1−0.30/0.75 = 0.60；recipient 统一乘 1+0.30/0.20 = **2.50**。

### 三个关键行为（与另外三臂对比）

**① `Wait`（低概率反思入口）被保护。** A 和 supn8 都因 u<0 压它（supn8 因归一化陷阱实际压不动，
但机制上是压）；floor 也保护它（谷区截 0）；**ST 完全不动它**（既不压，也不参与再分配）。

**② 替代续写获得竞争优势，但不按 u 排序。** Actually（u=0.69）和 However（u=1.92）都乘同一个
2.50 —— 与 supn8 的"等比接收"同构。对比 A：β=4 让 However 独吞 99.76%，是彻底的按 u 重排。

**③ 相对优势 ≠ 绝对概率（提案中强调过，此处验证）**：

$$\frac{a(\text{Wait})}{a(\text{Therefore})} = \frac{0.040}{0.600} = 0.0667
\;\longrightarrow\;
\frac{q(\text{Wait})}{q(\text{Therefore})} = \frac{0.040}{0.360} = 0.1111$$

相对 odds 提升 1.67×，但**该位置抽到 `Wait` 的绝对概率仍是 0.040**（丝毫未变）。
不能因相对优势改善就宣称"reflection 采样概率提高"。

---

## 四、训练配置

| 项 | 值 |
|---|---|
| 实验名 | `pair_stn8_6karmA`（短名 stn8） |
| 数据 / 视图 | `train_6karmA_pair.parquet` / pair，`null_scope=last`（与 supn8 完全相同） |
| 超参 | τ=0.10、ρ=0.10、**α_max=0.75**、ε=0.30 |
| batch × rollout | 48 × n8（与 supn8 相同） |
| GPU / SP | 3× RTXPro6000 / SP=1 |
| 步数 / seed | 51（1 epoch）/ 42 |
| launcher | `scripts/train_pair_st.sh` |
| 主对照 | **supn8**（同体制、同数据、同 seed，唯一差异 = 目标构造） |

---

## 五、剂量校准过程（三次冒烟）

因为"剂量混杂"是本季反复踩的坑（misalign 的 +0.5 被证明是扰动幅度效应），开训前先用
2 步冒烟把 δ 调到与 supn8 的实测 TV 对齐：

| 冒烟 | 配置 | δ (`st_delta_tv`) | coverage | 判断 |
|---|---|---|---|---|
| 1 | 默认（α_max=0.50） | 0.045 | 0.27 | 欠剂量（supn8 是 0.065） |
| 2 | ρ 0.10→**0.05** | 0.048 | 0.33 | 边际改善，放弃该旋钮 |
| 3 | **α_max 0.50→0.75** | **0.059** | 0.29 | **落入匹配带（±10%）→ 采用** |

ε=0.30 从未触顶（M_D 均值 ~0.34，α_max·M_D 先到），故真旋钮只有 α_max。

正式训练中 δ 全程 0.056-0.070（与 supn8 的 0.063-0.090 重叠），coverage 稳定 ~0.29，
内存平台 ~600G。**当时的结论是"剂量已对齐，唯一变量是结构" —— 第九节说明这个推断错了。**

---

## 六、评测结果（judge 口径；**规则分不入表**）

| step | stn8 V\* | supn8 V\* | A(48×n2) V\* | stn8 TB | supn8 TB | A TB |
|---|---|---|---|---|---|---|
| 30 | 90.05 | 86.91 | 92.15 | 45.93 | 47.65 | 48.64 |
| 40 | 89.53 | 89.53 | 91.10 | 46.91 | 48.40 | 47.65 |
| 50 | **89.01** | **91.62** | 91.62 | **43.95** | 48.64 | 49.38 |
| 均值 | 89.5 | 89.4 | 91.6 | **45.6** | 48.2 | 48.6 |

---

## 七、判负：四条独立证据

1. **末端双基准同向低**：@50 V\* −2.6（89.01 vs 91.62；V\* 稳定性 ±0.7，**可读**）、
   TB 均值 −2.6（45.6 vs 48.2，超出"均值差 <2 不可读"带）。两个独立基准同向。
2. **趋势相反**：stn8 单调下滑（V\* 90.05→89.53→89.01；TB 45.93→46.91→43.95），
   supn8 单调上升至平台（86.91→89.53→91.62）。
   **越训越差 = 目标构造有系统偏差，不是欠训练。**
3. **逐题翻转不对称**：V\*@50 对齐 190 题，**stn8 赢 2 / supn8 赢 7（净 −5）**。
   参照：A vs supn8 是 5:5 完全对称（同分的标志）。不对称 = 系统性劣化，非噪声。
4. **TB 维度系统性下降**：Comparison **−18.2**(n=44)、Ordering −7.0(n=57)、
   Spatial Containment −6.9(n=29)、OCR −4.4(n=68)；仅 Physical State +4.3(n=23)。
   下降项覆盖大样本维度。

附带：stn8 枚举式作答 15/191 vs supn8 7/191（翻倍），均长 306 vs 336。

---

## 八、最大产出：TV 标量匹配 ≠ 干预结构匹配

我们用 α_max 把 δ 校准到 0.059 以匹配 supn8 的 TV 0.065，并据此声称"剂量已对齐"。
**这个推断是错的** —— 两者 TV 相同，但改写结构完全不同：

| | supn8 | **stn8** |
|---|---|---|
| 干预的位置占比 | ~100%（每位置都改写） | **29%**（71% 位置 q=a，**退化为普通蒸馏**） |
| 干预的 token 占比 | 全 support 温和改写 | **0.4%**（`st_donor_tokens_frac`） |
| 单 token 改写幅度 | 小（e^{β·u} 连续） | **大**（少数 token 被扣 75% 质量） |
| TV(q,a) | 0.065 | 0.059 |

> **同一个 TV 标量，一个是弥散的温和改写，一个是极稀疏的剧烈搬运。**

**规范（已写入项目记忆）**：一切"剂量匹配"设计必须同时检查**覆盖率与改写结构**，不能只看 TV。
这条与 misalign 的教训并列但指向不同 —— 那次是幅度未控，这次是幅度控住了而结构未控。

---

## 九、归因：三个变化里哪个有害

ST 相对 A 同时改了三件事：

1. 关"谷区·负向"（= floor 的机制）
2. 关"正向指数抬升"，改为等比接收（= supn8 的机制，但形式不同）
3. softmax 归一化 → **守恒质量转移**（全新机制，无先例）

已知约束：**supn8 已证 ② 单独做 = 与 A 持平**（V\* 91.62 同分）→ ② 不是罪魁；
floor（① 单独做，非 pair 体制）TB 峰 51.60 疑似有益但归因未解。
故**嫌疑集中在 ③（守恒转移本身）**，或第八节所述的"干预过度稀疏"。

四象限图中，**ST ≈ floor ∩ supn8 的守恒版**，唯一主动力源只剩"头部·负向"一格。本轮结果说明
把力源压缩到这一格（哪怕 TV 相同）是有害的：

> **视觉对比的抑制作用需要弥散地作用在整个分布上（softmax 式软抑制），
> 而非集中搬运少数 token。71% 位置 q=a 等价于丢掉那些位置的全部监督信号。**

### 对原提案核心假设的回答

提案假设：*"视觉对比更适合识别不该继续占据高概率的续写，而非规定整个词表的排序"*，
并推论"精细化抑制结构（共同头部限定 + 低概率入口保护 + 守恒）优于裸 tilt"。

- 前半（**不需要精确排序**）：**得到支持** —— supn8 删掉全部正向指数排序仍与 A 持平。
- 后半（**精细结构更优**）：**被否** —— 结构化后双基准同向劣化，且末端下滑。

---

## 十、诚实边界与未尽方向

1. 单 seed（42）、k≈1 体制；负结果的强度由"双基准同向 + 趋势 + 翻转不对称"四条支撑，
   但仍是一次训练。
2. **超参未扫**：只测了 (τ,ρ,α_max,ε)=(0.10,0.10,0.75,0.30) 一组。coverage 0.29 是否
   本质偏低、还是可通过放宽 τ/ρ 救回，**未知**。
3. 若要救 ST，唯一合理方向是**提高覆盖率 + 保留弥散项**（例如守恒转移与软 tilt 混合），
   但需先有独立证据说明稀疏性确实是病因；当前不建议继续投入。
4. 48×n8 的 A 基线（`pair_ahaAn8_6karmA`）在跑 —— 它是四臂表的锚点，出分后本档案的
   对照列需更新。

---

## 十一、相关文件

- 实现：`verl/trainer/ppo/core_algos.py`（gate `counterfactual_st_enable`）
- 配置：`verl/workers/config/actor.py`（5 字段 + 互斥校验）+ `actor.yaml`
- 单测：`scripts/test_st.py`（七组，含 golden case）
- launcher：`scripts/train_pair_st.sh`
- 训练日志：`Vision-OPD-setup/logs/train_stn8.log`
- 逐题材料：`Vision-OPD-setup/review_stn8/{treebench,vstar}/`
- 终审：`Vision-OPD/docs/09_24_stn8_verdict.md`
- 姊妹档案：`docs/supn8_algorithm_reference.md`、`docs/results_board.md`、
  `Vision-OPD-OPSA/docs/floor_algorithm_reference.md`
