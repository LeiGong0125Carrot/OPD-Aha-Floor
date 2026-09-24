# supn8 算法档案：suppression-only u⁻ tilt

**结论先行：本季唯一"不塌"的机制替换臂 —— 终点与 OPD-Aha 原版双基准持平
（V\* 91.62 完全同分），证明 Aha 的收益不需要"正向视觉增量对替代续写的指数重排"。**

代码：`verl/trainer/ppo/core_algos.py`，gate `counterfactual_u_clip_pos`（commit `0bb76e3`）。
基线：`483d70f`（官方 OPD-Aha main，我们所有实验分叉前的精确基线）。写作日期 2026-09-24。

---

## 一、算法定义

记 p⁺ = 特权视图 teacher 分布（本臂视图为 pair：[全图, GT crop 2.42×]），
p⁰ = null 视图 teacher 分布（[全图, 均值色块]，`null_scope=last` 只遮末图），
二者都在**学生 top-100 支持** + tail 桶上取值（全词表 log_softmax 后 gather，再精确补 tail）。

视觉增量：

$$u(v) = \log p^+(v) - \log p^0(v)$$

原版 OPD-Aha（A 臂）的重建目标：

$$q_A \propto p^+ \cdot \exp(\beta \cdot u), \qquad \beta = 4$$

**suppression-only 把 u 砍掉正半边**：

$$\boxed{\;q_S \propto p^+ \cdot \exp\big(\beta \cdot \min(u,\,0)\big)\;}$$

即：**视觉支持（u>0）的 token 不再被指数抬升，只有视觉反对（u<0）的 token 被压制**；
被压走的质量经 softmax 归一化按 p⁺ 比例分配给所有未被压制的 token。

其余一切与 A 逐字节相同：anchor 仍是 p⁺、β 不变、support 定义不变、tail 参与 clamp 与 tilt、
下游仍是 α=0.5 的 JSD、长度归一化不变。**gate off 时与基线逐位一致**（`atol=0` 机器验证）。

---

## 二、代码（核心 3 行）

```python
u_term = teacher_real_distill_log_probs - teacher_null_distill_log_probs
if counterfactual_u_clip_pos:
    # Only the negative half of the visual contrast enters the target; tokens the real
    # image supports (u > 0) keep their p_real mass and receive the suppressed mass
    # proportionally via renormalization.
    with torch.no_grad():
        sup_frac_u_pos_per_token = (u_term > 0).float().mean(dim=-1)
    u_term = torch.clamp(u_term, max=0.0)
teacher_distill_log_probs = F.log_softmax(
    teacher_real_distill_log_probs + counterfactual_extrapolation_beta * u_term, dim=-1)
```

**实现要点**

1. **必须用全词表归一化后的 log-prob**，不能用 raw logits 差。u 的正负号是本臂唯一使用的信息，
   位置相关的 log Z 项会改变符号集合。本仓库的 dump 与训练路径都是全词表 `log_softmax`
   后 gather 到学生 top-k，天然满足。
2. **不可用 `add_tail=False`**（support 内重归一化）。那会给 u 注入
   $\log p^0(E) - \log p^+(E)$ 的偏移，可能翻转符号。生产恒为 `add_tail=True`。
3. **tail 桶同样参与 clamp 与 tilt**（保守，与 A 的现行为一致）。
4. 目标构造全在 `no_grad` 的 teacher 张量上，梯度只经 JSD 的 student 一侧回流。

**仪表盘**：`self_distillation/sup_frac_u_pos` = support 内 u>0 的 token 占比（被 clip 掉的
信号量）。若 ≈0 则本臂与 A 无差、结果不可读。实测 0.53→0.39-0.43（近半信号被 clip，剂量充足）。

**单测**：`scripts/test_sup.py`，七组 —— off 逐位一致（含 absent-key 回退）、u≡0 退化为普通蒸馏、
numpy 全式对拍（含 tail 列）、组间等比性质、q/a 对 u 的单调性、仪表盘、config 校验。

---

## 三、详细举例：四臂在同一张表上

人造六 token 例（β=4，学生分布取 s=a 以模拟师生同被错误前缀牵引）：

| token | a=p⁺ | u | **A（原版）** | floor | **supn8** | ST |
|---|---|---|---|---|---|---|
| Therefore | 0.600 | −0.154 | 0.00089 | 0.00089 | **0.5416** | 0.360 |
| Thus | 0.150 | −0.288 | 0.00013 | 0.00013 | **0.0792** | 0.090 |
| Wait | 0.040 | −0.118 | 0.00007 | 0.00011 | **0.0417** | 0.040 |
| Actually | 0.030 | +0.693 | 0.00132 | 0.00132 | **0.0501** | 0.075 |
| Indeed | 0.010 | −0.405 | 0.00001 | 0.00003 | **0.0033** | 0.010 |
| However | 0.170 | +1.917 | **0.99759** | 0.99753 | **0.2841** | 0.425 |
| | | TV(·,a) | 0.828 | 0.828 | **0.136** | 0.300 |

### 读法一：A 的指数 tilt 极其剧烈

β=4 下 However（u=1.92）独吞 **99.76%** 的质量，其余全部被压到 0.1% 以下，TV=0.83 —— 目标
分布几乎变成一个 one-hot。这解释了实测现象：**A 的 `counterfactual_target_tv` ≈ 0.25，而
supn8 只有 0.065（约 1/4）—— 正半边贡献了绝大部分干预幅度。**

（附带发现：这个例子里 floor 与 A 几乎重合。原因是 floor 只保护谷区负向，而当存在一个 u 极大
的 token 时，指数 tilt 的归一化被它完全支配，谷区保护的效果被淹没。这说明 **floor 的作用高度
依赖 u 的分布形状**，在有极端正向 u 的位置上它近乎空操作。）

### 读法二：u≥0 的 token 共享同一个接收因子

Actually 和 However 都恰好乘 **1.6712 = 1/Z**（Z = Σ a·e^{β·min(u,0)} = 0.5984）。于是

$$\frac{q_S(v)}{q_S(w)} = \frac{a(v)}{a(w)} \qquad \forall\, v,w:\ u \ge 0$$

**正向内部的相对排序完全不变** —— 视觉信号只用来"指出该压谁"，不用来"排序替代项"。
这是本臂想检验的核心假设的直接实现。

### 读法三（重要）：归一化陷阱确实发生

看 **Wait（u=−0.118，被惩罚）**，概率却从 0.040 **升到 0.0417**。

因为它的惩罚因子 $e^{4 \times (-0.118)} = 0.624$ **弱于**全局归一化因子 $Z = 0.5984$，
除完净增。精确判据：

$$q_S(v) > a(v) \iff \exp(\beta \min(u,0)) > Z \iff u > \frac{\ln Z}{\beta} = -0.128$$

**只有 $u < \ln Z/\beta$ 的 token 才真正被压低**；落在 $(-0.128,\,0)$ 这条窄带内的弱负向
token 反而上升。所以：

> **supn8 不是概率空间的"抑制"，它是 tilt 框架内"抑制假设"的最小实现。**

真正守恒的抑制是 ST 臂（见 `st_algorithm_reference.md`）—— 而 ST 的实测更差，这让本条边界
从"实现缺陷"变成了一个有信息量的发现：**弥散的软抑制优于精确的守恒扣减。**

---

## 四、训练配置

| 项 | 值 |
|---|---|
| 实验名 | `pair_supn8_6karmA`（短名 supn8） |
| 数据 | `train_6karmA_pair.parquet`（2459 题高清 6karmA，用户标准训练集） |
| 视图 | pair：p⁺=[全图, GT crop 2.42×]，p⁰=[全图, 均值块]（`null_scope=last`） |
| β | 4.0（与 A 相同，不重标定） |
| batch × rollout | 48 × **n8**（贴合论文 rollout 轴） |
| GPU / SP | 3× RTXPro6000 / ulysses SP=1 |
| 步数 | 51（1 epoch），save_freq=5 |
| seed | 42（`data.seed`，用户规则：不做多 seed） |
| α (JSD) | 0.5 |
| launcher | `scripts/train_pair_sup.sh` |

**体制变更史（踩坑记录）**：原计划 96×n8（作者论文设置），step1 即 driver CPU
**934G/1024G**，step2 被 `oom_kill`（作者称 1T 够，在他的部署形态下成立；我们 3 卡
colocated vLLM + 全套 offload 装不下 768 seqs）。降级 48×n8 后内存平台 **~590G**，安全。

---

## 五、训练动力学

| 指标 | 起点 | 全程 | 终点 |
|---|---|---|---|
| `sup_frac_u_pos` | 0.53 | 0.39-0.53 带内震荡 | 0.43 |
| `counterfactual_target_tv` | 0.064 | 0.063-0.090 | 0.068 |
| `cpu_memory_used_gb` | 501 | 平台 ~590 | 598 |
| response_length | 152 | 稳定 | — |

`sup_frac_u_pos` 从 0.53 缓降到 0.39-0.43 = policy 漂移使 support 构成变化，非塌缩；
target_tv 全程稳定、无剪刀差信号；无 SAD 式的 resp_len 失稳。

---

## 六、全量评测结果（judge 口径，gpt-oss-120b；**规则分不入表**）

11 个 checkpoint（TB n=405，V\* n=191）：

| step | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 51 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **V\*** | 81.68 | 83.77 | 86.39 | 85.86 | 87.96 | 86.91 | **91.10** | 89.53 | 88.48 | **91.62** | 91.10 |
| **TB** | 46.91 | 45.93 | 45.19 | 45.19 | 46.17 | 47.65 | 46.17 | 48.40 | 47.90 | **48.64** | 48.64 |

- **V\* 单调爬坡至平台**：81.7（≈base 83.25 附近）→ step35 首达 91.10 → 末端 91.6/91.1
  稳定，**无回落、无塌陷**；平台期（35-51）均值 90.4。
- **TB**：早期小凹（45.2@15-20，噪声带内）后爬至 48.4-48.6 平台，末端与 A 家族一致。
- 平台到达步数 ~35（A 在其体制下 ~30 即到），慢约 5 步。

### 对照（全部 judge 口径）

| 臂 | 视图/体制 | V\* @50 | TB @50 | 判定 |
|---|---|---|---|---|
| base Qwen3.5-4B | — | 83.25 | 44.94 | 起点 |
| **A（我们自训，48×n2）** | pair | **91.62** | 49.38 | 基线 |
| A-rr1（同配置重跑） | pair | 90.58 | 49.14 | 噪声底 ±1.5 |
| **supn8（本臂，48×n8）** | pair | **91.62** | 48.64 | **持平** |
| ST（48×n8） | pair | 89.01 | 43.95 | 判负 |
| SAD（λ=0 学生锚，48×n2） | pair | 93.19 | 47.99 | 持平/V\* 微优 |
| 官方 pair ckpt | pair | 93.72(峰) | 50.37 | 上沿 |

---

## 七、Case study：与 A@50 的逐题对照

V\*@50，对齐 190/191：

- **翻转完全对称：supn8 赢 5 / A 赢 5，净差 0** —— 同分不是数字巧合，是两个功能等价、
  误差结构同规模的模型（与 A vs A-rr1 重训的 8v8 对称同款）。
- **逐字相同答案 0/190** —— 模型实质不同，"同分 ≠ 同模型"。
- 翻转样例均为普通属性色/左右位置分歧，无系统模式。

**风格（SAD 病理未复发）**：

| 指标 | supn8@50 | A@50 |
|---|---|---|
| V\* 答案均长 | 336 字符 | 366 |
| 枚举式作答（≥3 选项串） | 7/191 | 12/191 |
| judge_source | 规则 172 + LLM 19 | 同量级 |

首字母抽取器在 supn8 输出上工作正常（TB@50：规则 197 + LLM 救回 0），**无 SAD 式假阴性病理**。
对照 SAD（换 anchor → 枚举式风格漂移），得到一条干净的归因：
**风格漂移来自 anchor，不来自 tilt 的正半边。**

**TB 维度指纹**（supn8@50 vs A@50）：Comparison +11.4、Physical State +8.7、Attributes +6.9 /
Ordering −8.8、Object Retrieval −6.2、Perspective −4.7；总分 −0.74，翻转 29:32 对称。
各维摆动均在小样本噪声内（n=13-85，σ≈5-14pp），不构成指纹证据。

---

## 八、科学意义

四象限图（把 support 按 p⁺ 头部/谷区 × u 正/负 切开，看每臂施加的力）：

| 象限 | A | **supn8** | floor | ST |
|---|---|---|---|---|
| 头部 · u>0 | β·u | **0** | β·u | 等比接收 |
| 头部 · u<0 | β·u | β·u | β·u | 守恒扣减 δ |
| 谷区 · u>0 | β·u | **0** | β·u | 等比接收 |
| 谷区 · u<0 | β·u | β·u | **0** | 0（不动） |

supn8 关掉**整列正向**（两格）；floor 关"谷区·负向"一格 —— 两者裁剪方向互补、不重叠。

**本臂的结论**：终点与 A 双基准持平 ⇒ **Aha 收益不需要"正向 u 对替代续写的指数重排"**。

与同季其他臂合并读：

| 改动 | 结果 | 推论 |
|---|---|---|
| 换 anchor（SAD，λ=0） | 同分，**风格变**（枚举式） | 风格由 anchor 决定 |
| 删正半边（supn8） | 同分，风格不变 | 正向重排非必要 |
| 结构化守恒转移（ST） | **变差**，末端下滑 | 抑制需**弥散**，不能集中搬运 |

> **合并结论：Aha 的有效成分是负向集合的抑制作用；它需要弥散地作用在整个分布上
> （softmax 式软抑制），而非压缩为少数 token 的精确搬运。输出风格由 anchor 承载，
> 能力平台由负向抑制承载。**

---

## 九、诚实边界

1. **同分是"不劣化"的证据，不是"更优"**。单 seed（42，用户规则），只报配对不确定性。
2. **爬坡慢约 5 步**（step30 差 5.2 点）与剂量只有 A 的 1/4 定性一致，但**混杂 n8 vs n2 的
   体制差异** —— 需同体制 A 基线（`pair_ahaAn8_6karmA`，进行中）才能封上这个缺口。
3. **非守恒**：第三节的归一化陷阱说明本臂不是真正的概率抑制；落在
   $(\ln Z/\beta,\,0)$ 窄带内的弱负向 token 反而上升。
4. **低概率负向 token 无保护**（Wait 类照压，只是压不动）—— 这是与 floor / ST 的已知差异。
5. k≈1 体制（1 epoch）只采样了累积风险的低区，多 epoch 不外推。

---

## 十、相关文件

- 实现：`verl/trainer/ppo/core_algos.py`（gate `counterfactual_u_clip_pos`）
- 配置：`verl/workers/config/actor.py` + `verl/trainer/config/actor/actor.yaml`
- 单测：`scripts/test_sup.py`（七组）
- launcher：`scripts/train_pair_sup.sh`
- 训练日志：`Vision-OPD-setup/logs/train_supn8.log`
- 逐题材料：`Vision-OPD-setup/review_supn8/{treebench,vstar}/`（44 个文件）
- case study：`Vision-OPD/docs/09_23_supn8_casestudy.md`
- 姊妹档案：`docs/st_algorithm_reference.md`、`docs/results_board.md`、
  `Vision-OPD-OPSA/docs/floor_algorithm_reference.md`
