# Negative-History OPD 进展与实验报告（2026-10-02）

> 本文汇总本季已确立的结论、negative_history 线的设计/实现/验证、目前的实验结果与分析、
> 在跑任务和待定事项。所有准确率为 **gpt-oss-120b LLM-as-judge 口径**；判读口径为
> **`{30,40,50}` 三个 checkpoint 取峰值 + ×2 同侧规则**（两次峰值都 >51.60 判增益，都 <49.88 判负，
> 否则不可判读）。

---

## 一、背景：本季已确立的结论

### 1.1 锚点

| 锚点 | 各次峰值（TB） | V\* 峰 | 说明 |
|---|---|---|---|
| **A**：原版 OPD-Aha + pair + n2 | 49.88 / 51.11 / 51.11 / 51.60 | 91.6 – 94.8 | 中位 51.11；同配置重跑峰差 1.23 |
| **B**：suppression-only (supn2) | 50.37 / 47.41 | 91.62 / 89.01 | B 两次差 2.96，超噪声带 |
| V0：原版 StdOPD | 49.38 / 48.64 | 87.4 – 88.5 | 两次都 < A 最小观测 |

固定配置：训练集 2459 题高清 6karmA、seed=42、batch 48、51 步、lr 2e-6、n=2、TB 用 TreeVGR 官方 HF 推理协议（greedy）、V\* 用 OPD-Aha 官方 `infer.py`（vLLM, temperature=0）。

### 1.2 主线结论

1. **A 是目前最优配置**：8 个目标构造改造臂（floor / sup / tanh / γ / SAD / ST / setshift / C-temporal 等）峰值无一超过 A 中位数 51.11；tanh、floor+pair 两次都 <49.88 判负。目标构造这一自由度已基本穷尽。
2. **软分布贡献约 1.4 点**：γ=50 硬标签三次均值 49.55 vs A 均值 50.93，V\* 无损。OPD-Aha 的增益主体来自"特权视图 argmax 上的 CE"。
3. **tilt 相对 V0 的增益真实**：V0 两次 TB 都低于 A，V\* 低 3–6 点，双基准一致（混杂：V0 用 hide 视图，A 用 pair）。
4. **lr 需随有效 batch 缩放**：n8+lr8e-6 ≈ n2+lr2e-6；n8 无额外收益且开销 4×。

---

## 二、negative_history 设计（docs/negative_history/02–07）

以 **suppression-only（Arm B）** 为底座：`q ∝ p⁺ · exp(β_t · min(u, 0))`，`u = log p⁺ − log p⁰`。
从**实际采样 token** 的视觉冲突 `c_t = [−u_t(y_t)]₊` 派生两个模块：

| 臂 | 模块 | 公式 |
|---|---|---|
| **C** | 历史自适应 β | `N_t = Σ_{k<t} c_k`（exclusive），`s_t = N_t/(1+N_t)`，`β_t = β(1+s_t) ∈ [β, 2β)` |
| **D** | 未来持续性 JSD 加权 | `F̄_t = Σ_{k>t} c_k / K_t`（exclusive 后缀均值），`r_t = F̄_t/(1+F̄_t)`，`w_t = 1+r_t ∈ [1,2)`，`L = Σ M w ℓ / Σ M w` |
| **E** | C + D | — |
| Cmis | C 的 misalign 控制 | 行内打乱 `c_t` 顺序（保边际、毁时序） |

B/C/D/E 构成 2×2 因子设计。用户决策：B 不补第三次（基线取现有两次）；E 条件跑（C 或 D 至少一个有信号才跑）。

---

## 三、实现与验证

### 3.1 代码（OPD-Aha-Floor:sup，commits `8ee4dfd`、`89eef5f`）

- `verl/workers/actor/dp_actor.py`（3 行）：把 null 视图 realized-token log-prob（`teacher_null_outputs["log_probs"]`，此前已计算但未使用）传入 loss。与 real 侧 realized log-prob 同源同对齐；`c_t` 精确，不受采样 token 落在 top-100 support 之外的影响。
- `verl/trainer/ppo/core_algos.py`：轨迹状态分支（`no_grad`）、C 的 `β_t` 注入 tilt、D 的加权聚合（加权分母按比例缩放 `batch_num_tokens`）、misalign shuffle、运行时守卫、`nh/*` 机制指标。
- 配置：`counterfactual_hist_adaptive_beta / hist_alpha / hist_shuffle / future_weight / future_alpha`（dataclass + yaml）。
- 启动脚本 `scripts/train_pair_neghist.sh`：`HIST/FUT/HIST_SHUFFLE` 环境变量，实验名自动带臂标识。

### 3.2 运行时守卫（写在 loss 函数里——dataclass 校验在本训练路径不执行）

- 任一门开必须 `u_clip_pos=True`，且必须有 null realized log-prob；
- 与 ST / floor / tanh / γ 互斥（单变量纪律）；shuffle 必须配 hist；
- **有效位出现非有限 `β_t` 或 `w_t` 立即抛 `FloatingPointError`**。

### 3.3 测试（`scripts/test_neghist.py`，45 项全绿）

- 门关时与 sup bit-identical；缺新 key 回落关闭；
- 文档工作例逐数对拍（02 §7、04 §4/§6/§8、05 §7/§9/§15）；
- C/D/E 生产 loss 与独立参照实现对拍 <1e-5；
- **4 个变异体必须被抓**：inclusive cumsum、future 含当前位、分母漏加权、c 不乘 mask；
- padding 毒化（有限值与 ±inf）loss 不变；有效位 NaN 必须报错；
- shuffle 保边际、毁时序；守卫全触发。
- 回归：test_sup / test_floor / test_tanh / test_gamma 全绿。

### 3.4 code review 发现并修复

| 严重度 | 问题 | 修复 |
|---|---|---|
| 中 | padding 位 log-prob 为 ±inf 时 `inf×0=NaN`，经 D 的反向 cumsum 扩散到整行有效位 | 改用 `torch.where`；另加非有限值即报错守卫 |
| 低（潜伏） | D 的加权分母直接替换 `batch_num_tokens`，若传入全局计数会按 dp 改变 loss 尺度 | 改为按权重比例缩放 |
| 低（指标） | `nh_N_last_mean` 把整行被 mask 的样本按 0 计入 | 排除空行 |

真实 2 步冒烟（E 全开）通过，日志无 NaN/inf。

---

## 四、实验结果（第一次跑；按 ×2 规则尚不能判定）

配置：3 卡 / SP1、n2、lr2e-6、seed42、51 步。

| 臂 | TB（30/40/50） | TB 峰 | V\* 峰 | 机制指标（51 步均值） |
|---|---|---|---|---|
| A | — | 49.88 – 51.60 | 91.6 – 94.8 | — |
| B | — | 50.37 / 47.41 | 91.62 / 89.01 | target_tv ≈ 0.07 |
| **nhC** | 47.16 / 44.94 / 47.90 | **47.90** | 91.10 | β_t 6.31，c>0 占比 0.44，target_tv ≈ 0.085 |
| **nhD** | 46.42 / 49.14 / 47.90 | **49.14** | 89.53 | w_t 1.037，c>0 占比 0.45 |

初步读法：
- nhC 峰低于 A 最小观测，落在 B 两次之间；
- nhD 与 B 同带，加权剂量只有约 4%，"机制太弱"与"机制无效"无法区分；
- **V\* 侧**：nhD 89.53、supn2r2 89.01 都低于 A 族 2–5 点，提示 suppression-only 底座本身可能压低 V\*。

---

## 五、nhC case study（详见 `nhC_r1_casestudy.md`）

1. **参照应为 B**：逐题池化 TB 准确率 A 49.2 / B 47.7 / C 46.7。C−B = −1.0（噪声内）；"C 差"主要是 sup 底座本身低于 A。
2. **C ≈ 加大剂量的 B**：`N_t` 几十 token 即饱和，`β_t` 在轨迹内近似常数（6.6→6.1）；训练动力学仅 `target_tv` 比 B 高约 27%，其余（grad_norm、max_prob、回复长度）同 B。**历史模块实际测到的是剂量，不是时间自适应。**
3. **类别偏移**（单次跑，类别 SE≈5pt，仅提示）：Attributes +9.2、Physical State +6.8；Spatial Containment −8.4、Ordering −7.2、OCR −5.9、Contact&Occlusion −5.4（均相对 A）。
4. **输出形态无变化**：词数、字母分布、无效答案率均同 A/B，不是格式塌陷。
5. **翻转题**：A 9/9 对而 C 错 21 题（反向 11 题）。典型模式是自信地编造细节——q85 断言"holding a camera"、q154 断言手指与保险杠间"small but distinct space"。

---

## 六、在跑与排队（3 卡 hold 20743823，约 10-03 14:10 到期）

| 任务 | 目的 | 预计出分 |
|---|---|---|
| nhCr2 | C 的第二次 | ~10-02 17:30 |
| nhDr2 | D 的第二次 | ~10-02 23:30 |
| supb63 | **剂量对照**：B + 常数 β=6.3 | ~10-03 05:30 |
| supb63r2 | 剂量对照第二次 | ~10-03 11:30（距 hold 到期仅约 2.5h，可能需补申） |

剂量对照判读：supb63 ≈ nhC → 历史模块无独立贡献；supb63 明显好于 nhC → 历史调节有害；明显差于 nhC → 历史调节确有作用。

编排：sbatch 托管（20743949 段 1、20760958 剂量对照），每臂评测后自动回收中间 checkpoint。TB 推理保持 HF（与 TreeBench 官方协议和全季已有分数一致，不切 vLLM）。

---

## 七、待定事项

1. **C 是否仅为剂量效应**：等 supb63。
2. **E 是否跑**：按预登记条件，C/D 均无信号则不跑；目前看大概率不跑。
3. **备选方向（未实现，等结果再定）**：把历史调节放到 **A 底座**上、只作用于负半边：
   `q ∝ p⁺ · exp(β·u⁺ + β_t·u⁻)`，并把 `N_t` 换成均值版 `N_t / t` 以消除饱和，使 `β_t` 在轨迹内真正变化。动机：sup 底座本身 TB −1.5、V\* −2~4；且累积版历史在任何底座上都会退化为剂量。
4. **共性现象**：suppression-only 底座可能系统性压低 V\*，值得在总账中单列。
