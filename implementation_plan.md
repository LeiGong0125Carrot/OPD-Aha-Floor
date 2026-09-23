# Implementation Plan: Selective Suppression v2 — 共同头部守恒质量转移臂(ST)

日期: 2026-09-23 凌晨 | 状态: **计划(等 supn8 全量评测收尾后实施)**
基底: `/scratch/nkw3mr/Vision-OPD/OPD-Aha-sup`(分支 sup,483d70f=A 臂精确基线)

## 0. 今夜时间线(用户睡 9h,~09:00 醒)

| 时刻(约) | 事件 | GPU |
|---|---|---|
| ~03:45 | supn8 训练收尾(51 步) | 3 卡训练中 |
| ~04:00-06:00 | 主链 stage2-4:step 30/40/50 评测+双 judge+归集 | 逐段 |
| ~06:00-09:00 | **扩展评测**:其余 8 个 ckpt(5,10,15,20,25,35,45,51)全量 TB+V\*+judge | GPU0=TB, GPU1=V\*, GPU2=judge |
| 06:00 起(CPU 并行) | case study(supn8 vs A/base 逐题分析)+ ST 实现+测试+review | 无 GPU |
| ~09:00 | ST 冒烟 2 步 → 正式 51 步(48×n8 同体制) | 3 卡 |

扩展评测脚本(night_eval.sh)已挂门控自动跑,不依赖会话存活;ST 实现依赖我在场
(monitor 会叫醒),若会话死亡则评测照跑、训练顺延到用户唤醒。

## 1. 方法(用户提案 §3 的直接实现)

v1(supn8,在跑)= tilt 内最小版:q ∝ p⁺·e^{β·min(u,0)},无保护、无守恒。
**v2(ST)= 概率空间的有界扣减+组内等比守恒转移,完全替换指数 tilt:**

记 a=p⁺(教师真实视图分布,student top-100+tail support 上),s=学生分布(detach),
u=log p⁺−log p⁰(全词表归一化后 gather,tail 精确聚合)。

- 相对头部:H(r;ρ) = {v: r(v) ≥ ρ·max_w r(w)},ρ=0.10
- 显著负向:N = {v: u(v) < −τ},τ=0.10
- **Donor D = N ∩ H(a;ρ) ∩ H(s;ρ)**(explicit 列;tail 永不入 D)
- **Recipient R = {v: u(v) ≥ −τ}**(explicit 列;tail 永不入 R)
- 其余 F(含 tail 桶):q(v)=a(v) 不动
- M_D=Σ_D a,M_R=Σ_R a;**δ = min(ε, α_max·M_D)**,ε=0.30,α_max=0.50
- D 空或 M_R=0 → δ=0,q=a(零转移守卫)
- q(v) = a(v)·(1−δ/M_D) [v∈D];a(v)·(1+δ/M_R) [v∈R];a(v) [v∈F]。**无全局 softmax。**

可机器验证的性质:①守恒 Σq=1;②donor floor q≥(1−α_max)a;③R 内比例不变
q(v)/q(w)=a(v)/a(w);④TV(q,a)=δ 精确;⑤组内等比=固定组间转移量下 min KL(q‖a) 解。

超参 τ/ρ/α_max/ε 用提案示例值起步(**未调优,冒烟后按 δ 实测校准 ε**,目标 δ 均值
落 0.10-0.25——介于 supn8 的 0.065 与 A 的 0.25 之间,可跨臂解读)。β 在本臂不使用。

## 2. 改动清单(全部 gated,off=bit-identical)

1. **config**(actor.py dataclass + actor.yaml 双处):
   `counterfactual_st_enable: bool=False`、`counterfactual_st_tau: float=0.10`、
   `counterfactual_st_head_ratio: float=0.10`、`counterfactual_st_alpha_max: float=0.50`、
   `counterfactual_st_eps: float=0.30`。
   校验:st_enable 需 null_mode+full_logit;**与 counterfactual_u_clip_pos 互斥**;
   τ≥0、0<ρ≤1、0<α_max<1、0<ε≤1。
2. **core_algos.py**(与 u_clip 同一分支点,st_enable 时替换整个 tilt 构造):
   - 概率空间张量运算,全 no_grad 之外仅 q 本身(输入均 detach:a、u 来自 no_grad 教师,
     s 显式 .detach() 后仅用于 H(s) 指示集);
   - explicit_mask = 前 K 列(tail=最后一列恒 F);
   - piecewise 缩放后直接 log(q+1e-12) 交给现有 JSD(q 已归一,数值上再做一次
     renormalize 防浮点漂移:q/=Σq)。
3. **仪表盘**(masked mean,键名避 min/max):
   `st/delta_tv`(实际 δ)、`st/donor_coverage`(δ>0 位置占比)、`st/donor_tokens_frac`、
   `st/mass_donor`(M_D)、`st/mass_recipient`(M_R)。
4. **launcher `scripts/train_pair_st.sh`**:仿 train_pair_sup.sh,48×n8、3 GPU、SP=1、
   51 步、save_freq=5、seed42、EXPERIMENT=`pair_stn8_6karmA`(短名 stn8),
   唯一新增 override `counterfactual_st_enable=True` + 四个超参。
5. **测试 `scripts/test_st.py`**(七组,numpy 全式对拍为核):
   ① off == 基线 bit-identical(含 absent-key);② D 空构造 → q==a 精确;
   ③ numpy 参考实现对拍(提案数值例:六 token 表,q=(0.36,0.09,0.04,0.075,0.01,0.425)
   直接作为 golden case);④ 守恒 Σq=1 + TV(q,a)==δ;⑤ donor floor + R 内比例不变;
   ⑥ 仪表盘五指标存在有限;⑦ config 校验(互斥/范围)。
6. **eval 接入**:eval_temporal_interactive.sh short_name 加 `pair_stn8_6karmA→stn8`;
   chain 复制为 chain_st.sh(同门控结构,judge-only)。

## 3. 判读(judge-only;主对照=supn8,同 48×n8 体制)

| 比较 | 回答的问题 |
|---|---|
| ST vs supn8(主) | 头部限定+低概率入口保护+守恒,是否优于裸 u⁻ tilt |
| ST vs A(48×n2,弱参照) | 跨体制仅方向性参考,不下结论 |
| st/donor_coverage | ≈0 → 机制大部分位置恒等,结果作废(预登记) |
| δ 实测 | <0.05 → 欠剂量,先调 ε 再判 |

**缺口(诚实声明)**:48×n8 体制的 A 基线尚未跑;supn8/ST 对 A 的绝对差需等同体制
A 重跑才可读。本轮先回答 ST vs supn8 的机制内比较。

## 4. Case study 规格(supn8,评测出分后做)

1. V\* 逐题:supn8@30/40/50 vs A@30(91.62) 的 judge 文件对齐——翻转题清单(双向)、
   文本风格(是否 SAD 式枚举、box 使用、长度)、judge_source 分布;
2. TB 维度指纹:各 step 的 per-dimension 得分 vs base 44.94 口径(OCR/Attributes 等轴);
3. 规则-judge 分歧行清单(抽取器病理是否复发);
4. 训练动力学联读:frac_u_pos 0.53→0.39 漂移段(step5-7) 与行为变化对齐;
5. 产出:docs/09_23_supn8_casestudy.md + review_supn8/ 全量文件供用户人工复核。

## 5. 风险与预案

- ST 冒烟 OOM(概率低,张量运算同量级)→ 与 supn8 同配方,不应有新增内存项;
- δ 实测过小(head 交集太苛)→ 预案:ρ 0.10→0.05 或 selector 降级 teacher-head(预登记,
  只动一个旋钮);
- 时间线滑动:ST 训练 ~10h,用户醒时最多到 step ~5,可视结果决定续/停;
- hold 剩余 ~1d3h(09:00 时)足够 ST 全程+评测。
