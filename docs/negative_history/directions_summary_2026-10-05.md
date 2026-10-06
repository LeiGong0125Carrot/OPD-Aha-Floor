# 方向总账：到 2026-10-05 为止试过的所有方向与结果

> 口径：准确率一律 gpt-oss-120b LLM-as-judge（规则分不进结论表）；TB = TreeBench 405 题（TreeVGR 官方 HF greedy），V\* = 191 题（OPD-Aha 官方 `infer.py`）。固定训练配置：Qwen3.5-4B，2459 题高清 6karmA pair 数据，seed 42，冻结初始教师，JSD α=0.5，学生 top-100 + tail，batch 48，lr 2e-6，51 步；除注明外 n=2、3 卡/SP1。每臂 ×2（同配置同 seed）。
> 峰值口径：本表全部为 {30,40,50} 三点峰值（10-05 20:45 起新臂改为全部 ckpt 取峰，见 `amis_results_2026-10-05.md`）。
> 目标（用户 10-03）：做出与 OPD-Aha 不同的方法论，持平 A 即可（两次 TB ≥ 50.5 且 V\* ≥ 93.5）；不靠调超参；永不再做"A + 常数剂量"对照。

## 1. 参照

| 臂 | 设定 | TB 峰 | V\* 峰 |
|---|---|---|---|
| **A**（ahaAn2 / rr1 / rr2 / Ahf 当第 4 次） | OPD-Aha 原版：pair 视图 [全图, GT crop]，null = [全图, 均值色块]，β=4 | 49.88 / 51.11 / 51.11 / 51.60 | 94.76 / 93.19 / 94.24 / 93.72 |
| A @ n=8（ahaAn8 / an8lr8 ×2 / an8sp1） | 同上，n=8（lr8 = lr 8e-6） | 48.15 / 51.60 / 50.12 / 48.64 | 93.72 / 91.62 / 91.62 / 92.15 |
| **V0**（StdOPD） | 目标 = p⁺，无 null、无 tilt | 49.38 / 48.64 | 87.4–88.5 |
| base（不训练） | — | ≈44.9（direct） | 83.25 |

A 相对 V0 的收益：V\* +5，TB +1–2。同配置同 seed 的 A 四次 TB 相差 1.7 点、V\* 相差 1.6 点——这是评测噪声的尺度，后面所有"持平"都要在这个尺度下读。

## 2. 目标函数 / 剂量 / 形状类（在 A 的 u 上做文章）

| 方向 | 假设 | 设定 | TB 峰（×2） | V\* 峰（×2） | 判定 | 文档 |
|---|---|---|---|---|---|---|
| floor（floorn2；flAn2 为 A 底座版） | 给 tilt 加下限防过压 | `counterfactual_floor_alpha` | 49.63 / 49.38；50.12 / 50.62 | 91.62 / 92.15；92.15 / 91.62 | ≈A 以下，无增益 | 09_08 verdict |
| γ 锐化（gam50 ×3） | 对目标再做温度锐化 | `target_gamma`=0.5 | 50.37 / 48.89 / 49.38 | 92.67 / 93.72 / 92.67 | ≈A；γ×4 剂量反应为负 | gamma_sharpening_plan |
| tanh 压缩 u（tanhn2 ×2） | 压 u 的尾部避免爆炸 | `tanh_scale` | 47.65 / 48.15 | 91.62 / 92.15 | 判负 | tanh_algorithm_reference |
| **suppression-only**（supn2 ×2；supn8；supb63 ×2） | 只留 u 负半边 | `u_clip_pos=True` | 50.37 / 47.41；48.64；50.12 / 47.16 | 91.62 / 89.01；91.62；91.62 / 91.62 | **终点 V\* 与 A 同分（91.62）→ 正半边非必要**；TB 方差大 | sup-arm 记录、09_23 casestudy |
| ST 守恒转移（stn8） | 用共同头部质量转移替代 tilt | `st_enable` | 46.91 | 90.05 | 判负；TV 匹配 ≠ 结构匹配 | 09_24_stn8_verdict |
| 负历史 C/D（nhC ×2、nhD ×2） | 按历史冲突调 β（cumsum / mean） | `hist_adaptive_beta` | 47.90 / 49.14；49.14 / 48.40 | 91.10 / 92.15；89.53 / 91.10 | 判负 | results_2026-10-03 |
| Ahm / **Ahf** / X1m（×2 各） | 历史+未来残余调 β（负半边）；X1 = 正半边 | `hist_mode=hf` 等 | Ahm 48.64 / 49.88；**Ahf 51.60 / 50.62**；X1m 47.65 / 49.14 | 94.24 / 93.19；**93.72 / 92.67**；92.67 / 92.15 | Ahf 持平 A（非超过）；Ahm 剂量↑TB 单调↓；X1 判负 | summary_history_future_on_full_aha |
| C-temporal（时间信用，旧仓） | 把后续 JSD 成本归因给早期动作 | PG 正交项 | 49.88 vs 打乱对照 50.62 | 同 A | 判死：控制臂 ≥ 主臂，纯幅度效应 | 09_10_results |
| TZR / setshift / γ×4（旧仓） | "精确定位"类三轴 | — | 均 ≤ A | — | 六连败，三轴关闭 | tzr-plan |
| PREF（margin-ranking 替代 β·u，旧仓） | 排序损失替代 CFG 放大 | λ=0.1 | 对照裸 pair 52.35 / 93.72 | — | 未超 | pref-arm |
| Q-contrast（答案位对比，旧仓） | 在答案位做对比 | — | Δans −8.5 nats 反向 | — | no-go；定律 = 负视图必须共享前缀承诺结构 | qcontrast-line |
| rollout-disagree / TGNVMT 门控（旧仓） | 用组内分歧或门控定位纠正 | n2 vs n8 | n2 峰 V\* 94.76（全季最高）；n8 末端崩 | — | 判死 | rollout-disagree |

**跨线结论**：A 的 u 在剂量、形状、位置、状态四个轴上的所有改法，没有一个在噪声尺度之外超过 A；唯一干净的减法是"正半边可以去掉"。

## 3. 去 visual-null 类（换参照、换底座、换来源）

| 方向 | 假设 | 设定 | TB 峰（×2） | V\* 峰（×2） | 判定 | 文档 |
|---|---|---|---|---|---|---|
| S（学生参照 ×2） | 用学生自身分布代替 p⁰ | `reference=student` | 48.40 / 50.12 | 87.96 / 87.96 | 判负：V\* 回到 V0；参照随学生漂移、u→0 | candidate_s_results |
| St（S + tail u=0 ×2） | 排除尾部病理 | `reference_tail_u_zero` | 48.64 / 48.40 | 87.96 / 87.43 | 判负；尾部病理不是主因 | 同上 §5.1 |
| S2（学生底座 + hidebox 教师） | 底座换学生 | `reference=teacher` | — | — | 第 6 步起塌缩到仅 EOS（正反馈增益 1+β） | 同上 §5.2 |
| **IR 路线一**（IRf λ=1 / IRfl2 / IRfl4） | 单次教师前向内部残差 [12,20) 的线性读出做纠正方向 | `teacher_target_mode=internal_residual` | 47.65 / 46.17 / 46.67 | 87.43 / 89.01 / 85.86 | 判负：TV 到 A 量级仍双降 + 格式漂移；区间扫描取消 | ir_results_2026-10-05 |
| 层先验探针（400 题离线） | real 前向某层读出 ≈ p⁰ | 逐层 logit-lens / 全局 block 权重 | — | — | 判负：无一层比 p⁺ 更近 p⁰（ℓ≤20 TV≈1，ℓ=31 0.36）；R² 0.008 | ir_layer_prior_probe_plan §6 |
| 镜像探针（400 题离线） | 全图前向某层读出 ≈ p⁺ | 零参数 | — | — | 判负：无一层比 p_full 更近 p⁺ | full_to_priv_probe §1 |
| 表征/注意力探针（149 题） | 全图前向中层是否盯住 GT 区域 | 8 个 full-attn 层 | — | — | **成立**：中层 lift 6×，但集中度越高差距越大（看了但没看清）；信息在 20–31 层才成偏好 | full_to_priv_probe §3 |
| per-token baseline（离线预检） | u 的跨题一致风格分量可剥离 | b(v)=E[u(v)] | — | — | 判否：held-out 解释力 −6% | opd_aha_mechanism_conclusion §5 |
| batch 信号（作者建议，三种读法） | 从同 batch 其他样本拿 null/特权信号 | 读法 1 = Table 7；读法 2 = 序列级加权（离开框架、已有工作）；读法 3 = 共识当伪特权（TTRL 族） | 共识投票 52.1% vs 单条 47.6%，全错一致组 12.5% | — | 均不在框架内或信号弱；转为 Amis | group_relative_teacher_signal_plan |
| **Amis**（mismatched-crop null ×2） | null 第二张图换成同 batch 另一 prompt 的 crop，抵消风格通道 | `null_mode=mismatch_crop` | 49.14 / 47.90 | 89.53 / 90.58 | **判负**：V\* 只保住 A 增益的 1/3，TB ≤ V0；真实 crop 必带假证据，色块的价值正是"无内容"；漂移形态变为裸选项收尾 | amis_results_2026-10-05 |

**跨线结论**：null 是一个反事实，只存在于"换输入再算一次"里，且其内容必须**不含证据**（Amis 证明真实 crop 做 null 会引入假证据）；残差流不储存"换个输入会说什么"；学生/batch 来源的参照要么随学生漂移、要么退化为对 V0 的加权。省算力的唯一精确解是前缀共享的双分支（未实现）。

## 4. 更早的线（旧仓库，均已关闭）

| 方向 | 结果 | 文档 |
|---|---|---|
| 语言特权（教师看文字提示而非 crop） | focus 措辞优于 json，但线关闭（09-27） | lang-privilege |
| e1 整轮（定位格式训练） | 定位学会了但 JSON 塌 + 作答代价；格式漂移是 reward-free 蒸馏内生 | e1-format-drift |
| 多框变体 | 多框自由未被使用，漂移复发 | multibox |
| OPSA / SA-OPD / EVT-neg | 三变体全崩；advantage 形式 0 胜 4 负；高清 6K 同款崩溃 | 09_07_opsa_verdict、evtneg |
| 官方 9B 复核 | judge 49.63 ≈ base 49.38；零迁移；指纹 OCR +8.8 / Attributes −13.8 | official9b |
| 框尺度轴 | 本质变量是 GT 框尺度非分辨率：V\* 0.08% vs TB 5.19%；hide 在两基准方向相反 | box-scale-axis |

## 5. 机制层面已确立的事实（可直接写进论文分析）

1. **A 学的是压制"没证据时乱说"的冲动**：sup-only 与 A 终点同分；p⁰ 离 p⁺ 0.065、离语言先验 0.23（null 不是语言先验）；u 的质量集中在推理前段的感知描述 token（首 10% 位置 TV 0.127 vs 末 10% 0.019），答案字母位几乎为零。
2. **u 同时携带作答风格**（首 token "The vs C" 0.57→0.99；zoom/cropped 措辞），同一个 β 把两者一起放大，是格式漂移与 OCR 丢分的来源；风格分量在 token 粒度上无法统计剥离。
3. **模型中层已经知道该看哪里**（全图前向对 GT 框 lift 6×），差距来自那里的内容分辨率；反事实只能靠换输入再算，内部读出无法替代。
4. **评测噪声尺度**：同配置 A 四次 TB 差 1.7、V\* 差 1.6；"持平"结论多数在噪声内。

## 6. 流程规则（现行）
- 永远 2459 题 6karmA、seed 42；准确率只报 judge 口径；每方法 ×2；判读报范围不报单点。
- ckpt：训练中仅磁盘 <500G 删最老中间 ckpt；评测完每臂只留 TB 峰 + V\* 峰（`trim_ckpt_peaks.sh`）；10-05 起峰值取全部 ckpt。
- 文档只推 `negative_history`；代码推 `sup`；每次功能改动后 code review；编排器 sbatch 到 standard，不引用 `SLURM_*`。
