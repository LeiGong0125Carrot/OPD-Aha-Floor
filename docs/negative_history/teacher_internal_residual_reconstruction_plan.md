# 路线一｜单次冻结 Teacher 前向的内部残差重构：算法与实施计划

**版本：** v0.1  
**整理日期：** 2026-10-05  
**文档位置：** `negative_history` / `docs/negative_history/`  
**状态：** 研究设计与待实施工程规格；尚未实现、运行 GPU 兼容性测试或验证性能。

> 本文整理已讨论的路线一：保留冻结 privileged teacher 的最终分布作为 anchor，从同一次正常 teacher forward 中读取内部残差贡献，替换 OPD-Aha 的 real–null 纠正方向。不是 Student-anchor、head-only 训练，也不是 FP-OPD 的扰动投影。
>
> **本次提交仅增加计划文档。** 不修改训练代码，不启动实验，不改写既有结果。本文新提出的配置名、返回字段、文件名和测试名均为实施建议，不应当作已存在的 API。层区间与修正强度尚未确定，不自动继承 A 的 β=4。

## 0. 来源、版本与阅读约定

- 文档基线：`negative_history@913efe104d00e578f0beda4568874b3c66d3b1f7`。[D1]
- 实现核对基线：`sup@187a9529319e6db12216f7319924075ea957f8a7`；后续开发开始前重新核对分支与运行环境，避免覆盖并行修改。[C1–C5]
- 已存在：student rollout、冻结 teacher 评分、real/null 或 S 系列 target 路径、top-100＋tail、JSD、IS correction、梯度累积和优化器更新。[C1–C4]
- 待新增：只读残差采集、实际最终归一化下的 logit 贡献读出、内部重构 target、独立路由与测试。
- 外部接口依据：Hugging Face Qwen3.5、PyTorch hooks/compile、vLLM 与 verl 文档；这些只支持工程接口判断，不证明本项目已兼容通过。[E1–E6]
- 所有数值例子均为合成演算。历史实验数值来自仓库报告；数学推导和新设计在下文明确标出。
- 不把此前失败表述扩大为“所有 null-free 方法必然失败”，也不把内部残差叫作已经识别的 pure visual evidence。

## 1. 研究动机与已知边界

现有 Aha 用一次 teacher-real 与一次 teacher-null 评分，得到额外的条件差分。当前 pair 配置的 null 保留全图、仅清空第二张 crop，因此它主要对应 crop 在全图之外的条件增量。[D1]

目前报告显示：S 与 S-tail 没有恢复 A 的 V* 表现，S2 的最终输出则塌缩为结束符。它们支持停止继续依赖最简 student-reference/student-anchor 外推；不构成对所有替代信号的排除证明。用户已明确本轮不再采用 student anchor。[D1]

**候选假设 H：** 冻结 privileged teacher 的同一次前向包含不止最终分布，还包含实际形成该分布的内部计算增量。对某个事先规定的计算阶段作贡献分解，再有控制地强调其词表相对偏好，可能提供优于普通 privileged OPD 的监督，且不需要 teacher-null 前向。

该假设有三个尚未证明的部分：

1. 某个内部计算阶段的贡献确实有任务价值，而非主要是语言风格或一般置信度变化。
2. 额外强调这个已经进入最终 logits 的贡献，能够改善而不是破坏 teacher target。
3. 读取状态与额外词表投影的实际成本，低于被省掉的 null 前向。

**目标是 null-free 地取得接近 A 的任务效果，不是声称单次前向精确恢复原 real–null 差值。**

## 2. 不变项与明确排除项

| 项目 | 第一版约束 |
|---|---|
| Student 输入 | 原有全图、问题、student 自己生成的 prefix |
| Teacher 输入 | 原有全图＋crop；不是 hidebox、纯文本或额外扰动视图 |
| Teacher 参数 | 冻结；正向 anchor 始终为其最终分布 p⁺ |
| Student 参数 | 沿用当前 A 的可训练范围；不新增 head-only 限制 |
| Rollout | 保留 vLLM，原有采样、模板、长度和权重同步规则 |
| 蒸馏损失 | JSD，α=0.5；沿用现有 IS、mask、聚合与优化器 |
| 新增信息 | 只来自一次 teacher 正常前向中的真实内部状态 |
| 不使用 | Null/无图/扰动参照前向、teacher backward、JVP、候选回答枚举 |
| 不新增 | 可训练读出 probe、新 LM head、表示匹配辅助 loss、PPO/history/future |
| 推理部署 | 训练后的 student 保持原结构；不需要 teacher 或 hook |

“只做一次 teacher forward”按**每个 actor microbatch 的 teacher 序列评分**计数；不表示整个训练 step 只有一次调用，也不包括 student rollout 的自回归计算。

## 3. 与 OPD-Aha 的融合位置

令 h_t 表示问题与 y_<t；令 z⁺_t 为冻结 teacher 在 privileged 输入下的原始 logits，τ 为既有评分温度：

$$
p_t^+=\operatorname{softmax}(z_t^+/\tau),\qquad p_t^S=p_\theta(\cdot\mid I,h_t).
$$

原 Aha：

$$
u_t=\log p_t^+-\log p_t^0,\qquad q_t^A\propto p_t^+\exp(\beta u_t).
$$

新候选：

$$
\boxed{q_t^{\mathrm{int}}(v)\propto p_t^+(v)\exp[\lambda r_t(v)]}
$$

$$
\boxed{\ell_t=\operatorname{JSD}(\operatorname{sg}(q_t^{\mathrm{int}}),p_t^S).}
$$

λ 是新信号的强度，替代原方向的 β，不是同时叠加两个纠正。r 的单位必须与 z⁺/τ 一致：若先计算原始 logit 贡献 c，则 r=c/τ。

保留：teacher anchor、exponential reconstruction、JSD。替换：纠正方向的来源。删除：null 图像构造和 null teacher forward。

### 3.1 Pairwise 解释

$$
\log\frac{q_t(v)}{q_t(w)}=\log\frac{p_t^+(v)}{p_t^+(w)}+\lambda[r_t(v)-r_t(w)].
$$

r 是有符号 score，不是概率。给所有词表项加相同常数不改变重构 target；但不能据其绝对正负直接复用原来的 historical visual-conflict 标签。

### 3.2 与 S/S2 的区别

本方法的 p⁺ 与 r 都来自冻结 teacher，student 分布不参与纠正公式。固定输入、prefix、温度与完整词表时，target 不因 student 更新而漂移。

仍需区分：student 改变 rollout prefix，或改变压缩支持集，会改变实际训练状态/表示方式。这不等于 student 被用作 anchor。

λ=0 或 r 为词表公共常数时，target 回到 p⁺，即普通 privileged OPD，而不是 student 模仿自己。

## 4. r 的定义：实际残差贡献，不是中间层独立预测

### 4.1 层边界约定

用 L 个 decoder blocks，采用零基索引 [0,L)。h⁰ 是进入 block 0 的 residual stream，h^j 是完成前 j 个 blocks 后、尚未经过最终 norm 的状态。

选定连续区间 [a,b)，要求 0≤a<b≤L。该区间的实际总更新是：

$$
\Delta h_t^{a:b}=h_t^b-h_t^a=\sum_{j=a}^{b-1}(h_t^{j+1}-h_t^j).
$$

第一版只实现一个连续区间、完整 block 边界。不分别拆 attention/DeltaNet/MLP，不按未验证的标签指定“某种层就是视觉”。连续区间只需要两个端点，可避免保留全部层。

### 4.2 使用实际最终 RMSNorm 缩放

对已核对的 Qwen3.5 类型，最终输出在理想实数运算下可写为：

$$
s_t=\left(\frac1d\sum_i(h_{t,i}^{L})^2+\epsilon\right)^{-1/2},\quad D_t=\operatorname{diag}(w_{\mathrm{eff}}s_t),
$$

$$
z_t^+=W_{\mathrm{out}}D_t h_t^L+b_{\mathrm{out}}.
$$

定义：

$$
\boxed{r_t=\tau^{-1}W_{\mathrm{out}}D_t(h_t^b-h_t^a).}
$$

外部已核对的 HF Qwen3.5 实现采用 zero-centered RMSNorm，即 w_eff=1+weight；实际训练环境必须确认，不能把通用 RMSNorm 权重公式直接套用。它也不同于 DeltaNet 内部的 gated norm。[E2]

**不得调用 RMSNorm(h^b−h^a) 代替 D_t(h^b−h^a)**：前者使用残差自身的尺度，会改变贡献定义。不得将 post-final-norm 的 h 与 pre-final-norm 的 h 相减。输出 bias 只属于原 anchor，不应再次加入 r。

BF16 cast 与 fused kernels 不满足严格线性可加性。以上是实数域贡献定义；实现须以 FP32 数学参照校验，并记录与实际低精度 logits 的偏差，不许宣称逐比特的因果分解。

### 4.3 为什么这不是随意的 hidden-state→token 映射

这里只使用 teacher 本来就有的 residual 坐标、最终实际归一化和冻结 LM head，不把中间状态当成一个经过独立校准的词表模型，也不训练新的映射。[E7]

但它只是实际 forward 的直接贡献分解，不等于删除这些 blocks 后重跑模型的效果。后续非线性响应、归一化变化和视觉/语言交互没有被因果分离。

尤其应诚实表述：r 的贡献已经存在于 z⁺。新 target 相当于 z⁺/τ+λr，即额外重加权这部分贡献；没有获得新的图像内容。是否值得重复强调，正是实验假设。

### 4.4 必须避免的退化选择

不要令信号直接等于整个最终归一化 hidden state 的读出；在无 bias 等简化条件下，这会退化为对 teacher logits 的温度锐化。若选取过宽区间接近这种行为，也必须与一般锐化区分，而不是称为新的视觉归因。

## 5. 完整数值例子与极限条件

教学词表为 striped、diamond、plain；评分温度为 1：

| Token | p⁺ | r |
|---|---:|---:|
| striped | 0.30 | +0.35 |
| diamond | 0.60 | −0.25 |
| plain | 0.10 | 0 |

取 λ=2 **只为演算**。未归一化权重为 0.604125812、0.363918396、0.1，总和 1.068044208；归一化：

$$
q=[0.565637459,\ 0.340733458,\ 0.093629083].
$$

相对偏好：

$$
q(\texttt{striped})/q(\texttt{diamond})=0.5e^{2(0.35+0.25)}\approx1.660058.
$$

Teacher 原本偏向 diamond，新 target 偏向 striped。并不意味着所选层在真实数据上必然正确。

如 r(v)−r(w)≤0，正 λ 不会额外促进 v 相对 w；若它恰好增强错误候选，target 也可能变坏。冻结 anchor 只避免 S2 那种特定自我参照反馈，不保证稳定或正确。

## 6. 词表支持集与 tail：必须在实现前规定的部分

前面讨论只固定了全词表公式，没有确定新信号如何定义 tail。**以下是本计划新增的工程规格建议**，不能误称现有 Aha 代码已采用该顺序。

### 6.1 建议主规格：先重构完整词表，再压缩到现有 K＋tail

令 K_t 为当前 student top-100 的 token ID 集合。在所有词表项上先定义：

$$
s_t(v)=z_t^+(v)/\tau+\lambda r_t(v),\quad q_t^V=\operatorname{softmax}(s_t).
$$

传入 JSD 的 target 为：

$$
\bar q_t(v)=q_t^V(v)\;(v\in K_t),\qquad \bar q_t(\mathrm{tail})=\sum_{v\notin K_t}q_t^V(v).
$$

Student 与普通 teacher 分布也在同一 K_t 上保留显式概率和补集总质量。Student 梯度仍流经其 tail 质量，不增加一个可生成的 tail token。

旧 Aha/S 路径先将 real/reference 概率合并成 tail，再重构。[C3] 本主规格与之顺序不同，因为新 r 是词表 logit 贡献而不是两个已经合并的概率差。必须在实验记录中说明；保持默认 legacy A 原路径不变，不能称整个实现只改了一行 u。

### 6.2 精确的 tail 计算与可选等价表示

使用 logsumexp，直接累积补集质量，避免 1−sum(head) 在极小 tail 下的消减误差：

$$
\log\bar q_t(\mathrm{tail})=\operatorname{LSE}_{v\notin K_t}s_t(v)-\operatorname{LSE}_{v\in V}s_t(v).
$$

若必须写成 tail 纠正，它的等价项在 λ≠0 时为：

$$
r_{t,\mathrm{tail}}=\frac1\lambda\log\frac{\sum_{v\notin K_t}p_t^+(v)e^{\lambda r_t(v)}}{\sum_{v\notin K_t}p_t^+(v)}.
$$

这不是尾部 r 的算术平均，也不是把某个真实 token 的 r 复制进 tail。实际实现优先直接累积 q_tail，不计算可能小数相除的等价式。

不允许静默设置 r_tail=0、固定 q_tail=p⁺_tail、去掉 tail 或切换成 head-only renormalization。这些都是不同算法；若成本迫使改规格，应先说明并单列实验。

### 6.3 高效实现，不长期保存 full-vocabulary 张量

按 response 行分块，必要时再按词表分块：复用正常 teacher logits，计算 r_chunk，形成 s_chunk，累积全词表/尾部 LSE，并记录 K_t 项。最终只返回 [B,T,K+1] target 与少量统计。

若正常 teacher forward 的包装只暴露 top-k，不能仅凭 top-k 复原精确 tail；应在原始 logits 被丢弃前完成新 target 计算。若必须重做一次 LM-head 投影取原 logits，记录额外成本，不能记作零开销。

λ=0 的 baseline 应直接走普通 teacher 分布的既有 tail 处理，以便回归。λ≠0 时若沿用旧 1e−7 概率钳位，需明确记录数值政策和偏差，不能把数值 floor 当成隐形算法模块。

## 7. 训练循环与梯度路径

1. vLLM 用当前 student 权重生成 full-image response，保留原始 response token IDs。
2. Actor 正常 student forward，得到有梯度的预测与 top-k IDs。
3. 同一 microbatch 执行一次冻结 privileged teacher forward；同步采集选定 residual 端点和最终 pre-norm 状态。
4. 在 no_grad 范围内读出 r、构造 q，再聚合为同一 K＋tail 支持。
5. 使用现有 JSD α=0.5、IS、mask 和归一化路径，累积 student 梯度。
6. 沿用既有 optimizer step、checkpoint 与 student→vLLM 权重同步；teacher 不更新。

Target 的 r、q、选定支持集 ID 都不反传。Teacher 参数不进入 student 优化器。Student 主干正常训练，不冻结为“共享主干”方案。

保持现有 correction：`student_log_probs/old_log_probs` 的 detached IS 比值与 batch 中的 `rollout_is_weights` 不属于新视觉方向，不得为新方法重写其语义。[C1,C3]

不得把各 microbatch 的 token mean 草率等同于完整分布式 batch 的全局平均；沿用当前分母与缩放，并以不等长度、不同切分的测试明确实际行为。新方法只接入 target，不暗中修复或更改原优化口径。

## 8. vLLM / verl 改动边界

### 8.1 vLLM 核心：第一版不改

当前仓库将 `actor_rollout_ref.rollout.name` 设为 vLLM；teacher 评分实际发生在 actor 的 PyTorch `_forward_micro_batch()`。官方 vLLM/verl 文档也区分 rollout 引擎与训练 worker。[C1,C4,E4,E5]

因此，不在 vLLM generation API 新增逐层 hidden-state 返回，不导出 attention，不改变 paged KV cache 或采样器，不把 teacher 部署到额外 vLLM 服务。不要为本计划切换 speculative hidden-state extraction 接口。

**vLLM 方向实际要做的是集成验收而非改内核：** 原 rollout engine/采样参数/权重同步继续工作；新增 teacher buffer 在 rollout 前释放；checkpoint 不混入 hook 状态；更新后的 student 可以直接用原评测脚本加载。

### 8.2 Teacher 评分侧：需要扩展

在独立冻结 teacher 实例上读取完整 block 边界，不读取 student 的隐藏状态作为 target 来源。推荐用局部只读 hook 原型验证，再为生产训练提供显式 capture adapter。[E1,E3]

采集量：h^a、h^b、h^L（若 b=L 可复用）；最终 norm 的参数/epsilon；原最终 logits 或其分块接口。只保留用来预测有效 response token 的位置。

不要依赖 `output_hidden_states` 的 tuple 最后一项一定是 pre-final-norm；以运行版本的实际 forward 和 hook 位置确认。Qwen3.5 的 linear-attention/full-attention blocks 都有完整 residual 边界，但其内部含义不能简单标为“视觉/语言”。[E2]

### 8.3 Hook 生命周期

Hook 只读，返回 None；不进行 in-place 修改、随机采样或改变 RNG。使用 per-forward collector，不使用全局可变缓存；采集后 detach 并在可能被复用的视图上 clone。异常路径也必须清空缓存和恢复开关。Teacher 参数冻结不等于其 dropout 自动关闭；记录实际 eval/dropout 状态，并与匹配 baseline 保持一致，不静默改变原 A 的模型模式。

若 teacher 与 actor 指向同一个可训练对象，拒绝运行 internal 模式，不能因 `self.teacher_module or self.actor_module` 的兜底而悄悄使用 student。

第一版训练不长期开启全部层的 `output_hidden_states=True`。只允许小 batch 正确性检查使用全部状态，并将额外内存计入检查说明。

## 9. 位置对齐：必须预测 y_t，而不是在看完 y_t 后读取状态

Teacher 和 student 的 prompt 长度不同；应复用现有 `teacher_response_start_idx` 和 `_select_response_positions()` 语义，不能直接沿用 student 的绝对位置。[C1]

假设 teacher prompt 有 P 个 token，完整输入为 prompt,y_0,...：预测 y_0 的状态位于 P−1；预测 y_t 的状态位于 P+t−1。采集到 y_t 自己所在的位置会泄漏该 token，造成监督错位。

Remove-padding 路径需要相同的 packed index/恢复顺序/样本边界；禁止把一条样本的最后 prompt 与下一条样本拼成预测上下文。Mask 为零的 padding 不参与重构、统计或梯度。

第一版保留当前 SP1。SP>1 若尚未实现 residual 的切分/汇集映射，应 fail-fast，而不是以近似张量继续训练。仍需实际验证多 GPU DP/FSDP；SP1 不等于单 GPU。

## 10. FSDP、融合内核与编译适配

| 项目 | 第一版要求 |
|---|---|
| FSDP 参数分片 | 额外 W_out 投影在正确的参数 materialization 范围内完成；不得在 reshard 后直接取本地 shard 冒充完整词表 |
| LM-head 切块 | 对应全局 vocab ID；若词表分片，LSE 必须全局正确；先支持现有 backend，不假设任意 TP 已兼容 |
| FlashAttention | 只采集 block 边界，不要求 attention 矩阵；不默认退回 eager attention |
| Fused model forward | 核对 patched forward 是否跳过 module hook 或改变输出布局；不支持则明确报错/单列已说明的适配，不默默换公式 |
| torch.compile | 在实际运行前注册采集接口；确认 hook 触发次数与图捕获；若发生 graph break，记录代价。[E6] |
| Gradient checkpointing | Student 既有设置不变；teacher 无梯度采集不得在重算时残留或重复累计 |
| CPU offload | norm/head 参数与 residual 使用正确设备；避免每行 CPU↔GPU 往返 |
| 混合精度 | 原 teacher anchor 用原前向；residual 差、RMS 缩放及 LSE 用明确的精度策略；记录 BF16 对 FP32 参照误差 |

不得通过全程 unshard 整个 teacher 来宣称省显存；允许先在小模型/单卡验证公式，但不能将其性能结论外推到正式 FSDP 环境。

## 11. 拟议文件改动清单

本表是后续实施清单；**此次文档提交不会修改这些代码**。

| 文件 | 计划职责 |
|---|---|
| `verl/workers/config/actor.py` | 核对并扩展 SelfDistillationConfig；新增独立 target mode 与 internal 参数；配置校验 |
| `verl/trainer/config/actor/actor.yaml` | 默认保持 legacy；internal 字段默认 disabled/未配置，不污染原 A/S 路径 |
| `verl/trainer/ppo/ray_trainer.py` | internal 模式只构造原 teacher-real 输入，跳过 null builder；manifest 记录方法与运行参数 |
| `verl/workers/actor/dp_actor.py` | Teacher 专用 capture 生命周期；残差投影；返回预构造 target；保留原 student/IS/optimizer 流程 |
| `verl/trainer/ppo/core_algos.py` | 增加可选 detached target 输入；internal 路径直接用它计算既有 JSD，不再走 reference subtraction/重构二次处理 |
| `verl/utils/teacher_residual_capture.py`（新） | 模型结构识别、选定边界采集、pre-norm 校验；按模型版本建立明确 adapter |
| `verl/utils/teacher_residual_target.py`（新） | 数学核：贡献读出、温度一致性、chunked target/LSE、支持集聚合 |
| `scripts/test_internal_residual.py`（新） | 独立公式、梯度、路由、tail、chunk、mask、指标和 legacy 回归测试 |
| `scripts/train_pair_internal_residual.sh`（新） | 包装既有 pair 训练启动方式；强制真实 privileged 数据、禁 null，要求显式 a,b,λ；不含机器路径 |
| vLLM 包与 rollout 实现 | 不计划修改；仅做同步、内存与部署兼容性测试 |

项目中的 Transformers monkey patch 位置需实施前按实际安装源文件定位，不能猜测一个模块路径后直接修改 site-packages。适配代码应版本可控、可关闭。

## 12. 配置建议与互斥保护

推荐新增独立 selector，避免继续用 `counterfactual_null_mode=mean_color` 表示“内部残差开关”。以下**不是已可运行配置**：

```yaml
actor_rollout_ref:
  rollout:
    name: vllm                         # 保留
  actor:
    policy_loss:
      loss_mode: vopd                  # 保留
    self_distillation:
      teacher_target_mode: internal_residual  # 新；缺省 legacy
      teacher_regularization: frozen
      teacher_update_rate: 0.0
      counterfactual_null_mode: null   # internal 走独立路由
      counterfactual_reference: "null" # legacy 字符串占位；internal 不消费此值
      alpha: 0.5
      distillation_topk: 100
      distillation_add_tail: true
      teacher_internal:
        start_block: null             # 正式训练必须显式填写 a
        end_block_exclusive: null     # 正式训练必须显式填写 b
        strength: null                # 正式训练必须显式填写 λ
        tail_policy: full_vocab_then_coarsen
        capture_backend: selective
        response_chunk_size: null     # 按显存确定，属于执行参数
```

缺省 `teacher_target_mode=legacy` 必须逐比特保留当前分支的老路径（同一确定性测试环境）；不得将 legacy 的 null/S/S2 行为一并重构后称为无变化。

Internal 模式拒绝：非冻结 teacher、teacher 与 student 共享可训练实例、S/S2 参照、null 张量、floor/suppression/tanh/gamma/history/future/shuffle 等额外 target 干预。层端点、强度、norm adapter 或支持集规格缺失时 fail-fast。第一版要求 λ 有限且非负；负号翻转只可作为另行声明的研究对照。

内部强度 λ=0 仅用于 baseline/回归，不自动填成正式训练默认。旧 `counterfactual_extrapolation_beta` 在 internal 模式不生效，manifest 应明确标记，避免用户以为 β=4 仍在使用。

继续使用已有 `VOPD_FORBID_NULL=1`，并验证它在所有 worker 中生效。任何额外 null 图像构造或模型评分都应报错，而不是只把日志数置零。

## 13. 拟议数据接口

Teacher forward 的新增产物建议为：

| 字段 | 形状/类型 | 用途 |
|---|---|---|
| `teacher_internal_target_log_probs` | [B,T,K+1]，detached | 已完成重构与压缩的 q |
| `teacher_real_topk_log_probs` | 沿用原接口 | p⁺ anchor 的监督参照/诊断，不能被 q 覆盖丢失 |
| `teacher_internal_signal_topk` | 可选 [B,T,K]，detached | 调试与统计；不要求长期存盘 |
| `teacher_internal_stats` | 分子/分母与明确 worst-case 标量 | 指标汇总 |
| capture manifest | 模块名、端点、norm、dtype、温度、chunk 配置 | 复现与错误追踪 |

不得把 r 填进 `teacher_null_log_probs` 冒充 log-probability。r 不是分布，不能套 add_tail(logp) 推导它的尾部。

Loss 检查 target 的最后维、有限性、归一化、位置、supports IDs 与 detach 状态；避免已经 K+1 的 target 再次 add_tail。应单独保留原 teacher 分布，正确计算 TV(q,p⁺)，而不是误算成 target 与自身的零差。

## 14. 核心执行伪代码（仅表示控制流程）

```text
responses = existing_vllm_rollout(full_images, questions)
for actor_microbatch in existing_update_loop:
    student = normal_student_forward(actor_microbatch)      # 有梯度
    K = student.topk_indices
    with no_grad(), teacher_only_capture(a, b):
        teacher = normal_privileged_teacher_forward(same_responses)
        endpoints, h_final_pre_norm = capture.take_response_states()
        delta = endpoints.after - endpoints.before
        r = frozen_head_readout_with_actual_final_scale(delta) / score_temperature
        q = reconstruct_full_vocab_then_coarsen(teacher.logits, r, lambda, K, score_temperature)
    loss = existing_jsd_and_IS_with_precomputed_target(student, stopgrad(q))
    existing_microbatch_scaled_backward(loss)
existing_optimizer_step_and_student_weight_sync()
```

实际程序可在 teacher forward/head 的参数可用范围内流式完成 q，不能机械地把伪代码末端放到 FSDP reshard 之后。vLLM rollout 不携带这些 teacher 中间张量。

## 15. 验证矩阵与验收标准

以下测试是计划，不代表本次已运行。

### 15.1 CPU 独立数学与单元测试

| 测试 | 必须检查 |
|---|---|
| 三词例子 | §5 target 数值、pairwise ratio 与独立公式一致 |
| λ=0 / 常数 r | 回到普通 privileged OPD；loss 与 student 梯度一致 |
| 已知残差分解 | 合成 residual 网络下端点差、逐 block 求和一致 |
| RMSNorm | 真实最终 scale 与错误 RMSNorm(delta) 可被测试区分；zero-centered 权重正确 |
| 温度 | τ≠1 时 anchor 和 r 同步缩放，避免强度被多除或少除 τ |
| Bias | 原 LM-head bias 只计一次，不进入 residual contribution |
| Tail/chunk | full target 后合并与流式实现一致；覆盖极小 tail、K=V、K<V、不同 chunk |
| 对齐 | 首 response、EOS、padding、异长 prompt、packed 样本顺序与边界 |
| 梯度 | q/r/teacher 无梯度；student 有梯度；IS 与旧 target 路径一致 |
| 互斥 | 非冻结、S2、null 输入、不支持 SP、遗漏端点均报错 |
| 回归 | 新 selector 缺省/legacy 对原 A、S、St、S2 既有 loss/grad/metrics 不变 |
| 指标 | 不等 microbatch 长度和多级 reduction 下，与人工全局计数一致 |

数学 FP64 小张量建议 atol/rtol=1e−8；FP32 建议 1e−5，均作为测试起始规范。真实 BF16 logits 的可接受误差必须用同一 kernel 的参照测定后预先登记，不可失败后无限放宽。

### 15.2 一次小 batch 的真实模型正确性检查

- 同一输入、同一模型模式、同一确定性条件下，开/关只读采集不改变原始 teacher logits。
- 用 h^L、真实 norm 与 LM head 重建最终 logits，并比较范数、逐项误差和 top-k 对齐。
- 端点 hook 触发一次，collector 不混入 student forward；不增加任何 teacher 参照/扰动前向。
- teacher hidden states 没有未来 token 泄漏；packed 与 unpacked 的对应位置输出相符。
- 允许在这个小检查中额外跑同一真实图像的比较，不运行 null，不用其结果选择最优层或预测 benchmark 性能。

### 15.3 两步端到端 smoke test

使用实际 HF/FSDP/SP1/vLLM 组合检查：正常 rollout → 单次 teacher 评分 → 非零 student 梯度 → optimizer step → vLLM 同步后可继续生成。验证 teacher 参数不变、null 前向计数为零、hook buffer 及时释放、checkpoint 可正常加载。

这只是工程验收，不能据短程 loss 下降或正常生成就宣称视觉方法有效。正式训练另行确认。

## 16. 诊断指标：沿用已经发现的计数式原则

历史报告已经定位 `max` 子串触发最大值归约的问题，并记录了指标修订。[D1] 新字段不依赖名字推断聚合，应记录所有有效位置的分子和分母；跨 microbatch/rank/time 使用匹配权重的全局求和。若框架内部 mean-reduce，必须证明分子与分母经过完全相同的权重流程，不能仅凭名称认定正确。

建议必需统计：

| 统计 | 目的 |
|---|---|
| `internal_active_token_count` | 有效位置分母 |
| r 的绝对值和、平方和；显式报告统计支持 | 检查信号尺度，避免混淆全词表均匀与概率加权 |
| TV(q,p⁺) 和、entropy(q) 和、top1(q) 和 | 检查实际 target 位移与尖锐度 |
| q_tail 质量和、tail-top1 事件计数 | 区分明确 token 与集合质量 |
| stop-set 的 p⁺/q 质量和、首 token 停止比例 | 监测 S2 曾暴露的控制风险，不主动修改停止 token 的目标 |
| teacher-real/null 调用次数与时间 | 证明新的信号确实只用正常 teacher forward |
| capture/readout/target_compute 时间、峰值显存 | 核算额外成本，不与主干时间重复相加 |
| legacy IS/raw JSD/weighted JSD | 保留原训练解释 |

最坏 microbatch 值使用显式 MAX 类型并单独命名，不能当成全局均值。概率分位数须使用合法的汇总/抽样方案，不能把各 microbatch p90 再平均后称为全局 p90。

## 17. 实验计划：先验证新增监督，再追求 A 的水平

### 17.1 不自动填入的研究选择

正式运行前登记：a,b、λ、norm adapter、tail policy、精度、评分温度及执行 chunk。此前 λ=2 只是教学例子，A 的 β=4 也不自动适用。层选择不能通过 TB/V* 测试集反复选最好者。

优先先确定一个可解释、结构明确的连续区间；若需要探索，用另行声明的开发集与有限预算。当前没有证据确定“最后几层最视觉化”，因此本计划不伪造最佳区间或强度。

### 17.2 最小实验集合

| 条件 | 回答的问题 |
|---|---|
| 匹配 pair/训练配置的普通 privileged OPD | 去掉 null 后的稳定输出监督参照 |
| Internal route，λ=0 | 实现退回上述参照是否正确；不需另跑完整训练重复回归结论 |
| Internal route，一个预登记区间与强度，按既有两次运行口径 | 内部纠正是否真的提供新增训练收益 |
| 原 A，使用有效历史记录或经批准匹配复现 | 效果与计算目标，不更改其配方 |

核心方向有效后才考虑一个 layer-contrast/teacher-sharpening 对照或错位内部信号对照，用于区分一般锐化与该内部结构的价值；不在第一轮堆 history、future 或新的 gate。

同一工程/优化配置不代表全部目标数学相同：本计划已单列 target 重构顺序差异。旧 V0 hide 数值只能作历史参照，不能冒充匹配 pair 的单变量消融。[D1]

### 17.3 固定既有评测口径，单列限制

沿用报告中的 2459 题高清 6karmA、batch 48、n2、lr 2e−6、51 步、seed 42、3 GPU/SP1 等基础设置，以实际 launcher/manifest 确认为准。[D1] 两次运行同 seed 不是两个不同 seed；需如实标明重复方式。

TB 用既有 TreeVGR HF greedy 流程，V* 用既有 OPD-Aha infer 流程，使用 gpt-oss-120b judge，去重并核对样本计数。[D1]

主要报告预定 step 30/40/50、每次峰值、共同 checkpoint 结果与两次变化。初期筛选仍用两次 TB 峰值≥50.5 且两次 V* 峰值≥93.5；不同 benchmark 独立峰值不等于同一 checkpoint 同时达标，也不是统计等效性证明。[D1]

记录 OCR、Attributes、Ordering、Spatial Containment、Comparison 等类别和生成长度/结束行为。历史观察仅用于预先定义检查内容，不预先写成“本方法必然改善关系推理”。

## 18. 成本账与失败处理

对比固定硬件、microbatch、序列长度/视觉 token、精度、warmup 后的耗时。新的成本变化是：

$$
\Delta t=-t_{\mathrm{null}}+t_{\mathrm{capture}}+t_{\mathrm{extra\ readout}}+t_{\mathrm{new\ target}}+t_{\mathrm{extra\ communication}}.
$$

还要报告实际端到端 wall time、GPU-hours 和峰值显存。旧 S 与 Ahf 的节点差异不能用作本方案速度证据。[D1]

| 观察 | 处理 |
|---|---|
| 开采集就改变 anchor logits | 先修实现；不训练 |
| block/norm/response 读出不一致 | 明确版本/边界/精度问题；不靠调 λ 掩盖 |
| 实际比 A 更慢或显存不可接受 | 不宣称效率成功；优化采集/分块后重测，不暗中改 tail 数学 |
| target 非有限或归一化失败 | 直接报错；不以 nan_to_num 静默制造新监督 |
| 空输出、异常提前结束或循环 | 保留短程日志与 checkpoint，暂停该配置；不通过修改 min-length 掩盖 |
| 内部监督正常但性能只等于普通 OPD | 暂未证明新增信息有价值；不自动继续加 history/future |
| 接近 A 但层选择缺乏视觉证据 | 可报告有效内部重构，不声称 pure visual evidence 或已足够 novelty |

## 19. 里程碑与交付物

| 阶段 | 交付 | 进入下一阶段的条件 |
|---|---|---|
| M0：规格冻结 | 运行版本清单、真实模块图、a/b/λ/tail 决策记录 | 用户确认正式试验参数；不自动开跑 |
| M1：数学核 | 独立 target/readout 实现和 CPU 测试 | 数值、梯度、tail、极限条件通过 |
| M2：Teacher 采集 | 模型 adapter、只读 hook/显式返回、生命周期测试 | Anchor 无扰动、位置对齐、单次前向确认 |
| M3：训练侧接入 | selector、null 跳过、loss 接口、legacy 回归 | 旧路径不变，新路由互斥完整 |
| M4：真实环境 smoke | FSDP/SP1/vLLM 两步记录与成本表 | 无泄漏/NaN，teacher 冻结，权重同步和加载通过 |
| M5：直接训练与评测 | 两次运行 manifest、逐题 judge、结果表 | 按预定口径判断，不将计划写成成功报告 |

此文档只完成设计整理；M1–M5 尚未执行。实际代码开发应按新的实施提交管理，不在文档分支悄悄修改 sup/main。

## 20. 当前待确认项

1. 选定哪个连续 block 区间 [a,b)？现有讨论未给出证据支持的具体层号。
2. 首轮 λ 取多少？不得将教学值或旧 β 无说明地当成已确定值。
3. 接受 §6 的 full-vocabulary reconstruction→K＋tail 规格吗？若需更廉价近似，先明确改了什么。
4. 服务器实际 Transformers/PyTorch/vLLM、模型 revision、fused/compile/FSDP 组合是什么？本轮未进入服务器验证。

这些是正式训练的决策门槛，不影响现在提交计划。无需先训练 probe，也无需用 null 分支给层选择打标签。

## 21. 来源索引

### 仓库来源（固定版本）

- [D1] [去 visual-null 线总结](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/913efe104d00e578f0beda4568874b3c66d3b1f7/docs/negative_history/null_free_line_summary_2026-10-04.md)：既有设置、结果与指标问题。本文仅继承明确记录，不将其强因果解释当作已证明事实。
- [D2] [Candidate S 计划](candidate_s_null_free_implementation_plan.md)：历史工程背景，不是本方法定义。
- [C1] [dp_actor.py](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/187a9529319e6db12216f7319924075ea957f8a7/verl/workers/actor/dp_actor.py)：teacher/student forward、response 对齐、no_grad、IS 传递与优化器。
- [C2] [ray_trainer.py](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/187a9529319e6db12216f7319924075ea957f8a7/verl/trainer/ppo/ray_trainer.py)：teacher 输入、null 构造、VOPD_FORBID_NULL。
- [C3] [core_algos.py](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/187a9529319e6db12216f7319924075ea957f8a7/verl/trainer/ppo/core_algos.py)：既有支持集、重构与损失。
- [C4] [run_visual_counterfactual_unit.sh](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/187a9529319e6db12216f7319924075ea957f8a7/scripts/run_visual_counterfactual_unit.sh)：既有 rollout/训练配置；正式实验仍需解析上层 pair launcher。
- [C5] [AGENTS.md](https://github.com/LeiGong0125Carrot/OPD-Aha-Floor/blob/913efe104d00e578f0beda4568874b3c66d3b1f7/AGENTS.md)：保持默认配方，不提交权重、数据、日志、缓存、凭证与机器专用路径。

### 外部接口与方法背景（2026-10-05 核对）

- [E1] [HF Qwen3.5 文档](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)：hidden-state 返回接口。实际安装版本必须另行固定。
- [E2] [HF Qwen3.5 当前实现](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)：zero-centered RMSNorm、decoder residual 边界、最终 norm。main 是可变链接，不替代运行源码哈希。
- [E3] [PyTorch Module hooks](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html)：只读采集的框架接口；不是未经测试的 FSDP/compile 兼容性保证。
- [E4] [vLLM RLHF 文档](https://docs.vllm.ai/en/latest/training/rlhf/)：vLLM 负责 completions/rollouts 与权重同步集成。
- [E5] [verl Engine Workers](https://verl.readthedocs.io/en/latest/workers/engine_workers.html)：训练 worker 与 rollout 的职责区分；项目保留自己的 legacy actor，不因此迁移 backend。
- [E6] [PyTorch NNModule / compile](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_nn_module.html)：hooks 与编译行为需要验证。
- [E7] [Logit Prisms](https://neuralblog.github.io/logit-prisms/)：残差输出贡献分解的相关思想；不是本 VLM OPD 方法的性能证据。
- [E8] 用户提供的 FP-OPD PDF，§3.3 / §3.5：其方法是额外扰动前向、student-anchor 和 reverse KL；本计划没有继承其投影、probe 或损失。不能把 FP-OPD 的结果写成本计划的验证。

**一句话总结：保留冻结 teacher 的最终监督和原有 JSD 训练，只把昂贵的跨视图差分，替换为同一次 teacher 计算内部、经过原模型读出路径定义的贡献；先证明读出正确与计算划算，再检验它是否真的提供有价值的视觉监督。**
