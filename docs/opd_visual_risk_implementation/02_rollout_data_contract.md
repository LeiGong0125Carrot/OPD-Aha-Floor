# 02｜Rollout、视图、token 对齐与数据契约

**目标：** 每个公式都能对应一条实际保存的数据；避免 real/null、old/new 和 mask 混淆。
**依赖：** [01](01_scope_and_method_spec.md)。**输出：** 可供 teacher scorer 和 actor updater 共享的 RolloutBatch。

## 1. 索引与形状

用 $i$ 表示 prompt group，$j$ 表示组内独立 rollout，$t$ 表示 response 中的文本预测位置，$v$ 表示词表 token。推导使用 $t=1,\ldots,T$；代码数组从 0 开始。

批处理将 $(i,j)$ 展平成 $B$ 行。标量 token 量统一为 `[B,T_max]`；完整分布为 `[B,T_max,V]`。Image token 位置不进入 $H_t$ 或 $R_t$。

## 2. RolloutBatch 必需字段

| 字段 | 类型/形状 | 获取方式与用途 |
|---|---|---|
| `prompt_group_id` | int/string `[B]` | 同题同输入实例的唯一键，不能只凭问题字符串分组 |
| `rollout_id` | `[B]` | 组内唯一，跨卡不重复 |
| `policy_version` | scalar/hash | 本次实际采样 student 权重版本 |
| `response_ids` | int64 `[B,T]` | 原样保存已生成 token，不二次 tokenize |
| `response_mask` | bool `[B,T]` | 有效输出 token，包含实际采样的终止 token；pad=False |
| `response_length` | int `[B]` | mask.sum，不含 prompt/image token |
| `finish_reason` | `[B]` | EOS/合法 stop、长度截断、基础设施失败必须分开 |
| `prompt_token_ids` | ragged/tensor | 含既定 chat template，不得重新增减指令 |
| `student_image_ref` | 每行输入标识 | 既定 full image 与 processor 配置 |
| `privileged_image_ref` | 每行输入标识 | 当前项目已确定的 crop/hide 构造，不能猜替换 |
| `null_image_ref` | 每行输入标识 | 对 privileged evidence 的既定 mean-RGB 替换 |
| `old_policy_logp` | fp32 `[B,T]` | 采样行为策略对实际 token 的 log-probability，固定 |
| `sampling_spec` | JSON/hash | 温度、top-p/top-k、logits processors、stop rules |
| `processor_spec` | JSON/hash | 图像分辨率、图像 token 布局、tokenizer/template 版本 |

Teacher 评分后增加 `teacher_plus_logp`、`teacher_null_logp`；二者是实际 sampled token 在完整 teacher softmax 中的 logp，不能是 tail bucket 概率。

## 3. 原始模型概率与采样策略概率分开命名

- `student_raw_logp`：通常是温度 1 的完整 student softmax；用于明确指定的 JSD/GU 等分布计算。
- `old_policy_logp` / `new_policy_logp`：必须对应相同的、实际用于 rollout 的行为策略定义；用于 PPO 比值。
- `teacher_plus/null_logp`：同一 teacher scoring 温度、同一文本 prefix 的 real/null 概率；不是 student 行为策略。

**新增工程约束：** 若采样器使用温度、top-p/top-k 或其他 logits processors，不能默认拿未变换的 actor logp 除以截断采样器报告的 old logp。必须由现有 sampler adapter 明确定义并重放相同策略。

第一轮参数未更新时，应逐有效 token 验证 `new_policy_logp ≈ old_policy_logp`。这只能检查实现一致性，不单独证明 surrogate 无偏。若旧框架采样/评分策略本来不一致，必须显式说明并解决，不得为了通过测试悄悄改采样配置。

动态 top-p/top-k 可能改变支持集；数值零概率、mask 与不可微截断的处理需要 sampler 专用适配。本包不提供虚构的通用适配器，不能以 clip 概率代替它。

## 4. Next-token shift 与跨视图对齐

`response_ids[:, t]` 必须由“包含 prompt 和前 $t$ 个 response token、不包含当前 token”的 logits 行评分。标准 causal LM 的 logits 与 label 通常错开一位，但实际索引要由框架 adapter 返回。

Real/null teacher 的文本 response token IDs 必须相同。Student 与 teacher 的 image token 数或 prompt 长度可以不同，不能直接把它们的绝对序列索引当作相同的 response $t$。

要求 adapter 返回 `response_logit_indices`；按 response 相对位置 gather。不要让“第20个总输入 token”混入“第20个回答 token”。

## 5. Mask 与终止

- 第一条 response token 前 $H_1=0$，每条 rollout 独立重置。
- 真正生成的 EOS/stop-token 动作要按既定 tokenizer 记录，不能为方便把 EOS 从 loss 中全剔除。
- EOS 后 pad：cost、return、loss 为零；pad 不得改变 baseline 或平均分母。
- 长度上限截断：只评价已观测前缀，不伪造 EOS，不声称其未来成本为真实的零。
- 基础设施失败：不能按“零冲突、已终止”处理。保留失败记录；固定组训练需重试完整组或跳过完整组，不能只留下低成本 response。
- Packed samples 必须先按 rollout 边界恢复；不允许累计量串到下一条样本。

参考数学代码仅接受 True 连续前缀后接 False 的 response mask；prompt/image mask 已在 adapter 层剔除。

## 6. 独立采样与同题组

复用已有每题多条 rollout，不因使用 future 额外增加采样。LOO 需要每组至少 2 条独立完整样本。共享 prompt 是允许的；克隆同一生成结果、组内 beam competition、按 reward 保留 top-k 不满足同样的统计前提。

若当前配置 n=1，不能静默设置 n=2。应把 baseline 替代或采样预算变更列为训练前待决定项。

## 7. 验收

输入冻结后重复评分一致；初始 ratio≈1；response shift 单 token 手算一致；EOS/pad/变长/跨卡分组可追溯；序列重打包不改变每条的 cost/return。仓库级 GPU smoke test 尚未执行，必须在实际接入时完成。

**来源：** real/null 同 prefix 与 unchanged rollout 来自 [P1, §2.1、§3.1、§3.2]；tensor 字段、EOS、采样策略契约为本计划的显式工程补充 [E]。
