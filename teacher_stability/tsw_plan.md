# 教师稳定性加权蒸馏（TSW）：动机与实现计划

**状态：** 计划，尚未实现
**日期：** 2026-10-23
**设置：** 官方 Vision-OPD-6K（off6k）。学生 = 带红框全图 + "Only focus on the objects inside the red bounding box"；教师 = 同文本、图换成 GT 裁剪图（`bbox_images`），冻结 base Qwen3.5-4B；损失 = top-100 + 尾桶的 JSD（α=0.5）。

## 1. 一句话

按"教师在这道题上自己答得稳不稳"给每道题的蒸馏损失加权：教师不稳定的题降权，学生照常更新。教师冻结，所以这个权重是每道题的静态属性，训练前查表即可，零额外训练成本，不用 GT。

## 2. 动机（已有离线证据）

数据：off6k 训练集全部 6241 题（`data/train.parquet`），每题 base 模型 8 次采样。学生视图和教师视图各一份（vLLM，temperature 1.0，seed 42，max_tokens 512）。
- 学生：`eval/probe_hidden/base_full_1020/gen.json`
- 教师：`eval/probe_hidden/teacher_base_1021/gen.json`

教师稳定性 s = 8 次采样中多数字母的占比（未解析的样本算作不一致）。下表的正确率只作诊断，GT 不进入权重。

### 2.1 教师稳定性 ≈ 教师对这道题的可靠性

| 教师 s | 题数 | 教师正确率 | 学生正确率 | 教师 − 学生 | 学生比教师准的题 | GT 在教师前 2 个字母中 | 教师未解析 |
|---|---|---|---|---|---|---|---|
| 8/8 | 1753 | 0.921 | 0.549 | +0.37 | 3% | 92% | 0% |
| 6–7/8 | 2585 | 0.676 | 0.392 | +0.28 | 14% | 91% | 2% |
| 4–5/8 | 1655 | 0.430 | 0.305 | +0.13 | 25% | 86% | 4% |
| ≤3/8 | 248 | 0.291 | 0.261 | +0.03 | 31% | 77% | 15% |

- 教师越不稳，题越难：学生正确率也跟着下降。
- 同时教师相对学生的优势也在消失：到 ≤3/8 这一档，裁剪图的特权信息基本不起作用（裁剪图丢掉上下文、区域太小、题目有歧义或标注错误）。
- 相比之下，学生自己的稳定性不可靠：学生 8/8 一致的 871 题里，48% 是 8 次全错（笃定地看错）。所以选题信号要用教师侧，而不是学生侧。

### 2.2 训练收益与退步集中在哪里（`select_1022`，a83c9ff，6241 题）

训练效应 = off6k_PBDkF1n2 step 70 的 8 次采样正确率，减去 base 后 4 次的正确率（零假设）。表中每格分别为：学生不确定 / 中等 / 确定（按探针分三档）。

| 教师桶 | 效应 | 大幅退步（训练后 vs 零假设） |
|---|---|---|
| 教师一致 4/4 | +0.331 / +0.246 / +0.156 | 合计 43，零假设 329，远低于零假设 |
| 教师多数 3/4 | +0.149 / +0.127 / +0.101 | — |
| 教师分歧 ≤2/4 | +0.075 / +0.107 / +0.141 | 学生确定的格子 54 vs 40、17 vs 14，达到或超过零假设 |

教师一致的题收益大、退步少；教师不一致的题收益小，而且退步达到甚至超过零假设。

### 2.3 机制解释

OPD 用的是教师在**学生前缀**上的逐 token 分布，而不是教师自己的多数票。分叉探针（10-17）显示，教师被强制接在学生前缀后时会跟着前缀走（跟随率 0.51–0.84）。教师在两个选项之间摇摆时（6–7/8 和 4–5/8 两档平均只有 2.0–2.6 个候选字母），它在作答位置大概率追认学生已经走的方向。结果：教师信号对这类题没有纠正作用，噪声却照样回传。降低这类题的权重，相当于减少"追认式"的噪声梯度。

### 2.4 为什么不需要学生侧信号

- 学生稳定性随训练变化，必须在线估计：rollout_n=2 是否一致太弱（AUC 0.583）；hidden 探针（AUC 0.897）需要随训练重新校准。
- 学生稳定不等于对（见 2.1）。
- 用户决定（10-23）：只估计教师稳定性，教师不确定的题降权，学生正常更新。

## 3. 方法

### 3.1 权重

对第 i 题：`s_i` = 教师 8 次采样多数字母占比，取值 {1/8, …, 1}。全集均值 0.786，分布如下：

| 多数字母次数 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 题数 | 1 | 32 | 215 | 721 | 934 | 1146 | 1439 | 1753 |

- 主设定：`w_i = s_i`，连续值，不设硬阈值，不丢数据。
- 归一化（必须）：每个全局 batch 内按 token 加权归一，使 `Σ_tokens w_i / Σ_tokens 1 = 1`，即 `w̃_i = w_i · N_tok / Σ_j w_j · n_tok,j`。这样 batch 的总梯度强度不变，只在题之间重新分配，避免和"整体学习率下降约 21%"混杂（规则：不做"A + 常数剂量"对照）。
- 同一道题的 n=2 条 rollout 共享同一个权重。

### 3.2 对照臂（必须）

**TSW-shuf**：同一组 `s_i` 值，用固定 seed（42）在题之间随机打乱后再按 3.1 使用。权重的分布和方差与主臂完全相同，只是和"教师稳定性"不再对应。用来排除"收益来自权重方差或扰动幅度"（时间信用臂就是被错位对照打掉的）。

### 3.3 不做的事

- 不在 token 级别门控。之前逐 token 门控（TGNVMT）判负过，本方法是题级、静态的。
- 不用 GT，也不用学生侧信号。
- 不改教师目标本身（不换成教师多数票分布），只改权重。

## 4. 实现

仓库 `OPD-Aha-sup`（`sup` 分支）。

1. **生成权重数据**：新脚本 `scripts/build_tsw_parquet.py`
   - 读 `data/train.parquet` 和 `teacher_base_1021/gen.json`（按行号 `row` 对齐，6241/6241 全覆盖，已核对）。
   - 写 `data/train_tsw.parquet`：原样拷贝，`extra_info` 增加 `tsw_s`（原始 s）、`tsw_s_shuf`（打乱版）、`index`（行号）。
   - 断言：行数相等；prompt、images、bbox_images 逐行不变；`tsw_s ∈ (0, 1]`；打乱版是原值的一个排列。
   - 现状：`extra_info` 没有 `index`，数据集读到的 index 全是 0，所以必须写出新的 parquet。
2. **配置**：`SelfDistillationConfig` 和 `actor.yaml` 新增：
   - `tsw_enable: bool = False`
   - `tsw_key: str = "tsw_s"`，对照臂设为 `"tsw_s_shuf"`
   - `tsw_floor: float = 0.0`，可选的权重下限，本轮不用
   - 训练时 actor 配置是 DictConfig，`__post_init__` 不会运行，所以守卫要在 ray_trainer 里重复：
     - 开启时任一样本缺 key 就 raise；
     - 和 null 模式同开时，只记录、不禁止。
3. **ray_trainer**：在构造 self-distillation batch 处（`_maybe_build_self_distillation_batch`）：
   - 从 `batch.non_tensor_batch["extra_info"][i][tsw_key]` 取 `w_i`；
   - 按 3.1 用 `response_mask` 的 token 数做全局 batch 归一；
   - 写成张量 `tsw_weight`（[B]），随 batch 下发。
4. **dp_actor**：把 `tsw_weight` 加入选键列表，按 micro-batch 切分，传给 `compute_self_distillation_loss`。
5. **core_algos.compute_self_distillation_loss**：新增参数 `tsw_weight: Optional[Tensor] = None`。
   - 在聚合前乘到 `weighted_per_token_loss` 上，和 PBD 的 `_w` 同一位置。
   - 不改 `loss_mask`：`self_distillation_mask` 在别处被当作 0/1 使用（`<= 0.5` fallback）。
   - 聚合仍然是 token-mean，分母为全局 `batch_num_tokens`。因为 3.1 已经按全局 token 归一，平均权重为 1。
   - 指标：`self_distillation/tsw_weight_mean`（应约等于 1）、`tsw_weight_std`、`tsw_s_mean`（归一前），以及按 s 档的 JSD。
   - 指标名不得含 `max`/`min` 子串。
6. **启动脚本**：`scripts/train_off6k_tsw.sh`
   - 基于 off6k V0 配置：`TASK_TRAIN_FILE=data/train_tsw.parquet`，`teacher_image_key=bbox_images`，`swap_images`，null 模式关闭，`ALPHA=0.5`，`distillation_topk=100`。
   - 参数：rollout_n=2，lr 2e-6，seed 42，batch 48，mini-batch 48，3 卡，ULYSSES_SP=1，130 步，2 epoch 上限，save 10，resume_mode=auto。
   - `TSW=main|shuf|off`，tag 分别为 `off6k_TSWn2`、`off6k_TSWsn2`、`off6k_V0n2`，`RUN_SUFFIX=r2` 加后缀。

### 4.1 测试（单测在 GPU hold 上跑，登录节点只做语法检查）

`scripts/test_tsw.py`：
- 归一化：token 加权均值 = 1；全部权重相同时 `w̃ ≡ 1`。
- **关闭路径**：`tsw_enable=False` 或 `tsw_weight=None` 时，loss 和 metrics 与改动前逐比特相同。
- **开启路径**：权重全为 1 时与关闭路径相同；权重翻倍的行，梯度贡献也翻倍。
- 和 PBD `_w` 同时开启时，两者相乘，顺序无关。
- parquet 构建：对齐、打乱是排列、原字段不变。
- 回归：`scripts/run_neghist_tests.sh` 全绿。

然后做 code review（按审查清单），修完提交 `sup`。GPU 冒烟 2 步，检查：
- `tsw_weight_mean ≈ 1`；
- `tsw_s_mean` 与全集均值 0.786 同量级；
- 无 NaN；
- 每步耗时与 V0 相同。

## 5. 实验设计

| 臂 | tag | 权重 | 次数 |
|---|---|---|---|
| 基线 V0 | off6k_V0n2 | 无 | ×2 |
| 主臂 TSW | off6k_TSWn2 | `s_i`，batch 归一 | ×2 |
| 对照 TSW-shuf | off6k_TSWsn2 | 打乱的 `s_i`，batch 归一 | ×2 |

- 现有 off6k 运行都是 PBD 变体，没有不带 PBD 的 V0 n2 基线，所以基线要新跑。
- 资源：RTX Pro 6000，3 卡一组的 hold；每次约 130 步。如果卡紧，顺序为：主臂 r1 + 基线 r1 → 主臂 r2 + 基线 r2 → 对照 ×2。
- 评测：每个 ckpt 测 V\*、ZoomBench、HR-4K、HR-8K，gpt-oss-120b judge（pure-llm 口径）。写 `run_records/<tag>.yaml`，只保留 ZoomBench 最好的 ckpt。汇报按 V\* 峰值选单个 ckpt，其余基准报同一 ckpt，两次运行报范围。

### 5.1 判据（事先登记）

- **有效：** 主臂两次在 V\* 和 ZoomBench 上都不低于基线两次的范围上沿，且至少一项高于；对照臂落在基线范围内。
- **机制核对：** 用主臂和基线选定的 ckpt，在全部 6241 道训练题上各做 8 次采样（与 `F1_step70_full_1020` 同协议），用 `probe_difficulty_select.py` 重算 2.2 的表。预期：
  - 教师一致桶的收益不低于基线；
  - "教师分歧 × 学生确定"格的大幅退步降到零假设以下。
- **无效：** 主臂与对照臂无差别，或者都在基线范围内。此时记入 negative_history，结论是"题级静态教师可靠性不是瓶颈"。

## 6. 风险与已知盲区

- 教师 8/8 一致但答错的题约占 8%，会被全权重蒸馏到错误答案上。
- 权重来自教师**独立作答**的稳定性，不等于教师在学生前缀上的信号质量（教师会跟随前缀）。TSW 只决定"这道题信不信教师"，不修正逐位置的信号。
- 教师采样用的 max_tokens 是 512，训练时是 1024。截断导致的未解析样本会压低 s；≤3/8 档里有 15% 未解析，部分可能是截断造成的。可以在 1024 下重采样核对，成本约 1 卡 1 小时。
- 全部 6241 题的 data_source 都是同一个（`zwz_rl_vqa_bbox_teacher`），没有现成的类别字段，无法直接检查降权是否系统性地压低某类题（比如关系类或 OCR 类）。评测时看各基准的分类别分数。
- 归一化后，高 s 题的有效权重约为 1/0.786 ≈ 1.27 倍。对已经学会的题，过度强化可能带来格式漂移。监控推理期缺 `<answer>` 的比例和平均长度。

## 7. 后续（不在本轮）

- 教师 prompt 末位的 hidden 探针替代 8 次采样（让方法能扩展到没有预采样的数据）。
- 权重形状：`s^γ`、`(s − 0.25)/0.75`、下限 floor。
- 和学生侧信号组合（教师稳定性减学生稳定性）。
