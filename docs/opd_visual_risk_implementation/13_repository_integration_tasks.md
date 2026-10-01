# 13｜仓库接入任务与开发交付顺序

**状态：** 以下是建议的模块接口与任务，不是对现有仓库路径/API的描述。本次没有读取或修改用户的GitHub/HPC仓库。

## 1. 开始前的仓库审计

实施者先定位：现有Aha launcher、rollout sampler、teacher real/null scorer、distillation loss、actor updater、distributed grouping、evaluation脚本、checkpoint选择规则。

导出一份baseline manifest，确认实际数据规模/视图/初始化/n/seed/采样参数。不要从旧聊天路径猜当前branch或commit；不要从PDF默认表推断当前正在用的配置。

## 2. 建议模块结构（名称可适配实际框架）

```text
visual_risk/
  contracts.py             # 02: RolloutBatch与验证
  teacher_signal.py        # 03: aligned real/null gathered logp与cost
  history_risk.py          # 04: exclusive H与risk increment
  temporal_credit.py       # 05: reverse discounted return
  group_baseline.py        # 06: distributed complete-group LOO
  objectives.py            # 07/10: PPO或GU，互斥入口
  anchor.py                # 08: full或fixed top-k+tail JSD
  training_adapter.py      # 09: 现有框架接口
  config.py                # 11: required config与manifest
  monitoring.py            # 12: 训练内诊断
```

配套 `reference/math_core.py` 是这些数学模块的CPU oracle，不是可直接运行VLM的training_adapter。

## 3. 建议分批交付

### PR-1：契约与数学核心

实现03–06，迁移参考测试。输出应只包含数值函数和明确的tensor contract，不改sampler或现有Aha分支。验收：固定toy数组逐项一致，feedback全部detach。

### PR-2：Teacher与sampler适配

复用两视图，取完整softmax的sampled logp；保存真正old policy logp；验证image/prefix与response shift。验收：实际小batch teacher/actor对齐，首次ratio≈1。

### PR-3：Distributed baseline与PPO入口

完成完整prompt组通信，group→microbatch顺序正确，单卡/多卡baseline一致。PPO新入口不得复用OPSA bottom-20% mask或hidden reward normalization。

### PR-4：可选anchor与GU独立入口

实现固定支持或继承明确支持策略的JSD；GU local与PPO互斥。验收：梯度边界和tail mass正确，关闭anchor时不存不必要的分布cache。

### PR-5：训练循环、日志、恢复与正式配置

反馈在本批固定；每轮重新采样；checkpoint保存全部状态；配置缺值阻止启动。只在集成验收通过后进入既定seed与预算的正式训练比较。

## 4. 给实施代码代理的不可越界说明

- 不为方便额外增加teacher视图、critic或correctness reward。
- 不自动设置未锁定gamma/lambda/epsilon为聊天例子数字。
- 不把概率差代价替换成log代价而不改实验名。
- 不删除EOS、不把cap补成EOS、不按结果过滤rollout。
- 不将baseline在microbatch内临时计算。
- 不用student低概率/熵作为视觉反对标签。
- 不重新构造Aha q然后仍称主线是无重构的policy-risk优化。
- 不增加独立科学probe；工程测试只对实现做判定。

## 5. 每个实际训练run需要的manifest

真实git commit/diff hash；data与processor版本；student/teacher checkpoint；sampling spec；teacher score温度；cost/risk/gamma；baseline规则；loss reduction；PPO epochs/epsilon；lambda与anchor支持；EOS/cap处理；seed；batch与n的确切语义；judge/checkpoint选择规则。

## 6. 本次交付与未交付

已交付：逐模块实施文档、CPU数学参考、可执行单元测试、配置模板、精确数值例子。

未交付：当前仓库真实补丁、VLM模型适配、GPU/分布式验收、正式训练结果、性能或新颖性保证。不要把文档中的建议接口误认为已存在的工程功能。
