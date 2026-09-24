#!/usr/bin/env python
"""Rollout-disagreement 探针单测 (纯 CPU)。

① 人造正交向量 → R ≈ 1/√n        (独立基准, 判据的刻度)
② 人造同向向量 → R ≈ 1           (完全 agreement)
③ 人造反向对   → R ≈ 0           (完全抵消)
④ TV / JSD / add_tail 对拍 numpy 参考
⑤ q 的构造与生产 core_algos 一致 (同一 tilt 公式, 含 tail 列)
⑥ _mean_color_teacher_images(imgs,"last") 只改末图 (复用训练函数的行为验证)
⑦ build_messages_from_template 的占位符-图像配对与计数校验

Run: PYTHONPATH=$REPO python probing/test_rollout_disagree.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_rollout_disagree import R_of, add_tail, jsd, profile, spearman, tv  # noqa: E402
from probe_rollout_disagree import build_messages_from_template  # noqa: E402

rng = np.random.default_rng(7)
V = 500

# ① 正交 (近似: 随机高维向量两两近正交) → R ≈ 1/√n
for n in (2, 4, 8):
    gs = []
    for _ in range(n):
        v = rng.normal(size=V)
        gs.append({i: float(v[i]) for i in range(V)})
    R = R_of(gs)
    expect = 1 / np.sqrt(n)
    assert abs(R - expect) < 0.12, f"独立基准偏离: n={n} R={R:.3f} vs {expect:.3f}"
print("① 正交/独立基准: R ≈ 1/√n ✓")

# ② 完全同向 → R ≈ 1
base = rng.normal(size=V)
gs = [{i: float(base[i] * (1 + 0.01 * k)) for i in range(V)} for k in range(8)]
R = R_of(gs)
assert R > 0.99, f"同向应 R≈1, got {R:.4f}"
print(f"② 完全同向: R={R:.4f} ≈ 1 ✓")

# ③ 完全反向配对 (4 对 ±v) → R ≈ 0
gs = []
for k in range(4):
    v = rng.normal(size=V)
    gs.append({i: float(v[i]) for i in range(V)})
    gs.append({i: float(-v[i]) for i in range(V)})
R = R_of(gs)
assert R < 1e-9, f"反向对应 R≈0, got {R:.2e}"
print(f"③ 完全抵消: R={R:.2e} ≈ 0 ✓")

# ③b 稀疏 support 不同也能正确对齐 (键并集)
g1 = {1: 1.0, 2: 1.0}
g2 = {2: -1.0, 3: 1.0}
# mean = {1:0.5, 2:0.0, 3:0.5} -> ‖mean‖=0.7071; ‖g1‖=‖g2‖=1.4142 -> R=0.5
assert abs(R_of([g1, g2]) - 0.5) < 1e-9, R_of([g1, g2])
print("③b 稀疏键并集对齐 ✓")

# ④ TV / JSD / add_tail
a = np.array([[0.5, 0.3, 0.2]])
b = np.array([[0.2, 0.3, 0.5]])
assert abs(tv(a, b)[0] - 0.3) < 1e-12, tv(a, b)
assert abs(jsd(a, a)[0]) < 1e-12
assert jsd(a, b)[0] > 0
lp = np.log(np.array([[0.5, 0.25]]))            # support 内只占 0.75
at = add_tail(lp)
assert abs(np.exp(at).sum() - 1.0) < 1e-9, np.exp(at).sum()
assert abs(np.exp(at)[0, -1] - 0.25) < 1e-9, np.exp(at)[0, -1]
print("④ TV / JSD / add_tail 对拍 ✓")

# ⑤ q 的 tilt 构造与生产一致 (numpy 参考 vs analyze 内部路径)
T, k, beta = 3, 4, 4.0
lp_p = np.log(rng.dirichlet(np.ones(k), size=T) * 0.9)     # 留 10% 尾
lp_n = np.log(rng.dirichlet(np.ones(k), size=T) * 0.9)
P, N = add_tail(lp_p), add_tail(lp_n)
u = P - N
logq = P + beta * u
logq -= logq.max(-1, keepdims=True)
q_ref = np.exp(logq); q_ref /= q_ref.sum(-1, keepdims=True)
assert abs(q_ref.sum(-1).mean() - 1.0) < 1e-12
# 与 core_algos 的 log_softmax(log p+ + beta*u) 等价
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
q_prod = F.log_softmax(torch.tensor(P) + beta * torch.tensor(u), dim=-1).exp().numpy()
assert np.allclose(q_ref, q_prod, atol=1e-10), np.abs(q_ref - q_prod).max()
print("⑤ tilt 目标 q 与生产 log_softmax 路径一致 ✓")

# ⑤b 口径交叉验证: 把探针的三个 log-prob 喂给**生产函数**, 它自报的
#     counterfactual_target_tv 必须等于探针自算的 TV(q, p⁺) —— 这条直接证明
#     探针与训练同口径 (含 add_tail 与 tilt 的实现细节)。
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402


class _Cfg(dict):
    __getattr__ = dict.get


B, Tt, kk = 1, 5, 6
lp_stu_t = torch.log_softmax(torch.randn(B, Tt, kk + 4), -1).sort(-1, descending=True).values[..., :kk]
lp_pos_t = torch.log_softmax(torch.randn(B, Tt, kk + 4), -1).sort(-1, descending=True).values[..., :kk]
lp_nul_t = torch.log_softmax(torch.randn(B, Tt, kk + 4), -1).sort(-1, descending=True).values[..., :kk]
cfg = _Cfg({"full_logit_distillation": True, "distillation_topk": kk,
            "distillation_add_tail": True, "renorm_topk_log_probs": False,
            "alpha": 0.5, "is_clip": None, "counterfactual_null_mode": "mean_color",
            "counterfactual_extrapolation_beta": 4.0, "log_prob_dump_dir": None})
lpz = torch.randn(B, Tt) * 0.1 - 1.0
_, m_prod = compute_self_distillation_loss(
    student_log_probs=lpz, teacher_log_probs=lpz.clone(), response_mask=torch.ones(B, Tt),
    self_distillation_config=cfg, old_log_probs=lpz.clone(),
    student_topk_log_probs=lp_stu_t, teacher_topk_log_probs=lp_pos_t,
    teacher_null_topk_log_probs=lp_nul_t, self_distillation_mask=torch.ones(B),
    loss_agg_mode="token-mean")
# 探针侧同一公式 (numpy): add_tail -> tilt -> TV(q, p⁺)
P_ = add_tail(lp_pos_t[0].double().numpy())
N_ = add_tail(lp_nul_t[0].double().numpy())
lq = P_ + 4.0 * (P_ - N_)
lq -= lq.max(-1, keepdims=True)
q_ = np.exp(lq); q_ /= q_.sum(-1, keepdims=True)
tv_probe = float(tv(q_, np.exp(P_)).mean())
tv_prod = m_prod["self_distillation/counterfactual_target_tv"]
assert abs(tv_probe - tv_prod) < 1e-5, f"探针 TV(q,p+)={tv_probe:.6f} != 生产 {tv_prod:.6f}"
print(f"⑤b 与生产 compute_self_distillation_loss 同口径: TV={tv_probe:.6f} ✓")

# ⑥ null 视图: 只改末图
from PIL import Image  # noqa: E402

from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # noqa: E402

full = Image.new("RGB", (32, 24), (10, 20, 30))
crop = Image.new("RGB", (8, 8), (200, 100, 50))
nulls = RayPPOTrainer._mean_color_teacher_images([full, crop], "last")
assert len(nulls) == 2
assert np.array_equal(np.asarray(nulls[0]), np.asarray(full)), "首图必须原样保留"
assert nulls[1].size == crop.size, "末图尺寸必须不变"
assert len(set(nulls[1].getdata())) == 1, "末图必须是纯色"
assert nulls[1].getpixel((0, 0)) == (200, 100, 50), nulls[1].getpixel((0, 0))
nulls_all = RayPPOTrainer._mean_color_teacher_images([full, crop], "all")
assert len(set(nulls_all[0].getdata())) == 1, "null_scope=all 时首图也应被遮"
print("⑥ _mean_color_teacher_images(last) 只改末图 ✓")

# ⑦ 占位符-图像配对
msgs = build_messages_from_template("<image>\nZoomed: <image>\nQ?", [full, crop])
c = msgs[0]["content"]
assert [x["type"] for x in c] == ["image", "text", "image", "text"], [x["type"] for x in c]
assert c[0]["image"] is full and c[2]["image"] is crop, "图像顺序必须与占位符一致"
for bad in ([full], [full, crop, crop]):
    try:
        build_messages_from_template("<image>\n<image>\nQ?", bad)
        raise AssertionError(f"应拒绝数量不匹配: {len(bad)}")
    except ValueError:
        pass
print("⑦ 占位符-图像配对与计数校验 ✓")

# 附: profile / spearman
D = np.arange(100, dtype=float)
pf = profile(D, 10)
assert len(pf) == 10 and pf[0] < pf[-1]
assert abs(spearman(pf, pf) - 1.0) < 1e-9
assert abs(spearman(pf, -pf) + 1.0) < 1e-9
print("附 profile / spearman ✓")


# ⑧ Layer-2 轨迹状态量: 对拍设计文档 §5.3/§5.4 的手算表 (λ=0.8, τ_u=0.5)
from analyze_rollout_disagree import realized_state  # noqa: E402

doc = [("The", .90, -.03), ("red", .80, +.10), ("right", .75, -1.00),
       ("side", .65, -.20), ("so", .85, -.10), ("B", .70, -.80)]
doc_C = [0.011, 0.009, 0.152, 0.171, 0.170, 0.265]        # 文档 §5.4 表
T8, k8 = len(doc), 4
# 构造 dump: y_t 恒为 support 第 1 列(索引 0 是 student 的 top-1, 用第 col 列放 y)
ids8 = np.tile(np.array([[100, 101, 102, 103]]), (T8, 1)).astype(np.int32)
y8 = np.full(T8, 101, dtype=np.int32)                      # y 落在第 1 列
lp_s8 = np.zeros((T8, k8))
for t, (_, m, _u) in enumerate(doc):
    lp_s8[t, 0] = np.log(0.40)                             # max_v p_S
    lp_s8[t, 1] = np.log(0.40 * m)                         # p_S(y_t) = m * max
    lp_s8[t, 2] = lp_s8[t, 3] = np.log(1e-4)
lp_p8 = np.zeros((T8, k8)) + np.log(0.1)
lp_n8 = lp_p8.copy()
for t, (_, _m, u) in enumerate(doc):
    lp_n8[t, 1] = lp_p8[t, 1] - u                          # u(y_t) = lp_pos - lp_null
d8 = {"ids": ids8, "y": y8, "lp_stu": lp_s8.astype(np.float16),
      "lp_pos": lp_p8.astype(np.float16), "lp_null": lp_n8.astype(np.float16)}
s8 = realized_state(d8, beta=2.0, tau_u=0.5, lam=0.8)
assert s8["miss"] == 0, s8["miss"]
assert np.allclose(s8["m"], [m for _, m, _u in doc], atol=2e-3), s8["m"]
assert np.allclose(s8["C_ema"], doc_C, atol=2e-3), (s8["C_ema"], doc_C)
# max 变体: 强事件立即生效 -> 在 'right' 处就该 ≥ EMA
i_right = 2
assert s8["C_max"][i_right] > s8["C_ema"][i_right], (s8["C_max"][i_right], s8["C_ema"][i_right])
# e_t=0 时 EMA 应按 λ 衰减
assert abs(s8["C_ema"][1] - 0.8 * s8["C_ema"][0]) < 1e-9
print(f"⑧ Layer-2 C_t 对拍文档手算表: EMA={np.round(s8['C_ema'],3).tolist()} ✓")
print(f"   max 变体 @right: {s8['C_max'][i_right]:.3f} > EMA {s8['C_ema'][i_right]:.3f} "
      f"(强事件立即生效) ✓")

print("\ntest_rollout_disagree: ALL 8 GROUPS PASS")
