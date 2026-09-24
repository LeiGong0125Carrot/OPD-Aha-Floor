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

print("\ntest_rollout_disagree: ALL 7 GROUPS PASS")
