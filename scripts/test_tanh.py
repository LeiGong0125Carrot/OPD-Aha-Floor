#!/usr/bin/env python
"""tanh 有界化臂单测 —— 7 组, 全 CPU。

判据来自 docs/tanh_algorithm_reference.md:
  τ·tanh(u/τ) 把 exp(β·u) 限制在 [e^{-βτ}, e^{+βτ}], 小 |u| 保持线性, 只压极端;
  因为 tanh 是奇函数且非降, 它与 floor 的 (u<0) 掩码、sup 的 clamp(max=0) 可交换。

组 6 是**真正的数值对拍**: 独立参照实现 -> JSD 标量, 与生产函数的 loss 逐位比较。
没有这一条, 一个把 β 放进 tanh 内部 (剂量错 4×) 的实现也能通过全部方向性断言。
"""
import math, os, sys, torch, torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: FAIL.append(name)


class Cfg(dict):
    __getattr__ = dict.get


def cfg(**kw):
    """与 scripts/test_floor.py 逐字段同构, 只多一个 counterfactual_tanh_scale。"""
    d = {"full_logit_distillation": True, "distillation_topk": 8, "distillation_add_tail": True,
         "renorm_topk_log_probs": False, "alpha": 0.5, "is_clip": None,
         "counterfactual_null_mode": "mean_color", "counterfactual_extrapolation_beta": 4.0,
         "counterfactual_u_clip_pos": False, "counterfactual_st_enable": False,
         "counterfactual_floor_alpha": 0.0, "counterfactual_tanh_scale": 0.0,
         "log_prob_dump_dir": None}
    d.update(kw); return Cfg(d)


def add_tail(x):
    s = torch.logsumexp(x, -1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([x, torch.log(-torch.expm1(s))], -1)


def tanh_reference(log_h, log_f, beta, tau, clip_pos=False, floor_alpha=0.0):
    """独立参照实现 (不调用生产代码)。log_h/log_f 是加过尾桶的 (K+1) teacher log-probs。"""
    u = log_h - log_f
    if tau > 0:
        u = tau * torch.tanh(u / tau)
    if floor_alpha > 0:
        valley = log_h < math.log(floor_alpha) + log_h.max(dim=-1, keepdim=True).values
        u = torch.where(valley & (u < 0), torch.zeros_like(u), u)
    elif clip_pos:
        u = torch.clamp(u, max=0.0)
    return torch.log_softmax(log_h + beta * u, dim=-1)


def jsd_ref(log_q, log_s, alpha=0.5):
    q, s = log_q.exp(), log_s.exp()
    m = alpha * s + (1 - alpha) * q
    logm = (m + 1e-30).log()
    return (alpha * (s * (log_s - logm)).sum(-1) + (1 - alpha) * (q * (log_q - logm)).sum(-1)).mean()


print("=" * 74); print("tanh 有界化臂单测"); print("=" * 74)
BETA, TAU = 4.0, 0.5

# ---- 1. 极限行为 ----
print("\n[1] 极限行为")
lp_real = F.log_softmax(torch.tensor([[[2.0, 1.0, 0.0, -1.0]]]), -1)
lp_null = F.log_softmax(torch.tensor([[[0.0, 1.0, 2.0, -1.0]]]), -1)
check("τ→0 时 q → p_real",
      torch.allclose(tanh_reference(lp_real, lp_null, BETA, 1e-6).exp(), lp_real.exp(), atol=1e-4))
check("τ→∞ 时 q → 原始 A 臂",
      torch.allclose(tanh_reference(lp_real, lp_null, BETA, 1e4),
                     tanh_reference(lp_real, lp_null, BETA, 0.0), atol=1e-4))

# ---- 2. 有界性 ----
print("\n[2] 有界性 (本臂的核心性质)")
f = torch.exp(BETA * TAU * torch.tanh(torch.linspace(-20, 20, 4001) / TAU))
lo, hi = math.exp(-BETA * TAU), math.exp(BETA * TAU)
check(f"tilt 因子 ∈ [{lo:.3f}, {hi:.3f}]", bool((f >= lo - 1e-6).all() and (f <= hi + 1e-6).all()),
      f"实测 [{f.min():.4f}, {f.max():.4f}]")
check("无界版在同区间会爆掉", math.exp(BETA * 20) > 1e30, f"exp(4·20)={math.exp(80):.2e}")

# ---- 3. 小 |u| 线性性 ----
print("\n[3] 小 |u| 线性性")
us = torch.tensor([0.01, 0.05, 0.1])
rel = ((TAU * torch.tanh(us / TAU) - us).abs() / us)
check("|u|≤0.1 时相对偏差 <1.5%", bool((rel < 0.015).all()), f"偏差={[f'{r:.4%}' for r in rel]}")

# ---- 4. 保号保序 ----
print("\n[4] 保号保序 (可组合性的前提)")
u = torch.randn(5000) * 3
v = TAU * torch.tanh(u / TAU)
check("保号", bool((torch.sign(u) == torch.sign(v)).all()))
check("非降 (掩码可交换的充分条件)",
      bool((torch.diff(TAU * torch.tanh(torch.sort(u).values / TAU)) >= 0).all()))
check("与 clamp(max=0) 可交换",
      torch.allclose(TAU * torch.tanh(torch.clamp(u, max=0) / TAU), torch.clamp(v, max=0), atol=1e-6))
# 有界化的已知代价: float32 下 tanh 对 |x|≳9 返回恰好 ±1, 极端之间的序被抹平。
sat = TAU * torch.tanh(torch.tensor([4.0, 4.5, 5.0, 10.0]) / TAU)
check("极端区退化为并列 (有界化的已知代价, 故 τ 不能太小)", bool(sat[-1] == sat[-2] == TAU),
      f"τ={TAU}: u=5 与 u=10 都映到 {sat[-1]:.6f}")
check("2τ 处只是被压缩、不是饱和 (指标命名的依据)", abs(math.tanh(2.0) - 0.9640) < 1e-3,
      f"tanh(2)={math.tanh(2.0):.4f} → 序仍保留")

# ---- 5. 目标集中度 (动机) ----
print("\n[5] 目标集中度 (动机的直接验证)")
lp_r = F.log_softmax(torch.tensor([[[0.5, 0.3, 0.1, 0.0, -0.2, -0.4]]]), -1)
lp_n = F.log_softmax(torch.tensor([[[-1.42, 0.3, 0.1, 0.0, -0.2, -0.4]]]), -1)
mx_off = tanh_reference(lp_r, lp_n, BETA, 0.0).exp().max().item()
mx_on = tanh_reference(lp_r, lp_n, BETA, TAU).exp().max().item()
check("tanh 显著降低目标最大概率", mx_on < mx_off - 0.10,
      f"max u={(lp_r - lp_n).max():.2f}  q_max: {mx_off:.4f} -> {mx_on:.4f}")

# ---- 6. 生产函数数值对拍 ----
print("\n[6] 生产函数数值对拍 (与独立参照实现逐位比较)")
torch.manual_seed(11)
B, T, K = 2, 4, 8
def topk_logps():
    return torch.log_softmax(torch.randn(B, T, K + 5), -1).sort(-1, descending=True).values[..., :K]
student0, real0, null0 = topk_logps(), topk_logps(), topk_logps()
lp = torch.randn(B, T) * 0.1 - 1.0
mask = torch.ones(B, T)
def run(c):
    return compute_self_distillation_loss(
        student_log_probs=lp, teacher_log_probs=lp.clone(), response_mask=mask,
        self_distillation_config=c, old_log_probs=lp.clone(),
        student_topk_log_probs=student0, teacher_topk_log_probs=real0,
        teacher_null_topk_log_probs=null0,
        self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")

log_h, log_f, log_s = add_tail(real0), add_tail(null0), add_tail(student0)
l_on, m_on = run(cfg(counterfactual_tanh_scale=TAU))
ref_on = float(jsd_ref(tanh_reference(log_h, log_f, BETA, TAU), log_s))
check("loss 与参照实现一致 (<1e-5)", abs(l_on.item() - ref_on) < 1e-5,
      f"生产 {l_on.item():.8f} vs 参照 {ref_on:.8f}")
# 剂量对拍: β 放进 tanh 内部 (界变成 [e^-0.5, e^0.5], 剂量错 4×) 必须被这条抓住
wrong = float(jsd_ref(torch.log_softmax(log_h + TAU * torch.tanh(BETA * (log_h - log_f) / TAU), -1), log_s))
check("能区分'β 放错位置'的 4× 错剂量实现", abs(l_on.item() - wrong) > 1e-3,
      f"正确 {l_on.item():.6f} vs 错版 {wrong:.6f}, 差 {abs(l_on.item()-wrong):.6f}")

# 关闭时必须与 A 臂逐位相同, 且缺 key 要回落到关闭
l_base, m_off = run(cfg())
l_zero, _ = run(cfg(counterfactual_tanh_scale=0.0))
check("gate 关时与 A 臂 bit-identical", torch.allclose(l_base, l_zero, atol=0))
c_absent = cfg(); del c_absent["counterfactual_tanh_scale"]
l_abs, _ = run(c_absent)
check("缺 key 时回落到关闭 (保护旧 config/ckpt)", torch.allclose(l_base, l_abs, atol=0))
ref_off = float(jsd_ref(tanh_reference(log_h, log_f, BETA, 0.0), log_s))
check("关闭态也与参照一致", abs(l_base.item() - ref_off) < 1e-5,
      f"生产 {l_base.item():.8f} vs 参照 {ref_off:.8f}")

check("gate 关时无 tanh 指标", "self_distillation/tanh_compressed_frac" not in m_off)
check("gate 开时有 tanh 指标", "self_distillation/tanh_compressed_frac" in m_on)
check("target_max_prob 两侧都记录", "self_distillation/target_max_prob" in m_off
      and "self_distillation/target_max_prob" in m_on,
      f"{m_off['self_distillation/target_max_prob']:.4f} -> {m_on['self_distillation/target_max_prob']:.4f}")
check("tanh 降低 batch 级目标集中度",
      m_on["self_distillation/target_max_prob"] < m_off["self_distillation/target_max_prob"])
check("TV 随之下降", m_on["self_distillation/counterfactual_target_tv"]
      < m_off["self_distillation/counterfactual_target_tv"],
      f"{m_off['self_distillation/counterfactual_target_tv']:.4f} -> "
      f"{m_on['self_distillation/counterfactual_target_tv']:.4f}")

# compressed_frac 必须在非零情形下被真正验证 (随机 topk 的 |u| 太小, 永远是 0)
big = torch.tensor([[[0.0, -6.0, -12.0, -18.0, -24.0, -30.0, -36.0, -42.0]]]).log_softmax(-1)
sml = torch.tensor([[[-42.0, -36.0, -30.0, -24.0, -18.0, -12.0, -6.0, 0.0]]]).log_softmax(-1)
_, m_big = compute_self_distillation_loss(
    student_log_probs=lp[:1, :1], teacher_log_probs=lp[:1, :1].clone(),
    response_mask=torch.ones(1, 1), self_distillation_config=cfg(counterfactual_tanh_scale=TAU),
    old_log_probs=lp[:1, :1].clone(), student_topk_log_probs=student0[:1, :1],
    teacher_topk_log_probs=big, teacher_null_topk_log_probs=sml,
    self_distillation_mask=torch.ones(1), loss_agg_mode="token-mean")
u_big = (add_tail(big) - add_tail(sml))[0, 0]
expect = float((u_big.abs() > 2 * TAU).float().mean())
check("compressed_frac 在非零情形下数值正确",
      abs(m_big["self_distillation/tanh_compressed_frac"] - expect) < 1e-6,
      f"实测 {m_big['self_distillation/tanh_compressed_frac']:.4f} vs 手算 {expect:.4f}")

# ---- 7. 组合与互斥 ----
print("\n[7] 组合与互斥")
l_fl, m_fl = run(cfg(counterfactual_tanh_scale=TAU, counterfactual_floor_alpha=0.1))
check("tanh + floor 与参照一致",
      abs(l_fl.item() - float(jsd_ref(tanh_reference(log_h, log_f, BETA, TAU, floor_alpha=0.1), log_s))) < 1e-5)
l_sp, m_sp = run(cfg(counterfactual_tanh_scale=TAU, counterfactual_u_clip_pos=True))
check("tanh + sup 与参照一致",
      abs(l_sp.item() - float(jsd_ref(tanh_reference(log_h, log_f, BETA, TAU, clip_pos=True), log_s))) < 1e-5)
_, m_sp0 = run(cfg(counterfactual_u_clip_pos=True))
check("保号 => sup 的 u>0 占比不受 tanh 影响",
      abs(m_sp["self_distillation/sup_frac_u_pos"] - m_sp0["self_distillation/sup_frac_u_pos"]) < 1e-12)
_, m_fl0 = run(cfg(counterfactual_floor_alpha=0.1))
check("保号 => floor 的 clamp 占比不受 tanh 影响",
      abs(m_fl["self_distillation/floor_clamped_frac"] - m_fl0["self_distillation/floor_clamped_frac"]) < 1e-12)
from verl.workers.config.actor import SelfDistillationConfig as SDC  # noqa: E402
ok = False
try:
    SDC(counterfactual_null_mode="mean_color", counterfactual_tanh_scale=0.5,
        counterfactual_st_enable=True)
except Exception as e:
    ok = "incompatible" in str(e) or "st_enable" in str(e)
check("tanh 与 ST 互斥 (dataclass 拦截)", ok)

print("\n" + "=" * 74)
print("全部通过" if not FAIL else "失败: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
