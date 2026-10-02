#!/usr/bin/env python
"""Negative-history 臂单测 —— 9 组, 全 CPU。

对拍纪律 (tanh review 学费): 独立参照实现逐式照抄 docs/negative_history/02-05 的公式,
与生产 loss 数值对拍 <1e-5; 并断言能抓住 4 个变异体:
  ① N 用 inclusive cumsum (04 §4 要求 exclusive)
  ② future 含当前位 (05 §6 要求 exclusive 后缀)
  ③ w 漏加权分母 (05 §14 要求 Σ M w ℓ / Σ M w)
  ④ c 不乘 mask (02 §11 要求 padding 不进任何统计)
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
    d = {"full_logit_distillation": True, "distillation_topk": 8, "distillation_add_tail": True,
         "renorm_topk_log_probs": False, "alpha": 0.5, "is_clip": None,
         "counterfactual_null_mode": "mean_color", "counterfactual_extrapolation_beta": 4.0,
         "counterfactual_u_clip_pos": True,           # 底座: 负向抑制
         "counterfactual_st_enable": False, "counterfactual_floor_alpha": 0.0,
         "counterfactual_tanh_scale": 0.0, "counterfactual_target_gamma": 1.0,
         "counterfactual_hist_adaptive_beta": False, "counterfactual_hist_alpha": 1.0,
         "counterfactual_hist_shuffle": False,
         "counterfactual_future_weight": False, "counterfactual_future_alpha": 1.0,
         "log_prob_dump_dir": None}
    d.update(kw); return Cfg(d)


def add_tail(x):
    s = torch.logsumexp(x, -1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([x, torch.log(-torch.expm1(s))], -1)


# ---------- 独立参照实现 (逐式照抄文档) ----------
def ref_traj(tlp, nlp, mask, incl_hist=False, incl_fut=False, unmasked_c=False):
    """c (02§6), N exclusive (04§4), s (04§6), Fbar exclusive (05§6), r (05§8)."""
    c = torch.relu(-(tlp - nlp))
    if not unmasked_c:
        c = c * mask
    N = torch.cumsum(c, 1)
    if not incl_hist:
        N = N - c                      # exclusive
    s = N / (1 + N)
    rev = lambda x: x.flip(1).cumsum(1).flip(1)
    Fs = rev(c); K = rev(mask)
    if not incl_fut:
        Fs = Fs - c; K = K - mask      # exclusive
    Fbar = Fs / K.clamp(min=1)
    r = Fbar / (1 + Fbar)
    return c, N, s, r


def ref_loss(log_h, log_f, log_s, tlp, nlp, mask, beta, hist=False, fut=False,
             a_H=1.0, a_F=1.0, mutant=None):
    """参照 loss: sup tilt (β 或 β_t) -> JSD -> (加权)聚合。"""
    c, N, s, r = ref_traj(tlp, nlp, mask,
                          incl_hist=(mutant == "incl_hist"),
                          incl_fut=(mutant == "incl_fut"),
                          unmasked_c=(mutant == "unmasked_c"))
    u = log_h - log_f
    beta_t = beta * (1 + a_H * s) if hist else torch.full_like(s, beta)
    log_q = torch.log_softmax(log_h + beta_t.unsqueeze(-1) * torch.clamp(u, max=0.0), -1)
    q, ps = log_q.exp(), log_s.exp()
    m = 0.5 * ps + 0.5 * q
    logm = (m + 1e-30).log()
    l = 0.5 * (ps * (log_s - logm)).sum(-1) + 0.5 * (q * (log_q - logm)).sum(-1)  # [B,T]
    w = (1 + a_F * r) if fut else torch.ones_like(r)
    num = (l * mask * w).sum()
    den = (mask * w).sum() if mutant != "no_wden" else mask.sum()
    return num / den


print("=" * 76); print("Negative-history 臂单测"); print("=" * 76)
BETA = 4.0

# ---------- fixture: 变长 2 条 rollout ----------
torch.manual_seed(11)
B, T, K = 2, 6, 8
def topk_logps():
    return torch.log_softmax(torch.randn(B, T, K + 5), -1).sort(-1, descending=True).values[..., :K]
student0, real0, null0 = topk_logps(), topk_logps(), topk_logps()
lp = torch.randn(B, T) * 0.1 - 1.0                  # student realized
tlp = torch.randn(B, T) * 0.3 - 1.2                 # teacher real realized
nlp = tlp + torch.randn(B, T) * 0.8                 # teacher null realized (u_realized 可正可负)
mask = torch.tensor([[1,1,1,1,1,1],[1,1,1,0,0,0]], dtype=torch.float32)   # 02 §11 变长
log_h, log_f, log_s = add_tail(real0), add_tail(null0), add_tail(student0)

def run(c, null_lp=nlp):
    return compute_self_distillation_loss(
        student_log_probs=lp, teacher_log_probs=tlp, response_mask=mask,
        self_distillation_config=c, old_log_probs=lp.clone(),
        student_topk_log_probs=student0, teacher_topk_log_probs=real0,
        teacher_null_topk_log_probs=null0, teacher_null_log_probs=null_lp,
        self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")

# ---- 1. 全门关 ≡ sup bit-identical ----
print("\n[1] 门关恒等 (Arm B 回归保护)")
l_sup, m_sup = run(cfg())
c_absent = cfg()
for k in ("counterfactual_hist_adaptive_beta", "counterfactual_hist_alpha",
          "counterfactual_hist_shuffle", "counterfactual_future_weight",
          "counterfactual_future_alpha"):
    del c_absent[k]
l_abs, _ = run(c_absent)
check("缺全部新 key 时 bit-identical (保护旧 config)", torch.allclose(l_sup, l_abs, atol=0))
l_nonull, _ = run(cfg(), null_lp=None)
check("门关时不传 teacher_null_log_probs 也不受影响", torch.allclose(l_sup, l_nonull, atol=0))
rs = ref_loss(log_h, log_f, log_s, tlp, nlp, mask, BETA)
check("门关与参照一致", abs(l_sup.item() - rs.item()) < 1e-5,
      f"生产 {l_sup.item():.8f} vs 参照 {rs.item():.8f}")

# ---- 2. 文档工作例逐数对拍 ----
print("\n[2] 文档工作例")
ur = torch.tensor([[0.03, 0.05, 0.0, -0.486, -0.02]])
c_doc = torch.relu(-ur)
check("02 §7: c 序列", torch.allclose(c_doc, torch.tensor([[0,0,0,0.486,0.02]]), atol=1e-6),
      f"c={c_doc.tolist()}")
H = torch.cumsum(c_doc, 1) - c_doc
check("02 §7: H 序列 (exclusive, H5=0.486)", abs(H[0,4].item() - 0.486) < 1e-6
      and H[0,3].item() == 0.0)
N_doc = sum([0, 0.1, 0.5, 0.2])
check("04 §4: N([0,.1,.5,.2])=0.8", abs(N_doc - 0.8) < 1e-9)
for Nv, sv in ((0.3, 0.231), (2.0, 0.667), (5.0, 0.833)):
    check(f"04 §6: s({Nv})≈{sv}", abs(Nv/(1+Nv) - sv) < 1e-3)
check("04 §8: β=2,s=0.667 → β_t≈3.334", abs(2*(1+2/3) - 3.334) < 1e-3)
check("05 §7: 早/晚位置同均值 → F̄ 相同 (16/160 = 4/40)", abs(16/160 - 4/40) < 1e-12)
check("05 §9: F̄=0.15 → r≈0.130", abs(0.15/1.15 - 0.130) < 1e-3)
L_doc = (1*0.05 + 1.5*0.10 + 1.8*0.20) / (1 + 1.5 + 1.8)
check("05 §15: 加权例 ≈0.130", abs(L_doc - 0.1302) < 1e-3, f"{L_doc:.4f}")

# ---- 3. 界与边界条件 ----
print("\n[3] 界与边界")
_, N_, s_, r_ = ref_traj(tlp, nlp, mask)
check("s ∈ [0,1)", bool((s_ >= 0).all() and (s_ < 1).all()))
check("β_t ∈ [β, 2β) (α_H=1)", bool(((BETA*(1+s_) >= BETA) & (BETA*(1+s_) < 2*BETA)).all()))
check("w ∈ [1,2) (α_F=1)", bool(((1+r_ >= 1) & (1+r_ < 2)).all()))
# 末位 K_t=0 ⇒ w=1 (05 §6)
last_valid = [5, 2]
check("每行最后有效位 r=0 ⇒ w=1", all(abs(r_[b, last_valid[b]].item()) < 1e-9 for b in range(B)))
check("首位 N=0 ⇒ β_t=β (04 §7: 历史为空时抑制不关闭)",
      all(abs(s_[b, 0].item()) < 1e-9 for b in range(B)))

# ---- 4. Arm C 生产对拍 ----
print("\n[4] Arm C (历史自适应 β)")
l_C, m_C = run(cfg(counterfactual_hist_adaptive_beta=True))
r_C = ref_loss(log_h, log_f, log_s, tlp, nlp, mask, BETA, hist=True)
check("loss 与参照一致 (<1e-5)", abs(l_C.item() - r_C.item()) < 1e-5,
      f"生产 {l_C.item():.8f} vs 参照 {r_C.item():.8f}")
check("与 B 不同 (β_t 真的生效)", abs(l_C.item() - l_sup.item()) > 1e-7)
mu = ref_loss(log_h, log_f, log_s, tlp, nlp, mask, BETA, hist=True, mutant="incl_hist")
check("抓住变异①: N 用 inclusive cumsum", abs(l_C.item() - mu.item()) > 1e-7,
      f"正确 {l_C.item():.8f} vs 错版 {mu.item():.8f}")
check("机制指标上报", all(f"self_distillation/{k}" in m_C
      for k in ("nh_c_mean", "nh_c_frac_pos", "nh_N_last_mean", "nh_s_mean", "nh_beta_t_mean")))
check("beta_t_mean > β (有历史冲突时)", m_C["self_distillation/nh_beta_t_mean"] > BETA)

# ---- 5. Arm D 生产对拍 ----
print("\n[5] Arm D (未来持续性加权)")
l_D, m_D = run(cfg(counterfactual_future_weight=True))
r_D = ref_loss(log_h, log_f, log_s, tlp, nlp, mask, BETA, fut=True)
check("loss 与参照一致 (<1e-5)", abs(l_D.item() - r_D.item()) < 1e-5,
      f"生产 {l_D.item():.8f} vs 参照 {r_D.item():.8f}")
mu = ref_loss(log_h, log_f, log_s, tlp, nlp, mask, BETA, fut=True, mutant="incl_fut")
check("抓住变异②: future 含当前位", abs(l_D.item() - mu.item()) > 1e-7)
mu = ref_loss(log_h, log_f, log_s, tlp, nlp, mask, BETA, fut=True, mutant="no_wden")
check("抓住变异③: 分母漏加权 (05 §14)", abs(l_D.item() - mu.item()) > 1e-7,
      f"正确 {l_D.item():.8f} vs 错版 {mu.item():.8f}")
check("w 指标上报且 ∈ (1,2)", 1.0 < m_D["self_distillation/nh_w_mean"] < 2.0,
      f"w_mean={m_D['self_distillation/nh_w_mean']:.4f} p90={m_D['self_distillation/nh_w_p90']:.4f}")

# ---- 6. Arm E 生产对拍 ----
print("\n[6] Arm E (全开)")
l_E, m_E = run(cfg(counterfactual_hist_adaptive_beta=True, counterfactual_future_weight=True))
r_E = ref_loss(log_h, log_f, log_s, tlp, nlp, mask, BETA, hist=True, fut=True)
check("loss 与参照一致 (<1e-5)", abs(l_E.item() - r_E.item()) < 1e-5,
      f"生产 {l_E.item():.8f} vs 参照 {r_E.item():.8f}")

# ---- 7. 变长 mask: padding 不进任何统计 (02 §11) ----
print("\n[7] padding 隔离")
mu = ref_loss(log_h, log_f, log_s, tlp, nlp, mask, BETA, hist=True, fut=True, mutant="unmasked_c")
check("抓住变异④: c 不乘 mask", abs(l_E.item() - mu.item()) > 1e-7)
# 把第二行 padding 区的 null lp 改成极端值, loss 必须一位不变
nlp_poison = nlp.clone(); nlp_poison[1, 3:] = -50.0
l_E2, _ = run(cfg(counterfactual_hist_adaptive_beta=True, counterfactual_future_weight=True),
              null_lp=nlp_poison)
check("毒化 padding 位的 null lp, loss bit-identical", torch.allclose(l_E, l_E2, atol=0))
nlp_inf = nlp.clone(); nlp_inf[1, 3:] = float("inf")       # u=-inf at padding -> relu=inf
tlp_inf = tlp.clone(); tlp_inf[1, 3:] = float("-inf")
l_E3, _ = compute_self_distillation_loss(
    student_log_probs=lp, teacher_log_probs=tlp_inf, response_mask=mask,
    self_distillation_config=cfg(counterfactual_hist_adaptive_beta=True, counterfactual_future_weight=True),
    old_log_probs=lp.clone(), student_topk_log_probs=student0, teacher_topk_log_probs=real0,
    teacher_null_topk_log_probs=null0, teacher_null_log_probs=nlp_inf,
    self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")
check("padding 位为 ±inf 也不产生 NaN (inf*0 陷阱)", torch.isfinite(l_E3).all() and torch.allclose(l_E, l_E3, atol=0))

# ---- 8. misalign 控制 (shuffle) ----
print("\n[8] misalign 控制")
cS = cfg(counterfactual_hist_adaptive_beta=True, counterfactual_hist_shuffle=True)
torch.manual_seed(0); lS1, mS1 = run(cS)
torch.manual_seed(1); lS2, _ = run(cS)
check("shuffle 两次 loss 不同 (随机重排生效)", abs(lS1.item() - lS2.item()) > 1e-9)
lC1, _ = run(cfg(counterfactual_hist_adaptive_beta=True))
lC2, _ = run(cfg(counterfactual_hist_adaptive_beta=True))
check("C (无 shuffle) 两次 bit-identical (确定性)", torch.allclose(lC1, lC2, atol=0))
check("shuffle 保边际: nh_c_mean 与 C 相同",
      abs(mS1["self_distillation/nh_c_mean"] - m_C["self_distillation/nh_c_mean"]) < 1e-9)

# ---- 9. 守卫 ----
print("\n[9] 运行时守卫")
for kw, why in ((dict(counterfactual_hist_adaptive_beta=True, counterfactual_u_clip_pos=False),
                 "hist 无 u_clip_pos"),
                (dict(counterfactual_future_weight=True, counterfactual_u_clip_pos=False),
                 "fut 无 u_clip_pos"),
                (dict(counterfactual_hist_adaptive_beta=True, counterfactual_floor_alpha=0.1),
                 "与 floor 互斥"),
                (dict(counterfactual_future_weight=True, counterfactual_tanh_scale=0.5),
                 "与 tanh 互斥"),
                (dict(counterfactual_future_weight=True, counterfactual_target_gamma=50.0),
                 "与 γ 互斥"),
                (dict(counterfactual_hist_shuffle=True), "shuffle 无 hist")):
    ok = False
    try: run(cfg(**kw))
    except ValueError: ok = True
    check(f"拦住: {why}", ok)
ok = False
try: run(cfg(counterfactual_hist_adaptive_beta=True), null_lp=None)
except ValueError as e: ok = "teacher_null_log_probs" in str(e)
check("拦住: 门开但无 null realized lp", ok)

print("\n" + "=" * 76)
print("全部通过" if not FAIL else "失败: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
