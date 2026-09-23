#!/usr/bin/env python
"""ST (selective suppression + conserved transfer) unit tests.

① st off (default & absent keys) == baseline bit-identical
② empty-donor guard: u >= -tau everywhere -> q == a exactly (delta = 0)
③ golden case: the proposal's six-token worked example reproduces
   q = (0.36, 0.09, 0.04, 0.075, 0.01, 0.425) with tau=0.10 rho=0.10 amax=0.5 eps=0.30
④ conservation: sum q = 1 AND TV(q, a) == delta exactly (random tensors)
⑤ donor floor q >= (1-alpha_max)·a on D; recipient ratio invariance q(v)/q(w)=a(v)/a(w)
⑥ dashboard: st_* five metrics present, finite; delta matches counterfactual_target_tv
⑦ config validation (st without null_mode; st + u_clip_pos mutual exclusion; ranges)

Run: PYTHONPATH=$REPO python scripts/test_st.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402


class Cfg(dict):
    __getattr__ = dict.get


def cfg(**kw):
    d = {"full_logit_distillation": True, "distillation_topk": 8, "distillation_add_tail": True,
         "renorm_topk_log_probs": False, "alpha": 0.5, "is_clip": None,
         "counterfactual_null_mode": "mean_color", "counterfactual_extrapolation_beta": 4.0,
         "counterfactual_u_clip_pos": False, "counterfactual_st_enable": False,
         "counterfactual_st_tau": 0.10, "counterfactual_st_head_ratio": 0.10,
         "counterfactual_st_alpha_max": 0.50, "counterfactual_st_eps": 0.30,
         "log_prob_dump_dir": None}
    d.update(kw); return Cfg(d)


torch.manual_seed(7)
B, T, K = 2, 4, 8


def topk_logps():
    raw = torch.randn(B, T, K + 5)
    return torch.log_softmax(raw, -1).sort(-1, descending=True).values[..., :K]


student0, real0, null0 = topk_logps(), topk_logps(), topk_logps()
lp = torch.randn(B, T) * 0.1 - 1.0
mask = torch.ones(B, T)


def run(c, student=None, real=None, null=None):
    st = (student if student is not None else student0)
    return compute_self_distillation_loss(
        student_log_probs=lp, teacher_log_probs=lp.clone(), response_mask=mask,
        self_distillation_config=c, old_log_probs=lp.clone(),
        student_topk_log_probs=st,
        teacher_topk_log_probs=(real if real is not None else real0),
        teacher_null_topk_log_probs=(null if null is not None else null0),
        self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")


def add_tail(x):
    s = torch.logsumexp(x, -1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([x, torch.log(-torch.expm1(s))], -1)


def st_numpy(a, s, u, tau=0.10, rho=0.10, amax=0.50, eps=0.30, tail_col=True):
    """Reference construction over K(+tail) columns; tail (last col) always F."""
    a, s, u = (np.asarray(x, dtype=np.float64) for x in (a, s, u))
    exp = np.ones(a.shape[-1], bool)
    if tail_col: exp[-1] = False
    ae = np.where(exp, a, 0.0); se = np.where(exp, s, 0.0)
    head_a = ae >= rho * ae.max(); head_s = se >= rho * se.max()
    D = (u < -tau) & head_a & head_s & exp
    R = (u >= -tau) & exp
    md, mr = (a * D).sum(), (a * R).sum()
    if md <= 0 or mr <= 0: return a.copy(), 0.0
    delta = min(eps, amax * md)
    q = a.copy()
    q[D] *= (1 - delta / md); q[R] *= (1 + delta / mr)
    return q, delta  # conserved: sum q == sum a (renorm only in the torch impl, as float hygiene)


# ① off == baseline bit-identical (incl. absent keys)
l_base, m_base = run(cfg())
l_off, m_off = run(cfg(counterfactual_st_enable=False))
assert torch.allclose(l_base, l_off, atol=0), "st off must == baseline"
assert not any("st_" in k for k in m_off), "no st metrics when off"
c_absent = cfg()
for k in list(c_absent):
    if k.startswith("counterfactual_st"): del c_absent[k]
l_abs, _ = run(c_absent)
assert torch.allclose(l_base, l_abs, atol=0), "absent st keys must fall back to off"

# ② empty donors: real==null -> u≡0 >= -tau -> q == a; loss equals plain distill vs p⁺
l_eq, m_eq = run(cfg(counterfactual_st_enable=True), null=real0.clone())
l_plain, _ = run(cfg(counterfactual_null_mode=None))
assert abs(l_eq.item() - l_plain.item()) < 1e-6, "empty-D must give q==a (plain distill)"
assert m_eq["self_distillation/st_delta_tv"] < 1e-9, "delta must be 0 with empty D"

# ③ golden case (proposal §4 six-token table; no tail in the example -> emulate with
# near-zero tail via probs summing to 1)
a6 = np.array([0.60, 0.15, 0.04, 0.03, 0.01, 0.17])
u6 = np.array([-0.154, -0.288, -0.118, 0.693, -0.405, 1.917])
q_ref, d_ref = st_numpy(np.append(a6, 1e-9), np.append(a6, 1e-9), np.append(u6, 0.0))
gold = np.array([0.36, 0.09, 0.04, 0.075, 0.01, 0.425])
assert abs(d_ref - 0.30) < 1e-9 and np.allclose(q_ref[:6], gold, atol=1e-6), \
    f"golden case mismatch: {q_ref[:6]} vs {gold}"
# same through the torch path: build B=T=1 tensors with 6 explicit cols + tail
la6 = torch.log(torch.tensor(a6, dtype=torch.float32)).view(1, 1, 6)
ln6 = la6 - torch.tensor(u6, dtype=torch.float32).view(1, 1, 6)
ls6 = la6.clone()
c6 = cfg(counterfactual_st_enable=True, distillation_topk=6)
l6, m6 = compute_self_distillation_loss(
    student_log_probs=torch.zeros(1, 1) - 1.0, teacher_log_probs=torch.zeros(1, 1) - 1.0,
    response_mask=torch.ones(1, 1), self_distillation_config=c6,
    old_log_probs=torch.zeros(1, 1) - 1.0,
    student_topk_log_probs=ls6, teacher_topk_log_probs=la6, teacher_null_topk_log_probs=ln6,
    self_distillation_mask=torch.ones(1), loss_agg_mode="token-mean")
assert abs(m6["self_distillation/st_delta_tv"] - 0.30) < 1e-4, \
    f"torch golden delta {m6['self_distillation/st_delta_tv']} != 0.30"
assert abs(m6["self_distillation/counterfactual_target_tv"] - 0.30) < 1e-4, \
    "TV(q,a) must equal delta on golden case"

# ④ conservation on random tensors
l_st, m_st = run(cfg(counterfactual_st_enable=True))
ls_, lr_, ln_ = add_tail(student0).numpy(), add_tail(real0).numpy(), add_tail(null0).numpy()
a_np, s_np, u_np = np.exp(lr_), np.exp(ls_), lr_ - ln_
deltas = []
for b in range(B):
    for t in range(T):
        a64 = a_np[b, t].astype(np.float64)
        s64 = s_np[b, t].astype(np.float64)
        u64 = u_np[b, t].astype(np.float64)
        q, d = st_numpy(a64, s64, u64)
        deltas.append(d)
        assert abs(q.sum() - a64.sum()) < 1e-12, "mass must be conserved (sum q == sum a)"
        assert abs(0.5 * np.abs(q - a64).sum() - d) < 1e-9, "TV(q,a) must == delta"
        # ⑤ floor + recipient ratio invariance
        exp = np.ones(K + 1, bool); exp[-1] = False
        ae = np.where(exp, a64, 0.0); se = np.where(exp, s64, 0.0)
        D = (u64 < -0.10) & (ae >= 0.10 * ae.max()) & (se >= 0.10 * se.max()) & exp
        R = (u64 >= -0.10) & exp
        assert np.all(q[D] >= (1 - 0.50) * a64[D] - 1e-12), "donor floor violated"
        ir = np.where(R)[0]
        if len(ir) >= 2:
            r0 = q[ir] / a64[ir]
            assert np.allclose(r0, r0[0], rtol=1e-9), "recipient ratios must be preserved"
assert abs(m_st["self_distillation/st_delta_tv"] - float(np.mean(deltas))) < 1e-5, \
    f"st_delta_tv {m_st['self_distillation/st_delta_tv']} != numpy mean {np.mean(deltas)}"

# ⑥ dashboard completeness + TV consistency
for k in ("st_delta_tv", "st_donor_coverage", "st_donor_tokens_frac", "st_mass_donor", "st_mass_recipient"):
    v = m_st[f"self_distillation/{k}"]
    assert np.isfinite(v), f"{k} not finite"
assert abs(m_st["self_distillation/st_delta_tv"] - m_st["self_distillation/counterfactual_target_tv"]) < 1e-5, \
    "existing TV metric must equal delta under ST"

# ⑦ config validation (real dataclass)
from verl.workers.config.actor import SelfDistillationConfig  # noqa: E402
try:
    SelfDistillationConfig(full_logit_distillation=True, distillation_topk=100,
                           counterfactual_st_enable=True)
    raise AssertionError("st without null_mode must be rejected")
except ValueError:
    pass
try:
    SelfDistillationConfig(full_logit_distillation=True, distillation_topk=100,
                           counterfactual_null_mode="mean_color",
                           counterfactual_st_enable=True, counterfactual_u_clip_pos=True)
    raise AssertionError("st + u_clip_pos must be rejected")
except ValueError:
    pass
try:
    SelfDistillationConfig(full_logit_distillation=True, distillation_topk=100,
                           counterfactual_null_mode="mean_color",
                           counterfactual_st_enable=True, counterfactual_st_alpha_max=1.5)
    raise AssertionError("alpha_max out of range must be rejected")
except ValueError:
    pass
SelfDistillationConfig(full_logit_distillation=True, distillation_topk=100,
                       counterfactual_null_mode="mean_color", counterfactual_null_scope="last",
                       counterfactual_st_enable=True)

print("test_st: ALL 7 GROUPS PASS")
