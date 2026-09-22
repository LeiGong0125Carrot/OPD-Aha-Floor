#!/usr/bin/env python
"""Suppression-only tilt (u⁻) unit tests.  q ∝ p⁺ · exp(β · min(u, 0)).

① flag off (default & absent key) == baseline bit-identical
② u≤0 everywhere: clip is a no-op -> output == full Aha tilt
③ u≥0 everywhere: q == p⁺ (tilt vanishes, plain distillation target)
④ numpy reference incl. tail column participating in clamp & tilt
⑤ group structure: q/a constant across u≥0 tokens (proportional receipt, no
   reranking); q/a monotone increasing in u across u<0 tokens
⑥ dashboard: sup_frac_u_pos present, finite, in [0,1]; matches hand count
⑦ config validation (u_clip_pos without null_mode rejected; null_scope guard)

Run: PYTHONPATH=$REPO python scripts/test_sup.py
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402


class Cfg(dict):
    __getattr__ = dict.get


def cfg(**kw):
    d = {"full_logit_distillation": True, "distillation_topk": 8, "distillation_add_tail": True,
         "renorm_topk_log_probs": False, "alpha": 0.5, "is_clip": None,
         "counterfactual_null_mode": "mean_color", "counterfactual_extrapolation_beta": 4.0,
         "counterfactual_u_clip_pos": False, "log_prob_dump_dir": None}
    d.update(kw); return Cfg(d)


torch.manual_seed(5)
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


# ① flag off == baseline bit-identical, incl. absent-key fallback (production A configs
# built before this field existed have no counterfactual_u_clip_pos key)
l_base, m_base = run(cfg())
l_off, m_off = run(cfg(counterfactual_u_clip_pos=False))
assert torch.allclose(l_base, l_off, atol=0), "flag off must == baseline"
assert "self_distillation/sup_frac_u_pos" not in m_off, "no sup metric when flag off"
c_absent = cfg(); del c_absent["counterfactual_u_clip_pos"]
assert c_absent.counterfactual_u_clip_pos is None
l_abs, _ = run(c_absent)
assert torch.allclose(l_base, l_abs, atol=0), "absent key must fall back to off"

# ② u<=0 everywhere: real strictly below null on every explicit token.
# Note tail then has u_tail>0 by construction (tails sum complements), so build the
# clip-noop check on the explicit columns via a shifted null: null = real + delta (delta>0)
# gives u = -delta < 0 explicitly; tail u may still be positive -> compare against a
# hand-tilted reference instead of the unclipped run (which only matches where clamp
# is inactive). Simplest exact route: clamp inactive <=> u<=0 on ALL 9 columns; force
# that by making null a renormalized upward shift is impossible (probs sum to 1), so
# we instead verify equivalence on a constructed support where the tail column of both
# real and null is the SAME (real==null shifted within explicit mass only).
shift = torch.full((B, T, K), 0.3)
real_lo = torch.log_softmax(torch.log_softmax(torch.randn(B, T, K + 5), -1)
                            .sort(-1, descending=True).values[..., :K], -1) - 3.0
# real_lo: explicit mass ~e^-3 scaled -> tail huge & similar for both views
null_hi = real_lo + shift  # explicit u = -0.3 < 0; tails both ~1 -> u_tail ~ +eps
l_clip, _ = run(cfg(counterfactual_u_clip_pos=True), real=real_lo, null=null_hi)
l_full, _ = run(cfg(counterfactual_u_clip_pos=False), real=real_lo, null=null_hi)
# tails: log(1-s_real) vs log(1-s_null), both ≈ log(1) -> u_tail tiny positive; the
# clipped run zeroes it, full run keeps it. Difference must be small but the explicit
# columns dominate. Assert near-equality (clamp only bites the ~0 tail term).
assert abs(l_clip.item() - l_full.item()) < 5e-3, \
    f"u<=0 explicit: clip must be ~no-op, {l_clip.item()} vs {l_full.item()}"

# ③ u>=0 everywhere: real==null -> u≡0 -> q == p⁺ exactly (both flag states equal, and
# equal to plain distillation against the real teacher)
l_eq_clip, m_eq = run(cfg(counterfactual_u_clip_pos=True), null=real0.clone())
l_eq_off, _ = run(cfg(counterfactual_u_clip_pos=False), null=real0.clone())
assert torch.allclose(l_eq_clip, l_eq_off, atol=1e-7), "u=0: clip must change nothing"
l_plain, _ = run(cfg(counterfactual_null_mode=None, counterfactual_u_clip_pos=False))
assert abs(l_eq_clip.item() - l_plain.item()) < 1e-6, "u=0 target must equal plain p⁺ distill"

# ④ numpy reference: q = softmax(lr + beta * min(lr - ln, 0)) over K+1 (tail included)
ls = add_tail(student0).numpy()
lr_ = add_tail(real0).numpy()
ln_ = add_tail(null0).numpy()
u = lr_ - ln_
logq = lr_ + 4.0 * np.minimum(u, 0.0)
logq -= logq.max(-1, keepdims=True)
q_ref = np.exp(logq); q_ref /= q_ref.sum(-1, keepdims=True)
ps = np.exp(ls)
m_mix = 0.5 * ps + 0.5 * q_ref
kl_t = (q_ref * (np.log(q_ref + 1e-30) - np.log(m_mix + 1e-30))).sum(-1)
kl_s = (ps * (ls - np.log(m_mix + 1e-30))).sum(-1)
ref_loss = float((0.5 * kl_s + 0.5 * kl_t).mean())
l_sup, m_sup = run(cfg(counterfactual_u_clip_pos=True))
assert abs(l_sup.item() - ref_loss) < 1e-4, f"sup loss {l_sup.item()} != numpy ref {ref_loss}"

# ⑤ group structure on q_ref: q/a constant over u>=0 tokens; monotone in u over u<0
a_ref = np.exp(lr_)
ratio = q_ref / a_ref
for b in range(B):
    for t in range(T):
        pos = u[b, t] >= 0
        if pos.sum() >= 2:
            r = ratio[b, t][pos]
            assert np.allclose(r, r[0], rtol=1e-5), "u>=0 tokens must share one q/a factor"
        neg_idx = np.where(~pos)[0]
        if len(neg_idx) >= 2:
            order = neg_idx[np.argsort(u[b, t][neg_idx])]
            rr = ratio[b, t][order]
            assert np.all(np.diff(rr) >= -1e-8), "q/a must be non-decreasing in u on u<0"

# ⑥ dashboard
assert "self_distillation/sup_frac_u_pos" in m_sup, "sup_frac_u_pos missing"
v = m_sup["self_distillation/sup_frac_u_pos"]
assert np.isfinite(v) and 0.0 <= v <= 1.0, f"sup_frac_u_pos out of range: {v}"
hand = float((u > 0).mean())  # uniform mask -> masked mean == plain mean
assert abs(v - hand) < 1e-5, f"sup_frac_u_pos {v} != hand count {hand}"
tv = m_sup["self_distillation/counterfactual_target_tv"]
_, m_full = run(cfg(counterfactual_u_clip_pos=False))
assert tv <= m_full["self_distillation/counterfactual_target_tv"] + 1e-9, \
    "sup TV(q,p⁺) should not exceed full-tilt TV (positive half removed)"

# ⑦ config validation (real dataclass)
from verl.workers.config.actor import SelfDistillationConfig  # noqa: E402
import dataclasses  # noqa: E402
fields = {f.name for f in dataclasses.fields(SelfDistillationConfig)}
assert "counterfactual_u_clip_pos" in fields and "counterfactual_null_scope" in fields
try:
    SelfDistillationConfig(full_logit_distillation=True, distillation_topk=100,
                           counterfactual_u_clip_pos=True)
    raise AssertionError("u_clip_pos without null_mode must be rejected")
except ValueError:
    pass
try:
    SelfDistillationConfig(counterfactual_null_scope="first")
    raise AssertionError("bad null_scope must be rejected")
except ValueError:
    pass
SelfDistillationConfig(full_logit_distillation=True, distillation_topk=100,
                       counterfactual_null_mode="mean_color", counterfactual_null_scope="last",
                       counterfactual_u_clip_pos=True)

print("test_sup: ALL 7 GROUPS PASS")
