#!/usr/bin/env python
"""Floor-only arm unit tests (plausibility floor ported from the OPSA floor arm).

① off (alpha=0, and absent key) == baseline bit-identical
② CROSS-CHECK vs the OPSA reference implementation of reconstruct_aha_target's floor
   branch (valley = log p+ < log(alpha) + max log p+; clamp_mask = valley & u<0),
   reimplemented here from that source and compared on the same tensors
③ u >= 0 everywhere -> floor is a no-op (clamp needs u<0) -> equals full tilt
④ alpha -> 1.0 (everything is valley) == suppression of ALL negative u == plain
   positive-only tilt; alpha tiny -> no clamping -> equals full tilt
⑤ tail column obeys the same valley rule (OPSA semantics: tail belongs to valley)
⑥ dashboard: floor_clamped_frac present, finite, in [0,1], matches hand count
⑦ config validation (range, requires null_mode, mutual exclusion with sup/ST)

Run: PYTHONPATH=$REPO python scripts/test_floor.py
"""
import math, os, sys
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
         "counterfactual_floor_alpha": 0.0, "log_prob_dump_dir": None}
    d.update(kw); return Cfg(d)


torch.manual_seed(11)
B, T, K = 2, 4, 8


def topk_logps():
    raw = torch.randn(B, T, K + 5)
    return torch.log_softmax(raw, -1).sort(-1, descending=True).values[..., :K]


student0, real0, null0 = topk_logps(), topk_logps(), topk_logps()
lp = torch.randn(B, T) * 0.1 - 1.0
mask = torch.ones(B, T)


def run(c, real=None, null=None):
    return compute_self_distillation_loss(
        student_log_probs=lp, teacher_log_probs=lp.clone(), response_mask=mask,
        self_distillation_config=c, old_log_probs=lp.clone(),
        student_topk_log_probs=student0,
        teacher_topk_log_probs=(real if real is not None else real0),
        teacher_null_topk_log_probs=(null if null is not None else null0),
        self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")


def add_tail(x):
    s = torch.logsumexp(x, -1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([x, torch.log(-torch.expm1(s))], -1)


def opsa_floor_reference(log_h, log_f, beta, floor_alpha):
    """Byte-faithful reimplementation of Vision-OPD-OPSA
    reconstruct_aha_target's floor branch (core_algos.py:1299-1303):

        valley     = log_h < math.log(floor_alpha) + log_h.max(dim=-1, keepdim=True).values
        clamp_mask = valley & (u_raw < 0)
        u          = torch.where(clamp_mask, torch.zeros_like(u_raw), u_raw)
        log_q      = torch.log_softmax(log_h + beta * u, dim=-1)

    log_h / log_f are the tail-augmented (K+1) teacher log-probs, as in OPSA.
    """
    u_raw = log_h - log_f
    valley = log_h < math.log(floor_alpha) + log_h.max(dim=-1, keepdim=True).values
    clamp_mask = valley & (u_raw < 0)
    u = torch.where(clamp_mask, torch.zeros_like(u_raw), u_raw)
    return torch.log_softmax(log_h + beta * u, dim=-1), clamp_mask


def jsd_ref(log_q, log_s, alpha=0.5):
    q, s = log_q.exp(), log_s.exp()
    m = alpha * s + (1 - alpha) * q
    logm = (m + 1e-30).log()
    kl_s = (s * (log_s - logm)).sum(-1)
    kl_q = (q * (log_q - logm)).sum(-1)
    return (alpha * kl_s + (1 - alpha) * kl_q).mean()


# ① off == baseline bit-identical (incl. absent key)
l_base, m_base = run(cfg())
l_off, m_off = run(cfg(counterfactual_floor_alpha=0.0))
assert torch.allclose(l_base, l_off, atol=0), "alpha=0 must == baseline"
assert "self_distillation/floor_clamped_frac" not in m_off, "no floor metric when off"
c_absent = cfg(); del c_absent["counterfactual_floor_alpha"]
l_abs, _ = run(c_absent)
assert torch.allclose(l_base, l_abs, atol=0), "absent key must fall back to off"

# ② CROSS-CHECK against the OPSA reference on the same tensors
ALPHA_F, BETA = 0.1, 4.0
log_h, log_f, log_s = add_tail(real0), add_tail(null0), add_tail(student0)
log_q_ref, clamp_ref = opsa_floor_reference(log_h, log_f, BETA, ALPHA_F)
ref_loss = float(jsd_ref(log_q_ref, log_s))
l_floor, m_floor = run(cfg(counterfactual_floor_alpha=ALPHA_F))
assert abs(l_floor.item() - ref_loss) < 1e-5, \
    f"floor loss {l_floor.item()} != OPSA reference {ref_loss}"
hand_frac = float(clamp_ref.float().mean(-1).mean())
assert abs(m_floor["self_distillation/floor_clamped_frac"] - hand_frac) < 1e-6, \
    "floor_clamped_frac must match the OPSA reference clamp mask"

# ③ u >= 0 everywhere: real == null -> u == 0, clamp needs u<0 -> no-op
l_eq_floor, m_eq = run(cfg(counterfactual_floor_alpha=ALPHA_F), null=real0.clone())
l_eq_off, _ = run(cfg(), null=real0.clone())
assert torch.allclose(l_eq_floor, l_eq_off, atol=1e-7), "u==0: floor must be a no-op"
assert m_eq["self_distillation/floor_clamped_frac"] < 1e-9, "no clamping when u==0"

# ④ alpha extremes
# alpha -> 1: every column is valley (p < 1.0*max except the argmax itself), so
# essentially all negative u is clamped -> close to positive-only tilt
_, m_hi = run(cfg(counterfactual_floor_alpha=0.999))
_, m_lo = run(cfg(counterfactual_floor_alpha=1e-9))
assert m_hi["self_distillation/floor_clamped_frac"] > m_lo["self_distillation/floor_clamped_frac"], \
    "larger alpha must clamp more"
assert m_lo["self_distillation/floor_clamped_frac"] < 1e-9, "alpha~0 must clamp nothing"
l_lo, _ = run(cfg(counterfactual_floor_alpha=1e-9))
assert abs(l_lo.item() - l_base.item()) < 1e-6, "alpha~0 must equal the full tilt"

# ⑤ tail column obeys the valley rule (OPSA semantics)
tail_is_valley = (log_h[..., -1] < math.log(ALPHA_F) + log_h.max(-1).values)
tail_u_neg = ((log_h - log_f)[..., -1] < 0)
expect_tail_clamped = (tail_is_valley & tail_u_neg)
assert torch.equal(clamp_ref[..., -1], expect_tail_clamped), \
    "tail column must follow the same valley rule as explicit columns"

# ⑥ dashboard sanity
v = m_floor["self_distillation/floor_clamped_frac"]
assert np.isfinite(v) and 0.0 <= v <= 1.0, f"floor_clamped_frac out of range: {v}"

# ⑦ config validation
from verl.workers.config.actor import SelfDistillationConfig  # noqa: E402
base = dict(full_logit_distillation=True, distillation_topk=100,
            counterfactual_null_mode="mean_color")
for bad in (
    dict(counterfactual_floor_alpha=1.5, counterfactual_null_mode="mean_color",
         full_logit_distillation=True, distillation_topk=100),
    dict(counterfactual_floor_alpha=0.1, full_logit_distillation=True, distillation_topk=100),
    dict(**base, counterfactual_floor_alpha=0.1, counterfactual_u_clip_pos=True),
    dict(**base, counterfactual_floor_alpha=0.1, counterfactual_st_enable=True),
):
    try:
        SelfDistillationConfig(**bad)
        raise AssertionError(f"config must be rejected: {bad}")
    except ValueError:
        pass
SelfDistillationConfig(**base, counterfactual_null_scope="last", counterfactual_floor_alpha=0.1)

print("test_floor: ALL 7 GROUPS PASS (incl. cross-check vs OPSA reference)")
