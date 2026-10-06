#!/usr/bin/env python
"""P1 Counterfactual-Flip Distillation (CFD) unit tests -- CPU only.
Spec: docs/proposals_2026-10-06/01_P1_counterfactual_flip.md §3-§4 and 06_P1_revision.md §2, §5.
Groups: [1] flip-set determination (tail exclude/include, t=0 exclusion, padding);
        [2] targets (flip: gamma-sharpened explicit columns with tail mass kept; flip_u == A target on F; p+ off F);
        [3] weights (1+lambda on F, weighted denominator; lambda=0 & gamma=1 == plain OPD loss);
        [4] A-t0 control (only column 0 changes; equals A elsewhere);
        [5] guards; [6] mutants; [7] tilt path bit-identical with the pinned previous commit (5db52df).
"""
import math, os, sys, subprocess, importlib.util, tempfile
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402
from verl.workers.config.actor import SelfDistillationConfig  # noqa: E402

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: FAIL.append(name)
def raises(fn, exc=Exception):
    try:
        fn(); return False
    except exc:
        return True
class Cfg(dict):
    __getattr__ = dict.get
def cfg(**kw):
    d = dict(full_logit_distillation=True, distillation_topk=5, distillation_add_tail=True, alpha=0.5, is_clip=None,
             counterfactual_null_mode="mean_color", counterfactual_extrapolation_beta=4.0, counterfactual_null_scope="last",
             counterfactual_u_clip_pos=False, counterfactual_reference="null", teacher_target_mode="legacy",
             counterfactual_target_mode="tilt", flip_gamma=50.0, flip_lambda=1.0, flip_tail_policy="exclude", counterfactual_t0_beta_zero=False)
    d.update(kw); return Cfg(d)
torch.manual_seed(11)
B, T, K = 3, 7, 5
def make(seed=0):
    # full-vocab log-softmax, then GATHER the student's top-K so the tail bucket is small (like training: ~0.3%)
    g = torch.Generator().manual_seed(seed)
    zs = torch.randn(B, T, 40, generator=g) * 3; zt = zs + torch.randn(B, T, 40, generator=g) * 1.5; zn = zt + torch.randn(B, T, 40, generator=g) * 1.5
    idx = zs.topk(K, -1).indices
    st = torch.log_softmax(zs, -1).gather(-1, idx).clone().requires_grad_(True)
    te = torch.log_softmax(zt, -1).gather(-1, idx)
    nu = torch.log_softmax(zn, -1).gather(-1, idx)
    mask = torch.ones(B, T); mask[2, 5:] = 0          # padding on the last row
    return st, te, nu, mask
def run(c, st, te, nu, mask, **kw):
    out = compute_self_distillation_loss(student_log_probs=st.sum(-1), teacher_log_probs=te.sum(-1), response_mask=mask, self_distillation_config=c,
                                         old_log_probs=st.sum(-1).detach(), student_topk_log_probs=st, teacher_topk_log_probs=te, teacher_null_topk_log_probs=nu,
                                         teacher_null_log_probs=nu.sum(-1), self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean", **kw)
    loss, met = out[0], out[1]; g = torch.autograd.grad(loss, st, retain_graph=True)[0]; return loss.detach(), g, met
def add_tail(lp):
    ls = torch.logsumexp(lp, -1, keepdim=True).clamp(max=-1e-7); return torch.cat([lp, torch.log(-torch.expm1(ls))], -1)
st, te, nu, mask = make()
TE, NU = add_tail(te), add_tail(nu)
a_real, a_null = TE.argmax(-1), NU.argmax(-1)

print("[1] flip set")
l, g, m = run(cfg(counterfactual_target_mode="flip"), st, te, nu, mask)
ref_flip = (a_real != a_null) & (mask > 0) & (a_real != K); ref_flip[:, 0] = False
assert ref_flip.sum() >= 0.2 * mask.sum(), f"toy flip set too small ({int(ref_flip.sum())} of {int(mask.sum())})"
check("toy data: flip set covers >= 20% of valid tokens and the tail is small", ref_flip.sum() >= 0.2 * mask.sum() and float(TE.exp()[..., -1].mean()) < 0.2, f"flips={int(ref_flip.sum())}/{int(mask.sum())} tail={float(TE.exp()[..., -1].mean()):.3f}")
check("flip_count == reference (tail excluded, t=0 excluded, padding excluded)", m["self_distillation/flip_count_sum"] == ref_flip.float().sum().item())
check("flip_raw_count counts t=0 and tail-argmax flips too", m["self_distillation/flip_raw_count_sum"] == ((a_real != a_null) & (mask > 0)).float().sum().item())
check("flip_t0_raw reports the raw t=0 flips (before tail exclusion)", m["self_distillation/flip_t0_raw_sum"] == ((a_real != a_null)[:, 0]).float().sum().item())
check("flip_frac = count / valid tokens", abs(m["self_distillation/flip_frac"] - ref_flip.float().sum().item() / mask.sum().item()) < 1e-9)
# tail include
l2, g2, m2 = run(cfg(counterfactual_target_mode="flip", flip_tail_policy="include"), st, te, nu, mask)
ref_inc = (a_real != a_null) & (mask > 0); ref_inc[:, 0] = False
check("tail_policy=include keeps tail-argmax flips", m2["self_distillation/flip_count_sum"] == ref_inc.float().sum().item())
check("flip_tail_excluded_sum consistent", m["self_distillation/flip_tail_excluded_sum"] == ((a_real != a_null) & (mask > 0) & (a_real == K)).float().sum().item())
# force a t=0 flip and make sure it is excluded from the target
te_t0 = te.clone(); nu_t0 = nu.clone(); te_t0[:, 0, 0] = 5.0; nu_t0[:, 0, 1] = 5.0   # argmaxes differ at t=0 for all rows
l3, g3, m3 = run(cfg(counterfactual_target_mode="flip"), st, te_t0, nu_t0, mask)
check("t=0 never enters the flip set", m3["self_distillation/flip_t0_raw_sum"] == B and m3["self_distillation/flip_count_sum"] == ((add_tail(te_t0).argmax(-1) != add_tail(nu_t0).argmax(-1)) & (mask > 0) & (add_tail(te_t0).argmax(-1) != K))[:, 1:].float().sum().item())
check("flip mode reports beta=0 (unused) while flip_u keeps beta", m["self_distillation/counterfactual_extrapolation_beta"] == 0.0 and run(cfg(counterfactual_target_mode="flip_u"), st, te, nu, mask)[2]["self_distillation/counterfactual_extrapolation_beta"] == 4.0)

print("[2] targets")
# reference implementation of the CFD target
def cfd_target(TE, flip, gamma):
    ex = torch.ones_like(TE, dtype=torch.bool); ex[..., -1] = False
    ninf = torch.finfo(TE.dtype).min
    mass = torch.logsumexp(TE.masked_fill(~ex, ninf), -1, keepdim=True); sc = gamma * TE
    z = torch.logsumexp(sc.masked_fill(~ex, ninf), -1, keepdim=True); sharp = torch.where(ex, sc - z + mass, TE)
    return torch.where(flip.unsqueeze(-1), sharp, TE)
def jsd(target, s):
    S = add_tail(s); mix = torch.logsumexp(torch.stack([S + math.log(0.5), target + math.log(0.5)]), 0)
    return 0.5 * (F.kl_div(mix, S, reduction="none", log_target=True) + F.kl_div(mix, target, reduction="none", log_target=True)).sum(-1)
def ref_loss(target, s, mask, w):
    per = jsd(target, s); W = mask * w; return (per * W).sum() / W.sum()
tgt = cfd_target(TE, ref_flip, 50.0)
check("flip target: tail mass preserved on F", torch.allclose(tgt[..., -1][ref_flip], TE[..., -1][ref_flip]))
check("flip target: explicit mass renormalised (sums to 1)", torch.allclose(tgt.exp().sum(-1), torch.ones(B, T), atol=1e-5))
_ex_mass = tgt.exp()[..., :-1].sum(-1); _top_after = tgt.exp()[..., :-1].amax(-1) / _ex_mass; _top_before = TE.exp()[..., :-1].amax(-1) / TE.exp()[..., :-1].sum(-1)
check("flip target: argmax on F is argmax p+ and its explicit share rises (gamma=50 sharpening)", bool((tgt.argmax(-1)[ref_flip] == a_real[ref_flip]).all()) and bool((_top_after[ref_flip] >= _top_before[ref_flip] - 1e-6).all()) and bool((_top_after[ref_flip] > 0.9).all()), f"shares {_top_after[ref_flip].tolist()}")
w = 1.0 + 1.0 * ref_flip.float()
L_ref = ref_loss(tgt, st, mask, w)
check("flip mode loss == reference (target + 1+lambda weights with weighted denominator)", torch.allclose(l, L_ref.detach(), atol=1e-6), f"{float(l):.6f} vs {float(L_ref):.6f}")
# flip_u: A target on F, p+ elsewhere
lu, gu, mu = run(cfg(counterfactual_target_mode="flip_u", flip_lambda=0.0), st, te, nu, mask)
A_t = torch.log_softmax(TE + 4.0 * (TE - NU), -1); tgt_u = torch.where(ref_flip.unsqueeze(-1), A_t, TE)
check("flip_u loss == reference (A target on F, p+ off F, uniform weights)", torch.allclose(lu, ref_loss(tgt_u, st, mask, torch.ones(B, T)).detach(), atol=1e-6))
check("flip_u differs from A and from flip (gamma)", not torch.allclose(lu, run(cfg(), st, te, nu, mask)[0]) and not torch.allclose(lu, run(cfg(counterfactual_target_mode="flip", flip_lambda=0.0), st, te, nu, mask)[0]))
check("flip_target_tv_mean matches TV(q, p+) averaged over F", abs(m["self_distillation/flip_target_tv_mean"] - float((0.5 * (tgt.exp() - TE.exp()).abs().sum(-1))[ref_flip].mean())) < 1e-6)

print("[3] weights / degenerate settings")
l0, g0, m0 = run(cfg(counterfactual_target_mode="flip", flip_gamma=1.0, flip_lambda=0.0), st, te, nu, mask)
lv, gv, mv = run(cfg(counterfactual_null_mode=None, counterfactual_target_mode="tilt"), st, te, nu, mask)
check("flip with gamma=1, lambda=0 == plain OPD (V0) loss and gradient", torch.allclose(l0, lv, atol=1e-7) and torch.allclose(g0, gv, atol=1e-7))
l1, g1, m1 = run(cfg(counterfactual_target_mode="flip", flip_lambda=0.0), st, te, nu, mask)
l3b, _, _ = run(cfg(counterfactual_target_mode="flip", flip_lambda=3.0), st, te, nu, mask)
check("lambda changes the loss through the weights (lambda 0 vs 1 vs 3 differ)", not torch.allclose(l1, l) and not torch.allclose(l, l3b))
w3 = 1.0 + 3.0 * ref_flip.float()
check("lambda=3 loss == reference with weighted denominator", torch.allclose(l3b, ref_loss(tgt, st, mask, w3).detach(), atol=1e-6))
check("flip_pos_rel_mean in [0,1]", 0.0 <= m["self_distillation/flip_pos_rel_mean"] <= 1.0)
check("no metric key contains max/min", not any(("max" in k or "min" in k) for k in m if k.startswith("self_distillation/flip")))

print("[4] A-t0 control")
la, ga, ma = run(cfg(), st, te, nu, mask)
lt0, gt0, mt0 = run(cfg(counterfactual_t0_beta_zero=True), st, te, nu, mask)
tgt_t0 = A_t.clone(); tgt_t0[:, 0] = TE[:, 0]
check("A-t0 loss == A target with column 0 replaced by p+", torch.allclose(lt0, ref_loss(tgt_t0, st, mask, torch.ones(B, T)).detach(), atol=1e-6))
check("A-t0 gradient equals A's at t>=1, differs at t=0", torch.allclose(gt0[:, 1:], ga[:, 1:], atol=1e-7) and not torch.allclose(gt0[:, 0], ga[:, 0]))
check("metric t0_beta_zero reported", mt0["self_distillation/t0_beta_zero"] == 1.0 and ma["self_distillation/t0_beta_zero"] == 0.0)

print("[5] guards")
check("unknown target mode raises", raises(lambda: run(cfg(counterfactual_target_mode="foo"), st, te, nu, mask), ValueError))
check("flip without null_mode raises", raises(lambda: run(cfg(counterfactual_target_mode="flip", counterfactual_null_mode=None), st, te, nu, mask), ValueError))
check("flip + u_clip_pos raises", raises(lambda: run(cfg(counterfactual_target_mode="flip", counterfactual_u_clip_pos=True), st, te, nu, mask), ValueError))
check("flip + target_gamma!=1 raises", raises(lambda: run(cfg(counterfactual_target_mode="flip", counterfactual_target_gamma=0.5), st, te, nu, mask), ValueError))
check("flip + t0_beta_zero raises", raises(lambda: run(cfg(counterfactual_target_mode="flip", counterfactual_t0_beta_zero=True), st, te, nu, mask), ValueError))
check("flip_gamma<=0 raises", raises(lambda: run(cfg(counterfactual_target_mode="flip", flip_gamma=0.0), st, te, nu, mask), ValueError))
check("flip_lambda<0 raises", raises(lambda: run(cfg(counterfactual_target_mode="flip", flip_lambda=-1.0), st, te, nu, mask), ValueError))
check("bad tail policy raises", raises(lambda: run(cfg(counterfactual_target_mode="flip", flip_tail_policy="drop"), st, te, nu, mask), ValueError))
check("t0_beta_zero without null_mode raises", raises(lambda: run(cfg(counterfactual_t0_beta_zero=True, counterfactual_null_mode=None), st, te, nu, mask), ValueError))
check("t0_beta_zero + st raises", raises(lambda: run(cfg(counterfactual_t0_beta_zero=True, counterfactual_st_enable=True), st, te, nu, mask), ValueError))
check("t0_beta_zero + hist raises", raises(lambda: run(cfg(counterfactual_t0_beta_zero=True, counterfactual_hist_adaptive_beta=True), st, te, nu, mask), ValueError))
check("flip + future_weight raises", raises(lambda: run(cfg(counterfactual_target_mode="flip", counterfactual_future_weight=True), st, te, nu, mask), ValueError))
check("flip + student reference raises", raises(lambda: run(cfg(counterfactual_target_mode="flip", counterfactual_reference="student"), st, te, nu, mask), ValueError))
base = dict(full_logit_distillation=True, distillation_topk=100)
check("dataclass: flip accepted with null mode", SelfDistillationConfig(**base, counterfactual_null_mode="mean_color", counterfactual_null_scope="last", counterfactual_target_mode="flip").flip_gamma == 50.0)
check("dataclass: flip without null mode rejected", raises(lambda: SelfDistillationConfig(**base, counterfactual_target_mode="flip"), ValueError))
check("dataclass: flip + st rejected", raises(lambda: SelfDistillationConfig(**base, counterfactual_null_mode="mean_color", counterfactual_null_scope="last", counterfactual_target_mode="flip_u", counterfactual_st_enable=True), ValueError))

print("[6] mutants must differ from the implementation")
# (a) flip judged by sign of u at the sampled argmax instead of argmax change: build a case where u(a+)>0 but argmax unchanged
flip_sign = ((TE - NU).gather(-1, a_real.unsqueeze(-1)).squeeze(-1) > 0) & (mask > 0); flip_sign[:, 0] = False
check("sign-of-u mutant gives a different flip set on this data", not torch.equal(flip_sign, ref_flip))
# (b) unweighted denominator mutant
L_mut = ((jsd(tgt, st) * mask * w).sum() / mask.sum())
check("unweighted-denominator mutant differs from the implementation", not torch.allclose(l, L_mut.detach(), atol=1e-6))
# (c) tail participating in argmax without exclusion: construct a case where argmax p+ is the tail
te_tail = te.clone(); te_tail[1, 3, :] = -8.0            # all explicit tiny -> tail bucket is the argmax at (1,3)
nu_tail = nu.clone(); nu_tail[1, 3, :] = -8.0; nu_tail[1, 3, 2] = -0.01   # null's argmax is explicit token 2 -> a flip whose a+ is the tail
TEt = add_tail(te_tail); assert int(TEt.argmax(-1)[1, 3]) == K
l_ex, _, m_ex = run(cfg(counterfactual_target_mode="flip"), st, te_tail, nu_tail, mask)
l_in, _, m_in = run(cfg(counterfactual_target_mode="flip", flip_tail_policy="include"), st, te_tail, nu_tail, mask)
NUt = add_tail(nu_tail); _raw = (TEt.argmax(-1) != NUt.argmax(-1)) & (mask > 0); _raw[:, 0] = False
_exp_in = _raw.float().sum().item(); _exp_ex = (_raw & (TEt.argmax(-1) != K)).float().sum().item()
check("tail-argmax position is excluded under 'exclude' and counted under 'include'", bool(_raw[1, 3]) and m_ex["self_distillation/flip_count_sum"] == _exp_ex and m_in["self_distillation/flip_count_sum"] == _exp_in and _exp_in > _exp_ex)

print("[7] tilt path bit-identical with the pinned previous commit (5db52df)")
try:
    _REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _src = subprocess.check_output(["git", "-C", _REPO, "show", "5db52df:verl/trainer/ppo/core_algos.py"], text=True)
    _tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False); _tmp.write(_src); _tmp.close()
    _spec = importlib.util.spec_from_file_location("core_algos_old", _tmp.name); _old = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_old)
    ok = True
    for kw in (dict(), dict(counterfactual_u_clip_pos=True), dict(counterfactual_null_mode=None), dict(counterfactual_target_gamma=0.5),
               dict(counterfactual_floor_alpha=0.1), dict(counterfactual_tanh_scale=1.0), dict(counterfactual_extrapolation_beta=1.0)):
        for seed in (0, 1):
            s_, t_, n_, m_ = make(seed); c = cfg(**kw)
            args = dict(student_log_probs=s_.sum(-1), teacher_log_probs=t_.sum(-1), response_mask=m_, self_distillation_config=c, old_log_probs=s_.sum(-1).detach(),
                        student_topk_log_probs=s_, teacher_topk_log_probs=t_, teacher_null_topk_log_probs=n_, teacher_null_log_probs=n_.sum(-1), self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")
            o_new = compute_self_distillation_loss(**args); o_old = _old.compute_self_distillation_loss(**args)
            g_new = torch.autograd.grad(o_new[0], s_)[0]; g_old = torch.autograd.grad(o_old[0], s_)[0]
            same = torch.equal(o_new[0].detach(), o_old[0].detach()) and torch.equal(g_new, g_old)
            for k_, v_ in o_old[1].items():
                if k_ in o_new[1] and isinstance(v_, (int, float)) and isinstance(o_new[1][k_], (int, float)) and o_new[1][k_] != v_:
                    same = False; print("    metric differs:", k_, v_, o_new[1][k_])
            ok &= same
    check("7 configs x 2 seeds: tilt-path loss / gradient / existing metrics bit-identical with 5db52df", ok)
except Exception as e:
    check("pinned regression (could not load 5db52df)", False, repr(e))

print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAIL: {FAIL}")
sys.exit(1 if FAIL else 0)
