#!/usr/bin/env python
"""Candidate S (student-reference, null-free) unit tests -- CPU only.

Reference: docs/negative_history/candidate_s_null_free_implementation_plan.md (v0.1) §4-§7, §13.
Discipline (as in test_neghist.py): an INDEPENDENT reference implementation written from the
document's formulas is compared against the production loss; mutants must be caught.
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
         "counterfactual_reference": "null",            # A by default
         "counterfactual_u_clip_pos": False, "counterfactual_st_enable": False,
         "counterfactual_floor_alpha": 0.0, "counterfactual_tanh_scale": 0.0,
         "counterfactual_target_gamma": 1.0,
         "counterfactual_hist_adaptive_beta": False, "counterfactual_hist_alpha": 1.0,
         "counterfactual_hist_shuffle": False, "counterfactual_future_weight": False,
         "counterfactual_future_alpha": 1.0, "log_prob_dump_dir": None}
    d.update(kw); return Cfg(d)


def add_tail(x):
    s = torch.logsumexp(x, -1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([x, torch.log(-torch.expm1(s))], -1)


def jsd_rows(log_q, log_s):
    q, ps = log_q.exp(), log_s.exp()
    m = 0.5 * ps + 0.5 * q; logm = (m + 1e-30).log()
    return 0.5 * (ps * (log_s - logm)).sum(-1) + 0.5 * (q * (log_q - logm)).sum(-1)   # [B,T]


def ref_S(log_t_topk, log_s_topk, mask, beta, mutant=None, tail=True):
    """Independent reference (plan §7 kernel): q_S = softmax(log_t + beta*(log_t - sg(log_s))) on
    the add_tail'ed support; JSD(q_S, p_S) token-mean. Returns loss (with grad wrt log_s_topk)."""
    log_s = add_tail(log_s_topk) if tail else log_s_topk
    log_t = add_tail(log_t_topk) if tail else log_t_topk
    ref = log_s.detach()
    if mutant == "no_detach":
        ref = log_s
    elif mutant == "teacher_ref":
        ref = log_t.detach()                                  # degenerates to self-sharpening
    elif mutant == "ref_no_tail":
        ref = add_tail(log_s_topk.detach() * 0.97)            # a reference NOT on the same support/mass
    u = log_t - ref
    if mutant == "sign":
        u = -u
    log_q = torch.log_softmax(log_t + beta * u, -1)
    if mutant != "no_detach":
        log_q = log_q.detach()
    l = jsd_rows(log_q, log_s)
    return (l * mask).sum() / mask.sum()


print("=" * 76); print("Candidate S (student-reference, null-free) 单测"); print("=" * 76)
BETA = 4.0
torch.manual_seed(7)
B, T, K = 2, 6, 8
def topk_logps():
    return torch.log_softmax(torch.randn(B, T, K + 5), -1).sort(-1, descending=True).values[..., :K]
student0, real0, null0 = topk_logps(), topk_logps(), topk_logps()
lp = torch.randn(B, T) * 0.1 - 1.0
tlp = torch.randn(B, T) * 0.3 - 1.2
nlp = tlp + torch.randn(B, T) * 0.8
mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.float32)

def run(c, student=student0, real=real0, null=null0, null_lp=nlp, msk=mask, lpp=lp, tl=None):
    tl = tlp if tl is None else tl
    return compute_self_distillation_loss(
        student_log_probs=lpp, teacher_log_probs=tl, response_mask=msk,
        self_distillation_config=c, old_log_probs=lpp.clone(),
        student_topk_log_probs=student, teacher_topk_log_probs=real,
        teacher_null_topk_log_probs=null, teacher_null_log_probs=null_lp,
        self_distillation_mask=torch.ones(msk.shape[0]), loss_agg_mode="token-mean")

cS = cfg(counterfactual_reference="student")
def runS(c=cS, student=student0, real=real0, msk=mask):
    return run(c, student=student, real=real, null=None, null_lp=None, msk=msk)

# ---- 1. A 默认回归 ----
print("\n[1] A 默认回归 (reference 缺省 / 'null' 不改变任何东西)")
cA = cfg(); cAbs = cfg(); del cAbs["counterfactual_reference"]
sA = student0.clone().requires_grad_(True); sB = student0.clone().requires_grad_(True)
lA, mA = run(cA, student=sA); lB, _ = run(cAbs, student=sB)
lA.backward(); lB.backward()
check("reference='null' 与缺省 key: loss bit-identical", torch.equal(lA, lB))
check("reference='null' 与缺省 key: 学生梯度 bit-identical", torch.equal(sA.grad, sB.grad))
# A 独立参照 (null reference)
_lt, _ln, _ls = add_tail(real0), add_tail(null0), add_tail(student0)
_lqA = torch.log_softmax(_lt + BETA * (_lt - _ln), -1)
_rA = (jsd_rows(_lqA, _ls) * mask).sum() / mask.sum()
check("A 路径与独立参照一致 (<1e-6)", abs(lA.item() - _rA.item()) < 1e-6, f"{lA.item():.8f} vs {_rA.item():.8f}")
check("A 路径指标 reference_is_student = 0", mA["self_distillation/reference_is_student"] == 0.0)
check("A 路径新增诊断指标存在 (u_abs_mean, teacher_student_kl, teacher_tail_mass)",
      all(f"self_distillation/{k}" in mA for k in ("counterfactual_u_abs_mean", "teacher_student_kl", "teacher_tail_mass")))

# ---- 2. S 数学 ----
print("\n[2] S 数学")
lS, mS = runS()
rS = ref_S(real0, student0, mask, BETA)
check("S loss 与独立参照一致 (<1e-5)", abs(lS.item() - rS.item()) < 1e-5, f"生产 {lS.item():.8f} vs 参照 {rS.item():.8f}")
check("S 指标 reference_is_student = 1", mS["self_distillation/reference_is_student"] == 1.0)
for mut, why in (("teacher_ref", "参照用教师 (自锐化)"), ("ref_no_tail", "参照不在同一支持集"), ("sign", "u 符号反")):
    rm = ref_S(real0, student0, mask, BETA, mutant=mut)
    check(f"抓住变异: {why}", abs(lS.item() - rm.item()) > 1e-6, f"正确 {lS.item():.8f} vs 错版 {rm.item():.8f}")
# β=0 -> q = p+ ; 与 null_mode=None (纯蒸馏) 相同
l0, _ = runS(cfg(counterfactual_reference="student", counterfactual_extrapolation_beta=0.0))
lV0, _ = run(cfg(counterfactual_null_mode=None), null=None, null_lp=None)
check("β=0 时 S ≡ 普通蒸馏到 p+ (<1e-6)", abs(l0.item() - lV0.item()) < 1e-6, f"{l0.item():.8f} vs {lV0.item():.8f}")
# p_S = p+ -> q = p+
lEq, mEq = runS(student=real0.clone())
lEqV0, _ = run(cfg(counterfactual_null_mode=None), student=real0.clone(), null=None, null_lp=None)
check("p_S = p+ 时 q = p+ (loss 同纯蒸馏, 且 ≈0)", abs(lEq.item() - lEqV0.item()) < 1e-6 and lEq.item() < 1e-6,
      f"{lEq.item():.3e}")
check("p_S = p+ 时 target_tv = 0", mEq["self_distillation/counterfactual_target_tv"] < 1e-6)

# ---- 3. 文档数值例 (§4.1 / §4.2 / §5 / §6.2) ----
print("\n[3] 文档数值例")
def odds_S(RT, RS, beta=BETA): return RT * (RT / RS) ** beta
check("§4.2: 学生 odds 0.5/1/2/4 -> 512 / 32 / 2 / 0.125",
      all(abs(odds_S(2.0, rs) - v) < 1e-9 for rs, v in ((0.5, 512), (1.0, 32), (2.0, 2), (4.0, 0.125))))
# §4.1 three tokens, NO tail (probabilities already sum to 1 -> renorm is a no-op)
cS_notail = cfg(counterfactual_reference="student", distillation_topk=3, distillation_add_tail=False)
pS = torch.tensor([.30, .60, .10]); pT = torch.tensor([.60, .30, .10])
s41 = pS.log().view(1, 1, 3); t41 = pT.log().view(1, 1, 3); m1 = torch.ones(1, 1)
_, m41 = run(cS_notail, student=s41, real=t41, null=None, null_lp=None, msk=m1, lpp=torch.zeros(1, 1), tl=torch.zeros(1, 1))
q41 = (pT * (pT / pS) ** BETA); q41 = q41 / q41.sum()
check("§4.1: q_S = [0.987781350, 0.001929260, 0.010289389]",
      torch.allclose(q41, torch.tensor([0.987781350, 0.001929260, 0.010289389]), atol=1e-8), f"{q41.tolist()}")
check("§4.1: 生产 target_top1_prob = 0.987781350", abs(m41["self_distillation/target_top1_prob"] - 0.987781350) < 1e-6,
      f"{m41['self_distillation/target_top1_prob']:.9f}")
# §5 ranking reversal
pS5 = torch.tensor([.75, .20, .05]); pT5 = torch.tensor([.60, .30, .10])
_, m5 = run(cS_notail, student=pS5.log().view(1, 1, 3), real=pT5.log().view(1, 1, 3), null=None, null_lp=None,
            msk=m1, lpp=torch.zeros(1, 1), tl=torch.zeros(1, 1))
q5 = pT5 * (pT5 / pS5) ** BETA; q5 = q5 / q5.sum()
check("§5: 排序反转 q_S ≈ [0.073044812, 0.451403027, 0.475552161], argmax=plain",
      torch.allclose(q5, torch.tensor([0.073044812, 0.451403027, 0.475552161]), atol=1e-8) and q5.argmax().item() == 2)
check("§5: 生产 target_top1_prob = 0.475552161", abs(m5["self_distillation/target_top1_prob"] - 0.475552161) < 1e-6)
# §6.2 tail example: explicit top-3 + tail
cS_k3 = cfg(counterfactual_reference="student", distillation_topk=3, distillation_add_tail=True)
s62 = torch.tensor([.40, .30, .20]).log().view(1, 1, 3); t62 = torch.tensor([.20, .40, .20]).log().view(1, 1, 3)
_, m62 = run(cS_k3, student=s62, real=t62, null=None, null_lp=None, msk=m1, lpp=torch.zeros(1, 1), tl=torch.zeros(1, 1))
w62 = torch.tensor([0.0125, 1.264197530864, 0.20, 3.20]); q62 = w62 / w62.sum()
check("§6.2: 合并后 target = [.002672826, .270318429, .042765220, .684243524]",
      torch.allclose(q62, torch.tensor([.002672826, .270318429, .042765220, .684243524]), atol=1e-8))
check("§6.2: 生产 target_tail_mass = 0.684243524 且 argmax 为 tail",
      abs(m62["self_distillation/target_tail_mass"] - 0.684243524) < 1e-6
      and m62["self_distillation/target_tail_top1_frac"] == 1.0, f"{m62['self_distillation/target_tail_mass']:.9f}")
check("§6.2: 生产 teacher_tail_mass = 0.20", abs(m62["self_distillation/teacher_tail_mass"] - 0.20) < 1e-6)
# aggregation order is NOT interchangeable
tail_s = torch.tensor([.05, .03, .02]); tail_t = torch.tensor([.15, .03, .02])
per_tok = (tail_t * (tail_t / tail_s) ** BETA).sum().item()          # reconstruct each tail token, then merge
merged = (0.20 * (0.20 / 0.10) ** BETA)                              # merge first, then reconstruct
check("§6.2: 先重构再合并 (12.20) ≠ 先合并再重构 (3.20)", abs(per_tok - 12.20) < 1e-5 and abs(merged - 3.20) < 1e-9, f"{per_tok:.6f} vs {merged:.6f}")

# ---- 4. 同一参照张量 -> 两路径恒等 (数学恒等, 不是\"第0步 S≡A\") ----
print("\n[4] 参照张量相同时 A 路径与 S 路径恒等")
lA_s, _ = run(cfg(), null=student0.detach().clone())
lS_s, _ = runS()
check("null := student 张量时, A 路径 == S 路径 (bit-identical)", torch.equal(lA_s, lS_s), f"{lA_s.item():.10f} vs {lS_s.item():.10f}")

# ---- 5. 梯度卫生 ----
print("\n[5] 梯度卫生")
sP = student0.clone().requires_grad_(True)
lP, _ = runS(student=sP); lP.backward()
sR = student0.clone().requires_grad_(True)
lR = ref_S(real0, sR, mask, BETA); lR.backward()
check("∂loss/∂student 与 '目标为常数' 的参照逐元素一致 (<1e-6)", torch.allclose(sP.grad, sR.grad, atol=1e-6),
      f"max|Δ|={(sP.grad - sR.grad).abs().max().item():.2e}")
check("学生路径有梯度 (mixture 未被 no_grad 吞掉)", sP.grad.abs().sum().item() > 0)
sM = student0.clone().requires_grad_(True)
lM = ref_S(real0, sM, mask, BETA, mutant="no_detach"); lM.backward()
check("抓住变异: 参照未 detach -> 梯度不同", not torch.allclose(sP.grad, sM.grad, atol=1e-6),
      f"max|Δ|={(sP.grad - sM.grad).abs().max().item():.2e}")
check("参照未 detach 时 loss 数值相同 (差异只在梯度)", abs(lM.item() - lP.item()) < 1e-6)

# ---- 6. 有限性 ----
print("\n[6] 有限性")
ext_s = student0.clone(); ext_t = real0.clone()
ext_s[0, 0, 0] = math.log(1e-4); ext_t[0, 0, 0] = math.log(1e-2)       # unnormalised weight ~1e6
lX, mX = runS(student=ext_s, real=ext_t)
check("极端比值 (p+=1e-2, p_S=1e-4) 下 loss/target 有限", math.isfinite(lX.item()) and math.isfinite(mX["self_distillation/target_entropy"]))
pad_s = student0.clone(); pad_t = real0.clone()
pad_s[1, 3:, :] = float("-inf"); pad_t[1, 4:, :] = float("inf")
lPad, _ = runS(student=pad_s, real=pad_t)
check("padding 位 ±inf 不影响有效位 loss", torch.allclose(lPad, lS, atol=1e-7), f"{lPad.item():.8f} vs {lS.item():.8f}")
bad_s = student0.clone(); bad_s[0, 2, 1] = float("nan")
ok = False
try: runS(student=bad_s)
except FloatingPointError: ok = True
check("有效位 NaN -> 立即报错", ok)

# ---- 7. 守卫 ----
print("\n[7] 运行时守卫")
for kw, why in ((dict(counterfactual_u_clip_pos=True), "u_clip_pos"), (dict(counterfactual_st_enable=True), "st"),
                (dict(counterfactual_floor_alpha=0.1), "floor"), (dict(counterfactual_tanh_scale=0.5), "tanh"),
                (dict(counterfactual_target_gamma=2.0), "gamma≠1"), (dict(counterfactual_hist_adaptive_beta=True), "hist"),
                (dict(counterfactual_future_weight=True, counterfactual_u_clip_pos=True), "future"),
                (dict(counterfactual_hist_adaptive_beta=True, counterfactual_hist_shuffle=True), "shuffle")):
    ok = False
    try: runS(cfg(counterfactual_reference="student", **kw))
    except ValueError as e: ok = "mutually exclusive" in str(e)      # must be THE exclusion guard, not a downstream one
    check(f"拦住: student + {why} (互斥守卫本身触发)", ok)
ok = False
try: run(cS)                       # null tensors passed although reference=student
except ValueError: ok = True
check("拦住: student 模式却传入 null 张量", ok)
ok = False
try: runS(cfg(counterfactual_reference="foo"))
except ValueError: ok = True
check("拦住: 非法 reference", ok)
ok = False
try: runS(cfg(counterfactual_reference="student", counterfactual_null_mode=None))
except ValueError: ok = True
check("拦住: student + null_mode=None (否则静默退回 V0)", ok)

# ---- 8. 聚合 / mask 回归 ----
print("\n[8] 聚合与 mask")
m2 = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)
l8, _ = runS(msk=m2); r8 = ref_S(real0, student0, m2, BETA)
check("变长 mask 下与参照一致", abs(l8.item() - r8.item()) < 1e-5)
check("诊断指标范围: u_abs_mean>0, teacher_student_kl≥0, teacher_tail_mass∈(0,1), u_p90≥u_abs_mean 量级",
      mS["self_distillation/counterfactual_u_abs_mean"] > 0 and mS["self_distillation/teacher_student_kl"] >= -1e-7
      and 0 < mS["self_distillation/teacher_tail_mass"] < 1 and mS["self_distillation/counterfactual_u_p90"] > 0)

# ---- 10. 指标管道: 名称分派 + 计数聚合 (candidate_s_results §7.2/§7.6) ----
print("\n[10] 指标管道 append_to_dict -> reduce_metrics")
from verl.utils.py_functional import append_to_dict
from verl.utils.metric import reduce_metrics, Metric
import re as _re
_, mA2 = run(cfg()); _, mS2 = runS()
bad = [k for k in list(mA2) + list(mS2) if _re.search(r"max|min", k.split("/")[-1])]
check("loss 函数输出的普通指标名不含 max/min 子串", not bad, f"{bad}")
check("worst-mb 指标是显式 MAX 的 Metric 对象", isinstance(mS2["self_distillation/target_top1_prob_worst_mb"], Metric))
# §7.2 合成例: 四个等长 micro-batch, tail-top1 比例 [0,0,0.02,0.08] -> 全局 2.5%, 最坏 8%
agg = {}
for frac in (0.0, 0.0, 0.02, 0.08):
    append_to_dict(agg, {"self_distillation/target_tail_top1_sum": frac * 100, "self_distillation/target_stat_count": 100.0,
                         "self_distillation/target_argmax_tail_frac_OLD": frac,
                         "self_distillation/worst_mb": Metric(aggregation="max", value=frac)})
red = reduce_metrics(agg)
g = red["self_distillation/target_tail_top1_sum"] / red["self_distillation/target_stat_count"]
check("计数聚合: mean(sum)/mean(count) = 2.5%", abs(g - 0.025) < 1e-12, f"{g:.4f}")
check("旧命名 (含 argmax) 会被 reducer 取 max = 8% (复现问题)", abs(red["self_distillation/target_argmax_tail_frac_OLD"] - 0.08) < 1e-12)
check("显式 MAX Metric 给出最坏 micro-batch 8%", abs(red["self_distillation/worst_mb"] - 0.08) < 1e-12)
# 不等长 micro-batch: token 加权 vs 等权
agg = {}
for frac, n in ((0.10, 10), (0.0, 190)):
    append_to_dict(agg, {"self_distillation/target_tail_top1_sum": frac * n, "self_distillation/target_stat_count": float(n),
                         "self_distillation/target_tail_top1_frac": frac})
red = reduce_metrics(agg)
g = red["self_distillation/target_tail_top1_sum"] / red["self_distillation/target_stat_count"]
check("不等长 micro-batch: 计数聚合 = token 加权 0.5%, 等权均值 = 5% (两者区分开)",
      abs(g - 0.005) < 1e-12 and abs(red["self_distillation/target_tail_top1_frac"] - 0.05) < 1e-12)
# 生产 loss 的 sum/count 与逐位置手算一致 (S 配置, 变长 mask)
_, mS3 = runS(msk=m2)
_lt2, _ls2 = add_tail(real0), add_tail(student0)
_lq2 = torch.log_softmax(_lt2 + BETA * (_lt2 - _ls2.detach()), -1); _q2 = _lq2.exp()
_top1 = (_q2.amax(-1) * m2).sum().item(); _tail = ((_q2.argmax(-1) == K).float() * m2).sum().item()
check("生产 target_top1_prob_sum / target_tail_top1_sum / count 与手算一致",
      abs(mS3["self_distillation/target_top1_prob_sum"] - _top1) < 1e-5 and abs(mS3["self_distillation/target_tail_top1_sum"] - _tail) < 1e-9
      and mS3["self_distillation/target_stat_count"] == m2.sum().item())

# ---- 9. 与改动前代码 (sup@bea75c6) 逐比特回归: null 路径 loss 与梯度必须完全不变 ----
print("\n[9] 与 bea75c6 的 core_algos 逐比特回归 (reference=null 路径)")
import subprocess, importlib.util, tempfile
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    _src = subprocess.check_output(["git", "-C", _REPO, "show", "bea75c6:verl/trainer/ppo/core_algos.py"], text=True)
    _tmp = tempfile.NamedTemporaryFile("w", suffix="_core_algos_old.py", delete=False); _tmp.write(_src); _tmp.close()
    _spec = importlib.util.spec_from_file_location("core_algos_old", _tmp.name)
    _old = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_old)
    OLD = _old.compute_self_distillation_loss
    variants = {
        "A": {}, "sup": dict(counterfactual_u_clip_pos=True), "floor": dict(counterfactual_floor_alpha=0.1),
        "tanh": dict(counterfactual_tanh_scale=0.5), "gamma": dict(counterfactual_target_gamma=3.0),
        "Ahm": dict(counterfactual_hist_adaptive_beta=True, counterfactual_hist_mode="mean", counterfactual_hist_kappa=0.1),
        "X1": dict(counterfactual_hist_adaptive_beta=True, counterfactual_hist_mode="mean", counterfactual_hist_half="pos"),
        "Ahf": dict(counterfactual_hist_adaptive_beta=True, counterfactual_hist_mode="hf"),
        "E": dict(counterfactual_u_clip_pos=True, counterfactual_hist_adaptive_beta=True, counterfactual_future_weight=True),
        "st": dict(counterfactual_st_enable=True), "V0": dict(counterfactual_null_mode=None), "notail": dict(distillation_add_tail=False),
    }
    all_ok = True
    for name, kw in variants.items():
        for agg in ("token-mean", "seq-mean-token-mean"):
            c = cfg(**kw); c["is_clip"] = 2.0
            s1 = student0.clone().requires_grad_(True); s2 = student0.clone().requires_grad_(True)
            kws = dict(student_log_probs=lp, teacher_log_probs=tlp, response_mask=mask, self_distillation_config=c,
                       old_log_probs=lp + 0.05, teacher_topk_log_probs=real0, teacher_null_topk_log_probs=null0,
                       teacher_null_log_probs=nlp, self_distillation_mask=torch.ones(B), loss_agg_mode=agg)
            l1, m1 = OLD(student_topk_log_probs=s1, **kws); l1.backward()
            l2, m2 = compute_self_distillation_loss(student_topk_log_probs=s2, **kws); l2.backward()
            _bad = [k for k in set(m1) & set(m2) if m1[k] != m2[k]]
            ok = torch.equal(l1, l2) and torch.equal(s1.grad, s2.grad) and not _bad
            all_ok &= ok
            if not ok: print(f"    mismatch: {name} / {agg} loss_eq={torch.equal(l1, l2)} grad_eq={torch.equal(s1.grad, s2.grad)} keys={_bad}")
    check("12 配置 × 2 聚合: loss / 学生梯度 / 既有指标 与 bea75c6 bit-identical", all_ok)
    os.unlink(_tmp.name)
except Exception as _e:   # noqa: BLE001
    check("bea75c6 逐比特回归 (无法加载旧版本)", False, repr(_e))

print("\n" + "=" * 76)
print("全部通过" if not FAIL else "失败: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
