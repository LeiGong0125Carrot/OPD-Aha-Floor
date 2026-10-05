#!/usr/bin/env python
"""Route 1 (internal residual reconstruction) unit tests -- CPU only.

Spec: docs/negative_history/teacher_internal_residual_reconstruction_plan.md §4-§6, §15.1.
Groups: [1] read-out math on a synthetic residual net (hooks, D_t, mutants, read-only);
        [2] target construction (doc §5 example, full-vocab -> K+tail, λ=0, tail edge cases);
        [3] loss integration (reference impl, λ=0 == plain OPD, gradients, guards);
        [4] packed response-position mask (predict y_t at P+t-1).
"""
import math, os, sys, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402
from verl.utils.teacher_residual import ResidualCapture, build_internal_targets, find_text_model_and_head  # noqa: E402

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: FAIL.append(name)

torch.manual_seed(3)

# ---------------- synthetic Qwen3.5-like stack ----------------
class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(0.1 * torch.randn(d)); self.eps = eps
    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)

class Block(nn.Module):
    def __init__(self, d):
        super().__init__(); self.lin = nn.Linear(d, d)
    def forward(self, hidden_states, **kw):
        return hidden_states + torch.tanh(self.lin(hidden_states))

class TextModel(nn.Module):
    def __init__(self, d, L):
        super().__init__(); self.layers = nn.ModuleList([Block(d) for _ in range(L)]); self.norm = ZeroCenteredRMSNorm(d)
    def forward(self, h):
        for layer in self.layers: h = layer(h)
        return self.norm(h)

class Inner(nn.Module):
    def __init__(self, d, L):
        super().__init__(); self.language_model = TextModel(d, L)

class Root(nn.Module):
    def __init__(self, d, V, L, bias=False):
        super().__init__(); self.model = Inner(d, L); self.lm_head = nn.Linear(d, V, bias=bias)
    def forward(self, h):
        return self.lm_head(self.model.language_model(h))

d, V, L, N = 16, 23, 6, 9
root = Root(d, V, L).eval()
for p_ in root.parameters(): p_.requires_grad_(False)
h0 = torch.randn(1, N, d)
mask = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=torch.bool)
TAU = 1.3

print("=" * 76); print("Route 1 内部残差重构 单测"); print("=" * 76)

# ---- 1. read-out math ----
print("\n[1] 读出数学 (合成残差网络)")
tm, nm, head = find_text_model_and_head(root)
check("定位 text_model / norm / lm_head", tm is root.model.language_model and head is root.lm_head)
a, b = 2, 5
with torch.no_grad():
    logits_plain = root(h0)
    cap = ResidualCapture(root, a, b, TAU, response_mask_packed=mask)
    with cap: logits_cap = root(h0)
check("只读 hook 不改变 logits (bit-identical)", torch.equal(logits_plain, logits_cap))
check("四个 hook 各触发一次", all(v == 1 for v in cap.calls.values()), f"{cap.calls}")
# manual reference
with torch.no_grad():
    hs = [h0]
    for layer in tm.layers: hs.append(layer(hs[-1]))
    hL = hs[-1]; delta = hs[b] - hs[a]                     # = sum of per-block updates in [a,b)
    per_block = sum(hs[j + 1] - hs[j] for j in range(a, b))
    check("Δh = 逐 block 更新之和 (端点差)", torch.allclose(delta, per_block, atol=1e-6))
    s = torch.rsqrt(hL.float().pow(2).mean(-1, keepdim=True) + nm.eps)
    D = (1.0 + nm.weight.float()) * s
    r_ref = F.linear((delta.float() * D)[0][mask], head.weight.float()) / TAU
check("r = τ⁻¹ W D_t Δh 与独立计算一致 (<1e-6)", torch.allclose(cap.r, r_ref, atol=1e-6), f"max|Δ|={(cap.r - r_ref).abs().max():.2e}")
# mutant: RMSNorm(delta) (normalising the difference by its own scale) must differ
with torch.no_grad():
    r_mut = F.linear(nm(delta)[0][mask].float(), head.weight.float()) / TAU
check("抓住变异: 用 RMSNorm(Δh) 代替 D_t·Δh", not torch.allclose(cap.r, r_mut, atol=1e-4))
# the anchor logits are reproduced by the same read-out of h^L (sanity of D_t)
with torch.no_grad():
    z_rebuilt = F.linear((hL.float() * D)[0], head.weight.float())
check("用 h^L·D_t·W 重建最终 logits", torch.allclose(z_rebuilt, logits_plain[0].float(), atol=1e-5))
# bias must not enter r
rootb = Root(d, V, L, bias=True).eval(); rootb.load_state_dict(root.state_dict(), strict=False)
with torch.no_grad():
    rootb.lm_head.bias.copy_(torch.randn(V)); capb = ResidualCapture(rootb, a, b, TAU, mask)
    with capb: rootb(h0)
check("LM-head bias (若存在) 不进入 r -- Qwen3.5 本身无 bias, 此为合成模型上的构造性检查", torch.allclose(capb.r, cap.r, atol=1e-6))
# contribution of the whole stack == z - head(D*h0)
with torch.no_grad():
    capall = ResidualCapture(root, 0, L, 1.0, mask)
    with capall: root(h0)
    z_full = logits_plain[0][mask].float(); z_in = F.linear((h0.float() * D)[0][mask], head.weight.float())
check("[0,L) 端点 hook 放置正确: 贡献 = z − W·D_t·h^0 (同一 D_t 下的恒等式, 检验的是 hook 位置)", torch.allclose(capall.r, z_full - z_in, atol=1e-5))
# temperature scaling
with torch.no_grad():
    cap1 = ResidualCapture(root, a, b, 1.0, mask)
    with cap1: root(h0)
check("τ≠1 时 r 同步缩放 (r_τ = r_1/τ)", torch.allclose(cap.r, cap1.r / TAU, atol=1e-6))
# guards
for args, why in (((5, 2), "a>=b"), ((0, L + 1), "b>L"), ((-1, 3), "a<0")):
    ok = False
    try: ResidualCapture(root, args[0], args[1], 1.0, mask)
    except ValueError: ok = True
    check(f"拦住: 非法区间 {why}", ok)
with torch.no_grad():
    capx = ResidualCapture(root, a, b, 1.0, torch.ones(5, dtype=torch.bool)); ok = False
    try:
        with capx: root(h0)
    except ValueError: ok = True
check("拦住: 掩码长度与 packed token 数不符", ok)
check("退出上下文后 hook 已注销 (再前向不触发)", (lambda: (root(h0), cap.calls["head"])[1] == 1)())

# ---- 2. target construction ----
print("\n[2] 目标构造")
# doc §5: p+ = [.3,.6,.1], r = [.35,-.25,0], λ=2, τ=1, K=V
z = torch.log(torch.tensor([[0.30, 0.60, 0.10]])); r5 = torch.tensor([[0.35, -0.25, 0.0]])
out = build_internal_targets(z, r5, 2.0, torch.tensor([[0, 1, 2]]), "full")
q = out["target_log_probs"].exp()[0]
check("§5 三词例 q = [0.565637459, 0.340733458, 0.093629083]",
      torch.allclose(q[:3], torch.tensor([0.565637459, 0.340733458, 0.093629083]), atol=1e-7), f"{q[:3].tolist()}")
check("§5 K=V 时 tail ≈ 0 且有限", q[3].item() < 1e-6 and math.isfinite(out["target_log_probs"][0, 3].item()))
check("§5 striped/diamond odds = 0.5·e^{1.2} ≈ 1.660058", abs((q[0] / q[1]).item() - 1.660058) < 1e-5)
# full-vocab -> coarsen vs explicit
Vb, Kb, n = 11, 4, 5
zb = torch.randn(n, Vb); rb = 0.5 * torch.randn(n, Vb); tk = torch.stack([torch.randperm(Vb)[:Kb] for _ in range(n)])
outb = build_internal_targets(zb, rb, 1.0, tk, "full")
qV = torch.softmax(zb + rb, -1)
qK = torch.gather(qV, -1, tk); qtail = 1.0 - qK.sum(-1, keepdim=True)
check("full: 显式项 = softmax 后 gather, tail = 补集总质量", torch.allclose(outb["target_log_probs"].exp(), torch.cat([qK, qtail], -1), atol=1e-6))
check("full: 目标归一化", torch.allclose(outb["target_log_probs"].exp().sum(-1), torch.ones(n), atol=1e-6))
check("r_topk = r 在 top-k 上的 gather", torch.equal(outb["r_topk"], torch.gather(rb, -1, tk)))
outa = build_internal_targets(zb, rb, 1.0, tk, "aha")
check("aha: 不返回 target, 只返回 r_topk", "target_log_probs" not in outa and torch.equal(outa["r_topk"], outb["r_topk"]))
# λ=0 -> coarsened p+
out0 = build_internal_targets(zb, rb, 0.0, tk, "full"); pV = torch.softmax(zb, -1)
check("λ=0: 目标 = 压缩后的 p+", torch.allclose(out0["target_log_probs"].exp(), torch.cat([torch.gather(pV, -1, tk), 1 - torch.gather(pV, -1, tk).sum(-1, keepdim=True)], -1), atol=1e-6))
# tiny tail stability
zt = torch.tensor([[30.0, 29.0, -40.0, -41.0]]); outt = build_internal_targets(zt, torch.zeros_like(zt), 1.0, torch.tensor([[0, 1]]), "full")
# numerical policy (plan §6.3): the tail gap is clamped at -1e-7 exactly like the legacy add_tail(), so a
# vanishing tail floors at ~1e-7 instead of underflowing; finite, monotone, and recorded as a declared floor.
_tail = outt["target_log_probs"][0, 2].exp().item(); _true = torch.softmax(zt, -1)[0, 2:].sum().item()
check("极小 tail 稳定: 有限且落在 [真值, 1e-7 钳位] 内 (与 legacy add_tail 同一数值政策)",
      torch.isfinite(outt["target_log_probs"]).all() and _true <= _tail <= 1.01e-7, f"tail={_tail:.3e} true={_true:.3e}")
for kw, why in ((dict(lam=-1.0, tp="full"), "λ<0"), (dict(lam=float("nan"), tp="full"), "λ=nan"), (dict(lam=1.0, tp="foo"), "非法 tail_policy")):
    ok = False
    try: build_internal_targets(zb, rb, kw["lam"], tk, kw["tp"])
    except ValueError: ok = True
    check(f"拦住: {why}", ok)

# ---- 3. loss integration ----
print("\n[3] loss 集成")
class Cfg(dict):
    __getattr__ = dict.get
def cfg(**kw):
    dd = {"full_logit_distillation": True, "distillation_topk": 8, "distillation_add_tail": True,
          "renorm_topk_log_probs": False, "alpha": 0.5, "is_clip": None,
          "counterfactual_null_mode": None, "counterfactual_extrapolation_beta": 4.0,
          "counterfactual_reference": "null", "counterfactual_u_clip_pos": False, "counterfactual_st_enable": False,
          "counterfactual_floor_alpha": 0.0, "counterfactual_tanh_scale": 0.0, "counterfactual_target_gamma": 1.0,
          "counterfactual_hist_adaptive_beta": False, "counterfactual_hist_alpha": 1.0, "counterfactual_hist_shuffle": False,
          "counterfactual_future_weight": False, "counterfactual_future_alpha": 1.0, "log_prob_dump_dir": None,
          "teacher_target_mode": "legacy", "teacher_internal": {"start_block": 2, "end_block_exclusive": 5, "strength": 1.0, "tail_policy": "full"}}
    dd.update(kw); return Cfg(dd)
def add_tail(x):
    s_ = torch.logsumexp(x, -1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([x, torch.log(-torch.expm1(s_))], -1)
def jsd_rows(log_q, log_s):
    q_, ps = log_q.exp(), log_s.exp(); m = 0.5 * ps + 0.5 * q_; logm = (m + 1e-30).log()
    return 0.5 * (ps * (log_s - logm)).sum(-1) + 0.5 * (q_ * (log_q - logm)).sum(-1)

B, T, K, VV = 2, 6, 8, 20
zT = torch.randn(B, T, VV) * 1.5                      # teacher full logits (/tau)
rT = 0.4 * torch.randn(B, T, VV)                     # internal read-out
tkS = torch.stack([torch.stack([torch.randperm(VV)[:K] for _ in range(T)]) for _ in range(B)])
teacher_topk = torch.gather(torch.log_softmax(zT, -1), -1, tkS)                 # [B,T,K]
student_topk = torch.log_softmax(torch.randn(B, T, VV), -1).gather(-1, tkS)
lp = torch.randn(B, T) * 0.1 - 1.0; tlp = torch.randn(B, T) * 0.3 - 1.2
maskBT = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.float32)
itf = build_internal_targets(zT.reshape(-1, VV), rT.reshape(-1, VV), 1.0, tkS.reshape(-1, K), "full")
target_full = itf["target_log_probs"].reshape(B, T, K + 1); r_topk = itf["r_topk"].reshape(B, T, K)

def run(c, student=student_topk, target=None, rk=None, nulls=False):
    return compute_self_distillation_loss(
        student_log_probs=lp, teacher_log_probs=tlp, response_mask=maskBT, self_distillation_config=c,
        old_log_probs=lp.clone(), student_topk_log_probs=student, teacher_topk_log_probs=teacher_topk,
        teacher_null_topk_log_probs=(torch.randn(B, T, K) if nulls else None), teacher_null_log_probs=(tlp if nulls else None),
        teacher_internal_target_log_probs=target, teacher_internal_r_topk=rk,
        self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")

cF = cfg(teacher_target_mode="internal_residual")
cA = cfg(teacher_target_mode="internal_residual", teacher_internal={"start_block": 2, "end_block_exclusive": 5, "strength": 1.0, "tail_policy": "aha"})
lF, mF = run(cF, target=target_full, rk=r_topk)
log_s = add_tail(student_topk)
rF = (jsd_rows(target_full, log_s) * maskBT).sum() / maskBT.sum()
check("IRf loss 与独立参照 JSD(q_full, p_S) 一致 (<1e-5)", abs(lF.item() - rF.item()) < 1e-5, f"{lF.item():.8f} vs {rF.item():.8f}")
lA, mA = run(cA, rk=r_topk)
qA = torch.log_softmax(add_tail(teacher_topk) + torch.cat([1.0 * r_topk, torch.zeros(B, T, 1)], -1), -1)
rA = (jsd_rows(qA, log_s) * maskBT).sum() / maskBT.sum()
check("IRa loss 与独立参照 (压缩后 tilt, tail r=0) 一致 (<1e-5)", abs(lA.item() - rA.item()) < 1e-5, f"{lA.item():.8f} vs {rA.item():.8f}")
check("IRf ≠ IRa (tail 顺序不同是真实差异)", abs(lF.item() - lA.item()) > 1e-6)
check("指标: teacher_target_mode_internal=1, internal_lambda=1, tail flag", mF["self_distillation/teacher_target_mode_internal"] == 1.0 and mF["self_distillation/internal_lambda"] == 1.0 and mF["self_distillation/internal_tail_is_full"] == 1.0 and mA["self_distillation/internal_tail_is_full"] == 0.0)
check("指标: target_tv/top1/tail 计数式字段存在", all(f"self_distillation/{k}" in mF for k in ("counterfactual_target_tv", "target_top1_prob_sum", "target_tail_top1_sum", "target_stat_count", "internal_r_abs_sum")))
# λ = 0 -> plain privileged OPD
c0 = cfg(); l0, _ = run(c0)
it0 = build_internal_targets(zT.reshape(-1, VV), rT.reshape(-1, VV), 0.0, tkS.reshape(-1, K), "full")
lF0, _ = run(cfg(teacher_target_mode="internal_residual", teacher_internal={"start_block": 2, "end_block_exclusive": 5, "strength": 0.0, "tail_policy": "full"}), target=it0["target_log_probs"].reshape(B, T, K + 1), rk=it0["r_topk"].reshape(B, T, K))
lA0, _ = run(cfg(teacher_target_mode="internal_residual", teacher_internal={"start_block": 2, "end_block_exclusive": 5, "strength": 0.0, "tail_policy": "aha"}), rk=r_topk)
check("λ=0: IRf ≡ 普通特权 OPD (<1e-6)", abs(lF0.item() - l0.item()) < 1e-6, f"{lF0.item():.8f} vs {l0.item():.8f}")
check("λ=0: IRa ≡ 普通特权 OPD (<1e-6)", abs(lA0.item() - l0.item()) < 1e-6, f"{lA0.item():.8f} vs {l0.item():.8f}")
# legacy regression pinned to the pre-route-1 commit (187a952): loss, student grad and metrics must be unchanged
import subprocess, importlib.util, tempfile
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    _src = subprocess.check_output(["git", "-C", _REPO, "show", "187a952:verl/trainer/ppo/core_algos.py"], text=True)
    _tmp = tempfile.NamedTemporaryFile("w", suffix="_core_algos_old.py", delete=False); _tmp.write(_src); _tmp.close()
    _spec = importlib.util.spec_from_file_location("core_algos_old_ir", _tmp.name); _old = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_old)
    OLD = _old.compute_self_distillation_loss
    _ok = True
    _null_topk = teacher_topk - 0.1
    for kw in ({}, dict(counterfactual_null_mode="mean_color"), dict(counterfactual_null_mode="mean_color", counterfactual_u_clip_pos=True),
               dict(counterfactual_null_mode="mean_color", counterfactual_reference="student")):
        c = cfg(**kw); _nulls = kw.get("counterfactual_null_mode") is not None and kw.get("counterfactual_reference", "null") == "null"
        res = []
        for fn in (OLD, compute_self_distillation_loss):
            st_ = student_topk.clone().requires_grad_(True)
            l_, m_ = fn(student_log_probs=lp, teacher_log_probs=tlp, response_mask=maskBT, self_distillation_config=c, old_log_probs=lp.clone(),
                        student_topk_log_probs=st_, teacher_topk_log_probs=teacher_topk,
                        teacher_null_topk_log_probs=_null_topk if _nulls else None, teacher_null_log_probs=(tlp - 0.1) if _nulls else None,
                        self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")
            l_.backward(); res.append((l_.detach(), st_.grad.clone(), m_))
        (l1, g1, m1), (l2, g2, m2) = res
        def _val(x):  # Metric objects compare by aggregated value
            return x.aggregate() if hasattr(x, "aggregate") else x
        _bad = [k for k in set(m1) & set(m2) if _val(m1[k]) != _val(m2[k])]
        if not (torch.equal(l1, l2) and torch.equal(g1, g2) and not _bad):
            _ok = False; print(f"    mismatch for {kw}: loss_eq={torch.equal(l1, l2)} grad_eq={torch.equal(g1, g2)} keys={_bad}")
    os.unlink(_tmp.name)
except Exception as _e:  # noqa: BLE001
    _ok = False; print("    legacy-pin error:", repr(_e))
check("legacy 路径 (A/sup/S/普通OPD) 与 187a952 的 loss / 学生梯度 / 指标 bit-identical", _ok)

# gradients: target is a constant
sG = student_topk.clone().requires_grad_(True); lG, _ = run(cF, student=sG, target=target_full, rk=r_topk); lG.backward()
sR = student_topk.clone().requires_grad_(True); lR = (jsd_rows(target_full.detach(), add_tail(sR)) * maskBT).sum() / maskBT.sum(); lR.backward()
check("IRf 梯度与常数目标参照一致", torch.allclose(sG.grad, sR.grad, atol=1e-6))
sG2 = student_topk.clone().requires_grad_(True); lG2, _ = run(cA, student=sG2, rk=r_topk); lG2.backward()
sR2 = student_topk.clone().requires_grad_(True); lR2 = (jsd_rows(qA.detach(), add_tail(sR2)) * maskBT).sum() / maskBT.sum(); lR2.backward()
check("IRa 梯度与常数目标参照一致", torch.allclose(sG2.grad, sR2.grad, atol=1e-6))
check("目标无梯度 (teacher_topk/r 不带 grad)", not teacher_topk.requires_grad and not r_topk.requires_grad)
# guards
def expect(c, why, msg, **kw):
    ok = False
    try: run(c, **kw)
    except (ValueError, FloatingPointError) as e: ok = msg in str(e)
    check(f"拦住: {why}", ok)
expect(cfg(teacher_target_mode="internal_residual", counterfactual_null_mode="mean_color"), "internal + null_mode 设置", "null-free", target=target_full, rk=r_topk)
expect(cfg(teacher_target_mode="internal_residual", counterfactual_reference="student"), "internal + student 参照", "null-free", target=target_full, rk=r_topk)
expect(cfg(teacher_target_mode="internal_residual", counterfactual_hist_adaptive_beta=True), "internal + hist", "mutually exclusive", target=target_full, rk=r_topk)
expect(cF, "internal(full) 缺目标张量", "teacher_internal_target_log_probs", rk=r_topk)
expect(cF, "internal 缺 r_topk", "teacher_internal_r_topk", target=target_full)
expect(cF, "internal 却传入 null 张量", "null-teacher tensors", target=target_full, rk=r_topk, nulls=True)
expect(cfg(teacher_target_mode="internal_residual", teacher_internal={"start_block": 2, "end_block_exclusive": 5, "strength": None, "tail_policy": "full"}), "λ 未设置", "strength", target=target_full, rk=r_topk)
expect(cfg(teacher_target_mode="internal_residual", teacher_internal={"start_block": 2, "end_block_exclusive": 5, "strength": -1.0, "tail_policy": "full"}), "λ<0", "strength", target=target_full, rk=r_topk)
expect(cfg(teacher_target_mode="internal_residual", teacher_internal={"start_block": 2, "end_block_exclusive": 5, "strength": 1.0, "tail_policy": "foo"}), "非法 tail_policy", "tail_policy", target=target_full, rk=r_topk)
expect(cfg(teacher_target_mode="foo"), "非法 teacher_target_mode", "teacher_target_mode")
bad = target_full.clone(); bad[0, 1, 0] = float("nan")
expect(cF, "目标含 NaN", "not finite/normalised", target=bad, rk=r_topk)
unnorm = target_full + 0.5
expect(cF, "目标未归一化", "not finite/normalised", target=unnorm, rk=r_topk)

# ---- 4. packed response-position mask ----
print("\n[4] packed 回答位置掩码 (预测 y_t 的位置 = P+t−1)")
try:
    from verl.utils.attention_utils import unpad_input
    Bm, Sm, Tm = 2, 10, 3
    attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    rsi = torch.tensor([6, 6])            # response starts at index 6 -> predicting positions 5,6,7
    _, indices, *_ = unpad_input(torch.arange(Bm * Sm).reshape(Bm, Sm, 1), attn)
    rp = rsi.unsqueeze(1) - 1 + torch.arange(Tm).unsqueeze(0)
    full = torch.zeros(Bm, Sm, dtype=torch.bool); full[torch.arange(Bm).unsqueeze(1), rp] = True
    packed = full.reshape(-1)[indices]
    flat_ids = torch.arange(Bm * Sm)[indices]
    sel = flat_ids[packed].tolist()
    check("掩码选中 [5,6,7] 与 [15,16,17] (每条样本预测 y_0..y_2 的位置)", sel == [5, 6, 7, 15, 16, 17], f"{sel}")
    check("掩码不含 y_t 自身所在位置 (无泄漏)", 8 not in sel and 18 not in sel)
    check("padding 位不在掩码内", 9 not in sel and 18 not in sel and 19 not in sel)
except Exception as e:  # noqa: BLE001
    check("packed 掩码测试 (unpad_input 不可用则跳过)", False, repr(e))

print("\n" + "=" * 76)
print("全部通过" if not FAIL else "失败: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
