#!/usr/bin/env python
"""Layer-prior probe unit tests -- CPU only.
Spec: docs/negative_history/ir_layer_prior_probe_plan_2026-10-05.md §5.6 (items 1-9; item 10 = training
integration is not implemented yet and is reported as SKIP).
Groups: [1] read-only multi-boundary collector; [2] read-out math; [3] per-layer controls (l=L, suffix-IR
identity, own-D breaks it); [4] tail; [5] beta=0 / constant shift; [6] regression; [7] positions & answers;
[8] gradients; [9] aggregation; [10] SKIP; [11] view construction vs the trainer.
"""
import math, os, sys, types
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils.layer_prior_probe import (  # noqa: E402
    BoundaryCollector, RegressionAccumulator, aha_target, centered, coarsen_log_probs, coarsen_logits, corr_pieces,
    final_scale_vec, find_answer_token_positions, omega_from_log_p_plus, own_scale_vec, paired_bootstrap, pooled_corr,
    readout_logits, tv_from_log, weighted_stats,
)
from verl.utils.teacher_residual import build_internal_targets  # noqa: E402
import probe_layer_prior as P  # noqa: E402  (scripts/ is on sys.path via __file__ dir)

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: FAIL.append(name)
torch.manual_seed(7); np.random.seed(7)

# ---------------- synthetic Qwen3.5-like stack with the HF call signature used by score() ----------------
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
    def __init__(self, d, L, V):
        super().__init__(); self.embed = nn.Embedding(V, d); self.layers = nn.ModuleList([Block(d) for _ in range(L)]); self.norm = ZeroCenteredRMSNorm(d)
    def forward(self, input_ids):
        h = self.embed(input_ids)
        for blk in self.layers: h = blk(h)
        return self.norm(h)
class Root(nn.Module):
    def __init__(self, d=16, L=4, V=37, bias=False):
        super().__init__(); self.model = TextModel(d, L, V); self.lm_head = nn.Linear(d, V, bias=bias); self.device = torch.device("cpu")
    def forward(self, input_ids, attention_mask=None, logits_to_keep=0, **kw):
        logits = self.lm_head(self.model(input_ids))
        if logits_to_keep: logits = logits[:, -logits_to_keep:]
        return types.SimpleNamespace(logits=logits)
d, L, V, K = 16, 4, 37, 5
model = Root(d, L, V).eval()
for p_ in model.parameters(): p_.requires_grad_(False)
ids = torch.randint(0, V, (1, 11)); S = ids.shape[1]
mask = torch.zeros(S, dtype=torch.bool); mask[3:9] = True; n = int(mask.sum())

print("[1] read-only multi-boundary collector")
ref = model(ids).logits.clone()
coll = BoundaryCollector(model, mask)
with coll:
    out = model(ids).logits
coll.check_single_forward()
check("logits identical with hooks on", torch.equal(out, ref))
check("hooks removed after exit", len(coll._handles) == 0)
h = coll.stacked(); check("stacked shape [L+1, n, d]", tuple(h.shape) == (L + 1, n, d))
# h^0 = embeddings, h^l = input of block l, h^L = final pre-norm; verify against a manual pass
hm = model.model.embed(ids)[0]; manual = [hm[mask].float()]
for blk in model.model.layers:
    hm = blk(hm); manual.append(hm[mask].float())
check("boundary states match a manual pass", all(torch.allclose(h[l], manual[l], atol=1e-6) for l in range(L + 1)))
calls_before = list(coll.calls); model(ids)
check("no capture outside the context", coll.calls == calls_before)
try:
    bad = BoundaryCollector(model, torch.zeros(S + 1, dtype=torch.bool))
    with bad: model(ids)
    check("mask length mismatch raises", False)
except ValueError:
    check("mask length mismatch raises", True)
check("hooks removed after an exception inside the context", len(bad._handles) == 0 and torch.equal(model(ids).logits, ref))
coll.reset(); check("reset() clears every boundary buffer and counter", all(x is None for x in coll.h) and coll.calls == [0] * (L + 1))
with coll: model(ids)
h2 = coll.stacked(); check("second sample re-captures (no stale buffers)", torch.allclose(h2, h, atol=1e-6))

print("[2] read-out math")
D = final_scale_vec(h[L], coll.norm_weight, coll.norm_eps)
z_rec = readout_logits(D * h[L], coll.head_weight, 1.0, coll.head_bias)
check("h^L read-out reconstructs the model logits", torch.allclose(z_rec, ref[0][mask].float(), atol=1e-5), f"max {float((z_rec-ref[0][mask]).abs().max()):.2e}")
r_sum = sum(readout_logits(D * (h[j + 1] - h[j]), coll.head_weight, 1.0) for j in range(L))
z0 = readout_logits(D * h[0], coll.head_weight, 1.0, coll.head_bias)
check("sum of block contributions + embedding read-out = logits", torch.allclose(z0 + r_sum, z_rec, atol=1e-5))
mut = model.model.norm(h[L] - h[0])                       # WRONG variant: RMSNorm applied to the delta itself
z_mut = readout_logits(mut, coll.head_weight, 1.0)
check("RMSNorm(delta h) mutant is detected (differs from the exact contribution)", not torch.allclose(z_mut, r_sum, atol=1e-3))
check("tau scales the read-out once", torch.allclose(readout_logits(D * h[L], coll.head_weight, 2.0, coll.head_bias), z_rec / 2.0, atol=1e-6))
mb = Root(d, L, V, bias=True).eval(); cb = BoundaryCollector(mb, mask)
with cb: refb = mb(ids).logits
Db = final_scale_vec(cb.stacked()[L], cb.norm_weight, cb.norm_eps)
check("bias enters the anchor read-out exactly once", torch.allclose(readout_logits(Db * cb.stacked()[L], cb.head_weight, 1.0, cb.head_bias), refb[0][mask].float(), atol=1e-5))
check("bias never enters a contribution", torch.allclose(readout_logits(Db * (cb.stacked()[L] - cb.stacked()[0]), cb.head_weight, 1.0), readout_logits(Db * cb.stacked()[L], cb.head_weight, 1.0) - readout_logits(Db * cb.stacked()[0], cb.head_weight, 1.0), atol=1e-6))

print("[3] per-layer controls")
z_plus = ref[0][mask].float()
topk = z_plus.topk(K, -1).indices
logP = coarsen_logits(z_plus, topk)
z_null = z_plus + 0.7 * torch.randn_like(z_plus)
logP0 = coarsen_logits(z_null, topk)
beta = 4.0
QA = aha_target(logP, logP0, beta)
logPL = coarsen_logits(readout_logits(D * h[L], coll.head_weight, 1.0, coll.head_bias), topk)
QL = aha_target(logP, logPL, beta)
check("l=L prior reproduces P+ and the target collapses to P+ (plain privileged OPD)", torch.allclose(logPL, logP, atol=1e-5) and torch.allclose(QL, logP, atol=1e-5))
for l in (0, 2):
    z_l = readout_logits(D * h[l], coll.head_weight, 1.0, coll.head_bias)
    q_prior = torch.log_softmax(z_plus + beta * (z_plus - z_l), -1)             # full-vocab "layer l as prior"
    r_suffix = readout_logits(D * (h[L] - h[l]), coll.head_weight, 1.0)
    q_ir = torch.log_softmax(z_plus + beta * r_suffix, -1)                        # suffix IR [l, L) with lambda = beta
    check(f"full-vocab identity: layer-{l} prior == suffix IR [{l},{L}) at lambda=beta", torch.allclose(q_prior, q_ir, atol=1e-5))
    t_ir = build_internal_targets(z_plus, r_suffix, beta, topk, "full")["target_log_probs"]
    check(f"  ... and equals build_internal_targets(full) after coarsening", torch.allclose(coarsen_log_probs(q_prior, topk), t_ir, atol=1e-5))
    z_own = readout_logits(own_scale_vec(h[l], coll.norm_weight, coll.norm_eps) * h[l], coll.head_weight, 1.0, coll.head_bias)
    check(f"  own-D read-out does not satisfy the identity", not torch.allclose(torch.log_softmax(z_plus + beta * (z_plus - z_own), -1), q_ir, atol=1e-3))
    # aggregate-first (main criterion) vs full-vocab-first differ in general (plan §2.7 last paragraph)
    agg_first = aha_target(logP, coarsen_logits(z_l, topk), beta)
    check(f"  aggregate-first target differs from coarsened full-vocab target (documented non-commutation)", not torch.allclose(agg_first, coarsen_log_probs(q_prior, topk), atol=1e-4))

print("[4] tail")
check("explicit K + tail sums to 1", torch.allclose(logP.exp().sum(-1), torch.ones(n), atol=1e-6))
p64 = torch.log_softmax(z_plus.double(), -1).exp(); tail64 = torch.log(1.0 - torch.gather(p64, -1, topk).sum(-1))
check("tail log-prob matches an FP64 exact reference", torch.allclose(logP[:, -1].double(), tail64, atol=1e-5))
# aggregate-first reconstruction == the original A code path (add_tail then softmax(log p + beta u))
def add_tail(lp):
    ls = torch.logsumexp(lp, -1, keepdim=True).clamp(max=-1e-7); return torch.cat([lp, torch.log(-torch.expm1(ls))], -1)
lpK = torch.gather(torch.log_softmax(z_plus, -1), -1, topk); lp0K = torch.gather(torch.log_softmax(z_null, -1), -1, topk)
QA_code = torch.log_softmax(add_tail(lpK) + beta * (add_tail(lpK) - add_tail(lp0K)), -1)
check("aggregate-first target == original A code path (add_tail + beta*u + softmax)", torch.allclose(QA, QA_code, atol=1e-6))
u_zero_tail = torch.cat([(logP - logP0)[:, :-1], torch.zeros(n, 1)], -1)
check("tail log-ratio is NOT zeroed (differs from the tail_u_zero variant)", not torch.allclose(QA, torch.log_softmax(logP + beta * u_zero_tail, -1), atol=1e-4))
full_idx = torch.arange(V).unsqueeze(0).expand(n, V)
cK = coarsen_logits(z_plus, full_idx)
check("K == V gives a finite (very negative) tail", torch.isfinite(cK).all() and bool((cK[:, -1] < -10).all()))
peaked = torch.zeros(n, V); peaked[:, 0] = 60.0
check("tiny tail stays finite", torch.isfinite(coarsen_logits(peaked, topk)).all())

print("[5] beta = 0 and constant shifts")
check("beta=0 returns P+", torch.allclose(aha_target(logP, logP0, 0.0), logP, atol=1e-6))
z_shift = z_null + 3.21
check("a common constant on the full-vocab candidate leaves the coarsened prior unchanged", torch.allclose(coarsen_logits(z_shift, topk), logP0, atol=1e-6))
partial = logP0.clone(); partial[:, :K] += 0.5           # shifting only explicit columns after aggregation is NOT an invariance
check("shifting only the explicit columns after aggregation changes the target (so it is never done)", not torch.allclose(aha_target(logP, partial, beta), QA, atol=1e-4))

print("[6] global regression")
Lr, Kr, Nfit, Nho = 6, 9, 400, 200
w_true = torch.randn(Lr, dtype=torch.float64)
def synth(N, noise):
    X = torch.randn(N, Kr, Lr); om = torch.softmax(torch.randn(N, Kr), -1)
    u = torch.einsum("nkl,l->nk", X, w_true.float()) + noise * torch.randn(N, Kr) + torch.randn(N, 1)   # per-position offset
    return centered(X, om), centered(u, om), om
Xf, uf, of = synth(Nfit, 0.05); Xh, uh, oh = synth(Nho, 0.05)
reg = RegressionAccumulator(Lr); reg.add(Xf[:150], uf[:150], of[:150]); reg.add(Xf[150:], uf[150:], of[150:])
sol = reg.solve()
check("fittable with sane conditioning", sol["fittable"] and sol["rank"] == Lr and math.isfinite(sol["cond"]))
w = torch.tensor(sol["w"], dtype=torch.float64)
check("recovers the shared w on synthetic data", torch.allclose(w, w_true, atol=0.05), f"max err {float((w-w_true).abs().max()):.3f}")
res, tot = RegressionAccumulator.r2_fixed(Xh, uh, oh, w); r2 = 1 - res / tot
check("held-out R^2 high for the true structure", r2 > 0.95, f"R2={r2:.3f}, in-sample {sol['r2_in_sample']:.3f}")
X2 = Xf + torch.randn(Nfit, 1, Lr); u2 = uf + torch.randn(Nfit, 1)                                   # add per-position constants
reg2 = RegressionAccumulator(Lr); reg2.add(centered(X2, of), centered(u2, of), of)
check("per-position constants do not change the centred fit", torch.allclose(torch.tensor(reg2.solve()["w"], dtype=torch.float64), w, atol=1e-6))
reg0 = RegressionAccumulator(Lr); reg0.add(Xf, torch.zeros_like(uf), of)
check("zero target energy -> not fittable (no fabricated score)", reg0.solve()["fittable"] is False)
Hbefore = reg.H.clone(); RegressionAccumulator.r2_fixed(Xh, uh, oh, w)
check("evaluating a fixed w never touches the accumulated statistics", torch.equal(Hbefore, reg.H))
pp = RegressionAccumulator.r2_pieces(Xh, uh, oh, w)
check("per-position R^2 pieces sum to the batch sums (subset R^2 is exact)", pp.shape == (Nho, 2) and abs(float(pp[:, 0].sum()) - res) < 1e-3 and abs(float(pp[:, 1].sum()) - tot) < 1e-3)
cp = corr_pieces(uh, uh, oh); check("Corr_omega(u,u) = 1", abs(pooled_corr(cp) - 1.0) < 1e-9)
check("Corr undefined on zero energy", pooled_corr(corr_pieces(torch.zeros_like(uh), uh, oh)) is None)

print("[7] positions and answers")
class FakeTok:
    def __init__(self, toks): self.toks = toks
    def dec(self, k): return "".join(self.toks[:k])
t1 = FakeTok(["Let", " me", " see", ".", " <answer>", "C", "</answer>", "<|im_end|>"])
a1 = find_answer_token_positions(list(range(len(t1.toks))), t1.dec)
check("<answer> tag: inner token only, tags excluded", a1["found"] and a1["positions"] == [5] and a1["source"] == "answer_tag")
t2 = FakeTok(["(", "B", ")", " white", "<|im_end|>"]); a2 = find_answer_token_positions(list(range(5)), t2.dec)
check("leading option letter with parentheses", a2["found"] and a2["positions"] == [1] and a2["source"] == "leading_letter")
t3 = FakeTok(["C", ".", " orange", "<|im_end|>"]); a3 = find_answer_token_positions(list(range(4)), t3.dec)
check("leading 'C. orange' -> the letter token", a3["positions"] == [0])
t4 = FakeTok(["The", " answer", " is", " B", "<|im_end|>"]); a4 = find_answer_token_positions(list(range(5)), t4.dec)
check("a letter inside the reasoning text is never taken as the final answer", not a4["found"] and a4["source"] == "missing")
t4b = FakeTok(["The sky", " is blue", ".\n\n", "A", ".", " blue", "<|im_end|>"]); a4b = find_answer_token_positions(list(range(7)), t4b.dec)
check("last line 'A. blue' -> that letter token (rule 3)", a4b["found"] and a4b["positions"] == [3] and a4b["source"] == "last_line_letter")
t4c = FakeTok(["It is brown", ".\n", "**", "B", "**", "<|im_end|>"]); a4c = find_answer_token_positions(list(range(6)), t4c.dec)
check("last line '**B**' -> the letter token", a4c["found"] and a4c["positions"] == [3])
t4d = FakeTok(["Looking at it", ", B is wrong", " so maybe D", "<|im_end|>"]); a4d = find_answer_token_positions(list(range(4)), t4d.dec)
check("last line not starting with a letter -> missing", not a4d["found"])
t4e = FakeTok(["...\n\n", "Correct", " answer", ":", " **", "D", ".", " cable", "**", "<|im_end|>"]); a4e = find_answer_token_positions(list(range(10)), t4e.dec)
check("'Correct answer: **D. cable**' -> the letter token", a4e["found"] and a4e["positions"] == [5])
t4f = FakeTok(["...\n\n", "\u2705", " Final", " Answer", ":", " **", "D", "**", "<|im_end|>"]); a4f = find_answer_token_positions(list(range(9)), t4f.dec)
check("'Final Answer: **D**' -> the letter token", a4f["found"] and a4f["positions"] == [6])
t4g = FakeTok(["...\n\n", "**", "Correct Answer", ": C", ". winter", " boots", "**", "<|im_end|>"]); a4g = find_answer_token_positions(list(range(8)), t4g.dec)
check("'**Correct Answer: C. winter boots**' -> token containing C", a4g["found"] and a4g["positions"] == [3])
t4h = FakeTok(["...\n\n", "The answer is not A", " but B", "<|im_end|>"]); a4h = find_answer_token_positions(list(range(4)), t4h.dec)
check("'The answer is not A but B' -> missing (no final-answer line form)", not a4h["found"])
t5 = FakeTok(["<answer>", "</answer>"]); check("empty tag -> missing", not find_answer_token_positions([0, 1], t5.dec)["found"])
t6 = FakeTok(["<answer>", "pur", "ple", " cup", "</answer>"]); a6 = find_answer_token_positions(list(range(5)), t6.dec)
check("multi-token answer span", a6["positions"] == [1, 2, 3])
t7 = FakeTok(["AB", "C"]); a7 = find_answer_token_positions([0, 1], t7.dec)
check("'ABC' is not an option letter", not a7["found"])
# score(): positions P-1 .. P+T-2 predict y_0 .. y_{T-1}; the collector mask covers exactly those
enc = {"input_ids": ids[:, :7], "attention_mask": torch.ones(1, 7, dtype=torch.long), "_raw": ""}
resp = ids[0, 7:].tolist(); T = len(resp)
coll2 = BoundaryCollector(model)
z_sc = P.score(model, enc, resp, 1.0, collector=coll2)
full = model(ids).logits[0].float()
check("score() returns logits at P-1..P+T-2", torch.allclose(z_sc, full[6:6 + T], atol=1e-6) and z_sc.shape == (T, V))
check("collector mask == the same positions", torch.equal(coll2.mask.nonzero().flatten(), torch.arange(6, 6 + T)))
check("never reads the position of y_t itself", not torch.allclose(z_sc[0], full[7], atol=1e-6))
check("tau divides score() once", torch.allclose(P.score(model, enc, resp, 2.0), z_sc / 2.0, atol=1e-6))
class RootMM(Root):   # records the kwargs it was called with
    def forward(self, input_ids, attention_mask=None, logits_to_keep=0, **kw):
        self.seen = {"input_ids": input_ids, "attention_mask": attention_mask, **kw}; return super().forward(input_ids, attention_mask, logits_to_keep, **kw)
mm = RootMM(d, L, V).eval(); mm.load_state_dict(model.state_dict())
enc_mm = dict(enc); enc_mm["mm_token_type_ids"] = torch.tensor([[0, 1, 1, 1, 0, 0, 0]])
P.score(mm, enc_mm, resp, 1.0)
check("mm_token_type_ids extended with zeros for the response", mm.seen["mm_token_type_ids"].tolist() == [[0, 1, 1, 1, 0, 0, 0] + [0] * T] and mm.seen["attention_mask"].shape[1] == 7 + T and mm.seen.get("use_cache") is False)
try:
    P.score(mm, {"input_ids": ids[:, :7].repeat(2, 1), "attention_mask": torch.ones(2, 7, dtype=torch.long)}, resp, 1.0); check("batch dim != 1 is rejected", False)
except RuntimeError:
    check("batch dim != 1 is rejected", True)
class RootBad(Root):   # simulates a transformers version that ignores logits_to_keep
    def forward(self, input_ids, attention_mask=None, logits_to_keep=0, **kw):
        return super().forward(input_ids, attention_mask, 0, **kw)
try:
    P.score(RootBad(d, L, V).eval(), enc, resp, 1.0); check("logits_to_keep semantics are asserted (shape check)", False)
except RuntimeError:
    check("logits_to_keep semantics are asserted (shape check)", True)
gen, trunc = P.generate(types.SimpleNamespace(device=torch.device("cpu"), generate=lambda **kw: torch.cat([kw["input_ids"], torch.tensor([[5, 9, 2, 4]])], 1)), enc, 4, [2], 0)
if gen is not None:
    check("generate() cuts at the first EOS (inclusive) and flags truncation", gen == [5, 9, 2] and trunc is False)
gen2, trunc2 = P.generate(types.SimpleNamespace(device=torch.device("cpu"), generate=lambda **kw: torch.cat([kw["input_ids"], torch.tensor([[5, 9, 4, 4]])], 1)), enc, 4, [2], 0)
check("generate() without EOS -> truncated=True, full output kept", gen2 == [5, 9, 4, 4] and trunc2 is True)
t8 = FakeTok(["caf", "\u00e9", "\n", "A", "<|im_end|>"])
a8 = find_answer_token_positions(list(range(5)), lambda k: ("caf\ufffd" if k == 1 else "".join(t8.toks[:k])))
check("drifting prefix decodes -> 'unmappable', never a wrong position", a8["source"] == "unmappable" and a8["positions"] == [])

print("[8] gradients")
student_logits = torch.randn(n, V, requires_grad=True)
def jsd_loss(target_log, student_logits):
    s = coarsen_logits(student_logits, topk); m = torch.logaddexp(s, target_log) - math.log(2)
    return 0.5 * (F.kl_div(m, s, log_target=True, reduction="sum") + F.kl_div(m, target_log, log_target=True, reduction="sum"))
tA = QA.detach(); tB = torch.log_softmax(logP + beta * (logP - logP0), -1).detach()   # same numbers, two constructions
gA = torch.autograd.grad(jsd_loss(tA, student_logits), student_logits)[0]
gB = torch.autograd.grad(jsd_loss(tB, student_logits), student_logits)[0]
check("identical detached targets -> identical student gradients", torch.allclose(gA, gB, atol=1e-7))
check("target carries no gradient", not QA.requires_grad and not tA.requires_grad)
check("student branch has gradient", gA.abs().sum() > 0)
zp_req = z_plus.clone().requires_grad_(True)
q_req = aha_target(coarsen_logits(zp_req, topk), logP0, beta)
check("teacher-side quantities would carry grad only if asked (we detach via no_grad in the script)", q_req.requires_grad)

print("[9] aggregation")
x1 = torch.rand(7); x2 = torch.rand(13); qid = torch.tensor([0] * 7 + [1] * 13)
ws = weighted_stats(torch.cat([x1, x2]), qid)
check("token mean = sum/count over unequal micro-batches", abs(ws["token_mean"] - float((x1.double().sum() + x2.double().sum()) / 20)) < 1e-6)
check("per-question mean is the mean of question means (not the token mean)", abs(ws["question_mean"] - float((x1.double().mean() + x2.double().mean()) / 2)) < 1e-6)
check("p90 is global", abs(ws["token_p90"] - float(torch.quantile(torch.cat([x1, x2]).double(), 0.9))) < 1e-9)
diff = torch.cat([torch.full((7,), 0.3), torch.full((13,), -0.1)]); bs = paired_bootstrap(diff, qid, 200, 0)
check("bootstrap over groups: mean is token-weighted, CI spans both group values", abs(bs["mean"] - float(diff.double().mean())) < 1e-6 and bs["ci95"][0] <= -0.1 + 1e-6 and bs["ci95"][1] >= 0.3 - 1e-6 and not bs["significant"])

print("[10] training integration: SKIP (internal_prior target mode not implemented yet; see plan §5.2)")

print("[11] view construction vs the trainer")
from PIL import Image
imgs = [Image.new("RGB", (8, 6), (10, 20, 30)), Image.fromarray(np.random.randint(0, 255, (6, 8, 3), dtype=np.uint8))]
nullv = P.mean_colour_last(imgs)
mean_rgb = tuple(np.rint(np.asarray(imgs[1], dtype=np.float32).mean(axis=(0, 1))).astype(np.uint8).tolist())
check("null_scope=last: first image kept, last replaced by its mean colour", nullv[0] is imgs[0] and nullv[1].size == imgs[1].size and nullv[1].getpixel((0, 0)) == mean_rgb)
try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    ref_null = RayPPOTrainer._mean_color_teacher_images(imgs, null_scope="last")
    check("  == RayPPOTrainer._mean_color_teacher_images(null_scope='last')", np.array_equal(np.asarray(ref_null[1]), np.asarray(nullv[1])) and np.array_equal(np.asarray(ref_null[0]), np.asarray(nullv[0])))
    msgs_ref = RayPPOTrainer._build_teacher_messages_from_template(RayPPOTrainer, [{"role": "user", "content": "<image>\nZoomed: <image>\nQ?"}], imgs)
    msgs_new = P.messages_from_template("<image>\nZoomed: <image>\nQ?", imgs)
    same = [(a["type"], a.get("text")) for a in msgs_ref[0]["content"]] == [(b["type"], b.get("text")) for b in msgs_new[0]["content"]]
    check("  teacher message structure == trainer's _build_teacher_messages_from_template", same)
except Exception as e:  # ray etc. not importable here
    if os.environ.get("PROBE_TEST_STRICT") == "1":
        check(f"trainer comparison importable ({type(e).__name__})", False)
    else:
        print(f"  SKIP  trainer comparison ({type(e).__name__})")
m_txt, txt = P.text_only_messages("<image>\nWhat color?\n\nA. x\nB. y")
check("text view drops the placeholder and keeps the question verbatim", txt == "What color?\n\nA. x\nB. y" and m_txt[0]["content"][0]["type"] == "text")
rows = [{"images": [{"path": f"img{i % 4}.jpg"}], "_row": i} for i in range(12)]
sp = P.assign_splits(rows, (6, 3, 3), 1)
grp = {g: {r["_split"] for r in rows if r["images"][0]["path"] == g} for g in {r["images"][0]["path"] for r in rows}}
check("splits keep every image group intact", all(len(v) == 1 for v in grp.values()) and sum(len(v) for v in sp.values()) == 12)

print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAIL: {FAIL}")
sys.exit(1 if FAIL else 0)
