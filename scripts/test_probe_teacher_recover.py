#!/usr/bin/env python
"""Tests for scripts/probe_teacher_recover.py (CPU; analyze needs the cached Qwen tokenizer). 用法: python scripts/test_probe_teacher_recover.py"""
import json, os, sys, tempfile, types
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import probe_teacher_recover as R

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); cond or fails.append(name)

g = torch.Generator().manual_seed(0)
a = torch.log_softmax(torch.randn(5, 50, generator=g), -1); b = torch.log_softmax(torch.randn(5, 50, generator=g), -1)
idx = a.topk(10, -1).indices
check("jsd identical = 0", torch.allclose(R.jsd_topk(a, a, idx), torch.zeros(5), atol=1e-6))
j = R.jsd_topk(a, b, idx)
check("jsd in (0, ln2]", bool((j > 0).all() and (j <= np.log(2) + 1e-6).all()))
check("jsd alpha=0.5 symmetric", torch.allclose(j, R.jsd_topk(b, a, idx), atol=1e-6))
# full support (K = V, tail ~ 0) matches a direct JSD
pa, pb = a.exp(), b.exp(); m = 0.5 * (pa + pb)
direct = 0.5 * (pa * (pa / m).log()).sum(-1) + 0.5 * (pb * (pb / m).log()).sum(-1)
check("jsd top-V == direct JSD", torch.allclose(R.jsd_topk(a, b, a.topk(50, -1).indices), direct, atol=1e-4))
check("superset_content", R.superset_content("<image>\nQ?") == "<image><image>\nQ?")
try:
    R.superset_content("<image><image>\nQ"); check("superset_content rejects 2 images", False)
except ValueError:
    check("superset_content rejects 2 images", True)
f = R.gap_fractions([1, 1, 2], [0, 1, 3])
check("gap_fractions", abs(f["gap_pos"] - 0.25) < 1e-9 and abs(f["gap_neg"] - 0.25) < 1e-9)
check("REL regex", bool(R.REL.search("What is to the left of the cup?")) and not R.REL.search("What color is the cup?"))

# analyze on synthetic tokens: gap+ planted on pre-commit tokens; trained ckpt lowers exactly the suppressed tokens
rng = np.random.RandomState(0); N = 4000; d = tempfile.mkdtemp()
t = np.tile(np.arange(40), N // 40); commit = np.full(N, 20)
sc = rng.rand(N) * 0.1; sf = sc.copy(); pre = t < 20; sf[pre] = sc[pre] * 0.2
lC = np.log(rng.rand(N) * 0.5 + 0.01); lF = lC.copy(); supp = pre & (rng.rand(N) < 0.3); lF[supp] = np.log(0.6); lC[supp] = np.log(0.01)
lS = np.log(np.full(N, 0.3)); lS2 = lS.copy(); lS2[supp] -= 1.0
np.savez_compressed(os.path.join(d, "tokens.npz"), jsd_SC=sc, jsd_SF=sf, jsd_CF=sc, lS=lS, lC=lC, lF=lF, lS2=lS2, ent_S=sc,
                    row=np.repeat(np.arange(N // 40), 40), s=np.zeros(N, int), t=t, tok=np.full(N, 785), commit=commit, rel=np.zeros(N, bool))
json.dump(dict(skipped={}, n_rows=N // 40, trained="x", K=100), open(os.path.join(d, "meta.json"), "w"))
R.part_analyze(types.SimpleNamespace(out=d)); s = json.load(open(os.path.join(d, "summary.json")))
check("analyze: gap+ > gap- (planted)", s["all"]["gap_pos"] > 0.3 and s["all"]["gap_neg"] < 1e-9)
check("analyze: post-commit gap+ = 0", s["buckets"]["承诺后"]["gap_pos"] < 1e-9)
check("analyze: suppressed only pre-commit", s["suppressed"]["frac_post"] == 0 and s["suppressed"]["frac_pre"] > 0.2)
check("analyze: train effect on suppressed = -1", abs(s["train_effect"]["sup"] + 1) < 1e-6)
check("analyze: suppressed tokens carry all the push-down", abs(s["train_effect"]["sup_share_of_pushdown"] - 1) < 1e-9)
check("analyze: commit-unknown bucket absent when all known", "承诺位置未知" not in s["buckets"])
print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
