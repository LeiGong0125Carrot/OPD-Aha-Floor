#!/usr/bin/env python
"""CPU tests for scripts/probe_teacher_stability.py. 用法: python scripts/test_probe_teacher_stability.py"""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import probe_teacher_stability as S
import probe_hidden_answer as H

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); cond or fails.append(name)

check("spearman monotone = 1", abs(S.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9)
check("spearman ties / constant -> nan", np.isnan(S.spearman([1, 1, 1], [1, 2, 3])))
check("spearman reversed = -1", abs(S.spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-9)
Z = S.zscore(np.array([[1.0, 5], [3, 5], [5, 5]]))
check("zscore: mean 0, constant column finite", np.allclose(Z.mean(0), 0) and np.isfinite(Z).all())

# transfer: shared direction transfers across a view offset only after per-view z-scoring
rng = np.random.RandomState(0); N, D = 400, 6
lat = rng.randn(N); y = (lat > 0).astype(int); y[::9] = -1
XA = rng.randn(N, D); XA[:, 0] += 2 * lat
XB = rng.randn(N, D) * 3 + 50; XB[:, 0] += 6 * lat                  # same direction, other offset / scale
folds = np.arange(N) % 5; ok = y >= 0
p = S.transfer(S.zscore(XA), y, S.zscore(XB), folds)
check("transfer: shared direction (z-scored) AUC > 0.9", np.isfinite(p).all() and H.auc(p[ok], y[ok]) > 0.9)
XC = rng.randn(N, D); XC[:, 3] += 2 * lat                             # view-specific direction
p2 = S.transfer(S.zscore(XA), y, S.zscore(XC), folds)
check("transfer: view-specific direction ~ chance", abs(H.auc(p2[ok], y[ok]) - 0.5) < 0.1)

# end-to-end main() on a tiny synthetic pair of views
d = tempfile.mkdtemp(); NR, L = 120, 5
for v in ("s", "t"): os.makedirs(os.path.join(d, v))
HS = rng.randn(NR, 3, L, 8).astype(np.float32); HT = rng.randn(NR, 3, L, 8).astype(np.float32)
GS, GT = [], []
for r in range(NR):
    stable_t = r % 2 == 0; stable_s = r % 3 == 0
    HT[r, 1, 2, 0] += 3 if stable_t else -3; HS[r, 1, 2, 1] += 3 if stable_s else -3
    lt = list("AAAAAAAA") if stable_t else list("ABCDABCD"); ls = list("BBBBAAAA") if stable_s else list("ABCDBCDA")
    if r == 5: ls = ["A", None, None, "A", "A", "A", "A", "A"]           # det4 = -1 (2 of samples 0..3 parsed): excluded
    GT.append(dict(row=r, group=f"g{r}", gt="A", opts=list("ABCD"), letters=lt)); GS.append(dict(row=r, group=f"g{r}", gt="A", opts=list("ABCD"), letters=ls))
# teacher view stored in a permuted row order (readout rows / hidden / gen.json) -> catches cross-view index misalignment
perm = rng.permutation(NR)
for v, G, Hh, order in (("s", GS, HS, np.arange(NR)), ("t", GT, HT, perm)):
    json.dump([G[i] for i in order[::-1]], open(os.path.join(d, v, "gen.json"), "w")); np.save(os.path.join(d, v, "hidden.npy"), Hh[order])
    np.savez(os.path.join(d, v, "readout.npz"), rows=order, letter_logp=np.log(np.full((NR, 6), 0.2)))
sys.argv = ["x", "--student", os.path.join(d, "s"), "--teacher", os.path.join(d, "t"), "--out", os.path.join(d, "o"), "--layers", "1,2"]
S.main(); r = json.load(open(os.path.join(d, "o", "summary.json")))
check("main: T->T high on planted layer", r["layer"]["2"]["TT"] > 0.9 and r["layer"]["2"]["SS"] > 0.9)
check("main: different planted dims -> transfer ~ chance", abs(r["layer"]["2"]["S2T"] - 0.5) < 0.2)
check("main: positives counted on samples 0..3, det4=-1 excluded", abs(r["pos"]["T"] - 0.5) < 1e-9 and r["n_valid"]["S"] == NR - 1 and abs(r["pos"]["S"] - 40 / 119) < 1e-9)
check("main: teacher correctness diag (late samples) = stable rows", r["ref"]["T"]["share4_acc"] > 0.99)
check("main: correctness uses samples 4..7 only", abs(r["ref"]["S"]["acc_pos"] - 41 / 120) < 1e-9)
check("main: quartiles cover all rows", sum(v["n"] for v in r["teacher_probe_quartiles"].values()) == NR)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
