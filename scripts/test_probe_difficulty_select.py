#!/usr/bin/env python
"""CPU tests for scripts/probe_difficulty_select.py helpers. 用法: python scripts/test_probe_difficulty_select.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import probe_difficulty_select as S

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); cond or fails.append(name)

L = list("AABBAAAA")
check("agree 0,1", S.agree(L, (0, 1), 2) == 1 and S.agree(L, (1, 2), 2) == 0)
check("agree rest", S.agree(L, range(2, 8), 3) == 0 and S.agree(list("AAAAAAAA"), range(2, 8), 3) == 1)
check("agree too few parsed", S.agree([None, "A", None, None, None, None, "A", None], range(2, 8), 3) == -1)
check("maj_share counts unparsed against", S.maj_share(["A"] * 6 + [None, None]) == 0.75 and S.maj_share([None] * 8) == 0.0)
M = np.array([[True, True, False, False, False, False], [True, True, True, True, False, False]])
lp = np.log(np.array([[0.45, 0.45, 0.1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25, 0, 0]]) + 1e-30)
mx, ne = S.t0_scores(lp, M)
check("t0 max renormalised over valid options", np.allclose(mx, [0.5, 0.25]))
check("t0 -entropy: ln2 / ln4, finite", np.allclose(ne, [-np.log(2), -np.log(4)]) and np.isfinite(ne).all())
t = S.tertile(np.arange(9.0))
check("tertile", list(t) == [0, 0, 0, 1, 1, 1, 2, 2, 2])
sb = np.array([0, 0, 1, -1]); ts = np.array([1.0, 0.5, 1.0, 1.0])
rn = np.array([0.5, 0.25, 1, 1]); rs = np.array([1.0, 0.0, 1, 0]); rt = np.array([1.0, 0.0, 1, 1]); rb = np.array([1.0, 0.5, 1, 1])
rs4 = np.array([1.0, 0.0, 1, 0])
c = S.cells(sb, ["u", "s"], ts, rn, rs, rt, rb, rs4)
check("cells: invalid student bin (-1) excluded", sum(v["n"] for v in c.values()) == 3)
check("cells: effect / drops", c["u|教师一致(4/4)"]["effect"] == 0.5 and c["u|教师分歧(≤2/4)"]["drops"] == 1 and c["u|教师分歧(≤2/4)"]["drops_null"] == 0)
check("TB bins 4-sample shares", [int(v.sum()) for v in S.TB(np.array([1, .75, .5, .25])).values()] == [1, 1, 2])

# crossfit_all scores every row (including y = -1) and learns a planted direction
rng = np.random.RandomState(0); N = 300; X = rng.randn(N, 8).astype(np.float32); y = (X[:, 0] > 0).astype(int); y[::7] = -1
folds = np.arange(N) % 5; p = S.crossfit_all(X, y, folds)
check("crossfit_all: finite for all rows (incl. y=-1)", np.isfinite(p).all())
ok = y >= 0
check("crossfit_all: learns planted signal", S.H.auc(p[ok], y[ok]) > 0.9 and S.H.auc(p[~ok], X[~ok, 0] > 0) > 0.9)
check("tertile on full probe", set(S.tertile(p)) == {0, 1, 2})

# end-to-end main() on a tiny synthetic layout
import json, tempfile
d = tempfile.mkdtemp(); bd, td, kd, od = [os.path.join(d, x) for x in ("b", "t", "k", "o")]
for x in (bd, td, kd): os.makedirs(x)
NR = 60; rows = list(range(NR)); Hh = rng.randn(NR, 3, 4, 8).astype(np.float32)
mk = lambda r, ls: dict(row=r, group=f"g{r}", gt="A", opts=list("ABCD"), letters=ls)
B, T, K = [], [], []
for r in rows:
    det = r % 2 == 0; Hh[r, 1, 2, 0] += 3 if det else -3
    B.append(mk(r, list("AAAAAAAA") if det else (list("ABCDABCD") if r % 4 == 1 else ["A", "B", None, None, None, "A", "B", None])))
    T.append(mk(r, list("AAAABBBB"))); K.append(mk(r, list("AAAAAAAB")))
for x, g in ((bd, B), (td, T), (kd, K)): json.dump(g, open(os.path.join(x, "gen.json"), "w"))
np.save(os.path.join(bd, "hidden.npy"), Hh); np.savez(os.path.join(bd, "readout.npz"), rows=np.array(rows), letter_logp=np.log(np.full((NR, 6), 0.2)))
sys.argv = ["x", "--base", bd, "--trained", kd, "--teacher", td, "--layer", "2", "--out", od]; S.main()
r = json.load(open(os.path.join(od, "select_summary.json")))
check("main: probe AUC high on planted layer", r["auc"]["probe"] > 0.9)
check("main: all rows binned", sum(v.get("n", 0) for v in r["cells"]["probe三分位"].values()) == NR)
check("main: teacher bin uses samples 0..3 (all 4/4)", r["teacher_consistency"]["教师一致(4/4)"]["n"] == NR and r["teacher_consistency"]["教师一致(4/4)"]["teacher_right"] == 0.0)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
