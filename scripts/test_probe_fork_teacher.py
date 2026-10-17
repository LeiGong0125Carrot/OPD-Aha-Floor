#!/usr/bin/env python
"""CPU tests for scripts/probe_fork_teacher.py pure helpers + analyze on a synthetic dump. 用法: python scripts/test_probe_fork_teacher.py"""
import json, os, sys, tempfile, types
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import probe_fork_teacher as F

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); cond or fails.append(name)

check("first_divergence basic", F.first_divergence([1, 2, 3], [1, 5, 3]) == 1)
check("first_divergence t=0", F.first_divergence([4, 2], [1, 2]) == 0)
check("first_divergence prefix -> None", F.first_divergence([1, 2], [1, 2, 3]) is None)
check("first_divergence identical -> None", F.first_divergence([1, 2], [1, 2]) is None)
tr = [("S", 0, [1, 2, 3]), ("S", 1, [1, 2, 4]), ("T", 0, [1, 9]), ("T", 1, [1, 2, 3])]
sp = F.fork_pairs(tr)
check("fork_pairs count C(4,2)=6", len(sp) == 6)
kinds = sorted(s["kind"] for s in sp)
check("fork_pairs kinds 1 SS / 1 TT / 4 ST", kinds == ["SS", "ST", "ST", "ST", "ST", "TT"])
st = [s for s in sp if s["kind"] == "ST"]
check("ST pairs: student first", all(s["a"][0] == "S" and s["b"][0] == "T" for s in st))
s01 = [s for s in st if s["a"] == ("S", 0) and s["b"] == ("T", 0)][0]
check("ST tokens a=student b=teacher", s01["t_div"] == 1 and s01["tok_a"] == 2 and s01["tok_b"] == 9)
same = [s for s in st if s["a"] == ("S", 0) and s["b"] == ("T", 1)][0]
check("identical pair -> no fork", same["t_div"] is None and same["tok_a"] is None)
# entropy-matched baseline: if fork entropies all fall in the top bin, baseline = mean of vals in that bin
ent = np.linspace(0, 1, 1000); vals = (ent > 0.9).astype(float)
check("entropy-matched baseline picks the fork bins", abs(F.entropy_matched_baseline(vals, ent, np.full(20, 0.95)) - 1.0) < 1e-9)
check("entropy-matched baseline low bins", abs(F.entropy_matched_baseline(vals, ent, np.full(20, 0.05))) < 1e-9)

# analyze on a synthetic dump (2 rows, 2 student + 2 teacher trajectories each)
d = tempfile.mkdtemp(); rng = np.random.default_rng(0)
gen, forks, answers, pos = [], [], [], {k: [] for k in ("tv_ST", "tv_TN", "lS", "lT", "lN", "ent_S", "top1_agree", "row", "kind", "traj", "t", "T")}
for row in (3, 7):
    S = [[1, 2, 3, 4, 5], [1, 2, 6, 7]]; T = [[1, 8, 9], [1, 2, 3, 4, 5]]
    gen.append(dict(row=row, S=S, T=T, S_text=["x <answer>A</answer>", "y <answer>B</answer>"], T_text=["zoomed <answer>B</answer>", "z <answer>B</answer>"]))
    for kind, idx, y in [("S", 0, S[0]), ("S", 1, S[1]), ("T", 0, T[0]), ("T", 1, T[1])]:
        n = len(y)
        for k in ("tv_ST", "tv_TN", "ent_S", "top1_agree"): pos[k].append(rng.random(n))
        for k in ("lS", "lT", "lN"): pos[k].append(-rng.random(n))
        pos["row"].append(np.full(n, row)); pos["kind"].append(np.full(n, kind)); pos["traj"].append(np.full(n, idx)); pos["t"].append(np.arange(n)); pos["T"].append(np.full(n, n))
        answers.append(dict(row=row, kind=kind, idx=idx, t=n - 1, T=n, letter={"S": "AB", "T": "BB"}[kind][idx], lS=-0.1, lT=-0.05, lN=-0.2))
    for s in F.fork_pairs([("S", i, y) for i, y in enumerate(S)] + [("T", j, y) for j, y in enumerate(T)]):
        if s["t_div"] is not None:
            s.update(lS_a=-1.0, lS_b=-1.5, lT_a=-0.2, lT_b=-2.0, lN_a=-0.5, lN_b=-1.0, tv_ST=0.4, ent_S=0.5, row=row, a_txt="a", b_txt="b")
        s["a"] = list(s["a"]); s["b"] = list(s["b"]); s.setdefault("row", row); forks.append(s)
np.savez_compressed(os.path.join(d, "positions.npz"), **{k: np.concatenate(v) for k, v in pos.items()})
json.dump(forks, open(os.path.join(d, "forks.json"), "w")); json.dump(answers, open(os.path.join(d, "answers.json"), "w")); json.dump(gen, open(os.path.join(d, "gen.json"), "w"))
F.part_analyze(types.SimpleNamespace(out=d))
res = json.load(open(os.path.join(d, "summary.json")))
check("analyze: SS fork stats present", res["forks"]["SS"]["n_pairs"] == 2 and res["forks"]["SS"]["no_fork"] == 0)
check("analyze: d = ΔT−ΔS = 1.8−0.5 = 1.3", abs(res["forks"]["SS"]["abs_d_mean"] - 1.3) < 1e-9)
check("analyze: ST teacher prefers student token here (ΔT>0)", res["forks"]["ST"]["teacher_prefers_teacher_tok"] == 0.0)
check("analyze: ST no-fork pairs counted (S0 vs T1 identical)", res["forks"]["ST"]["no_fork"] == 2)
check("analyze: E agree/disagree split (teacher majority B: S1 agrees, S0 disagrees)", res["E"]["n_agree"] == 2 and res["E"]["n_dis"] == 2)
check("analyze: D computed", "D_d" in res and res["D_d"]["n"] == 2)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
