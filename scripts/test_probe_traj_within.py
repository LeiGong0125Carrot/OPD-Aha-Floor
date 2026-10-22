#!/usr/bin/env python
"""CPU tests for scripts/probe_traj_within.py helpers + tanalyze / cross on synthetic data. 用法: python scripts/test_probe_traj_within.py"""
import json, os, sys, tempfile, types
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import probe_traj_within as W

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); cond or fails.append(name)

q = "<image>\nWhat color?\n\nA. black\nB. white\nC. red\nD. purple\n\nAnswer with the option's letter from the given choices."
ot = W.option_texts(q)
check("option_texts", ot == {"A": "black", "B": "white", "C": "red", "D": "purple"})
t1 = "The floor is lit with a purple hue.\n\n**Correct Answer: D. purple**"
check("commit at option text before letter", W.commit_char(t1, "D", ot["D"]) == t1.index("purple"))
check("commit answer-first", W.commit_char("D. purple", "D", ot["D"]) == 0)
t2 = "A red light covers the floor. Answer: A"
check("answer phrase commits ('Answer: A')", W.commit_char(t2, "A", ot["A"]) == t2.index("Answer"))
t3 = "A red light. So: A"
check("article 'A red' is not a commit", W.commit_char(t3, "A", "zzz") == t3.rindex("A"))
check("lower-case 'answer is a bit' is not a commit", W.commit_char("The answer is a bit unclear; B", "A", "zzz") is None)
check("option text whole word only", W.commit_char("blackish tones; so B", "A", "black") is None)
check("no commit -> None", W.commit_char("hard to say", "B", "white") is None)
check("char_to_token", W.char_to_token([(0, 3), (3, 5), (5, 9)], 4) == 1 and W.char_to_token([(0, 3)], 3) == 1)
p0, p5 = W.positions(0), W.positions(5)
check("positions answer-first: only k0", p0 == {"k0": 0})
check("positions ct=5", p5 == {"k0": 0, "k1": 1, "k2": 2, "k4": 4, "half": 2, "pre1": 4, "pre": 5})
check("within_auc perfect", W.within_auc([1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]) == 1.0)
check("within_auc ignores between-group offset", W.within_auc([10, 9, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]) == 1.0)
check("within_auc shared score -> 0.5", W.within_auc([3, 3, 1, 1], [1, 0, 1, 0], [0, 0, 1, 1]) == 0.5)
check("within_auc single-class groups skipped", np.isnan(W.within_auc([1, 2], [1, 1], [0, 0])))
g = lambda gt, ls: dict(gt=gt, letters=ls)
check("mixed", W.mixed(g("A", ["A", "B"])) and not W.mixed(g("A", ["A", None])) and not W.mixed(g("A", ["B", "C"])))
gen = {i: g("A", ["A", "B"]) for i in range(10)}; pair = {i: g("A", ["A", "B"] if i % 2 else ["A", "A"]) for i in range(10)}
check("select_rows: mixed in both", W.select_rows(gen, pair, 0) == [1, 3, 5, 7, 9])
check("select_rows: subsample deterministic", W.select_rows(gen, pair, 2) == W.select_rows(gen, pair, 2) and len(W.select_rows(gen, pair, 2)) == 2)

# tanalyze: correctness planted in layer index 2 at positions k4/pre only (k0 shared within prompt -> within AUC ~0.5)
rng = np.random.RandomState(0); NP, S, D = 120, 8, 12; d = tempfile.mkdtemp(); T = []; X = np.zeros((NP * S, len(W.POS_NAMES), len(W.LAYERS), D), np.float16)
for p in range(NP):
    shared = rng.randn(D)
    for s in range(S):
        c = s % 2 == 0; j = len(T); let = "A" if c else "B"
        X[j] = rng.randn(len(W.POS_NAMES), len(W.LAYERS), D) * 0.3
        X[j, 0] = shared                                                     # k0 = prompt state, identical in the group
        for pn in ("k4", "pre"): X[j, W.POS_NAMES.index(pn), 2, 0] += 3.0 if c else -3.0
        T.append(dict(row=p, s=s, letter=let, gt="A", opts=list("ABCD"), correct=c, commit_tok=10, valid=[pn in W.positions(10) for pn in W.POS_NAMES]))
np.save(os.path.join(d, "traj_hidden.npy"), X)
json.dump(dict(model="x", layers=W.LAYERS, pos=W.POS_NAMES, skipped={}, traj=T), open(os.path.join(d, "traj_meta.json"), "w"))
W.part_tanalyze(types.SimpleNamespace(out=d))
r = json.load(open(os.path.join(d, "traj_summary.json")))
check("tanalyze: k0 within AUC ~0.5 (shared state)", abs(r["best"]["k0"]["win_corr"] - 0.5) < 0.1)
check("tanalyze: planted pre within AUC > 0.95 at layer index 2", r["best"]["pre"]["win_corr"] > 0.95 and r["best"]["pre"]["layer"] == W.LAYERS[2])
check("tanalyze: shuffled ~ chance", abs(r["best"]["pre"]["shuffled"] - 0.5) < 0.15)
check("tanalyze: k32/k64 invalid (commit_tok=10) skipped", "k32" not in r["best"] and "k64" not in r["best"])

# cross: bins from base samples 0-3, null from base 4-7, treatment = trained
mk = lambda gt, ls: dict(gt=gt, letters=ls, opts=list("ABCD"))
base = {0: mk("A", list("AABBAABB")), 1: mk("A", list("AABBAABB")), 2: mk("A", list("BBBBBBBB"))}
trn = {0: mk("A", list("CCCCCCCC")), 1: mk("A", list("AAAAAAAA")), 2: mk("A", list("ABBBBBBB"))}
tch = {0: mk("A", list("CCCA")), 1: mk("A", list("AAAA")), 2: mk("A", list("AAAB"))}
c = W.cross_table(base, trn, tch)
cw, cr = c["cells"]["教师多数错(<0.5)|base前4部分对"], c["cells"]["教师多数对(≥0.5)|base前4部分对"]
check("cross: teacher-wrong & base-partial: null 0.5, trained 0, effect -0.5", cw["n"] == 1 and cw["base_null"] == 0.5 and cw["trained"] == 0 and cw["effect"] == -0.5)
check("cross: teacher-right & base-partial: effect +0.5", cr["effect"] == 0.5)
w = c["teacher_wrong_letter"]
check("cross: wrong-sample share on teacher letter 0 -> 1, uniform 1/3", w["n"] == 1 and w["base_share"] == 0 and w["trained_share"] == 1 and abs(w["uniform"] - 1 / 3) < 1e-9)
check("cross: big drop counted (eligible, teacher wrong), null 0", c["big_drops"]["n"] == 1 and c["big_drops"]["teacher_wrong_frac"] == 1 and c["big_drops"]["null_n"] == 0)
# missed-commit cases from review 10-21
for txt, L in (("Option D is correct", "D"), ("D is the answer", "D"), ("The answer is A because", "A"), ("Option A is the black one", "A")):
    check(f"commit_char '{txt}' at the letter or the answer phrase", W.commit_char(txt, L, "zzz") is not None and W.commit_char(txt, L, "zzz") <= txt.rindex(L))
check("positions ct=5 has pre1=4", W.positions(5).get("pre1") == 4)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
