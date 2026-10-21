#!/usr/bin/env python
"""CPU tests for scripts/probe_hidden_answer.py helpers + analyze on a synthetic dump. 用法: python scripts/test_probe_hidden_answer.py"""
import json, os, sys, tempfile, types
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import probe_hidden_answer as H

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); cond or fails.append(name)

q = "<image>\nWhat color?\n\nOnly focus on ...\n\nA. black\nB. white\nC. blue\nD. purple\n\nAnswer with the option's letter from the given choices."
O = H.parse_options(q)
check("parse_options A-D", O == list("ABCD"))
check("parse_letter bare", H.parse_letter("D", O) == "D")
check("parse_letter 'D.'", H.parse_letter("D. purple", O) == "D")
check("parse_letter '(C)'", H.parse_letter("(C)", O) == "C")
check("parse_letter <answer>", H.parse_letter("The light is blue. <answer>C</answer>", O) == "C")
check("parse_letter 'answer is B'", H.parse_letter("Looking closely, the answer is B.", O) == "B")
check("parse_letter article 'A red' is not A", H.parse_letter("A red light covers the floor, so D", O) == "D")
check("parse_letter article only -> None", H.parse_letter("A red light covers the floor.", O) is None)
check("parse_letter out-of-options -> None", H.parse_letter("E", O) is None)
check("parse_letter ambiguous -> None", H.parse_letter("Between B and C it is hard to say", O) is None)
check("majority basic", H.majority(["A", "A", "B", None]) == "A")
check("majority tie -> None", H.majority(["A", "B"]) is None)
check("majority empty -> None", H.majority([None, None]) is None)
g = ["a", "a", "b", "c", "c", "c", "d", "e", "f", "g"]
f = H.group_folds(g, k=3)
check("group_folds groups intact", all(len(set(f[[i for i, x in enumerate(g) if x == k]])) == 1 for k in set(g)))
check("group_folds deterministic", (f == H.group_folds(g, k=3)).all())
check("auc perfect", H.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0)
check("auc ties 0.5", H.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5)
check("auc one class nan", np.isnan(H.auc([1, 2], [1, 1])))

# linear probe recovers a planted signal; masks respected; shuffle -> chance
rng = np.random.RandomState(0); N, D = 600, 20
X = rng.randn(N, D).astype(np.float32); y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)   # 4 classes
M = np.zeros((N, 6), bool); M[:, :4] = True; folds = np.arange(N) % 5
P = H.cv_probe(X, y, M, folds, 6)
acc = (P.argmax(1) == y).mean()
check(f"cv_probe recovers planted signal (acc {acc:.2f} > 0.85)", acc > 0.85)
check("cv_probe masked classes never predicted", (P.argmax(1) < 4).all())
Ps = H.cv_probe(X, y, M, folds, 6, shuffle=True)
check(f"cv_probe shuffled ~ chance (acc {(Ps.argmax(1) == y).mean():.2f} < 0.4)", (Ps.argmax(1) == y).mean() < 0.4)
y2 = y.copy(); y2[:50] = -1
P2 = H.cv_probe(X, y2, M, folds, 6)
check("cv_probe excluded rows are nan", np.isnan(P2[:50]).all() and not np.isnan(P2[50:]).any())

# analyze end-to-end on a synthetic dump: GT linearly planted at layer 2 of the 'last' position only
d = tempfile.mkdtemp(); N, L1, D = 300, 4, 16
gt_idx = rng.randint(0, 4, N); hid = rng.randn(N, 3, L1, D).astype(np.float32)
hid[np.arange(N), 1, 2, gt_idx] += 4.0
gen = []
for i in range(N):
    good = rng.rand() < 0.6
    letters = [("ABCD"[gt_idx[i]] if good else "ABCD"[(gt_idx[i] + 1) % 4]) for _ in range(4)]
    gen.append(dict(row=i, group=f"img{i // 2}", gt="ABCD"[gt_idx[i]], opts=list("ABCD"), texts=letters, letters=letters, first_tok=[0] * 4))
lp = np.log(np.full((N, 6), 1e-3)); lp[np.arange(N), [("ABCD".index(g["letters"][0])) for g in gen]] = np.log(0.9)
json.dump(gen, open(os.path.join(d, "gen.json"), "w")); np.save(os.path.join(d, "hidden.npy"), hid)
np.savez(os.path.join(d, "readout.npz"), rows=np.arange(N), letter_logp=lp)
H.part_analyze(types.SimpleNamespace(out=d, layer_step=1))
res = json.load(open(os.path.join(d, "summary.json")))
check("analyze: best last/gt layer = planted layer 2", res["best"]["last/gt"]["layer"] == 2)
check("analyze: planted GT recovered (>0.9)", res["best"]["last/gt"]["value"] > 0.9)
check("analyze: vis position GT ~ chance (<0.45)", max(res["sweep"][f"vis/{l}"]["gt"] for l in range(L1)) < 0.45)
check("analyze: txt position GT ~ chance (<0.45)", max(res["sweep"][f"txt/{l}"]["gt"] for l in range(L1)) < 0.45)
check("analyze: know-more reports text/prior controls", "hit_txt" in res["know_more/t0 输出 argmax≠GT"] and "hit_prior" in res["know_more/t0 输出 argmax≠GT"])
check("analyze: shuffled control ~ chance", res["best"]["last/gt"]["shuffled"] < 0.45)
check("analyze: t0 readout baseline = sample accuracy (~0.6)", abs(res["baseline"]["gt_out"] - np.mean([g["letters"][0] == g["gt"] for g in gen])) < 1e-9)
check("analyze: know-more on wrong rows (>0.9)", res["know_more/t0 输出 argmax≠GT"]["hit"] > 0.9)
check("analyze: traj between-prompt var frac = 1 (all samples per prompt identical)", abs(res["traj"]["between_prompt_var_frac"] - 1) < 1e-9)
gd = [dict(gt="A", opts=list("ABCD"), letters=["A", None, None, None])]
yd, _, _ = H.build_targets(gd)
check("det excluded when fewer than half parsed", yd["det"][0] == -1)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
