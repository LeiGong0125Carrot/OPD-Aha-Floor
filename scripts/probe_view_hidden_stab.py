#!/usr/bin/env python
"""Does the student-vs-teacher view hidden gap track teacher stability / the TSW score / correctness? (user 10-25, zero-training
diagnostic before any OPRD-style hidden-state distillation). Pure analysis of saved arrays, no model forward.

Inputs
  --viewhid  probe_view_hidden.py output dir (tokens.npz: per student-trajectory token x layer nd/cos for S-C (right crop) and
             S-M (mismatched crop), JSD S||C; row = parquet row, s = sample index into --gen texts/letters, t = response pos)
  --scores   export_tsw_scores.py scores.npz (row, p = teacher L28 OOF det4 score, det4, share8; no GT used to build them)
  --gen      student-view gen.json the trajectories were sampled from (letters[s] = that trajectory's answer)
  --tgen / --sgen  1024-token 8-sample teacher-view / student-view gen.json (question-level accuracies)
GT enters ONLY diagnostic columns (trajectory correct, teacher/student accuracy and their gap), as allowed for diagnostics.

Per question (mean over its tokens, t>=1 and t=0 separately) per layer:
  gap = nd_SC, content = nd_SM - nd_SC (part of the gap the RIGHT crop removes vs a same-shaped wrong crop), jsd = JSD S||C.
Reported: Spearman of gap / content / jsd with p, share8, teacher acc, student acc, teacher-minus-student gap; AUC of gap for
low25 membership (TSW set U) and for teacher det4; per-trajectory: gap/content of right vs wrong trajectories (Mann-Whitney AUC);
question-type split (count / color / text / other, keyword heuristic on the question text).
用法: python scripts/probe_view_hidden_stab.py --viewhid <dir> --scores <scores.npz> --gen <gen.json> --tgen <gen.json>
      --sgen <gen.json> --parquet <train_tsw.parquet> --out <dir>
"""
import argparse, collections, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_teacher_stability import spearman


def auc(score, y):
    """P(score[y=1] > score[y=0]) with ties 1/2; nan if a class is empty."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    a, b = score[y == 1], score[y == 0]
    if len(a) == 0 or len(b) == 0: return float("nan")
    allv = np.concatenate([a, b]); o = np.argsort(allv, kind="mergesort"); r = np.empty(len(allv)); s = allv[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def qtype(q):
    """Keyword type of the question stem only (options block and answer instruction removed)."""
    q = re.split(r"\n\s*\(?[A-F][.)]", q)[0].split("Answer with")[0].lower()
    if re.search(r"how many|number of|count", q): return "count"
    if re.search(r"colou?r", q): return "color"
    if re.search(r"text|written|word|\bsay|read|sign|logo|brand|name|number|letters? on", q): return "text"
    return "other"


BANDS = (1, 4, 8, 16, 32, 64, 128, 10 ** 9)


def band_z(x, t):
    """z-score x within response-position bands (t=0 kept as its own band) so that per-question means are not driven by
    how long the trajectories are (nd and JSD depend strongly on t)."""
    x = np.asarray(x, float).copy(); b = np.digitize(t, BANDS)
    for k in np.unique(b):
        m = b == k; mu, sd = x[m].mean(0), x[m].std(0)
        x[m] = (x[m] - mu) / np.where(sd > 0, sd, 1.0)
    return x


def per_question(z, mask):
    """Mean over the masked tokens of each question of the position-band z-scored values:
    dict row -> ([L] gap, [L] content, jsd)."""
    out = {}
    rows = z["row"][mask]
    gap, sm, jsd = z["zSC"][mask], z["zcon"][mask], z["zjsd"][mask]
    for r in np.unique(rows):
        k = rows == r
        out[int(r)] = (gap[k].mean(0), sm[k].mean(0), float(jsd[k].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    for a in ("viewhid", "scores", "gen", "tgen", "sgen", "parquet", "out"): ap.add_argument("--" + a, required=True)
    args = ap.parse_args(); say = lambda s="": print(s, flush=True)
    z = dict(np.load(os.path.join(args.viewhid, "tokens.npz")))
    meta = json.load(open(os.path.join(args.viewhid, "meta.json"))); layers = meta["layers"]
    assert os.path.realpath(args.gen) == os.path.realpath(meta["gen"]), f"--gen {args.gen} != trajectories' gen {meta['gen']}"
    z["zSC"] = band_z(z["nd_SC"], z["t"]); z["zcon"] = band_z(z["nd_SM"] - z["nd_SC"], z["t"]); z["zjsd"] = band_z(z["jsd_SC"], z["t"])
    sc = np.load(args.scores)
    pos = {int(r): i for i, r in enumerate(sc["row"])}
    G = {g["row"]: g for g in json.load(open(args.gen))}
    TG = {g["row"]: g for g in json.load(open(args.tgen))}; SG = {g["row"]: g for g in json.load(open(args.sgen))}
    import pyarrow.parquet as pq
    tb = pq.read_table(args.parquet, columns=["extra_info", "images"])
    ei = tb.column("extra_info").to_pylist(); imgs = tb.column("images").to_pylist()
    for r in np.unique(z["row"]):   # identity: same question across files
        r = int(r); assert G[r]["gt"] == TG[r]["gt"] == SG[r]["gt"] == ei[r]["answer"], f"row {r} gt mismatch"
        assert G[r]["group"] == TG[r]["group"] == imgs[r][0]["path"], f"row {r} image mismatch"
        assert abs(ei[r]["tsw_p"] - float(sc["p"][pos[r]])) < 1e-6, f"row {r} score mismatch"
    acc = lambda g: np.mean([x == g["gt"] for x in g["letters"][4:8]])   # samples 4..7: disjoint from det4 (0..3)
    res = {"layers": layers}
    for name, mask in (("t>=1", z["t"] >= 1), ("t=0", z["t"] == 0)):
        Q = per_question(z, mask); rows = sorted(Q)
        cols = dict(p=np.array([sc["p"][pos[r]] for r in rows]), share8=np.array([sc["share8"][pos[r]] for r in rows]),
                    t_acc=np.array([acc(TG[r]) for r in rows]), s_acc=np.array([acc(SG[r]) for r in rows]))
        cols["t_minus_s"] = cols["t_acc"] - cols["s_acc"]
        low = np.array([ei[r]["tsw_low25"] for r in rows]); det4 = np.array([sc["det4"][pos[r]] for r in rows])
        gap = np.stack([Q[r][0] for r in rows]); con = np.stack([Q[r][1] for r in rows]); jsd = np.array([Q[r][2] for r in rows])
        say(f"\n== 题级 ({name}, {len(rows)} 题; U 内 {int(low.sum())} 题): Spearman ρ  [gap=nd_SC | content=nd_SM−nd_SC; 按位置段 z 化后取题内均值]")
        say(f"   {'':6s}" + "".join(f"{c:>18s}" for c in cols) + f"{'AUC(gap→U)':>14s}{'AUC(gap→det4)':>15s}")
        R = {}
        for li, l in enumerate(layers):
            R[l] = {c: (spearman(gap[:, li], v), spearman(con[:, li], v)) for c, v in cols.items()}
            R[l]["auc_gap_U"] = auc(gap[:, li], low); m = det4 >= 0; R[l]["auc_gap_det4"] = auc(gap[m, li], det4[m])
            say(f"   L{l:<4d}" + "".join(f"{R[l][c][0]:+8.3f}|{R[l][c][1]:+8.3f}" for c in cols)
                + f"{R[l]['auc_gap_U']:14.3f}{R[l]['auc_gap_det4']:15.3f}")
        Rj = {c: spearman(jsd, v) for c, v in cols.items()}
        say("   JSD S‖C " + " ".join(f"{c} {v:+.3f}" for c, v in Rj.items()) + f" | AUC(jsd→U) {auc(jsd, low):.3f}")
        # question type
        qt = np.array([qtype(ei[r]["question"]) for r in rows]); li = layers.index(28) if 28 in layers else len(layers) - 2
        say(f"   题型 (L{layers[li]}): " + " | ".join(
            f"{t} n={int((qt == t).sum())} gap {gap[qt == t, li].mean():.3f} content {con[qt == t, li].mean():+.3f} U占 {low[qt == t].mean():.2f}"
            for t in sorted(set(qt))))
        res[name] = dict(n=len(rows), layer={str(l): R[l] for l in layers}, jsd=Rj, auc_jsd_U=auc(jsd, low))
    # per trajectory (t>=1), WITHIN question: among questions whose scored trajectories disagree in correctness, how often is
    # the wrong trajectory's (position-band z-scored) value higher than the right one's. Pooling across questions would let
    # question difficulty and trajectory length leak in.
    m = z["t"] >= 1; rows_t, s_t = z["row"][m], z["s"][m]; gz, cz = z["zSC"][m], z["zcon"][m]
    by_q = collections.defaultdict(list)
    for r, s in sorted(set(zip(rows_t.tolist(), s_t.tolist()))):
        let = G[r]["letters"][s]
        if let is None: continue
        k = (rows_t == r) & (s_t == s)
        by_q[r].append((int(let == G[r]["gt"]), gz[k].mean(0), cz[k].mean(0)))
    win_g, win_c, npair = np.zeros(len(layers)), np.zeros(len(layers)), 0
    for r, tr in by_q.items():
        for a in tr:
            for b in tr:
                if a[0] == 0 and b[0] == 1:   # a wrong, b right
                    win_g += (a[1] > b[1]) + 0.5 * (a[1] == b[1]); win_c += (a[2] > b[2]) + 0.5 * (a[2] == b[2]); npair += 1
    nq = sum(1 for tr in by_q.values() if len({x[0] for x in tr}) == 2)
    say(f"\n== 轨迹级, 题内对错不一致的配对 ({nq} 题, {npair} 对; t≥1 位置段 z 化): P(答错轨迹的值更高) gap | content (0.5 = 无信号)")
    if npair:
        say("   " + " | ".join(f"L{l} {win_g[i] / npair:.3f}/{win_c[i] / npair:.3f}" for i, l in enumerate(layers)))
    res["traj_within"] = dict(n_questions=nq, n_pairs=npair,
                              layer={str(l): (win_g[i] / max(npair, 1), win_c[i] / max(npair, 1)) for i, l in enumerate(layers)})
    res["args"] = vars(args)
    os.makedirs(args.out, exist_ok=True); json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    say(f"[saved] {args.out}/summary.json")


if __name__ == "__main__":
    main()
