#!/usr/bin/env python
"""Teacher-view stability probe (user 10-23). Is the frozen teacher's (crop view) answer stability readable from its
last-prompt-token hidden state, does a stability direction transfer between the student view (red-box full image + hint)
and the teacher view, and how does each probe relate to correctness?

Inputs (same 6241 off6k rows, 8 samples per row and view, base Qwen3.5-4B):
  --student  dir with gen.json, hidden.npy [N, 3 pos, 33 layers, D], readout.npz (student view, probe_hidden_answer)
  --teacher  dir with gen.json, hidden.npy, readout.npz (crop view, probe_hidden_answer --view teacher)
Position: last prompt token (index 1). Layers: --layers.

Label-free targets (no GT): det4_V = samples 0..3 of view V all agree (>=3 parsed). Correctness (GT) is DIAGNOSTIC only
(user exception, nothing here feeds training): acc_V = >=half of samples 4..7 of view V right, and the mean accuracy on
samples 4..7 -- disjoint from the samples that define the target, so stability->correctness is not mechanical.
Reported per layer:
  (a) within view: cross-fitted probe AUC for det4_S on S hidden, det4_T on T hidden; shuffled-label control (best layer);
      baselines: t0 readout max prob of each view, sample-based maj share of samples 0..3 is the target itself (n/a).
  (b) transfer: probe fit on view A hidden / det4_A, applied to view B hidden (held-out folds) vs det4_B. Each view's
      features are z-scored with that view's own statistics (label-free, all rows) so a constant view offset is removed;
      the raw (un-z-scored, A-standardised) transfer is reported too.
  (c) cross-label: S hidden -> det4_T (does the student-view state know whether the TEACHER will be stable?).
  (d) correctness diagnostics: AUC of each cross-fitted score vs acc_V (samples 4..7) and Spearman vs mean accuracy,
      next to the sample-based stability (maj share of samples 0..3) as the reference.
用法: python scripts/probe_teacher_stability.py --student <dir> --teacher <dir> --out <dir> [--layers 8,12,16,20,24,28,31,32]
"""
import argparse, collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_hidden_answer as H
from probe_difficulty_select import agree, maj_share, t0_scores, crossfit_all

LAST = 1


def spearman(a, b):
    """Spearman rho with average ranks (scipy-free); nan if constant."""
    def rk(x):
        x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); s = x[o]; i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
            r[o[i:j + 1]] = (i + j) / 2; i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    if ra.std() == 0 or rb.std() == 0: return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def zscore(X):
    """Per-feature z-score with the matrix's own statistics (label-free)."""
    return (X - X.mean(0)) / X.std(0).clip(1e-6)


def transfer(XA, yA, XB, folds, lams=(1e-3, 1e-2, 1e-1, 1.0)):
    """P(y=1) for every row of view B from a probe trained on view A's rows of the OTHER folds (valid yA only; lam by
    inner held-out CE on view A). Held-out questions only, so no row is scored by a probe that saw it in either view."""
    out = np.full(len(yA), np.nan); ok = yA >= 0; two = np.ones((len(yA), 2), bool)
    for f in np.unique(folds):
        tr, te = ok & (folds != f), folds == f
        inner = folds[tr]; fi = np.unique(inner)[0]; a, v = inner != fi, inner == fi
        best = min(lams, key=lambda lam: -np.mean(H.fit_predict(XA[tr][a], yA[tr][a], two[tr][a], XA[tr][v], two[tr][v], lam, 2)[np.arange(v.sum()), yA[tr][v]]))
        out[te] = np.exp(H.fit_predict(XA[tr], yA[tr], two[tr], XB[te], two[te], best, 2)[:, 1])
    return out


def load_view(d):
    gen = {g["row"]: g for g in json.load(open(os.path.join(d, "gen.json")))}
    z = np.load(os.path.join(d, "readout.npz"))
    return gen, [int(r) for r in z["rows"]], z["letter_logp"], np.load(os.path.join(d, "hidden.npy"), mmap_mode="r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="8,12,16,20,24,28,31,32")
    args = ap.parse_args(); say = lambda s="": print(s, flush=True)
    layers = [int(x) for x in args.layers.split(",")]
    gS, rS, lpS, HS = load_view(args.student); gT, rT, lpT, HT = load_view(args.teacher)
    assert HS.shape[1:] == HT.shape[1:] and max(layers) < HS.shape[2], (HS.shape, HT.shape, layers)
    iS, iT = {r: i for i, r in enumerate(rS)}, {r: i for i, r in enumerate(rT)}
    rows = [r for r in rS if r in iT and len(gS[r]["letters"]) >= 8 and len(gT[r]["letters"]) >= 8]
    kS, kT = [iS[r] for r in rows], [iT[r] for r in rows]
    for r in rows[:50]: assert gS[r]["gt"] == gT[r]["gt"] and gS[r]["opts"] == gT[r]["opts"], f"row {r}: views disagree on gt/opts"
    M = np.zeros((len(rows), 6), bool)
    for i, r in enumerate(rows):
        for o in gS[r]["opts"]: M[i, H.LETTERS.index(o)] = True
    folds = H.group_folds([gS[r]["group"] for r in rows])
    V = {"S": gS, "T": gT}
    det4 = {v: np.array([agree(V[v][r]["letters"], range(4), 3) for r in rows]) for v in V}
    share4 = {v: np.array([maj_share(V[v][r]["letters"][:4]) for r in rows]) for v in V}
    acc_late = {v: np.array([np.mean([x == V[v][r]["gt"] for x in V[v][r]["letters"][4:8]]) for r in rows]) for v in V}
    t0max = {"S": t0_scores(lpS[kS], M)[0], "T": t0_scores(lpT[kT], M)[0]}
    res = dict(n=len(rows), layers=layers, pos={v: float(det4[v][det4[v] >= 0].mean()) for v in V},
               n_valid={v: int((det4[v] >= 0).sum()) for v in V}, layer={}, ref={})
    say(f"== {len(rows)} 题; det4 (样本0..3全一致, ≥3解析) 正类: 学生 {res['pos']['S']:.3f} (n={res['n_valid']['S']}) | 教师 {res['pos']['T']:.3f} (n={res['n_valid']['T']})")
    # reference: label-free sample/readout scores vs targets and correctness (GT diagnostic: samples 4..7)
    for v in V:
        m = det4[v] >= 0; y = acc_late[v] >= 0.5
        res["ref"][v] = dict(t0max_det4=H.auc(t0max[v][m], det4[v][m]), share4_acc=H.auc(share4[v], y), share4_rho=spearman(share4[v], acc_late[v]),
                             det4_acc=H.auc(det4[v][m], y[m]), t0max_acc=H.auc(t0max[v], y), acc_pos=float(y.mean()))
        say(f"   参照 [{v}]: t0读出 max→det4 AUC {res['ref'][v]['t0max_det4']:.3f} | 采样稳定性(前4多数占比)→对(后4过半) AUC {res['ref'][v]['share4_acc']:.3f} "
            f"ρ {res['ref'][v]['share4_rho']:.3f} | t0 max→对 AUC {res['ref'][v]['t0max_acc']:.3f} | 后4过半对 {res['ref'][v]['acc_pos']:.3f}")
    say("\n== 逐层 (prompt 末 token). 稳定性 AUC: S→S, T→T, 迁移 S⇒T / T⇒S (各视图自 z-score; 括号 = 不 z-score), 学生hidden→教师det4 | 对错诊断 AUC/ρ: 教师探针→教师对, 学生探针→学生对")
    # one read per view (strided memmap reads are slow on weka): [rows, layers] at the last prompt token, ~0.5 GB per view
    LS = np.stack([np.asarray(HS[:, LAST, l], dtype=np.float32) for l in layers], 1)[kS]
    LT = np.stack([np.asarray(HT[:, LAST, l], dtype=np.float32) for l in layers], 1)[kT]
    best = {}
    for li, l in enumerate(layers):
        XS, XT = LS[:, li], LT[:, li]
        ZS, ZT = zscore(XS), zscore(XT)
        pSS = crossfit_all(ZS, det4["S"], folds); pTT = crossfit_all(ZT, det4["T"], folds)
        pST, pTS = transfer(ZS, det4["S"], ZT, folds), transfer(ZT, det4["T"], ZS, folds)
        rST, rTS = transfer(XS, det4["S"], XT, folds), transfer(XT, det4["T"], XS, folds)
        pSx = crossfit_all(ZS, det4["T"], folds)
        mS, mT = det4["S"] >= 0, det4["T"] >= 0
        row = dict(SS=H.auc(pSS[mS], det4["S"][mS]), TT=H.auc(pTT[mT], det4["T"][mT]),
                   S2T=H.auc(pST[mT], det4["T"][mT]), T2S=H.auc(pTS[mS], det4["S"][mS]),
                   S2T_raw=H.auc(rST[mT], det4["T"][mT]), T2S_raw=H.auc(rTS[mS], det4["S"][mS]),
                   Sh2Tdet=H.auc(pSx[mT], det4["T"][mT]),
                   TT_acc=H.auc(pTT, acc_late["T"] >= 0.5), TT_rho=spearman(pTT, acc_late["T"]),
                   SS_acc=H.auc(pSS, acc_late["S"] >= 0.5), SS_rho=spearman(pSS, acc_late["S"]),
                   TT_share_late=spearman(pTT, [maj_share(gT[r]["letters"][4:8]) for r in rows]))
        res["layer"][str(l)] = row; best[l] = (pSS, pTT, li)
        say(f"   L{l:<3d} S→S {row['SS']:.3f} | T→T {row['TT']:.3f} | S⇒T {row['S2T']:.3f} ({row['S2T_raw']:.3f}) | T⇒S {row['T2S']:.3f} ({row['T2S_raw']:.3f}) | "
            f"学生hidden→教师det4 {row['Sh2Tdet']:.3f} || 教师探针→对 {row['TT_acc']:.3f}/ρ{row['TT_rho']:.3f} | 学生探针→对 {row['SS_acc']:.3f}/ρ{row['SS_rho']:.3f} | 教师探针~教师后4稳定 ρ{row['TT_share_late']:.3f}")
    # shuffled-label control at the best layer of each within-view probe
    res["shuffled"] = {}
    for key, v in (("SS", "S"), ("TT", "T")):
        l = max(layers, key=lambda l: res["layer"][str(l)][key]); Z = zscore((LS if v == "S" else LT)[:, best[l][2]]); m = det4[v] >= 0
        Psh = H.cv_probe(Z, det4[v], np.ones((len(rows), 2), bool), folds, 2, shuffle=True)[:, 1]
        res["shuffled"][key] = dict(layer=l, value=res["layer"][str(l)][key], shuffled=H.auc(Psh[m], det4[v][m]))
        say(f"   打乱标签对照 {key}: L{l} {res['layer'][str(l)][key]:.3f} vs 打乱 {res['shuffled'][key]['shuffled']:.3f}")
    # teacher probe bins vs teacher correctness (diagnostic), at the best T->T layer
    l = res["shuffled"]["TT"]["layer"]; pTT = best[l][1]; q = np.quantile(pTT, [0.25, 0.5, 0.75]); b = np.digitize(pTT, q)
    res["teacher_probe_quartiles"] = {int(k): dict(n=int((b == k).sum()), teacher_acc_late=float(acc_late["T"][b == k].mean()),
                                                   student_acc_late=float(acc_late["S"][b == k].mean()), share_late=float(np.mean([maj_share(gT[r]["letters"][4:8]) for r, bb in zip(rows, b) if bb == k])))
                                      for k in range(4)}
    say(f"\n== 教师探针 (L{l}) 四分位 → 教师后4正确率 / 学生后4正确率 / 教师后4样本多数占比 (GT 仅诊断)")
    for k, v in res["teacher_probe_quartiles"].items():
        say(f"   Q{k + 1}: n={v['n']} 教师 {v['teacher_acc_late']:.3f} | 学生 {v['student_acc_late']:.3f} | 教师后4稳定性 {v['share_late']:.3f}")
    os.makedirs(args.out, exist_ok=True); json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    say("\n[tstab] saved summary.json")


if __name__ == "__main__":
    main()
