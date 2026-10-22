#!/usr/bin/env python
"""Offline check (user 10-22): can a pre-generation, label-free signal pick the questions where OPD helps / hurts?

(1) Student uncertainty, label-free. Target det_rest = base samples 2..7 all agree (>=5 parsed; no GT). Scores, all
    available before / without those samples:
      t0_max   max over options of the t=0 letter readout (renormalised over the valid options)
      t0_ent   -entropy of that readout
      probe    cross-fitted linear probe on the last-prompt-token hidden state (one layer), trained on det_rest
      n2       base samples 0,1 agree (what rollout_n=2 already gives for free during training)
    det_rest needs >=5 of the 6 parsed (so it is not mostly a parse-rate target). The probe is cross-fitted on rows with a
    valid det_rest but scores EVERY row (part (2) bins all rows; dropping rows would select on samples 4..7 parsing).
    Reported: AUC per score, and the AUC of t0/probe inside the n2-agree / n2-disagree strata (do they add to n2?).
(2) Where does training gain / lose? Questions binned by student uncertainty (probe tertiles; n2 as an alternative) x
    teacher self-consistency (majority share of teacher-view samples 0..3; label-free; the diagnostic teacher-right
    column uses teacher samples 4..7 so it is not mechanically tied to the bin). Outcome uses GT
    (diagnostic exception, as in probe_traj_within --part cross): null = base samples 4..7, treatment = trained 8 samples.
    Bins never use base samples 4..7, so the null carries no regression-to-the-mean bias. Big-drop counts compare
    equal sample sizes: trained samples 0..3 vs base 4..7, each minus base 0..3.
GT is used only for the diagnostic outcome columns; nothing here feeds training.
用法: python scripts/probe_difficulty_select.py --base <dir> --trained <dir> --teacher <dir> --layer 31 --out <dir>
"""
import argparse, collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_hidden_answer as H
from probe_traj_within import rate, rate_of


def agree(letters, idx, min_parsed):
    """1 if the parsed letters at idx all agree, 0 if not, -1 if fewer than min_parsed parsed."""
    p = [letters[i] for i in idx if letters[i] is not None]
    if len(p) < min_parsed: return -1
    return int(len(set(p)) == 1)


def maj_share(letters):
    """Share of the most frequent letter among ALL samples (unparsed count against consistency); 0 if nothing parsed."""
    c = collections.Counter(x for x in letters if x is not None)
    return c.most_common(1)[0][1] / len(letters) if c else 0.0


def t0_scores(lp, M):
    """Renormalised t0 readout over valid options -> (max prob, -entropy)."""
    lpv = np.where(M, lp, -np.inf); lpv = lpv - np.logaddexp.reduce(lpv, 1, keepdims=True)
    p = np.exp(lpv); ent = -np.where(M, p * np.where(M, lpv, 0), 0).sum(1)
    return p.max(1), -ent


def crossfit_all(X, y, folds, lams=(1e-3, 1e-2, 1e-1, 1.0)):
    """Binary probe P(y=1) for EVERY row: per fold, train on valid (y>=0) rows of the other folds (lam by inner held-out
    CE, as H.cv_probe), predict all rows of the fold including y=-1 ones."""
    out = np.full(len(y), np.nan); ok = y >= 0; two = np.ones((len(y), 2), bool)
    for f in np.unique(folds):
        tr, te = ok & (folds != f), folds == f
        inner = folds[tr]; fi = np.unique(inner)[0]; a, v = inner != fi, inner == fi
        best = min(lams, key=lambda lam: -np.mean(H.fit_predict(X[tr][a], y[tr][a], two[tr][a], X[tr][v], two[tr][v], lam, 2)[np.arange(v.sum()), y[tr][v]]))
        out[te] = np.exp(H.fit_predict(X[tr], y[tr], two[tr], X[te], two[te], best, 2)[:, 1])
    return out


def tertile(x):
    """0/1/2 by within-sample tertiles (low / mid / high)."""
    q = np.quantile(x, [1 / 3, 2 / 3]); return np.digitize(x, q, right=True)


TB = lambda ts: {"教师一致(4/4)": ts == 1, "教师多数(3/4)": ts == 0.75, "教师分歧(≤2/4)": ts < 0.75}


def cells(stud_bin, stud_names, tshare, rn, rs, rt, rb, rs4):
    """Per (student bin x teacher-consistency bin): n, teacher >=half right (diag), base null (samples 4-7), trained,
    effect, big drops (trained samples 0-3 minus base 0-3 <= -0.5, among base0-3 >= 0.5) and the null count (base 4-7)."""
    tb = TB(tshare)
    out = {}; elig = rb >= 0.5
    for si, sn in enumerate(stud_names):
        for tn, tm in tb.items():
            m = (stud_bin == si) & tm
            if not m.any(): out[f"{sn}|{tn}"] = dict(n=0); continue
            out[f"{sn}|{tn}"] = dict(n=int(m.sum()), teacher_right=float((rt[m] >= 0.5).mean()), base_null=float(rn[m].mean()),
                                    trained=float(rs[m].mean()), effect=float((rs - rn)[m].mean()),
                                    drops=int((((rs4 - rb) <= -0.5) & elig & m).sum()), drops_null=int((((rn - rb) <= -0.5) & elig & m).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--trained", required=True); ap.add_argument("--teacher", required=True)
    ap.add_argument("--layer", type=int, default=31); ap.add_argument("--out", required=True)
    args = ap.parse_args(); say = lambda s="": print(s, flush=True)
    ld = lambda d: {g["row"]: g for g in json.load(open(os.path.join(d, "gen.json")))}
    base, trn, tch = ld(args.base), ld(args.trained), ld(args.teacher)
    z = np.load(os.path.join(args.base, "readout.npz")); rows_h = [int(r) for r in z["rows"]]
    keep = [i for i, r in enumerate(rows_h) if r in trn and r in tch and len(base[r]["letters"]) >= 8]
    rows = [rows_h[i] for i in keep]; lp = z["letter_logp"][keep]
    M = np.zeros((len(rows), 6), bool)
    for i, r in enumerate(rows):
        for o in base[r]["opts"]: M[i, H.LETTERS.index(o)] = True
    # hidden at one layer, last prompt token (position index 1) -- memmap, one strided read
    Hm = np.load(os.path.join(args.base, "hidden.npy"), mmap_mode="r")
    assert Hm.shape[0] == len(rows_h) and 0 <= args.layer < Hm.shape[2], (Hm.shape, len(rows_h), args.layer)
    assert M.any(1).all(), "row without options"
    assert all(len(tch[r]["letters"]) >= 8 and len(trn[r]["letters"]) >= 8 for r in rows)
    X = np.asarray(Hm[keep, 1, args.layer], dtype=np.float32)
    say(f"== {len(rows)} 题; 探针层 L{args.layer} (prompt 末 token)")

    # ---------------- (1) label-free uncertainty
    det_rest = np.array([agree(base[r]["letters"], range(2, 8), 5) for r in rows])
    n2 = np.array([agree(base[r]["letters"], (0, 1), 2) for r in rows])
    t0_max, t0_negent = t0_scores(lp, M)
    folds = H.group_folds([base[r]["group"] for r in rows]); two = np.ones((len(rows), 2), bool)
    probe = crossfit_all(X, det_rest, folds); assert np.isfinite(probe).all()
    Psh = H.cv_probe(X, det_rest, two, folds, 2, shuffle=True)[:, 1]
    m = det_rest >= 0
    res = dict(n=len(rows), layer=args.layer, n_det_rest=int(m.sum()), det_rest_pos=float(det_rest[m].mean()), auc={}, strata={})
    sc = dict(t0_max=t0_max, t0_negent=t0_negent, probe=probe, probe_shuffled=Psh)
    for k, v in sc.items(): res["auc"][k] = H.auc(v[m], det_rest[m])
    mn = m & (n2 >= 0); res["auc"]["n2"] = H.auc(n2[mn], det_rest[mn])
    for sn, sm in (("n2一致", mn & (n2 == 1)), ("n2不一致", mn & (n2 == 0))):
        res["strata"][sn] = dict(n=int(sm.sum()), pos=float(det_rest[sm].mean()), **{k: H.auc(sc[k][sm], det_rest[sm]) for k in ("t0_max", "probe")})
    say(f"\n== (1) 预测 base 采样 2..7 是否全一致 (无 GT); n={res['n_det_rest']}, 正类 {res['det_rest_pos']:.3f}")
    say("   AUC: " + " | ".join(f"{k} {v:.3f}" for k, v in res["auc"].items()))
    for sn, s in res["strata"].items():
        say(f"   分层 {sn}: n={s['n']} 正类 {s['pos']:.3f} | t0_max {s['t0_max']:.3f} | probe {s['probe']:.3f}")

    # ---------------- (2) where training gains / loses
    rb = np.array([rate_of(base[r], range(4)) for r in rows]); rn = np.array([rate_of(base[r], range(4, 8)) for r in rows])
    rs = np.array([rate(trn[r]) for r in rows])
    rs4 = np.array([rate_of(trn[r], range(4)) for r in rows])
    tshare = np.array([maj_share(tch[r]["letters"][:4]) for r in rows])          # bin: teacher samples 0..3
    rt = np.array([rate_of(tch[r], range(4, 8)) for r in rows])                  # diagnostic: teacher samples 4..7
    res["teacher_consistency"] = {k: dict(n=int(v.sum()), teacher_right=float((rt[v] >= 0.5).mean())) for k, v in TB(tshare).items()}
    res["teacher_consistency_auc_right"] = H.auc(tshare, rt >= 0.5)
    say(f"\n== (2) 训练效应 = 训练后 8 条 − base 后 4 条 (GT 仅诊断); 分桶不用 base 后 4 条; 教师桶用教师前 4 条, 教师对错(诊断)用教师后 4 条")
    say("   教师自一致性 → 教师后4条≥半数对 的比例: " + " | ".join(f"{k} n={v['n']} {v['teacher_right']:.3f}" for k, v in res["teacher_consistency"].items())
        + f" | AUC {res['teacher_consistency_auc_right']:.3f}")
    stud = {"probe三分位": (tertile(probe), ["学生不确定(probe低)", "学生中等", "学生确定(probe高)"]),
            "n2": (np.where(n2 == 1, 1, 0), ["n2不一致", "n2一致"])}
    res["cells"] = {}
    for key, (b, names) in stud.items():
        valid = np.ones(len(rows), bool) if key != "n2" else n2 >= 0
        c = cells(np.where(valid, b, -1), names, tshare, rn, rs, rt, rb, rs4); res["cells"][key] = c
        say(f"\n   [{key}] 学生桶 × 教师桶: n | 教师≥半数对(诊断) | base 后4 | 训练后 | 效应 | 大幅退步 训练后前4 (零假设 base 后4)")
        for k, v in c.items():
            if v["n"] == 0: continue
            say(f"   {k:34s} n={v['n']:5d} | {v['teacher_right']:.3f} | {v['base_null']:.3f} | {v['trained']:.3f} | {v['effect']:+.3f} | {v['drops']:3d} ({v['drops_null']})")
    os.makedirs(args.out, exist_ok=True); json.dump(res, open(os.path.join(args.out, "select_summary.json"), "w"), indent=1, ensure_ascii=False)
    say("\n[select] saved select_summary.json")


if __name__ == "__main__":
    main()
