#!/usr/bin/env python
"""Export the frozen teacher's per-question stability score for TSW (teacher_stability/tsw_initial_training_plan.md §3.2).

score p_i = out-of-fold P(det4=1 | teacher-view (crop) last-prompt-token hidden at --layer), cross-fitted with the same
image-grouped 5 folds as probe_teacher_stability.py (crossfit_all; features are standardised inside each fold's train split
by H.fit_predict, the regulariser is picked on an inner held-out fold of the train split). det4 = samples 0..3 all agree
(>=3 parsed); rows with det4=-1 get an out-of-fold score but are not trained on. No GT anywhere (gt/opts are only copied
for the parquet identity check).
Also written for later rounds / checks: share8 (majority share of all 8 samples, unparsed count against), det4, fold, group.

用法: python scripts/export_tsw_scores.py --teacher <dir: gen.json, hidden.npy, readout.npz> --out <dir> [--layer 28]
      [--summary <tstab summary.json> to check the OOF AUC against the recorded T->T value]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_hidden_answer as H
from probe_difficulty_select import agree, maj_share, crossfit_all
from probe_teacher_stability import load_view, LAST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=28); ap.add_argument("--summary", default=None)
    ap.add_argument("--auc_tol", type=float, default=0.01)
    args = ap.parse_args(); say = lambda s="": print(s, flush=True)
    gen, rows_h, _, HT = load_view(args.teacher)
    rows = [r for r in rows_h if len(gen[r]["letters"]) >= 8]
    assert len(rows) == len(rows_h), f"{len(rows_h) - len(rows)} rows with <8 samples"
    assert len(set(rows)) == len(rows), "duplicate rows in readout"
    k = list(range(len(rows_h)))                                       # hidden index i <-> rows_h[i]
    X = np.asarray(HT[:, LAST, args.layer], dtype=np.float32)[k]
    groups = [gen[r]["group"] for r in rows]
    folds = H.group_folds(groups)
    det4 = np.array([agree(gen[r]["letters"], range(4), 3) for r in rows])
    share8 = np.array([maj_share(gen[r]["letters"][:8]) for r in rows])
    p = crossfit_all(X, det4, folds)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "scores must be finite and in [0, 1] (only the ranking is used)"
    m = det4 >= 0
    oof_auc = H.auc(p[m], det4[m])
    say(f"[tsw_scores] {len(rows)} 题, L{args.layer}, det4 有效 {int(m.sum())} 正类 {det4[m].mean():.3f}; OOF AUC {oof_auc:.4f}; "
        f"p 均值 {p.mean():.3f} 分位 {np.quantile(p, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()}")
    meta = dict(layer=args.layer, n=len(rows), n_valid=int(m.sum()), oof_auc=oof_auc, teacher_dir=os.path.abspath(args.teacher),
                hidden_shape=list(HT.shape), p_mean=float(p.mean()), spearman_p_share8=None)
    if args.summary:
        rec = json.load(open(args.summary))["layer"][str(args.layer)]["TT"]
        meta["summary_TT"] = rec
        say(f"[tsw_scores] summary.json T->T L{args.layer} = {rec:.4f} (|Δ| {abs(rec - oof_auc):.4f})")
        assert abs(rec - oof_auc) <= args.auc_tol, f"OOF AUC {oof_auc:.4f} != recorded {rec:.4f}"
    from probe_teacher_stability import spearman
    meta["spearman_p_share8"] = spearman(p, share8)
    say(f"[tsw_scores] Spearman(p, 8 样本多数占比) = {meta['spearman_p_share8']:.3f}")
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "scores.npz"), row=np.array(rows), p=p, det4=det4, share8=share8, fold=folds,
             group=np.array(groups), gt=np.array([gen[r]["gt"] for r in rows]),
             opts=np.array(["".join(gen[r]["opts"]) for r in rows]))
    json.dump(meta, open(os.path.join(args.out, "scores_meta.json"), "w"), indent=1)
    say(f"[tsw_scores] saved {args.out}/scores.npz")


if __name__ == "__main__":
    main()
