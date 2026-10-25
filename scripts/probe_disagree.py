#!/usr/bin/env python
"""Teacher-student answer disagreement precheck (user 10-25): before gating "distil on the teacher's own trajectory" on the
questions where the teacher's opinion would be masked by the student's prefix (prefix dominance, fork probe 10-17), check
  (1) label-free disagreement D: answer distribution of the student view (red-box full image + hint) vs the teacher view
      (crop), base Qwen3.5-4B, 1024-token T=1 samples. D is built from samples 0..3 of each view;
  (2) can it be read before generation: cross-fitted linear probes on the last-prompt-token hidden of the student view /
      teacher view / both, and the label-free t=0 readout distance TV(p_S^0, p_T^0) (available at training time: the teacher
      forward already scores the first response token);
  (3) DIAGNOSTIC ONLY (GT, user exception; nothing feeds training): on samples 4..7 (disjoint from D), is the teacher right
      more often than the student where they disagree? Plus majority-of-0..3 correctness split inside D=1.
用法: python scripts/probe_disagree.py --student <dir> --teacher <dir> --scores <tsw scores.npz> --out <dir>
"""
import argparse, collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_hidden_answer as H
from probe_difficulty_select import crossfit_all
from probe_teacher_stability import load_view, LAST, spearman

NONE = "_"   # unparsed answer is its own category


def dist(letters):
    """Empirical answer distribution (unparsed = NONE)."""
    c = collections.Counter(NONE if x is None else x for x in letters); n = len(letters)
    return {k: v / n for k, v in c.items()}


def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def maj(letters, min_parsed=3):
    """Unique most frequent parsed letter with >= min_parsed parsed (as det4), else None (tie / too few parsed)."""
    c = collections.Counter(x for x in letters if x is not None).most_common()
    if sum(v for _, v in c) < min_parsed or (len(c) > 1 and c[0][1] == c[1][1]): return None
    return c[0][0]


def t0_dist(lp, opts):
    """Renormalised t=0 letter readout over the question's options -> [6] probabilities (0 off-option)."""
    m = np.array([l in opts for l in H.LETTERS]); v = np.where(m, lp, -np.inf)
    v = v - np.logaddexp.reduce(v); return np.where(m, np.exp(v), 0.0)


def main():
    ap = argparse.ArgumentParser()
    for a in ("student", "teacher", "scores", "out"): ap.add_argument("--" + a, required=True)
    ap.add_argument("--layers", default="8,16,20,24,28,31,32")
    args = ap.parse_args(); say = lambda s="": print(s, flush=True)
    layers = [int(x) for x in args.layers.split(",")]
    gS, rS, lpS, HS = load_view(args.student); gT, rT, lpT, HT = load_view(args.teacher)
    iS, iT = {r: i for i, r in enumerate(rS)}, {r: i for i, r in enumerate(rT)}
    rows = [r for r in rS if r in iT and len(gS[r]["letters"]) >= 8 and len(gT[r]["letters"]) >= 8]
    for r in rows:
        assert gS[r]["gt"] == gT[r]["gt"] and gS[r]["opts"] == gT[r]["opts"] and gS[r]["group"] == gT[r]["group"], f"row {r}"
    kS, kT = [iS[r] for r in rows], [iT[r] for r in rows]
    folds = H.group_folds([gS[r]["group"] for r in rows])
    n = len(rows); say(f"== {n} 题 (两视图各 8 样本)")

    # (1) label-free disagreement on samples 0..3
    tv4 = np.array([tv(dist(gS[r]["letters"][:4]), dist(gT[r]["letters"][:4])) for r in rows])
    tv8 = np.array([tv(dist(gS[r]["letters"][:8]), dist(gT[r]["letters"][:8])) for r in rows])
    tvh = np.array([tv(dist(gS[r]["letters"][4:8]), dist(gT[r]["letters"][4:8])) for r in rows])   # held-out reference
    unp = {v: np.array([sum(x is None for x in g[r]["letters"][:8]) / 8 for r in rows]) for v, g in (("S", gS), ("T", gT))}
    mS4 = [maj(gS[r]["letters"][:4]) for r in rows]; mT4 = [maj(gT[r]["letters"][:4]) for r in rows]
    D = np.array([-1 if a is None or b is None else int(a != b) for a, b in zip(mS4, mT4)])
    ok = D >= 0
    say(f"   TV4 均值 {tv4.mean():.3f}, TV(样本4..7) 均值 {tvh.mean():.3f}, ρ(TV4, TV4..7) {spearman(tv4, tvh):.3f} (两视图 vLLM 同 seed=42, 不一致度偏低估)")
    say(f"   D4 (两视图 0..3 的唯一多数答案不同, 需 ≥3 可解析): 有效 {ok.sum()} 题 (无定义 {(~ok).mean():.3f}), D=1 占 {D[ok].mean():.3f}")
    for nm, m in (("D=0", D == 0), ("D=1", D == 1), ("D=-1", D < 0)):
        say(f"   未解析率 {nm}: 学生 {unp['S'][m].mean():.3f} 教师 {unp['T'][m].mean():.3f}")

    # (2) readable before generation?
    pS0 = np.stack([t0_dist(lpS[k], gS[r]["opts"]) for k, r in zip(kS, rows)])
    pT0 = np.stack([t0_dist(lpT[k], gT[r]["opts"]) for k, r in zip(kT, rows)])
    tv0 = 0.5 * np.abs(pS0 - pT0).sum(1)
    massS = np.array([np.exp(lpS[k]).sum() for k in kS]); massT = np.array([np.exp(lpT[k]).sum() for k in kT])
    say(f"   t=0 原始字母总质量 (重归一化前): 学生 中位 {np.median(massS):.3f} 教师 中位 {np.median(massT):.3f}")
    sc = np.load(args.scores); pos = {int(r): i for i, r in enumerate(sc["row"])}
    assert all(r in pos for r in rows), "rows missing from TSW scores"
    p_tsw = np.array([sc["p"][pos[r]] for r in rows])
    res = dict(n=n, D_valid=int(ok.sum()), D_rate=float(D[ok].mean()), tv4_mean=float(tv4.mean()), layer={})
    say(f"\n== 生成前可读性 (AUC→D4=1, ρ→TV(样本4..7, 与 D 不相交); 交叉拟合, 按图分组 5 折; 主报 L28, 其余层仅参考)")
    base = {"t0 读出 TV(p_S⁰,p_T⁰)": tv0, "TSW 分数 1−p": 1 - p_tsw}
    for k, v in base.items():
        say(f"   {k:24s} AUC {H.auc(v[ok], D[ok]):.3f}  ρ(TVh) {spearman(v, tvh):+.3f}  (n={int(ok.sum())})")
        res[k] = dict(auc=H.auc(v[ok], D[ok]), rho_tvh=spearman(v, tvh), n=int(ok.sum()))
    best = {}
    for l in layers:
        XS = np.asarray(HS[:, LAST, l], np.float32)[kS]; XT = np.asarray(HT[:, LAST, l], np.float32)[kT]
        out = {}
        for nm, X in (("S", XS), ("T", XT), ("S+T", np.concatenate([XS, XT], 1))):
            p = crossfit_all(X, D, folds); out[nm] = (H.auc(p[ok], D[ok]), spearman(p, tvh))
            if nm == "S" and l == 28: pS28 = p
            if nm not in best or out[nm][0] > best[nm][1]: best[nm] = (l, out[nm][0], p)
        res["layer"][str(l)] = out
        say(f"   L{l:<3d} " + " | ".join(f"{nm} AUC {a:.3f} ρ {b:+.3f}" for nm, (a, b) in out.items()))
    sh = []
    for seed in (0, 1):   # null for the max-over-layers selection: shuffle, every layer, take the max
        Dsh = D.copy(); Dsh[ok] = np.random.default_rng(seed).permutation(D[ok])
        sh.append(max(H.auc(crossfit_all(np.asarray(HS[:, LAST, l], np.float32)[kS], Dsh, folds)[ok], Dsh[ok]) for l in layers))
    say(f"   打乱标签对照 (S, 各层取最大, 2 个 seed): AUC {', '.join(f'{a:.3f}' for a in sh)}")
    res["shuffle_max_auc"] = sh

    # (3) diagnostic: who is right where they disagree (samples 4..7, disjoint from D)
    gt = [gS[r]["gt"] for r in rows]
    aS = np.array([np.mean([x == g for x in gS[r]["letters"][4:8]]) for r, g in zip(rows, gt)])
    aT = np.array([np.mean([x == g for x in gT[r]["letters"][4:8]]) for r, g in zip(rows, gt)])
    say(f"\n== 诊断 (GT, 样本 4..7): 教师 acc / 学生 acc / 差")
    say(f"   全部        n={n:5d}  {aT.mean():.3f} / {aS.mean():.3f} / {aT.mean() - aS.mean():+.3f}")
    for nm, m in (("D=0 (一致)", D == 0), ("D=1 (不一致)", D == 1), ("D 无唯一多数", D < 0)):
        say(f"   {nm:12s} n={m.sum():5d}  {aT[m].mean():.3f} / {aS[m].mean():.3f} / {aT[m].mean() - aS[m].mean():+.3f}")
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):   # TV4 is on a 1/4 grid
        m = (tv4 >= lo) & (tv4 < hi)
        if m.any(): say(f"   TV4∈[{lo:.2f},{min(hi, 1):.2f}) n={m.sum():5d}  {aT[m].mean():.3f} / {aS[m].mean():.3f} / {aT[m].mean() - aS[m].mean():+.3f}")
    m1 = D == 1
    tr = np.array([a == g for a, g in zip(mT4, gt)]); sr = np.array([a == g for a, g in zip(mS4, gt)])
    cats = {"教师对 学生错": tr & ~sr, "教师错 学生对": ~tr & sr, "都错": ~tr & ~sr}
    say(f"   D=1 内 (0..3 多数答案): " + " | ".join(f"{k} {(v & m1).sum()} ({(v & m1).sum() / m1.sum():.2f})" for k, v in cats.items()))
    # gate quality: top-25% of the pre-registered L28 student-view score, and of the t=0 readout TV
    pS_best = pS28; top = pS_best >= np.quantile(pS_best, 0.75)
    t0top = tv0 >= np.quantile(tv0, 0.75)
    say(f"   t=0 读出 TV 前 25% 题: D=1 占 {D[t0top & ok].mean():.3f}; 教师−学生 acc {aT[t0top].mean() - aS[t0top].mean():+.3f} (其余 {aT[~t0top].mean() - aS[~t0top].mean():+.3f}); "
        f"教师对学生错 {(cats['教师对 学生错'] & t0top).sum()} / 教师错学生对 {(cats['教师错 学生对'] & t0top).sum()}")
    say(f"   学生视图探针 (L28) 前 25% 题 (含 D 无定义行): D=1 占 {D[top & ok].mean():.3f} (全体 {D[ok].mean():.3f}); 教师−学生 acc {aT[top].mean() - aS[top].mean():+.3f} (其余 {aT[~top].mean() - aS[~top].mean():+.3f}); "
        f"教师对学生错 {(cats['教师对 学生错'] & top).sum()} / 教师错学生对 {(cats['教师错 学生对'] & top).sum()}")
    res["diag"] = dict(acc_T=float(aT.mean()), acc_S=float(aS.mean()),
                       D1=dict(acc_T=float(aT[m1].mean()), acc_S=float(aS[m1].mean()), **{k: int((v & m1).sum()) for k, v in cats.items()}))
    res["best"] = {k: (v[0], v[1]) for k, v in best.items()}; res["args"] = vars(args)
    os.makedirs(args.out, exist_ok=True); json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    np.savez(os.path.join(args.out, "scores.npz"), row=np.array(rows), D=D, tv4=tv4, tv8=tv8, tvh=tvh, tv0=tv0, p_S28=pS28)
    say(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
