#!/usr/bin/env python
"""分析 rollout-disagreement dump: 特权 discrepancy 在 n 条 rollout 间是否一致?

指标 (全部 judge-free, 纯分布几何):
  1. 逐位置 discrepancy   D_t^view = TV(p⁺_t, p_S,t) / D_t^tgt = TV(q_t, p_S,t)
  2. 累积 discrepancy      C_i = mean_t D_t, S_i = sum_t D_t; 读数 = 8 条之间 CV
  3. 信号方向一致性 R      g_i = Σ_t (p_S,t − q_t) (= KL(q‖p_S) 对 student logits 的精确梯度)
                           R_n = ‖mean_i g_i‖ / mean_i‖g_i‖
                           R≈1 全一致 | R≈1/√n 独立 | R<1/√n 系统抵消
  4. 位置剖面              D_t 按相对位置 t/T 分 10 箱, 两两 Spearman
  5. n 体制模拟            R_8 (全 8 条) vs R_2 (C(8,2) 对的均值)

判据 (预登记): R_8 ≥0.7 强 agreement(假设否定) | 0.354–0.7 部分独立 | <0.354 系统抵消(假设成立)

用法: python probing/analyze_rollout_disagree.py --dump results/rd_dump [--beta 4.0]
"""
import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np

# 稠密 g 向量的长度。不硬编码模型词表: 取一个覆盖 Qwen3.5 (151936) 的上界即可 ——
# bincount 的 minlength 只要 ≥ 出现过的最大 token id, 且同一次分析里所有 rollout
# 必须用同一个长度 (否则 R_of 的向量无法对齐)。8×这个长度的 float64 ≈ 10MB, 可忽略。
VOCAB = 200_000


def add_tail(lp):
    """(T,k) logp -> (T,k+1), 与 core_algos.py:1141-1147 同口径。"""
    s = np.logaddexp.reduce(lp, axis=-1, keepdims=True)
    s = np.minimum(s, -1e-7)
    tail = np.log(-np.expm1(s))
    return np.concatenate([lp, tail], axis=-1)


def tv(a, b):
    return 0.5 * np.abs(a - b).sum(-1)


def jsd(p, q, alpha=0.5):
    m = alpha * p + (1 - alpha) * q
    lm = np.log(m + 1e-30)
    kl_p = (p * (np.log(p + 1e-30) - lm)).sum(-1)
    kl_q = (q * (np.log(q + 1e-30) - lm)).sum(-1)
    return alpha * kl_p + (1 - alpha) * kl_q


def load(dump):
    rows = defaultdict(list)
    for line in open(os.path.join(dump, "meta.jsonl")):
        r = json.loads(line)
        rows[r["row"]].append(r)
    return rows


def rollout_signal(d, beta):
    """返回 (g_sparse dict[token_id]->float, C_view, C_tgt, prof_view, prof_tgt, tail_mass)"""
    ids = d["ids"].astype(np.int64)
    lp_s, lp_p, lp_n = (add_tail(d[k].astype(np.float64)) for k in ("lp_stu", "lp_pos", "lp_null"))
    u = lp_p - lp_n
    logq = lp_p + beta * u
    logq -= logq.max(-1, keepdims=True)
    q = np.exp(logq)
    q /= q.sum(-1, keepdims=True)
    ps = np.exp(lp_s)
    ps /= ps.sum(-1, keepdims=True)
    pp = np.exp(lp_p)
    pp /= pp.sum(-1, keepdims=True)

    D_view = tv(pp, ps)
    D_tgt = tv(q, ps)
    # g = Σ_t (p_S − q) ∈ R^V, 只在 explicit 列 (丢 tail 列 —— 它不对应具体 token id)。
    # bincount 向量化: 同一 token 在不同位置的贡献自动累加 (比 Python 双循环快 ~100x)。
    delta = (ps - q)[:, :-1]
    g = np.bincount(ids.ravel(), weights=delta.ravel(), minlength=VOCAB)
    tail_mass = float(np.median(1.0 - np.exp(np.logaddexp.reduce(d["lp_stu"].astype(np.float64), -1))))
    return g, D_view, D_tgt, tail_mass


def R_of(gs):
    """R = ‖mean_i g_i‖ / mean_i ‖g_i‖。

    gs: list of 稠密 vocab 向量 (np.ndarray, 同长) 或 sparse dict (单测用)。
    标定: 完全同向 → 1; 独立 → 1/√n; 成对反向 → 0。
    """
    if len(gs) < 2:
        return float("nan")
    if isinstance(gs[0], dict):                      # 单测路径: 稀疏 dict, 取键并集
        keys = sorted(set().union(*(set(g) for g in gs)))
        M = np.array([[g.get(k, 0.0) for k in keys] for g in gs], dtype=np.float64)
    else:
        M = np.asarray(gs, dtype=np.float64)
    num = np.linalg.norm(M.mean(0))
    den = np.linalg.norm(M, axis=1).mean()
    return float(num / den) if den > 0 else float("nan")


def profile(D, nbin=10):
    """D_t 按相对位置分 nbin 箱。T < nbin 时返回全 nan (剖面无意义, 下游会跳过)。"""
    T = len(D)
    if T < nbin:
        return np.full(nbin, np.nan)
    edges = np.linspace(0, T, nbin + 1).astype(int)
    return np.array([D[edges[i]:edges[i + 1]].mean() if edges[i + 1] > edges[i] else np.nan
                     for i in range(nbin)])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok], b[ok]
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load(args.dump)
    L = []
    say = L.append
    say("=" * 78)
    say("Rollout-Disagreement 探针报告  (student=teacher=base, step-0 状态)")
    say(f"dump={args.dump}  beta={args.beta}  rows={len(rows)}")
    say("=" * 78)

    per_row = []
    tails = []
    for row in sorted(rows):
        recs = sorted(rows[row], key=lambda r: r["rollout_id"])
        gs, Cv, Ct, profs = [], [], [], []
        for r in recs:
            d = np.load(os.path.join(args.dump, r["npz"]))
            g, D_view, D_tgt, tm = rollout_signal(d, args.beta)
            gs.append(g)
            Cv.append(float(D_view.mean()))
            Ct.append(float(D_tgt.mean()))
            profs.append(profile(D_tgt))
            tails.append(tm)
        n = len(gs)
        R_full = R_of(gs)
        pairs = [R_of([gs[i], gs[j]]) for i, j in itertools.combinations(range(n), 2)]
        R_pair = float(np.nanmean(pairs)) if pairs else float("nan")
        sp = [spearman(profs[i], profs[j]) for i, j in itertools.combinations(range(n), 2)]
        cv_v = float(np.std(Cv) / np.mean(Cv)) if np.mean(Cv) > 0 else float("nan")
        cv_t = float(np.std(Ct) / np.mean(Ct)) if np.mean(Ct) > 0 else float("nan")
        per_row.append(dict(row=row, n=n, R_full=R_full, R_pair=R_pair,
                            C_view=float(np.mean(Cv)), C_tgt=float(np.mean(Ct)),
                            cv_view=cv_v, cv_tgt=cv_t,
                            spearman=float(np.nanmean(sp)) if sp else float("nan"),
                            box_frac=recs[0]["box_frac"], task=recs[0]["task"]))

    say("")
    say(f"{'row':>5}{'n':>3}{'R_n':>8}{'R_2':>8}{'TV(p+,pS)':>11}{'TV(q,pS)':>10}"
        f"{'CV(C)':>8}{'ρ(剖面)':>9}{'box%':>7}")
    for p in per_row:
        say(f"{p['row']:>5}{p['n']:>3}{p['R_full']:>8.3f}{p['R_pair']:>8.3f}"
            f"{p['C_view']:>11.4f}{p['C_tgt']:>10.4f}{p['cv_tgt']:>8.3f}"
            f"{p['spearman']:>9.3f}{100*p['box_frac']:>7.2f}")

    Rf = np.array([p["R_full"] for p in per_row])
    Rp = np.array([p["R_pair"] for p in per_row])
    n0 = int(np.median([p["n"] for p in per_row]))
    indep_n, indep_2 = 1 / np.sqrt(n0), 1 / np.sqrt(2)
    say("")
    say("-" * 78)
    say(f"R_n (n={n0}): mean={np.nanmean(Rf):.3f}  median={np.nanmedian(Rf):.3f}  "
        f"min={np.nanmin(Rf):.3f}  max={np.nanmax(Rf):.3f}")
    say(f"  独立基准 1/√{n0} = {indep_n:.3f}   完全一致 = 1.000")
    say(f"R_2: mean={np.nanmean(Rp):.3f}   独立基准 1/√2 = {indep_2:.3f}")
    # bootstrap CI (按 row 聚类)
    rng = np.random.default_rng(0)
    bs = [np.nanmean(rng.choice(Rf, size=len(Rf), replace=True)) for _ in range(2000)]
    say(f"R_n 均值 95% CI (row-clustered bootstrap): "
        f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
    say(f"support tail mass 中位数: {np.median(tails):.5f}  (应 <0.02)")
    say(f"CV(C_tgt) 均值: {np.nanmean([p['cv_tgt'] for p in per_row]):.3f}  (>0.5 = 强度差异大)")
    say(f"剖面 Spearman 均值: {np.nanmean([p['spearman'] for p in per_row]):.3f}")

    m = np.nanmean(Rf)
    say("")
    say("判读 (预登记判据):")
    if m >= 0.7:
        say(f"  R_n={m:.3f} ≥ 0.70 → **强 agreement**: rollout 间信号方向高度一致,")
        say("     n=8 的平均是有效降噪, 不稀释。→ disagreement 假设 **否定**,")
        say("     n8 的 TB 劣势需另找原因 (如 lr 未随有效批缩放)。")
    elif m >= indep_n:
        say(f"  {indep_n:.3f} ≤ R_n={m:.3f} < 0.70 → **部分独立**: 降噪为主但有稀释,")
        say("     边界情形; 需看 R_2 与 CV 辅助判断。")
    else:
        say(f"  R_n={m:.3f} < 1/√{n0}={indep_n:.3f} → **系统性抵消**: rollout 间 discrepancy")
        say("     方向互相对消, n=8 的平均在稀释信号。→ disagreement 假设 **成立**。")
    if np.nanmean([p["cv_tgt"] for p in per_row]) > 0.5:
        say("  辅助: CV(C_tgt) > 0.5 → 8 条信号强度差异大, 平均会被个别强信号主导。")

    # 按 box_frac 分层 (TreeBench 对齐轴)
    bf = np.array([p["box_frac"] for p in per_row])
    if np.isfinite(bf).sum() >= 4 and np.nanstd(bf) > 0:
        med = np.nanmedian(bf)
        lo, hi = Rf[bf <= med], Rf[bf > med]
        if len(lo) and len(hi):
            say("")
            say(f"按 GT 框占比分层 (中位 {100*med:.2f}%):")
            say(f"  小框 (≤中位): R_n={np.nanmean(lo):.3f} (n={len(lo)})")
            say(f"  大框 (>中位): R_n={np.nanmean(hi):.3f} (n={len(hi)})")

    txt = "\n".join(L)
    print(txt)
    out = args.out or os.path.join("results", "rollout_disagree_report.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(txt + "\n")
    print(f"\n>>> 报告已写入 {out}")


if __name__ == "__main__":
    main()
