#!/usr/bin/env python
"""Rollout-level loss 加权的候选 w(τ) 摸底 (纯 CPU, 复用已有 dump, 零 GPU)。

背景: OPD-Aha 的 loss 是 **token-mean** (actor.yaml:220), 即整批所有 token 的 loss 求和
除以总 token 数 —— 没有任何 rollout 级权重, 且**长 rollout 自动获得更大权重**
(T=291 的轨迹对梯度的影响是 T=4 的 73 倍, 纯由长度决定)。若要做
    w(τ) = 1 + η·B(τ),  L = Σ_τ w(τ)·mean_t L_t / Σ_τ w(τ)
就必须先回答: 候选 w(τ) 到底在加权什么?

本探针**不测** w(τ) 与正确性的关系 (那是上一版探针的设计偏差 —— 一条中途走错又纠正回来的
轨迹最终答对, 但它的纠错过程恰恰最有学习价值)。这里测三件事:

  1. **长度混杂**: corr(w, T)。若某个 w 与 T 高度相关, 它加权的其实是长度,
     那就退化回现状的 token-mean, 没有增量。
  2. **候选间冗余**: 两两 Spearman。高度相关的候选只需留一个。
  3. **动态范围**: w 的分散度 (CV、p90/p10)。若所有 rollout 的 w 几乎相同,
     加权是恒等操作, 再怎么设 η 也没有效果 (stn8 空转陷阱的另一种形态)。

候选 (全部只用已 dump 的 p_S / p⁺ / p⁰, 无需新概念):
  d_tgt   = mean_t TV(q_t, p_S,t)      目标与学生的分歧 = "可学的量"
  d_view  = mean_t TV(p⁺_t, p_S,t)     纯视图分歧
  u_abs   = mean_t mean_v |u_t(v)|     视觉证据强度
  u_neg   = mean_t Σ_v p⁺·[-u]_+       负向视觉质量 (supn8 的作用面)
  B_ema   = mean_t C_t  (TGNVMT)       轨迹冲突累积
  B_max   = max_t C_t
  d_peak  = max_t TV(q_t, p_S,t)       单点最剧烈分歧 (关键决策点?)
  ent_s   = mean_t H(p_S,t) (support内) 学生不确定性

用法 (计算节点, 无 GPU):
  PYTHONPATH=$REPO python probing/probe_rollout_weight.py --dumps results/rd_dump ...
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_rollout_disagree import add_tail, realized_state, tv  # noqa: E402


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok], b[ok]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def candidates(d, beta):
    """返回该 rollout 的所有候选 w 值 + T。"""
    lp_s = add_tail(d["lp_stu"].astype(np.float64))
    lp_p = add_tail(d["lp_pos"].astype(np.float64))
    lp_n = add_tail(d["lp_null"].astype(np.float64))
    u = lp_p - lp_n
    logq = lp_p + beta * u
    logq -= logq.max(-1, keepdims=True)
    q = np.exp(logq); q /= q.sum(-1, keepdims=True)
    ps = np.exp(lp_s); ps /= ps.sum(-1, keepdims=True)
    pp = np.exp(lp_p); pp /= pp.sum(-1, keepdims=True)

    D_tgt = tv(q, ps)
    D_view = tv(pp, ps)
    out = {
        "T": float(lp_s.shape[0]),
        "d_tgt": float(D_tgt.mean()),
        "d_view": float(D_view.mean()),
        "u_abs": float(np.abs(u).mean()),
        "u_neg": float((pp * np.maximum(-u, 0.0)).sum(-1).mean()),
        "d_peak": float(D_tgt.max()),
        "ent_s": float((-(ps * np.log(ps + 1e-30)).sum(-1)).mean()),
    }
    if "y" in d:
        s = realized_state(d, beta)
        out["B_ema"] = float(s["C_ema"].mean())
        out["B_max"] = float(s["C_max"].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--out", default="results/rollout_weight_report.txt")
    args = ap.parse_args()

    rows = []
    for dump in args.dumps:
        mp = os.path.join(dump, "meta.jsonl")
        if not os.path.exists(mp):
            continue
        for line in open(mp):
            r = json.loads(line)
            p = os.path.join(dump, r["npz"])
            if not os.path.exists(p):
                continue
            c = candidates(np.load(p), args.beta)
            c["dump"] = os.path.basename(dump)
            c["row"] = r["row"]
            rows.append(c)
    if not rows:
        print("无数据"); return

    keys = [k for k in ("d_tgt", "d_view", "u_abs", "u_neg", "d_peak", "ent_s", "B_ema", "B_max")
            if k in rows[0]]
    arr = {k: np.array([r[k] for r in rows]) for k in keys + ["T"]}
    n = len(rows)

    L = []
    say = L.append
    say("=" * 84)
    say(f"Rollout 加权候选 w(τ) 摸底   (n={n} 条 rollout, dumps={len(set(r['dump'] for r in rows))})")
    say("现状: loss_agg_mode=token-mean -> 无 rollout 级权重, 且长度即隐式权重")
    say("=" * 84)

    say("")
    say(f"T (rollout 长度): mean={arr['T'].mean():.1f}  p10={np.percentile(arr['T'],10):.0f}  "
        f"p90={np.percentile(arr['T'],90):.0f}  max={arr['T'].max():.0f}")
    say(f"  -> token-mean 下最长/最短 rollout 的梯度权重比 = "
        f"{arr['T'].max()/max(arr['T'].min(),1):.0f}x  (纯由长度决定)")

    say("")
    say("① 动态范围 与 ② 长度混杂")
    say(f"{'候选':<10}{'mean':>9}{'CV':>7}{'p90/p10':>9}{'corr(w,T)':>11}  判读")
    for k in keys:
        v = arr[k]
        cv = v.std() / max(abs(v.mean()), 1e-12)
        p10, p90 = np.percentile(v, 10), np.percentile(v, 90)
        ratio = p90 / p10 if p10 > 0 else float("inf")
        rho = spearman(v, arr["T"])
        flag = []
        if cv < 0.15:
            flag.append("范围过窄(近恒等)")
        if abs(rho) > 0.5:
            flag.append(f"**与长度强相关**")
        elif abs(rho) > 0.3:
            flag.append("与长度中度相关")
        if not flag:
            flag.append("可用")
        say(f"{k:<10}{v.mean():>9.4f}{cv:>7.2f}{ratio:>9.2f}{rho:>11.3f}  {', '.join(flag)}")

    say("")
    say("③ 候选间 Spearman (高相关 => 冗余, 只需留一个)")
    hdr = "          " + "".join(f"{k[:7]:>9}" for k in keys)
    say(hdr)
    for i, k1 in enumerate(keys):
        line = f"{k1:<10}"
        for j, k2 in enumerate(keys):
            line += "        -" if i == j else f"{spearman(arr[k1], arr[k2]):>9.2f}"
        say(line)

    # 与长度去相关后还剩多少信号: 对 T 回归取残差, 看残差的 CV
    say("")
    say("④ 去长度后的剩余变异 (对 log T 线性回归的残差 CV; 越大 = 越不是长度的替身)")
    lt = np.log(np.maximum(arr["T"], 1))
    for k in keys:
        v = arr[k]
        A = np.vstack([lt, np.ones_like(lt)]).T
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        resid = v - A @ coef
        r2 = 1 - resid.var() / max(v.var(), 1e-30)
        say(f"  {k:<10} R²(logT)={r2:>6.3f}   残差 CV={resid.std()/max(abs(v.mean()),1e-12):>6.3f}")

    say("")
    say("=" * 84)
    say("选型判据:")
    say("  - corr(w,T) 绝对值 >0.5  -> 它加权的是长度, 与 token-mean 现状差别不大, 弃")
    say("  - CV <0.15               -> 动态范围过窄, 加权近恒等, 弃")
    say("  - 与已选候选 |ρ|>0.8     -> 冗余, 只留一个")
    say("  - 三关都过 & R²(logT) 低 -> 真正携带长度之外信息的候选")
    say("")
    say("注: 本探针不判定 w(τ) 是否**有效** (那要训练), 只排除三类先天不可用的候选。")

    txt = "\n".join(L)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")
    print(f"\n>>> 报告已写入 {args.out}")


if __name__ == "__main__":
    main()
