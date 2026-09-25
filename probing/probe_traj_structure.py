#!/usr/bin/env python
"""轨迹级结构测量 (纯 CPU, 复用已有 dump, 零 GPU) —— 描述性, 不需要任何正确性标签。

动机: aha-opd 的 u_t(v) 逐位置独立计算、loss 按 token-mean 聚合, 所以整条管线里
**没有任何地方能看到视觉证据沿轨迹的趋势**。一条 u 持续为负的轨迹与一条 u 正负交替的
轨迹, 在 token-mean 下贡献相同的平均信号, 但轨迹级语义截然不同。

测四组:

第一组 两通道分解 (aha-opd 的"抬升"与"压低"两半, 显式拆开):
    P(τ) = mean_t Σ_v p⁺(v)·[u]_+     抬升通道
    N(τ) = mean_t Σ_v p⁺(v)·[-u]_+    压低通道
  关键: corr(P,N) 跨轨迹。若高相关 => 两通道在轨迹级是同一个量 (分别加权无意义);
  若低相关 => 存在"抬升主导"与"压低主导"两类轨迹, 这才是 reweight 的抓手。
  (supn8 删掉整个 P 通道后终点持平 —— 若轨迹级存在 P 主导的一类, 其损失可能被
   N 主导轨迹的无损所平均掉, 这种异质性正是加权能利用的。)

第二组 时序结构:
    自相关 ρ(1)/ρ(5)  —— **最关键**: TGNVMT 的 EMA 隐含假设 u 有时序持续性,
                         但从未验证。若 ρ(1)≈0, 任何轨迹级聚合都只是在平均噪声。
    趋势斜率          —— u/N 对 t/T 的回归斜率: 视觉分歧随推理积累还是消解?
    通道时序分离      —— P_t 与 N_t 的峰值位置差、互相关: 是否"前半建立解释、后半被否定"

第三组 集中度: N_t 沿 t 的 Gini / 峰值相对位置 (时间轴上的"弥散 vs 尖峰";
  ST 的教训是词表轴上"抑制需要弥散", 这里看时间轴的天然结构)

第四组 可用性: 跨轨迹 CV、与长度 T 的相关、两两冗余

用法: PYTHONPATH=$REPO python probing/probe_traj_structure.py --dumps results/rd_* ...
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_rollout_disagree import add_tail  # noqa: E402


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


def autocorr(x, lag):
    """序列自相关 (Pearson, 去均值)。序列太短返回 nan。"""
    x = np.asarray(x, float)
    if len(x) <= lag + 2:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def gini(x):
    x = np.sort(np.maximum(np.asarray(x, float), 0))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def traj_features(d):
    """单条 rollout 的全部结构量。"""
    lp_p = add_tail(d["lp_pos"].astype(np.float64))
    lp_n = add_tail(d["lp_null"].astype(np.float64))
    u = lp_p - lp_n                                   # (T, k+1)
    pp = np.exp(lp_p); pp /= pp.sum(-1, keepdims=True)
    T = u.shape[0]

    # 逐位置的两通道质量 (p⁺ 加权)
    P_t = (pp * np.maximum(u, 0.0)).sum(-1)           # 抬升
    N_t = (pp * np.maximum(-u, 0.0)).sum(-1)          # 压低
    u_bar_t = (pp * u).sum(-1)                        # 净签名信号

    rel = np.arange(T) / max(T - 1, 1)                # 相对位置 0..1
    def slope(y):
        if T < 4:
            return float("nan")
        A = np.vstack([rel, np.ones_like(rel)]).T
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(c[0])

    f = {
        "T": float(T),
        # --- 第一组: 两通道 ---
        "P": float(P_t.mean()),
        "N": float(N_t.mean()),
        "PN_ratio": float(P_t.mean() / max(N_t.mean(), 1e-12)),
        "u_net": float(u_bar_t.mean()),
        # --- 第二组: 时序 ---
        "ac1_N": autocorr(N_t, 1),
        "ac5_N": autocorr(N_t, 5),
        "ac1_u": autocorr(u_bar_t, 1),
        "ac5_u": autocorr(u_bar_t, 5),
        "slope_N": slope(N_t),
        "slope_u": slope(u_bar_t),
        # P_t 与 N_t 的同步性 (同一轨迹内的逐位置相关) 与峰值位置差
        "PN_sync": float(np.corrcoef(P_t, N_t)[0, 1]) if T > 3 and P_t.std() > 0 and N_t.std() > 0 else float("nan"),
        "peak_pos_N": float(rel[int(np.argmax(N_t))]) if T > 3 else float("nan"),
        "peak_pos_P": float(rel[int(np.argmax(P_t))]) if T > 3 else float("nan"),
        # --- 第三组: 集中度 ---
        "gini_N": gini(N_t),
        "gini_P": gini(P_t),
    }
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--min-T", type=int, default=10, help="短于此的 rollout 不参与时序统计")
    ap.add_argument("--out", default="results/traj_structure_report.txt")
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
            f = traj_features(np.load(p))
            f["dump"] = os.path.basename(dump)
            rows.append(f)
    if not rows:
        print("无数据"); return

    A = {k: np.array([r.get(k, np.nan) for r in rows], float)
         for k in rows[0] if k != "dump"}
    n = len(rows)
    long_mask = A["T"] >= args.min_T
    nl = int(long_mask.sum())

    L = []
    say = L.append
    def stat(k, mask=None):
        v = A[k] if mask is None else A[k][mask]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return None
        return v

    say("=" * 86)
    say(f"轨迹级结构测量   n={n} 条 rollout ({nl} 条 T>={args.min_T} 参与时序统计)")
    say("描述性测量, 无正确性标签。dump 覆盖 base/ahaAn8/supn8 多个训练阶段。")
    say("=" * 86)

    # ---------------- 第二组先报: 自相关是整条路的前提 ----------------
    say("")
    say("【第二组 时序结构】—— 自相关是前提: 若 ρ(1)≈0, 任何轨迹级聚合只是在平均噪声")
    say(f"{'量':<12}{'mean':>9}{'p50':>9}{'p10':>9}{'p90':>9}  说明")
    for k, desc in (("ac1_N", "压低通道 lag-1 自相关"), ("ac5_N", "压低通道 lag-5"),
                    ("ac1_u", "净信号 lag-1"), ("ac5_u", "净信号 lag-5")):
        v = stat(k, long_mask)
        if v is None:
            continue
        say(f"{k:<12}{v.mean():>9.3f}{np.median(v):>9.3f}{np.percentile(v,10):>9.3f}"
            f"{np.percentile(v,90):>9.3f}  {desc}")
    a1 = stat("ac1_N", long_mask)
    if a1 is not None:
        frac_pos = float((a1 > 0.1).mean())
        say(f"  -> ρ(1)>0.1 的轨迹占比: {100*frac_pos:.1f}%")
        if a1.mean() < 0.05:
            say("  ** ρ(1)≈0 => 压低信号在时序上无持续性, 轨迹级聚合/EMA 在平滑噪声 **")
        elif a1.mean() > 0.2:
            say("  ** ρ(1) 显著为正 => 存在真实的时序持续结构, 轨迹级聚合有依据 **")
        else:
            say("  -> ρ(1) 弱正: 有一定持续性但不强")

    say("")
    say("趋势斜率 (对相对位置 t/T 回归; >0 = 沿推理积累, <0 = 消解)")
    for k, desc in (("slope_N", "压低通道"), ("slope_u", "净信号")):
        v = stat(k, long_mask)
        if v is None:
            continue
        say(f"  {k:<10} mean={v.mean():>+8.4f}  p50={np.median(v):>+8.4f}  "
            f">0 占比={100*float((v>0).mean()):>5.1f}%   {desc}")

    say("")
    say("两通道的时序关系")
    for k, desc in (("PN_sync", "同轨迹内 P_t 与 N_t 的逐位置相关"),
                    ("peak_pos_P", "P 峰值相对位置 (0=开头,1=结尾)"),
                    ("peak_pos_N", "N 峰值相对位置")):
        v = stat(k, long_mask)
        if v is None:
            continue
        say(f"  {k:<12} mean={v.mean():>7.3f}  p50={np.median(v):>7.3f}   {desc}")

    # ---------------- 第一组 ----------------
    say("")
    say("【第一组 两通道分解】—— corr(P,N) 跨轨迹: 高相关=同一个量, 低相关=存在通道主导的分化")
    P, N = A["P"], A["N"]
    say(f"  P (抬升) mean={P.mean():.4f}  CV={P.std()/max(P.mean(),1e-12):.2f}")
    say(f"  N (压低) mean={N.mean():.4f}  CV={N.std()/max(N.mean(),1e-12):.2f}")
    say(f"  **corr(P,N) 跨轨迹 (Spearman) = {spearman(P,N):+.3f}**")
    r = A["PN_ratio"]; r = r[np.isfinite(r)]
    say(f"  P/N 比值: p10={np.percentile(r,10):.2f}  p50={np.median(r):.2f}  "
        f"p90={np.percentile(r,90):.2f}  (>1 = 抬升主导)")
    say(f"    抬升主导(P/N>1) 的轨迹占比: {100*float((r>1).mean()):.1f}%")
    if abs(spearman(P, N)) > 0.7:
        say("  ** 两通道高度相关 => 轨迹级上它们是同一个'视觉证据强度', 分别加权无增量 **")
    elif abs(spearman(P, N)) < 0.4:
        say("  ** 两通道近独立 => 存在通道主导的轨迹分化, 是 reweight 的可用抓手 **")

    # ---------------- 第三组 ----------------
    say("")
    say("【第三组 集中度】—— 时间轴上负向质量是弥散还是尖峰 (Gini: 0=完全均匀, 1=全在一点)")
    for k, desc in (("gini_N", "压低通道"), ("gini_P", "抬升通道")):
        v = stat(k, long_mask)
        if v is None:
            continue
        say(f"  {k:<10} mean={v.mean():.3f}  p50={np.median(v):.3f}  "
            f"p90={np.percentile(v,90):.3f}   {desc}")

    # ---------------- 第四组 ----------------
    say("")
    say("【第四组 可用性筛选】CV / 与长度相关 / 冗余")
    keys = ["P", "N", "PN_ratio", "u_net", "ac1_N", "slope_N", "PN_sync", "gini_N", "peak_pos_N"]
    say(f"{'量':<12}{'CV':>7}{'corr(·,T)':>11}  判读")
    ok_keys = []
    for k in keys:
        v = A[k]
        m = np.isfinite(v) & long_mask
        if m.sum() < 10:
            continue
        vv = v[m]
        cv = vv.std() / max(abs(vv.mean()), 1e-12)
        rho = spearman(vv, A["T"][m])
        flag = []
        if cv < 0.15:
            flag.append("范围窄")
        if abs(rho) > 0.5:
            flag.append("长度替身")
        if not flag:
            flag.append("可用"); ok_keys.append(k)
        say(f"{k:<12}{cv:>7.2f}{rho:>11.3f}  {', '.join(flag)}")

    if len(ok_keys) >= 2:
        say("")
        say("存活量的两两 Spearman (|ρ|>0.8 => 冗余):")
        say("            " + "".join(f"{k[:9]:>11}" for k in ok_keys))
        for k1 in ok_keys:
            line = f"{k1:<12}"
            for k2 in ok_keys:
                if k1 == k2:
                    line += f"{'-':>11}"
                else:
                    m = np.isfinite(A[k1]) & np.isfinite(A[k2]) & long_mask
                    line += f"{spearman(A[k1][m], A[k2][m]):>11.2f}"
            say(line)

    txt = "\n".join(L)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")
    print(f"\n>>> 报告已写入 {args.out}")


if __name__ == "__main__":
    main()


def signal_position_mismatch(dumps, min_T=20):
    """量化"信号分布 vs loss 权重分布"的错配。

    token-mean 下每个 token 权重相等 -> 前 f 比例的位置获得恰好 f 的 loss 权重。
    但视觉信号 N_t 沿位置递减 (96% 轨迹 slope<0, 峰值在前 10-14%), 所以前 f 位置
    携带的信号占比 S(f) > f。差 S(f)-f 就是错配。

    另报信号质心 E[t/T | 按 N_t 加权] —— token-mean 的质心恒为 ~0.5, 两者之差
    是最直观的单一错配指标。
    """
    import glob
    out = {}
    for dump in dumps:
        mp = os.path.join(dump, "meta.jsonl")
        if not os.path.exists(mp):
            continue
        S10, S20, S50, cen, Ts = [], [], [], [], []
        for line in open(mp):
            r = json.loads(line)
            p = os.path.join(dump, r["npz"])
            if not os.path.exists(p):
                continue
            d = np.load(p)
            lp_p = add_tail(d["lp_pos"].astype(np.float64))
            lp_n = add_tail(d["lp_null"].astype(np.float64))
            u = lp_p - lp_n
            pp = np.exp(lp_p); pp /= pp.sum(-1, keepdims=True)
            N_t = (pp * np.maximum(-u, 0.0)).sum(-1)
            T = len(N_t)
            if T < min_T or N_t.sum() <= 0:
                continue
            c = np.cumsum(N_t) / N_t.sum()
            S10.append(float(c[max(int(0.1 * T) - 1, 0)]))
            S20.append(float(c[max(int(0.2 * T) - 1, 0)]))
            S50.append(float(c[max(int(0.5 * T) - 1, 0)]))
            rel = np.arange(T) / max(T - 1, 1)
            cen.append(float((rel * N_t).sum() / N_t.sum()))
            Ts.append(T)
        if S10:
            out[os.path.basename(dump)] = dict(
                n=len(S10), T=float(np.mean(Ts)),
                S10=float(np.mean(S10)), S20=float(np.mean(S20)),
                S50=float(np.mean(S50)), cen=float(np.mean(cen)))
    return out
