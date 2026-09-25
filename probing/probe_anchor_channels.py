#!/usr/bin/env python
"""两通道构成 × 语义锚点 —— 对前一版测量的两处修正 (纯 CPU, 零 GPU)。

修正 1: 用**通道构成**而非通道绝对值。
  上一版用 P=Σp⁺[u]₊ 和 N=Σp⁺[-u]₊ 的绝对值, 并从 corr(P,N)=0.784 判"两通道不可分离"。
  但 P − N = Σ_v p⁺·u = KL(p⁺‖p⁰) ≥ 0 是**恒等式**, 所以 P≥N 恒成立、且两者被总强度
  绑定 —— 那个论证是循环的。这里改用去掉总强度后的构成量:
      frac_N = N / (P + N)      压低通道占信号总质量的比例
      PN     = P / N            (等价刻画, 便于与 §手算例的 2.4 对照)

修正 2: 用**语义锚点**而非几何位置 t/T。
  aha-opd 论文 (Fig.4) 的位置结论是"reflection **之前**压制最强";我们自己的 qcontrast
  探针测 GT 对象名"提及前 1.97 vs 后 0.59 (3.3×)" —— 两者都是**语义锚定**。
  上一版按 t/T 对齐, 把不同轨迹的语义事件平均掉了, 只得到 1.3× 的富集。
  本探针用三个锚点:
      A1 answer  : 答案声明位 ("final answer"/"correct answer"/末尾孤立选项字母)
      A2 optword : **GT 选项文本首次提及位** (= qcontrast 的"GT 对象提及"等价物)
      A3 reflect : 反思词首现位 (数据中约 23% 轨迹有)

对每个锚点报: 锚点前 vs 后的 N / P / frac_N, 以及效应比 —— 直接与 t/T 的 1.3× 对照。

用法: PYTHONPATH=$REPO python probing/probe_anchor_channels.py --dumps results/rd_* ...
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_rollout_disagree import add_tail  # noqa: E402

PARQUET = "/sfs/weka/scratch/nkw3mr/Vision-OPD/data/TreeVGR-RL-37K/train_6karmA_pair.parquet"
REFL = re.compile(r"(wait|actually|hmm|alternatively|let me|looking again|re-?exam|"
                  r"on second thought|re-?check|instead|correction|reconsider)", re.I)
ANSWER = re.compile(r"(final answer|correct answer|the answer is|answer:)", re.I)


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


def parse_options(prompt_text):
    """从 prompt 解析 {letter: option_text}, 例: 'A. black' -> {'A': 'black'}"""
    out = {}
    for m in re.finditer(r"(?m)^\s*([A-D])[.)]\s*(.+?)\s*$", prompt_text):
        out[m.group(1)] = m.group(2).strip()
    return out


def find_anchors(y, tk, gt_option_text):
    """增量解码, 返回三个锚点的 token index (找不到则 None)。"""
    s = ""
    a_ans = a_opt = a_ref = None
    gt_low = (gt_option_text or "").lower()
    for i, t in enumerate(y):
        s += tk.decode([int(t)])
        low = s.lower()
        if a_ans is None and ANSWER.search(low):
            a_ans = i
        if a_opt is None and gt_low and len(gt_low) >= 3 and gt_low in low:
            a_opt = i
        if a_ref is None and REFL.search(low):
            a_ref = i
    return {"answer": a_ans, "optword": a_opt, "reflect": a_ref}


def channels(d):
    """逐位置两通道质量 (p⁺ 加权)。"""
    lp_p = add_tail(d["lp_pos"].astype(np.float64))
    lp_n = add_tail(d["lp_null"].astype(np.float64))
    u = lp_p - lp_n
    pp = np.exp(lp_p); pp /= pp.sum(-1, keepdims=True)
    P_t = (pp * np.maximum(u, 0.0)).sum(-1)
    N_t = (pp * np.maximum(-u, 0.0)).sum(-1)
    return P_t, N_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--win", type=int, default=15, help="锚点前/后各取多少 token 作窗口")
    ap.add_argument("--min-T", type=int, default=40)
    ap.add_argument("--out", default="results/anchor_channels_report.txt")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    df = pd.read_parquet(PARQUET)

    per_traj = []          # 轨迹级构成量
    anchor_acc = {k: {"pre_N": [], "post_N": [], "pre_P": [], "post_P": [],
                      "pre_fN": [], "post_fN": [], "pos": []}
                  for k in ("answer", "optword", "reflect")}
    for dump in args.dumps:
        mp = os.path.join(dump, "meta.jsonl")
        if not os.path.exists(mp):
            continue
        for line in open(mp):
            r = json.loads(line)
            p = os.path.join(dump, r["npz"])
            if not os.path.exists(p):
                continue
            d = np.load(p)
            if "y" not in d:
                continue
            P_t, N_t = channels(d)
            T = len(N_t)
            if T < args.min_T:
                continue
            Ps, Ns = float(P_t.mean()), float(N_t.mean())
            per_traj.append({"T": float(T), "P": Ps, "N": Ns,
                             "frac_N": Ns / max(Ps + Ns, 1e-12),
                             "PN": Ps / max(Ns, 1e-12)})
            # 语义锚点
            gt = str(r.get("gt", "")).strip().upper()[:1]
            opts = parse_options(df.iloc[int(r["row"])]["prompt"][0]["content"])
            anc = find_anchors(d["y"], tk, opts.get(gt, ""))
            w = args.win
            for k, idx in anc.items():
                if idx is None or idx < w or idx + w >= T:
                    continue
                pre_P, pre_N = P_t[idx - w:idx].mean(), N_t[idx - w:idx].mean()
                po_P, po_N = P_t[idx + 1:idx + 1 + w].mean(), N_t[idx + 1:idx + 1 + w].mean()
                a = anchor_acc[k]
                a["pre_N"].append(pre_N); a["post_N"].append(po_N)
                a["pre_P"].append(pre_P); a["post_P"].append(po_P)
                a["pre_fN"].append(pre_N / max(pre_P + pre_N, 1e-12))
                a["post_fN"].append(po_N / max(po_P + po_N, 1e-12))
                a["pos"].append(idx / max(T - 1, 1))

    if not per_traj:
        print("无数据"); return
    A = {k: np.array([t[k] for t in per_traj], float) for k in per_traj[0]}
    n = len(per_traj)

    L = []
    say = L.append
    say("=" * 84)
    say(f"两通道构成 × 语义锚点   n={n} 条 rollout (T>={args.min_T}), 窗口 ±{args.win} token")
    say("=" * 84)

    say("")
    say("【修正 1: 通道构成 (去掉总强度)】")
    say(f"  提醒: P−N = KL(p⁺‖p⁰) ≥ 0 是恒等式, 故 P≥N 恒成立 —— 上一版的'100% 抬升主导'")
    say(f"        是平凡结论。以下用构成量。")
    for k, desc in (("frac_N", "压低占信号总质量的比例 N/(P+N)"), ("PN", "P/N 比值")):
        v = A[k]
        say(f"  {k:<8} mean={v.mean():.3f}  p10={np.percentile(v,10):.3f}  "
            f"p50={np.median(v):.3f}  p90={np.percentile(v,90):.3f}  "
            f"CV={v.std()/max(abs(v.mean()),1e-12):.2f}   {desc}")
    say(f"  frac_N 的动态范围 p90/p10 = {np.percentile(A['frac_N'],90)/max(np.percentile(A['frac_N'],10),1e-12):.2f}x")
    say(f"  corr(frac_N, T) = {spearman(A['frac_N'], A['T']):+.3f}   "
        f"corr(P+N, T) = {spearman(A['P']+A['N'], A['T']):+.3f}")
    say(f"  **corr(frac_N, 总强度 P+N) = {spearman(A['frac_N'], A['P']+A['N']):+.3f}** "
        f"(接近 0 => 构成与强度确实是两个独立维度)")

    say("")
    say("【修正 2: 语义锚点 前 vs 后】(论文 Fig.4: reflection 之前压制最强)")
    say(f"{'锚点':<10}{'n':>5}{'锚点位置':>9}{'N前':>9}{'N后':>9}{'比':>7}"
        f"{'fN前':>8}{'fN后':>8}{'比':>7}")
    for k, desc in (("answer", "答案声明位"), ("optword", "GT选项文本首现"), ("reflect", "反思词首现")):
        a = anchor_acc[k]
        if len(a["pre_N"]) < 5:
            say(f"{k:<10}{len(a['pre_N']):>5}  样本不足")
            continue
        pN, qN = np.mean(a["pre_N"]), np.mean(a["post_N"])
        pf, qf = np.mean(a["pre_fN"]), np.mean(a["post_fN"])
        say(f"{k:<10}{len(a['pre_N']):>5}{np.mean(a['pos']):>9.3f}{pN:>9.4f}{qN:>9.4f}"
            f"{pN/max(qN,1e-12):>7.2f}{pf:>8.3f}{qf:>8.3f}{pf/max(qf,1e-12):>7.2f}   {desc}")

    say("")
    say("对照: 上一版按几何位置 t/T 的富集只有 1.27-1.34x (前 10-20% 位置)")
    say("      qcontrast 旧探针 (GT对象提及前/后) 测得 |u| 1.97 vs 0.59 = 3.34x")
    say("      若语义锚点的比值显著 >1.34, 说明结构确实是语义锚定而非几何的。")

    txt = "\n".join(L)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")
    print(f"\n>>> 报告已写入 {args.out}")


if __name__ == "__main__":
    main()
