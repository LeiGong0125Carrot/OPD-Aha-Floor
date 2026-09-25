#!/usr/bin/env python
"""TGNVMT 门控前提的判别性检验 (纯 CPU, 复用现有 rollout dump, 零 GPU)。

检验对象: `trajectory_gated_negative_visual_mass_transfer.md` 的**核心前提** ——
    "C_t 高 = 学生正走在一条被视觉证据否定的错误推理分支上"

若该前提成立, 则文档 §11 自己提出的轨迹级冲突负担
    B(τ) = (1/T)Σ_t C_t      或      B_max(τ) = max_t C_t
**应当能区分最终答错与答对的 rollout**。若区分不了 (AUROC≈0.5), 那么 C_t 并不指示
"错误分支", Layer-3 的状态门控就失去依据 —— 无论怎么调 λ/τ_u 都不会有机制收益。

这是一个**必要条件检验**: 通过不代表方法有效 (C_t 可能只是与难度相关);
不通过则门控的立论基础不成立。

判据 (预登记):
  AUROC(错>对) ≥ 0.60  → C_t 有判别力, 前提成立, 可进入训练阶段 (需先重标剂量)
  0.55 - 0.60          → 弱信号, 需扩样 (--rows 更多) 才能判
  ≈ 0.50 (0.45-0.55)   → **C_t 不指示错误分支, 门控失去依据 → no-go**
  ≤ 0.45               → 反向 (C_t 高反而更容易答对), 更强的 no-go

用法 (务必在计算节点上跑):
  srun --overlap --jobid=$JID --ntasks=1 --cpus-per-task=2 bash -c \
    'source 00_env.sh; cd $REPO; PYTHONPATH=$REPO $VOPD_PY probing/probe_tgnvmt_gate.py \
       --dumps results/rd_dump results/rd_ahaAn8-step30 results/rd_supn8-step50'
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_rollout_disagree import realized_state  # noqa: E402


def auroc(pos, neg):
    """P(随机取的 pos > 随机取的 neg), 秩公式, 含并列修正。"""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = np.concatenate([pos, neg])
    order = np.argsort(a)
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # 并列取平均秩
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.bincount(inv, weights=ranks)
        ranks = (sums / cnt)[inv]
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return (float("nan"), float("nan"))
    vals = [auroc(rng.choice(pos, len(pos), replace=True),
                  rng.choice(neg, len(neg), replace=True)) for _ in range(n)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def extract_answer(text):
    """pair prompt 要求 "Answer with the option's letter from the given choices."
    取文本中**最后**一个孤立的 A-D 作为最终答案 (与我们评测侧的 first-letter 规则不同 ——
    这里是生成式续写, 结论在末尾)。返回 None 表示无法判定, 计入 unknown 而非猜测。"""
    m = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text.upper())
    return m[-1] if m else None


def run_one(dump, tk, beta, tau_u, lam):
    metas = [json.loads(l) for l in open(os.path.join(dump, "meta.jsonl"))]
    feats = {k: {"w": [], "r": []} for k in ("B_ema", "B_max", "Bmax_ema", "frac_hi")}
    unknown = 0
    for r in metas:
        d = np.load(os.path.join(dump, r["npz"]))
        if "y" not in d:
            continue
        txt = tk.decode(d["y"].tolist(), skip_special_tokens=True)
        pred = extract_answer(txt)
        gt = str(r.get("gt", "")).strip().upper()[:1]
        if pred is None or not gt:
            unknown += 1
            continue
        side = "r" if pred == gt else "w"
        s = realized_state(d, beta, tau_u, lam)
        feats["B_ema"][side].append(float(s["C_ema"].mean()))
        feats["B_max"][side].append(float(s["C_max"].mean()))
        feats["Bmax_ema"][side].append(float(s["C_ema"].max()))
        feats["frac_hi"][side].append(float((s["delta_ema"] > 0.01).mean()))
    return feats, unknown, len(metas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="仅用于 tokenizer (CPU)")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--tau-u", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--out", default="results/tgnvmt_gate_report.txt")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.model)

    L = []
    say = L.append
    say("=" * 78)
    say("TGNVMT 门控前提的判别性检验: B(τ) 能否区分答错/答对的 rollout?")
    say(f"τ_u={args.tau_u} λ={args.lam} β={args.beta}")
    say("判据: AUROC≥0.60 前提成立 | 0.55-0.60 需扩样 | ≈0.50 门控失去依据(no-go)")
    say("=" * 78)

    verdicts = []
    for dump in args.dumps:
        if not os.path.exists(os.path.join(dump, "meta.jsonl")):
            say(f"\n[跳过] {dump}: 无 meta.jsonl")
            continue
        feats, unknown, total = run_one(dump, tk, args.beta, args.tau_u, args.lam)
        nw, nr = len(feats["B_ema"]["w"]), len(feats["B_ema"]["r"])
        say("")
        say(f"--- {dump}  (答错 {nw} / 答对 {nr} / 无法判定 {unknown}, 共 {total}) ---")
        if nw < 2 or nr < 2:
            say("  样本不足 (某一侧 <2), 跳过判读")
            continue
        for key, lbl in (("B_ema", "B(τ)=mean C_t (EMA)"),
                         ("B_max", "B(τ)=mean C_t (max变体)"),
                         ("Bmax_ema", "B_max(τ)=max_t C_t (EMA)"),
                         ("frac_hi", "有效干预位置占比")):
            w, r = feats[key]["w"], feats[key]["r"]
            a = auroc(w, r)
            lo, hi = boot_ci(w, r)
            say(f"  {lbl:<26} 答错 {np.mean(w):.4f} vs 答对 {np.mean(r):.4f}   "
                f"AUROC={a:.3f} [{lo:.3f}, {hi:.3f}]")
            if key == "B_ema":
                verdicts.append((dump, a, lo, hi, nw, nr))

    say("")
    say("=" * 78)
    say("判读 (以 B(τ)=mean C_t EMA 为主指标):")
    if not verdicts:
        say("  无可判读的 dump")
    else:
        for dump, a, lo, hi, nw, nr in verdicts:
            tag = os.path.basename(dump)
            if a >= 0.60:
                v = "前提成立 (C_t 有判别力)"
            elif a >= 0.55:
                v = "弱信号, 需扩样"
            elif a >= 0.45:
                v = "**C_t 不指示错误分支 → 门控失去依据 (no-go)**"
            else:
                v = "**反向 (C_t 高反而更易答对) → 更强的 no-go**"
            sig = "CI 不含 0.5" if (lo > 0.5 or hi < 0.5) else "CI 含 0.5 (与随机不可区分)"
            say(f"  {tag:<24} AUROC={a:.3f}  n={nw}错/{nr}对  {sig}")
            say(f"      → {v}")
        am = float(np.mean([v[1] for v in verdicts]))
        say("")
        say(f"  跨 dump 平均 AUROC = {am:.3f}")
        say("  注: 这是**必要条件**检验。通过不代表方法有效 (C_t 可能只与题目难度相关);")
        say("      不通过则 Layer-3 状态门控的立论基础不成立, 调 λ/τ_u 也救不回来。")

    txt = "\n".join(L)
    print(txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt + "\n")
    print(f"\n>>> 报告已写入 {args.out}")


if __name__ == "__main__":
    main()
