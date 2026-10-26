#!/usr/bin/env python
"""Which response segment should OPRD-style hidden distillation supervise in VLM OPD? (user 10-26; OPRD arXiv 2606.06021
Fig. 8 analogue). Pure analysis of probe_view_hidden.py tokens.npz, no model forward, no GT.

Setting: off6k, base Qwen3.5-4B, student view S (red-box full image + hint) vs training-teacher view C (crop, same text),
teacher-forced on the student's own T=1 trajectories; M = a same-shaped crop of ANOTHER question (view-type control);
S2 = an OPD-trained checkpoint under the student view (see meta.json 'trained').
Per trajectory of length T (positions t = 0..T-1; t=0 is the last prompt token = pre-generation state):
  segments  all = every t;  first25 = t < ceil(0.25 T);  last50 = t >= floor(0.5 T);  (and t=0 alone, all_t>=1)
  per layer: cos_SC, nd_SC (normalised squared distance, probe_view_hidden.ndist), content share (nd_SM - nd_SC)/nd_SM
  (part of the S-C distance that the RIGHT crop removes relative to a same-shaped wrong crop; the rest is "a crop-shaped
  image" style; an operational definition relative to a wrong-crop baseline, not a true decomposition), output JSD S||C, and
  distances of the trained checkpoint S2 to C, to its own crop view C2, and to base S (weight-drift baseline).
Aggregation: mean over tokens inside each trajectory's segment, then mean over trajectories (each trajectory counts once,
so long responses do not dominate); token-weighted means are printed for reference. Paper-style first-k / last-k curves
(absolute k) at one layer. Trajectories truncated at --max_len (512 in the probe) are excluded (their T is unknown).
用法: python scripts/probe_oprd_segments.py --viewhid eval/probe_hidden/viewhid_1022 --out <dir>
"""
import argparse, json, math, os
import numpy as np

SEGS = ("all", "first25", "first25_t>=1", "last50", "t0", "all_t>=1")


def seg_masks(t, T):
    """Per-token boolean masks for each segment given position t and its trajectory length T."""
    f25 = t < np.ceil(0.25 * T)
    return {"all": np.ones_like(t, bool), "first25": f25, "first25_t>=1": f25 & (t >= 1), "last50": t >= np.floor(0.5 * T),
            "t0": t == 0, "all_t>=1": t >= 1}


def traj_mean(v, key, mask):
    """Mean over tokens per trajectory key (masked), then mean over trajectories that have >=1 token in the mask.
    v: [N] or [N, L]. Returns (traj-weighted mean, token-weighted mean, n_traj)."""
    v, key = v[mask], key[mask]
    if len(v) == 0:
        return np.nan, np.nan, 0
    u, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv).astype(float)
    if v.ndim == 1:
        s = np.bincount(inv, weights=v); return float(np.mean(s / w)), float(v.mean()), len(u)
    s = np.stack([np.bincount(inv, weights=v[:, j]) for j in range(v.shape[1])], 1)
    return (s / w[:, None]).mean(0), v.mean(0), len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewhid", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--curve_layer", type=int, default=28)
    a = ap.parse_args(); say = lambda s="": print(s, flush=True)
    z = dict(np.load(os.path.join(a.viewhid, "tokens.npz"))); meta = json.load(open(os.path.join(a.viewhid, "meta.json")))
    L = meta["layers"]
    key = z["row"] * 1000 + z["s"]
    u, inv, T = np.unique(key, return_inverse=True, return_counts=True)
    Ttok = T[inv]
    keep = Ttok < a.max_len                     # truncated trajectories: true length unknown
    say(f"== {len(u)} 条轨迹 ({(T >= a.max_len).sum()} 条截断于 {a.max_len} 被排除), 长度中位 {int(np.median(T[T < a.max_len]))}, "
        f"均值 {T[T < a.max_len].mean():.1f}; trained = {meta.get('trained')}")
    t = z["t"]; M = {k: m & keep for k, m in seg_masks(t, Ttok).items()}
    com = z["commit"]; okc = keep & (t == 0) & (com >= 0)
    rel = com[okc] / Ttok[okc]
    say(f"   答案承诺 token 相对位置: 中位 {np.median(rel):.2f}, <25% 占 {np.mean(rel < 0.25):.2f}, ≥50% 占 {np.mean(rel >= 0.5):.2f} "
        f"(n={okc.sum()}, 可定位承诺的轨迹)")
    res = {"layers": L, "n_traj": int(keep[t == 0].sum()), "seg": {}}
    eps = 1e-12
    for nm in ("cos_SC", "nd_SC", "jsd_SC"):
        say(f"\n== {nm}  (轨迹加权; 行 = 段, 列 = 层)")
        say("   " + f"{'':10s}" + "".join(f"L{l:<7d}" for l in L))
        for sg in SEGS:
            m = M[sg]
            v = z[nm]
            tm, kw, n = traj_mean(v, key, m)
            res["seg"].setdefault(sg, {})[nm] = dict(traj=np.atleast_1d(tm).tolist(), tok=np.atleast_1d(kw).tolist(), n=n)
            if np.ndim(tm) == 0:
                say(f"   {sg:10s} {tm:.4f} (token 加权 {kw:.4f}, n={n})")
            else:
                say(f"   {sg:10s}" + "".join(f"{x:<8.3f}" for x in tm) + f" n={n}")
    # content share on the ratio of means (robust to tiny per-token nd_SM)
    say("\n== content (均值之比: 1 - mean nd_SC / mean nd_SM, 轨迹加权)")
    for sg in SEGS:
        a1, _, _ = traj_mean(z["nd_SC"], key, M[sg]); a2, _, _ = traj_mean(z["nd_SM"], key, M[sg])
        r = 1 - np.asarray(a1) / np.maximum(np.asarray(a2), eps); res["seg"][sg]["content_ratio"] = r.tolist()
        say(f"   {sg:10s}" + "".join(f"{x:<8.3f}" for x in r))
    # by response style: short (direct answer, T <= 16) vs long (reasoning first), ratio-of-means metrics only
    say("\n== 按回答长度分组 (短 T<=16 = 直接作答 / 长 T>16 = 先推理); 每格: nd_SC 轨迹加权/token 加权 | content 比 (1 - nd_SC/nd_SM, 均值之比)")
    res["by_len"] = {}
    for gname, gm in (("short", Ttok <= 16), ("long", Ttok > 16)):
        say(f"   -- {gname}: {int((gm & keep & (t == 0)).sum())} 条轨迹; 层 " + " ".join(f"L{l}" for l in L))
        for sg in SEGS:
            m = M[sg] & gm
            sc, sck, n = traj_mean(z["nd_SC"], key, m); sm, _, _ = traj_mean(z["nd_SM"], key, m)
            s2, _, _ = traj_mean(z["nd_S2C"], key, m); s22, _, _ = traj_mean(z["nd_S2C2"], key, m); ss2, _, _ = traj_mean(z["nd_SS2"], key, m)
            j, _, _ = traj_mean(z["jsd_SC"], key, m)
            sc, sck, sm, s2, s22, ss2 = (np.asarray(x) for x in (sc, sck, sm, s2, s22, ss2))
            con = 1 - sc / np.maximum(sm, eps)
            res["by_len"].setdefault(gname, {})[sg] = dict(nd_SC=sc.tolist(), nd_SC_tok=sck.tolist(), content=con.tolist(),
                                                          nd_S2C=s2.tolist(), nd_S2C2=s22.tolist(), nd_SS2=ss2.tolist(), jsd=float(j), n=n)
            say(f"   {sg:13s} n={n:4d} JSD {j:.3f} | " + "  ".join(f"{a1:.3f}/{a4:.3f}|{a2:+.2f}" for a1, a4, a2 in zip(sc, sck, con)))
        say(f"   -- {gname}: 训练后模型 (S2, {meta.get('trained', '?').split('/')[-1]}) 的距离, 均值之比: S2C/SC | S2C2/SC | SS2/SC (权重漂移基线)")
        for sg in ("first25_t>=1", "last50", "t0"):
            d = res["by_len"][gname][sg]; sc = np.asarray(d["nd_SC"])
            say(f"   {sg:13s} " + "  ".join(f"{a:.2f}|{b:.2f}|{c:.2f}" for a, b, c in zip(np.asarray(d["nd_S2C"]) / sc, np.asarray(d["nd_S2C2"]) / sc, np.asarray(d["nd_SS2"]) / sc)))
    # paper-style first-k / last-k curves (absolute k) at one layer
    res["curve"] = {}
    for lay in sorted({a.curve_layer, L[-1]}):
        li = L.index(lay)
        for start in (0, 1):
            say(f"\n== OPRD Fig.8 式曲线 (L{lay}, cos_SC, 轨迹加权; 只用 T >= 2k + {start} 的轨迹, 两段不重叠; first-k 从 t={start} 起)")
            cur = {}
            for k in (1, 2, 4, 8, 16, 32, 64, 128):
                ok = keep & (Ttok >= 2 * k + start)
                f = ok & (t >= start) & (t < start + k); l = ok & (t >= Ttok - k)
                cf, _, nf = traj_mean(z["cos_SC"][:, li], key, f); cl, _, nl = traj_mean(z["cos_SC"][:, li], key, l)
                cur[str(k)] = (cf, cl, nf); say(f"   k={k:<4d} first {cf:.4f}  last {cl:.4f}  (n={nf})")
            res["curve"][f"L{lay}_t{start}"] = cur
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    say(f"[saved] {a.out}/summary.json")


if __name__ == "__main__":
    main()
