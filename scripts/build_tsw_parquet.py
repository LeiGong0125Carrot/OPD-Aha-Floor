#!/usr/bin/env python
"""Freeze the TSW round-1 question sets into a training parquet (teacher_stability/tsw_initial_training_plan.md §4).

Input: the off6k train parquet (6241 rows) and scores.npz from export_tsw_scores.py (row = parquet row index).
Output: a copy of the parquet whose extra_info gains
  tsw_row     parquet row index (identity)
  tsw_p       teacher L28 out-of-fold det4 score (probe)
  tsw_share8  teacher 8-sample majority share (sampling; round 2)
  tsw_low25   1 if the row is among the floor(0.25 N) lowest tsw_p (ties broken by a fixed GT-free random order)
  tsw_rand25  1 for a fixed random floor(0.25 N) subset of all rows (independent of scores; not forced disjoint)
plus <out>.sets.json (row ids of both sets, seeds, sha256 of inputs). All other columns are copied unchanged.
No GT is used: gt/opts in scores.npz are compared to the parquet only as an identity check.

用法: python scripts/build_tsw_parquet.py --parquet data/train.parquet --scores <dir>/scores.npz --out data/train_tsw.parquet
"""
import argparse, hashlib, json, os, re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FRAC = 0.25
TIE_SEED = 20261024      # tie-break order for equal scores (GT-free)
RAND_SEED = 42           # random-set seed, separate stream from the training seed


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def lowest_k(scores, k, seed=TIE_SEED):
    """Indices of the k lowest scores; equal scores ordered by a fixed random permutation."""
    tie = np.random.default_rng(seed).permutation(len(scores))
    return np.lexsort((tie, scores))[:k]


def random_k(n, k, seed=RAND_SEED):
    return np.sort(np.random.default_rng([seed, 25]).choice(n, size=k, replace=False))


def opts_of(question):
    """Option letters listed in the question text ('A. ...' lines)."""
    return "".join(re.findall(r"^([A-F])\.", question, flags=re.M))


def build(parquet, scores, out):
    t = pq.read_table(parquet); n = t.num_rows
    z = np.load(scores, allow_pickle=False)
    row, p, share8 = z["row"].astype(int), z["p"].astype(float), z["share8"].astype(float)
    assert len(row) == n and sorted(row.tolist()) == list(range(n)), f"scores cover {len(row)} rows, parquet has {n}"
    order = np.argsort(row); p, share8, gt, opts = p[order], share8[order], z["gt"][order], z["opts"][order]
    group = z["group"][order] if "group" in z.files else None
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()   # only the ranking is used
    ei = t.column("extra_info").to_pylist()
    for i in range(n):   # identity check: answer letter and option set must match the generation file row by row
        assert ei[i]["answer"] == str(gt[i]), f"row {i}: answer {ei[i]['answer']} != gen gt {gt[i]}"
        assert opts_of(ei[i]["question"]) == str(opts[i]), f"row {i}: options {opts_of(ei[i]['question'])} != gen {opts[i]}"
    if group is not None:   # strong identity: gen.json group = the student image path of the same parquet row
        imgs = t.column("images").to_pylist()
        for i in range(n):
            assert imgs[i][0]["path"] == str(group[i]), f"row {i}: image {imgs[i][0]['path']} != gen group {group[i]}"
    k = int(np.floor(FRAC * n))
    low = lowest_k(p, k); rnd = random_k(n, k)
    low_m = np.zeros(n, int); low_m[low] = 1; rnd_m = np.zeros(n, int); rnd_m[rnd] = 1
    new_ei = [dict(e, tsw_row=i, tsw_p=float(p[i]), tsw_share8=float(share8[i]), tsw_low25=int(low_m[i]), tsw_rand25=int(rnd_m[i]))
              for i, e in enumerate(ei)]
    cols = {c: (pa.array(new_ei) if c == "extra_info" else t.column(c)) for c in t.column_names}
    t2 = pa.table(cols)
    for c in t.column_names:
        if c != "extra_info": assert t2.column(c).equals(t.column(c)), c
    pq.write_table(t2, out)
    info = dict(n=n, k=k, frac=FRAC, tie_seed=TIE_SEED, rand_seed=[RAND_SEED, 25], low25_rows=low.tolist(), rand25_rows=rnd.tolist(),
                overlap=int((low_m & rnd_m).sum()), p_cut=float(np.sort(p)[k - 1]), p_mean_low=float(p[low].mean()),
                p_mean_rest=float(p[low_m == 0].mean()), share8_mean_low=float(share8[low].mean()), share8_mean_rest=float(share8[low_m == 0].mean()),
                sha256=dict(parquet=sha256(parquet), scores=sha256(scores), out=sha256(out)))
    json.dump(info, open(out + ".sets.json", "w"), indent=1)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--scores", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = build(a.parquet, a.scores, a.out)
    print(f"[tsw_parquet] {info['n']} 行 -> {a.out}; low25 {info['k']} 题 (p ≤ {info['p_cut']:.3f}, 均值 {info['p_mean_low']:.3f} vs 其余 "
          f"{info['p_mean_rest']:.3f}; 8 样本多数占比 {info['share8_mean_low']:.3f} vs {info['share8_mean_rest']:.3f}); rand25 {info['k']} 题, "
          f"与 low25 交集 {info['overlap']}", flush=True)


if __name__ == "__main__":
    main()
