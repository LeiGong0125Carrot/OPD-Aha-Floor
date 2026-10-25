#!/usr/bin/env python
"""Freeze the DG offline inputs into a training parquet (OPD-Aha-Floor teacher_stability/disagreement_gate_plan.md v2).

Inputs: the off6k train parquet (6241 rows) and the frozen base teacher's crop-view probe dir (probe_hidden_answer.py
--view teacher: gen.json = 8 samples x 1024 tokens at T=1 per row, readout.npz = full-vocab log_softmax at the option-letter
tokens of the last prompt position). The scoring teacher is frozen, so both are fixed for the whole run.
Output: a copy of the parquet whose extra_info gains
  dg_row       parquet row index (identity)
  dg_opts      option letters of the question ('ABCD')
  dg_pT0       [6] teacher t=0 answer readout over A-F renormalised on the options (0 off-option)
  dg_traj_ok   1 if an offline teacher trajectory is available
  dg_traj      that trajectory's text: uniformly among samples 4..7 whose parsed answer equals the teacher's 8-sample majority
               letter (teacher self-consistency; no GT), seed (42, row); "" when dg_traj_ok = 0
  dg_traj_s    its sample index (-1 when none)
plus <out>.meta.json (counts, sha256 of inputs). All other columns are copied unchanged. GT is used only as an identity check.

用法: python scripts/build_dg_parquet.py --parquet data/train.parquet --teacher eval/probe_hidden/teacher_hidden_1023 --out data/train_dg.parquet
"""
import argparse, hashlib, json, os, re, sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verl.utils import dg as G

SEED = 42


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def opts_of(question):
    return "".join(re.findall(r"^([A-F])\.", question, flags=re.M))


def build(parquet, teacher, out):
    t = pq.read_table(parquet); n = t.num_rows
    gen = {g["row"]: g for g in json.load(open(os.path.join(teacher, "gen.json")))}
    z = np.load(os.path.join(teacher, "readout.npz"))
    lp = {int(r): z["letter_logp"][i] for i, r in enumerate(z["rows"])}
    assert sorted(gen) == list(range(n)) and sorted(lp) == list(range(n)), f"teacher dir must cover all {n} rows"
    ei = t.column("extra_info").to_pylist(); imgs = t.column("images").to_pylist()
    new_ei, n_ok, n_nomaj, src = [], 0, 0, []
    for i, e in enumerate(ei):
        g = gen[i]; opts = opts_of(e["question"])
        assert e["answer"] == g["gt"], f"row {i}: answer {e['answer']} != gen gt {g['gt']}"
        assert opts == "".join(g["opts"]), f"row {i}: options {opts} != gen {g['opts']}"
        assert imgs[i][0]["path"] == g["group"], f"row {i}: image {imgs[i][0]['path']} != gen group {g['group']}"
        assert len(g["letters"]) == len(g["texts"]) == 8, f"row {i}: expected 8 teacher samples"
        pT = G.letter_dist_from_logp(lp[i], opts)
        s = G.choose_teacher_traj(g["letters"], g["texts"], i, seed=SEED)
        n_nomaj += int(G.majority(g["letters"]) is None)
        ok = s is not None; n_ok += int(ok); src.append(-1 if s is None else s)
        new_ei.append(dict(e, dg_row=i, dg_opts=opts, dg_pT0=[float(x) for x in pT], dg_traj_ok=int(ok),
                           dg_traj=(g["texts"][s] if ok else ""), dg_traj_s=(int(s) if ok else -1)))
    cols = {c: (pa.array(new_ei) if c == "extra_info" else t.column(c)) for c in t.column_names}
    t2 = pa.table(cols)
    for c in t.column_names:
        if c != "extra_info": assert t2.column(c).equals(t.column(c)), c
    pq.write_table(t2, out)
    info = dict(n=n, traj_ok=n_ok, no_majority=n_nomaj, seed=SEED, pool=[4, 5, 6, 7],
                sample_hist={str(k): int(v) for k, v in zip(*np.unique(src, return_counts=True))},
                sha256=dict(parquet=sha256(parquet), gen=sha256(os.path.join(teacher, "gen.json")),
                            readout=sha256(os.path.join(teacher, "readout.npz")), out=sha256(out)))
    json.dump(info, open(out + ".meta.json", "w"), indent=1)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--teacher", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = build(a.parquet, a.teacher, a.out)
    print(f"[dg_parquet] {info['n']} 行 -> {a.out}; 有教师轨迹 {info['traj_ok']} 题 ({info['traj_ok'] / info['n']:.3f}), "
          f"教师 8 样本无唯一多数 {info['no_majority']} 题; 样本来源 {info['sample_hist']}", flush=True)


if __name__ == "__main__":
    main()
