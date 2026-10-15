#!/usr/bin/env python
"""Write a YAML record of one training arm: config (from the training log) + per-ckpt judge scores on
V* / ZoomBench / HR-Bench 4K / 8K (rule-first and pure-llm) + peaks + which ckpt is kept (ZoomBench best, pure-llm).
用法: python record_run.py --tag TAG --sn SN [--ckpt_dir DIR] [--train_log LOG] [--out run_records/TAG.yaml] [--kept STEP]"""
import argparse, datetime, json, os, re, subprocess, sys
import yaml
EV = "/sfs/weka/scratch/nkw3mr/Vision-OPD/eval"; SETUP = "/sfs/weka/scratch/nkw3mr/Vision-OPD-setup"
REPO = "/scratch/nkw3mr/Vision-OPD/OPD-Aha-sup"
BENCH_N = {"vstar": 191, "zoombench": 845, "hrbench-4k": 800, "hrbench-8k": 800}
KEYS = ["experiment_name", "train_files", "train_batch_size", "ppo_mini_batch_size", "save_freq", "seed", "alpha",
        "distillation_topk", "teacher_regularization", "teacher_update_rate", "teacher_model_source", "teacher_image_key", "counterfactual_null_mode",
        "counterfactual_null_scope", "counterfactual_extrapolation_beta", "counterfactual_hist_adaptive_beta",
        "counterfactual_hist_mode", "counterfactual_hist_kappa", "counterfactual_target_mode", "max_response_length",
        "max_prompt_length", "pbd_enable", "pbd_mode", "pbd_ratio", "pbd_coverage", "pbd_lambda", "pbd_max_cont_len",
        "pbd_leak_mask", "pbd_cont_view"]

def acc(f, n):
    if not os.path.exists(f): return None
    try: r = json.load(open(f))
    except Exception: r = [json.loads(l) for l in open(f)]
    if len(r) != n: return None
    return round(100 * sum(str(x.get("judge", "")).strip().lower().startswith("yes") for x in r) / len(r), 2)

def lit(v):
    import ast
    try: return ast.literal_eval(v)
    except Exception: return v


def parse_log(path):
    cfg = {"launcher_lines": []}
    if not path or not os.path.exists(path): return cfg
    head = open(path, errors="ignore").read(4_000_000)
    for line in head.splitlines()[:6]:
        if line.strip() and "START" not in line: cfg["launcher_lines"].append(line.strip())
    for k in KEYS:
        vals = re.findall(rf"'{k}': ([^,}}\n]+(?:\][^,}}\n]*)?)", head)
        vals = [lit(v.strip()) for v in vals]
        if vals:
            uniq = []
            for v in vals:
                if v not in uniq: uniq.append(v)
            cfg[k] = uniq[0] if len(uniq) == 1 else uniq
    m = re.search(r"'total_training_steps': (\d+)", head); cfg["total_training_steps"] = int(m.group(1)) if m else None
    m = re.search(r"rollout_n=(\d+) lr=([0-9.e-]+)", head)
    if m: cfg["rollout_n"], cfg["actor_lr"] = int(m.group(1)), m.group(2)
    m = re.search(r"START (.+)", head); cfg["train_start"] = m.group(1).strip() if m else None
    tail = open(path, errors="ignore").read()[-20000:]
    m = re.search(r"EXIT=(\d+) (.+)", tail); cfg["train_exit"] = (m.group(1) + " " + m.group(2).strip()) if m else None
    return cfg

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--sn", required=True)
    ap.add_argument("--ckpt_dir"); ap.add_argument("--train_log"); ap.add_argument("--out"); ap.add_argument("--kept", type=int)
    ap.add_argument("--steps", help="space-separated steps (default: from ckpt dir)"); ap.add_argument("--note", default="")
    a = ap.parse_args()
    ck = a.ckpt_dir or f"{REPO}/checkpoints/{a.tag}"
    steps = [int(x) for x in a.steps.split()] if a.steps else sorted(int(d.split("_")[-1]) for d in os.listdir(ck) if d.startswith("global_step_"))
    res, peaks = {}, {}
    for s in steps:
        res[s] = {}
        for b, n in BENCH_N.items():
            res[s][b] = {p: acc(f"{EV}/judge/{p}/{b}/{a.sn}-step{s}_seed42_answer.jsonl", n) for p in ("rule-first", "pure-llm")}
    for b in BENCH_N:
        sc = [(res[s][b]["pure-llm"], s) for s in steps if res[s][b]["pure-llm"] is not None]
        if sc:
            best = max(sc, key=lambda t: (t[0], -t[1])); peaks[b] = {"step": best[1], "pure-llm": best[0], "rule-first": res[best[1]][b]["rule-first"]}
    try: head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception: head = None
    rec = {"tag": a.tag, "short_name": a.sn, "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "repo_head_at_record": head, "ckpt_dir": ck, "note": a.note,
           "config": parse_log(a.train_log or f"{SETUP}/logs/train_{a.tag}.log"),
           "eval_protocol": "OPD-Aha eval/infer.py + gpt-oss-120b judge (judge_qwenlm.py); peaks by pure-llm",
           "results": res, "peaks": peaks,
           "kept_ckpt": {"step": a.kept, "rule": "ZoomBench best (pure-llm), user 10-07"} if a.kept is not None else None}
    out = a.out or f"{SETUP}/run_records/{a.tag}.yaml"; os.makedirs(os.path.dirname(out), exist_ok=True)
    yaml.safe_dump(rec, open(out, "w"), allow_unicode=True, sort_keys=False)
    print(out)

if __name__ == "__main__":
    main()
