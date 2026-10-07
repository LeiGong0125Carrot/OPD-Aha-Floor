#!/usr/bin/env python
"""Offline checker for the one-step real-GPU PBD smoke run (docs/proposals_2026-10-06/14_P5_pre_run_integration_test_plan.md).
Reads the PBD_SMOKE_DUMP directory written by the production code path (trainer / agent loop / actor dumps) and asserts
§4-§14 invariants; prints the §20 diagnostic block.  用法: python scripts/test_pbd_integration.py --dump DIR --mode keep|replace
"""
import argparse, glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: FAIL.append(name)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); ap.add_argument("--mode", required=True, choices=["keep", "replace"])
    ap.add_argument("--tol", type=float, default=2e-3)
    a = ap.parse_args(); D = a.dump
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    tr_files = sorted(glob.glob(os.path.join(D, "trainer_step*.json")))
    check("trainer dump present", len(tr_files) >= 1, str(tr_files)); 
    if not tr_files: sys.exit(1)
    T = json.load(open(tr_files[0])); rows = T["rows"]
    check("mode matches", T["mode"] == a.mode, T["mode"])
    check("B + N_branch divisible by dp", (T["B"] + len(T["cand"])) % T["dp"] == 0, f"B={T['B']} n_b={len(T['cand'])} dp={T['dp']}")
    check("N_total = B + branch rows", T["n_total"] == T["B"] + len(T["cand"]))
    check("pbd_scale = N_total/N_v0", abs(T["pbd_scale"] - T["n_total"] / T["B"]) < 1e-9)
    check("pbd_scale_b = N_total/N_branch(with loss)", abs(T["pbd_scale_b"] - T["n_total"] / max(T["n_b"], 1)) < 1e-9)
    check("cand indices are increasing (parent order preserved)", all(x < y for x, y in zip(T["cand"], T["cand"][1:])))
    check("merged is_branch = [0]*B + [1]*n_b", T["merged_is_branch"] == [0.0] * T["B"] + [1.0] * len(T["cand"]))
    vis_special = {151652, 151653, 151654, 151655, 151656}   # vision_start/end/pad, image_pad, video_pad (Qwen)
    # ---- A/B/D per row
    for r in rows:
        Tn = r["T"]; k = r["t_star"]; pre = r["prefix_ids"]; pr = r["parent_response_ids"]
        check(f"[row {r['parent_idx']}] T>0 and prefix inside response", Tn > 0 and 0 < k < Tn - 1, f"T={Tn} t*={k}")
        check(f"[row {r['parent_idx']}] t* >= ceil(0.1T)", k >= r["raw_target"], f"{k} >= {r['raw_target']}")
        check(f"[row {r['parent_idx']}] prefix_ids == parent[:t*] (exact, no retokenisation)", pre == pr[:k])
        check(f"[row {r['parent_idx']}] token at t* starts a word", (r["first_token_after"] or "").startswith((" ", "\n")), repr(r["first_token_after"]))
        check(f"[row {r['parent_idx']}] no visual special tokens in parent/branch responses", not (set(pr) & vis_special) and not (set(r["branch_response_ids"]) & vis_special))
        n_teach = T.get("teacher_images_per_row", 2)
        check(f"[row {r['parent_idx']}] student view = 1 image, teacher view = {n_teach} image(s)", r["student_images"] == 1 and r["teacher_images"] == n_teach, f"{r['student_images']}/{r['teacher_images']}")
        br = r["branch_response_ids"]; yT = r["cont_ids"]; n = len(br)
        check(f"[row {r['parent_idx']}] branch = prefix ⊕ y^T (up to truncation)", br[:k] == pr[:k] and br[k:] == yT[: n - k])
        lm = r["branch_loss_mask"]; rm = r["branch_response_mask"]
        check(f"[row {r['parent_idx']}] branch response_mask = 1 on all real tokens", rm[:n] == [1] * n and sum(rm) == n)
        exp_lm = [0.0] * k + [1.0] * (n - k) + [0.0] * (len(lm) - n)
        if T["leak_mask_on"]:
            check(f"[row {r['parent_idx']}] branch loss mask (leak-masked)", lm[:k] == [0.0] * k and sum(lm[n:]) == 0)
        else:
            check(f"[row {r['parent_idx']}] branch loss mask: 0 on prefix, 1 on y^T, 0 on pad (leak tokens kept)", lm == exp_lm)
        check(f"[row {r['parent_idx']}] attention valid = parent prompt + n", r["branch_attn_valid"] == r["parent_attn_valid"] - Tn + n, f"{r['branch_attn_valid']} vs {r['parent_attn_valid']-Tn+n}")
        pm = r["parent_loss_mask"]; prm = r["parent_response_mask"]
        if a.mode == "keep":
            check(f"[row {r['parent_idx']}] keep parent: full response loss", pm == [float(x) for x in prm])
        else:
            check(f"[row {r['parent_idx']}] replace parent: t<t* only", pm == [1.0] * k + [0.0] * (len(pm) - k))
        check(f"[row {r['parent_idx']}] position ids shape (4, L)", r["pos_shape"][0] == 4 and len(r["pos_shape"]) == 2, str(r["pos_shape"]))
    # ---- C agent loop
    al = [json.load(open(f)) for f in glob.glob(os.path.join(D, "agentloop_*.json"))]
    check("agent-loop dumps present (>= branch rows, incl. padding)", len(al) >= len(rows), f"{len(al)} >= {len(rows)}")
    n_gen = T.get("teacher_images_per_row", 2) if T.get("cont_view", "crop_append") == "teacher_prompt" else 2
    for x in al:
        check(f"[gen {x['uid'][:8]}] continuation view = {n_gen} image(s), images before text", x["n_images"] == n_gen and x["content_types"][:n_gen] == ["image"] * n_gen and x["content_types"][-1] == "text", str(x["content_types"]))
        check(f"[gen {x['uid'][:8]}] prompt tail == prefix ids; final = template + prefix", x["prompt_tail_equals_prefix"] and x["final_prompt_len"] == x["prompt_len_before_prefix"] + x["prefix_len"])
        check(f"[gen {x['uid'][:8]}] continuation <= cap", x["cont_len"] <= x["max_cont"])
    # match continuation ids between agent loop and trainer rows
    # two rollouts of one prompt (n=2) often share the first ~10 tokens, and padding duplicates rows: match on
    # (prefix, continuation) rather than prefix alone
    by_pre = {}
    for x in al: by_pre.setdefault(tuple(x["prefix_ids"]), []).append(x)
    for r in rows:
        cands = by_pre.get(tuple(r["prefix_ids"]), [])
        check(f"[row {r['parent_idx']}] trainer y^T == an agent-loop continuation with the same prefix",
              any(r["cont_ids"] == x["cont_ids"][: len(r["cont_ids"])] for x in cands), f"{len(cands)} same-prefix records")
    # ---- E/F/G actor
    ac = [json.load(open(f)) for f in glob.glob(os.path.join(D, "actor_rank*_mb*.json"))]
    check("actor dumps present for every row", len(ac) == T["n_total"], f"{len(ac)} vs {T['n_total']}")
    br_rows = [x for x in ac if x["is_branch"] > 0.5]; v0_rows = [x for x in ac if x["is_branch"] < 0.5]
    check("actor: branch rows count", len(br_rows) == len(T["cand"])); check("actor: V0 rows count", len(v0_rows) == T["B"])
    for x in ac:
        tag = f"[actor r{x['rank']} {'B' if x['is_branch']>0.5 else 'V0'} n={x['response_valid']}]"
        check(f"{tag} student and teacher score the same response ids", x["student_response_ids"] == x["teacher_response_ids"])
        check(f"{tag} student view full-image only (1), teacher view {T.get('teacher_images_per_row', 2)} image(s)", x["student_images"] == 1 and x["teacher_images"] == T.get("teacher_images_per_row", 2), f"{x['student_images']}/{x['teacher_images']}")
        check(f"{tag} no null forward / tensors", (not x["null_present"]) and x["null_forward_ran"] == 0.0 and not x["teacher_null_keys"])
        check(f"{tag} finite logits/loss", all(x["finite"].values()))
        check(f"{tag} scales and lambda", abs(x["pbd_scale"] - T["pbd_scale"]) < 1e-9 and abs(x["pbd_scale_b"] - T["pbd_scale_b"]) < 1e-9 and abs(x["lam"] - T["lam"]) < 1e-12)
    br_ids = {tuple(r["branch_response_ids"]) for r in rows}
    check("actor branch response ids ⊆ trainer branch rows", all(tuple(x["student_response_ids"]) in br_ids for x in br_rows))
    # ---- G/H objective reconstruction: backward total = mean over rows (each micro-batch loss already has 1/GA; ranks average)
    ranks = sorted({x["rank"] for x in ac})
    per_rank = {rk: sum(x["loss"] for x in ac if x["rank"] == rk) for rk in ranks}
    ga = {rk: [x["grad_accum"] for x in ac if x["rank"] == rk] for rk in ranks}
    check("grad_accum == rows per rank", all(all(g == len(ga[rk]) for g in ga[rk]) for rk in ranks), str({rk: (len(ga[rk]), set(ga[rk])) for rk in ranks}))
    backward_total = float(np.mean(list(per_rank.values())))
    lam = T["lam"]
    v0_terms = []
    for x in v0_rows:
        frac = (x["pbd_loss_mask_sum"] / max(x["response_valid"], 1)) if a.mode == "replace" else 1.0
        v0_terms.append(x["raw_jsd_token_mean"] * frac)
    b_terms = [x["raw_jsd_token_mean"] for x in br_rows if x["pbd_loss_mask_sum"] > 0]
    recon = float(np.mean(v0_terms)) + lam * (float(np.mean(b_terms)) if b_terms else 0.0)
    check("backward total == mean_v0 (+frac) + lambda * mean_branch (reconstructed from per-row raw JSD)", abs(backward_total - recon) <= a.tol * max(1.0, abs(recon)), f"{backward_total:.6f} vs {recon:.6f}")
    if a.mode == "replace":
        for x in v0_rows:
            check(f"[actor r{x['rank']} V0] replace: loss tokens == t* region only", x["pbd_loss_mask_sum"] <= x["response_valid"])
    # ---- diagnostic block
    r0 = rows[0]
    print("\n[PBD smoke]")
    print(f"uid: {r0['uid']}\nmode: {a.mode}\nparent_idx: {r0['parent_idx']}\nT: {r0['T']}\nt_star: {r0['t_star']}\nt_star/T: {r0['t_star']/r0['T']:.3f}")
    print(f"\nstudent_view_images: {r0['student_images']}\nbranch_generation_images: {al[0]['n_images'] if al else '?'}\nteacher_scoring_images: {br_rows[0]['teacher_images'] if br_rows else '?'}")
    print(f"\nN_v0: {T['B']}\nN_branch: {len(T['cand'])} (with loss: {T['n_b']}; dropped_for_dp: {T['n_drop']})\nN_total: {T['n_total']}\npbd_scale: {T['pbd_scale']:.4f}\npbd_scale_b: {T['pbd_scale_b']:.4f}\nlambda: {lam}")
    print(f"\nmean_v0_jsd: {float(np.mean([x['raw_jsd_token_mean'] for x in v0_rows])):.5f}\nmean_branch_jsd: {float(np.mean(b_terms)) if b_terms else float('nan'):.5f}\nreconstructed_total: {recon:.6f}\nbackward_total: {backward_total:.6f}\nabs_diff: {abs(backward_total-recon):.2e}")
    print(f"\nparent_text:\n{r0['parent_text'][:600]}\n\nbranch_prefix: {tok.decode(r0['prefix_ids'])!r}\nbranch_continuation:\n{r0['cont_text'][:600]}")
    print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAIL: {FAIL[:6]}")
    sys.exit(1 if FAIL else 0)

if __name__ == "__main__":
    main()
