#!/usr/bin/env python
"""Layer-prior probe (plan v0.2: docs/negative_history/ir_layer_prior_probe_plan_2026-10-05.md).

Offline, no training. For N training questions: base model greedy response on the full image, then
four teacher-forced scorings of that SAME response with the same frozen weights:
  student view  [full image]            -> top-K support K_t
  real view     [full image, crop]      -> anchor z+, all block boundaries h^0..h^L (read-only hooks)
  null view     [full image, mean-colour block]  (null_scope=last, as in A)  -> p^0
  text view     question only           -> p_text (diagnostic)
and evaluates candidate internal priors (final-D per layer, own-D per layer, global block weights)
against Q_A on fit / selection / held-out splits grouped by original image.

Usage (GPU):
  python scripts/probe_layer_prior.py --out eval/probe_layer_prior/<run_id> [--limit 8]   # smoke
  python scripts/probe_layer_prior.py --out eval/probe_layer_prior/<run_id>               # full 400
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils.layer_prior_probe import (  # noqa: E402
    BoundaryCollector, RegressionAccumulator, aha_target, centered, coarsen_logits, corr_pieces,
    final_scale_vec, find_answer_token_positions, omega_from_log_p_plus, own_scale_vec, paired_bootstrap,
    pooled_corr, readout_logits, tv_from_log, weighted_stats,
)
from verl.utils.teacher_residual import build_internal_targets  # noqa: E402

IMAGE_TAG = re.compile(r"(<image>)")


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------
def load_rows(parquet: str, n: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(parquet)
    rows = t.to_pylist()
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(rows))[:n]
    out = []
    for i in sorted(idx.tolist()):
        r = rows[i]
        r["_row"] = i
        out.append(r)
    return out


def group_key(row: dict) -> str:
    return row["images"][0]["path"]


def assign_splits(rows: list[dict], sizes=(200, 100, 100), seed: int = 42) -> dict[str, list[dict]]:
    """Group by original image; groups are shuffled (seeded) and filled fit -> selection -> held-out,
    keeping every group intact. Reports actual counts (may deviate from the target sizes)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    keys = sorted(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)
    names = ["fit", "selection", "heldout"]
    out = {k: [] for k in names}
    si = 0
    for k in keys:
        while si < 2 and len(out[names[si]]) >= sizes[si]:
            si += 1
        out[names[si]].extend(groups[k])
    for name in names:
        for r in out[name]:
            r["_split"] = name
    return out


# ------------------------------------------------------------------------------------------
# views
# ------------------------------------------------------------------------------------------
def load_rgb(path: str):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def mean_colour_last(images: list):
    """null_scope=last: keep every image but the last, replace the last by its mean colour (A's null)."""
    from PIL import Image
    out = list(images[:-1])
    last = images[-1]
    mean_rgb = np.asarray(last, dtype=np.float32).mean(axis=(0, 1))
    fill = tuple(np.rint(mean_rgb).clip(0, 255).astype(np.uint8).tolist())
    out.append(Image.new("RGB", last.size, fill))
    return out


def messages_from_template(content: str, images: list) -> list[dict]:
    """Replicates RayPPOTrainer._build_teacher_messages_from_template for a single user turn."""
    parts = [s for s in IMAGE_TAG.split(content) if s != ""]
    cl, k = [], 0
    for s in parts:
        if s == "<image>":
            cl.append({"type": "image", "image": images[k]}); k += 1
        else:
            cl.append({"type": "text", "text": s})
    if k != len(images):
        raise ValueError(f"image placeholder count {k} != images {len(images)}")
    return [{"role": "user", "content": cl}]


def text_only_messages(content: str) -> tuple[list[dict], str]:
    txt = "".join(s for s in IMAGE_TAG.split(content) if s != "<image>").lstrip("\n")
    return [{"role": "user", "content": [{"type": "text", "text": txt}]}], txt


class Views:
    def __init__(self, processor, max_prompt_len: int, image_patch_size: int):
        self.p = processor
        self.max_prompt_len = max_prompt_len
        self.image_patch_size = image_patch_size

    def encode(self, messages: list[dict], images: list | None) -> dict:
        raw = self.p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = dict(self.p(text=[raw], images=images or None, return_tensors="pt", truncation=True, max_length=self.max_prompt_len))
        # mm_token_type_ids (1 = image token) is required by the HF Qwen3.5 forward whenever image_grid_thw is given;
        # response tokens are appended as 0 (text) in score().
        if enc["input_ids"].shape[1] >= self.max_prompt_len:
            raise RuntimeError(f"prompt hit max_prompt_len={self.max_prompt_len} (truncation would corrupt the view)")
        enc["_raw"] = raw
        return enc

    def student(self, row: dict) -> dict:
        from verl.utils.dataset.vision_utils import process_image
        imgs = [process_image(dict(d), image_patch_size=self.image_patch_size) for d in row["images"]]
        return self.encode(messages_from_template(row["prompt"][0]["content"], imgs), imgs)

    def real(self, row: dict) -> dict:
        imgs = [load_rgb(d["path"]) for d in row["bbox_images"]]
        return self.encode(messages_from_template(row["teacher_prompt"][0]["content"], imgs), imgs)

    def null(self, row: dict) -> dict:
        imgs = mean_colour_last([load_rgb(d["path"]) for d in row["bbox_images"]])
        return self.encode(messages_from_template(row["teacher_prompt"][0]["content"], imgs), imgs)

    def text(self, row: dict) -> tuple[dict, str]:
        msgs, txt = text_only_messages(row["prompt"][0]["content"])
        return self.encode(msgs, None), txt


# ------------------------------------------------------------------------------------------
# model calls
# ------------------------------------------------------------------------------------------
@torch.no_grad()
def generate(model, enc: dict, max_new_tokens: int, eos_ids: list[int], pad_id: int) -> tuple[list[int], bool]:
    inp = {k: v.to(model.device) for k, v in enc.items() if k != "_raw"}
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos_ids, pad_token_id=pad_id)
    gen = out[0, inp["input_ids"].shape[1]:].tolist()
    cut, truncated = len(gen), True
    for i, t in enumerate(gen):
        if t in eos_ids:
            cut, truncated = i + 1, False
            break
    return gen[:cut], truncated


@torch.no_grad()
def score(model, enc: dict, response_ids: list[int], tau: float, collector: BoundaryCollector | None = None) -> torch.Tensor:
    """Teacher-force `response_ids` after the encoded prompt; return logits / tau at the T positions that
    predict y_0..y_{T-1} (P-1 .. P+T-2) as fp32 [T, V]. With a collector, h^l are captured at the same positions."""
    dev = model.device
    P = enc["input_ids"].shape[1]
    T = len(response_ids)
    resp = torch.tensor(response_ids, dtype=enc["input_ids"].dtype).unsqueeze(0)
    inp = {k: v.to(dev) for k, v in enc.items() if k != "_raw"}
    inp["input_ids"] = torch.cat([inp["input_ids"], resp.to(dev)], dim=1)
    inp["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(resp).to(dev)], dim=1)
    if "mm_token_type_ids" in inp:
        inp["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(resp).to(dev)], dim=1)
    S = P + T
    if inp["input_ids"].shape[0] != 1:
        raise RuntimeError("score(): single unpadded sequence expected (batch dim 1)")
    if collector is not None:
        mask = torch.zeros(S, dtype=torch.bool)
        mask[P - 1 : P - 1 + T] = True
        collector.reset()
        collector.set_mask(mask)
        with collector:
            out = model(**inp, logits_to_keep=T + 1, use_cache=False)
        collector.check_single_forward()
    else:
        out = model(**inp, logits_to_keep=T + 1, use_cache=False)
    if tuple(out.logits.shape[:2]) != (1, T + 1):
        raise RuntimeError(f"score(): logits_to_keep semantics changed? expected [1,{T+1},V], got {tuple(out.logits.shape)}")
    logits = out.logits[0, :T].float()           # kept = last T+1 positions = S-T-1 .. S-1 ; first T of them = P-1 .. P+T-2
    return logits / tau


# ------------------------------------------------------------------------------------------
# per-question evaluation
# ------------------------------------------------------------------------------------------
class Store:
    """Flat per-position arrays (CPU) + per-question records. Column rows are committed per question
    (all columns or none) so that a failure mid-question can never misalign the arrays."""

    def __init__(self, L: int):
        self.L = L
        self.cols: dict[str, list] = defaultdict(list)
        self.questions: list[dict] = []
        self._pending: dict[str, np.ndarray] = {}

    def add(self, name: str, x):
        self._pending[name] = np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x)

    def commit(self) -> None:
        n = {v.shape[0] for v in self._pending.values()}
        if len(n) != 1:
            raise RuntimeError(f"Store.commit: inconsistent row counts {n}")
        if self.cols and set(self._pending) != set(self.cols):
            raise RuntimeError(f"Store.commit: column set changed: {sorted(set(self._pending) ^ set(self.cols))}")
        for k, v in self._pending.items():
            self.cols[k].append(v)
        self._pending = {}

    def discard(self) -> None:
        self._pending = {}

    def finalize(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v, axis=0) for k, v in self.cols.items()}


def evaluate_question(model, views: Views, row: dict, args, tok, eos_ids, pad_id, coll: BoundaryCollector,
                      store: Store, reg: RegressionAccumulator, w_global: torch.Tensor | None, qid: int, split: str):
    t0 = time.time()
    dev = model.device
    K, beta, tau = args.topk, args.beta, args.tau
    L = coll.L
    rec = {"qid": qid, "row": row["_row"], "split": split, "group": group_key(row), "gt": row["reward_model"]["ground_truth"]}

    # ---- generation on the student view
    enc_s = views.student(row)
    resp, truncated = generate(model, enc_s, args.max_new_tokens, eos_ids, pad_id)
    T = len(resp)
    rec.update(T=T, truncated=truncated, response=tok.decode(resp, skip_special_tokens=False), prompt_len_student=int(enc_s["input_ids"].shape[1]))
    if qid == 0:
        rec["raw_prompts"] = {"student": enc_s["_raw"]}
    if T == 0:
        rec["skipped"] = "empty response"; store.questions.append(rec); return
    ans = find_answer_token_positions(resp, lambda k: tok.decode(resp[:k], skip_special_tokens=False))
    rec["answer"] = ans
    t_gen = time.time()

    # ---- student scoring -> support
    z_s = score(model, enc_s, resp, tau)
    topk = z_s.topk(K, dim=-1).indices                                   # [T, K]
    del z_s
    # ---- real (with boundaries), null, text
    enc_r = views.real(row); rec["prompt_len_real"] = int(enc_r["input_ids"].shape[1])
    if qid == 0:
        rec["raw_prompts"]["real"] = enc_r["_raw"]
    z_plus = score(model, enc_r, resp, tau, collector=coll)
    h = coll.stacked()                                                   # [L+1, T, d]
    coll.reset()
    enc_0 = views.null(row)
    z_0 = score(model, enc_0, resp, tau)
    enc_t, txt = views.text(row); rec["prompt_len_text"] = int(enc_t["input_ids"].shape[1]); rec["text_prompt"] = txt
    if qid == 0:
        rec["raw_prompts"]["null"] = enc_0["_raw"]; rec["raw_prompts"]["text"] = enc_t["_raw"]
    z_t = score(model, enc_t, resp, tau)
    t_fwd = time.time()

    # ---- coarsened distributions on K+1
    logP = coarsen_logits(z_plus, topk)
    logP0 = coarsen_logits(z_0, topk)
    logPt = coarsen_logits(z_t, topk)
    del z_0, z_t
    QA = aha_target(logP, logP0, beta)
    u_k = (logP - logP0)[:, :K]
    omega = omega_from_log_p_plus(logP[:, :K])
    a_t = tv_from_log(logP, logP0)

    W = coll.head_weight; bias = coll.head_bias
    D = final_scale_vec(h[L], coll.norm_weight, coll.norm_eps)           # [T, d]
    # reconstruction check (bf16 model logits vs fp32 read-out of h^L)
    z_rec = readout_logits(D * h[L], W, tau, bias)
    rec["recon_max_abs"] = float((z_rec - z_plus).abs().max()); rec["recon_tv"] = float(tv_from_log(torch.log_softmax(z_rec, -1), torch.log_softmax(z_plus, -1)).mean())
    if rec["recon_tv"] > args.recon_tol:
        raise RuntimeError(f"l=L read-out does not reconstruct the model logits (mean TV {rec['recon_tv']:.4f} > {args.recon_tol}); hooks/norm/head mismatch")
    del z_rec
    # baselines: no correction (P+) and the IRfl4 target (full-vocab [a,b) λ then coarsen)
    d_base = tv_from_log(logP, QA)
    r_ir = readout_logits(D * (h[args.ir_b] - h[args.ir_a]), W, tau)
    logQ_ir = build_internal_targets(z_plus, r_ir, args.ir_lambda, topk, "full")["target_log_probs"]
    d_ir = tv_from_log(logQ_ir, QA)
    del r_ir, logQ_ir

    n = T
    base_cols = dict(qid=np.full(n, qid), t=np.arange(n), T=np.full(n, T), is_eos=np.array([i == T - 1 and resp[-1] in eos_ids for i in range(n)]),
                     is_answer=np.isin(np.arange(n), ans["positions"]), a_t=a_t, d_base=d_base, d_ir=d_ir,
                     tv_plus_null=a_t, tv_plus_text=tv_from_log(logP, logPt), tv_null_text=tv_from_log(logP0, logPt))
    for k_, v_ in base_cols.items():
        store.add(k_, v_)

    # ---- candidate A / B: per-layer priors
    fam_tv0, fam_tvt, fam_d, fam_c = [], [], [], []
    for fam in ("final", "own"):
        tv0_l, tvt_l, d_l, c_l = [], [], [], []
        for l in range(L + 1):
            Dl = D if fam == "final" else own_scale_vec(h[l], coll.norm_weight, coll.norm_eps)
            z_l = readout_logits(Dl * h[l], W, tau, bias)
            logPh = coarsen_logits(z_l, topk); del z_l
            Qh = aha_target(logP, logPh, beta)
            tv0_l.append(tv_from_log(logPh, logP0)); tvt_l.append(tv_from_log(logPh, logPt))
            d_l.append(tv_from_log(Qh, QA)); c_l.append(corr_pieces((logP - logPh)[:, :K], u_k, omega))
        fam_tv0.append(torch.stack(tv0_l, 1)); fam_tvt.append(torch.stack(tvt_l, 1)); fam_d.append(torch.stack(d_l, 1)); fam_c.append(torch.stack(c_l, 1))
    store.add("tv_null_layer", torch.stack(fam_tv0, 1))      # [n, 2, L+1]
    store.add("tv_text_layer", torch.stack(fam_tvt, 1))
    store.add("d_layer", torch.stack(fam_d, 1))
    store.add("corr_layer", torch.stack(fam_c, 1))           # [n, 2, L+1, 3]

    # ---- candidate C: block contributions on K (regression) and the global-w prior
    Wk = W[topk].float()                                     # [T, K, d]
    dh = (h[1:] - h[:-1]) * D.unsqueeze(0)                   # [L, T, d]
    r_k = torch.einsum("tkd,ltd->tkl", Wk, dh) / tau         # [T, K, L]
    # (an output bias, absent in Qwen3.5, would belong to the anchor only and never enter r)
    X = centered(r_k, omega); ubar = centered(u_k, omega)
    if split == "fit":
        reg.add(X, ubar, omega)
        store.add("r2_pieces", np.full((n, 2), np.nan, dtype=np.float32))
        store.add("tv_null_w", np.full(n, np.nan, dtype=np.float32)); store.add("tv_text_w", np.full(n, np.nan, dtype=np.float32))
        store.add("d_w", np.full(n, np.nan, dtype=np.float32)); store.add("corr_w", np.full((n, 3), np.nan, dtype=np.float32))
    else:
        if w_global is None:
            raise RuntimeError("global w must be fitted before selection / held-out")
        wg = w_global.to(dev)
        store.add("r2_pieces", RegressionAccumulator.r2_pieces(X, ubar, omega, wg))   # per-position (residual, total) energy
        z_hat0 = z_plus - readout_logits(D * torch.einsum("l,ltd->td", wg.float(), h[1:] - h[:-1]), W, tau)
        logPw = coarsen_logits(z_hat0, topk); del z_hat0
        Qw = aha_target(logP, logPw, beta)
        store.add("tv_null_w", tv_from_log(logPw, logP0)); store.add("tv_text_w", tv_from_log(logPw, logPt))
        store.add("d_w", tv_from_log(Qw, QA)); store.add("corr_w", corr_pieces((logP - logPw)[:, :K], u_k, omega))
    del Wk, dh, r_k, X, h
    rec.update(t_gen=round(t_gen - t0, 2), t_fwd=round(t_fwd - t_gen, 2), t_eval=round(time.time() - t_fwd, 2),
               peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)
    store.commit()
    store.questions.append(rec)


# ------------------------------------------------------------------------------------------
# aggregation / selection
# ------------------------------------------------------------------------------------------
def subset_masks(cols: dict, split_q: set, a_star: float | None) -> dict[str, np.ndarray]:
    qsel = np.isin(cols["qid"], list(split_q))
    m = {"all": qsel, "answer": qsel & cols["is_answer"], "eos": qsel & cols["is_eos"], "other": qsel & ~cols["is_answer"] & ~cols["is_eos"]}
    if a_star is not None:
        m["active"] = qsel & (cols["a_t"] >= a_star)
    return m


def cand_arrays(cols: dict, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (tv_null, tv_text, d, corr_pieces[n,3]) for a candidate name."""
    if name == "global_w":
        return cols["tv_null_w"], cols["tv_text_w"], cols["d_w"], cols["corr_w"]
    fam, l = name.split("_"); f = 0 if fam == "final" else 1; l = int(l)
    return cols["tv_null_layer"][:, f, l], cols["tv_text_layer"][:, f, l], cols["d_layer"][:, f, l], cols["corr_layer"][:, f, l]


def stats_for(cols: dict, mask: np.ndarray, name: str) -> dict:
    tv0, tvt, d, c = cand_arrays(cols, name)
    ok = mask & ~np.isnan(d)
    if ok.sum() == 0:
        return {"n_tokens": 0}
    qid = torch.from_numpy(cols["qid"][ok])
    out = {"tv_null": weighted_stats(torch.from_numpy(tv0[ok]), qid), "tv_text": weighted_stats(torch.from_numpy(tvt[ok]), qid),
           "d": weighted_stats(torch.from_numpy(d[ok]), qid), "corr": pooled_corr(torch.from_numpy(c[ok]))}
    return out


def judge(cols: dict, masks: dict, name: str, thr, active_degenerate: bool = False) -> dict:
    """M1 / M3 / M4 / active / answer checks for one candidate on one split (plan §3-§4.1)."""
    res = {}
    s_all = stats_for(cols, masks["all"], name)
    if "d" not in s_all:
        return {"pass": False, "reason": "no_data", "d_token_mean": None, "M1": None, "M3": None, "M4": None, "active": None, "M5": None, "n_answer": 0}
    ref_tv = float(cols["tv_plus_null"][masks["all"]].mean())
    base = float(cols["d_base"][masks["all"]].mean()); ir = float(cols["d_ir"][masks["all"]].mean())
    res["M1"] = s_all["tv_null"]["token_mean"] < ref_tv
    res["M3"] = (s_all["d"]["token_median"] < thr.m3_median_tv) and (s_all["corr"] is not None and s_all["corr"] > thr.m3_corr)
    res["M4"] = s_all["d"]["token_mean"] < min(base, ir)
    if "active" in masks and masks["active"].sum() > 0 and not active_degenerate:
        s_act = stats_for(cols, masks["active"], name)
        res["active"] = s_act["d"]["token_mean"] < min(float(cols["d_base"][masks["active"]].mean()), float(cols["d_ir"][masks["active"]].mean()))
    else:
        res["active"] = None
    n_ans = int(masks["answer"].sum())
    if n_ans >= thr.m5_min_answer_tokens:
        s_ans = stats_for(cols, masks["answer"], name)
        res["M5"] = (s_ans["d"]["token_median"] < thr.m3_median_tv) and (s_ans["corr"] is not None and s_ans["corr"] > thr.m3_corr)
    else:
        res["M5"] = None
    res["n_answer"] = n_ans
    res["pass"] = bool(res["M1"] and res["M3"] and res["M4"] and res["active"] is True and res["M5"] is True)
    res["d_token_mean"] = s_all["d"]["token_mean"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--parquet", default="/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet")
    ap.add_argument("--chat-template", default=os.path.join(os.path.dirname(__file__), "..", "chat_templates", "perception_chat_template_qwen35.jinja"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--sizes", default="200,100,100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N questions of each split (0 = all)")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--tau", type=float, default=1.0, help="scoring temperature (= rollout temperature in training, 1.0)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-prompt-len", type=int, default=10240)
    ap.add_argument("--image-patch-size", type=int, default=14, help="verl data.image_patch_size default used by the student path")
    ap.add_argument("--ir-a", type=int, default=12); ap.add_argument("--ir-b", type=int, default=20); ap.add_argument("--ir-lambda", type=float, default=4.0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--recon-tol", type=float, default=0.02, help="max mean TV between the h^L read-out and the model logits (bf16 vs fp32)")
    ap.add_argument("--max-consecutive-errors", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for stale in ("questions.jsonl",):
        if os.path.exists(os.path.join(args.out, stale)):
            os.remove(os.path.join(args.out, stale))          # never append a re-run onto an old run
    torch.manual_seed(args.seed)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from verl.utils.layer_prior_probe import Thresholds
    thr = Thresholds()
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    tpl = open(args.chat_template).read()
    processor.chat_template = tpl; processor.tokenizer.chat_template = tpl
    tok = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model = model.cuda().eval() if torch.cuda.is_available() else model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    eos_ids = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")})
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]
    coll = BoundaryCollector(model)
    L = coll.L
    print(f"[probe] model loaded in {time.time()-t0:.0f}s  L={L}  V={coll.head_weight.shape[0]}  eos={eos_ids}", flush=True)

    rows = load_rows(args.parquet, args.n, args.seed)
    splits = assign_splits(rows, tuple(int(x) for x in args.sizes.split(",")), args.seed)
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    views = Views(processor, args.max_prompt_len, args.image_patch_size)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        commit = None
    import hashlib as _h, platform
    pq_sha1 = _h.sha1(open(args.parquet, "rb").read()).hexdigest()
    manifest = {
        "args": vars(args), "commit": commit, "model": args.model, "model_commit_hash": getattr(model.config, "_commit_hash", None), "L": L, "torch": torch.__version__,
        "python": platform.python_version(), "cuda": torch.version.cuda, "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "parquet": {"path": args.parquet, "sha1": pq_sha1, "rows": len(rows), "mtime": os.path.getmtime(args.parquet)},
        "generation_config": {k: v for k, v in model.generation_config.to_dict().items() if k in ("do_sample", "temperature", "top_p", "top_k", "eos_token_id", "pad_token_id", "bos_token_id")},
        "thresholds": vars(thr), "tie_break": "min token-mean d on selection; then final < own < global_w; then lower layer",
        "transformers": __import__("transformers").__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "chat_template_sha1": hashlib.sha1(tpl.encode()).hexdigest(), "eos_ids": eos_ids, "generation": "HF greedy, do_sample=False",
        "split_sizes": {k: len(v) for k, v in splits.items()}, "split_groups": {k: len({group_key(r) for r in v}) for k, v in splits.items()},
        "split_rows": {k: [r["_row"] for r in v] for k, v in splits.items()},
        "text_view": "student prompt with every <image> placeholder removed (leading newlines stripped); no extra hint",
        "null_view": "teacher bbox_images with the LAST image replaced by its mean colour (null_scope=last)",
        "student_view": "verl process_image(image_patch_size=%d) on `images`; teacher views use raw RGB PIL like the trainer" % args.image_patch_size,
        "positions": "logits at P-1..P+T-2 predict y_0..y_{T-1}; EOS included when generated",
        "answer_rule": "(1) first <answer>..</answer> content; (2) leading option letter; (3) option letter starting the LAST non-empty line; else missing",
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1, ensure_ascii=False)

    store = Store(L); reg = RegressionAccumulator(L); w_global = None; regression = {}
    a_star = None; consecutive_errors = 0; n_oom = 0; active_degenerate = True
    for split in ("fit", "selection", "heldout"):
        for i, row in enumerate(splits[split]):
            qid = len(store.questions)
            try:
                evaluate_question(model, views, row, args, tok, eos_ids, pad_id, coll, store, reg, w_global, qid, split)
            except Exception as e:  # record and continue; a crash in ONE question must not lose the run ...
                store.discard()
                store.questions.append({"qid": qid, "row": row["_row"], "split": split, "group": group_key(row), "error": repr(e)})
                print(f"[probe] !! {split} q{qid} row{row['_row']}: {e!r}", flush=True)
                coll.reset()
                consecutive_errors += 1
                if isinstance(e, torch.cuda.OutOfMemoryError) if hasattr(torch.cuda, "OutOfMemoryError") else False:
                    torch.cuda.empty_cache(); n_oom += 1
                # ... but a systematic failure must stop early, not after an hour of error records
                n_done = len(store.questions)
                if consecutive_errors >= args.max_consecutive_errors or (n_done >= 5 and all("error" in q for q in store.questions[:5])):
                    raise RuntimeError(f"systematic failure: {consecutive_errors} consecutive errors (last: {e!r})")
            else:
                consecutive_errors = 0
            assert len(store.questions) == qid + 1, "exactly one record per question"
            r = store.questions[-1]
            n_err = sum("error" in q for q in store.questions)
            print(f"[probe] {split} {i+1}/{len(splits[split])} row={r['row']} T={r.get('T')} ans={(r.get('answer') or {}).get('source')} "
                  f"gen={r.get('t_gen')}s fwd={r.get('t_fwd')}s eval={r.get('t_eval')}s mem={r.get('peak_mem_gb')}G recon_tv={r.get('recon_tv')} errors={n_err}", flush=True)
            with open(os.path.join(args.out, "questions.jsonl"), "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if split == "fit":
            regression = reg.solve()
            if regression.get("fittable"):
                w_global = torch.tensor(regression["w"], dtype=torch.float64)
            cols_fit = store.finalize()
            fit_mask = np.isin(cols_fit["qid"], [q["qid"] for q in store.questions if q.get("split") == "fit" and "T" in q and "error" not in q])
            a_vals = cols_fit["a_t"][fit_mask]
            a_star = float(np.quantile(a_vals, 0.75)) if fit_mask.sum() else None
            active_degenerate = (a_star is None) or (a_star < 1e-3) or (int((a_vals >= a_star).sum()) < 50)
            regression["a_star"] = a_star
            regression["a_t_quantiles_fit"] = {q: float(np.quantile(a_vals, q)) for q in (0.5, 0.75, 0.9, 0.95)} if fit_mask.sum() else None
            regression["active_degenerate"] = bool(active_degenerate)
            json.dump(regression, open(os.path.join(args.out, "regression.json"), "w"), indent=1)
            print(f"[probe] fit done: w fittable={regression.get('fittable')} r2_in={regression.get('r2_in_sample')} a*={a_star}", flush=True)
            if w_global is None:
                w_global = torch.zeros(L, dtype=torch.float64)     # keep the pipeline alive; candidate C then equals P+ (reported as unfittable)

    cols = store.finalize()
    np.savez_compressed(os.path.join(args.out, "positions.npz"), **cols)
    qs = {s: {q["qid"] for q in store.questions if q.get("split") == s and "error" not in q and "skipped" not in q} for s in ("fit", "selection", "heldout")}
    cand_names = [f"{fam}_{l}" for fam in ("final", "own") for l in range(L + 1)] + ["global_w"]
    metrics = {"a_star": a_star, "active_degenerate": bool(active_degenerate), "splits": {}, "thresholds": vars(thr),
               "global_w_fittable": bool(regression.get("fittable", False)), "n_oom": n_oom}
    rows_csv = []
    for split in ("fit", "selection"):
        masks = subset_masks(cols, qs[split], a_star)
        sp = {"n_questions": len(qs[split]), "subsets": {k: int(v.sum()) for k, v in masks.items()},
              "reference": {k: {"tv_plus_null": float(cols["tv_plus_null"][m].mean()), "tv_plus_text": float(cols["tv_plus_text"][m].mean()),
                                "tv_null_text": float(cols["tv_null_text"][m].mean()), "d_base": weighted_stats(torch.from_numpy(cols["d_base"][m]), torch.from_numpy(cols["qid"][m])),
                                "d_ir": weighted_stats(torch.from_numpy(cols["d_ir"][m]), torch.from_numpy(cols["qid"][m]))} for k, m in masks.items() if m.sum()},
              "candidates": {}}
        for name in cand_names:
            if split == "fit" and name == "global_w":
                continue
            sp["candidates"][name] = {k: stats_for(cols, m, name) for k, m in masks.items() if m.sum()}
            sp["candidates"][name]["judge"] = judge(cols, masks, name, thr, active_degenerate) if masks["all"].sum() else None
            for k, m in masks.items():
                st = sp["candidates"][name].get(k, {})
                if st.get("n_tokens", 1) == 0 or "d" not in st:
                    continue
                rows_csv.append({"split": split, "candidate": name, "subset": k, "n_tokens": st["d"]["n_tokens"], "n_questions": st["d"]["n_questions"],
                                 "tv_null_mean": st["tv_null"]["token_mean"], "tv_text_mean": st["tv_text"]["token_mean"], "d_mean": st["d"]["token_mean"],
                                 "d_median": st["d"]["token_median"], "d_p90": st["d"]["token_p90"], "d_qmean": st["d"]["question_mean"], "corr": st["corr"]})
        if split != "fit" and masks["all"].sum():
            pieces = cols["r2_pieces"][masks["all"]]
            res, tot = float(np.nansum(pieces[:, 0])), float(np.nansum(pieces[:, 1]))
            sp["r2_global_w"] = (1.0 - res / tot) if tot > 1e-12 else None
        metrics["splits"][split] = sp

    # ---- selection (plan §4.1): on `selection`, among passing candidates pick min token-mean d; ties -> final < own < global_w, then lower layer
    sel = {k: v for k, v in metrics["splits"].get("selection", {}).get("candidates", {}).items() if v.get("judge") and v["judge"].get("d_token_mean") is not None}
    order = {"final": 0, "own": 1, "global": 2}
    def sort_key(nm):   # pre-registered tie-break: smaller token-mean d, then final < own < global_w, then lower layer
        j = sel[nm]["judge"]; fam = nm.split("_")[0]; l = int(nm.split("_")[1]) if fam != "global" else -1
        return (round(j["d_token_mean"], 6), order[fam], l)
    passing = [nm for nm in sel if sel[nm]["judge"]["pass"]]
    best_any = min(sel, key=sort_key) if sel else None
    locked = min(passing, key=sort_key) if passing else None
    n_ans_sel = int(subset_masks(cols, qs["selection"], a_star)["answer"].sum()) if qs["selection"] else 0
    selection = {"passing": passing, "locked": locked, "best_by_d_regardless_of_gates": best_any, "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "w_sha1": hashlib.sha1(json.dumps(regression.get("w")).encode()).hexdigest() if regression.get("w") else None,
                 "tie_break": "min token-mean d on selection; then final < own < global_w; then lower layer",
                 "answer_tokens_selection": n_ans_sel, "active_degenerate": bool(active_degenerate),
                 "selection_judge": {nm: sel[nm]["judge"] for nm in ([locked] if locked else [])}}
    # held-out (plan §4.1 step 3): ONLY the locked candidate and the two fixed baselines are evaluated here.
    masks_h = subset_masks(cols, qs["heldout"], a_star)
    if masks_h["all"].sum():
        metrics["splits"]["heldout"] = {"n_questions": len(qs["heldout"]), "subsets": {k: int(v.sum()) for k, v in masks_h.items()},
            "reference": {k: {"tv_plus_null": float(cols["tv_plus_null"][m].mean()), "tv_plus_text": float(cols["tv_plus_text"][m].mean()),
                              "tv_null_text": float(cols["tv_null_text"][m].mean()), "d_base": weighted_stats(torch.from_numpy(cols["d_base"][m]), torch.from_numpy(cols["qid"][m])),
                              "d_ir": weighted_stats(torch.from_numpy(cols["d_ir"][m]), torch.from_numpy(cols["qid"][m]))} for k, m in masks_h.items() if m.sum()},
            "candidates": {}}
        pieces = cols["r2_pieces"][masks_h["all"]]
        res, tot = float(np.nansum(pieces[:, 0])), float(np.nansum(pieces[:, 1]))
        metrics["splits"]["heldout"]["r2_global_w"] = (1.0 - res / tot) if tot > 1e-12 else None
        # every other candidate's held-out numbers go to a separate post-hoc file and must never be used for selection
        post_hoc = {name: {k: stats_for(cols, m, name) for k, m in masks_h.items() if m.sum()} for name in cand_names}
        json.dump(post_hoc, open(os.path.join(args.out, "heldout_all_candidates_post_hoc_DO_NOT_SELECT.json"), "w"), indent=1)
    if locked:
        if masks_h["all"].sum():
            metrics["splits"]["heldout"]["candidates"][locked] = {k: stats_for(cols, m, locked) for k, m in masks_h.items() if m.sum()}
            selection["heldout_judge"] = judge(cols, masks_h, locked, thr, active_degenerate)
            _, _, d_c, _ = cand_arrays(cols, locked)
            m = masks_h["all"] & ~np.isnan(d_c)
            gnames = [store.questions[int(q)]["group"] for q in cols["qid"][m]]
            g = torch.from_numpy(np.unique(np.array(gnames), return_inverse=True)[1])
            selection["heldout_bootstrap"] = {
                "cand_minus_base": paired_bootstrap(torch.from_numpy(d_c[m] - cols["d_base"][m]), g, args.n_boot, args.seed),
                "cand_minus_ir": paired_bootstrap(torch.from_numpy(d_c[m] - cols["d_ir"][m]), g, args.n_boot, args.seed),
            }
            selection["heldout_pass"] = selection["heldout_judge"]["pass"]
    if locked and selection.get("heldout_pass"):
        selection["verdict"] = "pass"
    elif locked:
        selection["verdict"] = "selection_pass_heldout_fail"
    elif n_ans_sel < thr.m5_min_answer_tokens:
        selection["verdict"] = "insufficient_answer_coverage"       # plan §3.5: 待补充, not a pass / fail on M5
    else:
        selection["verdict"] = "no_candidate_passed_selection"
    metrics["selection"] = selection
    metrics["questions"] = {"n": len(store.questions), "errors": sum("error" in q for q in store.questions), "skipped": sum("skipped" in q for q in store.questions),
                            "truncated": sum(bool(q.get("truncated")) for q in store.questions)}
    src = defaultdict(int)
    for q in store.questions:
        src[(q.get("answer") or {}).get("source", "n/a")] += 1
    metrics["questions"]["answer_sources"] = dict(src)
    metrics["cost"] = {"wall_s": round(time.time() - t0), "peak_mem_gb": max((q.get("peak_mem_gb") or 0) for q in store.questions) if store.questions else None,
                       "mean_t_gen": float(np.mean([q["t_gen"] for q in store.questions if "t_gen" in q])) if any("t_gen" in q for q in store.questions) else None,
                       "mean_t_fwd": float(np.mean([q["t_fwd"] for q in store.questions if "t_fwd" in q])) if any("t_fwd" in q for q in store.questions) else None,
                       "mean_t_eval": float(np.mean([q["t_eval"] for q in store.questions if "t_eval" in q])) if any("t_eval" in q for q in store.questions) else None}
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    json.dump(selection, open(os.path.join(args.out, "selection.json"), "w"), indent=1)
    with open(os.path.join(args.out, "per_layer.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()) if rows_csv else ["split"])
        wr.writeheader(); wr.writerows(rows_csv)
    print(f"[probe] done in {time.time()-t0:.0f}s  verdict={selection['verdict']} locked={locked} best={best_any}", flush=True)


if __name__ == "__main__":
    main()
