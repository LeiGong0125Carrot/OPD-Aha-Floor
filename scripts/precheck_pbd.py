#!/usr/bin/env python
"""P5 / PBD label-free pre-check (13_P5_assessment §15-§17): on sequences produced by speed_probe_pbd.py --part vllm.
Per row and branch ratio: student (full view) and teacher (crop view, no hint) distributions on
  V0 row      = y^S                      and   branch row = prefix + y^T
coarsened to the student top-100 + tail support (training口径). Reports
  J        : JSD_0.5(p+ || pS) density on NEW states (positions after the first divergent token t_div) vs V0-row density
  J_tail   : last-8-token JSD on the branch row vs last-8 on the V0 row
  D_hat    : P4 symmetric-KL sample estimate (window K=64 and all) from realized log-probs of y^T / y^S-suffix under both views
  r_tilde  : rewrite magnitude minus same-view resample baseline (diagnostic only)
  leak     : crop-wording fraction in y^T
No GT, no answer extraction.
用法: python scripts/precheck_pbd.py --inp eval/precheck_pbd [--n_rows 200]
"""
import argparse, json, math, os, re, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import probe_layer_prior as P  # noqa: E402
from speed_probe_pbd import PARQUET, MODEL, TPL, LEAK, crop_content, rmag  # noqa: E402
from verl.utils.layer_prior_probe import coarsen_logits  # noqa: E402


def jsd(logp, logq, alpha=0.5):
    p, q = logp.exp(), logq.exp(); m = alpha * p + (1 - alpha) * q
    logm = m.clamp_min(1e-12).log()
    return alpha * (p * (logp - logm)).sum(-1) + (1 - alpha) * (q * (logq - logm)).sum(-1)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0: return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inp", default=os.path.join(ROOT, "eval/precheck_pbd"))
    ap.add_argument("--n", type=int, default=200, help="same --n as the speed_probe vllm run (row sampling)"); ap.add_argument("--n_rows", type=int, default=10**9)
    ap.add_argument("--K", type=int, default=100); ap.add_argument("--win", type=int, default=64)
    args = ap.parse_args()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(MODEL); tpl = open(TPL).read(); processor.chat_template = tpl; processor.tokenizer.chat_template = tpl
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    views = P.Views(processor, 10240, 14)
    seqs = json.load(open(os.path.join(args.inp, "sequences.json")))[: args.n_rows]
    rows_by = {r["_row"]: r for r in P.load_rows(PARQUET, args.n, 42)}
    missing = [s["row"] for s in seqs if s["row"] not in rows_by]
    if missing: raise RuntimeError(f"{len(missing)} sequence rows not in load_rows(n={args.n}); pass the same --n as the vllm run")
    ratios = sorted(seqs[0]["branches"].keys(), key=float)
    out = {r: {k: [] for k in ("J_new", "J_v0", "J_tail_br", "J_tail_v0", "D_win", "D_all", "r_tilde", "leak", "yT_len", "n_new", "t_div_rel", "dup_frac", "J_new_masked", "leak_pos_frac")} for r in ratios}
    tok = processor.tokenizer
    def leak_mask(ids):
        """True at token positions that fall inside a crop-wording match (decoded with offsets)."""
        text = tok.decode(ids); spans = [m.span() for m in LEAK.finditer(text)]
        if not spans: return torch.zeros(len(ids), dtype=torch.bool)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        # map decoded-text tokens back by position (re-tokenisation may differ slightly; approximate by char offsets)
        mask = torch.zeros(len(enc["input_ids"]), dtype=torch.bool)
        for i, (a, b) in enumerate(enc["offset_mapping"]):
            if any(a < e and b > s0 for s0, e in spans): mask[i] = True
        if len(mask) != len(ids):  # fallback: mark nothing if alignment broke
            m2 = torch.zeros(len(ids), dtype=torch.bool); n = min(len(mask), len(ids)); m2[:n] = mask[:n]; return m2
        return mask

    def dists(enc, resp):
        """student-support-coarsened log-probs for teacher/student plus realized log-probs; resp predicted at positions 0..T-1"""
        return P.score(model, enc, resp, 1.0)

    for i, s in enumerate(seqs):
        row = rows_by[s["row"]]; es = views.student(row)
        imgs = [P.load_rgb(d["path"]) for d in row["bbox_images"]]
        ec = views.encode(P.messages_from_template(crop_content(row["prompt"][0]["content"]), imgs), imgs)
        yS = s["yS"]
        with torch.no_grad():
            zs = dists(es, yS); topk = zs.topk(args.K, -1).indices
            zt = dists(ec, yS); lS = coarsen_logits(zs, topk); lT = coarsen_logits(zt, topk)
            j_v0 = jsd(lT, lS); T = len(yS); idxS = torch.tensor(yS, device=zs.device).unsqueeze(-1)
            realS_S = torch.log_softmax(zs.float(), -1).gather(-1, idxS).squeeze(-1)
            realT_S = torch.log_softmax(zt.float(), -1).gather(-1, idxS).squeeze(-1)
            for r in ratios:
                b = s["branches"][r]; pre, yT, yS2 = b["prefix"], b["yT"], b["yS2"]; k = len(pre)
                if len(yT) == 0: continue
                resp = pre + yT
                zsb = dists(es, resp); topkb = zsb.topk(args.K, -1).indices
                lSb = coarsen_logits(zsb, topkb); ztb = dists(ec, resp); lTb = coarsen_logits(ztb, topkb)
                j_br = jsd(lTb, lSb)                       # positions 0..len(resp)-1 ; prefix part duplicates V0
                suf = yS[k:]; div = next((d for d in range(min(len(yT), len(suf))) if yT[d] != suf[d]), min(len(yT), len(suf)))
                t_div = k + div                            # first divergent position in the branch row
                new = j_br[t_div + 1:] if t_div + 1 < len(resp) else j_br[len(resp):]
                o = out[r]
                o["J_new"].append(float(new.mean()) if len(new) else float("nan")); o["n_new"].append(int(len(new)))
                lm = leak_mask(yT).to(j_br.device); lm_full = torch.cat([torch.zeros(k, dtype=torch.bool, device=j_br.device), lm])
                keep = ~lm_full[t_div + 1:] if t_div + 1 < len(resp) else torch.zeros(0, dtype=torch.bool, device=j_br.device)
                # also drop the 2 positions after a leak token (the continuation of the leaked phrase)
                if len(lm_full):
                    after = torch.zeros_like(lm_full); idxs = torch.where(lm_full)[0]
                    for d_ in (1, 2):
                        j_ = idxs + d_; after[j_[j_ < len(after)]] = True
                    keep = keep & ~after[t_div + 1:] if t_div + 1 < len(resp) else keep
                o["J_new_masked"].append(float(new[keep].mean()) if len(new) and keep.any() else float("nan"))
                o["leak_pos_frac"].append(float(lm.float().mean()) if len(lm) else 0.0)
                o["J_v0"].append(float(j_v0[k:].mean()) if T > k else float("nan"))      # V0 density on the same suffix region
                o["J_tail_br"].append(float(j_br[-8:].mean())); o["J_tail_v0"].append(float(j_v0[-8:].mean()))
                o["dup_frac"].append(float((t_div - k) / max(len(yT), 1)))               # share of y^T that just copies y^S
                o["t_div_rel"].append(float(t_div / len(resp)))
                # D_hat (P4): mean realized log-prob over continuation window under both views
                idx = torch.tensor(resp, device=zsb.device).unsqueeze(-1)
                realS_T = torch.log_softmax(zsb.float(), -1).gather(-1, idx).squeeze(-1)[k:]
                realT_T = torch.log_softmax(ztb.float(), -1).gather(-1, idx).squeeze(-1)[k:]
                sS, sT = realS_S[k:], realT_S[k:]
                for name, W in (("D_win", args.win), ("D_all", 10**9)):
                    lT_yT, lT_yS = realT_T[:W].mean(), sT[:W].mean(); lS_yS, lS_yT = sS[:W].mean(), realS_T[:W].mean()
                    o[name].append(float(0.5 * ((lT_yT - lT_yS) + (lS_yS - lS_yT))))
                o["r_tilde"].append(rmag(yT, suf) - rmag(yS2, suf)); o["leak"].append(float(bool(LEAK.search(processor.tokenizer.decode(yT)))))
                o["yT_len"].append(len(yT))
        if (i + 1) % 10 == 0: print(f"[precheck] {i+1}/{len(seqs)}", flush=True)

    summ = {}
    print(f"\n[precheck] n={len(seqs)} rows; JSD alpha=0.5 on student top-{args.K}+tail")
    print(f"{'ratio':>6s} {'J_new':>7s} {'J_v0':>7s} {'ratio':>6s} {'pass(b′)':>9s} {'Jtail_br':>9s} {'Jtail_v0':>9s} {'x':>5s} {'dup':>5s} {'leak':>5s} {'|yT|':>5s} {'ρ(D̂,J)':>8s} {'ρ(D̂all,J)':>10s} {'ρ(r̃,J)':>8s} {'ρ(J,len)':>9s}")
    for r in ratios:
        o = out[r]; Jn, Jv = np.array(o["J_new"]), np.array(o["J_v0"]); ok = ~np.isnan(Jn) & ~np.isnan(Jv)
        Jm = np.array(o["J_new_masked"]); noleak = np.array(o["leak"]) == 0
        d = dict(J_new=float(np.nanmean(Jn)), J_v0=float(np.nanmean(Jv)), J_ratio=float(np.nanmean(Jn) / np.nanmean(Jv)),
                 J_new_masked=float(np.nanmean(Jm)), J_masked_ratio=float(np.nanmean(Jm) / np.nanmean(Jv)),
                 J_new_noleakrows=float(np.nanmean(Jn[noleak])) if noleak.any() else float("nan"), J_v0_noleakrows=float(np.nanmean(Jv[noleak])) if noleak.any() else float("nan"),
                 leak_pos_frac=float(np.mean(o["leak_pos_frac"])),
                 pass_b1=float(np.mean(Jn[ok] >= 0.5 * Jv[ok])), J_tail_br=float(np.mean(o["J_tail_br"])), J_tail_v0=float(np.mean(o["J_tail_v0"])),
                 tail_x=float(np.mean(o["J_tail_br"]) / max(np.mean(o["J_tail_v0"]), 1e-9)), dup_frac=float(np.mean(o["dup_frac"])), leak=float(np.mean(o["leak"])),
                 yT_len=float(np.mean(o["yT_len"])), rho_Dwin_J=spearman(np.array(o["D_win"])[ok], Jn[ok]), rho_Dall_J=spearman(np.array(o["D_all"])[ok], Jn[ok]),
                 rho_rt_J=spearman(np.array(o["r_tilde"])[ok], Jn[ok]), rho_J_len=spearman(np.array(o["yT_len"])[ok], Jn[ok]), n=int(ok.sum()))
        summ[r] = d
        print(f"{float(r):6.2f} {d['J_new']:7.4f} {d['J_v0']:7.4f} {d['J_ratio']:6.2f} {d['pass_b1']:9.2f} {d['J_tail_br']:9.4f} {d['J_tail_v0']:9.4f} {d['tail_x']:5.2f} {d['dup_frac']:5.2f} {d['leak']:5.2f} {d['yT_len']:5.0f} {d['rho_Dwin_J']:8.2f} {d['rho_Dall_J']:10.2f} {d['rho_rt_J']:8.2f} {d['rho_J_len']:9.2f}")
    print("\n泄漏控制:  J_new(mask 泄漏位+后2位) / J_v0 ;  仅无泄漏行: J_new / J_v0")
    for r in ratios:
        d = summ[r]; print(f"{float(r):6.2f}  masked {d['J_new_masked']:.4f} / {d['J_v0']:.4f} = {d['J_masked_ratio']:.2f}   no-leak rows {d['J_new_noleakrows']:.4f} / {d['J_v0_noleakrows']:.4f} = {d['J_new_noleakrows']/max(d['J_v0_noleakrows'],1e-9):.2f}   leak token frac {d['leak_pos_frac']:.3f}")
    print("\n门槛: (b′) J_new ≥ 0.5·J_v0 (列 ratio≥0.5 / pass 比例);  (b″) J_tail_br ≥ 1.5·J_tail_v0 (列 x≥1.5);  泄漏 >5% 需 mask;  D̂ 与 J 显著正相关才可作选择器")
    json.dump({"summary": summ, "per_row": out}, open(os.path.join(args.inp, "precheck.json"), "w"), indent=1)
    print("[precheck] saved", os.path.join(args.inp, "precheck.json"))


if __name__ == "__main__":
    main()
