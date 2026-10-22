#!/usr/bin/env python
"""Student-view vs teacher-view hidden-state probe (user 10-22, motivated by OPRD arXiv 2606.06021). Label-free: no GT.

Question: OPRD distills hidden states (per layer, per response position) instead of next-token distributions. In off6k the
teacher is the SAME frozen base under a different view (crop instead of the red-box full image), so the hidden-state gap is
a view gap, not a weight gap. How large is it, where (layer / position), how much of it is crop CONTENT vs just "a crop-shaped
image" (style), does the output-level gap see all of it, and did OPD training (base -> trained ckpt) already close it?

Per student trajectory (base model's own T=1.0 samples, probe_hidden_answer gen.json), teacher-forced, hidden states at the
positions that predict y_t (P-1+t; t=0 is the last prompt token = the pre-generation state), layers LAYERS:
  S  = base, student view (red-box full image + hint)           C  = base, training teacher view (crop, same text)
  M  = base, mismatched crop (crop of ANOTHER question, same text: view-type control, as arm Amis)
  S2 = trained ckpt, student view                               C2 = trained ckpt, crop view
  M is the donor crop RESIZED to this question's crop size (identical image grid, prompt length and positions as C; only
  pixel content differs), donor = next row with a different source image and crop.
Per token x layer, per pair (SC, SM, CM, S2C, S2C2, SS2): raw cosine and the normalised squared distance
  nd = ||a - b||^2 / s_l,  s_l = E||h_S - mu_S||^2 over the S-view tokens of the first --warm rows (per-layer scale only;
  centring-free, so no view/model offset leaks in via a reference mean; huge constant dims cancel in a - b).
Content share (per layer) = (nd_SM - nd_SC) / nd_SM: fraction of the student-vs-crop-teacher distance that the RIGHT crop
removes relative to a same-shaped wrong crop. Output-level JSD (top-K+tail) S-C, S-M, S2-C. --gen must be STUDENT-view gen.
用法: python scripts/probe_view_hidden.py --part score --gen <base gen.json> --trained <hf> --out <dir> [--n 600 --per 2]
      python scripts/probe_view_hidden.py --part analyze --out <dir>
"""
import argparse, collections, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import probe_hidden_answer as H
import probe_traj_within as W
from probe_teacher_recover import jsd_topk

LAYERS = (1, 4, 8, 12, 16, 20, 24, 28, 31, 32)      # hidden_states index (0 = embeddings, 32 = after the final norm)
PAIRS = {"SC": ("S", "C"), "SM": ("S", "M"), "CM": ("C", "M"), "S2C": ("S2", "C"), "S2C2": ("S2", "C2"), "SS2": ("S", "S2")}


# ------------------------------------------------------------------------------------------------ pure helpers (unit-tested)
def cos_rows(a, b):
    """Row-wise cosine of [..., D] tensors."""
    import torch
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


def ndist(a, b, scale):
    """||a - b||^2 / scale per row; a, b [L, T, D], scale [L] -> [L, T]."""
    return ((a.float() - b.float()) ** 2).sum(-1) / scale.unsqueeze(-1)


def other_row(i, keys):
    """Mismatched-crop donor for row i: the next row (cyclic) whose key (source image, crop path) differs in BOTH parts."""
    n = len(keys)
    for k in range(1, n):
        j = (i + k) % n
        if keys[j][0] != keys[i][0] and keys[j][1] != keys[i][1]: return j
    raise ValueError("no row with a different source image and crop")


def forward(model, enc, y, layers):
    """Teacher-force y after the encoded prompt. Returns (log-probs [T,V] fp32, hidden [len(layers), T, D] fp32) at the
    positions predicting y_0..y_{T-1} (P-1 .. P+T-2)."""
    import torch
    dev = model.device; P = enc["input_ids"].shape[1]; T = len(y)
    resp = torch.tensor(y, dtype=enc["input_ids"].dtype).unsqueeze(0)
    inp = {k: v.to(dev) for k, v in enc.items() if k != "_raw"}
    inp["input_ids"] = torch.cat([inp["input_ids"], resp.to(dev)], 1)
    inp["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(resp).to(dev)], 1)
    if "mm_token_type_ids" in inp: inp["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(resp).to(dev)], 1)
    out = model(**inp, output_hidden_states=True, logits_to_keep=T + 1, use_cache=False)
    if tuple(out.logits.shape[:2]) != (1, T + 1): raise RuntimeError(f"logits_to_keep semantics: {tuple(out.logits.shape)}")
    hs = out.hidden_states
    if len(hs) <= max(layers) or hs[0].shape[1] != P + T: raise RuntimeError(f"hidden_states layout: {len(hs)} x {tuple(hs[0].shape)}")
    hid = torch.stack([hs[l][0, P - 1:P - 1 + T] for l in layers]).float()
    return torch.log_softmax(out.logits[0, :T].float(), -1), hid


def bucket_masks(t, commit):
    """Disjoint: t=0 and 1-7 take precedence; t>=8 split by the commit token (unknown commit = own bucket)."""
    known = commit >= 0; late = t >= 8
    return {"t=0 (生成前)": t == 0, "1–7": (t >= 1) & (t < 8), "承诺前 (≥8)": late & known & (t < commit),
            "承诺 token (≥8)": late & known & (t == commit), "承诺后 (≥8)": late & known & (t > commit), "承诺未知 (≥8)": late & ~known}


# ------------------------------------------------------------------------------------------------ score (GPU)
def part_score(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText
    import probe_layer_prior as P
    if "teacher" in os.path.basename(os.path.dirname(os.path.abspath(args.gen))): raise SystemExit("--gen must be a student-view gen.json")
    processor = H.setup_processor(); ps = processor.image_processor.patch_size; tok = processor.tokenizer
    gen = {g["row"]: g for g in json.load(open(args.gen))}
    rows = [r for r in H.load_rows(args.n) if r["_row"] in gen]
    mk = lambda p: AutoModelForImageTextToText.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    base, trained = mk(H.BASE), mk(args.trained)
    views = P.Views(processor, 12288, ps); rng = np.random.RandomState(42)
    crop_of = lambda r: P.load_rgb(r["bbox_images"][0]["path"])
    tok_ids = lambda text: tok(text, add_special_tokens=False)["input_ids"][:args.max_len]
    # per-layer scale s_l = E||h_S - mu_S||^2 from the S-view tokens of the first --warm rows (sample 0 of each; these
    # rows are also scored below -- the scale is a per-layer constant, so it does not change any within-layer comparison)
    hs_w = []
    with torch.no_grad():
        for r in rows[:args.warm]:
            y = tok_ids(gen[r["_row"]]["texts"][0])
            if len(y) < 2: continue
            try:
                hs_w.append(forward(base, views.student(r), y, LAYERS)[1])
            except (RuntimeError, ValueError) as e:
                print(f"[warm] skip row {r['_row']}: {e}", flush=True)
    assert hs_w, "no warm-up row survived"
    hw = torch.cat(hs_w, 1); cnt = hw.shape[1]
    scale = ((hw - hw.mean(1, keepdim=True)) ** 2).sum(-1).mean(1); del hw, hs_w     # [L]
    keys = [(gen[r["_row"]]["group"], r["bbox_images"][0]["path"]) for r in rows]
    cols, n_skip, vstat = collections.defaultdict(list), collections.Counter(), collections.defaultdict(list)
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    for ri, r in enumerate(rows):
        g = gen[r["_row"]]; content = r["prompt"][0]["content"]; donor = rows[other_row(ri, keys)]
        try:
            crop = crop_of(r); mis = crop_of(donor).resize(crop.size, Image.BICUBIC)
            enc = dict(S=views.student(r), C=views.encode(P.messages_from_template(content, [crop]), [crop]),
                       M=views.encode(P.messages_from_template(content, [mis]), [mis]))
            if enc["C"]["input_ids"].shape != enc["M"]["input_ids"].shape or not torch.equal(enc["C"]["image_grid_thw"], enc["M"]["image_grid_thw"]):
                raise RuntimeError("mismatched crop does not share the crop view's grid / length")
        except (RuntimeError, ValueError) as e:
            print(f"[score] skip row {r['_row']}: {e}", flush=True); n_skip["row"] += 1; continue
        for v in "SCM":
            ids = enc[v]["input_ids"][0]; vstat[f"{v}_prompt"].append(int(len(ids))); vstat[f"{v}_img"].append(int((ids == pad).sum()))
        otext = W.option_texts(content)
        for si in sorted(rng.permutation(len(g["texts"]))[:args.per].tolist()):
            text, let = g["texts"][si], g["letters"][si]; y = tok_ids(text)
            if len(y) < 2: n_skip["short"] += 1; continue
            t = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ci = W.commit_char(text, let, otext.get(let, "")) if let else None
            ct = W.char_to_token(t["offset_mapping"], ci) if ci is not None else -1
            with torch.no_grad():
                ls, hd = {}, {}
                for v, m, e in (("S", base, "S"), ("C", base, "C"), ("M", base, "M"), ("S2", trained, "S"), ("C2", trained, "C")):
                    ls[v], hd[v] = forward(m, enc[e], y, LAYERS)
                for pr, (a, b) in PAIRS.items():
                    cols[f"cos_{pr}"].append(cos_rows(hd[a], hd[b]).T.cpu().numpy().astype(np.float32))          # [T, L]
                    cols[f"nd_{pr}"].append(ndist(hd[a], hd[b], scale).T.cpu().numpy().astype(np.float32))
                idx = ls["S"].topk(args.K, -1).indices
                cols["jsd_SC"].append(jsd_topk(ls["S"], ls["C"], idx).cpu().numpy())
                cols["jsd_SM"].append(jsd_topk(ls["S"], ls["M"], idx).cpu().numpy())
                idx2 = ls["S2"].topk(args.K, -1).indices
                cols["jsd_S2C"].append(jsd_topk(ls["S2"], ls["C"], idx2).cpu().numpy())

            n = len(y)
            cols["row"].append(np.full(n, r["_row"])); cols["s"].append(np.full(n, si)); cols["t"].append(np.arange(n))
            cols["tok"].append(np.array(y)); cols["commit"].append(np.full(n, ct)); n_skip["commit_beyond_max_len"] += int(ct >= len(y))
            del ls, hd
        if (ri + 1) % 50 == 0: print(f"[score] {ri+1}/{len(rows)}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "tokens.npz"), **{k: np.concatenate(v) for k, v in cols.items()})
    json.dump(dict(skipped=dict(n_skip), n_rows=len(rows), trained=args.trained, K=args.K, layers=LAYERS, warm_tokens=int(cnt), gen=args.gen,
                   scale=[float(x) for x in scale],
                   views={k: dict(mean=float(np.mean(v)), median=float(np.median(v))) for k, v in vstat.items()}),
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(f"[score] saved {sum(len(x) for x in cols['t'])} tokens; skipped {dict(n_skip)}", flush=True)


# ------------------------------------------------------------------------------------------------ analyze (CPU)
def part_analyze(args):
    z = dict(np.load(os.path.join(args.out, "tokens.npz"))); meta = json.load(open(os.path.join(args.out, "meta.json")))
    z = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in z.items()}
    say = lambda s="": print(s, flush=True); L = meta["layers"]; res = dict(meta=meta)
    say(f"== {len(z['t'])} token / {len(np.unique(z['row']))} 题; 跳过/计数 {meta['skipped']}")
    v = meta["views"]
    say(f"   视图 prompt 长度 (中位): 学生 {v['S_prompt']['median']:.0f} (图像 token {v['S_img']['median']:.0f}) | "
        f"教师裁剪图 {v['C_prompt']['median']:.0f} (图像 token {v['C_img']['median']:.0f}) | 错配裁剪图 (同尺寸) {v['M_prompt']['median']:.0f}")
    # 1. per layer
    res["layer"] = {}
    say("\n== 逐层: 原始余弦 / 归一化距离 nd=||a−b||²/E||h_S−μ_S||² (nd 越小越像; 两个独立学生视图 token 约 2)"
        "\n   SC=学生视图 vs 教师视图 (同一 base 权重) SM=学生 vs 错配裁剪图 CM=教师 vs 错配 S2C=训练后学生 vs base 教师"
        "\n   S2C2=训练后模型自己两视图 SS2=学生训练前后; 内容份额=(nd_SM−nd_SC)/nd_SM")
    say("   层   " + "  ".join(f"{p:>13s}" for p in PAIRS) + "   内容份额")
    for li, l in enumerate(L):
        row = {p: (float(z[f"cos_{p}"][:, li].mean()), float(z[f"nd_{p}"][:, li].mean())) for p in PAIRS}
        dSC, dSM = row["SC"][1], row["SM"][1]; row["content_share"] = float((dSM - dSC) / dSM) if dSM > 0 else float("nan")
        res["layer"][l] = row
        say(f"   L{l:<3d} " + "  ".join(f"{row[p][0]:.3f}/{row[p][1]:6.3f}" for p in PAIRS) + f"   {row['content_share']:+.3f}")
    # 2. by position
    bm = bucket_masks(z["t"], z["commit"]); show = [l for l in (8, 16, 24, 31) if l in L]
    res["pos"] = {}
    say("\n== 按位置: nd  SC → S2C (训练后学生), 括号内 SM 与内容份额; 层 " + ", ".join(f"L{l}" for l in show))
    for name, m in bm.items():
        if not m.any(): continue
        d = {}
        for l in show:
            k = L.index(l); sc, s2c, sm = (float(z[f"nd_{p}"][m, k].mean()) for p in ("SC", "S2C", "SM"))
            d[l] = dict(SC=sc, S2C=s2c, SM=sm, content_share=(sm - sc) / sm if sm > 0 else float("nan"))
        d["n"] = int(m.sum()); res["pos"][name] = d
        say(f"   {name:14s} n={m.sum():6d} " + " | ".join(f"L{l} {d[l]['SC']:.3f}→{d[l]['S2C']:.3f} (SM {d[l]['SM']:.3f}, 内容 {d[l]['content_share']:+.2f})" for l in show))
    # 3. does the output-level gap see the hidden gap? within-trajectory Spearman (t>=1), averaged
    from scipy.stats import spearmanr
    j = z["jsd_SC"]; keep = z["t"] >= 1; traj = z["row"] * 100 + z["s"]
    res["out_vs_hidden"] = {}
    for li, l in enumerate(L):
        rs = []
        for tr in np.unique(traj[keep]):
            m = keep & (traj == tr)
            if m.sum() >= 10: rs.append(spearmanr(j[m], z["nd_SC"][m, li]).correlation)
        res["out_vs_hidden"][l] = dict(within=float(np.nanmean(rs)), n_traj=len(rs))
        lo_j = keep & (j < np.quantile(j[keep], 0.25)); hi_h = keep & (z["nd_SC"][:, li] > np.quantile(z["nd_SC"][keep, li], 0.75))
        res["out_vs_hidden"][l]["low_jsd_high_hidden"] = float(hi_h[lo_j].mean())
    say("\n== 输出差异 (JSD S‖C = 训练时的蒸馏信号) vs hidden 差异 (nd_SC), t≥1: 轨迹内 Spearman 均值 | 输出最一致 25% 的 token 中 hidden 差异最高 25% 的比例 (独立 0.25)")
    say("   " + " | ".join(f"L{l} {res['out_vs_hidden'][l]['within']:.3f}/{res['out_vs_hidden'][l]['low_jsd_high_hidden']:.3f}" for l in L))
    res["jsd"] = dict(SC=float(j.mean()), SM=float(z["jsd_SM"].mean()), S2C=float(z["jsd_S2C"].mean()))
    say(f"   平均 JSD: S‖C {res['jsd']['SC']:.4f} | S‖M (错配) {res['jsd']['SM']:.4f} | 训练后 S2‖C {res['jsd']['S2C']:.4f}")
    json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    say("\n[analyze] saved summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["score", "analyze"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gen"); ap.add_argument("--trained"); ap.add_argument("--n", type=int, default=600); ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--K", type=int, default=100); ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--warm", type=int, default=20)
    args = ap.parse_args()
    {"score": part_score, "analyze": part_analyze}[args.part](args)


if __name__ == "__main__":
    main()
