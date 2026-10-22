#!/usr/bin/env python
"""Teacher recoverability probe (user 10-21). Label-free: no GT, distribution quantities only.

Question: the off6k scoring teacher sees ONLY the crop (+ the student's text), the student sees the full red-box image. When
the student talks about something outside the crop ("the watermelon"), the crop teacher cannot see it, so its disagreement
there is "teacher can't see" rather than "student is wrong". How much of the distillation signal is of that kind, where is
it, and did training (base -> trained ckpt) push those tokens down?

Per student trajectory (base model's own T=1.0 samples from probe_hidden_answer gen.json), teacher-forced scoring by the
frozen base under
  C  = training teacher: student text, image -> crop                         (off6k _swap_images_in_messages)
  F  = superset teacher: student text, images -> [red-box full image][crop]  (sees everything the student sees + the crop)
  S  = the student itself (base, student view) and S' = the trained ckpt on the same student view (training effect)
Per token: JSD_alpha=0.5 on the student's top-K + tail bucket (= training's distillation_topk=100, alpha=0.5) between S and C,
S and F; log p of the sampled token under every view.
  gap+ = relu(jsd_SC - jsd_SF)  signal the crop teacher adds that the superset teacher does not ("can't see")
  gap- = relu(jsd_SF - jsd_SC)  the reverse (control: view-change noise in the other direction)
用法: python scripts/probe_teacher_recover.py --part score --gen <base gen.json> --trained <hf> --out <dir> [--n 600 --per 2]
      python scripts/probe_teacher_recover.py --part analyze --out <dir>
"""
import argparse, collections, json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import probe_hidden_answer as H
import probe_traj_within as W

REL = re.compile(r"\b(left|right|above|below|behind|in front|next to|between|near|nearest|closest|farthest|beside|relative|"
                 r"direction|side of|on top|under|beneath|adjacent|outside)\b", re.I)


# ------------------------------------------------------------------------------------------------ pure helpers (unit-tested)
def jsd_topk(ls_s, ls_t, idx, alpha=0.5):
    """JSD_alpha between student and teacher on the student's top-K support + one tail bucket (log-probs in, nats out)."""
    import torch
    def coarse(ls):
        top = ls.gather(-1, idx)
        tail = torch.log1p(-top.exp().sum(-1, keepdim=True).clamp(max=1 - 1e-6))
        return torch.cat([top, tail], -1)
    a, b = coarse(ls_s), coarse(ls_t)
    m = torch.logsumexp(torch.stack([a + np.log(alpha), b + np.log(1 - alpha)]), 0)
    kl = lambda p, q: (p.exp() * (p - q)).sum(-1)
    return alpha * kl(a, m) + (1 - alpha) * kl(b, m)


def superset_content(content: str) -> str:
    """Student text with its single <image> doubled: [red-box full image][crop]."""
    if content.count("<image>") != 1: raise ValueError(f"expected one <image>, got {content.count('<image>')}")
    return content.replace("<image>", "<image><image>", 1)


def gap_fractions(sc, sf):
    sc, sf = np.asarray(sc, float), np.asarray(sf, float); tot = sc.sum()
    return dict(gap_pos=float(np.maximum(sc - sf, 0).sum() / tot), gap_neg=float(np.maximum(sf - sc, 0).sum() / tot),
                mean_sc=float(sc.mean()), mean_sf=float(sf.mean()))


# ------------------------------------------------------------------------------------------------ score (GPU)
def part_score(args):
    import torch
    from transformers import AutoModelForImageTextToText
    import probe_layer_prior as P
    processor = H.setup_processor(); ps = processor.image_processor.patch_size; tok = processor.tokenizer
    gen = {g["row"]: g for g in json.load(open(args.gen))}
    rows = [r for r in H.load_rows(args.n) if r["_row"] in gen]
    mk = lambda p: AutoModelForImageTextToText.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    base, trained = mk(H.BASE), mk(args.trained)
    views = P.Views(processor, 12288, ps)
    cols, n_skip, rng = collections.defaultdict(list), collections.Counter(), np.random.RandomState(42)
    for ri, r in enumerate(rows):
        g = gen[r["_row"]]; content = r["prompt"][0]["content"]
        try:
            full, crop = H.student_images(r, ps)[0], P.load_rgb(r["bbox_images"][0]["path"])   # full = the image exactly as the student sees it
            enc = dict(S=views.student(r), C=views.encode(P.messages_from_template(content, [crop]), [crop]),
                       F=views.encode(P.messages_from_template(superset_content(content), [full, crop]), [full, crop]))
        except (RuntimeError, ValueError) as e:
            print(f"[score] skip row {r['_row']}: {e}", flush=True); n_skip["row"] += 1; continue
        otext = W.option_texts(content); rel = bool(REL.search(content.split("\n\n")[0]))
        for si in sorted(rng.permutation(len(g["texts"]))[:args.per].tolist()):
            text, let = g["texts"][si], g["letters"][si]
            y = tok(text, add_special_tokens=False)["input_ids"][:args.max_len]
            if len(y) < 2: n_skip["short"] += 1; continue
            t = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ci = W.commit_char(text, let, otext.get(let, "")) if let else None
            ct = W.char_to_token(t["offset_mapping"], ci) if ci is not None else -1   # -1 = commit unknown: excluded from pre/post buckets
            with torch.no_grad():
                ls = {v: torch.log_softmax(P.score(base, enc[v], y, 1.0), -1) for v in "SCF"}
                ls["S2"] = torch.log_softmax(P.score(trained, enc["S"], y, 1.0), -1)
                idx = ls["S"].topk(args.K, -1).indices
                yi = torch.tensor(y, device=ls["S"].device).unsqueeze(-1)
                rec = dict(jsd_SC=jsd_topk(ls["S"], ls["C"], idx), jsd_SF=jsd_topk(ls["S"], ls["F"], idx),
                           jsd_CF=jsd_topk(ls["F"], ls["C"], idx),          # on the STUDENT's top-K support (not F's own)
                           **{f"l{v}": ls[v].gather(-1, yi).squeeze(-1) for v in ("S", "C", "F", "S2")},
                           ent_S=-(ls["S"].exp() * ls["S"]).sum(-1))
                for k, v in rec.items(): cols[k].append(v.float().cpu().numpy())
            n = len(y)
            cols["row"].append(np.full(n, r["_row"])); cols["s"].append(np.full(n, si)); cols["t"].append(np.arange(n))
            cols["tok"].append(np.array(y)); cols["commit"].append(np.full(n, ct)); cols["rel"].append(np.full(n, rel))
            del ls
        if (ri + 1) % 50 == 0: print(f"[score] {ri+1}/{len(rows)}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "tokens.npz"), **{k: np.concatenate(v) for k, v in cols.items()})
    json.dump(dict(skipped=dict(n_skip), n_rows=len(rows), trained=args.trained, K=args.K), open(os.path.join(args.out, "meta.json"), "w"))
    print(f"[score] saved {sum(len(x) for x in cols['t'])} tokens; skipped {dict(n_skip)}", flush=True)


# ------------------------------------------------------------------------------------------------ analyze (CPU)
def part_analyze(args):
    from transformers import AutoTokenizer
    z = dict(np.load(os.path.join(args.out, "tokens.npz"))); meta = json.load(open(os.path.join(args.out, "meta.json")))
    say = lambda s="": print(s, flush=True); res = dict(meta=meta)
    sc, sf = z["jsd_SC"], z["jsd_SF"]; gp = np.maximum(sc - sf, 0)
    known = z["commit"] >= 0; pre = known & (z["t"] < z["commit"]); post = known & ~pre; early = z["t"] < 8
    res["all"] = gap_fractions(sc, sf)
    say(f"== {len(sc)} token / {len(np.unique(z['row']))} 题; 跳过 {meta['skipped']}; 关系类题目占比 {np.mean([z['rel'][z['row'] == r][0] for r in np.unique(z['row'])]):.3f}")
    say(f"   蒸馏信号 (JSD 学生 vs 教师, top-{meta['K']}+尾): 只看裁剪图教师 C {res['all']['mean_sc']:.4f} | 全图+裁剪图教师 F {res['all']['mean_sf']:.4f}")
    say(f"   C 的信号中 '看不到' 份额 gap+ = Σrelu(C−F)/ΣC {res['all']['gap_pos']:.3f} | 反向对照 gap− = Σrelu(F−C)/ΣC {res['all']['gap_neg']:.3f}"
        f" | 净 {res['all']['gap_pos'] - res['all']['gap_neg']:+.3f}")
    say("\n== 按位置 / 题型: token 占比 | C 信号占比 | gap+ | gap− | 净")
    buckets = {"前 8 token": early, "承诺前 (≥8)": pre & ~early, "承诺后": post, "承诺位置未知": ~known, "关系类题": z["rel"].astype(bool), "非关系类题": ~z["rel"].astype(bool)}
    res["buckets"] = {}
    for name, m in buckets.items():
        if not m.any(): continue
        f = gap_fractions(sc[m], sf[m]); f.update(tok_frac=float(m.mean()), sig_frac=float(sc[m].sum() / sc.sum())); res["buckets"][name] = f
        say(f"   {name:10s} {f['tok_frac']:.3f} | {f['sig_frac']:.3f} | {f['gap_pos']:.3f} | {f['gap_neg']:.3f} | {f['gap_pos'] - f['gap_neg']:+.3f}")
    # which tokens carry the "can't see" mass
    tok = AutoTokenizer.from_pretrained(H.BASE); mass = collections.Counter(); cnt = collections.Counter()
    for t_, g_ in zip(z["tok"], gp): mass[int(t_)] += float(g_); cnt[int(t_)] += 1
    top = mass.most_common(25)
    res["top_tokens"] = [(tok.decode([t]), round(m / gp.sum(), 4), cnt[t]) for t, m in top]
    say("\n== gap+ 质量最大的 token (占 gap+ 总量, 出现次数):")
    say("   " + "  ".join(f"{s!r} {m:.3f}/{c}" for s, m, c in res["top_tokens"]))
    # sampled-token view: crop teacher suppresses tokens the superset teacher accepts
    dl = z["lF"] - z["lC"]; sup = (dl > 2.0) & (z["lF"] > np.log(0.1))   # superset teacher p>0.1 and >= e^2 x the crop teacher
    res["suppressed"] = dict(frac=float(sup.mean()), frac_pre=float(sup[pre].mean()), frac_post=float(sup[post].mean()) if post.any() else float("nan"))
    say(f"\n== 被采样 token 中 '裁剪图教师压、全图教师不压' (lF−lC>2 且 pF>0.1): 全部 {sup.mean():.3f} | 承诺前 {res['suppressed']['frac_pre']:.3f} | 承诺后 {res['suppressed']['frac_post']:.3f}")
    # training effect. Training only ever sees S vs C, so its push on a token is a function of (lS - lC), not of F.
    # (a) raw: how much did the trained student lower the 'can't see' tokens, and what share of the total push-down is theirs;
    # (b) matched: within (lS - lC) decile bins, 'can't see' tokens vs tokens the crop and superset teachers agree on
    #     (|lF - lC| < 0.5). Null (training is blind to F) = 0; nonzero = something beyond the S-vs-C gap (review 10-21).
    d2 = z["lS2"] - z["lS"]; agree = np.abs(dl) < 0.5
    down = np.minimum(d2, 0)
    gapc = z["lS"] - z["lC"]; bins = np.digitize(gapc, np.quantile(gapc, np.linspace(0, 1, 11)[1:-1]))
    diffs, wts, per = [], [], []
    for b in np.unique(bins):
        mb = bins == b; a, c = mb & sup, mb & agree
        per.append((int(b), int(a.sum()), int(c.sum())))
        if a.sum() >= 20 and c.sum() >= 20: diffs.append(d2[a].mean() - d2[c].mean()); wts.append(min(a.sum(), c.sum()))
    res["train_effect"] = dict(sup=float(d2[sup].mean()) if sup.any() else float("nan"), agree=float(d2[agree].mean()), rest=float(d2[~sup].mean()),
                               sup_share_of_pushdown=float(down[sup].sum() / down.sum()) if down.sum() < 0 else float("nan"),
                               matched_diff=float(np.average(diffs, weights=wts)) if diffs else float("nan"), bins_n=per)
    te = res["train_effect"]
    say(f"   训练后学生 log p 变化 (step70 − base): '教师看不到' token {te['sup']:+.3f} | 两教师一致的 token {te['agree']:+.3f} | 其余 {te['rest']:+.3f}")
    say(f"   '教师看不到' token 占训练总下压量 {te['sup_share_of_pushdown']:.3f} (占 token {res['suppressed']['frac']:.3f}); "
        f"按 (lS−lC) 分桶匹配后与 '两教师一致' token 的差 {te['matched_diff']:+.3f} (训练看不到 F, 零假设 = 0)")
    json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    say("\n[analyze] saved summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["score", "analyze"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gen"); ap.add_argument("--trained"); ap.add_argument("--n", type=int, default=600); ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--K", type=int, default=100); ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()
    {"score": part_score, "analyze": part_analyze}[args.part](args)


if __name__ == "__main__":
    main()
