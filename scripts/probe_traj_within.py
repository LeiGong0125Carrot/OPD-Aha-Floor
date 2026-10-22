#!/usr/bin/env python
"""Within-group trajectory probe + teacher cross-table (user 10-21). Diagnostic only -- uses GT by the same explicit user
exception as probe_hidden_answer.py; nothing here feeds a selector or a training target.

Q1 (part=thidden/tanalyze): the 8 trajectories of one prompt share the pre-generation state, so prompt-level probes cannot
   tell which of them will be right (within-group AUC = 0.5 by construction). Per trajectory, take the hidden state after
   reading its first k response tokens (k = 0,1,2,4,8,16,32,64, half-way to commitment, and the token right before
   commitment) and ask: within a group with mixed outcomes, does that state rank the correct trajectories above the wrong
   ones? Commitment = first char where the response reveals its answer: the committed option letter as a standalone token OR
   the committed option's text (e.g. 'purple' before 'D'), whichever comes first; positions at/after it are excluded (a probe
   there just reads the text). Prompts = groups mixed (>=1 right and >=1 wrong parsed answer) in BOTH models' gens, sampled.
   Scores: (a) correctness probe (binary), (b) final-letter probe p(GT). Folds grouped by prompt.
Q2 (part=cross, CPU): teacher-view samples of the frozen base (probe_hidden_answer --part gen --view teacher) vs base and
   trained student samples, per question: does the trained student lose questions the base partly solved when the teacher
   is wrong there, and does it move toward the teacher's wrong letter?
用法: python scripts/probe_traj_within.py --part thidden --model <hf> --gen <dir>/gen.json --pair_gen <other>/gen.json --out <dir>
      python scripts/probe_traj_within.py --part tanalyze --out <dir>
      python scripts/probe_traj_within.py --part cross --base <dir> --trained <dir> --teacher <dir> --out <dir>
"""
import argparse, collections, json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import probe_hidden_answer as H

KS = (0, 1, 2, 4, 8, 16, 32, 64)
POS_NAMES = tuple(f"k{k}" for k in KS) + ("half", "pre1", "pre")   # pre = state that emits the commit token, pre1 = one before
LAYERS = (8, 12, 16, 20, 24, 28, 32)


# ------------------------------------------------------------------------------------------------ pure helpers (unit-tested)
def option_texts(question: str) -> dict:
    """{'A': 'black', ...} from 'A. black' lines."""
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^\s*([A-F])\.\s+(.+?)\s*$", question, re.M)}


def commit_char(text: str, letter: str, opt_text: str):
    """First char index where the response reveals `letter`: 'answer/option/choice (is|:) X'; a standalone letter token
    (for 'A' not the article 'A word', as parse_letter); or the option text as whole words (case-insensitive, >= 3 chars;
    may cut early when the option word also appears in another sense, e.g. 'red' in 'red bounding box' -- conservative).
    None if none occurs. thidden additionally drops trajectories whose kept prefix still parses to a letter."""
    tail = r"(?![A-Za-z]|\s+[a-z])" if letter == "A" else r"(?![A-Za-z])"
    c = [m.start() for m in [re.search(rf"(?<![A-Za-z]){letter}{tail}", text)] if m]
    m = re.search(rf"(?i:answer|option|choice)\s*(?:is|:)?\s*\**\(?{letter}(?![A-Za-z])", text)   # keyword case-insensitive, letter not (no "answer is a bit")
    if m: c.append(m.start())
    if opt_text and len(opt_text) >= 3:
        m = re.search(rf"(?<![A-Za-z0-9]){re.escape(opt_text)}(?![A-Za-z0-9])", text, re.I)
        if m: c.append(m.start())
    return min(c) if c else None


def char_to_token(offsets, ci: int) -> int:
    """Index of the token whose span contains char ci (first token ending after ci)."""
    for t, (a, b) in enumerate(offsets):
        if b > ci: return t
    return len(offsets)


def positions(commit_tok: int) -> dict:
    """Response-token counts k (state after reading k tokens; valid iff k <= commit_tok) per position name."""
    out = {f"k{k}": k for k in KS if k <= commit_tok}
    if commit_tok >= 2: out["half"] = commit_tok // 2; out["pre1"] = commit_tok - 1
    if commit_tok >= 1: out["pre"] = commit_tok
    return out


def within_auc(score, label, group):
    """Pair-weighted within-group AUC: over groups with both classes, sum_g AUC_g * n1_g * n0_g / sum_g n1_g * n0_g."""
    score, label, group = np.asarray(score, float), np.asarray(label, bool), np.asarray(group)
    num = den = 0.0
    for g in np.unique(group):
        m = group == g; n1 = label[m].sum(); n0 = m.sum() - n1
        if n1 == 0 or n0 == 0: continue
        num += H.auc(score[m], label[m]) * n1 * n0; den += n1 * n0
    return float(num / den) if den else float("nan")


def mixed(g: dict) -> bool:
    c = [x == g["gt"] for x in g["letters"] if x is not None]
    return any(c) and not all(c)


def select_rows(gen: dict, pair: dict, n: int, seed: int = 42) -> list:
    rows = sorted(r for r in gen if r in pair and mixed(gen[r]) and mixed(pair[r]))
    if n > 0 and len(rows) > n:
        rows = sorted(np.random.RandomState(seed).permutation(rows)[:n].tolist())
    return rows


def rate(g: dict) -> float:
    return float(np.mean([x == g["gt"] for x in g["letters"]]))         # unparsed = wrong


# ------------------------------------------------------------------------------------------------ thidden (GPU)
def part_thidden(args):
    import torch
    from transformers import AutoModelForImageTextToText
    import probe_layer_prior as P
    processor = H.setup_processor(); ps = processor.image_processor.patch_size; tok = processor.tokenizer
    gen = {g["row"]: g for g in json.load(open(args.gen))}; pair = {g["row"]: g for g in json.load(open(args.pair_gen))}
    sel = set(select_rows(gen, pair, args.n_prompts)); rows = [r for r in H.load_rows(0) if r["_row"] in sel]
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    views = P.Views(processor, 10240, ps); L1 = model.config.get_text_config().num_hidden_layers + 1
    assert max(LAYERS) < L1, f"layers {LAYERS} vs {L1} boundaries"
    n_tr = sum(len(gen[r["_row"]]["texts"]) for r in rows)
    os.makedirs(args.out, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(args.out, "traj_hidden.tmp.npy"), mode="w+", dtype=np.float16,
                                  shape=(n_tr, len(POS_NAMES), len(LAYERS), model.config.get_text_config().hidden_size))
    meta, j, n_skip = [], 0, collections.Counter()
    for ri, r in enumerate(rows):
        g = gen[r["_row"]]; otext = option_texts(r["prompt"][0]["content"])
        try:
            enc = views.student(r)
        except (RuntimeError, ValueError) as e:
            print(f"[thidden] skip row {r['_row']}: {e}", flush=True); n_skip["row"] += 1; continue
        Pn = enc["input_ids"].shape[1]; enc_g = {k: v.cuda() for k, v in enc.items() if k != "_raw"}
        for si, (text, let) in enumerate(zip(g["texts"], g["letters"])):
            if let is None: n_skip["unparsed"] += 1; continue
            ci = commit_char(text, let, otext.get(let, "")); ci_l = commit_char(text, let, "")
            if ci is None: n_skip["no_commit"] += 1; continue
            t = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ct = char_to_token(t["offset_mapping"], ci); pos = positions(ct)
            if ct > 0 and H.parse_letter(text[:t["offset_mapping"][ct - 1][1]], g["opts"]) == let:
                n_skip["prefix_reveals"] += 1; continue                      # commit_char missed an earlier reveal
            resp = torch.tensor(t["input_ids"][:ct], dtype=enc["input_ids"].dtype).unsqueeze(0)
            inp = dict(enc_g)
            inp["input_ids"] = torch.cat([inp["input_ids"], resp.cuda()], 1)
            inp["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(resp).cuda()], 1)
            if "mm_token_type_ids" in inp:
                inp["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(resp).cuda()], 1)
            with torch.no_grad():
                hs = model(**inp, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
            idx = [Pn - 1 + pos.get(p, 0) for p in POS_NAMES]                 # invalid positions filled with k=0, masked out
            v = torch.stack([hs[l][0, idx] for l in LAYERS], 1).float()       # [pos, layers, D]
            if not torch.isfinite(v).all(): raise RuntimeError(f"non-finite hidden at row {r['_row']} sample {si}")
            v16 = v.cpu().numpy().astype(np.float16)
            if not np.isfinite(v16).all(): raise RuntimeError(f"fp16 overflow at row {r['_row']} sample {si}")
            X[j] = v16
            meta.append(dict(row=r["_row"], s=si, letter=let, gt=g["gt"], opts=g["opts"], correct=bool(let == g["gt"]), via_text=bool(ci_l is None or ci < ci_l),
                             commit_tok=int(ct), valid=[p in pos for p in POS_NAMES])); j += 1
        if (ri + 1) % 100 == 0: print(f"[thidden] {ri+1}/{len(rows)} prompts, {j} traj", flush=True)
    np.save(os.path.join(args.out, "traj_hidden.npy"), np.asarray(X[:j])); del X; os.remove(os.path.join(args.out, "traj_hidden.tmp.npy"))
    json.dump(dict(model=args.model, layers=LAYERS, pos=POS_NAMES, skipped=dict(n_skip), traj=meta), open(os.path.join(args.out, "traj_meta.json"), "w"))
    ct = np.array([m["commit_tok"] for m in meta])
    print(f"[thidden] saved {j} traj from {len(rows)} prompts; skipped {dict(n_skip)}; commit_tok==0 (answer-first) {np.mean(ct == 0):.3f}, "
          f"commit via option text {np.mean([m['via_text'] for m in meta]):.3f}, "
          f"median commit_tok {np.median(ct):.0f}", flush=True)


# ------------------------------------------------------------------------------------------------ tanalyze
def part_tanalyze(args):
    meta = json.load(open(os.path.join(args.out, "traj_meta.json"))); T = meta["traj"]
    X = np.load(os.path.join(args.out, "traj_hidden.npy"))
    say = lambda s="": print(s, flush=True)
    grp = np.array([t["row"] for t in T]); corr = np.array([t["correct"] for t in T]).astype(int)
    V = np.array([t["valid"] for t in T]); ct = np.array([t["commit_tok"] for t in T])
    M = np.zeros((len(T), 6), bool); yl = np.array([H.LETTERS.index(t["letter"]) for t in T]); gt = np.array([H.LETTERS.index(t["gt"]) for t in T])
    for i, t in enumerate(T):
        for o in t["opts"]: M[i, H.LETTERS.index(o)] = True
    folds = H.group_folds(grp.tolist()); two = np.ones((len(T), 2), bool)
    res = dict(via_text=float(np.mean([t.get("via_text", False) for t in T])), n_traj=len(T), n_prompts=int(len(np.unique(grp))), answer_first=float(np.mean(ct == 0)), median_commit_tok=float(np.median(ct)),
               skipped=meta["skipped"], sweep={}, best={})
    say(f"== {len(T)} 条轨迹 / {res['n_prompts']} 题 (两模型都是有对有错的题); 正确率 {corr.mean():.3f}; 先作答 (commit_tok=0) {res['answer_first']:.3f}; "
        f"承诺位置中位数 {res['median_commit_tok']:.0f} token; 跳过 {meta['skipped']}")
    say("   每格: 组内 AUC 对错探针 / 组内 AUC 字母探针 p(GT) / 字母探针预测最终字母准确率   (组内 AUC 只在该位置仍有对有错的题上算)")
    for pi, pn in enumerate(meta["pos"]):
        v = V[:, pi]; ug, inv = np.unique(grp, return_inverse=True); vg = np.bincount(inv, weights=v)[inv]   # keep groups with >= 2 valid traj
        m = v & (vg >= 2)
        if m.sum() < 50: say(f"   {pn:5s} n={m.sum()} 太少, 跳过"); continue
        say(f"   {pn:5s} n={m.sum():5d} 题={len(np.unique(grp[m])):4d}  " + "  ".join(f"L{l}" for l in meta["layers"]))
        cells = []
        for li, l in enumerate(meta["layers"]):
            Xl = X[m, pi, li].astype(np.float32)
            Pc = H.cv_probe(Xl, corr[m], two[m], folds[m], 2); Pl = H.cv_probe(Xl, yl[m], M[m], folds[m], 6)
            r = dict(win_corr=within_auc(Pc[:, 1], corr[m], grp[m]), win_pgt=within_auc(Pl[np.arange(m.sum()), gt[m]], corr[m], grp[m]),
                     letter_acc=float((Pl.argmax(1) == yl[m]).mean()), pooled_corr=H.auc(Pc[:, 1], corr[m]), n=int(m.sum()))
            res["sweep"][f"{pn}/L{l}"] = r; cells.append(f"{r['win_corr']:.3f}/{r['win_pgt']:.3f}/{r['letter_acc']:.2f}")
        say("         " + "  ".join(cells))
        lb = max(meta["layers"], key=lambda l: np.nan_to_num(res["sweep"][f"{pn}/L{l}"]["win_corr"], nan=-1))
        # shuffled control: max over the same layers (matches the max-over-layers selection of the real probe)
        sh = np.nanmax([within_auc(H.cv_probe(X[m, pi, li].astype(np.float32), corr[m], two[m], folds[m], 2, shuffle=True)[:, 1], corr[m], grp[m])
                 for li in range(len(meta["layers"]))])
        # length baseline: commit_tok alone (the state at half/pre1/pre encodes position); report the stronger direction
        la = within_auc(ct[m], corr[m], grp[m])
        res["best"][pn] = dict(layer=lb, win_corr=res["sweep"][f"{pn}/L{lb}"]["win_corr"], shuffled=sh, len_auc=max(la, 1 - la),
                               win_pgt=res["sweep"][f"{pn}/L{lb}"]["win_pgt"], letter_acc=res["sweep"][f"{pn}/L{lb}"]["letter_acc"], n=int(m.sum()))
    say(f"\n== 各位置最优层 (按对错探针选): 组内 AUC 对错探针 | 打乱对照 (各层最大) | 仅长度 commit_tok 基线 | 同层组内 AUC p(GT) | 同层最终字母准确率;  承诺点来自选项文字 {res['via_text']:.3f}")
    for pn, b in res["best"].items():
        say(f"   {pn:5s} L{b['layer']:2d}  {b['win_corr']:.3f} | 打乱 {b['shuffled']:.3f} | 长度 {b['len_auc']:.3f} | {b['win_pgt']:.3f} | {b['letter_acc']:.3f}  (n={b['n']})")
    json.dump(res, open(os.path.join(args.out, "traj_summary.json"), "w"), indent=1)
    say("\n[tanalyze] saved traj_summary.json")


# ------------------------------------------------------------------------------------------------ cross (CPU)
def rate_of(g: dict, idx) -> float:
    return float(np.mean([g["letters"][i] == g["gt"] for i in idx]))         # unparsed = wrong


def cross_table(base: dict, trained: dict, teacher: dict) -> dict:
    """Regression-to-the-mean control (review 10-21): base samples 0-3 assign the bin, base samples 4-7 are the null
    outcome (same model, fresh samples) and the trained student's 8 samples the treatment outcome."""
    rows = sorted(r for r in set(base) & set(trained) & set(teacher) if len(base[r]["letters"]) >= 8)
    rb = np.array([rate_of(base[r], range(4)) for r in rows]); rn = np.array([rate_of(base[r], range(4, 8)) for r in rows])
    rs = np.array([rate(trained[r]) for r in rows]); rt = np.array([rate(teacher[r]) for r in rows])
    out = dict(n=len(rows), teacher_acc=float(rt.mean()), base_acc=float(np.mean([rate(base[r]) for r in rows])), trained_acc=float(rs.mean()), cells={})
    tb = {"教师多数对(≥0.5)": rt >= 0.5, "教师多数错(<0.5)": rt < 0.5, "教师全错(0)": rt == 0}
    bb = {"base前4全错": rb == 0, "base前4部分对": (rb > 0) & (rb < 1), "base前4全对": rb == 1}
    f = lambda x, m: float(x[m].mean()) if m.any() else float("nan")
    for tn, tm in tb.items():
        for bn, bm in bb.items():
            m = tm & bm
            out["cells"][f"{tn}|{bn}"] = dict(n=int(m.sum()), bin=f(rb, m), base_null=f(rn, m), trained=f(rs, m), effect=f(rs - rn, m))
    # among WRONG samples: share on the teacher's wrong majority letter, base vs trained, vs uniform over wrong options
    pb, ps, uni, n = [], [], [], 0
    for r in rows:
        tmj = H.majority(teacher[r]["letters"]); gt = teacher[r]["gt"]
        if tmj is None or tmj == gt: continue
        wb = [x for x in base[r]["letters"] if x is not None and x != gt]; ws = [x for x in trained[r]["letters"] if x is not None and x != gt]
        if not wb or not ws: continue
        n += 1; pb.append(np.mean([x == tmj for x in wb])); ps.append(np.mean([x == tmj for x in ws]))
        uni.append(1 / max(len(base[r]["opts"]) - 1, 1))
    out["teacher_wrong_letter"] = dict(n=n, base_share=float(np.mean(pb)) if n else float("nan"),
                                       trained_share=float(np.mean(ps)) if n else float("nan"), uniform=float(np.mean(uni)) if n else float("nan"))
    big = (rs - rb) <= -0.5; elig = rb >= 0.5    # same statistic as the null (rn - rb): both carry the same regression to the mean
    out["big_drops"] = dict(n=int((big & elig).sum()), teacher_wrong_frac=f((rt < 0.5).astype(float), big & elig),
                            teacher_wrong_frac_eligible=f((rt < 0.5).astype(float), elig),
                            null_n=int((((rn - rb) <= -0.5) & elig).sum()))
    return out


def part_cross(args):
    ld = lambda d: {g["row"]: g for g in json.load(open(os.path.join(d, "gen.json")))}
    res = cross_table(ld(args.base), ld(args.trained), ld(args.teacher)); say = lambda s="": print(s, flush=True)
    say(f"== {res['n']} 题; 单条采样正确率: 教师视图 {res['teacher_acc']:.3f} | base 学生 {res['base_acc']:.3f} | 训练后学生 {res['trained_acc']:.3f}")
    say("   分桶用 base 前 4 条采样; 零假设 = base 后 4 条 (同模型重采样, 吸收均值回归); 训练效应 = 训练后 8 条 − base 后 4 条")
    say("   教师 × base 桶: n | 桶内 base 前4 | base 后4 (零假设) | 训练后 | 训练效应")
    for k, c in res["cells"].items():
        say(f"   {k:30s} n={c['n']:5d} | {c['bin']:.3f} | {c['base_null']:.3f} | {c['trained']:.3f} | {c['effect']:+.3f}")
    w = res["teacher_wrong_letter"]
    say(f"   教师多数答案是错字母的题 n={w['n']}: 学生答错时选这个字母的占比 base {w['base_share']:.3f} → 训练后 {w['trained_share']:.3f} (错选项均匀 {w['uniform']:.3f})")
    b = res["big_drops"]
    say(f"   大幅退步 (训练后比 base 前4 降 ≥0.5, base 前4 ≥0.5) n={b['n']} (零假设 base 后4 比前4 降 ≥0.5: {b['null_n']}): "
        f"其中教师多数错 {b['teacher_wrong_frac']:.3f} (同条件全体 {b['teacher_wrong_frac_eligible']:.3f})")
    say("   注: 教师自生成 ≠ 训练信号 (训练时教师在学生前缀上 teacher-forcing 打分, 10-17 探针: 前缀主导 0.51-0.84)")
    os.makedirs(args.out, exist_ok=True); json.dump(res, open(os.path.join(args.out, "cross_summary.json"), "w"), indent=1)
    say("\n[cross] saved cross_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["thidden", "tanalyze", "cross"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=H.BASE); ap.add_argument("--gen"); ap.add_argument("--pair_gen"); ap.add_argument("--n_prompts", type=int, default=1500)
    ap.add_argument("--base"); ap.add_argument("--trained"); ap.add_argument("--teacher")
    args = ap.parse_args()
    {"thidden": part_thidden, "tanalyze": part_tanalyze, "cross": part_cross}[args.part](args)


if __name__ == "__main__":
    main()
