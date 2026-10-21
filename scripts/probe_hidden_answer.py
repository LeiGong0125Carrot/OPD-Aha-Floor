#!/usr/bin/env python
"""Hidden-state answer probe (user 10-20). Diagnostic only -- uses GT labels by explicit user exception to the 10-06
no-correctness-label rule; nothing here feeds a selector or a training target.

Question: many off6k trajectories commit to the answer at t=0. Is the answer (GT letter / the model's own sampled answer /
whether it will be right) already linearly readable from the hidden states BEFORE generation, and does any layer know more
than the output distribution says?

Setting: official Vision-OPD-6K (data/train.parquet), student view (red-box image + hint text), frozen model given by --model.
Positions: last visual token (last <|image_pad|>; the question/options are not visible yet), the last prompt token
(after the generation prompt; its next-token distribution IS the t=0 answer distribution when the model answers first), and
the last prompt token of a TEXT-ONLY view of the same prompt (control: how much of the GT is decodable from the question and
option text alone -- text priors / letter skew -- without the image).

part=gen     (vLLM)  n samples per row at temperature 1.0 (seed 42), parsed to option letters
part=hidden  (HF)    per row one image forward (vis + last) and one text-only forward (txt): hidden states of every layer
                     boundary (fp32) + log-probs of the option-letter tokens at the last prompt token (the model's own t=0 readout)
part=analyze (torch) 5-fold CV linear probes (folds grouped by image path; in off6k every row has its own image, so this is
                     plain 5-fold) per (position, layer) for four targets:
                     gt  = GT letter (label)          maj = model's majority sampled letter (label-free)
                     acc = >=half of samples correct  det = all parsed samples agree (label-free)
                     baselines = model's own t=0 letter readout, GT-letter prior, chance; shuffled-label control;
                     per-trajectory correctness AUC from per-prompt scores. Note: the best layer is picked from the same CV
                     predictions it is reported on (max over layers, ~+1 pt optimism); the shuffled control runs at that layer.
用法: python scripts/probe_hidden_answer.py --part gen|hidden|analyze --model <hf path> --out eval/probe_hidden/<id> [--n 0]
"""
import argparse, collections, json, math, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))

PARQUET = "/scratch/nkw3mr/Vision-OPD/data/train.parquet"
BASE = "Qwen/Qwen3.5-4B"
TPL = os.path.join(ROOT, "chat_templates/perception_chat_template_qwen35.jinja")
LETTERS = "ABCDEF"
POSITIONS = ("vis", "last", "txt")   # last visual token, last prompt token, last prompt token of the text-only view
TARGETS = ("gt", "maj", "acc", "det")


# ------------------------------------------------------------------------------------------------ pure helpers (unit-tested)
def parse_options(text: str) -> list:
    """Option letters offered in the question ('A. black' lines), in order, deduplicated."""
    return list(dict.fromkeys(re.findall(r"^\s*([A-F])\.\s", text, re.M)))


def parse_letter(text: str, options: list):
    """Option letter the response commits to, or None. Priority: <answer>X</answer>; a leading bare letter ('D', 'D.',
    '(D)'); the last 'answer is X' / 'option X'; otherwise a unique standalone option letter (not the article 'A word')."""
    cands = []
    m = re.search(r"<answer>\s*\(?([A-F])(?![A-Za-z])", text)
    if m: cands.append(m.group(1))
    m = re.match(r"\s*\(?([A-F])(?:[).:,]|\s*$|\s*\n)", text)
    if m: cands.append(m.group(1))
    ms = re.findall(r"(?:answer|option)\s*(?:is|:)?\s*\**\(?([A-F])(?![A-Za-z])", text, re.I)
    if ms: cands.append(ms[-1])
    if not cands:
        alone = set(re.findall(r"(?<![A-Za-z])([A-F])(?![A-Za-z]|\s+[a-z])", text))
        if len(alone) == 1: cands.append(alone.pop())
    for c in cands:
        if c in options:
            return c
    return None


def majority(letters: list):
    """Most frequent non-None letter; None on a tie for first place or when nothing parsed."""
    c = collections.Counter(x for x in letters if x is not None).most_common()
    if not c or (len(c) > 1 and c[0][1] == c[1][1]):
        return None
    return c[0][0]


def group_folds(groups: list, k: int = 5, seed: int = 42) -> np.ndarray:
    """Fold id per row; all rows of a group share a fold (groups shuffled with a fixed seed, dealt round-robin by size)."""
    keys = sorted(set(groups)); rng = np.random.RandomState(seed); rng.shuffle(keys)
    size = collections.Counter(groups); load = [0] * k; fold_of = {}
    for g in sorted(keys, key=lambda g: -size[g]):        # stable: shuffled order, then largest first
        f = int(np.argmin(load)); fold_of[g] = f; load[f] += size[g]
    return np.array([fold_of[g] for g in groups])


def auc(score, label):
    """ROC AUC via the rank-sum statistic (ties get average ranks); nan if one class is empty."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(score, kind="mergesort"); r = np.empty(len(score)); s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_predict(Xtr, ytr, mtr, Xte, mte, lam, n_cls, iters=100):
    """Multinomial logistic regression (torch LBFGS, L2 = lam * ||W||^2 / 2, features standardized on the train split).
    m* = bool [N, n_cls] valid-class masks (invalid options get -inf). Returns test log-probs [N_te, n_cls]."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=dev); Xte = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6); Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    y = torch.as_tensor(ytr, device=dev); Mtr = torch.as_tensor(mtr, device=dev); Mte = torch.as_tensor(mte, device=dev)
    W = torch.zeros(Xtr.shape[1], n_cls, device=dev, requires_grad=True); b = torch.zeros(n_cls, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        z = (Xtr @ W + b).masked_fill(~Mtr, -1e9)
        loss = torch.nn.functional.cross_entropy(z, y) + 0.5 * lam * (W * W).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        return torch.log_softmax((Xte @ W + b).masked_fill(~Mte, -1e9), -1).cpu().numpy()


def cv_probe(X, y, M, folds, n_cls, lams=(1e-3, 1e-2, 1e-1, 1.0), shuffle=False, seed=0):
    """Cross-fitted log-probs for every row. lam picked per outer fold on an inner held-out fold (by CE); shuffle permutes the
    training labels (control). Rows with y < 0 are excluded (returned as nan)."""
    out = np.full((len(y), n_cls), np.nan); ok = y >= 0; rng = np.random.RandomState(seed)
    for f in np.unique(folds):
        tr, te = ok & (folds != f), ok & (folds == f)
        if not te.any() or not tr.any(): continue
        ytr = y[tr].copy()
        if shuffle: ytr = ytr[rng.permutation(len(ytr))]
        Mtr = M[tr] if not shuffle else np.ones_like(M[tr])
        inner = folds[tr]; fi = np.unique(inner)[0]; a, v = inner != fi, inner == fi
        best = min(lams, key=lambda lam: -np.mean(fit_predict(X[tr][a], ytr[a], Mtr[a], X[tr][v], Mtr[v], lam, n_cls)[np.arange(v.sum()), ytr[v]]))
        out[te] = fit_predict(X[tr], ytr, Mtr, X[te], M[te], best, n_cls)
    return out


# ------------------------------------------------------------------------------------------------ data
def load_rows(n: int):
    import pyarrow.parquet as pq
    rows = pq.read_table(PARQUET).to_pylist()
    idx = range(len(rows)) if n <= 0 else sorted(np.random.RandomState(42).permutation(len(rows))[:n].tolist())
    out = []
    for i in idx:
        r = rows[i]; r["_row"] = i
        rm = r["reward_model"]; rm = rm if isinstance(rm, dict) else eval(rm)
        r["_gt"] = str(rm["ground_truth"]).strip()
        r["_opts"] = parse_options(r["prompt"][0]["content"])
        out.append(r)
    return out


def setup_processor():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(BASE); tpl = open(TPL).read()
    processor.chat_template = tpl; processor.tokenizer.chat_template = tpl
    return processor


def student_images(r, ps):
    from verl.utils.dataset.vision_utils import process_image
    return [process_image(dict(d), image_patch_size=ps) for d in r["images"]]


# ------------------------------------------------------------------------------------------------ gen / hidden (GPU)
def part_gen(args):
    from vllm import LLM, SamplingParams
    import probe_layer_prior as P
    processor = setup_processor(); ps = processor.image_processor.patch_size
    rows = load_rows(args.n); reqs = []
    for r in rows:
        imgs = student_images(r, ps)
        reqs.append({"prompt": processor.apply_chat_template(P.messages_from_template(r["prompt"][0]["content"], imgs), tokenize=False, add_generation_prompt=True),
                     "multi_modal_data": {"image": imgs}})
    llm = LLM(model=args.model, tokenizer=BASE, gpu_memory_utilization=0.6, max_model_len=12288, limit_mm_per_prompt={"image": 1},
              trust_remote_code=True, enable_prefix_caching=True, dtype="bfloat16")
    outs = llm.generate(reqs, SamplingParams(n=args.n_samples, temperature=1.0, max_tokens=args.max_tokens, seed=42))
    res = []
    for r, o in zip(rows, outs):
        texts = [x.text for x in o.outputs]
        res.append(dict(row=r["_row"], group=r["images"][0]["path"], gt=r["_gt"], opts=r["_opts"], texts=texts,
                        letters=[parse_letter(t, r["_opts"]) for t in texts], first_tok=[x.token_ids[0] if x.token_ids else None for x in o.outputs]))
    os.makedirs(args.out, exist_ok=True); json.dump(res, open(os.path.join(args.out, "gen.json"), "w"))
    L = [x for g in res for x in g["letters"]]
    print(f"[gen] rows {len(res)}  samples {len(L)}  unparsed {np.mean([x is None for x in L]):.3f}  "
          f"sample acc {np.mean([x == g['gt'] for g in res for x in g['letters']]):.3f}  gt∈opts {np.mean([g['gt'] in g['opts'] for g in res]):.3f}", flush=True)


def part_hidden(args):
    import torch
    from transformers import AutoModelForImageTextToText
    import probe_layer_prior as P
    processor = setup_processor(); ps = processor.image_processor.patch_size; tok = processor.tokenizer
    letter_ids = [tok.encode(L) for L in LETTERS]
    assert all(len(x) == 1 for x in letter_ids), f"option letters must be single tokens: {letter_ids}"
    letter_ids = [x[0] for x in letter_ids]; pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    gen = {g["row"]: g for g in json.load(open(os.path.join(args.out, "gen.json")))}
    rows = [r for r in load_rows(args.n) if r["_row"] in gen]
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    views = P.Views(processor, 10240, ps)
    H, LP, kept = None, [], []
    for i, r in enumerate(rows):
        try:
            enc = views.student(r)
        except (RuntimeError, ValueError) as e:
            print(f"[hidden] skip row {r['_row']}: {e}", flush=True); continue
        ids = enc["input_ids"][0]; vis = (ids == pad_id).nonzero()
        if len(vis) == 0:
            print(f"[hidden] skip row {r['_row']}: no image tokens", flush=True); continue
        pos = [int(vis[-1]), len(ids) - 1]
        with torch.no_grad():
            out = model(**{k: v.cuda() for k, v in enc.items() if k != "_raw"}, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            enc_t, _ = views.text(r)
            out_t = model(**{k: v.cuda() for k, v in enc_t.items() if k != "_raw"}, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        hs = torch.cat([torch.stack([h[0, pos] for h in out.hidden_states], 1),
                        torch.stack([h[0, -1:] for h in out_t.hidden_states], 1)], 0).float()   # [3 positions, L+1, D]
        if not torch.isfinite(hs).all():
            raise RuntimeError(f"non-finite hidden state at row {r['_row']}")
        if H is None:
            H = np.lib.format.open_memmap(os.path.join(args.out, "hidden.tmp.npy"), mode="w+", dtype=np.float32, shape=(len(rows),) + tuple(hs.shape))
        H[len(kept)] = hs.cpu().numpy()
        LP.append(torch.log_softmax(out.logits[0, -1].float(), -1)[letter_ids].cpu().numpy()); kept.append(r["_row"])
        if (i + 1) % 200 == 0: print(f"[hidden] {i+1}/{len(rows)}", flush=True)
    if H is None:
        raise RuntimeError("no row survived encoding")
    np.save(os.path.join(args.out, "hidden.npy"), np.asarray(H[:len(kept)])); del H; os.remove(os.path.join(args.out, "hidden.tmp.npy"))
    np.savez(os.path.join(args.out, "readout.npz"), rows=np.array(kept), letter_logp=np.array(LP))
    print(f"[hidden] saved {len(kept)} rows, letter mass at t0 {np.exp(np.array(LP)).sum(1).mean():.3f}", flush=True)


# ------------------------------------------------------------------------------------------------ analyze
def build_targets(gen_rows):
    """Per row: class targets (index into LETTERS, -1 = excluded), valid-option masks, sample correctness."""
    n = len(gen_rows); M = np.zeros((n, 6), bool); y = {t: np.full(n, -1) for t in TARGETS}; corr = []
    for i, g in enumerate(gen_rows):
        for o in g["opts"]: M[i, LETTERS.index(o)] = True
        if g["gt"] in g["opts"]: y["gt"][i] = LETTERS.index(g["gt"])
        mj = majority(g["letters"])
        if mj is not None: y["maj"][i] = LETTERS.index(mj)
        c = [x == g["gt"] for x in g["letters"]]; corr.append(c)            # unparsed = wrong
        if g["gt"] in g["opts"]: y["acc"][i] = int(np.mean(c) >= 0.5)
        parsed = [x for x in g["letters"] if x is not None]
        if parsed and 2 * len(parsed) >= len(g["letters"]): y["det"][i] = int(len(set(parsed)) == 1)
    return y, M, corr


def part_analyze(args):
    gen = {g["row"]: g for g in json.load(open(os.path.join(args.out, "gen.json")))}
    z = np.load(os.path.join(args.out, "readout.npz")); rows = z["rows"]; lp = z["letter_logp"]
    Hm = np.load(os.path.join(args.out, "hidden.npy"))          # ~6 GB fp32, load once (strided memmap reads are slow on weka)
    g_rows = [gen[int(r)] for r in rows]
    y, M, corr = build_targets(g_rows)
    folds = group_folds([g["group"] for g in g_rows]); two = np.ones((len(rows), 2), bool)
    say = lambda s="": print(s, flush=True); res = dict(n=len(rows), layers=int(Hm.shape[2]))
    lpv = np.where(M, lp, -np.inf); lpv = lpv - np.logaddexp.reduce(lpv, 1, keepdims=True)   # t0 readout renormalised over options
    out_arg = lpv.argmax(1)
    for t in TARGETS: res[f"n_{t}"] = int((y[t] >= 0).sum())
    say(f"== {len(rows)} 题, 每题 {len(g_rows[0]['letters'])} 条采样; 有效标签 gt {res['n_gt']} / maj {res['n_maj']} / acc {res['n_acc']} / det {res['n_det']}")
    say(f"   采样正确率 {np.mean([np.mean(c) for c in corr]):.3f}; 未解析 {np.mean([x is None for g in g_rows for x in g['letters']]):.3f}; "
        f"t0 字母质量 {np.exp(lp).sum(1).mean():.3f}; acc 正类 {y['acc'][y['acc']>=0].mean():.3f}; det 正类 {y['det'][y['det']>=0].mean():.3f}")
    # baselines
    base = {}
    m = y["gt"] >= 0; base["gt_chance"] = float(np.mean(1 / M[m].sum(1))); base["gt_out"] = float((out_arg[m] == y["gt"][m]).mean())
    prior = np.zeros(len(rows), int)
    for f in np.unique(folds):
        tr = m & (folds != f); cnt = np.bincount(y["gt"][tr], minlength=6)
        prior[folds == f] = [int(np.argmax(np.where(M[i], cnt, -1))) for i in np.where(folds == f)[0]]
    base["gt_prior"] = float((prior[m] == y["gt"][m]).mean())
    m2 = y["maj"] >= 0; base["maj_out"] = float((out_arg[m2] == y["maj"][m2]).mean()); base["maj_chance"] = float(np.mean(1 / M[m2].sum(1)))
    m3 = y["acc"] >= 0; base["acc_out_auc"] = auc(lpv[m3, y["gt"][m3]], y["acc"][m3])
    m4 = y["det"] >= 0; base["det_out_auc"] = auc(lpv[m4].max(1), y["det"][m4])
    res["baseline"] = base
    say(f"   基线: GT 机会 {base['gt_chance']:.3f} | GT 先验 {base['gt_prior']:.3f} | t0 输出 argmax=GT {base['gt_out']:.3f} | "
        f"t0 输出 argmax=多数答案 {base['maj_out']:.3f} | acc AUC(p_t0(GT)) {base['acc_out_auc']:.3f} | det AUC(max p_t0) {base['det_out_auc']:.3f}")
    # layer sweep
    layers = list(range(Hm.shape[2])) if args.layer_step <= 1 else sorted(set(list(range(0, Hm.shape[2], args.layer_step)) + [Hm.shape[2] - 1]))
    metric = lambda t, P: (float((P[y[t] >= 0].argmax(1) == y[t][y[t] >= 0]).mean()) if t in ("gt", "maj") else auc(P[y[t] >= 0, 1], y[t][y[t] >= 0]))
    res["sweep"] = {}; preds = {}
    for pi, pname in enumerate(POSITIONS):
        say(f"\n== 位置 {pname} ({dict(vis='最后一个视觉 token', last='prompt 最后一个 token', txt='纯文本 prompt 最后一个 token (无图对照)')[pname]}); gt/maj = 准确率, acc/det = AUC")
        say("   层  " + "  ".join(f"{t:>6s}" for t in TARGETS))
        for l in layers:
            X = np.asarray(Hm[:, pi, l], dtype=np.float32); row = {}
            for t in TARGETS:
                n_cls, MM = (6, M) if t in ("gt", "maj") else (2, two)
                P = cv_probe(X, y[t], MM, folds, n_cls); row[t] = metric(t, P); preds[(pname, l, t)] = P
            res["sweep"][f"{pname}/{l}"] = row
            say(f"   {l:2d}  " + "  ".join(f"{row[t]:6.3f}" for t in TARGETS))
    # best layer per (position, target): shuffled control + diagnostics
    res["best"] = {}
    for pi, pname in enumerate(POSITIONS):
        for t in TARGETS:
            l = max(layers, key=lambda l: res["sweep"][f"{pname}/{l}"][t])
            X = np.asarray(Hm[:, pi, l], dtype=np.float32)
            n_cls, MM = (6, M) if t in ("gt", "maj") else (2, two)
            sh = metric(t, cv_probe(X, y[t], MM, folds, n_cls, shuffle=True))
            res["best"][f"{pname}/{t}"] = dict(layer=l, value=res["sweep"][f"{pname}/{l}"][t], shuffled=sh)
    say("\n== 各目标最优层 (值 | 打乱标签对照)")
    for k, v in res["best"].items(): say(f"   {k:10s} 层 {v['layer']:2d}  {v['value']:.3f} | 打乱 {v['shuffled']:.3f}")
    # does the best last-token GT probe know more than the output? rows where the t0 readout / the sample majority is wrong
    l = res["best"]["last/gt"]["layer"]; P = preds[("last", l, "gt")]; m = y["gt"] >= 0
    pa = P.argmax(1)
    for name, wrong in (("t0 输出 argmax≠GT", m & (out_arg != y["gt"])), ("采样多数≠GT", m & (y["maj"] >= 0) & (y["maj"] != y["gt"]))):
        others = np.maximum(M[wrong].sum(1) - 1, 1)
        Pt = preds[("txt", res["best"]["txt/gt"]["layer"], "gt")]
        say(f"   {name}: n={wrong.sum()}  探针命中 GT {(pa[wrong] == y['gt'][wrong]).mean():.3f}  (纯文本探针 {(Pt.argmax(1)[wrong] == y['gt'][wrong]).mean():.3f}; "
            f"GT 先验 {(prior[wrong] == y['gt'][wrong]).mean():.3f}; 机会 {np.mean(1 / M[wrong].sum(1)):.3f}; "
            f"排除模型自选项后的机会 {np.mean(1 / others):.3f})  探针=模型自选 {(pa[wrong] == (out_arg[wrong] if 't0' in name else y['maj'][wrong])).mean():.3f}")
        res[f"know_more/{name}"] = dict(n=int(wrong.sum()), hit=float((pa[wrong] == y["gt"][wrong]).mean()), hit_txt=float((Pt.argmax(1)[wrong] == y["gt"][wrong]).mean()),
                                        hit_prior=float((prior[wrong] == y["gt"][wrong]).mean()), chance=float(np.mean(1 / M[wrong].sum(1))))
    # per-trajectory: prompt-level scores vs each trajectory's correctness (prompt hidden is shared by all its trajectories)
    tr_lab, s_probe, s_out = [], [], []
    for i in np.where(m)[0]:
        for c in corr[i]: tr_lab.append(c); s_probe.append(P[i, y["gt"][i]]); s_out.append(lpv[i, y["gt"][i]])
    rate = np.array([np.mean(c) for c in corr])[m]; tr_lab = np.array(tr_lab)
    between = np.var(rate) / max(np.var(tr_lab.astype(float)), 1e-12)
    res["traj"] = dict(n=len(tr_lab), auc_probe=auc(s_probe, tr_lab), auc_out=auc(s_out, tr_lab), between_prompt_var_frac=float(between))
    say(f"\n== 逐轨迹: {len(tr_lab)} 条, 正确率方差中题间占比 {between:.3f} (prompt 末 hidden 只能解释这一部分)")
    say(f"   轨迹正确 AUC: 探针 p(GT) {res['traj']['auc_probe']:.3f} | t0 输出 p(GT) {res['traj']['auc_out']:.3f}")
    json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    say("\n[analyze] saved summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["gen", "hidden", "analyze"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=BASE); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=8); ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--layer_step", type=int, default=1)
    args = ap.parse_args()
    {"gen": part_gen, "hidden": part_hidden, "analyze": part_analyze}[args.part](args)


if __name__ == "__main__":
    main()
