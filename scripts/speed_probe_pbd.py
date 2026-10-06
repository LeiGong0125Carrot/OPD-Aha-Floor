#!/usr/bin/env python
"""P5 / PBD speed probe (no sharing, no training): measures the per-row cost of branch rows vs V0 rows on ONE GPU.
part=vllm : student rollouts y^S (full-image view), then crop-view continuations y^T from a prefix at ratio r of |y^S|
            (snapped to a word boundary) and student-view resamples y^S' from the same prefix; times + lengths + leak rate +
            normalized-LCS rewrite magnitude (diagnostic only).
part=hf   : HF bf16 + gradient checkpointing; per-row fwd+bwd on V0 rows (student view + y^S) and on branch rows
            (student view + prefix + y^T); teacher (crop view, no grad) forward on both. Prints a projection table.
Crop view = [full image][crop] + question, NO hint sentence (user constraint for P5).
用法: python scripts/speed_probe_pbd.py --part vllm|hf [--n 48] [--ratios 5,10,20] [--out eval/speed_pbd]
"""
import argparse, json, math, os, re, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import probe_layer_prior as P  # noqa: E402

PARQUET = "/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet"
MODEL = "Qwen/Qwen3.5-4B"
TPL = os.path.join(ROOT, "chat_templates/perception_chat_template_qwen35.jinja")
LEAK = re.compile(r"zoom|close-up|closeup|crop|enlarg|magnif|inset|second image|zoomed", re.I)


def crop_content(student_content: str) -> str:
    """[full][crop] + question, no hint sentence."""
    body = student_content[len("<image>\n"):] if student_content.startswith("<image>\n") else P.IMAGE_TAG.sub("", student_content).lstrip("\n")
    return "<image>\n<image>\n" + body


def build(processor, rows):
    from verl.utils.dataset.vision_utils import process_image
    out = []
    for r in rows:
        imgs_s = [process_image(dict(d), image_patch_size=14) for d in r["images"]]
        imgs_t = [P.load_rgb(d["path"]) for d in r["bbox_images"]]
        msg_s = P.messages_from_template(r["prompt"][0]["content"], imgs_s)
        msg_t = P.messages_from_template(crop_content(r["prompt"][0]["content"]), imgs_t)
        out.append(dict(row=r["_row"], raw_s=processor.apply_chat_template(msg_s, tokenize=False, add_generation_prompt=True), imgs_s=imgs_s,
                        raw_t=processor.apply_chat_template(msg_t, tokenize=False, add_generation_prompt=True), imgs_t=imgs_t))
    return out


def snap_prefix(tok, ids, ratio):
    k = max(1, math.ceil(ratio * len(ids)))
    while k < len(ids) and not tok.decode([ids[k]]).startswith((" ", "\n")):
        k += 1
    return ids[:k]


def lcs(a, b):
    n, m = len(a), len(b); prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1); ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def rmag(a, b):
    return 1 - lcs(a, b) / max(len(a), len(b), 1)


def part_vllm(args, processor, items):
    from vllm import LLM, SamplingParams
    tok = processor.tokenizer
    llm = LLM(model=MODEL, gpu_memory_utilization=0.6, max_model_len=12288, limit_mm_per_prompt={"image": 2},
              trust_remote_code=True, enable_prefix_caching=True, dtype="bfloat16")
    sp_s = SamplingParams(temperature=1.0, max_tokens=1024, seed=42)
    sp_c = SamplingParams(temperature=1.0, max_tokens=args.max_cont, seed=42)
    torch.cuda.synchronize(); t0 = time.time()
    outs = llm.generate([{"prompt": it["raw_s"], "multi_modal_data": {"image": it["imgs_s"]}} for it in items], sp_s)
    t_s = time.time() - t0
    for it, o in zip(items, outs):
        it["yS"] = list(o.outputs[0].token_ids); it["yS_text"] = o.outputs[0].text
    res = {"n": len(items), "gen_S_total_s": t_s, "gen_S_per_row_s": t_s / len(items),
           "yS_len_mean": float(np.mean([len(it["yS"]) for it in items])), "ratios": {}}
    print(f"[vllm] y^S: {len(items)} rows in {t_s:.1f}s ({t_s/len(items):.2f} s/row), mean len {res['yS_len_mean']:.0f}", flush=True)
    for ratio in args.ratios:
        pre = [snap_prefix(tok, it["yS"], ratio) for it in items]
        reqT = [{"prompt": it["raw_t"] + tok.decode(p), "multi_modal_data": {"image": it["imgs_t"]}} for it, p in zip(items, pre)]
        reqS = [{"prompt": it["raw_s"] + tok.decode(p), "multi_modal_data": {"image": it["imgs_s"]}} for it, p in zip(items, pre)]
        torch.cuda.synchronize(); t0 = time.time(); oT = llm.generate(reqT, sp_c); tT = time.time() - t0
        torch.cuda.synchronize(); t0 = time.time(); oS = llm.generate(reqS, sp_c); tS2 = time.time() - t0
        yT = [list(o.outputs[0].token_ids) for o in oT]; yS2 = [list(o.outputs[0].token_ids) for o in oS]
        suf = [it["yS"][len(p):] for it, p in zip(items, pre)]
        r_T = [rmag(a, b) for a, b in zip(yT, suf)]; r_S = [rmag(a, b) for a, b in zip(yS2, suf)]
        leak = float(np.mean([bool(LEAK.search(o.outputs[0].text)) for o in oT]))
        d = dict(gen_T_total_s=tT, gen_T_per_row_s=tT / len(items), gen_S2_total_s=tS2, prefix_len_mean=float(np.mean([len(p) for p in pre])),
                 yT_len_mean=float(np.mean([len(y) for y in yT])), yT_trunc_frac=float(np.mean([len(y) >= args.max_cont for y in yT])),
                 yT_leak_frac=leak, r_T_mean=float(np.mean(r_T)), r_S2_mean=float(np.mean(r_S)), r_tilde_mean=float(np.mean(np.array(r_T) - np.array(r_S))))
        res["ratios"][str(ratio)] = d
        print(f"[vllm] ratio {ratio}: prefix {d['prefix_len_mean']:.0f} tok; y^T gen {tT:.1f}s ({d['gen_T_per_row_s']:.2f} s/row), len {d['yT_len_mean']:.0f}, trunc {d['yT_trunc_frac']:.2f}, "
              f"leak {leak:.2f}; r(T) {d['r_T_mean']:.2f} vs r(S') {d['r_S2_mean']:.2f}; y^S' gen {tS2:.1f}s", flush=True)
        for it, p, a, b in zip(items, pre, yT, yS2):
            it.setdefault("branches", {})[str(ratio)] = dict(prefix=p, yT=a, yS2=b)
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "vllm_speed.json"), "w"), indent=1)
    json.dump([{k: v for k, v in it.items() if k in ("row", "yS", "yS_text", "branches")} for it in items], open(os.path.join(args.out, "sequences.json"), "w"))
    print("[vllm] saved", args.out, flush=True)


def part_hf(args, processor, items):
    from transformers import AutoModelForImageTextToText
    seqs = {s["row"]: s for s in json.load(open(os.path.join(args.out, "sequences.json")))}
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    views = P.Views(processor, 10240, 14)
    rows_by = {r["_row"]: r for r in P.load_rows(PARQUET, args.n, 42)}

    def enc_student(r): return views.student(r)
    def enc_crop(r):
        imgs = [P.load_rgb(d["path"]) for d in r["bbox_images"]]
        return views.encode(P.messages_from_template(crop_content(r["prompt"][0]["content"]), imgs), imgs)

    def fwd_bwd(enc, resp, grad):
        dev = model.device; resp_t = torch.tensor(resp, dtype=enc["input_ids"].dtype).unsqueeze(0)
        inp = {k: v.to(dev) for k, v in enc.items() if k != "_raw"}
        inp["input_ids"] = torch.cat([inp["input_ids"], resp_t.to(dev)], 1)
        inp["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(resp_t).to(dev)], 1)
        if "mm_token_type_ids" in inp: inp["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(resp_t).to(dev)], 1)
        T = len(resp)
        torch.cuda.synchronize(); t0 = time.time()
        if grad:
            model.train(); out = model(**inp, logits_to_keep=T + 1, use_cache=False)
            logp = torch.log_softmax(out.logits[0, :T].float(), -1).gather(-1, resp_t[0].to(dev).unsqueeze(-1)).mean()
            (-logp).backward(); model.zero_grad(set_to_none=True)
        else:
            model.eval()
            with torch.no_grad(): model(**inp, logits_to_keep=T + 1, use_cache=False)
        torch.cuda.synchronize(); return time.time() - t0

    sel = [it["row"] for it in items][: args.n_hf]
    t = {"v0_fwdbwd": [], "v0_teacher": []}; t.update({f"br{r}_fwdbwd": [] for r in args.ratios}); t.update({f"br{r}_teacher": [] for r in args.ratios})
    for i, row in enumerate(sel):
        r = rows_by[row]; s = seqs[row]; es = enc_student(r); ec = enc_crop(r)
        if i == 0:  # warm-up
            fwd_bwd(es, s["yS"], True); fwd_bwd(ec, s["yS"], False)
        t["v0_fwdbwd"].append(fwd_bwd(es, s["yS"], True)); t["v0_teacher"].append(fwd_bwd(ec, s["yS"], False))
        for ratio in args.ratios:
            b = s["branches"][str(ratio)]; resp = b["prefix"] + b["yT"]
            t[f"br{ratio}_fwdbwd"].append(fwd_bwd(es, resp, True)); t[f"br{ratio}_teacher"].append(fwd_bwd(ec, resp, False))
        print(f"[hf] row {i+1}/{len(sel)} P_s={es['input_ids'].shape[1]} P_c={ec['input_ids'].shape[1]} |yS|={len(s['yS'])}", flush=True)
    m = {k: float(np.mean(v)) for k, v in t.items()}
    vs = json.load(open(os.path.join(args.out, "vllm_speed.json")))
    print("\n[hf] per-row seconds (1 GPU, bf16, grad ckpt):"); [print(f"   {k:16s} {v:6.2f}") for k, v in m.items()]
    v0 = m["v0_fwdbwd"] + m["v0_teacher"] + vs["gen_S_per_row_s"]; null = m["v0_teacher"]
    print(f"\n[proj] V0 row cost = fwd+bwd {m['v0_fwdbwd']:.2f} + teacher {m['v0_teacher']:.2f} + gen {vs['gen_S_per_row_s']:.2f} = {v0:.2f} s; null forward (A extra) = {null:.2f} s/row")
    print(f"{'ratio':>6s} {'cov':>5s} {'branch row s':>13s} {'step/V0':>8s} {'step/A':>7s} {'in null units':>14s}")
    for ratio in args.ratios:
        br = m[f"br{ratio}_fwdbwd"] + m[f"br{ratio}_teacher"] + vs["ratios"][str(ratio)]["gen_T_per_row_s"]
        for cov in (0.25, 0.5, 1.0):
            print(f"{ratio:6.2f} {cov:5.2f} {br:13.2f} {(v0+cov*br)/v0:8.2f} {(v0+cov*br)/(v0+null):7.2f} {cov*br/null:14.2f}")
    m["projection_inputs"] = {"gen_S_per_row_s": vs["gen_S_per_row_s"], **{f"gen_T_per_row_s_{r}": vs["ratios"][str(r)]["gen_T_per_row_s"] for r in args.ratios}}
    json.dump(m, open(os.path.join(args.out, "hf_speed.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["vllm", "hf"], required=True)
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--n_hf", type=int, default=12)
    ap.add_argument("--ratios", default="0.05,0.10,0.20"); ap.add_argument("--max_cont", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/speed_pbd"))
    args = ap.parse_args(); args.ratios = [float(x) for x in args.ratios.split(",")]
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL); tpl = open(TPL).read()
    processor.chat_template = tpl; processor.tokenizer.chat_template = tpl
    rows = P.load_rows(PARQUET, args.n, 42)
    items = build(processor, rows)
    (part_vllm if args.part == "vllm" else part_hf)(args, processor, items)


if __name__ == "__main__":
    main()
