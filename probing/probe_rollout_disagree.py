#!/usr/bin/env python
"""Rollout-disagreement 探针: 特权 discrepancy 在同一 prompt 的 n 条 rollout 间是否一致?

科学问题: n=8 的梯度平均, 是在降噪 (rollout 间 agreement) 还是在稀释信号
(rollout 间 disagreement 互相抵消)? 所有 TB 峰 >50 的臂都是 n=2, 三个 n=8 臂全 ≤48.64。

做法 (student=teacher=base, 即 step-0 状态, discrepancy 纯由视图差异产生):
  1. 取 pair parquet 的 K 行, 每行用 HF generate 采 n 条 rollout (训练同采样参数)
  2. 对每条 rollout teacher-force 三次前向:
       p_S  = [全图]            + prompt          (定 support = top-100, 与训练一致)
       p⁺   = [全图, GT crop]   + teacher_prompt  (bbox_images)
       p⁰   = [全图, 均值色块]  + teacher_prompt  (现算! 训练不读 null_images 列)
  3. dump per-position (T, k) log-prob, 分析交给 analyze_rollout_disagree.py

口径对齐训练的三处关键:
  - support = student top-k, teacher/null gather 到 student ids (dp_actor.py:498,525)
  - log-prob 全词表归一化后 gather (dp_actor.py:526-527), 不是 support 内重归一化
  - null 视图 = _mean_color_teacher_images(bbox_images, "last") 现算 (ray_trainer.py:732)
  - 自定义 chat template 注入 processor.chat_template (ray_trainer.py:361-365)

用法:
  PYTHONPATH=$REPO python probing/probe_rollout_disagree.py --rows 16 --n 8 --out results/rd_dump
  PYTHONPATH=$REPO python probing/probe_rollout_disagree.py --row 0 --n 8 --out results/rd_smoke
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_PARQUET = "/sfs/weka/scratch/nkw3mr/Vision-OPD/data/TreeVGR-RL-37K/train_6karmA_pair.parquet"
DEFAULT_TEMPLATE = "/sfs/weka/scratch/nkw3mr/Vision-OPD/chat_templates/perception_chat_template_qwen35.jinja"


def trim_at_eos(ids, eos_ids):
    """截到首个 eos (含), 与 gen_rollouts_treebench.py:37-42 同口径。"""
    for i, t in enumerate(ids):
        if t in eos_ids:
            return ids[: i + 1]
    return ids


def build_messages_from_template(content: str, images):
    """把 '<image>' 占位符按序换成 PIL 图 —— 与 ray_trainer._build_teacher_messages_from_template
    (:777) 同逻辑 (该方法是实例方法但只调 static, 这里直接复刻以免依赖 trainer 实例)。"""
    parts = [s for s in re.split(r"(<image>)", content) if s != ""]
    out, off = [], 0
    for seg in parts:
        if seg == "<image>":
            if off >= len(images):
                raise ValueError(f"image count < placeholders: {len(images)=} {off=}")
            out.append({"type": "image", "image": images[off]})
            off += 1
        else:
            out.append({"type": "text", "text": seg})
    if off != len(images):
        raise ValueError(f"image count != placeholders: {len(images)=} {off=}")
    return [{"role": "user", "content": out}]


def encode(processor, device, messages):
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)
    txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[txt], images=image_inputs, videos=video_inputs,
                     padding=True, return_tensors="pt").to(device)


@torch.no_grad()
def forced_logp(model, inputs, cont_ids, support_ids=None, topk=100, chunk=128):
    """Teacher-force 续写, 返回 (ids[T,k], logp[T,k]) on CPU。

    support_ids=None -> 本视图自己的 top-k (student 用, 定 support);
    support_ids 给定 -> 在该 support 上 gather (teacher/null 用)。
    logp = topk_logits - logsumexp(全词表) —— 与 dp_actor.py:526-527 同口径。
    对齐是 continuation-relative, 因此 prompt 长度不同 (1 图 vs 2 图) 也能逐位置对上。
    """
    full = torch.cat([inputs["input_ids"], cont_ids.unsqueeze(0).to(model.device)], dim=1)
    kw = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    if "mm_token_type_ids" in kw:
        pad = torch.zeros((1, cont_ids.shape[0]), dtype=kw["mm_token_type_ids"].dtype,
                          device=kw["mm_token_type_ids"].device)
        kw["mm_token_type_ids"] = torch.cat([kw["mm_token_type_ids"], pad], dim=1)
    T = cont_ids.shape[0]
    with torch.inference_mode():
        try:
            out = model(input_ids=full, attention_mask=torch.ones_like(full),
                        logits_to_keep=T + 1, **kw)
            logits = out.logits[0, :T]
        except TypeError:
            out = model(input_ids=full, attention_mask=torch.ones_like(full), **kw)
            p0 = inputs["input_ids"].shape[1]
            logits = out.logits[0, p0 - 1: p0 - 1 + T]
    out_ids = torch.empty((T, topk), dtype=torch.long)
    out_lp = torch.empty((T, topk))
    for i in range(0, T, chunk):
        lg = logits[i:i + chunk].float()
        lse = torch.logsumexp(lg, dim=-1, keepdim=True)   # 全词表 logZ
        if support_ids is None:
            v, idx = lg.topk(topk, dim=-1)
            out_ids[i:i + chunk] = idx.cpu()
            out_lp[i:i + chunk] = (v - lse).cpu()
        else:
            sid = support_ids[i:i + chunk].to(lg.device)
            out_ids[i:i + chunk] = sid.cpu()
            out_lp[i:i + chunk] = (lg.gather(-1, sid) - lse).cpu()
    del logits, out
    return out_ids, out_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--chat-template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--rows", type=int, default=16, help="随机抽多少行 (--row 指定时忽略)")
    ap.add_argument("--row", type=int, default=None, help="只跑这一行 (smoke)")
    ap.add_argument("--n", type=int, default=8, help="每行 rollout 条数")
    ap.add_argument("--topk", type=int, default=100, help="support 宽度, 训练用 100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="results/rd_dump")
    args = ap.parse_args()

    from PIL import Image  # noqa: F401  (ray_trainer 的 static 方法需要)
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_parquet(args.parquet)
    if args.row is not None:
        rows = [args.row]
    else:
        rng = np.random.default_rng(args.seed)
        rows = sorted(rng.choice(len(df), size=min(args.rows, len(df)), replace=False).tolist())
    print(f">>> parquet {len(df)} 行, 本次跑 rows={rows}, n={args.n}, topk={args.topk}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.chat_template and os.path.exists(args.chat_template):
        with open(args.chat_template) as f:
            tpl = f.read()
        processor.chat_template = tpl          # 与 ray_trainer.py:361-365 同款注入
        tokenizer.chat_template = tpl
        print(f">>> 自定义 chat template: {args.chat_template}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    eos_ids = {tokenizer.eos_token_id}
    for t in ("<|im_end|>",):
        i = tokenizer.convert_tokens_to_ids(t)
        if i is not None and i >= 0:
            eos_ids.add(i)

    meta_path = os.path.join(args.out, "meta.jsonl")
    done = set()
    if os.path.exists(meta_path):
        for line in open(meta_path):
            r = json.loads(line)
            done.add((r["row"], r["rollout_id"]))
    mf = open(meta_path, "a")

    for row in rows:
        item = df.iloc[row]
        stu_content = item["prompt"][0]["content"]
        tch_content = item["teacher_prompt"][0]["content"]
        stu_imgs = [RayPPOTrainer._normalize_teacher_image(x) for x in item["images"]]
        pos_imgs = [RayPPOTrainer._normalize_teacher_image(x) for x in item["bbox_images"]]
        # 训练口径: null 现算, 只遮末图 (不读 parquet 的 null_images 列 —— 那是死列)
        null_imgs = RayPPOTrainer._mean_color_teacher_images(list(item["bbox_images"]), "last")

        msg_stu = build_messages_from_template(stu_content, stu_imgs)
        msg_pos = build_messages_from_template(tch_content, pos_imgs)
        msg_null = build_messages_from_template(tch_content, null_imgs)
        in_stu = encode(processor, model.device, msg_stu)
        in_pos = encode(processor, model.device, msg_pos)
        in_null = encode(processor, model.device, msg_null)

        # ---- rollout 生成 (student 视图), 训练同采样参数; n 条一次批量, OOM 降级逐条 ----
        gen_kw = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                      repetition_penalty=1.0, max_new_tokens=args.max_new_tokens, use_cache=True)
        torch.manual_seed(args.seed * 1_000_000 + row)
        try:
            with torch.inference_mode():
                gen = model.generate(**in_stu, num_return_sequences=args.n, **gen_kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[row {row}] n={args.n} 批量 OOM, 降级逐条", flush=True)
            seqs = []
            for sid in range(args.n):
                torch.manual_seed(args.seed * 1_000_000 + row + 7919 * (sid + 1))
                with torch.inference_mode():
                    seqs.append(model.generate(**in_stu, num_return_sequences=1, **gen_kw)[0])
            L = max(s.shape[0] for s in seqs)
            gen = torch.full((args.n, L), list(eos_ids)[0], dtype=seqs[0].dtype, device=seqs[0].device)
            for sid, s in enumerate(seqs):
                gen[sid, : s.shape[0]] = s
        p0 = in_stu["input_ids"].shape[1]

        for sid in range(args.n):
            if (row, sid) in done:
                continue
            ids = trim_at_eos(gen[sid][p0:].tolist(), eos_ids)
            if len(ids) < 2:
                print(f"[row {row} rid {sid}] 空续写, 跳过", flush=True)
                continue
            cont = torch.tensor(ids, dtype=torch.long)
            # student 定 support, teacher/null gather 到同一 support (训练口径)
            sup_ids, lp_stu = forced_logp(model, in_stu, cont, None, args.topk)
            _, lp_pos = forced_logp(model, in_pos, cont, sup_ids, args.topk)
            _, lp_null = forced_logp(model, in_null, cont, sup_ids, args.topk)
            assert lp_stu.shape == lp_pos.shape == lp_null.shape == (len(ids), args.topk)

            npz = f"{row}_{sid}.npz"
            np.savez_compressed(os.path.join(args.out, npz),
                                ids=sup_ids.numpy().astype(np.int32),
                                lp_stu=lp_stu.numpy().astype(np.float16),
                                lp_pos=lp_pos.numpy().astype(np.float16),
                                lp_null=lp_null.numpy().astype(np.float16))
            text = tokenizer.decode(ids, skip_special_tokens=False)
            m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            rec = {"row": int(row), "rollout_id": sid, "T": len(ids),
                   "prediction": (m.group(1).strip().upper() if m else ""),
                   "gt": str(item["reward_model"].get("ground_truth", "")),
                   "box_frac": float(item["extra_info"].get("box_frac", -1)),
                   "task": str(item["extra_info"].get("task", "")),
                   "npz": npz, "model": args.model, "topk": args.topk}
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            print(f"  row {row} rid {sid}: T={len(ids)} pred={rec['prediction']} gt={rec['gt']}", flush=True)
        torch.cuda.empty_cache()

    mf.close()
    print(f">>> dump 完成: {args.out}", flush=True)


if __name__ == "__main__":
    main()
