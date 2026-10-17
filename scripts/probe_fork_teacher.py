#!/usr/bin/env python
"""Fork / teacher-own-trajectory probe (user 10-17). Label-free: no GT, no correctness, distribution quantities only.

Question: across student trajectories of one prompt, where does the frozen teacher's scoring differ, and is that difference
evidence or style? Teacher-forced scoring is prefix-dominated (teacher drifts toward the student as the shared prefix grows), so
the teacher's OWN trajectories (generated from t=0 under the privileged view) are added as an unconditioned reference.

Pair data (train_6karmA_pair): student view = [full] + q (no box, no hint), teacher view = teacher_prompt [full][crop],
null view = [full][mean-colour crop] (A's null_scope=last). Same frozen base model for all three.

part=gen     (vLLM)  per row: n_s student-view samples + n_t teacher-view samples (temperature 1.0, seed 42)
part=score   (HF)    every trajectory scored under S / T / N: realized log-probs, TV(S,T) and TV(T,N) on the student top-K + tail
                     support (training 口径), student entropy, and the exact log-probs of both branch tokens at every pairwise fork
                     (first divergent position of two trajectories of the same row; shared prefix => same conditioning)
part=analyze (CPU)   A position profiles on student vs teacher trajectories | B off-policy-ness | C forks SS/TT/ST
                     (where, how strong vs entropy-matched baseline, raw teacher preference vs null-contrasted u) |
                     D post-fork self-consistency | E answer-position prefix dominance (teacher-forced vs teacher-own answers) |
                     F fork tokens at student->teacher forks
用法: python scripts/probe_fork_teacher.py --part gen|score|analyze --out eval/probe_fork/<id> [--n 200 --n_s 4 --n_t 4]
"""
import argparse, collections, itertools, json, math, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))

PARQUET = "/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet"
MODEL = "Qwen/Qwen3.5-4B"
TPL = os.path.join(ROOT, "chat_templates/perception_chat_template_qwen35.jinja")
LEAK = re.compile(r"zoom|close-up|closeup|crop|enlarg|magnif|inset|second image|zoomed", re.I)
ABS_BINS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 10**9)]


# ------------------------------------------------------------------------------------------------ pure helpers (unit-tested)
def first_divergence(a: list, b: list):
    """First index where a and b differ; None when one is a prefix of the other (no fork, incl. identical)."""
    for t in range(min(len(a), len(b))):
        if a[t] != b[t]:
            return t
    return None


def fork_pairs(trajs: list):
    """trajs: list of (kind 'S'|'T', idx, ids). Returns fork specs over all unordered pairs; for mixed pairs the student
    trajectory comes first (a = student token, b = teacher token). kind = 'SS' | 'TT' | 'ST'. Pairs without a fork are
    returned with t_div=None so they can be counted."""
    out = []
    for (ka, ia, ya), (kb, ib, yb) in itertools.combinations(trajs, 2):
        if ka == "T" and kb == "S":
            (ka, ia, ya), (kb, ib, yb) = (kb, ib, yb), (ka, ia, ya)
        t = first_divergence(ya, yb)
        out.append(dict(kind=ka + kb, a=(ka, ia), b=(kb, ib), t_div=t,
                        tok_a=None if t is None else int(ya[t]), tok_b=None if t is None else int(yb[t]),
                        len_a=len(ya), len_b=len(yb)))
    return out


def entropy_matched_baseline(vals, ent, fork_ent, n_bins=10):
    """Mean of `vals` over all positions, re-weighted to the entropy histogram of the fork positions (quantile bins of ent)."""
    vals, ent, fork_ent = map(np.asarray, (vals, ent, fork_ent))
    edges = np.unique(np.quantile(ent, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return float(vals.mean())
    bi = np.clip(np.searchsorted(edges, ent, side="right") - 1, 0, len(edges) - 2)
    bf = np.clip(np.searchsorted(edges, fork_ent, side="right") - 1, 0, len(edges) - 2)
    w = np.bincount(bf, minlength=len(edges) - 1).astype(float); w /= max(w.sum(), 1)
    means = np.array([vals[bi == k].mean() if (bi == k).any() else np.nan for k in range(len(edges) - 1)])
    ok = ~np.isnan(means) & (w > 0)
    return float((means[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-12))


# ------------------------------------------------------------------------------------------------ gen / score (GPU)
def load_items(args, processor):
    import probe_layer_prior as P
    from verl.utils.dataset.vision_utils import process_image
    rows = P.load_rows(PARQUET, args.n, 42)
    ps = processor.image_processor.patch_size          # training: process_vision_info(image_patch_size=16) for the student
    return rows, P, ps, process_image


def part_gen(args, processor):
    from vllm import LLM, SamplingParams
    rows, P, ps, process_image = load_items(args, processor)
    reqS, reqT, kept = [], [], []
    views = P.Views(processor, 10240, ps)
    for r in rows:
        try:
            views.student(r); views.real(r); views.null(r)
        except RuntimeError as e:                       # prompt >= 10240 tokens: skip the row (same limit in score)
            print(f"[gen] skip row {r['_row']}: {e}", flush=True); continue
        kept.append(r)
        imgs_s = [process_image(dict(d), image_patch_size=ps) for d in r["images"]]
        imgs_t = [P.load_rgb(d["path"]) for d in r["bbox_images"]]
        reqS.append({"prompt": processor.apply_chat_template(P.messages_from_template(r["prompt"][0]["content"], imgs_s), tokenize=False, add_generation_prompt=True),
                     "multi_modal_data": {"image": imgs_s}})
        reqT.append({"prompt": processor.apply_chat_template(P.messages_from_template(r["teacher_prompt"][0]["content"], imgs_t), tokenize=False, add_generation_prompt=True),
                     "multi_modal_data": {"image": imgs_t}})
    llm = LLM(model=MODEL, gpu_memory_utilization=0.6, max_model_len=12288, limit_mm_per_prompt={"image": 2},
              trust_remote_code=True, enable_prefix_caching=True, dtype="bfloat16")
    oS = llm.generate(reqS, SamplingParams(n=args.n_s, temperature=1.0, max_tokens=args.max_tokens, seed=42))
    # disjoint seeds: vLLM uses seed+i for the i-th of n samples; equal seeds would Gumbel-couple student/teacher sample i (review 10-17)
    oT = llm.generate(reqT, SamplingParams(n=args.n_t, temperature=1.0, max_tokens=args.max_tokens, seed=42 + 10_000))
    out = []
    for r, a, b in zip(kept, oS, oT):
        out.append(dict(row=r["_row"], S=[list(o.token_ids) for o in a.outputs], T=[list(o.token_ids) for o in b.outputs],
                        S_text=[o.text for o in a.outputs], T_text=[o.text for o in b.outputs]))
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "gen.json"), "w"))
    L = lambda k: np.mean([len(y) for g in out for y in g[k]])
    print(f"[gen] rows {len(out)}  |yS| {L('S'):.0f}  |yT| {L('T'):.0f}  teacher leak {np.mean([bool(LEAK.search(t)) for g in out for t in g['T_text']]):.3f}", flush=True)


def part_score(args, processor):
    import torch
    from transformers import AutoModelForImageTextToText
    from verl.utils.layer_prior_probe import coarsen_logits, find_answer_token_positions
    rows, P, ps, _ = load_items(args, processor)
    rows_by = {r["_row"]: r for r in rows}
    gen = json.load(open(os.path.join(args.out, "gen.json")))
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    views = P.Views(processor, 10240, ps); tok = processor.tokenizer
    pos = collections.defaultdict(list); forks = []; answers = []
    for gi, g in enumerate(gen):
        row = rows_by[g["row"]]
        enc = {"S": views.student(row), "T": views.real(row), "N": views.null(row)}
        trajs = [("S", i, y) for i, y in enumerate(g["S"])] + [("T", j, y) for j, y in enumerate(g["T"])]
        specs = fork_pairs(trajs)
        queries = collections.defaultdict(list)                       # (kind, idx) -> spec indices scored on that trajectory
        for si, s in enumerate(specs):
            if s["t_div"] is not None: queries[s["a"]].append(si)
        for kind, idx, y in trajs:
            if len(y) == 0: continue
            with torch.no_grad():
                ls = {v: torch.log_softmax(P.score(model, enc[v], y, 1.0), -1) for v in "STN"}
                topk = ls["S"].topk(args.K, -1).indices
                cs = {v: coarsen_logits(ls[v], topk) for v in "STN"}      # log_softmax is idempotent: coarsen of normalised = same
                tv = lambda p, q: 0.5 * (cs[p].exp() - cs[q].exp()).abs().sum(-1)
                yi = torch.tensor(y, device=ls["S"].device).unsqueeze(-1)
                real = {v: ls[v].gather(-1, yi).squeeze(-1) for v in "STN"}
                ent = -(ls["S"].exp() * ls["S"]).sum(-1)
                rec = dict(tv_ST=tv("S", "T"), tv_TN=tv("T", "N"), lS=real["S"], lT=real["T"], lN=real["N"], ent_S=ent,
                           top1_agree=(ls["S"].argmax(-1) == ls["T"].argmax(-1)).float())
                for k, v in rec.items(): pos[k].append(v.float().cpu().numpy())
                pos["row"].append(np.full(len(y), g["row"])); pos["kind"].append(np.full(len(y), kind)); pos["traj"].append(np.full(len(y), idx))
                pos["t"].append(np.arange(len(y))); pos["T"].append(np.full(len(y), len(y)))
                for si in queries[(kind, idx)]:
                    s = specs[si]; t = s["t_div"]
                    for v in "STN":
                        s[f"l{v}_a"] = float(ls[v][t, s["tok_a"]]); s[f"l{v}_b"] = float(ls[v][t, s["tok_b"]])
                    s["tv_ST"] = float(rec["tv_ST"][t]); s["ent_S"] = float(ent[t]); s["row"] = g["row"]
                    s["a_txt"] = tok.decode([s["tok_a"]]); s["b_txt"] = tok.decode([s["tok_b"]])
                ans = find_answer_token_positions(y, lambda k: tok.decode(y[:k]))
                letter, t0 = None, None
                if ans["found"]:                                  # letter = first option letter inside the located span; t0 = the token carrying it
                    c0, c1 = ans["char_span"]; m = re.search(r"(?<![A-Za-z])[A-F](?![A-Za-z])", tok.decode(y)[c0:c1])   # standalone option letter, not "A red" / "Because"
                    if m:
                        letter = m.group(0); t0 = next((p for p in ans["positions"] if letter in tok.decode([y[p]])), None)
                rec_a = dict(row=g["row"], kind=kind, idx=idx, t=t0, T=len(y), letter=letter, src=ans["source"])
                if t0 is not None: rec_a.update({f"l{v}": float(real[v][t0]) for v in "STN"})
                answers.append(rec_a)
                del ls, cs
        for s in specs:
            s["a"] = list(s["a"]); s["b"] = list(s["b"]); s.setdefault("row", g["row"])
        forks.extend(specs)
        if (gi + 1) % 10 == 0: print(f"[score] {gi+1}/{len(gen)}", flush=True)
    np.savez_compressed(os.path.join(args.out, "positions.npz"), **{k: np.concatenate(v) for k, v in pos.items()})
    json.dump(forks, open(os.path.join(args.out, "forks.json"), "w")); json.dump(answers, open(os.path.join(args.out, "answers.json"), "w"))
    print("[score] saved", args.out, flush=True)


# ------------------------------------------------------------------------------------------------ analyze (CPU)
def part_analyze(args):
    z = np.load(os.path.join(args.out, "positions.npz")); forks = json.load(open(os.path.join(args.out, "forks.json")))
    answers = json.load(open(os.path.join(args.out, "answers.json"))); gen = json.load(open(os.path.join(args.out, "gen.json")))
    kind, t, T = z["kind"], z["t"], z["T"]; rel = t / T; res = {}
    say = lambda s="": print(s, flush=True)
    fm = lambda x, p=2: "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{p}f}"
    mn = lambda a: float(a.mean()) if len(a) else float("nan")
    say(f"== 样本: {len(gen)} 题, 学生轨迹 {sum(len(g['S']) for g in gen)} 条 (均长 {np.mean([len(y) for g in gen for y in g['S']]):.0f}), "
        f"教师自生成 {sum(len(g['T']) for g in gen)} 条 (均长 {np.mean([len(y) for g in gen for y in g['T']]):.0f}); "
        f"教师轨迹含 crop 措辞 {np.mean([bool(LEAK.search(x)) for g in gen for x in g['T_text']]):.3f}")
    # A / B: position profiles
    for name, col in (("A1 TV(S,T) 师生分布差", "tv_ST"), ("A2 TV(T,N) crop 证据效应", "tv_TN"), ("B1 学生视角下实现 token logp", "lS"),
                      ("B2 教师视角下实现 token logp", "lT"), ("A3 top-1 一致率", "top1_agree")):
        v = z[col]; say(f"\n== {name}: 按绝对位置 (学生轨迹 | 教师自生成轨迹)")
        res[col] = {}
        for a, b in ABS_BINS:
            cells = []
            for k in "ST":
                m = (kind == k) & (t >= a) & (t < b); cells.append(f"{v[m].mean():7.3f} (n={m.sum():6d})" if m.any() else "      -        ")
            say(f"  [{a},{b if b < 10**9 else '∞'}) " + " | ".join(cells)); res[col][f"{a}-{b}"] = [float(v[(kind == k) & (t >= a) & (t < b)].mean()) if ((kind == k) & (t >= a) & (t < b)).any() else None for k in "ST"]
        say("  相对位置十分位: " + "  ".join(f"{i/10:.1f}:{fm(mn(v[(kind=='S')&(rel>=i/10)&(rel<(i+1)/10)]),3)}/{fm(mn(v[(kind=='T')&(rel>=i/10)&(rel<(i+1)/10)]),3)}" for i in range(10)))
    # C: forks
    say("\n== C 分叉 (同题两条轨迹的第一个不同 token; 前缀相同 => 条件相同)")
    say("   Δ_V = log p_V(a) − log p_V(b);  教师改判 d = Δ_T − Δ_S;  去风格 u = Δ_T − Δ_N (A 的反事实对比)")
    say("   注: 统计按'轨迹对'计, n=4 时同一分叉可被多对重复计入 (非独立样本); 基线 = 与轨迹 a 同类的全部位置按分叉熵直方图重加权")
    ent_all, tv_all = z["ent_S"], z["tv_ST"]; res["forks"] = {}
    for k in ("SS", "TT", "ST"):
        fk = [f for f in forks if f["kind"] == k]; ok = [f for f in fk if f["t_div"] is not None and "lT_a" in f]
        if not ok: say(f"  {k}: 无分叉"); continue
        td = np.array([f["t_div"] for f in ok]); tdr = td / np.array([min(f["len_a"], f["len_b"]) for f in ok])
        dS = np.array([f["lS_a"] - f["lS_b"] for f in ok]); dT = np.array([f["lT_a"] - f["lT_b"] for f in ok]); dN = np.array([f["lN_a"] - f["lN_b"] for f in ok])
        d, u = dT - dS, dT - dN; ftv = np.array([f["tv_ST"] for f in ok]); fent = np.array([f["ent_S"] for f in ok])
        ka = k[0]                                            # positions of the same trajectory kind as a (SS/ST: student, TT: teacher)
        base = entropy_matched_baseline(tv_all[kind == ka], ent_all[kind == ka], fent)
        sig = np.abs(d) > 0.5
        r = dict(n_pairs=len(fk), no_fork=len(fk) - len(ok), t_div_median=float(np.median(td)), t_div_lt5=float((td < 5).mean()), t_div_lt20=float((td < 20).mean()),
                 t_div_rel_median=float(np.median(tdr)), tv_fork=float(ftv.mean()), tv_base_entropy_matched=base, abs_d_mean=float(np.abs(d).mean()), frac_abs_d_gt05=float(sig.mean()),
                 corr_d_u=float(np.corrcoef(d, u)[0, 1]) if len(d) > 2 and d.std() > 0 and u.std() > 0 else None,
                 sign_agree_d_u=float((np.sign(d[sig]) == np.sign(u[sig])).mean()) if sig.any() else None,
                 abs_u_over_abs_d=float(np.median(np.abs(u[sig]) / np.abs(d[sig]))) if sig.any() else None)
        if k == "ST":
            r["teacher_prefers_teacher_tok"] = float((dT < 0).mean()); r["u_prefers_teacher_tok"] = float((u < 0).mean())
        res["forks"][k] = r
        say(f"  {k}: 对数 {r['n_pairs']}, 不分叉 {r['no_fork']}; t_div 中位 {r['t_div_median']:.0f} (相对 {r['t_div_rel_median']:.2f}), <5: {r['t_div_lt5']:.2f}, <20: {r['t_div_lt20']:.2f}")
        say(f"      分叉处 TV(S,T) {r['tv_fork']:.3f} vs 同熵基线 {base:.3f};  |d| 均值 {r['abs_d_mean']:.2f}, |d|>0.5 占 {r['frac_abs_d_gt05']:.2f};  "
            f"corr(d,u) {fm(r['corr_d_u'])}, |d|>0.5 时 sign(d)=sign(u) {fm(r['sign_agree_d_u'])}, |u|/|d| 中位 {fm(r['abs_u_over_abs_d'])}")
        if k == "ST":
            say(f"      教师 (原始) 偏好教师 token {r['teacher_prefers_teacher_tok']:.2f};  去风格 u 仍偏好教师 token {r['u_prefers_teacher_tok']:.2f}")
    # D: post-fork self-consistency on SS forks
    say("\n== D 分叉后自洽 (SS): 教师偏好的那支 (d>0.5 选 a, d<-0.5 选 b), 分叉后剩余位置的平均 TV(S,T) 是否更低")
    grp = collections.defaultdict(list)                     # (row, kind, traj) -> position indices (in t order)
    for i_, key_ in enumerate(zip(z["row"].tolist(), kind.tolist(), z["traj"].tolist())): grp[key_].append(i_)
    def post_tv(row, k_, j_, t0):
        ii = np.asarray(grp.get((row, k_, j_), []), dtype=int); ii = ii[t[ii] > t0] if len(ii) else ii
        return float(tv_all[ii].mean()) if len(ii) else np.nan
    for lab, use in (("原始 d", "d"), ("去风格 u", "u")):
        pref, other = [], []
        for f in forks:
            if f["kind"] != "SS" or f["t_div"] is None or "lT_a" not in f: continue
            d = (f["lT_a"] - f["lT_b"]) - (f["lS_a"] - f["lS_b"]); u = (f["lT_a"] - f["lT_b"]) - (f["lN_a"] - f["lN_b"])
            s = d if use == "d" else u
            if abs(s) <= 0.5: continue
            ta, tb = post_tv(f["row"], "S", f["a"][1], f["t_div"]), post_tv(f["row"], "S", f["b"][1], f["t_div"])
            if np.isnan(ta) or np.isnan(tb): continue
            (pref.append(ta), other.append(tb)) if s > 0 else (pref.append(tb), other.append(ta))
        if pref:
            pref, other = np.array(pref), np.array(other)
            say(f"  按 {lab}: n={len(pref)}  偏好支 {pref.mean():.4f} vs 另一支 {other.mean():.4f}  (偏好支更低的比例 {(pref < other).mean():.2f})")
            res[f"D_{use}"] = dict(n=len(pref), pref=float(pref.mean()), other=float(other.mean()), frac_lower=float((pref < other).mean()))
    # E: answer-position prefix dominance
    say("\n== E 答案位: 教师 teacher-forcing 打分 vs 教师自己生成的答案分布 (只比字母一致性, 不用 GT)")
    by_row = collections.defaultdict(list)
    for a in answers: by_row[a["row"]].append(a)
    agree_pT, dis_pT, agree_pN, dis_pN, dis_q = [], [], [], [], []; n_tie = 0
    for row, lst in by_row.items():
        tl = [a["letter"] for a in lst if a["kind"] == "T" and a["letter"]]
        if not tl: continue
        q = collections.Counter(tl); mc = q.most_common(2)
        if len(mc) > 1 and mc[0][1] == mc[1][1]: n_tie += 1; continue      # tied teacher majority: excluded
        maj = mc[0][0]
        for a in lst:
            if a["kind"] != "S" or a["t"] is None or not a["letter"]: continue
            (agree_pT if a["letter"] == maj else dis_pT).append(math.exp(a["lT"])); (agree_pN if a["letter"] == maj else dis_pN).append(math.exp(a["lN"]))
            if a["letter"] != maj: dis_q.append(q[a["letter"]] / len(tl))
    if agree_pT or dis_pT:
        say(f"  学生答案 = 教师自生成多数答案: n={len(agree_pT)}  教师打分 p_T(答案 token | 学生前缀) 均值 {np.mean(agree_pT):.3f}")
        if dis_pT:
            say(f"  学生答案 ≠ 教师多数答案:       n={len(dis_pT)}  p_T 均值 {np.mean(dis_pT):.3f} (null 视图 {np.mean(dis_pN):.3f});  "
                f"教师自己生成时选该字母的频率 {np.mean(dis_q):.3f}")
        res["E"] = dict(n_agree=len(agree_pT), pT_agree=float(np.mean(agree_pT)) if agree_pT else None, n_dis=len(dis_pT),
                        pT_dis=float(np.mean(dis_pT)) if dis_pT else None, pN_dis=float(np.mean(dis_pN)) if dis_pN else None, q_dis=float(np.mean(dis_q)) if dis_q else None)
    miss = collections.Counter((a["kind"], a["t"] is None) for a in answers)
    say(f"  未定位到答案位: 学生 {miss[('S', True)]}, 教师 {miss[('T', True)]};  教师自生成答案平票而排除的题: {n_tie}")
    # F: tokens at student->teacher forks
    say("\n== F 学生->教师自生成 分叉处的 token 对 (学生 token → 教师 token), 前 15")
    st = [f for f in forks if f["kind"] == "ST" and f["t_div"] is not None and "a_txt" in f]
    for (a, b), c in collections.Counter((f["a_txt"], f["b_txt"]) for f in st).most_common(15):
        say(f"  {c:4d}  {a!r} → {b!r}")
    json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    say("\n[analyze] saved summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["gen", "score", "analyze"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--n_s", type=int, default=4); ap.add_argument("--n_t", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1024); ap.add_argument("--K", type=int, default=100)
    args = ap.parse_args()
    if args.part == "analyze":
        return part_analyze(args)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL); tpl = open(TPL).read()
    processor.chat_template = tpl; processor.tokenizer.chat_template = tpl
    (part_gen if args.part == "gen" else part_score)(args, processor)


if __name__ == "__main__":
    main()
