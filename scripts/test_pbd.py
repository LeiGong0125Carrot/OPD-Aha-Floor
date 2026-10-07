#!/usr/bin/env python
"""P5 PBD unit tests -- CPU only.
[1] helpers (snap prefix, leak mask, crop-view messages, coverage select); [2] branch-row tensors + parent masks;
[3] config validation; [4] core_algos: pbd-off bit-identical to the plain path; pbd-on loss = mean_v0 + lambda*mean_branch
    (rows aggregated as the trainer does: micro-batch = row, averaged over rows); loss mask excludes prefix / leak tokens;
    metrics present; null mode + pbd rows raises; [5] position ids without a processor; [6] source guards in trainer/actor.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils import pbd as P  # noqa: E402

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: FAIL.append(name)
def raises(fn, exc=Exception):
    try: fn(); return False
    except exc: return True

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

print("[1] helpers")
ids = tok("The sign above the entrance reads 'ELY DIOCESE'. The final letters are E-S-E. Therefore the answer is A.", add_special_tokens=False)["input_ids"]
k = P.snap_prefix_len(tok, ids, 0.10)
check("snap: k >= ceil(0.1*T)", k is not None and k >= int(np.ceil(0.1 * len(ids))), f"k={k} T={len(ids)}")
check("snap: token at k starts a word", tok.decode([ids[k]]).startswith((" ", "\n")), repr(tok.decode([ids[k]])))
check("snap: too-short response -> None", P.snap_prefix_len(tok, ids[:3], 0.1) is None)
check("snap: ratio near 1 -> None", P.snap_prefix_len(tok, ids, 0.99) is None)
yT = tok(" The zoomed-in view clearly shows the letters E-S-E. The close-up confirms it.", add_special_tokens=False)["input_ids"]
lm = P.leak_token_mask(tok, yT)
check("leak mask: length matches", len(lm) == len(yT))
check("leak mask: marks zoom/close-up tokens, not all", 0 < sum(lm) < len(yT), f"{sum(lm)}/{len(yT)}")
marked = tok.decode([t for t, f in zip(yT, lm) if f]).lower()
check("leak mask: marked text contains 'zoom' and 'close'", "zoom" in marked and "close" in marked, repr(marked))
check("leak mask: clean text -> all False", not any(P.leak_token_mask(tok, ids)))
raw = [{"role": "user", "content": [{"type": "image", "image": "/x/full.jpg"}, {"type": "text", "text": "\nWhat color?\nA. red\nB. blue"}]}]
cv = P.crop_view_messages(raw, {"path": "/x/crop.jpg"})
c = cv[-1]["content"]
check("crop view: [full][crop][text] order", [x["type"] for x in c] == ["image", "image", "text"])
check("crop view: crop dict normalised like the dataset ({type, image=path, path})", c[1].get("image") == "/x/crop.jpg" and isinstance(c[1]["image"], str), str(c[1]))
cv2 = P.crop_view_messages(raw, "/x/crop2.jpg"); check("crop view: plain path accepted", cv2[-1]["content"][1]["image"] == "/x/crop2.jpg")
check("crop view: dict without path/image rejected", raises(lambda: P.crop_view_messages(raw, {"foo": 1}), ValueError))
check("crop view: original not mutated", len(raw[-1]["content"]) == 2)
check("crop view: no hint sentence added", all("zoom" not in str(x.get("text", "")).lower() for x in c))
tv = P.teacher_view_messages([{"role": "user", "content": "<image>\nQ?\nA. x\nB. y"}], [{"path": "/h/hide.jpg"}])
check("teacher view: one image then text, path normalised", [c["type"] for c in tv[-1]["content"]] == ["image", "text"] and tv[-1]["content"][0]["image"] == "/h/hide.jpg" and tv[-1]["content"][1]["text"].startswith("\nQ?"))
check("teacher view: placeholder/image count mismatch rejected", raises(lambda: P.teacher_view_messages([{"role": "user", "content": "<image>\n<image>\nQ"}], [{"path": "/h/a.jpg"}]), ValueError))
check("teacher view: no hint sentence (hide parquet form)", "zoom" not in tv[-1]["content"][1]["text"].lower())
sv = P.swap_view_messages([{"role": "user", "content": [{"type": "image", "image": "/s/box.png", "path": "/s/box.png"}, {"type": "text", "text": "Q? Only focus on the objects inside the red bounding box"}]}], [{"path": "/t/crop.png"}])
check("swap view: image replaced by teacher crop, text (incl. hint) unchanged", sv[-1]["content"][0]["image"] == "/t/crop.png" and "red bounding box" in sv[-1]["content"][1]["text"] and len(sv[-1]["content"]) == 2)
check("swap view: count mismatch rejected", raises(lambda: P.swap_view_messages([{"role": "user", "content": [{"type": "image", "image": "a"}]}], [{"path": "x"}, {"path": "y"}]), ValueError))
check("coverage 1.0 -> all rows", np.array_equal(P.coverage_select(10, 1.0, 1, 1), np.arange(10)))
s1 = P.coverage_select(96, 0.5, 42, 7); check("coverage 0.5 -> 48 sorted unique rows", len(s1) == 48 and len(set(s1.tolist())) == 48 and np.all(np.diff(s1) > 0))
check("coverage: same (seed, step) deterministic, different step differs", np.array_equal(s1, P.coverage_select(96, 0.5, 42, 7)) and not np.array_equal(s1, P.coverage_select(96, 0.5, 42, 8)))

print("[2] branch-row tensors")
R, PL = 16, 6
prompts = torch.arange(100, 100 + PL); pattn = torch.ones(PL, dtype=torch.long)
pre, cont = [1, 2, 3], [4, 5, 6, 7, 8]
flags = [False, True, True, False, False]
rt = P.make_branch_row_tensors(prompts, pattn, pre, cont, R, pad_token_id=0, leak_flags=flags)
check("responses = prefix+cont then pad", rt["responses"].tolist() == pre + cont + [0] * (R - 8))
check("response_mask = 1 on all real tokens (prefix included)", rt["response_mask"].tolist() == [1] * 8 + [0] * (R - 8))
check("loss mask: prefix 0, cont 1 minus leak", rt["pbd_loss_mask"].tolist() == [0, 0, 0, 1, 0, 0, 1, 1] + [0] * (R - 8))
check("input_ids = prompts + responses", rt["input_ids"].tolist() == prompts.tolist() + rt["responses"].tolist())
check("t_star = len(prefix)", int(rt["pbd_t_star"]) == 3)
rt2 = P.make_branch_row_tensors(prompts, pattn, pre, cont, R, 0, None)
check("no leak flags -> cont fully in loss", rt2["pbd_loss_mask"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1] + [0] * (R - 8))
rt3 = P.make_branch_row_tensors(prompts, pattn, pre, list(range(50)), R, 0, None)
check("truncation to response_length", rt3["responses"].shape[0] == R and rt3["response_mask"].sum().item() == R and int(rt3["pbd_n_resp"]) == R)
ra = torch.tensor([1] * 10 + [0] * 6)
check("parent keep: full response", P.parent_loss_mask(ra, 4, "keep").tolist() == ra.float().tolist())
check("parent replace: t < t* only", P.parent_loss_mask(ra, 4, "replace").tolist() == [1.0] * 4 + [0.0] * 12)
check("parent replace, not branched: full", P.parent_loss_mask(ra, None, "replace").tolist() == ra.float().tolist())

print("[3] config validation")
from verl.workers.config.actor import SelfDistillationConfig  # noqa: E402
base = dict(full_logit_distillation=True, distillation_topk=100)
SelfDistillationConfig(**base, pbd_enable=True); check("pbd defaults accepted (null mode None)", True)
check("pbd + mean_color rejected", raises(lambda: SelfDistillationConfig(**base, pbd_enable=True, counterfactual_null_mode="mean_color"), ValueError))
check("bad ratio rejected", raises(lambda: SelfDistillationConfig(**base, pbd_enable=True, pbd_ratio=1.0), ValueError))
check("bad coverage rejected", raises(lambda: SelfDistillationConfig(**base, pbd_enable=True, pbd_coverage=0.0), ValueError))
check("bad mode rejected", raises(lambda: SelfDistillationConfig(**base, pbd_enable=True, pbd_mode="drop"), ValueError))
check("pbd off: mean_color still fine", SelfDistillationConfig(**base, counterfactual_null_mode="mean_color").pbd_enable is False)
check("default leak mask off (13 §20.2b)", SelfDistillationConfig(**base).pbd_leak_mask is False)

print("[4] core_algos loss")
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402
class Cfg(dict):
    __getattr__ = dict.get
def cfg(**kw):
    d = dict(full_logit_distillation=True, distillation_topk=5, distillation_add_tail=True, alpha=0.5, is_clip=None,
             counterfactual_null_mode=None, counterfactual_extrapolation_beta=0.0, counterfactual_null_scope="all",
             counterfactual_u_clip_pos=False, counterfactual_reference="null", teacher_target_mode="legacy", pbd_lambda=0.5)
    d.update(kw); return Cfg(d)
torch.manual_seed(0)
B, T, K = 4, 7, 5
st = torch.log_softmax(torch.randn(B, T, 50), -1)[..., :K].requires_grad_(True)
te = torch.log_softmax(torch.randn(B, T, 50), -1)[..., :K]
mask = torch.ones(B, T)
def run_rows(rows, **kw):
    """call per row (micro-batch = row), return list of (loss, grad)"""
    out = []
    for i in rows:
        s_i = st[i:i+1]; t_i = te[i:i+1]
        res = compute_self_distillation_loss(student_log_probs=s_i.sum(-1), teacher_log_probs=t_i.sum(-1), response_mask=mask[i:i+1],
                self_distillation_config=cfg(), old_log_probs=s_i.sum(-1).detach(), student_topk_log_probs=s_i, teacher_topk_log_probs=t_i,
                self_distillation_mask=torch.ones(1), loss_agg_mode="token-mean", **{k: (v[i:i+1] if torch.is_tensor(v) else v) for k, v in kw.items()})
        out.append(res)
    return out
plain = run_rows(range(B))
offpbd = run_rows(range(B), pbd_loss_mask=None, pbd_is_branch=None, pbd_scale=1.0)
check("pbd args None -> bit-identical loss to the plain call", all(torch.equal(a[0], b[0]) for a, b in zip(plain, offpbd)))
is_branch = torch.tensor([0.0, 0.0, 1.0, 1.0]); lmask = torch.ones(B, T); scale = 4 / 2
on = run_rows(range(B), pbd_loss_mask=lmask, pbd_is_branch=is_branch, pbd_scale=scale)
total_on = torch.stack([r[0] for r in on]).mean()
expect = torch.stack([plain[0][0], plain[1][0]]).mean() + 0.5 * torch.stack([plain[2][0], plain[3][0]]).mean()
check("pbd on: mean over rows == mean_v0 + lambda * mean_branch", torch.allclose(total_on, expect, atol=1e-6), f"{total_on.item():.6f} vs {expect.item():.6f}")
# N_b != N_v0 (review 3): 3 V0 rows + 1 branch row -> scale_v0 = 4/3, scale_b = 4/1
is_b31 = torch.tensor([0.0, 0.0, 0.0, 1.0])
on31 = run_rows(range(B), pbd_loss_mask=lmask, pbd_is_branch=is_b31, pbd_scale=4 / 3, pbd_scale_b=4 / 1)
tot31 = torch.stack([r[0] for r in on31]).mean(); exp31 = torch.stack([plain[i][0] for i in range(3)]).mean() + 0.5 * plain[3][0]
check("pbd on, N_b != N_v0: mean over rows == mean_v0 + lambda * mean_branch", torch.allclose(tot31, exp31, atol=1e-6), f"{tot31.item():.6f} vs {exp31.item():.6f}")
# Replace parent (review 4): a parent keeping 2/7 tokens is weighted by 2/7
lmr = torch.ones(B, T); lmr[0, 2:] = 0
onr = run_rows([0], pbd_loss_mask=lmr, pbd_is_branch=is_branch, pbd_scale=1.0, pbd_scale_b=1.0)[0][0]
ref0 = run_rows([0], pbd_loss_mask=lmr, pbd_is_branch=None, pbd_scale=1.0)[0][0]   # same mask, no row weighting
check("replace parent weighted by kept-token fraction", torch.allclose(onr, ref0 * (2 / 7), atol=1e-6), f"{onr.item():.6f} vs {(ref0*(2/7)).item():.6f}")
lm2 = torch.ones(B, T); lm2[2, :3] = 0; lm2[3, 5] = 0   # prefix / leak positions dropped on branch rows
on2 = run_rows([2, 3], pbd_loss_mask=lm2, pbd_is_branch=is_branch, pbd_scale=scale)
check("loss mask drops prefix/leak positions (loss changes, token count shrinks)",
      not torch.equal(on2[0][0], on[2][0]) and on2[0][1]["self_distillation/num_distill_tokens"] == 4 and on2[1][1]["self_distillation/num_distill_tokens"] == 6)
m = on[2][1]
check("metrics: pbd_* keys present", all(k in m for k in ("self_distillation/pbd_jsd_branch_sum", "self_distillation/pbd_jsd_v0_cnt", "self_distillation/pbd_jsd_tail8_branch_sum", "self_distillation/pbd_scale")))
check("metrics: branch row counted as branch", m["self_distillation/pbd_jsd_branch_cnt"] == T and m["self_distillation/pbd_jsd_v0_cnt"] == 0 and on[0][1]["self_distillation/pbd_jsd_v0_cnt"] == T)
g_on = torch.autograd.grad(total_on, st, retain_graph=True)[0]; g_ex = torch.autograd.grad(expect, st, retain_graph=True)[0]
check("gradient matches the weighted formula", torch.allclose(g_on, g_ex, atol=1e-6))
check("pbd rows + null mode raises", raises(lambda: compute_self_distillation_loss(student_log_probs=st[:1].sum(-1), teacher_log_probs=te[:1].sum(-1), response_mask=mask[:1],
      self_distillation_config=cfg(counterfactual_null_mode="mean_color"), old_log_probs=st[:1].sum(-1).detach(), student_topk_log_probs=st[:1], teacher_topk_log_probs=te[:1],
      teacher_null_topk_log_probs=te[:1], teacher_null_log_probs=te[:1].sum(-1), self_distillation_mask=torch.ones(1), loss_agg_mode="token-mean",
      pbd_loss_mask=lmask[:1], pbd_is_branch=is_branch[:1], pbd_scale=1.0), ValueError))

print("[5] position ids")
ids1 = torch.tensor([[0, 0, 5, 6, 7, 8]]); am = torch.tensor([[0, 0, 1, 1, 1, 1]])
pos = P.compute_position_ids(None, ids1, am, {})
check("no processor: incremental positions over valid tokens", pos.shape == (1, 6) and pos[0, 2:].tolist() == [0, 1, 2, 3])

print("[6] source guards")
import inspect  # noqa: E402
try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    src = inspect.getsource(RayPPOTrainer._maybe_append_pbd_branch_rows)
    check("trainer: branch rows built from parent copies (select_idxs) and concatenated", "select_idxs" in src and "DataProto.concat" in src)
    check("trainer: rollout log-probs dropped so old_log_probs are recomputed", 'merged.batch.pop("rollout_log_probs")' in src)
    check("trainer: pbd_scale = N_total/N_v0", 'merged.meta_info["pbd_scale"] = float(n_total) / float(n_v0)' in src)
    check("trainer: crop view uses the LAST teacher image", "crops[i][-1]" in src)
    fit = inspect.getsource(RayPPOTrainer.fit)
    check("fit: branch phase right after union, before response_mask/balance", fit.find("_maybe_append_pbd_branch_rows") > fit.find("batch = batch.union(gen_batch_output)") and fit.find("_maybe_append_pbd_branch_rows") < fit.find("self._balance_batch(batch, metrics=metrics)"))
except Exception as e:
    if os.environ.get("PROBE_TEST_STRICT") == "1": check(f"trainer import ({type(e).__name__}: {e})", False)
    else: print(f"  SKIP  trainer source checks ({type(e).__name__}: {e})")
from verl.workers.actor import dp_actor  # noqa: E402
dsrc = inspect.getsource(dp_actor)
check("dp_actor: single mini-batch under pbd rows", "mini_batches = [data]" in dsrc and "_pbd_rows" in dsrc)
check("dp_actor: passes pbd tensors + scale into the loss", "pbd_loss_mask=model_inputs.get" in dsrc and 'data.meta_info.get("pbd_scale"' in dsrc)

print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAIL: {FAIL}")
sys.exit(1 if FAIL else 0)
