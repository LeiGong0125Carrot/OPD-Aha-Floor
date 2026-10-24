#!/usr/bin/env python
"""TSW (question-level teacher-stability weighting) unit tests. CPU only.

① trainer row weights: lam=1 -> all ones; soft/hard normalise to row-mean 1 over valid rows; invalid rows 0;
   all valid rows in a zero-weight set -> raise
② _maybe_attach_tsw_weight on a DataProto: reads extra_info[key], respects self_distillation_mask, guards
   (missing key, non-0/1 value, PBD on, multi mini-batch, dynamic bsz, tsw off -> no tensor)
③ loss: tsw_weight=None and all-ones are bit-identical; per-row losses scale by the row weight; a zero-weight row has
   zero gradient; per-row (micro-batch=1) aggregation with all-ones weights == unweighted V0 mean
④ build_tsw_parquet: low25 = lowest scores with fixed tie order, rand25 size, other columns unchanged, identity check
⑤ config dataclass guards

Run: PYTHONPATH=$REPO python scripts/test_tsw.py
"""
import json, os, sys, tempfile, types
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True); cond or fails.append(name)

def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False

# ---------------- ① trainer row weights ----------------
from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # noqa: E402
W = RayPPOTrainer._tsw_row_weights
mem = np.array([1, 0, 0, 0, 1, 0, 0, 0], bool); val = np.ones(8, bool)
w, m = W(mem, val, 1.0)
check("① lam=1 -> all ones", np.array_equal(w, np.ones(8, np.float32)) and m["tsw/norm_coef"] == 1.0)
w, m = W(mem, val, 0.5)
check("① soft: row mean 1, member/non-member ratio 0.5", abs(w.mean() - 1) < 1e-6 and abs(w[0] / w[1] - 0.5) < 1e-6)
w, m = W(mem, val, 0.0)
check("① hard: members 0, others 8/6", np.all(w[mem] == 0) and np.allclose(w[~mem], 8 / 6) and m["tsw/zero_weight_rows"] == 2)
val2 = val.copy(); val2[1] = False
w, m = W(mem, val2, 0.5)
check("① invalid row -> 0, mean over valid rows = 1", w[1] == 0 and abs(w[val2].mean() - 1) < 1e-6)
check("① all valid rows in zero set -> raise", raises(lambda: W(np.ones(4, bool), np.ones(4, bool), 0.0), ValueError))

# ---------------- ② attach on a DataProto ----------------
from omegaconf import OmegaConf  # noqa: E402
from verl import DataProto  # noqa: E402

def fake_trainer(**sd):
    base = dict(tsw_enable=True, tsw_key="tsw_low25", tsw_lambda=0.5, pbd_enable=False)
    base.update(sd)
    cfg = OmegaConf.create({"actor_rollout_ref": {"actor": {"self_distillation": base, "policy_loss": {"loss_mode": "vopd"},
                                                            "use_dynamic_bsz": False, "ppo_epochs": 1, "ppo_mini_batch_size": 4},
                                                  "rollout": {"n": 2}}})
    return types.SimpleNamespace(config=cfg, _tsw_row_weights=RayPPOTrainer._tsw_row_weights)

def make_batch(flags, key="tsw_low25", sdm=None):
    ei = np.array([{key: f, "answer": "A"} for f in flags], dtype=object)
    t = {"self_distillation_mask": torch.tensor(sdm if sdm is not None else [1.0] * len(flags))}
    return DataProto.from_dict(tensors=t, non_tensors={"extra_info": ei})

attach = RayPPOTrainer._maybe_attach_tsw_weight
flags = [1, 1, 0, 0, 0, 0, 0, 0]          # one question (2 rollouts) of 4 in the set
b = make_batch(flags); m = attach(fake_trainer(), b)
tw = b.batch["tsw_weight"].numpy()
check("② weights: 2 rollouts of the member share the weight; mean 1", tw[0] == tw[1] and abs(tw.mean() - 1) < 1e-6 and abs(tw[0] / tw[2] - 0.5) < 1e-6)
check("② metrics present", abs(m["tsw/weight_mean"] - 1) < 1e-6 and m["tsw/member_row_frac"] == 0.25 and m["tsw/lambda"] == 0.5)
b = make_batch(flags, sdm=[1, 1, 0, 1, 1, 1, 1, 1]); attach(fake_trainer(), b)
check("② self_distillation_mask=0 row gets 0", b.batch["tsw_weight"][2].item() == 0.0)
b = make_batch(flags); r = attach(fake_trainer(tsw_enable=False), b)
check("② tsw off -> no tensor, no metrics", "tsw_weight" not in b.batch.keys() and r == {})
check("② missing key -> raise", raises(lambda: attach(fake_trainer(tsw_key="tsw_rand25"), make_batch(flags)), ValueError))
check("② non-0/1 value -> raise", raises(lambda: attach(fake_trainer(), make_batch([0.3] + flags[1:])), ValueError))
check("② pbd on -> raise", raises(lambda: attach(fake_trainer(pbd_enable=True), make_batch(flags)), ValueError))
check("② lambda out of range -> raise", raises(lambda: attach(fake_trainer(tsw_lambda=1.5), make_batch(flags)), ValueError))
check("② batch != one mini-batch -> raise", raises(lambda: attach(fake_trainer(), make_batch(flags[:6])), ValueError))
ft = fake_trainer(); ft.config.actor_rollout_ref.actor.use_dynamic_bsz = True
check("② dynamic bsz -> raise", raises(lambda: attach(ft, make_batch(flags)), ValueError))

# ---------------- ③ loss ----------------
class Cfg(dict):
    __getattr__ = dict.get

C = Cfg({"full_logit_distillation": True, "distillation_topk": 8, "distillation_add_tail": True, "renorm_topk_log_probs": False,
         "alpha": 0.5, "is_clip": None, "counterfactual_null_mode": None, "counterfactual_extrapolation_beta": 0.0,
         "log_prob_dump_dir": None})
torch.manual_seed(0); B, T, K = 4, 5, 8
def topk():
    return torch.log_softmax(torch.randn(B, T, K + 6), -1).sort(-1, descending=True).values[..., :K]
st0, te0 = topk(), topk(); lp = torch.randn(B, T) * 0.1 - 1.0
mask = torch.ones(B, T); mask[2, 3:] = 0

def loss(rows=slice(None), w=None, st=None):
    s = (st if st is not None else st0)[rows]
    return compute_self_distillation_loss(
        student_log_probs=lp[rows], teacher_log_probs=lp[rows].clone(), response_mask=mask[rows], self_distillation_config=C,
        old_log_probs=lp[rows].clone(), student_topk_log_probs=s, teacher_topk_log_probs=te0[rows],
        self_distillation_mask=torch.ones(s.shape[0]), loss_agg_mode="token-mean", tsw_weight=w)

l0, m0 = loss(); l1, m1 = loss(w=torch.ones(B))
check("③ all-ones == None (bit-identical loss)", torch.equal(l0, l1) and m0["self_distillation/raw_jsd_token_mean"] == m1["self_distillation/raw_jsd_token_mean"])
per_row = [loss(rows=slice(i, i + 1))[0] for i in range(B)]
ww = torch.tensor([0.5, 2.0, 0.0, 1.5])
per_row_w = [loss(rows=slice(i, i + 1), w=ww[i:i + 1])[0] for i in range(B)]
check("③ per-row loss scales by the row weight", all(torch.allclose(per_row_w[i], ww[i] * per_row[i]) for i in range(B)))
v0 = sum(per_row) / B; tw = sum(ww[i] * per_row[i] for i in range(B)) / B
check("③ micro-batch=1 aggregation: all-ones == V0 mean; weighted == mean(w_i L_i)",
      torch.allclose(sum(loss(rows=slice(i, i + 1), w=torch.ones(1))[0] for i in range(B)) / B, v0) and torch.allclose(sum(per_row_w) / B, tw))
st = st0.clone().requires_grad_(True)
lw, mw = loss(w=ww, st=st); lw.backward()
check("③ zero-weight row has zero gradient, others nonzero", st.grad[2].abs().sum().item() == 0.0 and st.grad[0].abs().sum().item() > 0)
check("③ metric tsw_row_weight / tsw_zero_row_frac", abs(mw["self_distillation/tsw_row_weight"] - ww.mean().item()) < 1e-6 and abs(mw["self_distillation/tsw_zero_row_frac"] - 0.25) < 1e-6)
check("③ wrong shape -> raise", raises(lambda: loss(w=torch.ones(B + 1)), ValueError))

# ---------------- ④ build_tsw_parquet ----------------
import pyarrow as pa, pyarrow.parquet as pq  # noqa: E402
import build_tsw_parquet as BP  # noqa: E402
d = tempfile.mkdtemp(); N = 40
rng = np.random.RandomState(1)
gts = [rng.choice(list("ABCD")) for _ in range(N)]
qs = [f"q{i}?\nA. x\nB. y\nC. z\nD. w\nAnswer with the option's letter from the given choices." for i in range(N)]
src = pa.table({"prompt": [f"p{i}" for i in range(N)], "images": [[{"path": f"i{i}.png"}] for i in range(N)],
                "extra_info": [{"answer": gts[i], "question": qs[i], "source_extra_info": {"answer": gts[i], "question": qs[i]}} for i in range(N)]})
pq.write_table(src, os.path.join(d, "train.parquet"))
p = rng.rand(N) * 0.8 + 0.1; p[[3, 7, 11]] = 0.05      # three-way tie at the bottom; k = 10
perm = rng.permutation(N)                                 # scores stored in a shuffled row order
grp = np.array([f"i{i}.png" for i in range(N)])
np.savez(os.path.join(d, "scores.npz"), row=perm, p=p[perm], share8=np.full(N, 0.75), gt=np.array(gts)[perm], opts=np.array(["ABCD"] * N)[perm], group=grp[perm])
info = BP.build(os.path.join(d, "train.parquet"), os.path.join(d, "scores.npz"), os.path.join(d, "out.parquet"))
out = pq.read_table(os.path.join(d, "out.parquet")); ei = out.column("extra_info").to_pylist()
low = sorted(i for i in range(N) if ei[i]["tsw_low25"] == 1)
check("④ low25 = 10 lowest scores", low == sorted(np.argsort(p)[:10].tolist()) and info["k"] == 10)
check("④ rand25 size 10, tsw_row identity, tsw_p aligned", sum(e["tsw_rand25"] for e in ei) == 10 and all(e["tsw_row"] == i and abs(e["tsw_p"] - p[i]) < 1e-12 for i, e in enumerate(ei)))
check("④ other columns and original extra_info keys unchanged", out.column("prompt").equals(src.column("prompt")) and out.column("images").equals(src.column("images"))
      and all(ei[i]["answer"] == gts[i] and ei[i]["source_extra_info"]["question"] == qs[i] for i in range(N)))
tie = np.full(N, 0.5); a1 = BP.lowest_k(tie, 5); a2 = BP.lowest_k(tie, 5)
check("④ tie order fixed (deterministic)", np.array_equal(a1, a2))
bad = np.array(gts)[perm].copy(); bad[0] = "E"
np.savez(os.path.join(d, "bad.npz"), row=perm, p=p[perm], share8=np.full(N, 0.75), gt=bad, opts=np.array(["ABCD"] * N)[perm])
check("④ gt mismatch -> AssertionError", raises(lambda: BP.build(os.path.join(d, "train.parquet"), os.path.join(d, "bad.npz"), os.path.join(d, "o2.parquet")), AssertionError))
g2 = grp[perm].copy(); g2[[0, 1]] = g2[[1, 0]]
np.savez(os.path.join(d, "badg.npz"), row=perm, p=p[perm], share8=np.full(N, 0.75), gt=np.array(gts)[perm], opts=np.array(["ABCD"] * N)[perm], group=g2)
check("④ image-path (group) mismatch -> AssertionError", raises(lambda: BP.build(os.path.join(d, "train.parquet"), os.path.join(d, "badg.npz"), os.path.join(d, "o3.parquet")), AssertionError))
check("④ sets.json written with hashes", "sha256" in json.load(open(os.path.join(d, "out.parquet.sets.json"))))

# ---------------- ⑤ config ----------------
from verl.workers.config.actor import SelfDistillationConfig  # noqa: E402
check("⑤ default tsw off", SelfDistillationConfig().tsw_enable is False)
def raises_tsw(fn):
    try: fn()
    except ValueError as e: return "tsw" in str(e).lower()
    return False
check("⑤ lambda > 1 rejected", raises_tsw(lambda: SelfDistillationConfig(tsw_enable=True, tsw_lambda=2.0)))
check("⑤ tsw + pbd rejected", raises_tsw(lambda: SelfDistillationConfig(tsw_enable=True, pbd_enable=True)))
check("⑤ valid tsw config accepted", not raises(lambda: SelfDistillationConfig(tsw_enable=True, tsw_lambda=0.0, tsw_key="tsw_rand25")))

print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
