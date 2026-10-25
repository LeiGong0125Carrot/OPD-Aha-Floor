#!/usr/bin/env python
"""DG (disagreement-gated teacher-trajectory distillation) unit tests. CPU only (run on a compute node, not the login node).

① t=0 readout: top-k -> option distribution (renormalised, missing option -> 0, no option -> uniform + flag, off-option
   letters ignored); offline full-vocab readout == probe_disagree.t0_dist; TV
② prompt selection: gate = top-g among eligible (fixed tie order), random = same count, ineligible never chosen,
   deterministic per (seed, step), count capped by #eligible
③ offline teacher trajectory choice: only pool samples carrying the 8-sample majority; tie / no carrier -> None
④ vLLM first-position top-k conversion
⑤ trainer hook on a synthetic DataProto: replaces exactly the LAST rollout of the selected prompts with
   tokenizer(teacher text)+end-of-turn (responses / input_ids / attention / response_mask / position_ids), untouched rows
   bit-identical, rollout_log_probs zeroed on replaced rows, dg_is_teacher, DR same count; guards
⑥ config dataclass guards
⑦ build_dg_parquet on a tiny synthetic parquet + teacher dir

Run: PYTHONPATH=$REPO python scripts/test_dg.py
"""
import json, os, sys, tempfile, types
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verl.utils import dg as G  # noqa: E402

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True); cond or fails.append(name)

def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False

# ---------------- ① readout ----------------
LID = [101, 102, 103, 104, 105, 106]   # fake letter token ids A..F
lp = {101: np.log(0.2), 102: np.log(0.1), 103: np.log(0.05), 104: np.log(0.05), 999: np.log(0.5)}
p, info = G.letter_dist_from_topk(lp, LID, "ABCD")
check("① renormalised over options", np.allclose(p[:4], np.array([0.2, 0.1, 0.05, 0.05]) / 0.4) and p[4:].sum() == 0)
check("① letter mass / coverage", abs(info["letter_mass"] - 0.4) < 1e-9 and info["covered"] == 1.0 and info["no_letter"] == 0.0)
p, info = G.letter_dist_from_topk({101: np.log(0.3), 105: np.log(0.3)}, LID, "ABC")
check("① missing option -> 0, off-option E ignored", np.allclose(p, [1, 0, 0, 0, 0, 0]) and abs(info["covered"] - 1 / 3) < 1e-9)
p, info = G.letter_dist_from_topk({999: -0.1}, LID, "ABCD")
check("① no option in top-k -> uniform + flag", np.allclose(p[:4], 0.25) and info["no_letter"] == 1.0)
p, info = G.letter_dist_from_topk(None, LID, "AB")
check("① empty top-k -> uniform + flag", np.allclose(p[:2], 0.5) and info["no_letter"] == 1.0)
check("① bad option letter raises", raises(lambda: G.letter_dist_from_topk(lp, LID, "AZ"), ValueError))
from probe_disagree import t0_dist  # noqa: E402
ll = np.log(np.array([0.1, 0.3, 0.2, 0.05, 0.01, 0.01]))
check("① offline readout == probe_disagree.t0_dist", np.allclose(G.letter_dist_from_logp(ll, "ABCD"), t0_dist(ll, ["A", "B", "C", "D"])))
check("① tv", abs(G.tv([1, 0, 0], [0, 1, 0]) - 1) < 1e-12 and G.tv([0.5, 0.5], [0.5, 0.5]) == 0)

# ---------------- ② selection ----------------
g = np.array([0.9, 0.1, 0.8, 0.95, 0.2, 0.7, 0.3, 0.6])
el = np.array([1, 1, 1, 0, 1, 1, 1, 1], bool)
s = G.select_prompts(g, el, 0.25, "gate", 42, 5)
check("② gate: top-2 eligible by g (index 3 ineligible)", s.tolist() == [0, 2])
r = G.select_prompts(g, el, 0.25, "random", 42, 5)
check("② random: same count, eligible only", len(r) == 2 and el[r].all())
check("② random deterministic per (seed, step)", np.array_equal(r, G.select_prompts(g, el, 0.25, "random", 42, 5)))
rs = {tuple(G.select_prompts(g, el, 0.25, "random", 42, st)) for st in range(20)}
check("② random varies with step", len(rs) > 1)
gt = np.zeros(8); s1 = G.select_prompts(gt, np.ones(8, bool), 0.5, "gate", 42, 1)
check("② all ties -> fixed random tie order, count 4", len(s1) == 4 and np.array_equal(s1, G.select_prompts(gt, np.ones(8, bool), 0.5, "gate", 42, 1)))
check("② count capped by eligible", len(G.select_prompts(g, np.array([1, 0, 0, 0, 0, 0, 0, 0], bool), 0.5, "gate", 42, 1)) == 1)
check("② n_select(48, .25) = 12", G.n_select(48, 0.25) == 12)
check("② bad mode raises", raises(lambda: G.select_prompts(g, el, 0.25, "foo", 42, 1), ValueError))

# ---------------- ③ teacher trajectory choice ----------------
L = ["A", "A", "B", "A", "B", "A", "A", None]; T = [f"t{i}" for i in range(8)]
picks = {G.choose_teacher_traj(L, T, row, seed=42) for row in range(50)}
check("③ only pool samples with the majority letter (5, 6)", picks <= {5, 6} and len(picks) == 2)
check("③ deterministic per row", G.choose_teacher_traj(L, T, 7) == G.choose_teacher_traj(L, T, 7))
check("③ tie -> None", G.choose_teacher_traj(["A", "B"] * 4, T, 0) is None)
check("③ majority absent from pool -> None", G.choose_teacher_traj(["A", "A", "A", "A", "B", "C", None, None], T, 0) is None)
check("③ empty text skipped", G.choose_teacher_traj(L, ["x"] * 5 + ["", "y", "z"], 0) == 6)

# ---------------- ④ vLLM conversion ----------------
LP = types.SimpleNamespace
check("④ first_topk_from_vllm", G.first_topk_from_vllm({5: LP(logprob=-0.5, rank=1), 7: LP(logprob=-2.0, rank=2)}) == {5: -0.5, 7: -2.0})

# ---------------- ⑤ trainer hook ----------------
from omegaconf import OmegaConf  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # noqa: E402
from verl.utils.model import compute_position_id_with_mask  # noqa: E402

tok = AutoTokenizer.from_pretrained(os.environ.get("DG_TEST_TOKENIZER", "Qwen/Qwen3.5-4B"))
LIDS = [tok.encode(c, add_special_tokens=False)[0] for c in G.LETTERS]
END = tok.convert_tokens_to_ids("<|im_end|>")
check("⑤ tokenizer: letters single tokens, <|im_end|> special", all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in G.LETTERS)
      and END in tok.all_special_ids)

def fake_trainer(**sd):
    base = dict(dg_enable=True, dg_mode="gate", dg_frac=0.5, dg_topk=20, pbd_enable=False, tsw_enable=False,
                counterfactual_null_mode=None)
    base.update(sd)
    cfg = OmegaConf.create({"actor_rollout_ref": {"actor": {"self_distillation": base, "policy_loss": {"loss_mode": "vopd"},
                                                            "use_dynamic_bsz": False, "ppo_epochs": 1, "ppo_mini_batch_size": 4},
                                                  "rollout": {"n": 2}},
                            "algorithm": {"rollout_correction": {"rollout_is": "token", "rollout_rs": None, "bypass_mode": False,
                                                                 "rollout_is_batch_normalize": False}},
                            "data": {"seed": 42}})
    t = types.SimpleNamespace(config=cfg, tokenizer=tok, processor=None, global_steps=3, async_rollout_mode=True,
                              _DG_ROW_KEYS=RayPPOTrainer._DG_ROW_KEYS)
    t._dg_cfg = types.MethodType(RayPPOTrainer._dg_cfg, t)
    return t

PL, R = 6, 24
TEXTS = ["The answer is B.", "Looking closely at the crop, it is clearly D.", "C", "A. red"]
def make_batch(pS_top, elig=(1, 1, 1, 1), extra_tensor=False, drop_topk=False):
    """4 prompts x n=2 interleaved; prompt j: pT0 one-hot on 'D'; student top-k from pS_top[j]."""
    rows = []
    for j in range(4):
        for k in range(2):
            prompt = [tok.pad_token_id or 0] * 2 + tok.encode(f"question {j}", add_special_tokens=False)[:PL - 2]
            prompt = ([tok.pad_token_id or 0] * PL + prompt)[-PL:]
            resp = tok.encode(f"student {j} {k} says A", add_special_tokens=False) + [END]
            rows.append((prompt, resp))
    B = len(rows)
    prompts = torch.tensor([p for p, _ in rows]); pattn = (prompts != (tok.pad_token_id or 0)).long()
    responses = torch.full((B, R), tok.pad_token_id or 0, dtype=torch.long); rattn = torch.zeros(B, R, dtype=torch.long)
    for i, (_, r) in enumerate(rows):
        responses[i, :len(r)] = torch.tensor(r); rattn[i, :len(r)] = 1
    attn = torch.cat([pattn, rattn], 1); ids = torch.cat([prompts, responses], 1)
    td = {"prompts": prompts, "responses": responses, "response_mask": rattn.clone(), "input_ids": ids, "attention_mask": attn,
          "position_ids": compute_position_id_with_mask(attn), "rollout_log_probs": torch.randn(B, R)}
    if extra_tensor: td["mystery"] = torch.zeros(B, R)
    ei, topk = [], []
    for j in range(4):
        for k in range(2):
            ei.append(dict(dg_row=100 + j, dg_opts="ABCD", dg_pT0=[0, 0, 0, 1, 0, 0], dg_traj_ok=int(elig[j]), dg_traj=TEXTS[j],
                           answer="D"))
            topk.append({LIDS[c]: float(np.log(v)) for c, v in pS_top[j].items()})
    nt = {"uid": np.array([f"u{j}" for j in range(4) for _ in range(2)], dtype=object), "extra_info": np.array(ei, dtype=object)}
    if not drop_topk:
        a = np.empty(B, dtype=object); a[:] = topk; nt["dg_first_topk"] = a
    return DataProto(batch=TensorDict(td, batch_size=(B,)), non_tensor_batch=nt)

# prompt 0: student on D (agree, g~0); 1: student on A (g=1); 2: student split A/D (g=.5); 3: student on B (g=1, ineligible below)
PS = [{3: 0.9}, {0: 0.9}, {0: 0.4, 3: 0.4}, {1: 0.9}]
b = make_batch(PS); ref = b.batch.clone()
apply = RayPPOTrainer._maybe_apply_dg
m = apply(fake_trainer(), b)
rep = b.batch["dg_is_teacher"].nonzero().flatten().tolist()
check("⑤ gate frac .5 of 4 prompts -> 2 prompts (1, 3: g = 1), last rollout replaced", rep == [3, 7] and m["dg/replaced_rows"] == 2)
check("⑤ g metrics", abs(m["dg/g_selected_mean"] - 1.0) < 1e-9 and m["dg/g_rest_mean"] < 0.5)
for r_, j in ((3, 1), (7, 3)):
    want = tok.encode(TEXTS[j], add_special_tokens=False) + [END]; n = len(want)
    check(f"⑤ row {r_}: responses = teacher text + <|im_end|>", b.batch["responses"][r_, :n].tolist() == want
          and (b.batch["responses"][r_, n:] == (tok.pad_token_id or 0)).all())
    check(f"⑤ row {r_}: masks / input_ids / position_ids consistent",
          b.batch["response_mask"][r_].sum() == n and b.batch["attention_mask"][r_, PL:].sum() == n
          and torch.equal(b.batch["attention_mask"][r_, :PL], ref["attention_mask"][r_, :PL])
          and torch.equal(b.batch["input_ids"][r_], torch.cat([ref["prompts"][r_], b.batch["responses"][r_]]))
          and torch.equal(b.batch["position_ids"][r_], compute_position_id_with_mask(b.batch["attention_mask"][r_:r_ + 1])[0])
          and (b.batch["rollout_log_probs"][r_] == 0).all())
keep = [i for i in range(8) if i not in rep]
check("⑤ untouched rows bit-identical", all(torch.equal(b.batch[k][keep], ref[k][keep]) for k in ref.keys()))
check("⑤ prompts never modified", torch.equal(b.batch["prompts"], ref["prompts"]))
check("⑤ leak monitored on the teacher text ('crop' in prompt 1's)", m["dg/teacher_leak_row_frac"] == 0.5)

b = make_batch(PS, elig=(1, 1, 1, 0)); m = apply(fake_trainer(), b)
check("⑤ ineligible prompt skipped -> next best g (prompt 2)", b.batch["dg_is_teacher"].nonzero().flatten().tolist() == [3, 5])
b = make_batch(PS); m = apply(fake_trainer(dg_mode="random"), b)
check("⑤ DR: same count, one row per prompt, always the last rollout", m["dg/replaced_rows"] == 2
      and all(i % 2 == 1 for i in b.batch["dg_is_teacher"].nonzero().flatten().tolist()))
b = make_batch(PS); m = apply(fake_trainer(dg_enable=False), b)
check("⑤ dg off -> no-op", m == {} and "dg_is_teacher" not in b.batch.keys())
check("⑤ missing dg_first_topk raises", raises(lambda: apply(fake_trainer(), make_batch(PS, drop_topk=True)), ValueError))
check("⑤ unknown per-token tensor raises", raises(lambda: apply(fake_trainer(), make_batch(PS, extra_tensor=True)), RuntimeError))
_tm = fake_trainer(); _tm.config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
check("⑤ multi mini-batch raises", raises(lambda: apply(_tm, make_batch(PS)), ValueError))
check("⑤ tsw on raises", raises(lambda: apply(fake_trainer(tsw_enable=True), make_batch(PS)), ValueError))
check("⑤ null mode raises", raises(lambda: apply(fake_trainer(counterfactual_null_mode="mean_color"), make_batch(PS)), ValueError))
check("⑤ rollout_is=token (as the V0 launcher) accepted", "dg/replaced_rows" in apply(fake_trainer(), make_batch(PS)))
_tr = fake_trainer(); _tr.config.algorithm.rollout_correction.rollout_rs = "token"
check("⑤ rollout_rs raises", raises(lambda: apply(_tr, make_batch(PS)), ValueError))
_tr = fake_trainer(); _tr.config.actor_rollout_ref.rollout.name = "sglang"
check("⑤ non-vLLM rollout raises", raises(lambda: apply(_tr, make_batch(PS)), ValueError))
m = apply(fake_trainer(), make_batch(PS))
check("⑤ style/dose metrics", 0 <= m["dg/teacher_direct_frac"] <= 1 and 0 < m["dg/replaced_token_frac"] < 1)
m = apply(fake_trainer(), make_batch(PS, elig=(0, 0, 0, 0)))
check("⑤ nothing eligible -> no replacement, replaced keys still logged", m["dg/replaced_rows"] == 0 and "dg/replaced_token_frac" in m)
bb = make_batch(PS); bb.batch["responses"][:, :] = 5; bb.batch["attention_mask"][:, PL:PL + 3] = 1; bb.batch["attention_mask"][:, PL + 3:] = 0
check("⑤ finished student responses not ending in <|im_end|> raises", raises(lambda: apply(fake_trainer(), bb), RuntimeError))
bb = make_batch(PS); bb.non_tensor_batch["uid"][1] = "zz"
check("⑤ uid group size != n raises", raises(lambda: apply(fake_trainer(), bb), RuntimeError))
bb = make_batch(PS); bb.non_tensor_batch["extra_info"][1] = dict(bb.non_tensor_batch["extra_info"][1], dg_row=999)
check("⑤ rollouts of one uid from different questions raises", raises(lambda: apply(fake_trainer(), bb), RuntimeError))
bb = make_batch(PS); bb.non_tensor_batch["extra_info"][0] = {"answer": "D"}
check("⑤ extra_info without dg_* raises", raises(lambda: apply(fake_trainer(), bb), ValueError))

# ---------------- ⑥ config ----------------
from verl.workers.config.actor import SelfDistillationConfig  # noqa: E402
check("⑥ default off ok", SelfDistillationConfig().dg_enable is False)
check("⑥ dg on ok", SelfDistillationConfig(dg_enable=True).dg_mode == "gate")
check("⑥ bad mode raises", raises(lambda: SelfDistillationConfig(dg_enable=True, dg_mode="x")))
check("⑥ frac 0 raises", raises(lambda: SelfDistillationConfig(dg_enable=True, dg_frac=0.0)))
check("⑥ topk 50 raises", raises(lambda: SelfDistillationConfig(dg_enable=True, dg_topk=50)))
check("⑥ + tsw raises", raises(lambda: SelfDistillationConfig(dg_enable=True, tsw_enable=True)))

# ---------------- ⑦ builder ----------------
import pyarrow as pa, pyarrow.parquet as pq  # noqa: E402
from build_dg_parquet import build  # noqa: E402
with tempfile.TemporaryDirectory() as d:
    qs = ["What color?\nA. red\nB. blue\nC. green\nD. black", "How many?\nA. 1\nB. 2"]
    t = pa.table({"prompt": ["p0", "p1"], "images": [[{"path": "/i0.png"}], [{"path": "/i1.png"}]],
                  "extra_info": [{"question": qs[0], "answer": "C"}, {"question": qs[1], "answer": "A"}]})
    pq.write_table(t, f"{d}/in.parquet"); os.makedirs(f"{d}/T")
    gen = [dict(row=0, group="/i0.png", gt="C", opts=["A", "B", "C", "D"], letters=["C", "C", "B", "C", "B", "C", "C", "D"],
                texts=[f"s{i} C" for i in range(8)]),
           dict(row=1, group="/i1.png", gt="A", opts=["A", "B"], letters=["A", "B"] * 4, texts=["x"] * 8)]
    json.dump(gen, open(f"{d}/T/gen.json", "w"))
    np.savez(f"{d}/T/readout.npz", rows=np.array([1, 0]), letter_logp=np.log(np.array([[.5, .5, .1, .1, .1, .1], [.1, .1, .6, .2, .1, .1]])))
    info = build(f"{d}/in.parquet", f"{d}/T", f"{d}/out.parquet")
    o = pq.read_table(f"{d}/out.parquet"); e = o.column("extra_info").to_pylist()
    check("⑦ row 0: traj from pool majority C (5 or 6), pT0 renorm over ABCD (rows reordered)",
          e[0]["dg_traj_ok"] == 1 and e[0]["dg_traj_s"] in (5, 6) and np.allclose(e[0]["dg_pT0"], [.1 / 1, .1 / 1, .6 / 1, .2 / 1, 0, 0]))
    check("⑦ row 1: tie -> no traj; pT0 over AB", e[1]["dg_traj_ok"] == 0 and e[1]["dg_traj"] == "" and np.allclose(e[1]["dg_pT0"][:2], .5))
    check("⑦ other columns unchanged + original extra_info kept", o.column("prompt").equals(t.column("prompt")) and e[0]["answer"] == "C")
    check("⑦ meta counts", info["traj_ok"] == 1 and info["no_majority"] == 1)
    bad = [dict(gen[0], gt="D"), gen[1]]; json.dump(bad, open(f"{d}/T/gen.json", "w"))
    check("⑦ identity mismatch raises", raises(lambda: build(f"{d}/in.parquet", f"{d}/T", f"{d}/o2.parquet"), AssertionError))

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
