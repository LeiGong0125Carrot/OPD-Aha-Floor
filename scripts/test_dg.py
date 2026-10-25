#!/usr/bin/env python
"""DG (disagreement-gated teacher-trajectory distillation) unit tests. CPU only (run on a compute node, not the login node).

① t=0 readout: top-k -> option distribution (renormalised, missing option -> 0, no option -> uniform + flag, off-option
   letters ignored); offline full-vocab readout == probe_disagree.t0_dist; TV
② prompt selection: gate = top-g among eligible (fixed tie order), random = same count, ineligible never chosen,
   deterministic per (seed, step), count capped by #eligible
③ answer parser (= probe_hidden_answer.parse_letter)
④ vLLM first-position top-k conversion
⑤ trainer hook on a synthetic DataProto with a mocked rollout manager: the teacher-view generation request (swapped
   images, empty prefix, pbd_branch loop, no top-k), replaces exactly the LAST rollout of the selected prompts with the
   generated ids (responses / input_ids / attention / response_mask / position_ids), untouched rows bit-identical,
   rollout_log_probs zeroed on replaced rows, dg_is_teacher, DR same count, monitoring metrics; guards
⑥ config dataclass guards
⑦ build_dg_parquet on a tiny synthetic parquet + teacher dir (p_T0 only)

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

# ---------------- ③ answer parser ----------------
from probe_hidden_answer import parse_letter as ref_parse  # noqa: E402
cases = ["<answer>C</answer>", "D. purple", "The answer is B.", "so option A", "I think (C)", "A big dog", "", "\n\n**B. white**"]
check("③ parse_letter == probe_hidden_answer.parse_letter", all(G.parse_letter(c, list("ABCD")) == ref_parse(c, list("ABCD")) for c in cases))
check("③ direct-answer detector", G.is_direct_answer("B. white") and G.is_direct_answer("**D**") and G.is_direct_answer("<answer>C</answer>")
      and G.is_direct_answer("\n\nA") and not G.is_direct_answer("A big dog is in the box") and not G.is_direct_answer("The answer is B."))
check("③ parse examples", G.parse_letter("D. purple", list("ABCD")) == "D" and G.parse_letter("A big dog", list("ABCD")) is None)

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
PAD = tok.pad_token_id or 0
check("⑤ tokenizer: letters single tokens, <|im_end|> special", all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in G.LETTERS)
      and END in tok.all_special_ids)

PL, R = 6, 24
GEN_TEXT = "Zooming into the crop, the floor is purple. D. purple"

class FakeManager:
    """Mocks async_rollout_manager.generate_sequences: records the request, answers every row with GEN_TEXT + <|im_end|>
    (row 'empty' -> nothing, row 'long' -> R+5 tokens without end)."""
    def __init__(self, special=None):
        self.req = None; self.special = special or {}
    def generate_sequences(self, gen):
        self.req = gen
        n = len(gen); resp = torch.full((n, R), PAD, dtype=torch.long); mask = torch.zeros(n, R, dtype=torch.long)
        for i in range(n):
            u = gen.non_tensor_batch["uid"][i] if "uid" in gen.non_tensor_batch else None
            kind = self.special.get(u, "ok")
            ids = {"ok": tok.encode(GEN_TEXT, add_special_tokens=False) + [END], "empty": [],
                   "long": tok.encode("word " * (R + 5), add_special_tokens=False)[:R]}[kind]
            resp[i, :len(ids)] = torch.tensor(ids, dtype=torch.long) if ids else resp[i, :0]; mask[i, :len(ids)] = 1
        return DataProto(batch=TensorDict({"responses": resp, "response_mask": mask}, batch_size=(n,)), meta_info={"timing": {"gen": 0.1}})

def fake_trainer(special=None, **sd):
    base = dict(dg_enable=True, dg_mode="gate", dg_frac=0.5, dg_topk=20, pbd_enable=False, tsw_enable=False,
                counterfactual_null_mode=None, teacher_image_key="bbox_images")
    base.update(sd)
    cfg = OmegaConf.create({"actor_rollout_ref": {"actor": {"self_distillation": base, "policy_loss": {"loss_mode": "vopd"},
                                                            "use_dynamic_bsz": False, "ppo_epochs": 1, "ppo_mini_batch_size": 4},
                                                  "rollout": {"n": 2, "name": "vllm", "agent": {"num_workers": 3}}},
                            "algorithm": {"rollout_correction": {"rollout_is": "token", "rollout_rs": None, "bypass_mode": False,
                                                                 "rollout_is_batch_normalize": False}},
                            "data": {"seed": 42}})
    t = types.SimpleNamespace(config=cfg, tokenizer=tok, processor=None, global_steps=3, async_rollout_mode=True,
                              _DG_ROW_KEYS=RayPPOTrainer._DG_ROW_KEYS, async_rollout_manager=FakeManager(special))
    t._dg_cfg = types.MethodType(RayPPOTrainer._dg_cfg, t)
    return t

def make_batch(pS_top, extra_tensor=False, drop_topk=False):
    """4 prompts x n=2 interleaved; prompt j: pT0 one-hot on 'D'; student top-k from pS_top[j]; student answers 'A'."""
    rows = []
    for j in range(4):
        for k in range(2):
            prompt = ([PAD] * PL + tok.encode(f"question {j}", add_special_tokens=False))[-PL:]
            resp = tok.encode(f"student {j} {k} says A", add_special_tokens=False) + [END]
            rows.append((prompt, resp))
    B = len(rows)
    prompts = torch.tensor([p for p, _ in rows]); pattn = (prompts != PAD).long()
    responses = torch.full((B, R), PAD, dtype=torch.long); rattn = torch.zeros(B, R, dtype=torch.long)
    for i, (_, r) in enumerate(rows):
        responses[i, :len(r)] = torch.tensor(r); rattn[i, :len(r)] = 1
    attn = torch.cat([pattn, rattn], 1); ids = torch.cat([prompts, responses], 1)
    td = {"prompts": prompts, "responses": responses, "response_mask": rattn.clone(), "input_ids": ids, "attention_mask": attn,
          "position_ids": compute_position_id_with_mask(attn), "rollout_log_probs": torch.randn(B, R)}
    if extra_tensor: td["mystery"] = torch.zeros(B, R)
    ei, topk, raw, crops = [], [], [], []
    for j in range(4):
        for k in range(2):
            ei.append(dict(dg_row=100 + j, dg_opts="ABCD", dg_pT0=[0, 0, 0, 1, 0, 0], answer="D"))
            topk.append({LIDS[c]: float(np.log(v)) for c, v in pS_top[j].items()})
            raw.append([{"role": "user", "content": [{"type": "image", "image": f"/full{j}.png"},
                                                     {"type": "text", "text": f"Look at the red box. question {j}"}]}])
            crops.append([{"path": f"/crop{j}.png"}])
    def obj(xs):
        a = np.empty(len(xs), dtype=object); a[:] = xs; return a
    nt = {"uid": np.array([f"u{j}" for j in range(4) for _ in range(2)], dtype=object), "extra_info": obj(ei),
          "raw_prompt": obj(raw), "bbox_images": obj(crops)}
    if not drop_topk:
        nt["dg_first_topk"] = obj(topk)
    return DataProto(batch=TensorDict(td, batch_size=(B,)), non_tensor_batch=nt)

# prompt 0: student on D (agree, g~0); 1: student on A (g=1); 2: student split A/D (g=.5); 3: student on B (g=1)
PS = [{3: 0.9}, {0: 0.9}, {0: 0.4, 3: 0.4}, {1: 0.9}]
apply = RayPPOTrainer._maybe_apply_dg
T = fake_trainer(); b = make_batch(PS); ref = b.batch.clone(); tr = {}
m = apply(T, b, tr)
rep = b.batch["dg_is_teacher"].nonzero().flatten().tolist()
check("⑤ gate frac .5 of 4 prompts -> prompts 1, 3 (g = 1), last rollout replaced", rep == [3, 7] and m["dg/replaced_rows"] == 2)
check("⑤ g metrics", abs(m["dg/g_selected_mean"] - 1.0) < 1e-9 and m["dg/g_rest_mean"] < 0.5)
q = T.async_rollout_manager.req
qn = q.non_tensor_batch
check("⑤ request padded to num_workers (2 -> 3), pbd_branch loop, empty prefix, full response budget, no top-k",
      len(q) == 3 and all(a == "pbd_branch" for a in qn["agent_name"][:2]) and all(len(p) == 0 for p in qn["pbd_prefix_ids"][:2])
      and all(int(x) == R for x in qn["pbd_max_cont_len"][:2]) and q.meta_info.get("dg_no_topk") is True and q.meta_info["validate"] is False)
msg = qn["raw_prompt"][0][-1]["content"]
check("⑤ teacher view = student text with images swapped for the crop (as the scoring teacher)",
      msg[0]["type"] == "image" and msg[0]["image"] == "/crop1.png" and msg[1]["text"] == "Look at the red box. question 1"
      and list(qn["uid"][:2]) == ["u1", "u3"])
check("⑤ dg_gen timing recorded", "dg_gen" in tr and "dg_gen/gen" in tr)
want = tok.encode(GEN_TEXT, add_special_tokens=False) + [END]; n = len(want)
for r_ in rep:
    check(f"⑤ row {r_}: responses = generated ids", b.batch["responses"][r_, :n].tolist() == want and (b.batch["responses"][r_, n:] == PAD).all())
    check(f"⑤ row {r_}: masks / input_ids / position_ids consistent, rollout log-probs zeroed",
          b.batch["response_mask"][r_].sum() == n and b.batch["attention_mask"][r_, PL:].sum() == n
          and torch.equal(b.batch["attention_mask"][r_, :PL], ref["attention_mask"][r_, :PL])
          and torch.equal(b.batch["input_ids"][r_], torch.cat([ref["prompts"][r_], b.batch["responses"][r_]]))
          and torch.equal(b.batch["position_ids"][r_], compute_position_id_with_mask(b.batch["attention_mask"][r_:r_ + 1])[0])
          and (b.batch["rollout_log_probs"][r_] == 0).all())
keep = [i for i in range(8) if i not in rep]
check("⑤ untouched rows bit-identical", all(torch.equal(b.batch[k][keep], ref[k][keep]) for k in ref.keys()))
check("⑤ prompts never modified", torch.equal(b.batch["prompts"], ref["prompts"]))
check("⑤ monitoring: leak ('Zooming', 'crop'), answer differs (D vs A), finished, not direct",
      m["dg/teacher_leak_row_frac"] == 1.0 and m["dg/teacher_student_answer_differ"] == 1.0 and m["dg/answer_parsed_pairs"] == 2
      and m["dg/teacher_unfinished_frac"] == 0.0 and m["dg/teacher_direct_frac"] == 0.0 and 0 < m["dg/replaced_token_frac"] < 1)

T = fake_trainer(special={"u3": "empty", "u1": "long"}); b = make_batch(PS); ref = b.batch.clone(); m = apply(T, b, {})
check("⑤ empty generation keeps the student row; long one truncated to R and counted unfinished",
      b.batch["dg_is_teacher"].nonzero().flatten().tolist() == [3] and m["dg/teacher_empty"] == 1
      and torch.equal(b.batch["responses"][7], ref["responses"][7]) and b.batch["attention_mask"][3, PL:].sum() == R
      and m["dg/teacher_unfinished_frac"] == 1.0)
T = fake_trainer(dg_mode="random"); b = make_batch(PS); m = apply(T, b, {})
check("⑤ DR: same count, one row per prompt, always the last rollout", m["dg/replaced_rows"] == 2
      and all(i % 2 == 1 for i in b.batch["dg_is_teacher"].nonzero().flatten().tolist()))
b = make_batch(PS); m = apply(fake_trainer(dg_enable=False), b, {})
check("⑤ dg off -> no-op", m == {} and "dg_is_teacher" not in b.batch.keys())
check("⑤ missing dg_first_topk raises", raises(lambda: apply(fake_trainer(), make_batch(PS, drop_topk=True), {}), ValueError))
check("⑤ unknown per-token tensor raises", raises(lambda: apply(fake_trainer(), make_batch(PS, extra_tensor=True), {}), RuntimeError))
_tm = fake_trainer(); _tm.config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
check("⑤ multi mini-batch raises", raises(lambda: apply(_tm, make_batch(PS), {}), ValueError))
check("⑤ tsw on raises", raises(lambda: apply(fake_trainer(tsw_enable=True), make_batch(PS), {}), ValueError))
check("⑤ null mode raises", raises(lambda: apply(fake_trainer(counterfactual_null_mode="mean_color"), make_batch(PS), {}), ValueError))
check("⑤ no teacher_image_key raises", raises(lambda: apply(fake_trainer(teacher_image_key=None), make_batch(PS), {}), ValueError))
_tr = fake_trainer(); _tr.config.algorithm.rollout_correction.rollout_rs = "token"
check("⑤ rollout_rs raises", raises(lambda: apply(_tr, make_batch(PS), {}), ValueError))
_tr = fake_trainer(); _tr.config.actor_rollout_ref.rollout.name = "sglang"
check("⑤ non-vLLM rollout raises", raises(lambda: apply(_tr, make_batch(PS), {}), ValueError))
bb = make_batch(PS); bb.non_tensor_batch["uid"][1] = "zz"
check("⑤ uid group size != n raises", raises(lambda: apply(fake_trainer(), bb, {}), RuntimeError))
bb = make_batch(PS); bb.non_tensor_batch["extra_info"][1] = dict(bb.non_tensor_batch["extra_info"][1], dg_row=999)
check("⑤ rollouts of one uid from different questions raises", raises(lambda: apply(fake_trainer(), bb, {}), RuntimeError))
bb = make_batch(PS); bb.non_tensor_batch["extra_info"][0] = {"answer": "D"}
check("⑤ extra_info without dg_* raises", raises(lambda: apply(fake_trainer(), bb, {}), ValueError))

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
    gen = [dict(row=0, group="/i0.png", gt="C", opts=["A", "B", "C", "D"], letters=["C"] * 8, texts=["x"] * 8),
           dict(row=1, group="/i1.png", gt="A", opts=["A", "B"], letters=["A"] * 8, texts=["x"] * 8)]
    json.dump(gen, open(f"{d}/T/gen.json", "w"))
    np.savez(f"{d}/T/readout.npz", rows=np.array([1, 0]), letter_logp=np.log(np.array([[.5, .5, .1, .1, .1, .1], [.1, .1, .6, .2, .1, .1]])))
    build(f"{d}/in.parquet", f"{d}/T", f"{d}/out.parquet")
    o = pq.read_table(f"{d}/out.parquet"); e = o.column("extra_info").to_pylist()
    check("⑦ pT0 renormalised over the options, rows matched by id", np.allclose(e[0]["dg_pT0"], [.1, .1, .6, .2, 0, 0])
          and np.allclose(e[1]["dg_pT0"], [.5, .5, 0, 0, 0, 0]) and e[0]["dg_opts"] == "ABCD" and e[1]["dg_row"] == 1)
    check("⑦ no offline trajectory fields", not any(k.startswith("dg_traj") for k in e[0]))
    check("⑦ other columns unchanged + original extra_info kept", o.column("prompt").equals(t.column("prompt")) and e[0]["answer"] == "C")
    bad = [dict(gen[0], gt="D"), gen[1]]; json.dump(bad, open(f"{d}/T/gen.json", "w"))
    check("⑦ identity mismatch raises", raises(lambda: build(f"{d}/in.parquet", f"{d}/T", f"{d}/o2.parquet"), AssertionError))

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
