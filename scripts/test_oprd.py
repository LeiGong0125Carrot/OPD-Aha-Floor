#!/usr/bin/env python
"""OPRD-Vanilla port unit tests (CPU; run on a compute node). Reference = verbatim OPRD main rep_distillation.py
(scripts/ref/oprd_rep_distillation_main.py, commit e1f7d22).

① layer_indices == reference get_rep_distillation_hidden_state_indices (all / last / even / odd)
② position_mask == reference build_rep_distillation_position_mask (all / last / last_k / first_k, ragged lengths)
③ mse_loss (normalize False / True) == reference multi_layer_normalized_mse_loss (main = plain MSE) on (B, L, T, D) with the
   reference's own masks; per-layer token-mean then layer mean; all-masked -> 0 (reference: nan)
④ gather_response_hidden: padded and remove-padding (packed) layouts give identical reprs; = manual indexing at start + t;
   padded response positions -> 0; a teacher row with a different prompt length / start
⑤ DataParallelPPOActor._oprd_response_token_positions: right-aligned layout (seqlen - R + t) and response_start_idx + t;
   valid = attention mask on the response block; out-of-bounds raises
⑥ gradients: teacher gets none (detached); student gradient only at supervised positions / selected layers
⑦ config dataclass guards; actor _oprd_cfg runtime guards
Run: PYTHONPATH=$REPO python scripts/test_oprd.py
"""
import os, sys, types
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref"))
from verl.utils import oprd as O  # noqa: E402
import oprd_rep_distillation_main as REF  # noqa: E402

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True); cond or fails.append(name)

def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False

torch.manual_seed(0)
# ① layers
check("① layer indices == reference (33 hidden states, 4 modes)",
      all(O.layer_indices(33, m) == REF.get_rep_distillation_hidden_state_indices(33, m) for m in ("all", "last", "even", "odd")))
check("① all = 1..32, embeddings excluded", O.layer_indices(33, "all") == list(range(1, 33)))
check("① bad mode raises", raises(lambda: O.layer_indices(33, "mid"), ValueError))

# ② positions
B, R = 4, 12
lens = torch.tensor([12, 5, 1, 8])
rm = (torch.arange(R).unsqueeze(0) < lens.unsqueeze(1)).float()
ok = True
for mode in ("all", "last", "last_k", "first_k"):
    for k in (1, 3, 2000):
        ok &= torch.equal(O.position_mask(rm, mode, last_k=k, first_k=k), REF.build_rep_distillation_position_mask(rm, mode, last_k=k, first_k=k))
check("② position masks == reference (4 modes x k in {1,3,2000})", bool(ok))
check("② last_k=2000 == all for responses <= 1024", torch.equal(O.position_mask(rm, "last_k", 2000), rm))
# first_frac (our ablation, user 10-28): per-sample first ceil(frac * n) tokens == reference first_k with that per-row k
ff = O.position_mask(rm, "first_frac", frac=0.25)
check("② first_frac 0.25: kept counts ceil(n/4) for n = 12,5,1,8 -> 3,2,1,2", ff.sum(1).tolist() == [3.0, 2.0, 1.0, 2.0])
ok = True
for i, n in enumerate(lens.tolist()):
    k = -(-n // 4)
    ok &= torch.equal(ff[i:i + 1], REF.build_rep_distillation_position_mask(rm[i:i + 1], "first_k", first_k=k))
check("② first_frac rows == reference first_k(k=ceil(n/4)) row by row", bool(ok))
check("② first_frac is a prefix inside the valid response", bool(((ff <= rm).all()) and (ff[:, 0] == 1).all()))
check("② first_frac 1.0 == all", torch.equal(O.position_mask(rm, "first_frac", frac=1.0), rm))
rm0 = torch.zeros(1, R)
check("② first_frac on an empty response -> nothing", O.position_mask(rm0, "first_frac", frac=0.25).sum().item() == 0)
lens2 = torch.tensor([100, 101, 4, 3]); rm2 = (torch.arange(128).unsqueeze(0) < lens2.unsqueeze(1)).float()
check("② first_frac counts 25,26,1,1 for n = 100,101,4,3", O.position_mask(rm2, "first_frac", frac=0.25).sum(1).tolist() == [25.0, 26.0, 1.0, 1.0])
check("② first_frac 0.07 on n=100 keeps 7 (float ceil guard)", O.position_mask((torch.arange(128).unsqueeze(0) < 100).float(), "first_frac", frac=0.07).sum().item() == 7)
check("② first_frac bad frac raises", raises(lambda: O.position_mask(rm, "first_frac", frac=0.0), ValueError) and raises(lambda: O.position_mask(rm, "first_frac", frac=1.5), ValueError))

# ③ loss vs reference (multi-layer path with explicit num_layers, (B, L, T, D))
L, D = 5, 16
s = torch.randn(B, L, R, D); t = torch.randn(B, L, R, D)
for mode in ("all", "last_k", "first_k"):
    pm = REF.build_rep_distillation_position_mask(rm, mode, last_k=3, first_k=3)
    ours = O.mse_loss(s, t, pm, normalize=False)
    ref = REF.multi_layer_normalized_mse_loss(s, t, position_mask=pm, loss_agg_mode="token-mean", num_layers=L)
    check(f"③ plain MSE == reference main ({mode})", torch.allclose(ours, ref, rtol=1e-5, atol=1e-7))
sn, tn = torch.nn.functional.normalize(s, dim=-1), torch.nn.functional.normalize(t, dim=-1)
check("③ normalize=True == plain MSE on L2-normalised vectors (Bridge-branch variant)",
      torch.allclose(O.mse_loss(s, t, rm, normalize=True), O.mse_loss(sn, tn, rm)))
check("③ all-masked -> 0 (finite)", float(O.mse_loss(s, t, torch.zeros_like(rm))) == 0.0)
manual = torch.stack([(((s[:, l] - t[:, l]) ** 2).mean(-1) * rm).sum() / rm.sum() for l in range(L)]).mean()
check("③ = mean over layers of masked token-mean of mean-over-D squared error", torch.allclose(O.mse_loss(s, t, rm), manual))

# ④ gather: padded vs packed layouts
S = 10; Bq = 2; Dq = 6; nh = 4
attn = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],      # left-pad 2, prompt 4, response 2 valid + 2 pad (R=4, start=6)
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])     # no left pad, prompt 6, response 3 valid + 1 pad (start=6)
Rq = 4; start = torch.tensor([6, 6])
hs_pad = tuple(torch.randn(Bq, S, Dq) for _ in range(nh))
idx = torch.nonzero(attn.reshape(-1), as_tuple=False).flatten()
hs_packed = tuple(h.reshape(Bq * S, Dq)[idx].unsqueeze(0) for h in hs_pad)
packed = torch.full((Bq * S,), -1, dtype=torch.long); packed[idx] = torch.arange(idx.numel())
abs_pos = start.unsqueeze(1) + torch.arange(Rq).unsqueeze(0)
valid = torch.gather(attn, 1, abs_pos).bool()
lay = [1, 3]
g_pad = O.gather_response_hidden(hs_pad, lay, abs_pos, valid)
g_pk = O.gather_response_hidden(hs_packed, lay, abs_pos, valid, packed_index=packed)
check("④ padded == packed layout", torch.allclose(g_pad, g_pk))
man = torch.stack([torch.stack([hs_pad[l][b, abs_pos[b]] * valid[b].unsqueeze(-1) for b in range(Bq)]) for l in lay], 1)
check("④ = manual hidden at start + t, padded response positions 0", torch.allclose(g_pad, man)
      and g_pad[0, :, 2:].abs().sum() == 0 and g_pad[1, :, 3:].abs().sum() == 0 and g_pad[1, :, :3].abs().sum() > 0)
check("④ shape (B, n_layers, R, D) fp32", tuple(g_pad.shape) == (Bq, 2, Rq, Dq) and g_pad.dtype == torch.float32)

# ④b vs the reference extractor on the right-aligned student layout (remove-padding path, all layers, no compaction)
attn_r = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]])   # prompt left-padded, response right-padded
Rr = 4
idx_r = torch.nonzero(attn_r.reshape(-1), as_tuple=False).flatten()
hs_full = tuple(torch.randn(Bq, S, Dq) for _ in range(nh))
hs_pk = tuple(h.reshape(Bq * S, Dq)[idx_r].unsqueeze(0) for h in hs_full)
rmask = attn_r[:, S - Rr:].float()
ref = REF.hidden_states_tuple_to_response_repr(hs_pk, rmask, "all", "all", indices=idx_r, batch_size=Bq, seqlen=S,
                                               use_remove_padding=True, compact=False)
pk = torch.full((Bq * S,), -1, dtype=torch.long); pk[idx_r] = torch.arange(idx_r.numel())
abs_r = torch.arange(S - Rr, S).repeat(Bq, 1); val_r = rmask.bool()
ours = O.gather_response_hidden(hs_pk, O.layer_indices(nh, "all"), abs_r, val_r, packed_index=pk)
check("④b == reference hidden_states_tuple_to_response_repr (rmpad, right-aligned) on valid tokens",
      torch.allclose(ours * val_r[:, None, :, None], ref * val_r[:, None, :, None]))

# ⑤ actor position helper
from verl.workers.actor.dp_actor import DataParallelPPOActor as A  # noqa: E402
mb = {"input_ids": torch.zeros(Bq, S, dtype=torch.long), "attention_mask": attn}
p1, v1 = A._oprd_response_token_positions(mb, None, Rq, S)
check("⑤ right-aligned layout: seqlen - R + t", torch.equal(p1, torch.arange(S - Rq, S).repeat(Bq, 1)) and torch.equal(v1, attn[:, S - Rq:].bool()))
p2, v2 = A._oprd_response_token_positions(mb, torch.tensor([3, 5]), Rq, S)
check("⑤ response_start_idx + t (teacher rows with their own prompt length)",
      p2[0].tolist() == [3, 4, 5, 6] and p2[1].tolist() == [5, 6, 7, 8] and torch.equal(v2, torch.gather(attn, 1, p2).bool()))
check("⑤ out-of-bounds raises", raises(lambda: A._oprd_response_token_positions(mb, torch.tensor([8, 8]), Rq, S), ValueError))

# ⑥ gradients
s2 = torch.randn(B, L, R, D, requires_grad=True); t2 = torch.randn(B, L, R, D, requires_grad=True)
pm = O.position_mask(rm, "first_k", first_k=3)
O.mse_loss(s2, t2, pm).backward()
check("⑥ teacher detached (no grad)", t2.grad is None)
gnz = s2.grad.abs().sum((1, 3)) > 0
check("⑥ student grad exactly at supervised positions", torch.equal(gnz, pm.bool()))

# ⑥b hidden-state guard
from verl.workers.actor.dp_actor import DataParallelPPOActor as A2  # noqa: E402
hs_g = tuple(torch.randn(1, 3, 4, requires_grad=True) for _ in range(5))
hs_ng = tuple(torch.randn(1, 3, 4) for _ in range(4)) + (torch.randn(1, 3, 4, requires_grad=True),)
check("⑥b guard: complete grad-carrying hidden states pass", raises(lambda: A2._oprd_check_hidden(hs_g, {"layers": "all", "expected_hidden_states": 5})) is False)
check("⑥b guard: wrong count raises", raises(lambda: A2._oprd_check_hidden(hs_g, {"layers": "all", "expected_hidden_states": 33}), RuntimeError))
check("⑥b guard: only the last layer has grad -> raises", raises(lambda: A2._oprd_check_hidden(hs_ng, {"layers": "all"}), RuntimeError))
with torch.no_grad():
    check("⑥b guard: teacher (no_grad) does not require grad", raises(lambda: A2._oprd_check_hidden(hs_ng, {"layers": "all"})) is False)
check("⑥b guard: None raises", raises(lambda: A2._oprd_check_hidden(None, {"layers": "all"}), RuntimeError))

# ⑦ config
from verl.workers.config.actor import SelfDistillationConfig as C  # noqa: E402
check("⑦ defaults off, paper values", C().oprd_enable is False and C().oprd_only and C().oprd_layers == "all"
      and C().oprd_positions == "last_k" and C().oprd_last_k == 2000 and C().oprd_normalize is False)
check("⑦ bad layers raises", raises(lambda: C(oprd_enable=True, oprd_layers="mid")))
check("⑦ bad positions raises", raises(lambda: C(oprd_enable=True, oprd_positions="frac")))
check("⑦ + tsw raises", raises(lambda: C(oprd_enable=True, tsw_enable=True)))
check("⑦ first_frac accepted", not raises(lambda: C(oprd_enable=True, oprd_positions="first_frac", oprd_frac=0.25)))
check("⑦ oprd_frac 1.5 raises", raises(lambda: C(oprd_enable=True, oprd_positions="first_frac", oprd_frac=1.5)))
from omegaconf import OmegaConf  # noqa: E402
fake = types.SimpleNamespace(use_ulysses_sp=False, use_fused_kernels=False)
cfg = lambda **kw: OmegaConf.create(dict(dict(oprd_enable=True), **kw))
c = A._oprd_cfg(fake, cfg())
check("⑦ runtime cfg defaults", c == dict(layers="all", positions="last_k", last_k=2000, first_k=2000, frac=0.25, coef=1.0, only=True, normalize=False, include_t0=False))
check("⑦ runtime first_frac passes frac", A._oprd_cfg(fake, cfg(oprd_positions="first_frac", oprd_frac=0.25))["frac"] == 0.25)
check("⑦ runtime: frac 0 raises", raises(lambda: A._oprd_cfg(fake, cfg(oprd_positions="first_frac", oprd_frac=0.0)), ValueError))
check("⑦ runtime: off -> None", A._oprd_cfg(fake, OmegaConf.create(dict(oprd_enable=False))) is None)
check("⑦ runtime: SP>1 raises", raises(lambda: A._oprd_cfg(types.SimpleNamespace(use_ulysses_sp=True, use_fused_kernels=False), cfg()), ValueError))
check("⑦ runtime: null mode raises", raises(lambda: A._oprd_cfg(fake, cfg(counterfactual_null_mode="mean_color")), ValueError))
check("⑦ runtime: bad positions raises", raises(lambda: A._oprd_cfg(fake, cfg(oprd_positions="x")), ValueError))

# ⑧ + t=0 (user 10-29): column 0 = start - 1 (last prompt token of each view), mask column 0 = rows with a response
check("⑧ config default include_t0 False", C().oprd_include_t0 is False)
check("⑧ runtime passes include_t0", A._oprd_cfg(fake, cfg(oprd_positions="first_frac", oprd_include_t0=True))["include_t0"] is True)
q1, w1 = A._oprd_response_token_positions(mb, torch.tensor([3, 5]), Rq, S, include_t0=True)
check("⑧ positions: [start-1, start, .., start+R-1], shape (B, R+1)",
      tuple(q1.shape) == (Bq, Rq + 1) and q1[0].tolist() == [2, 3, 4, 5, 6] and q1[1].tolist() == [4, 5, 6, 7, 8])
check("⑧ columns 1.. == include_t0=False positions/valid", torch.equal(q1[:, 1:], p2) and torch.equal(w1[:, 1:], v2))
check("⑧ t=0 column valid = attention at the last prompt token", torch.equal(w1[:, 0], torch.gather(attn, 1, q1[:, :1]).squeeze(1).bool()))
q3, _ = A._oprd_response_token_positions(mb, None, Rq, S, include_t0=True)
check("⑧ right-aligned layout: t=0 at seqlen - R - 1", q3[:, 0].tolist() == [S - Rq - 1] * Bq and torch.equal(q3[:, 1:], p1))
check("⑧ start 0 with t=0 -> out of bounds raises", raises(lambda: A._oprd_response_token_positions(mb, torch.tensor([0, 3]), Rq, S, include_t0=True), ValueError))
ff = O.position_mask(rm, "first_frac", frac=0.25)
m0 = O.with_t0(ff, rm)
check("⑧ with_t0: shape (B, R+1), column 0 = 1 for non-empty rows, rest == first_frac",
      tuple(m0.shape) == (B, R + 1) and torch.equal(m0[:, 1:], ff) and torch.equal(m0[:, 0], (rm.sum(1) > 0).float()))
check("⑧ with_t0 on an empty response -> nothing", O.with_t0(O.position_mask(rm0, "first_frac", frac=0.25), rm0).sum().item() == 0)
check("⑧ supervised count = ceil(n/4) + 1", m0.sum(1).tolist() == [4.0, 3.0, 2.0, 3.0])
# end-to-end on a padded layout: gathered t=0 column is the hidden state at start-1, loss counts it as one more token
g0 = O.gather_response_hidden(hs_pad, lay, q1, w1)
check("⑧ gathered t=0 = hidden at start-1", torch.allclose(g0[:, :, 0], torch.stack([torch.stack([hs_pad[l][b, q1[b, 0]] for b in range(Bq)]) for l in lay], 1))
      and torch.allclose(g0[:, :, 1:], O.gather_response_hidden(hs_pad, lay, p2, v2)))
check("⑧ t=0 gather: packed (remove-padding) layout == padded", torch.allclose(O.gather_response_hidden(hs_packed, lay, q1, w1, packed_index=packed), g0))
rmq = attn.gather(1, p2).float()
mq = O.with_t0(O.position_mask(rmq, "first_frac", frac=0.25), rmq)
sg = torch.randn(Bq, 2, Rq + 1, Dq, requires_grad=True); tg = torch.randn(Bq, 2, Rq + 1, Dq)
O.mse_loss(sg, tg, mq).backward()
check("⑧ student grad at t=0 and the first_frac tokens only", torch.equal(sg.grad.abs().sum((1, 3)) > 0, mq.bool()))
man0 = torch.stack([(((sg[:, l] - tg[:, l]) ** 2).mean(-1) * mq).sum() / mq.sum() for l in range(2)]).mean()
check("⑧ loss = token-mean incl. t=0", torch.allclose(O.mse_loss(sg, tg, mq), man0))

# ⑨ t=0 arms (user 10-31): V0 + t=0 hidden (positions=none + include_t0) and the y_0-only JSD control
pn = O.position_mask(rm, "none")
check("⑨ positions=none -> all zero", tuple(pn.shape) == tuple(rm.shape) and pn.sum().item() == 0)
mn = O.with_t0(pn, rm)
check("⑨ none + with_t0 -> only column 0, for non-empty rows", torch.equal(mn[:, 0], (rm.sum(1) > 0).float()) and mn[:, 1:].sum().item() == 0)
sg2 = torch.randn(B, 2, R + 1, 4, requires_grad=True); tg2 = torch.randn(B, 2, R + 1, 4)
O.mse_loss(sg2, tg2, mn).backward()
check("⑨ grad only at t=0", torch.equal(sg2.grad.abs().sum((1, 3)) > 0, mn.bool()))
check("⑨ config: none without include_t0 raises", raises(lambda: C(oprd_enable=True, oprd_positions="none")))
check("⑨ config: none + include_t0 + only=False accepted", not raises(lambda: C(oprd_enable=True, oprd_positions="none", oprd_include_t0=True, oprd_only=False)))
check("⑨ runtime: none without include_t0 raises", raises(lambda: A._oprd_cfg(fake, cfg(oprd_positions="none")), ValueError))
check("⑨ runtime: none + include_t0 ok", A._oprd_cfg(fake, cfg(oprd_positions="none", oprd_include_t0=True, oprd_only=False))["only"] is False)
check("⑨ config: sd_first_token_only default False, + OPRD raises", C().sd_first_token_only is False
      and raises(lambda: C(sd_first_token_only=True, oprd_enable=True, oprd_positions="none", oprd_include_t0=True)))
check("⑨ runtime: sd_first_token_only + OPRD raises", raises(lambda: A._oprd_cfg(fake, cfg(oprd_positions="none", oprd_include_t0=True, sd_first_token_only=True)), ValueError))
fm = O.first_token_mask(rm)
check("⑨ first_token_mask: column 0 of non-empty rows only", torch.equal(fm[:, 0], rm[:, 0]) and fm[:, 1:].sum().item() == 0 and fm.dtype == rm.dtype)
# y_0-only JSD normalisation: student = teacher everywhere except y_0 -> full-mask loss = sum JSD(y_0) / n_all,
# y_0-only loss = sum JSD(y_0) / n_rows (token-mean over the y_0 tokens)
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402
class _Cfg(dict):
    __getattr__ = dict.get
sdc = _Cfg({"full_logit_distillation": True, "distillation_topk": 8, "distillation_add_tail": True, "renorm_topk_log_probs": False,
            "alpha": 0.5, "is_clip": None, "counterfactual_null_mode": None, "counterfactual_extrapolation_beta": 0.0,
            "counterfactual_u_clip_pos": False, "log_prob_dump_dir": None})
Bj, Rj, K = 3, 6, 8
rmj = torch.tensor([[1.] * 6, [1.] * 4 + [0.] * 2, [1.] * 2 + [0.] * 4])
st = torch.log_softmax(torch.randn(Bj, Rj, K), -1) - 0.1; tt = st.clone(); tt[:, 0] = torch.log_softmax(torch.randn(Bj, K), -1) - 0.1
lpj = torch.randn(Bj, Rj)
def _jsd(mask):
    return compute_self_distillation_loss(student_log_probs=lpj, teacher_log_probs=lpj.clone(), response_mask=mask,
                                          self_distillation_config=sdc, old_log_probs=lpj.clone(), student_topk_log_probs=st,
                                          teacher_topk_log_probs=tt, self_distillation_mask=torch.ones(Bj), loss_agg_mode="token-mean")[0]
lf, l0 = _jsd(rmj), _jsd(O.first_token_mask(rmj))
check("⑨ y_0-only loss = full loss x n_all / n_rows (only y_0 differs)", lf.item() > 0 and torch.allclose(l0, lf * rmj.sum() / Bj, rtol=1e-4))

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
