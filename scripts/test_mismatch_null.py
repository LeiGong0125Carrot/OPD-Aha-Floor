#!/usr/bin/env python
"""Arm Amis (mismatched-crop visual null) unit tests -- CPU only.
Groups: [1] Sattolo derangement / donor picking; [2] image construction; [3] config validation;
        [4] two-view log-space averaging; [5] trainer static methods (mean_color path unchanged, mismatch path);
        [6] loss integration: mismatch changes nothing in core_algos (same tensors -> same loss as mean_color).
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils.mismatch_null import (  # noqa: E402
    average_null_log_probs, mismatch_crop_images, pick_donor_indices, sattolo_derangement,
)
from PIL import Image  # noqa: E402

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: FAIL.append(name)
def raises(fn, exc=Exception):
    try:
        fn(); return False
    except exc:
        return True

print("[1] derangement and donor picking")
for n in (2, 3, 5, 48):
    p = sattolo_derangement(n, np.random.default_rng(0))
    check(f"n={n}: permutation without fixed point", sorted(p.tolist()) == list(range(n)) and not np.any(p == np.arange(n)))
check("n=1 raises", raises(lambda: sattolo_derangement(1, np.random.default_rng(0)), ValueError))
uids = [f"p{i//2}" for i in range(96)]                      # 48 prompts x 2 rollouts, like training
d1 = pick_donor_indices(uids, seed=42, step=7)[0]
check("same (seed, step) -> same donors", np.array_equal(d1, pick_donor_indices(uids, seed=42, step=7)[0]))
check("different step -> different donors", not np.array_equal(d1, pick_donor_indices(uids, seed=42, step=8)[0]))
check("different seed -> different donors", not np.array_equal(d1, pick_donor_indices(uids, seed=43, step=7)[0]))
check("donor prompt != own prompt for every sample", all(uids[int(d1[i])] != uids[i] for i in range(96)))
check("both rollouts of a prompt share the donor", all(d1[2 * k] == d1[2 * k + 1] for k in range(48)))
check("donor index points at the FIRST rollout of the donor prompt", all(int(d1[i]) % 2 == 0 for i in range(96)))
d2a, d2b = pick_donor_indices(uids, seed=42, step=7, num_views=2)
check("num_views=2: first view identical to the single-view draw", np.array_equal(d2a, d1))
check("num_views=2: two donors differ from each other and from own", all(uids[int(d2a[i])] != uids[int(d2b[i])] and uids[int(d2a[i])] != uids[i] and uids[int(d2b[i])] != uids[i] for i in range(96)))
check("num_views=2 with 2 prompts raises", raises(lambda: pick_donor_indices(["a", "a", "b", "b"], 1, 1, num_views=2), ValueError))
check("single prompt raises", raises(lambda: pick_donor_indices(["a", "a"], 1, 1), ValueError))
check("num_views=3 raises", raises(lambda: pick_donor_indices(uids, 1, 1, num_views=3), ValueError))
# exhaustive: for n=3 all three steps' derangements have no fixed point and second view never equals first
ok = True
for step in range(50):
    a, b = pick_donor_indices(["x", "y", "z"], seed=5, step=step, num_views=2)
    ok &= all(a[i] != i and b[i] != i and a[i] != b[i] for i in range(3))
check("n=3 exhaustive over 50 steps: valid pairs", ok)

print("[2] image construction")
norm = lambda x: x.convert("RGB") if isinstance(x, Image.Image) else Image.open(x["path"]).convert("RGB")
full = Image.fromarray(np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8))
own = Image.fromarray(np.random.randint(0, 255, (16, 32, 3), dtype=np.uint8))
donor = Image.fromarray(np.random.randint(0, 255, (24, 10, 3), dtype=np.uint8))
out = mismatch_crop_images([full, own], donor, norm)
check("two images out", len(out) == 2)
check("full image kept pixel-identical", np.array_equal(np.asarray(out[0]), np.asarray(full)))
check("last image has the OWN crop's size", out[1].size == own.size)
check("last image is not the own crop", not np.array_equal(np.asarray(out[1]), np.asarray(own)))
mean_block = Image.new("RGB", own.size, tuple(np.rint(np.asarray(own, dtype=np.float32).mean(axis=(0, 1))).astype(np.uint8).tolist()))
check("last image is not the mean-colour block", not np.array_equal(np.asarray(out[1]), np.asarray(mean_block)))
check("last image equals the donor resized", np.array_equal(np.asarray(out[1]), np.asarray(donor.resize(own.size, Image.BICUBIC))))
same_size_donor = Image.fromarray(np.random.randint(0, 255, (16, 32, 3), dtype=np.uint8))
o2 = mismatch_crop_images([full, own], same_size_donor, norm)
check("same-size donor copied without resampling", np.array_equal(np.asarray(o2[1]), np.asarray(same_size_donor)))
check("donor == own object raises", raises(lambda: mismatch_crop_images([full, own], own, lambda x: x), RuntimeError))
check("donor path == own path raises", raises(lambda: mismatch_crop_images([{"path": "a.jpg"}, {"path": "c.jpg"}], {"path": "c.jpg"}, lambda d: Image.new("RGB", (4, 4))), RuntimeError))
check("empty list raises", raises(lambda: mismatch_crop_images([], donor, norm), ValueError))

print("[3] config validation")
from verl.workers.config.actor import SelfDistillationConfig  # noqa: E402
base = dict(full_logit_distillation=True, distillation_topk=100)
SelfDistillationConfig(**base, counterfactual_null_mode="mismatch_crop", counterfactual_null_scope="last")
check("mismatch_crop + last accepted", True)
check("mismatch_crop + scope=all rejected", raises(lambda: SelfDistillationConfig(**base, counterfactual_null_mode="mismatch_crop", counterfactual_null_scope="all"), ValueError))
check("unknown null mode rejected", raises(lambda: SelfDistillationConfig(**base, counterfactual_null_mode="foo"), ValueError))
check("num_views=2 requires mismatch_crop", raises(lambda: SelfDistillationConfig(**base, counterfactual_null_mode="mean_color", counterfactual_null_scope="last", counterfactual_null_num_views=2), ValueError))
SelfDistillationConfig(**base, counterfactual_null_mode="mismatch_crop", counterfactual_null_scope="last", counterfactual_null_num_views=2)
check("num_views=2 + mismatch_crop accepted", True)
check("num_views=3 rejected", raises(lambda: SelfDistillationConfig(**base, counterfactual_null_mode="mismatch_crop", counterfactual_null_scope="last", counterfactual_null_num_views=3), ValueError))
SelfDistillationConfig(**base, counterfactual_null_mode="mismatch_crop", counterfactual_null_scope="last", counterfactual_u_clip_pos=True)
check("mismatch_crop + suppression-only accepted", True)
check("default num_views is 1", SelfDistillationConfig(**base).counterfactual_null_num_views == 1)

print("[4] two-view averaging")
a = torch.log_softmax(torch.randn(4, 7, 300), -1)[..., :100]; b = torch.log_softmax(torch.randn(4, 7, 300), -1)[..., :100]
m = average_null_log_probs(a, b)
check("arithmetic mean of log-probs", torch.allclose(m, 0.5 * (a + b)))
check("top-k mass stays <= 1 (geometric <= arithmetic)", bool((m.exp().sum(-1) <= 1.0 + 1e-6).all()))
check("None/None passes through", average_null_log_probs(None, None) is None)
check("one None raises", raises(lambda: average_null_log_probs(a, None), ValueError))
check("shape mismatch raises", raises(lambda: average_null_log_probs(a, b[..., :50]), ValueError))
check("identical views -> identical result (num_views=2 with same donor would be a no-op)", torch.equal(average_null_log_probs(a, a), a))

print("[5] trainer static methods")
try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    mc = RayPPOTrainer._mean_color_teacher_images([full, own], null_scope="last")
    check("mean_color path unchanged: full kept, last = mean block of own size", np.array_equal(np.asarray(mc[0]), np.asarray(full)) and np.array_equal(np.asarray(mc[1]), np.asarray(mean_block)))
    mm = RayPPOTrainer._mismatch_crop_teacher_images([full, own], donor)
    check("_mismatch_crop_teacher_images: full kept, last = resized donor", np.array_equal(np.asarray(mm[0]), np.asarray(full)) and np.array_equal(np.asarray(mm[1]), np.asarray(donor.resize(own.size, Image.BICUBIC))))
    os.environ["VOPD_FORBID_NULL"] = "1"
    check("VOPD_FORBID_NULL=1 blocks the mismatch null too", raises(lambda: RayPPOTrainer._mismatch_crop_teacher_images([full, own], donor), RuntimeError))
    os.environ.pop("VOPD_FORBID_NULL", None)
    import inspect
    src = inspect.getsource(RayPPOTrainer._maybe_build_self_distillation_batch)
    check("gate accepts both null modes", 'in ("mean_color", "mismatch_crop")' in src)
    check("mismatch path reads uid + data seed + global_steps", "pick_donor_indices" in src and 'non_tensor_batch["uid"]' in src and "global_steps" in src)
    check("prefix-equality assertion still present for every view", "changed matched teacher" in src and "for view_k in range(null_num_views)" in src)
    check("mismatch path refuses batches with a sample lacking teacher images (donor safety)", "requires teacher images for every sample" in src)
except Exception as e:  # ray etc. not importable here
    if os.environ.get("PROBE_TEST_STRICT") == "1":
        check(f"trainer import ({type(e).__name__}: {e})", False)
    else:
        print(f"  SKIP  trainer statics ({type(e).__name__}: {e})")

print("[6] loss integration: core_algos is mode-agnostic")
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402
class Cfg(dict):
    __getattr__ = dict.get
def cfg(**kw):
    d = dict(full_logit_distillation=True, distillation_topk=5, distillation_add_tail=True, alpha=0.5, is_clip=2.0,
             counterfactual_null_mode="mean_color", counterfactual_extrapolation_beta=4.0, counterfactual_null_scope="last",
             counterfactual_u_clip_pos=False, counterfactual_reference="null", teacher_target_mode="legacy")
    d.update(kw); return Cfg(d)
torch.manual_seed(0)
B, T, K = 2, 5, 5
st = torch.log_softmax(torch.randn(B, T, 50), -1)[..., :K].requires_grad_(True)
te = torch.log_softmax(torch.randn(B, T, 50), -1)[..., :K]; nu = torch.log_softmax(torch.randn(B, T, 50), -1)[..., :K]
mask = torch.ones(B, T)
def run(c):
    out = compute_self_distillation_loss(student_log_probs=st.sum(-1), teacher_log_probs=te.sum(-1), response_mask=mask, self_distillation_config=c,
                                         old_log_probs=st.sum(-1).detach(), student_topk_log_probs=st, teacher_topk_log_probs=te, teacher_null_topk_log_probs=nu,
                                         teacher_null_log_probs=nu.sum(-1), self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")
    loss = out[0]; g = torch.autograd.grad(loss, st)[0]; return loss.detach(), g
l1, g1 = run(cfg(counterfactual_null_mode="mean_color")); l2, g2 = run(cfg(counterfactual_null_mode="mismatch_crop", counterfactual_null_num_views=2))
check("same null tensors -> bit-identical loss and gradient under mean_color vs mismatch_crop", torch.equal(l1, l2) and torch.equal(g1, g2))
l3, _ = run(cfg(counterfactual_null_mode="mismatch_crop", counterfactual_u_clip_pos=True))
check("suppression-only composes with mismatch_crop (loss changes)", not torch.equal(l1, l3))

print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAIL: {FAIL}")
sys.exit(1 if FAIL else 0)
