# P5 PBD Pre-Run Integration / Smoke Test Plan

**Date:** 2026-10-06  
**Target implementation:** `59178f3` on `sup`  
**Design reference:** `docs/proposals_2026-10-06/13_P5_assessment.md` §20  
**Scope:** validation required before starting the full 51-step Keep / Replace P5 training runs.

---

## 1. Purpose

The current P5 implementation spans the full rollout-to-update path:

[
	ext{student rollout}
ightarrow
	ext{crop-view continuation } y^T
ightarrow
	ext{branch-row construction}
ightarrow
	ext{student / frozen-teacher scoring}
ightarrow
	ext{PBD mask + weighting}
ightarrow
	ext{optimizer update}.
]

The CPU unit test in `scripts/test_pbd.py` validates most local helpers and the loss algebra, but it does **not** exercise the complete multimodal runtime path with Qwen3.5-4B. Before the 51-step run, perform one real GPU integration smoke test and verify the invariants below.

This document is a **test plan**, not a change to the frozen P5 arm. The frozen first-round settings remain:

- Keep main arm; Replace control.
- prefix ratio = 10%, snapped forward to a word boundary.
- branch coverage = 50% random.
- (lambda=0.5).
- (y^T) generated immediately after the student rollout by the current rollout weights.
- crop-view continuation input = ([full, crop]), no hint sentence.
- scoring teacher = frozen initial model.
- no null-view forward.
- continuation cap = 256.
- leak wording is monitored but not masked by default.

---

## 2. Test locations

### 2.1 Existing CPU/unit regression

Run:

```bash
CUDA_VISIBLE_DEVICES="" PROBE_TEST_STRICT=1 "$VOPD_PY" scripts/test_pbd.py
```

Relevant files:

- `scripts/test_pbd.py`
- `verl/utils/pbd.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/workers/actor/dp_actor.py`
- `verl/trainer/ppo/ray_trainer.py`

This test should remain the fast regression test for:

1. word-boundary prefix snapping;
2. branch-row tensor construction;
3. Keep / Replace parent masks;
4. branch prefix masking;
5. optional leak masking;
6. PBD configuration validation;
7. (L=mathrm{mean}_{V0}+lambda,mathrm{mean}_{branch}) weighting;
8. the (N_b
eq N_{V0}) case;
9. gradient equivalence to the intended weighted objective;
10. position-id helper behavior.

### 2.2 Test-suite exit-code guard

`scripts/run_neghist_tests.sh` currently stores the PBD test result in `r7`, but the final exit expression only includes `r1ldots r6`.

Before treating the suite as a gate, change:

```bash
exit $(( r1 | r2 | r3 | r4 | r5 | r6 ))
```

to:

```bash
exit $(( r1 | r2 | r3 | r4 | r5 | r6 | r7 ))
```

Otherwise a failing `test_pbd.py` can leave the aggregate script with exit code 0.

### 2.3 Required real-GPU integration test

Add a dedicated smoke-test launcher/script rather than modifying the frozen 51-step launcher. Suggested locations:

```text
scripts/test_pbd_integration.py
scripts/run_pbd_smoke.sh
```

The smoke run should use the same production code path as:

```text
scripts/train_pair_pbd.sh
```

but terminate after one optimizer step.

---

## 3. Minimal GPU smoke-test configuration

Use Qwen3.5-4B and the actual pair training parquet.

Recommended minimal configuration:

```text
MODEL_PATH=Qwen/Qwen3.5-4B
TRAIN_BATCH_SIZE=2 or 4
ROLLOUT_N=1
PBD_MODE=keep
PBD_RATIO=0.10
PBD_COVERAGE=1.0
PBD_LAMBDA=0.5
PBD_MAXCONT=256
PBD_LEAK_MASK=False
TRAINER_TOTAL_TRAINING_STEPS=1
ACTOR_USE_DYNAMIC_BSZ=False
```

For the smoke test only, `PBD_COVERAGE=1.0` is preferred so that every eligible parent produces a branch row and the branch path cannot be skipped by random coverage.

Run Keep first. After it passes, repeat once with:

```text
PBD_MODE=replace
```

No benchmark evaluation is needed for this test; the goal is runtime and objective correctness.

---

## 4. Instrumentation point A: immediately after student rollout

**Location:** `RayPPOTrainer.fit()`, immediately before:

```python
batch, _pbd_metrics = self._maybe_append_pbd_branch_rows(batch, timing_raw)
```

For each parent selected for the smoke test, record:

```text
uid
parent row index
student response token ids
student response decoded text
valid response length
```

### Assertions

For every selected parent:

1. response length is non-zero;
2. response mask exactly identifies real response tokens;
3. decoded response contains no visual special tokens in the generated response;
4. the row contains the expected full-image student multimodal input;
5. the teacher crop field exists.

This establishes the source trajectory (y^S).

---

## 5. Instrumentation point B: prefix selection

**Location:** `RayPPOTrainer._maybe_append_pbd_branch_rows()`, immediately after `snap_prefix_len()`.

Record:

```text
T = valid student-response length
raw target = ceil(0.10 * T)
t_star = snapped prefix length
t_star / T
prefix token ids
decoded prefix
first token after t_star
```

### Assertions

For every branched row:

[
t^* ge lceil 0.10Tceil,
]

and:

1. (t^* < T-1);
2. prefix ids are exactly `student_response[:t_star]`;
3. the first token at `student_response[t_star]` starts a new word/newline according to the current snapping rule;
4. no token is inserted, removed, or re-tokenized inside the preserved prefix.

The test should fail loudly if:

```python
prefix_ids != parent_response_ids[:t_star]
```

---

## 6. Instrumentation point C: crop-view continuation generation

**Location:** `PBDBranchAgentLoop.run()`.

Record for one or two rows:

```text
number of images in crop-view messages
prompt token length before prefix
prefix token length
final generation-prompt length
generated continuation length
generated continuation token ids
decoded continuation
```

### Required visual-view invariant

The continuation generator must receive:

```text
[full image, crop image] + original question
```

with no extra hint sentence.

Assert:

```text
student parent view:        1 image
PBD continuation view:      2 images [full, crop]
```

The exact ordering must be:

```text
full -> crop -> original text/question
```

### Prefix-continuation invariant

Inside `PBDBranchAgentLoop.run()`:

```python
prompt_ids = apply_chat_template(crop_view_messages)
prompt_ids = prompt_ids + prefix_ids
```

Assert that the sequence sent to vLLM ends in the **exact student prefix ids**.

No assistant turn should be closed and reopened between the preserved prefix and (y^T); (y^T) must be a direct continuation of that assistant prefix.

---

## 7. Instrumentation point D: branch-row construction

**Location:** after `make_branch_row_tensors()` in `_maybe_append_pbd_branch_rows()`.

For each branch row record:

```text
parent index
branch response length
t_star
branch response ids
pbd_loss_mask
response_mask
attention-mask valid length
```

### Exact token invariant

For valid branch response positions:

[
y^{branch}
=
y^S_{<t^*}
oplus
y^T.
]

Assert:

```python
branch.responses[:t_star] == parent.responses[:t_star]
branch.responses[t_star:t_star+len(yT)] == yT
```

up to the configured response-length truncation.

### Branch-loss-mask invariant

With `PBD_LEAK_MASK=False`:

```text
positions < t_star:        pbd_loss_mask = 0
positions t_star:valid:    pbd_loss_mask = 1
padding:                   pbd_loss_mask = 0
```

No prefix token may enter the branch loss.

---

## 8. Instrumentation point E: student multimodal view for the branch row

The branch row is constructed by copying the parent and replacing its response tensors.

This is intentional: the **student scoring view must stay full-image only**.

Immediately before the actor student forward, verify for at least one branch row:

```text
pbd_is_branch = 1
student input ids = original student prompt + prefix + yT
student multi_modal_inputs = parent's full-image inputs
```

### Critical assertion

The student branch forward must **not** inherit the crop-view generation multimodal inputs.

Expected:

```text
student branch scoring view: full image only
```

not:

```text
[full, crop]
```

This is one of the highest-priority integration checks because a silent crop leak into the student branch would invalidate the P5 experiment.

---

## 9. Instrumentation point F: frozen-teacher branch view

**Location:** teacher-input construction in `RayPPOTrainer` / actor update.

For a branch row, verify:

```text
teacher response token sequence = branch response token sequence
teacher image source = teacher_image_key / bbox_images
teacher visual view = [full, crop]
teacher weights = frozen initial model
null teacher tensors = absent
```

### Token-alignment invariant

Student and teacher must score the **same response tokens**:

```python
student_branch_response_ids == teacher_scored_response_ids
```

The only intended difference is the visual evidence available to the two models:

```text
student: full
teacher: full + crop
```

### No-null invariant

For the smoke run:

```text
counterfactual_null_mode is None
teacher_null_* tensors do not exist
teacher_null_forward_frac == 0
```

With `VOPD_FORBID_NULL=1`, any attempted null construction must hard-fail.

---

## 10. Instrumentation point G: Keep objective

Run one optimizer step with:

```text
PBD_MODE=keep
```

For a branched parent (i), assert:

### Parent mask

[
M_i^S(t)=1
]

for every valid student-response token.

### Branch mask

[
M_i^T(t)=
egin{cases}
0, & t<t^*\
1, & tge t^*
end{cases}
]

up to valid continuation tokens.

### Batch objective

The effective objective should be numerically reconstructable from logged raw losses:

[
L_{	ext{PBD-Keep}}
=
rac{1}{N_{V0}}sum_{iin V0} L_i
+
lambda
rac{1}{N_B}sum_{jin B} L_j^{branch}.
]

For the smoke batch, independently recompute this scalar from detached per-row values and compare with the loss entering backward within a small floating-point tolerance.

This is stronger than the existing synthetic unit test because it uses the real multimodal forwards and the real batching path.

---

## 11. Instrumentation point H: Replace objective

Repeat the one-step smoke test with:

```text
PBD_MODE=replace
```

For every branched parent:

[
M_i^S(t)=
egin{cases}
1, & t<t^*\
0, & tge t^*
end{cases}
]

while the branch row learns only on (tge t^*).

Verify the existing kept-token-fraction correction:

[
w_i^{parent}
=
rac{|M_i^S|}{|y_i^S|}.
]

This prevents a short retained prefix from being normalized as a full parent sequence.

### Replace-specific assertion

For one selected parent/branch pair, concatenate the two **loss supports** conceptually:

```text
parent contributes: yS[:t*]
branch contributes: yT[t*:]
```

There must be no duplicated loss support on the parent's (tge t^*) region.

---

## 12. Loss-scale / row-count check

Log:

```text
N_v0
N_branch
N_total
pbd_scale
pbd_scale_b
lambda
```

Expected:

[
pbd_scale=rac{N_{total}}{N_{v0}},
qquad
pbd_scale_b=rac{N_{total}}{N_{branch}}.
]

The real one-step test should include a case where:

[
N_{branch}
eq N_{v0}
]

if feasible, because that is where normalization bugs are easiest to hide.

For the ordinary 50% production setting this condition naturally holds.

---

## 13. DP / worker divisibility check

The implementation may drop branch candidates so:

[
B+N_{branch}
]

is divisible by actor data-parallel size.

Record:

```text
pbd/branch_candidates
pbd/branch_rows
pbd/dropped_for_dp
actor dp size
final merged batch size
```

Assert:

[
(B+N_{branch}) mod dp = 0.
]

Also verify that dropping candidates does not alter parent ordering and that branch rows correspond to the retained `cand` indices.

---

## 14. Position-id / multimodal positional check

The helper currently reconstructs Qwen multimodal position ids and performs a parent-row self-check.

For the GPU smoke test require both:

1. parent reconstruction is bit-identical to the rollout-produced parent `position_ids`;
2. every branch-row `position_ids` tensor has the same structural shape expected by the actor.

If either condition fails, stop before backward.

Do **not** downgrade this check to a warning: malformed MRoPE position ids can produce a run that executes but no longer corresponds to the intended Qwen multimodal sequence.

---

## 15. Leak monitoring check

The frozen arm is monitor-only:

```text
PBD_LEAK_MASK=False
```

Verify:

```text
pbd/leak_row_frac
pbd/leak_tok_frac
```

are produced, but leak tokens retain `pbd_loss_mask=1`.

For better §20 monitoring, add relative leak position statistics:

[
r_j = rac{j}{|y^T|}.
]

Suggested metrics:

```text
pbd/leak_pos_rel_mean
pbd/leak_pos_rel_p50
pbd/leak_pos_rel_p90
```

These distinguish leakage at the start of the continuation from leakage late in the reasoning trajectory.

---

## 16. Answer-format monitoring

§20 also calls for monitoring response length and missing `<answer>` rate.

For both parent and branch continuation, record at least:

```text
parent_response_len_mean
branch_response_len_mean
branch_missing_answer_frac
parent_missing_answer_frac
```

The smoke test should print the decoded parent and branch response for at least one pair so malformed continuation behavior is immediately visible.

This is monitoring rather than a hard training rejection criterion unless the output is structurally broken.

---

## 17. Runtime configuration guards

Before the full run, deliberately verify that the PBD path fails for invalid configurations:

```text
PBD_RATIO <= 0 or >= 1
PBD_COVERAGE <= 0 or > 1
PBD_LAMBDA < 0
PBD_MODE not in {keep, replace}
PBD_MAXCONT < 1
counterfactual_null_mode != None
use_dynamic_bsz=True
ppo_epochs != 1
```

The `SelfDistillationConfig` dataclass already validates most PBD fields. For robustness, mirror the essential range/mode checks in the runtime PBD path (`_pbd_cfg()` / `_maybe_append_pbd_branch_rows()`) so command-line overrides cannot silently reach generation with invalid values.

---

## 18. Random coverage versus \hat D

The first frozen training arm in §20.4 uses **50% random coverage**. The current implementation correctly uses `coverage_select()` and does not use (hat D) for targeting.

The wording in §20.1 that (hat D) is a usable selector should therefore be read as:

> (hat D) has been validated as a potential selector, but the first causal P5 training run intentionally uses random coverage; (hat D)-targeted branching is deferred to a later arm.

The smoke test should **not** introduce a (hat D) selector. Doing so would change the frozen experiment.

---

## 19. Pass/fail gate before the 51-step run

The full Keep run should start only if all of the following pass.

### Required

- [ ] `scripts/test_pbd.py` exits 0.
- [ ] aggregate test script propagates the PBD test exit code.
- [ ] one-step real-GPU Keep smoke run finishes forward, backward, and optimizer step.
- [ ] one-step real-GPU Replace smoke run finishes forward, backward, and optimizer step.
- [ ] exact parent-prefix identity verified.
- [ ] crop-view continuation receives exactly ([full,crop]).
- [ ] student branch scoring receives full-image view only.
- [ ] frozen teacher branch scoring receives privileged ([full,crop]) view.
- [ ] student and teacher score the same branch response token sequence.
- [ ] branch prefix has zero branch loss.
- [ ] Keep parent retains full valid response loss.
- [ ] Replace parent drops (tge t^*) loss.
- [ ] no null forward executes.
- [ ] real-batch loss agrees with independently reconstructed
  (mathrm{mean}_{V0}+lambda,mathrm{mean}_{branch}).
- [ ] position-id self-check passes.
- [ ] final merged batch is DP-divisible.
- [ ] no NaN / Inf in student logits, teacher logits, JSD, loss, or gradient norm.

### Monitoring only

- [ ] (y^T) length distribution.
- [ ] continuation truncation fraction.
- [ ] leak row/token frequency.
- [ ] leak relative-position statistics.
- [ ] branch/V0 raw JSD.
- [ ] tail-8 branch/V0 raw JSD.
- [ ] response length.
- [ ] missing `<answer>` fraction.

---

## 20. Recommended smoke-test output

For one representative parent/branch pair, print a compact diagnostic block:

```text
[PBD smoke]
uid:
mode:
parent_idx:
T:
t_star:
t_star/T:

student_view_images: 1
branch_generation_images: 2
teacher_scoring_images: 2

parent_prefix_ids == branch_prefix_ids: PASS
branch_prefix_loss_zero: PASS
student_branch_full_only: PASS
teacher_branch_full_plus_crop: PASS
student_teacher_response_ids_match: PASS
null_forward_absent: PASS
position_ids_valid: PASS

N_v0:
N_branch:
N_total:
pbd_scale:
pbd_scale_b:
lambda:

mean_v0_jsd:
mean_branch_jsd:
reconstructed_total:
backward_total:
abs_diff:

parent_text:
...
branch_continuation:
...
```

Keep this diagnostic for the first successful smoke run. It is useful later if the 51-step run behaves unexpectedly.

---

## 21. Decision rule

If the CPU suite and both one-step GPU smoke tests pass, the implementation is sufficiently validated to begin the frozen first-round P5 experiment.

If a smoke test fails, fix the integration path before changing any P5 algorithmic hyperparameter. In particular, do not respond to an integration failure by changing the 10% ratio, 50% random coverage, (lambda=0.5), Keep/Replace definitions, or leak policy; those are experimental variables already frozen by §20.
