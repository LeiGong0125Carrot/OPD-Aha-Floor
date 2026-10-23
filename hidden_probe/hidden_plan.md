# Hidden Probe Plan: Causal Visual Correction beyond Output-Level OPD

**Status:** reviewed pilot plan (v2); no training algorithm approved  
**Updated:** 2026-10-23  
**Revision:** incorporates 2026-10-22 per-token answer probe, off6k actual views, Qwen3.5-4B hybrid architecture, GT-free pilot selection  
**Repository context:** OPD-Aha; planned extension of visual-contrast on-policy distillation

## 1. Objective and research boundary

Test whether **privileged visual representations contain causally useful corrective information that is not adequately expressed by output-level teacher token preferences**.

Do **not** assume that subtracting two hidden states yields a semantic "visual direction." Vector subtraction is well-defined, but the resulting direction may contain input-layout, crop-style, position, content, and answer-planning effects. A hidden difference becomes a *candidate corrective signal* only after controlled intervention demonstrates (i) task-specific preference improvement and (ii) downstream behavioral recovery.

This project separates:

1. **Representation discrepancy:** How much do two views change hidden states?
2. **Causal usefulness:** Does intervening on a chosen state actually improve the student's visual decision?
3. **Optimization utility:** Can the student learn the effect by training without privileged input at deployment?

Only (2) justifies investigating (3). Representation distance by itself is insufficient.

## 2. Existing evidence: view-hidden probe (2026-10-22)

**Run:** 600 questions x 2 base trajectories = 1,200 trajectories; 74,919 generated tokens. Tests and smoke checks passed. Run finished in 00:58; GPU hold released. Reported artifact: **eval/probe_hidden/viewhid_1022/summary.json** (existing experiment output; presence in this GitHub repository has not been confirmed).

### 2.1 Representation similarity

- Student/teacher cross-view hidden cosine similarity exceeds 0.95 across layers.
- Example at L31: normalized distance (nd) from student to correct-crop teacher (SC) = 0.101; from student to a mismatched-crop teacher (SM) = 0.152.
- L1 nd ~0.005; deeper states are more sensitive to context and image changes.
- Distances between different-token states are on the order of 2, substantially larger than same-token cross-view distances.

**Interpretation constraint:** SC/SM around 0.66–0.72 means that SC is ~28–34% *lower than SM using SM as denominator*; at L31, (0.152 − 0.101)/0.152 = 33.6%. This is a controlled, view-matched **additional content-sensitive distance difference**, NOT an additive claim that hidden information is 1/3 content and 2/3 style. SC/SM does **not** identify an exact decomposition. Norms do not add linearly and differences can interact. The matched-crop control establishes that view configuration is a major confound and image content has an additional measurable effect, not a uniquely identified additive percentage.

### 2.2 Difference before and after output-level training

At L31, comparing base student vs. base teacher and step-70 student vs. **base teacher**:

| Response location | Base student vs. base teacher | Step-70 student vs. base teacher |
| --- | ---: | ---: |
| Pre-generation (t=0) | 0.197 | 0.464 |
| Before answer | 0.129 | 0.062 |
| After answer | 0.065 | 0.043 |

- Mean output JSD decreased from 0.0457 to 0.0197.
- Fraction of answer-first trajectories decreased from 0.51 to 0.41.
- JSD vs. hidden-distance within-trajectory rank correlation: L31 = 0.51; L16 = 0.22; L8 = 0.13.

**Interpretation constraints:** Lower JSD accompanied by lower hidden distance at some positions does not establish that JSD directly forces hidden alignment. The changed pre-generation distance shows that indiscriminate hidden MSE could oppose useful training-induced behavior. Low JSD/hidden-distance correlation does not establish that the output misses *corrective* information: it may simply miss irrelevant representation variation.

### 2.3 Per-token answer-position probe and decision anchors (2026-10-22)

A separate group-wise per-token probe found that **natural next-token answer decoding before the answer position is almost uninformative** (answer AUC ~0.50–0.54; choice-letter accuracy near chance). A strong answer signal appears just before the actual answer token (reported AUC ~0.84). In the base trajectories, **51% answer immediately at t=0** (step-70: 41%); the median answer position in the base is therefore t=0.

These figures are reported by the user from the existing per-token probe; they are **not re-estimated** in this document. Readout implementation is reported to exist in `probe_hidden_answer`.

**Consequences:** (a) t=0 is a main analysis stratum, not a style-only diagnostic; (b) for earlier anchors, use *forced answer readout* after the intervention plus natural free continuation, not natural next-token answer margin; (c) only the answer-decision anchor permits natural next-token choice logit comparison.

### 2.4 Immediate design consequence

**Do not launch raw, all-layer hidden-state MSE.** Do not treat a teacher-student or correct-null hidden difference as a validated visual correction vector. The next study must test causal influence on **forced-readout answer preference and downstream continuation**, with answer-first and reasoning-first trajectories reported separately.

## 3. Relationship to OPD-Aha

OPD-Aha builds an output-space visual contrast on the same student-sampled prefix:

\[
u_t(v)=\log p_T^+(v)-\log p_T^0(v),\qquad
q_t(v)\propto p_T^+(v)\exp(\beta u_t(v)).
\]

For visual-supported token \(v_+\) versus prefix-supported token \(v_-\), the log-odds change has a direct interpretation:

\[
\log \frac{q_t(v_+)}{q_t(v_-)}
-\log \frac{p_T^+(v_+)}{p_T^+(v_-)}
=\beta[u_t(v_+)-u_t(v_-)].
\]

By contrast, a hidden contrast \(d_l=h_T^{+,l}-h_T^{0,l}\) has **no guaranteed** token-preference effect. Locally, with \(M(h)\) the visual-supported minus prefix-supported output-logit margin:

\[
M(h_S+\alpha d_l)-M(h_S)
\approx \alpha\,\nabla_{h_S}M(h_S)^\top d_l.
\]

The sign is empirical and can vary by sample, layer, prefix, and direction. **The research question is whether some internal contrasts have reproducible corrective causal utility that OPD-Aha output contrast does not capture.**

## 4. Hypotheses and falsifiable alternatives

- **H1 / causal effect:** At matched student prefixes, controlled privileged-visual hidden interventions improve the correct image-supported answer margin more than matched random, shuffled, or misleading-view interventions.
- **H2 / content specificity:** Effects follow the *corresponding image content*, not just addition of crop tokens, image layout, or a generic answer/readout direction.
- **H3 / output complementarity:** Some states have weak, unreliable, or wrong output-level OPD-Aha targets yet exhibit positive, content-specific hidden intervention effects.
- **H4 / behavior transfer:** Positive local preference changes sometimes translate to durable, unpatched free-continuation recovery.
- **H5 / optimization utility (conditional):** Only after H1–H4, a selective hidden-informed target improves inference-time performance over OPD-Aha at matched compute/data.

**Null explanations:** random activation disruption; crop-vs-full-view style; mismatched image semantics; position/token-count differences; direct LM-head steering without improved computation; answer-token post hoc effects; noisy gains selected after viewing outcomes.

## 5. Actual off6k views, answer-position strata, and controls

### 5.1 Off6k view definitions (default pilot)

Use the **actual off6k setup**, rather than an invented full-image-plus-crop main teacher view.

| Name | Actual input | Role |
| --- | --- | --- |
| **S** | Red-box-marked **full image + the existing instruction/prompt text** | Base student; rollout generation and patched replay |
| **T+** | **Correct GT crop only**, with the existing off6k teacher prompt convention | Privileged teacher |
| **T0** | **Same-sized mean-color image block** in place of teacher crop, from the existing \`_mean_color_teacher_images\` implementation (OPD-Aha A-arm) | Null visual counterfactual |
| **Tmis** | Crop taken from a different question, resized to match the correct crop dimensions and visual-token settings (existing \`probe_view_hidden\` procedure) | Mismatched visual-content control |
| **Tuninfo** | An unrelated/uninformative crop with matched view layout, if available | Optional extra control |
| **Tsame** | An identical S-view replay with no effective modification | Deterministic/no-op check |
| **F** | Full image + crop | *Supplemental only*, not primary teacher view |

Prior recoverability probe: the F view was **farther** from S in output JSD (0.0523) than the original teacher-crop view (0.0457), and changed opening response style. This is a reason not to silently replace the crop-only teacher with F.

The teacher/student share exactly the **same student-generated textual response prefix**, but have **different multimodal prompt lengths and visual-token positions**. Align at the logical *response decision anchor*, not at equal absolute sequence index. Confirm per-view positional encodings, image scaling, special tokens, attention semantics, and extraction hook before any subtraction.

T0 and Tmis are different interventions: T0 replaces image evidence with a null image; Tmis inserts real, potentially contradictory evidence. Never assume their deltas have the same causal meaning.

### 5.2 Main strata based on existing automated answer positions

Do **not** require human annotation of "first wrong visual relation." Use the already-implemented answer-token locations:

1. **Answer-first:** answer position \(a=0\), i.e. the student answers on its first generated token; t=0 is a **primary stratum**.
2. **Reasoning-first:** \(a\ge 1\), with patch anchors at answer decision, \(a-1\), and \(a-4\) when valid.
3. **Post-answer:** descriptive stability/format diagnostics only; not prevention.

Define **t=0** as the **pre-first-response predictive state**, after the full multimodal prompt, not hidden_states[0] (embedding). Map each proposed anchor to the exact causal hidden row that predicts the relevant next token (avoid off-by-one errors between token index and prediction position). If an anchor duplicates another (e.g. answer at t=0), compute that intervention once.

The first wrong relation is not needed for primary grouping. Do not equate "before answer" with "before first mistake": without manual annotation or another verifier, refer to **pre-answer influence**, not guaranteed error prevention.

### 5.3 Checkpoint/split controls

- Primary view comparisons: the **same base Qwen3.5-4B checkpoint** for S and T+/T0/Tmis; freeze the policy during pilot.
- Analyze the step-70 student vs base teacher separately; it mixes training drift and view differences.
- Use a **development/holdout split** of the pilot questions. Select layer/dose using GT-free teacher-consistency only in development. Holdout GT diagnostics must not influence selection.
- Group splits by image/object pair where known; stratify answer-first/reasoning-first and answer-position coverage.

## 6. Phase 0 — model architecture, replay, alignment, and integrity

**Architecture in the current implementation:** Qwen3.5-4B, **32 decoder layers**, hidden size **2560**, mixed full-attention/linear-attention (DeltaNet) architecture. The 8 full-attention blocks are **3, 7, 11, 15, 19, 23, 27, 31** (1-based block numbering); all other blocks are linear attention / stateful DeltaNet.

**Indexing:** existing extraction uses \`hidden_states[l]\`, with \`hidden_states[0]\` being the **embedding output**, and \`hidden_states[l]\` corresponding to output after block \`l\` for \`l=1..32\`. Verify this against the exact loaded checkpoint and local hooks rather than assuming external conventions. Do not confuse the decoder **layer index** with answer **time t=0**.

**Candidate pilot layers:** \`8, 12, 15, 19, 23, 27\`; 15/19/23/27 are full-attention, 8/12 are linear/DeltaNet. L31 is descriptive for view-distance analyses but not part of this pilot's intervention grid.

**Critical stateful-model requirement:** **No approximate KV-cache-based patch-and-continue protocol.** Full-sequence replay must be used for every patched condition. Explicitly test whether a patch applied to the **block output at time t** changes later positions through the actual implementation: modifying the output residual stream is **not the same** as rewriting a DeltaNet block's internal recurrent state. Avoid claims that the patch automatically edits the past recurrence memory; validate actual forward effects in full replay.

- [ ] Inspect exact Qwen3.5-4B model config and hidden-state hook semantics (linear vs. full-attention blocks; pre/post norm).
- [ ] Verify the existing off6k prompts, red-box rendering, crop-only teacher, \`_mean_color_teacher_images\` null, \`probe_view_hidden\` mismatched-crop scaling, and color/tokenization handling.
- [ ] Verify *identical student response tokens* across all teacher-forced passes and no future-token leakage.
- [ ] Match logical token/answer anchors under different multimodal token counts and position IDs; log the actual indices per view.
- [ ] Verify layer-output patch is applied at the desired predictive time and impacts forced suffix positions correctly.
- [ ] Use full replay without KV-cache shortcuts; verify no-op bit/exact or tolerance-documented equivalence and random-control behavior.
- [ ] Verify answer parsing for t=0 and later choices and choices spanning multiple sub-tokens; report skipped examples and why.
- [ ] Save checkpoint/hash, image transform metadata, tensor shapes, dtype, baseline logits/JSD, hidden-state distances, and exact intervention masks.
- [ ] Keep base vs step-70 checkpoint comparisons separate from view controls.

## 7. Phase 1 — observational candidate directions (not corrections)

At each shared student-text decision anchor and candidate decoder layer:

\[
d_{\mathrm{TS}}^l=h_{T+}^l-h_S^l,\qquad
d_{0}^l=h_{T+}^l-h_{T0}^l,\qquad
d_{\mathrm{mis}}^l=h_{T+}^l-h_{Tmis}^l.
\]

**Pilot intervention grid:** \`d0 = T+ - T0\` and \`dmis = T+ - Tmis\`. Keep \`dTS\` descriptive / later ablation; it mixes differences between the red-box full-image student and crop-only teacher. Null and mismatched crops should have identical within-teacher layout so subtraction is interpretable as a *controlled input contrast*, not a confirmed visual semantic vector.

Record norms, covariance, direction cosines, and dependence on answer position and full-attention vs DeltaNet layers. Do not claim a content/style percentage decomposition from scalar norm ratios. Freeze any norm normalization rule and expected alpha scale before holdout evaluation.

**Caution:** subtracting Teacher states is a mathematical operation, not a claim that the vector improves the desired answer. Only Phase 2 interventions can establish a local causal effect.

## 8. Phase 2 — pilot causal intervention and answer readout

### 8.1 Intervention

For a matched causal decision anchor and layer:

\[
h_{S,\mathrm{patched}}^{l,t}=h_S^{l,t}+\alpha d^{l,t}
\]

Run the **complete replay** through the actual hybrid attention/DeltaNet architecture, with a well-defined teacher-forced suffix only for *readout*, not for changing the original rollout. DeltaNet state-cache shortcuts are disallowed. Use \`alpha in {0.5, 1.0}\` for candidate differences and **full-state T+ replacement** as a separate upper-bound diagnostic, **not** another additive alpha.

Controls at same tensor position, comparable norm, and scaling convention: \`T+ - T0\`, \`T+ - Tmis\`, norm-matched random, within-layer sample-shuffled direction, Tsame/no-op, and full-state teacher replacement. Preserve sample identity, direction, choice order, and view preprocessing in each matched comparison.

### 8.2 Outcome measurement: pre-answer vs. answer position

**Pre-answer and t=0 anchors:** Natural next-token answer-letter scores are **not valid primary readouts**; the existing probe reported AUC 0.50–0.54 before answer. Following the patch, append the **same forced answer-elicitation suffix** (e.g. \`So the answer is (\`) and read choice-letter probabilities, reusing the existing \`probe_hidden_answer\` method. Critical: the patch must occur at the original anchor, **before** the forced suffix; replay its downstream effects through the suffix. A fresh forced suffix that discards the patched computation does not test the intervention.

**Natural answer-decision anchor:** Read the next choice-token logits directly, using the exact causal predictor position *before* generating the selected answer token; where the chosen format requires it, also report the corresponding forced readout. Report whether the two readouts agree.

For both, use the probability distribution over the **actual candidate option letters** (with consistent prompt, choice mapping, and tokenizer handling). Record forced-readout entropy, KL/JSD between letter distributions, chosen-letter switch rates, and teacher/Student agreement. Avoid treating a single next-token natural logit at arbitrary early positions as an answer decision.

### 8.3 GT-free pilot selection, separated from GT diagnostics

To choose layer and dose on the development set, use **only quantities that do not involve GT**. Example: convergence toward the T+ teacher's *forced-answer letter distribution* under the same elicitation convention:

\[
G_{\mathrm{T+}} =
D_{\mathrm{KL}}(q_{\mathrm{T+}}^{\mathrm{read}}\|p_S^{\mathrm{read}})
-
D_{\mathrm{KL}}(q_{\mathrm{T+}}^{\mathrm{read}}\|p_{\mathrm{patched}}^{\mathrm{read}}).
\]

Higher values indicate greater **teacher agreement**, **not correctness**. Report collapse/entropy and no-op checks to prevent false convergence. The teacher can itself be wrong or uncertain, so improvement of this metric is **not an automatic Go**.

**GT-based answer margin/accuracy are only diagnostics on the held-out set after freezing layer/alpha/readout choice**. Never use GT to choose a direction, select samples, tune dose, set a gate, or construct a distillation target. Optional GT accuracy analysis on development must not be used for method selection.

Correctness-oriented diagnostic, at answer token or forced readout (as applicable):

\[
M_{\mathrm{GT}}=\log p(v_{\mathrm{GT}})-\log p(v_{\mathrm{alt}}),
\quad
\Delta M_{\mathrm{GT}}=M_{\mathrm{GT}}^{\mathrm{patched}}-M_{\mathrm{GT}}^{S}.
\]

Specify how alternative letters \`v_alt\` are aggregated and handle multi-token labels exactly; report accuracy/GT-class probability as separate checks. Never call a teacher-alignment gain a GT-correction gain.

### 8.4 Paired statistics and analysis

Analyze **answer-first (t=0) and reasoning-first (a>=1) separately**, then an appropriately weighted aggregate. Report effects at \`t=0\`, \`answer\`, \`answer-1\`, \`answer-4\` (invalid/duplicate anchors omitted) and by layer family. Use paired bootstrap intervals and paired tests; correct for multiple condition/layer/dose searches. No cherry-picking of positive patch cases to claim held-out gains.

Include matched output-level OPD-Aha agreement and output visual contrast where available, checking whether hidden interventions add *nonredundant* effects instead of merely replicating strong token-logit steering.

## 9. Phase 3 — continuation and mechanism checks

Use **50 predesignated questions** from the pilot holdout for patched and unpatched full-replay free continuation, balanced between answer-first and reasoning-first when possible. No teacher forcing / answer elicitation suffix in free continuation.

- [ ] At t=0, evaluate impact on **immediate answer-first behavior**, choice accuracy, and answer timing; do not treat t=0 as merely a style diagnostic.
- [ ] For reasoning-first, compare anchors \`answer\`, \`answer-1\`, \`answer-4\` where each exists, and distinguish interventions before and after committed decisions.
- [ ] Evaluate final GT accuracy *only as a held-out diagnostic*, explicit/silent recovery, reversals, choice switching, response style, and length.
- [ ] Report both conditional results and population frequency; do not report only selected "successful patches."
- [ ] Compare to teacher-output OPD-Aha contrast, same-norm controls, and if feasible equivalent direct logit-steering.
- [ ] Verify correct/mis/null effects track underlying visual content, not red-box/crop view format or generic answer bias.

A forced-readout letter flip is a **local causal readout effect**; it is not evidence of successful autonomous continuation. A single position intervention during response generation may interact with future DeltaNet recurrence; describe exactly what the full-replay implementation perturbs.

## 10. Minimal pilot scope, decision gate, and execution procedure

### 10.1 Fixed first-pass pilot scope (proposed)

| Aspect | Plan |
| --- | --- |
| Dataset | off6k, **200 questions x 1 base trajectory**, no new training |
| Split | Development vs. strictly separated holdout, by image/object group if possible (suggest 100/100; freeze before evaluating) |
| Model | **Qwen3.5-4B**, 32 layers, 2560 dim; hybrid DeltaNet/full attention |
| Layers | **8, 12, 15, 19, 23, 27** |
| Positions | **t=0**, answer-decision, answer−1, answer−4 where valid |
| Directions | T+−T0, T+−Tmis, norm-matched random, same-layer shuffled; Tsame no-op |
| Doses | **alpha 0.5 and 1.0**; T+ full-state replacement upper-bound |
| Pre-answer readout | Forced suffix **\`So the answer is (\`**, existing \`probe_hidden_answer\` scoring; full replay |
| Answer readout | Natural next-token choice distribution at the predictive anchor |
| GT use | **Held-out diagnosis only; never selector, direction construction, or target construction** |
| Continuation | **50 questions**, patched/unpatched full-replay free generation, no forced suffix |
| Compute | User's rough estimate **one RTX Pro 6000, 2–3 hours**, not independently measured; verify throughput/storage first |

The number of valid anchors depends on answer position; do not force \`answer−4\` when the answer occurs too early. Distinct \`t=0\` answer-first and reasoning-first results are mandatory.

### 10.2 Gate and selection rules

1. **Pilot/development:** choose one layer, alpha, direction, and readout convention using **GT-free** teacher-distribution agreement / stability metrics and controls. Explicitly state whether optimizing agreement with T+ can favor an erroneous teacher.
2. **Freeze:** save the selected config, sample IDs, metrics, and multiple-testing protocol *before* held-out evaluation.
3. **Held-out:** report GT-aware effects and paired uncertainty; compare to all controls and OPD-Aha. GT can be used to *evaluate* causal correctness, not to construct or select a target.
4. **Go only if:** held-out evidence is content-specific, beyond random/shuffled/no-op; yields actual answer/continuation improvement; and shows a credible complementary regime beyond output OPD-Aha.
5. **Partial Go:** strong teacher-consistent forced-readout change without reliable GT/continuation gain. Continue mechanism probing, **no training**.
6. **No-Go:** no specific causal effect, teacher agreement without correctness, or no incremental utility over OPD-Aha.

Do not claim error prevention merely from an answer-position anchor, or claim that GT-free teacher agreement guarantees factual correctness.

### 10.3 Execution workflow (separate from this document change)

1. Implement **one pilot script** reusing local \`probe_hidden_answer\` and \`probe_view_hidden\` functions **after checking they exist on the execution host**; do not invent GitHub paths or modify production training.
2. Perform static code review: causal indexing, forced-suffix injection, visual-token alignment, full-sequence replay, DeltaNet recurrent-state behavior, controls, deterministic seeding, output schemas.
3. On the HPC **login node: syntax/import checks only**, no GPU inference or substantive computation.
4. Request **one GPU hold**, run a tiny GPU smoke test, then the bounded 200-question pilot.
5. Release the hold and stop any recurring checks after the run; preserve provenance and resource logs.

No code execution, cluster login, hold request, or task scheduling is authorized by this document edit alone.

## 11. Conditional training extensions (NOT approved until Go)

Two distinct training ideas must be compared rather than conflated:

**A. Hidden-state representation target:** A selected, causally validated direction defines a detached target \(h_{\mathrm{target}}=\operatorname{sg}[h_{S,\mathrm{old}}+\alpha P_l d_l]\). Train only validated layers/positions/subspaces; do not do indiscriminate all-layer teacher-state MSE. Validate that target projection and representation scale are stable under model updates.

**B. Hidden-intervention-guided output target:** Apply a validated intervention to a frozen rollout-time student, produce \(q_t^H=\operatorname{sg}[\mathrm{softmax}(z_{\mathrm{patched}})]\), and distill the *unpatched* student output distribution toward \(q_t^H\). This is **output-level OPD with hidden intervention used to construct the target**, not direct hidden-state distillation.

The user's initial compute comparison suggests Arm D may add about one Student forward per training step and remain in the same broad cost class as OPD-Aha's A-arm three-forward procedure; **verify actual passes, memory and wall-clock before making efficiency claims**.

Run matched comparisons:

| Arm | Objective | Purpose |
| --- | --- | --- |
| A | OPD-Aha | Strong existing baseline |
| B | Selected-layer OPRD-like hidden MSE | Tests ordinary representation alignment |
| C | Causally selected visual-target hidden loss | Tests selective representation supervision |
| D | Detached hidden-steered output target | Tests targeted token-preference supervision |
| E | OPD-Aha + best validated extension | Tests complementarity |

Control rollout source, data, steps, token budget, model checkpoint, decoding, seeds, and GPU costs. At evaluation, the deployable student must take **only its ordinary full image + prompt**, without teacher, crop, hidden injection, probe labels, or privileged context.

## 12. Required outputs and reporting

Suggested paths (new work products; do not assume they already exist):

- **hidden_probe/hidden_plan.md** — this plan.
- **hidden_probe/analysis/** — layer/position effect tables, paired tests, confidence intervals, diagnostic figures.
- **hidden_probe/results/** — matched-view delta metadata, intervention metrics, held-out continuation summaries.
- **hidden_probe/decisions.md** — GT-free pilot choices, pre-registered held-out comparison, and final Go / Partial-Go / No-Go rationale.
- **hidden_probe/pilot_*.py** — optional later pilot scripts; not created by this plan revision.
- **eval/probe_hidden/viewhid_1022/summary.json** — existing reported experiment artifact on the research environment, not assumed to be in GitHub.

Always distinguish **observed probe findings**, **hypotheses**, **causal tests**, and **conditional training proposals** in reporting.

## 13. Related methods

- **OPD-Aha:** reconstructs teacher *output-space* visual preference using privileged vs. null-view probabilities.
- **OPRD (arXiv:2606.06021):** matches teacher/student hidden representations on on-policy trajectories; direct MSE is not evidence that a view-change delta is semantically corrective.
- **Activation/causal patching:** distinguishes whether internal representations affect behavior, but does not by itself establish learnability through weight updates.

**Central falsifiable claim to test:** A subset of privileged, content-sensitive hidden-state differences causally supports visual correction that is not adequately expressed by output-level teacher preference; selective use of those differences may improve VLM OPD only after that causal claim is validated.
