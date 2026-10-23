# Hidden Probe Plan: Causal Visual Correction beyond Output-Level OPD

**Status:** research plan / not yet an implemented algorithm  
**Updated:** 2026-10-23  
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

**Interpretation constraint:** SC/SM around 0.66–0.72 does **not** identify an exact "one-third content / two-thirds view style" decomposition. Norms do not add linearly and differences can interact. The matched-crop control establishes that view configuration is a major confound and image content has an additional measurable effect, not a uniquely identified additive percentage.

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

### 2.3 Immediate design consequence

**Do not launch raw, all-layer hidden-state MSE.** Do not treat a teacher-student or correct-null hidden difference as a validated visual correction vector. The next study should measure **causal effect on the visual decision**, not distance alone.

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

## 5. Data, view construction, and frozen controls

Use existing base student trajectories, with the exact sampled **text prefix** shared across student and all teacher conditions. Stratify positions into:

1. **Pre-commitment:** before the first explicit incorrect visual-relation decision (tests prevention).
2. **Conflict / before answer:** a prefix that may already contain an incorrect relation (tests recoverability).
3. **Post-answer:** diagnostic only, not evidence of early correction.
4. **Pre-generation t=0:** separate style/format diagnostic; do not mix with response-position results.

Student input: the original full image. Candidate teacher views, using a frozen matched checkpoint:

- **T+**: full image + correct GT crop (also pilot crop-only separately).
- **T0**: visual-null control with explicitly defined masking/compositing and as-matched-as-possible visual layout.
- **Tmis**: incorrect *cross-question* crop, resized and matched for visual-token count, image count, preprocessing, and resolution.
- **Tuninfo**: uninformative crop with matched view configuration.
- **Tsame**: same student view, for exact no-op sanity checks.

Do not equate Tmis and T0: an incorrect crop can introduce conflicting visual information, while null removes information. Ideally use counterbalanced same-format image replacements to estimate content-specific effects.

**Checkpoint confound:** Compare matched base checkpoints for the primary representation/patching contrast. Analyze step-70 student changes in a *separate* section; never interpret a trained-student vs. base-teacher difference as purely a change of visual view.

**Layer numbering:** Previous L17–L19 causal observations belonged to an earlier Qwen2.5-VL-7B probe setting. The new view-gap report uses L31; do not assume identical causal layer indices across model/checkpoint architectures. Record total layer count, layer hook location, norm convention, and model ID before selecting candidate layers.

## 6. Phase 0 — implementation and data integrity

- [ ] Verify exact model/checkpoint, preprocessing, tokenizer, prompt template, image order, positional IDs, attention masks, visual token counts, and the response-token alignment for every view.
- [ ] Confirm student/teacher teacher-forced forwards condition on identical **student-generated textual tokens**, without future token leakage.
- [ ] Save per-example identifiers, relative position, absolute token location, output JSD, logits/margins, hidden norm/distance, and view metadata.
- [ ] Verify repeated forward and no-op patch are numerically consistent; confirm cached-KV or full-replay intervention conventions are correct.
- [ ] Freeze task split and a small pilot/development subset separately from held-out evaluation.
- [ ] Measure enough natural conflict, error, and recovered examples exist; report population prevalence. Template-induced conflicts are mechanism tests, not substitutes for natural-rollout claims.

## 7. Phase 1 — candidate representation differences (observational)

For each candidate layer and aligned response position, cache:

\[
d_{\mathrm{TS}}^l=h_{T+}^l-h_S^l,\qquad
d_{\mathrm{null}}^l=h_{T+}^l-h_{T0}^l,\qquad
d_{\mathrm{mis}}^l=h_{T+}^l-h_{Tmis}^l.
\]

These are **candidate intervention vectors, not ground-truth visual axes**.

Report norm distributions, cosine similarities, layer/position dependence, correct-vs-mismatched condition contrasts, and residual confounding by view configuration. Compare output JSD and actual visual-answer margins; avoid inferring usefulness from hidden distance or its correlation with JSD.

Explore last-layer states for descriptive diagnostics, but prioritize a small number of empirically tested mid/deep layers for causal interventions. Choose the final layer/strength grid on development data and freeze it before held-out tests.

## 8. Phase 2 — local causal intervention (primary gate)

For a student activation \(h_S^{l,t}\), evaluate in a controlled replay:

\[
h_{\mathrm{patch}}^{l,t}=
h_S^{l,t}+\alpha d^{l,t},
\]

with the **rest of the student model executing normally**. Evaluate the modified downstream logits on the same position, with consistent attention-cache handling.

**Primary task metric:**

\[
M_{\mathrm{img}}=
\log p(v_{\mathrm{image}})-\log p(v_{\mathrm{prefix}}),
\qquad
\Delta M_{\mathrm{img}}=
M_{\mathrm{img}}(\mathrm{patched})-
M_{\mathrm{img}}(\mathrm{original}).
\]

For multi-token answers, specify tokenization and whether the reported metric uses the next discriminating token or complete candidate sequence log-likelihood; never compare incompatible scoring conventions.

**Controls (same position, norm and scale):**

- [ ] T+ vs. T0 candidate contrast.
- [ ] T+ vs. Tmis and T+ vs. Tuninfo candidate contrasts.
- [ ] Norm-matched random direction.
- [ ] Same-condition within-layer shuffled delta.
- [ ] Same-view/no-op patch.
- [ ] Full-state teacher replacement only as a causal upper-bound diagnostic, *not* a visual-specific effect.
- [ ] Direct-readout / local Jacobian-dot-direction prediction versus actual downstream effect, to distinguish trivial final-logit steering from nonlinear downstream computation.

**Metrics:** mean and median paired \(\Delta M_{\mathrm{img}}\); sign flip toward correct image relation; argmax flip; entropy; change to both candidate log-probabilities; effects by original teacher-token reliability and prefix correctness.

Use paired, within-sample comparisons, bootstrap CIs and paired permutation tests. Apply multiplicity correction for multiple layers/doses/conditions; pre-specify one primary layer/dose/metric after pilot tuning. Do not claim causal improvement from cherry-picked samples with positive effect.

## 9. Phase 3 — continuation and mechanism checks

Local logits are not a complete correction. On a held-out set:

- [ ] Separate **pre-error prevention** from **post-error recovery**. A patch after an already-generated wrong statement cannot retroactively prevent that statement.
- [ ] Continue student generation after one well-defined patched position, then also after a small set of validated positions; keep the baseline decoding and token budget identical.
- [ ] Compare image-consistent final answers, actual explicit correction, silent correction, repeated incorrect commitment, wrong-to-right-to-wrong reversals, answer timing, and trajectory length.
- [ ] Compare results to plain OPD-Aha output targets and to controlled pure-logit steering of equivalent immediate margin shift.
- [ ] Verify that benefits generalize across image directions, option ordering, and held-out image/object groups rather than reflecting a left/right lexical prior.
- [ ] Review cases where token-level OPD-Aha is uncertain/incorrect but hidden intervention is effective, and the converse.

An intervention that changes only a single next-token margin without improving free continuation counts as **local steering**, not trajectory repair.

## 10. Decision gate

**Go** to a minimal training study only if the chosen intervention:

1. Has a positive held-out paired causal effect against strong matched controls with uncertainty reported;
2. Shows image-content specificity (rather than view-format/style effects);
3. Produces meaningful unpatched-policy *training potential*: at least some improvement in patched free continuation, not just instantaneous logits;
4. Offers a measurable complementary case over output-space OPD-Aha; and
5. Has an implementation/resource path that does not require privileged views at inference.

**Partial Go:** Strong, content-specific local intervention but no continuation gain. Study downstream computation, position selection, and intervention consistency; do not yet claim an OPD algorithm advance.

**No-Go:** Random/misleading controls perform similarly; effects follow generic view layout or answer bias; local changes vanish in continuation; no additional value beyond OPD-Aha. Keep the existing output-level method as primary.

Exact effect-size minima and the primary alpha/layer must be registered from a pilot *before* held-out analysis; do not tune them to the final results.

## 11. Conditional training extensions (NOT approved until Go)

Two distinct training ideas must be compared rather than conflated:

**A. Hidden-state representation target:** A selected, causally validated direction defines a detached target \(h_{\mathrm{target}}=\operatorname{sg}[h_{S,\mathrm{old}}+\alpha P_l d_l]\). Train only validated layers/positions/subspaces; do not do indiscriminate all-layer teacher-state MSE. Validate that target projection and representation scale are stable under model updates.

**B. Hidden-intervention-guided output target:** Apply a validated intervention to a frozen rollout-time student, produce \(q_t^H=\operatorname{sg}[\mathrm{softmax}(z_{\mathrm{patched}})]\), and distill the *unpatched* student output distribution toward \(q_t^H\). This is **output-level OPD with hidden intervention used to construct the target**, not direct hidden-state distillation.

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
- **hidden_probe/decisions.md** — pilot choices, pre-registered primary comparison, and final Go / Partial-Go / No-Go rationale.

Always distinguish **observed probe findings**, **hypotheses**, **causal tests**, and **conditional training proposals** in reporting.

## 13. Related methods

- **OPD-Aha:** reconstructs teacher *output-space* visual preference using privileged vs. null-view probabilities.
- **OPRD (arXiv:2606.06021):** matches teacher/student hidden representations on on-policy trajectories; direct MSE is not evidence that a view-change delta is semantically corrective.
- **Activation/causal patching:** distinguishes whether internal representations affect behavior, but does not by itself establish learnability through weight updates.

**Central falsifiable claim to test:** A subset of privileged, content-sensitive hidden-state differences causally supports visual correction that is not adequately expressed by output-level teacher preference; selective use of those differences may improve VLM OPD only after that causal claim is validated.
