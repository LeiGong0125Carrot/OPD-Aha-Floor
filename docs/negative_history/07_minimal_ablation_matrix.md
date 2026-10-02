# Section 07 — Minimal Ablation Matrix

## 1. Purpose

This section defines the minimum experimental matrix needed to isolate the contribution of:

1. suppression-only visual evidence,
2. historical state-adaptive suppression strength,
3. future-persistence-based JSD weighting.

The goal is not to maximize the number of variants, but to obtain clean causal attribution between algorithmic components while keeping the OPD-Aha training regime fixed.

The main questions are:

\[
\boxed{
\text{Q1: Is positive visual amplification necessary?}
}
\]

\[
\boxed{
\text{Q2: Does historical conflict improve the choice of suppression strength?}
}
\]

\[
\boxed{
\text{Q3: Does future conflict persistence improve token-level supervision prioritization?}
}
\]

---

## 2. Core five-arm matrix

The first-stage matrix should contain only five arms:

| Arm | Negative-only | History-adaptive \(\beta_t\) | Future JSD weight \(w_t\) | Purpose |
|---|:---:|:---:|:---:|---|
| **A: OPD-Aha** | ✗ | ✗ | ✗ | Original reference |
| **B: Suppression-only** | ✓ | ✗ | ✗ | Test whether positive \(u_t(v)\) is necessary |
| **C: Suppression-only + History** | ✓ | ✓ | ✗ | Test history-adaptive suppression |
| **D: Suppression-only + Future** | ✓ | ✗ | ✓ | Test future-based supervision prioritization |
| **E: Full** | ✓ | ✓ | ✓ | Combined method |

Arms B/C/D/E form a clean \(2\times2\) factorial design:

| | Future OFF | Future ON |
|---|---|---|
| **History OFF** | B | D |
| **History ON** | C | E |

Arm A remains the original signed OPD-Aha reference.

---

## 3. Arm A — OPD-Aha

The signed visual preference is

\[
u_t(v)
=
\log
\frac{p_t^+(v)}
{p_t^0(v)}.
\]

The original reconstructed target is

\[
\boxed{
q_t^A(v)
\propto
p_t^+(v)
\exp
\left(
\beta u_t(v)
\right)
}
\]

with fixed

\[
\beta_t=\beta
\]

and uniform token weight

\[
w_t=1.
\]

The loss uses the existing OPD-Aha global token-mean aggregation:

\[
\boxed{
L_A
=
\frac{
\sum_{b,t}
M_{b,t}
D_{\mathrm{JS}}
\left(
q_{b,t}^A,
p_{b,t}^S
\right)
}{
\sum_{b,t}M_{b,t}
}.
}
\]

This arm is the reference for testing whether the positive half of the visual preference is necessary.

---

## 4. Arm B — Suppression-only

Define

\[
n_t(v)
=
[-u_t(v)]_+.
\]

The target becomes

\[
\boxed{
q_t^B(v)
\propto
p_t^+(v)
\exp
\left(
-\beta n_t(v)
\right).
}
\]

The other components remain unchanged:

\[
\beta_t=\beta,
\]

\[
w_t=1.
\]

Therefore A \(\rightarrow\) B changes only the visual signal used for reconstruction:

\[
\boxed{
u_t(v)
\rightarrow
[-u_t(v)]_+.
}
\]

The comparison

\[
\boxed{
B-A
}
\]

answers:

> Is explicit positive visual amplification necessary, or is negative visual suppression sufficient?

Existing suppression-only experiments motivate this comparison, but the formal ablation should be rerun under exactly matched training conditions to remove rollout-regime or implementation confounds.

---

## 5. Arm C — Suppression-only + History

Arm C keeps the same negative-only signal as B:

\[
n_t(v)
=
[-u_t(v)]_+.
\]

For the realized student token:

\[
c_t
=
n_t(y_t).
\]

Historical accumulated negative conflict is

\[
N_t
=
\sum_{k<t}c_k.
\]

The bounded history state is

\[
s_t
=
\frac{N_t}{1+N_t}.
\]

Then

\[
\boxed{
\beta_t
=
\beta(1+s_t)
}
\]

for the first zero-extra-hyperparameter version.

The reconstructed target is

\[
\boxed{
q_t^C(v)
\propto
p_t^+(v)
\exp
\left(
-\beta_t n_t(v)
\right).
}
\]

The token weight remains

\[
w_t=1.
\]

Thus

\[
\boxed{
C-B
}
\]

isolates the effect of replacing fixed suppression strength with historical state-adaptive suppression strength.

---

## 6. Arm D — Suppression-only + Future

Arm D uses exactly the same target geometry as B:

\[
\boxed{
q_t^D=q_t^B.
}
\]

Therefore

\[
\beta_t=\beta.
\]

Future information affects only the token-level JSD weight.

Define:

\[
F_t
=
\sum_{k>t}c_k,
\]

\[
K_t
=
\sum_{k>t}M_k,
\]

\[
\bar F_t
=
\frac{
F_t
}{
\max(K_t,1)
},
\]

\[
r_t
=
\frac{
\bar F_t
}{
1+\bar F_t
},
\]

and

\[
\boxed{
w_t
=
1+r_t.
}
\]

The final objective is the global weighted token mean:

\[
\boxed{
L_D
=
\frac{
\sum_{b,t}
M_{b,t}
w_{b,t}
D_{\mathrm{JS}}
\left(
q_{b,t}^B,
p_{b,t}^S
\right)
}{
\sum_{b,t}
M_{b,t}
w_{b,t}
}.
}
\]

Thus

\[
\boxed{
D-B
}
\]

isolates the contribution of retrospective future-persistence-based supervision prioritization.

---

## 7. Arm E — Full method

Arm E combines both temporal components on top of negative-only reconstruction.

Current negative evidence:

\[
n_t(v)
=
[-u_t(v)]_+.
\]

Realized conflict:

\[
c_t=n_t(y_t).
\]

History:

\[
N_t
=
\sum_{k<t}c_k,
\]

\[
s_t
=
\frac{N_t}{1+N_t},
\]

\[
\boxed{
\beta_t
=
\beta(1+s_t).
}
\]

Future:

\[
F_t
=
\sum_{k>t}c_k,
\]

\[
K_t
=
\sum_{k>t}M_k,
\]

\[
\bar F_t
=
\frac{F_t}{\max(K_t,1)},
\]

\[
r_t
=
\frac{\bar F_t}{1+\bar F_t},
\]

\[
\boxed{
w_t
=
1+r_t.
}
\]

The target is

\[
\boxed{
q_t^E(v)
\propto
p_t^+(v)
\exp
\left(
-\beta_tn_t(v)
\right)
}
\]

and the final loss is

\[
\boxed{
L_E
=
\frac{
\sum_{b,t}
M_{b,t}
w_{b,t}
D_{\mathrm{JS}}
\left(
q_{b,t}^E,
p_{b,t}^S
\right)
}{
\sum_{b,t}
M_{b,t}
w_{b,t}
}.
}
\]

---

## 8. Worked example across the five arms

Suppose at one response position the teacher distributions are:

| token | \(p_t^+\) | \(p_t^0\) |
|---|---:|---:|
| \(\texttt{diamond}\) | 0.35 | 0.55 |
| \(\texttt{striped}\) | 0.30 | 0.15 |
| \(\texttt{wait}\) | 0.10 | 0.05 |
| other | 0.25 | 0.25 |

Then

\[
u_t
\approx
[-0.452,\;0.693,\;0.693,\;0].
\]

Assume

\[
\beta=2.
\]

### Arm A

Using full signed OPD-Aha:

\[
q_A(v)
\propto
p^+(v)e^{2u(v)}.
\]

The approximate reconstructed distribution is

\[
\boxed{
q_A
\approx
[0.071,\;0.602,\;0.201,\;0.126].
}
\]

Positive visual evidence explicitly amplifies \(\texttt{striped}\) and \(\texttt{wait}\).

### Arm B

Negative-only signal:

\[
n_t
=
[0.452,\;0,\;0,\;0].
\]

Then

\[
q_B(v)
\propto
p^+(v)e^{-2n_t(v)}.
\]

Approximately,

\[
\boxed{
q_B
\approx
[0.179,\;0.379,\;0.126,\;0.316].
}
\]

Only \(\texttt{diamond}\) is explicitly suppressed.

### Arm C

Suppose historical accumulated conflict is

\[
N_t=0.8.
\]

Then

\[
s_t
=
\frac{0.8}{1.8}
\approx0.444,
\]

so

\[
\beta_t
=
2(1+0.444)
\approx2.889.
\]

Thus

\[
q_C(v)
\propto
p^+(v)e^{-2.889n_t(v)},
\]

giving approximately

\[
\boxed{
q_C
\approx
[0.127,\;0.403,\;0.134,\;0.336].
}
\]

History changes target geometry by strengthening suppression of the same current negative set.

### Arm D

The target remains

\[
q_D=q_B.
\]

Suppose

\[
\bar F_t=0.25.
\]

Then

\[
r_t
=
\frac{0.25}{1.25}
=
0.20,
\]

so

\[
\boxed{
w_t=1.20.
}
\]

The target is unchanged; only the token-level JSD contribution receives more relative weight.

### Arm E

The full arm uses

\[
q_E=q_C
\]

and

\[
w_t=1.20.
\]

Thus history changes target strength, while future changes supervision importance through a separate interface.

---

## 9. Interpreting the main comparisons

### A vs B

\[
\boxed{
B-A
}
\]

tests whether the positive half of \(u_t(v)\) is necessary.

Possible interpretations:

- \(B\approx A\): positive amplification is not necessary.
- \(B>A\): positive amplification may be harmful or noisy.
- \(B<A\): positive visual evidence contains useful corrective information that suppression-only discards.

### B vs C

\[
\boxed{
C-B
}
\]

tests historical adaptive suppression.

If \(C>B\), accumulated conflict may provide useful state information.

If \(C\approx B\), current \(u_t(v)\) may already encode enough prefix state, and explicit historical accumulation may be redundant.

### B vs D

\[
\boxed{
D-B
}
\]

tests future-based supervision prioritization.

If \(D>B\), persistent downstream visual conflict identifies positions worth emphasizing.

If \(D\approx B\), future persistence adds little beyond current token-level OPD supervision.

If \(D<B\), downstream conflict may be too weakly related to the current state's corrective value, causing misallocated supervision.

---

## 10. Factorial interaction analysis

Because B/C/D/E form a \(2\times2\) design, define the history effect without future:

\[
\Delta_H^{F=0}
=
C-B.
\]

History effect with future:

\[
\Delta_H^{F=1}
=
E-D.
\]

Future effect without history:

\[
\Delta_F^{H=0}
=
D-B.
\]

Future effect with history:

\[
\Delta_F^{H=1}
=
E-C.
\]

These comparisons determine whether the modules are:

- approximately additive,
- partially redundant,
- positively interacting,
- or negatively interacting.

A simple interaction term is

\[
\boxed{
I
=
E-C-D+B.
}
\]

However, small numerical interaction values should not be over-interpreted without paired uncertainty analysis.

---

## 11. Hypothetical example

Suppose a benchmark produces:

| Arm | Score |
|---|---:|
| A | 49.4 |
| B | 49.3 |
| C | 50.1 |
| D | 49.8 |
| E | 51.0 |

These are illustrative numbers only.

Then:

\[
B-A=-0.1,
\]

so suppression-only is approximately matched to Aha.

History effect:

\[
C-B=+0.8.
\]

Future effect:

\[
D-B=+0.5.
\]

Full improvement over B:

\[
E-B=+1.7.
\]

The simple additive expectation would be

\[
0.8+0.5=1.3.
\]

The remaining difference suggests a possible positive interaction, but this should be validated with paired examples and uncertainty rather than treated as conclusive from aggregate score alone.

---

## 12. Why not run eight arms immediately?

A full three-factor design would include:

- signed vs negative-only,
- history off/on,
- future off/on,

for

\[
2^3=8
\]

arms.

The first-stage scientific hypothesis, however, is specifically about using negative visual evidence as the reliable base signal and then adapting its strength with trajectory information.

Therefore the most informative first-stage structure is:

\[
A
\rightarrow
B
\]

followed by the \(2\times2\) matrix on B:

\[
\begin{matrix}
B & D\\
C & E
\end{matrix}.
\]

Signed-Aha + History/Future variants can be added later only if temporal adaptation shows a useful signal and the remaining question becomes whether the gain is specific to suppression-only reconstruction.

---

## 13. Matched constant-\(\beta\) control for history

History adaptation introduces an important dose confound.

Arm B uses fixed

\[
\beta.
\]

Arm C uses

\[
\beta_t\in[\beta,2\beta).
\]

Therefore if C improves over B, one alternative explanation is simply:

> The average suppression strength is larger.

Suppose training C yields

\[
\mathbb E[\beta_t]=5.4
\]

while B uses

\[
\beta=4.
\]

A follow-up control should use a constant

\[
\boxed{
\beta_{\mathrm{const}}
=
\mathbb E[\beta_t]
}
\]

without history adaptation.

Call this arm

\[
C_{\mathrm{const}}.
\]

Then:

- if \(C_{\mathrm{const}}\approx C\), the gain may be primarily a stronger average suppression dose;
- if \(C>C_{\mathrm{const}}\), it supports the claim that **when** suppression is increased matters, not only how much suppression is applied on average.

This control is especially important if history produces a clear performance gain.

---

## 14. Optional future-weight permutation control

Future weighting does not have the same average-strength confound because the final loss is normalized as a weighted token mean.

Still, a useful later diagnostic is to preserve the same weight distribution while randomly permuting the token-position assignment:

\[
w_t
\rightarrow
w_t^{\mathrm{perm}}.
\]

If the true future alignment helps while permuted weighting does not, that supports the interpretation that the future trajectory information, rather than generic non-uniform weighting, is useful.

This is a second-stage diagnostic rather than a first-stage main arm.

---

## 15. Training conditions that must remain fixed

All primary arms should use identical:

- training subset,
- image views,
- null view,
- student initialization,
- frozen teacher initialization,
- seed,
- batch size,
- number of rollouts per prompt,
- learning rate,
- number of steps / epochs,
- maximum response length,
- top-\(k\) + tail support,
- JSD \(\alpha\),
- base \(\beta\),
- checkpoint evaluation schedule,
- loss aggregation mode.

The loss aggregation must remain the OPD-Aha global token-level geometry:

- A/B/C: global token mean,
- D/E: global weighted token mean normalized by the global sum of token weights.

No rollout-wise normalization should be introduced.

---

## 16. Minimal config switches

A clean implementation can expose three booleans:

~~~yaml
negative_only: false
history_adaptive_beta: false
future_jsd_weight: false
~~~

### Arm A

~~~yaml
negative_only: false
history_adaptive_beta: false
future_jsd_weight: false
~~~

### Arm B

~~~yaml
negative_only: true
history_adaptive_beta: false
future_jsd_weight: false
~~~

### Arm C

~~~yaml
negative_only: true
history_adaptive_beta: true
future_jsd_weight: false
~~~

### Arm D

~~~yaml
negative_only: true
history_adaptive_beta: false
future_jsd_weight: true
~~~

### Arm E

~~~yaml
negative_only: true
history_adaptive_beta: true
future_jsd_weight: true
~~~

This structure makes bit-identical controls easier to verify.

---

## 17. Mechanism metrics to log

Performance alone is insufficient. Each arm should log mechanism dose and state statistics.

### Negative signal

Recommended metrics:

\[
\operatorname{frac}[u_t(v)<0],
\]

\[
\mathbb E[c_t].
\]

### History

Log:

\[
\mathbb E[N_t],
\]

\[
\mathbb E[s_t],
\]

\[
\mathbb E[\beta_t],
\]

and the upper quantiles / maximum of \(\beta_t\).

If

\[
s_t\approx1
\]

for most positions, the history transform is saturating too early and effectively collapses to approximately \(2\beta\).

### Future

Log:

\[
\mathbb E[\bar F_t],
\]

\[
\mathbb E[r_t],
\]

\[
\mathbb E[w_t],
\]

and the weight distribution.

If

\[
w_t\approx1
\]

almost everywhere, the future arm has negligible intervention dose.

### Target movement

Continue to record

\[
\boxed{
TV(q_t,p_t^+)
}
\]

or the existing target-TV metric.

This is necessary to determine whether a performance gain from history is truly state adaptation or merely larger average target movement.

---

## 18. Recommended experiment order

A practical order is:

### Stage 1

Run

\[
A,\quad B
\]

under exactly matched conditions to establish the suppression-only baseline.

### Stage 2

Run

\[
C,\quad D
\]

relative to B to isolate history and future contributions.

### Stage 3

Run

\[
E
\]

once at least one temporal module shows useful signal, or run it explicitly to test interaction.

### Stage 4

If C or E improves materially, run the matched constant-\(\beta\) control.

If D or E improves materially, optionally run the future-weight permutation diagnostic.

---

## 19. Recommended main table

| Arm | Neg-only | History | Future | TreeBench | V* | Target TV | Mean \(\beta_t\) | Mean \(w_t\) |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| A: Aha | ✗ | ✗ | ✗ | | | | \(\beta\) | 1 |
| B: Sup | ✓ | ✗ | ✗ | | | | \(\beta\) | 1 |
| C: Sup+Hist | ✓ | ✓ | ✗ | | | | measured | 1 |
| D: Sup+Future | ✓ | ✗ | ✓ | | | | \(\beta\) | measured |
| E: Full | ✓ | ✓ | ✓ | | | | measured | measured |

Performance and intervention dose should be shown together so that stronger performance is not confused with simply stronger target movement.

---

## 20. Main conclusion

The first-stage ablation should use:

\[
\boxed{
A
\rightarrow
B
}
\]

to test whether positive visual amplification is necessary, followed by the suppression-only \(2\times2\) matrix

\[
\boxed{
\begin{matrix}
B & D\\
C & E
\end{matrix}
}
\]

to isolate history and future effects.

The key comparisons are:

\[
\boxed{
C-B
=
\text{history effect}
}
\]

\[
\boxed{
D-B
=
\text{future effect}
}
\]

\[
\boxed{
E-D
=
\text{history effect given future}
}
\]

\[
\boxed{
E-C
=
\text{future effect given history}
}
\]

If history improves performance, the highest-priority follow-up is a matched constant-\(\beta\) control to separate state adaptation from increased average suppression strength.

If future weighting improves performance, a later weight-permutation control can test whether the gain comes from meaningful temporal alignment rather than generic non-uniform token weighting.
