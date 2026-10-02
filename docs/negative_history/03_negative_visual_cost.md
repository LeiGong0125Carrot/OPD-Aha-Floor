# Section 03 — Negative Visual Cost for Negative-History OPD

## 1. Purpose

This section defines how to extract the **negative direction** of OPD-Aha's real-vs-null visual preference and reuse it in two distinct places:

1. **Vocabulary-level suppression**, which modifies the reconstructed OPD target over candidate next tokens.
2. **Trajectory-level conflict accumulation**, which measures how much visual opposition the student's actually sampled response has accumulated over time.

The central hypothesis is:

> Visual evidence is more reliable as evidence against an image-inconsistent continuation than as a direct prescription of which alternative token should be amplified.

The implementation therefore keeps OPD-Aha's real/null visual signal, but discards its positive half when constructing the negative-history variant.

This is a candidate method design, not an already validated result.

---

## 2. Base visual preference from OPD-Aha

At response position (t), under the same student-generated prefix

[
h_t=(x,y_{<t}),
]

the frozen teacher is evaluated under:

- privileged visual evidence (I^+),
- matched visual null (I^0).

The two teacher distributions are

[
p_t^+(v)=p_phi(vmid I^+,h_t),
]

[
p_t^0(v)=p_phi(vmid I^0,h_t).
]

OPD-Aha defines the token-level visual preference

[
oxed{
u_t(v)
=
log p_t^+(v)-log p_t^0(v)
=
lograc{p_t^+(v)}{p_t^0(v)}
}
]

with the interpretation:

- (u_t(v)>0): privileged visual evidence increases support for token (v),
- (u_t(v)=0): privileged evidence leaves the token unchanged relative to null,
- (u_t(v)<0): privileged visual evidence suppresses token (v).

The negative-history method keeps this log-ratio representation unchanged.

---

## 3. Why use the log-ratio rather than probability difference?

Two natural negative visual costs are possible.

### 3.1 Probability-space gap

[
c_t^{mathrm{prob}}(v)
=
[p_t^0(v)-p_t^+(v)]_+.
]

This measures the amount of **absolute probability mass** removed by the visual evidence.

### 3.2 Log-ratio negative evidence

[
c_t^{log}(v)
=
[-u_t(v)]_+
=
left[
lograc{p_t^0(v)}{p_t^+(v)}
ight]_+.
]

This measures the **relative suppression** caused by the visual evidence.

Here

[
[x]_+=max(x,0).
]

The two quantities have different semantics and should not be mixed.

---

## 4. Worked example: high-probability token

Suppose the current candidate token is

[
v=	exttt{diamond}.
]

The null teacher gives

[
p_t^0(	exttt{diamond})=0.80,
]

and the privileged teacher gives

[
p_t^+(	exttt{diamond})=0.60.
]

Then the probability-gap cost is

[
c_t^{mathrm{prob}}
=
0.80-0.60
=
0.20.
]

The log-ratio cost is

[
c_t^{log}
=
lograc{0.80}{0.60}
approx
0.288.
]

The two numbers are not interchangeable:

- (0.20) means the evidence removed (0.20) absolute probability mass.
- (0.288) nats means the null condition supports the token about (1.33	imes) as strongly as the real-evidence condition.

Equivalently,

[
rac{p_t^+}{p_t^0}
=
0.75,
]

so the privileged evidence reduces the relative support for this token to (75%) of its null value.

---

## 5. Worked example: why probability gap and log-ratio differ

Consider two tokens.

### Token A

[
p^0(A)=0.80,
qquad
p^+(A)=0.40.
]

Then

[
c^{mathrm{prob}}(A)=0.40,
]

while

[
c^{log}(A)
=
lograc{0.80}{0.40}
=
log 2
approx0.693.
]

### Token B

[
p^0(B)=0.02,
qquad
p^+(B)=0.01.
]

Then

[
c^{mathrm{prob}}(B)=0.01,
]

while

[
c^{log}(B)
=
lograc{0.02}{0.01}
=
log 2
approx0.693.
]

Therefore:

- probability gap says token A receives much stronger negative evidence because much more mass moved;
- log-ratio says both tokens are suppressed by the same multiplicative factor, (2	imes).

For the negative-history method, the log-ratio form is the preferred primary signal because the historical trajectory statistic later relies on additive accumulation across autoregressive positions.

---

## 6. Main vocabulary-level negative signal

Define

[
oxed{
n_t(v)
=
[-u_t(v)]_+
}
]

or equivalently

[
n_t(v)
=
max
left(
lograc{p_t^0(v)}{p_t^+(v)},
0
ight).
]

This keeps only the negative half of the visual preference.

The cases are:

### Visual support

If

[
p_t^+(v)>p_t^0(v),
]

then

[
u_t(v)>0
]

and

[
n_t(v)=0.
]

The negative-history method does not directly amplify this token.

### Visual neutrality

If

[
p_t^+(v)=p_t^0(v),
]

then

[
u_t(v)=0
]

and

[
n_t(v)=0.
]

### Visual opposition

If

[
p_t^+(v)<p_t^0(v),
]

then

[
u_t(v)<0
]

and

[
n_t(v)=-u_t(v)>0.
]

The token becomes eligible for suppression.

---

## 7. Vocabulary-level suppression-only reconstruction

A direct negative-only reconstruction is

[
oxed{
q_t^{mathrm{sup}}(v)
propto
p_t^+(v)
exp
left(
-eta_t n_t(v)
ight)
}
]

with

[
n_t(v)=[-u_t(v)]_+.
]

The symbol (propto) means that the right-hand side gives an unnormalized weight. The normalized target is

[
q_t^{mathrm{sup}}(v)
=
rac{
p_t^+(v)
exp(-eta_t n_t(v))
}{
sum_w
p_t^+(w)
exp(-eta_t n_t(w))
}.
]

This construction has one important property:

[
exp(-eta_t n_t(v))
le1.
]

Therefore no token receives an explicit positive multiplier greater than (1).

The method only:

- suppresses tokens with negative visual evidence,
- leaves non-negative tokens unchanged before normalization.

Any relative increase of an alternative token occurs only because probability mass is redistributed during normalization.

---

## 8. Pairwise interpretation

For two tokens (v) and (w),

[
log
rac{q_t^{mathrm{sup}}(v)}
{q_t^{mathrm{sup}}(w)}
=
log
rac{p_t^+(v)}
{p_t^+(w)}
-
eta_t
left[
n_t(v)-n_t(w)
ight].
]

Consider

[
v=	exttt{striped},
qquad
w=	exttt{diamond}.
]

Suppose

[
u_t(	exttt{striped})=+1.0,
]

[
u_t(	exttt{diamond})=-0.5.
]

Then

[
n_t(	exttt{striped})=0,
]

[
n_t(	exttt{diamond})=0.5.
]

Therefore

[
log
rac{q_t^{mathrm{sup}}(	exttt{striped})}
{q_t^{mathrm{sup}}(	exttt{diamond})}
=
log
rac{p_t^+(	exttt{striped})}
{p_t^+(	exttt{diamond})}
+
0.5eta_t.
]

The relative improvement of (	exttt{striped}) comes entirely from suppressing (	exttt{diamond}).

For comparison, full OPD-Aha would use

[
u_t(	exttt{striped})-u_t(	exttt{diamond})
=
1.0-(-0.5)
=
1.5,
]

so its pairwise correction contains both:

- positive amplification of (	exttt{striped}),
- negative suppression of (	exttt{diamond}).

The negative-history variant retains only the second component.

---

## 9. Realized-token negative conflict

The trajectory branch does not need the full vocabulary signal.

Let the student actually sample token

[
y_t.
]

Gather the visual preference at the sampled token:

[
u_t(y_t).
]

Define the realized negative conflict

[
oxed{
c_t
=
[-u_t(y_t)]_+
}
]

or

[
c_t
=
n_t(y_t).
]

This is a scalar for each sampled response position.

Its interpretation is:

> How much negative real-vs-null visual evidence did the token actually chosen by the student receive?

This scalar is used only to characterize the sampled trajectory. It is distinct from the vocabulary-wide (n_t(v)) used for target reconstruction.

---

## 10. Worked trajectory example

Suppose the student generates

```text
The pattern is diamond because diagonal shapes
```

and the real/null teacher probabilities at the sampled tokens are:

| sampled token (y_t) | (p_t^0(y_t)) | (p_t^+(y_t)) |
|---|---:|---:|
| `The` | 0.60 | 0.62 |
| `pattern` | 0.55 | 0.58 |
| `is` | 0.70 | 0.70 |
| `diamond` | 0.50 | 0.25 |
| `because` | 0.60 | 0.55 |
| `diagonal` | 0.30 | 0.15 |
| `shapes` | 0.40 | 0.38 |

Then

[
u_t(y_t)
=
lograc{p_t^+(y_t)}{p_t^0(y_t)}.
]

The resulting realized negative conflicts are approximately:

| token | (u_t(y_t)) | (c_t=[-u_t(y_t)]_+) |
|---|---:|---:|
| `The` | (>0) | 0 |
| `pattern` | (>0) | 0 |
| `is` | 0 | 0 |
| `diamond` | (-0.693) | 0.693 |
| `because` | (-0.087) | 0.087 |
| `diagonal` | (-0.693) | 0.693 |
| `shapes` | (-0.051) | 0.051 |

The cumulative negative evidence over the realized trajectory is

[
0.693+0.087+0.693+0.051
=
1.524.
]

This does **not** mean all four negative-cost tokens are semantically wrong. The quantity is a visual-opposition proxy, not a ground-truth correctness label.

---

## 11. Why log-ratio is especially suitable for history

Without one-sided clipping, the full signed log-ratio satisfies

[
sum_k u_k(y_k)
=
sum_k
log
rac{p_k^+(y_k)}{p_k^0(y_k)}
]

[
=
log
prod_k
rac{p_k^+(y_k)}{p_k^0(y_k)}.
]

Thus additive accumulation in log space corresponds to multiplicative accumulation of real-vs-null likelihood ratios across autoregressive positions.

The one-sided history statistic

[
sum_k[-u_k(y_k)]_+
]

no longer equals the full signed trajectory log-ratio because positive evidence is intentionally discarded, but it retains the same per-token evidence scale and additivity.

By contrast,

[
sum_k
[p_k^0(y_k)-p_k^+(y_k)]_+
]

is a valid heuristic score but does not have the same natural sequence-likelihood interpretation.

For this reason, log-ratio is the default signal for the negative-history design.

---

## 12. Tail-token and numerical considerations

Log-ratios can become large when both probabilities are very small.

For example,

[
p_t^0(v)=10^{-6},
qquad
p_t^+(v)=10^{-9}
]

gives

[
n_t(v)
=
log 1000
approx6.91.
]

This is strong relative suppression, but the token is extremely low probability under both views.

The implementation should therefore preserve OPD-Aha's compressed distribution support for vocabulary-level reconstruction, such as:

- student top-(k) tokens,
- one aggregated tail bucket.

The exact support should match the existing Aha scorer to avoid introducing a new confound.

For the realized-token trajectory branch, this tail issue is less severe because (y_t) was actually sampled by the student and is therefore behaviorally relevant.

Still, all teacher log-probabilities must be finite before computing (u_t(v)).

---

## 13. Recommended tensor definitions

Assume the real/null teacher log-probabilities on the chosen support have shape

```text
[B, T, V']
```

with response mask

```text
[B, T]
```.

Compute:

```python
u_vocab = teacher_plus_logprobs - teacher_null_logprobs
negative_vocab = relu(-u_vocab)
```

where

[
u_{	ext{vocab}}[b,t,v]
=
u_t(v).
]

Then gather the actually sampled token:

```python
u_realized = gather(u_vocab, response_ids)
negative_realized = relu(-u_realized)
```

so that

[
	exttt{negative_realized}[b,t]
=
[-u_t(y_t)]_+.
]

Apply the response mask to all realized-token statistics.

---

## 14. Two logically separate uses of the same signal

The implementation must preserve two branches.

### Distribution branch

[
n_t(v)=[-u_t(v)]_+
]

is used to construct

[
q_t^{mathrm{sup}}(v).
]

This branch asks:

> Which candidate next tokens should have their target probability suppressed?

### Trajectory branch

[
c_t=n_t(y_t)
]

is used to construct historical or future trajectory statistics.

This branch asks:

> How much negative visual evidence did the student's actually sampled path accumulate?

The two quantities share the same real/null source but must remain conceptually and programmatically distinct.

---

## 15. Compatibility with OPD-Aha

This section can be integrated directly into the existing OPD-Aha implementation.

The following components remain unchanged:

- student on-policy rollout,
- privileged teacher forward,
- visual-null teacher forward,
- same-prefix scoring,
- compressed distribution support,
- student forward,
- JSD distillation geometry.

The main modification is:

[
u_t(v)
quadlongrightarrowquad
n_t(v)=[-u_t(v)]_+
]

inside target reconstruction.

At the same time, the implementation gathers

[
n_t(y_t)
]

from the same tensor to build trajectory statistics for later sections.

Therefore this section does not require a replacement training framework. It is an extension of the existing OPD-Aha scorer and target-construction path.

---

## 16. Main conclusion

The recommended primary negative visual signal is

[
oxed{
u_t(v)
=
log p_t^+(v)-log p_t^0(v)
}
]

followed by

[
oxed{
n_t(v)
=
[-u_t(v)]_+
}
]

and, for the sampled trajectory,

[
oxed{
c_t
=
n_t(y_t)
=
[-u_t(y_t)]_+.
}
]

This choice keeps OPD-Aha's already established log-ratio representation but makes a different information assumption:

> Only visual evidence that suppresses a continuation is trusted for direct target modification.

The same negative signal is then reused in two ways:

1. vocabulary-level suppression for OPD target reconstruction,
2. realized-token accumulation for history/future adaptation.

The next section should decide how the realized costs (c_t) are accumulated into a history state and how that state should modify reconstruction strength without introducing redundant or unstable scaling.
