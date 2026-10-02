# Section 05 — Future Conflict Persistence and Position-Weighted JSD

## 1. Purpose

This section defines how the observed continuation after response position \(t\) should affect the **importance** of OPD/JSD supervision at that position.

The design principle is:

> Current negative visual evidence decides **which tokens should be suppressed**; historical conflict decides **how strongly the target is reconstructed**; future conflict persistence decides **how much the current position should matter in the final JSD objective**.

Future information does **not** modify token membership and does **not** directly modify \(\beta_t\) in the first version.

The method remains pure OPD/JSD. No PPO-style action credit or policy-ratio objective is introduced.

---

## 2. Starting point: realized negative visual conflict

From Section 03,

\[
c_t
=
[-u_t(y_t)]_+,
\]

where

\[
u_t(y_t)
=
\log
\frac{
p_t^+(y_t)
}{
p_t^0(y_t)
}.
\]

Thus

\[
c_t\ge0
\]

measures the negative real-vs-null visual evidence received by the token actually sampled by the student at position \(t\).

---

## 3. Past, current, and future must be separated

For a response trajectory

\[
y_1,\ldots,y_T,
\]

the three temporal components are defined as:

### Past

\[
N_t
=
\sum_{k<t}c_k.
\]

### Current

\[
c_t.
\]

### Future

\[
\boxed{
F_t
=
\sum_{k>t}c_k
}
\]

The strict inequalities are important:

- history uses \(k<t\),
- current uses \(k=t\),
- future uses \(k>t\).

This prevents the current token from being counted twice.

---

## 4. Why history alone is insufficient

Consider two states with identical past and current conflict.

Suppose at position \(t\):

\[
N_t=2.0,
\qquad
c_t=0.5.
\]

Two student continuations can still behave very differently.

### Persistent continuation

Future realized costs:

\[
[0.4,\;0.3,\;0.2,\;0.1].
\]

Then

\[
F_t=1.0.
\]

### Recovering continuation

Future realized costs:

\[
[0.05,\;0,\;0,\;0].
\]

Then

\[
F_t=0.05.
\]

The past state and current conflict are identical, but the first continuation remains visually opposed while the second largely stops accumulating negative visual evidence.

Future information is therefore complementary to history.

---

## 5. Why future should measure persistence rather than total distance

The target spatial-reasoning regime usually produces responses of only a few hundred tokens. Extreme long-horizon variation is therefore not the main issue.

However, even within one 200-token response, different positions have different numbers of future tokens available.

For example,

\[
T=200.
\]

At

\[
t=40,
\]

there are approximately 160 future tokens.

At

\[
t=160,
\]

there are only approximately 40 future tokens.

If the average future negative conflict is the same,

\[
\mathbb E[c_k]=0.1,
\]

then raw sums are approximately

\[
F_{40}=16,
\]

and

\[
F_{160}=4.
\]

Raw \(F_t\) therefore confounds **future conflict persistence** with **remaining horizon length**.

For the future module, the relevant question is not:

> How much total conflict remains simply because more tokens are left?

but:

> How densely does visual conflict continue in the observed suffix?

This motivates mean future conflict rather than raw cumulative future distance.

---

## 6. Mean future conflict intensity

Let

\[
M_k\in\{0,1\}
\]

denote the valid-response mask.

Define the number of valid observed tokens after position \(t\):

\[
\boxed{
K_t
=
\sum_{k>t}M_k.
}
\]

Then define the mean future conflict intensity

\[
\boxed{
\bar F_t
=
\frac{
\sum_{k>t}c_k
}{
\max(K_t,1)
}
}
\]

or equivalently

\[
\bar F_t
=
\frac{F_t}{\max(K_t,1)}.
\]

This quantity asks:

> Among the observed future response tokens, how much negative visual conflict is accumulated per token on average?

For the last valid response token,

\[
K_t=0,
\]

so the convention gives

\[
\bar F_t=0.
\]

---

## 7. Position-bias example

Suppose two positions have the same mean future conflict:

\[
0.1.
\]

### Early position

\[
F_t=16,
\qquad
K_t=160.
\]

Then

\[
\bar F_t
=
\frac{16}{160}
=
0.1.
\]

### Late position

\[
F_t=4,
\qquad
K_t=40.
\]

Then

\[
\bar F_t
=
\frac4{40}
=
0.1.
\]

Thus

\[
\boxed{
\bar F_{\text{early}}
=
\bar F_{\text{late}}
}
\]

when the continuation has the same average conflict persistence.

This removes the trivial preference for earlier positions that appears under the raw future sum.

---

## 8. Bounded future persistence score

Mean future conflict can still vary in scale, so map it to

\[
\boxed{
r_t
=
\frac{
\bar F_t
}{
1+\bar F_t
}
}
\]

with

\[
0\le r_t<1.
\]

This quantity is called the **future conflict persistence score**.

Its interpretation is:

- \(r_t\approx0\): the observed continuation after \(t\) contains little negative visual conflict;
- larger \(r_t\): visual conflict remains denser in the continuation;
- \(r_t\) is bounded and cannot grow without limit.

No future threshold, discount factor, or window size is required.

---

## 9. Worked spatial-reasoning example

Assume the current position has 60 observed future tokens.

### Case A: persistent conflict

Suppose the average future conflict is

\[
0.15.
\]

Then

\[
F_t=60\times0.15=9.
\]

But the method uses

\[
\bar F_t
=
\frac9{60}
=
0.15.
\]

Therefore

\[
r_t
=
\frac{0.15}{1.15}
\approx0.130.
\]

### Case B: recovery

Suppose the average future conflict is only

\[
0.02.
\]

Then

\[
\bar F_t=0.02,
\]

and

\[
r_t
=
\frac{0.02}{1.02}
\approx0.0196.
\]

Thus the first state receives a substantially larger persistence score because its suffix continues to exhibit visual conflict more densely.

---

## 10. Why future does not modify token direction

Future statistics are computed from the actual sampled continuation.

They do not define a vocabulary-level visual preference over all candidate tokens at the current state.

Therefore future information should not redefine

\[
n_t(v)
=
[-u_t(v)]_+.
\]

Current real-vs-null visual evidence remains the only source used to decide which candidate tokens are visually opposed.

---

## 11. Why future does not modify \(\beta_t\) in the first version

Section 04 already defines

\[
\beta_t
=
\beta(1+s_t),
\]

where \(s_t\) is derived from historical accumulated conflict.

If future persistence also changes \(\beta_t\), both past and future information alter target geometry, making the two contributions difficult to separate experimentally.

The first version therefore uses:

- history \(\rightarrow\) target reconstruction strength,
- future \(\rightarrow\) position importance.

This keeps the interfaces distinct and reduces the risk of overly strong reconstruction.

---

## 12. Position-wise future weight

Define

\[
\boxed{
w_t
=
1+\alpha_F r_t
}
\]

where \(\alpha_F\ge0\) controls the maximum additional future emphasis.

For the first zero-extra-hyperparameter variant, fix

\[
\boxed{
\alpha_F=1.
}
\]

Then

\[
\boxed{
1\le w_t<2.
}
\]

This means future information can only **increase** the importance of a position relative to ordinary OPD/JSD supervision.

It never removes base supervision.

---

## 13. Why the base weight must be 1

A formulation such as

\[
w_t=r_t
\]

would give

\[
w_t\approx0
\]

when the continuation contains little conflict.

That would nearly remove OPD/JSD supervision from states followed by recovery.

The proposed method instead uses

\[
w_t=1+r_t,
\]

so that:

- ordinary supervision is always retained;
- persistent downstream conflict adds emphasis;
- future information acts as prioritization rather than gating.

---

## 14. Weighted JSD objective

The per-position distillation loss remains

\[
\ell_t
=
\operatorname{JSD}
\left(
q_t,
p_t^S
\right).
\]

The response-level objective becomes

\[
\boxed{
L
=
\frac{
\sum_{b,t}
M_{b,t}
w_{b,t}
\ell_{b,t}
}{
\sum_{b,t}
M_{b,t}
w_{b,t}
}
}
\]

where \(M_{b,t}\) is the valid-response mask.

The denominator must use the same weights.

Otherwise batches with more persistent conflict would automatically have larger overall loss magnitude, unintentionally changing the effective learning rate.

---

## 15. Weighted-loss example

Suppose three valid positions have

| position | JSD \(\ell_t\) | future weight \(w_t\) |
|---|---:|---:|
| \(t_1\) | 0.05 | 1.0 |
| \(t_2\) | 0.10 | 1.5 |
| \(t_3\) | 0.20 | 1.8 |

The weighted loss is

\[
L
=
\frac{
1(0.05)+1.5(0.10)+1.8(0.20)
}{
1+1.5+1.8
}
\]

\[
=
\frac{0.56}{4.3}
\approx0.130.
\]

Uniform averaging would give

\[
\frac{0.05+0.10+0.20}{3}
\approx0.117.
\]

The future weighting therefore increases the influence of positions whose continuation remains more visually conflicted without changing the loss family.

---

## 16. Efficient implementation with reverse exclusive cumulative sums

The future statistic does not require an \(O(T^2)\) loop.

Assume

~~~text
c:             [B, T]
response_mask: [B, T]
~~~

Define a reverse exclusive cumulative sum:

~~~python
def exclusive_reverse_cumsum(x):
    rev = torch.flip(x, dims=[1])
    rev_inclusive = torch.cumsum(rev, dim=1)
    inclusive = torch.flip(rev_inclusive, dims=[1])
    return inclusive - x
~~~

Then:

~~~python
future_sum = exclusive_reverse_cumsum(c)
future_len = exclusive_reverse_cumsum(response_mask)
~~~

which correspond to

\[
F_t
=
\sum_{k>t}c_k
\]

and

\[
K_t
=
\sum_{k>t}M_k.
\]

Compute:

~~~python
future_mean = future_sum / torch.clamp(future_len, min=1.0)
r_future = future_mean / (1.0 + future_mean)
w_future = 1.0 + alpha_F * r_future
~~~

with

~~~python
alpha_F = 1.0
~~~

for the first version.

---

## 17. Natural EOS

If the response terminates naturally at token \(T\), then for the final token:

\[
F_T=0,
\qquad
K_T=0,
\]

so

\[
\bar F_T=0,
\]

\[
r_T=0,
\]

and

\[
\boxed{
w_T=1.
}
\]

This is the correct boundary condition: no observed future implies ordinary base JSD supervision at the final position.

For earlier positions, the entire observed suffix up to natural termination contributes normally.

---

## 18. Maximum-length truncation

A response that reaches the maximum generation length does not have a naturally completed suffix.

For example:

~~~text
This is red. This is red. This is red. ...
~~~

may stop only because of the generation cap.

In this case, absence of tokens beyond the cap must **not** be interpreted as recovery.

The conservative first-version rule is:

1. use all observed future tokens before the truncation cap;
2. compute \(\bar F_t\) only from that observed suffix;
3. record that the rollout is censored through the finish-reason field;
4. do not invent unobserved future conflict or an extra truncation penalty.

Thus the measured future persistence for a truncated response should be interpreted as an **observed-suffix statistic**, not a complete trajectory estimate.

Because

\[
w_t\ge1,
\]

missing future can only reduce the additional emphasis relative to what a longer observed suffix might have produced. It cannot reduce supervision below the base OPD/JSD level.

---

## 19. Required finish-reason logging

At minimum store:

~~~text
finish_reason:
    eos
    max_length
    invalid
~~~

Analysis should distinguish:

~~~text
low future conflict + natural EOS
~~~

from

~~~text
low observed future conflict + max-length truncation
~~~

because the second case is censored and must not be described as confirmed recovery.

---

## 20. Gradient semantics

The future persistence score is a training-time trajectory statistic.

It should not receive gradient.

Conceptually:

~~~python
with torch.no_grad():
    c = relu(-u_realized) * response_mask

    future_sum = exclusive_reverse_cumsum(c)
    future_len = exclusive_reverse_cumsum(response_mask)

    future_mean = future_sum / torch.clamp(future_len, min=1.0)
    r_future = future_mean / (1.0 + future_mean)
    w_future = 1.0 + r_future
~~~

Then:

~~~python
jsd_per_token = JSD(q_target.detach(), p_student)

loss = (
    response_mask * w_future * jsd_per_token
).sum() / (
    response_mask * w_future
).sum()
~~~

Only the student distribution receives gradient through the JSD term.

---

## 21. Full current / past / future decomposition

The three modules now form:

### Current visual direction

\[
u_t(v)
=
\log p_t^+(v)-\log p_t^0(v),
\]

\[
\boxed{
n_t(v)
=
[-u_t(v)]_+.
}
\]

This decides **which candidate tokens are suppressed**.

### Historical accumulated deviation

\[
c_t=n_t(y_t),
\]

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

This decides **how strongly the current negative target is reconstructed**.

### Future conflict persistence

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

This decides **how much the current position contributes to the final JSD objective**.

---

## 22. Final pure-OPD objective

The reconstructed target is

\[
\boxed{
q_t(v)
\propto
p_t^+(v)
\exp
\left[
-\beta_t
[-u_t(v)]_+
\right].
}
\]

The final weighted OPD objective is

\[
\boxed{
L
=
\frac{
\sum_t
M_t
w_t
\operatorname{JSD}
\left(
q_t,
p_t^S
\right)
}{
\sum_t
M_t
w_t
}.
}
\]

No PPO probability ratio, reward model, or policy-gradient term is introduced.

---

## 23. Interpretation and causal limitation

The future statistic does **not** prove that the current sampled token caused downstream conflict.

No intervention is performed at position \(t\).

The appropriate interpretation is:

> Future persistence identifies states on sampled trajectories that are followed by sustained visual conflict and increases the importance of visual-corrective distillation at those states.

Thus the mechanism is best described as:

\[
\boxed{
\text{retrospective supervision prioritization}
}
\]

rather than

\[
\text{causal action credit assignment}.
\]

---

## 24. Main conclusion

The first future module uses

\[
\boxed{
F_t
=
\sum_{k>t}c_k
}
\]

only as an intermediate sum, then removes remaining-horizon bias through

\[
\boxed{
\bar F_t
=
\frac{F_t}{\max(K_t,1)}.
}
\]

It then constructs

\[
\boxed{
r_t
=
\frac{\bar F_t}{1+\bar F_t}
}
\]

and

\[
\boxed{
w_t
=
1+r_t.
}
\]

The final division of responsibility is therefore:

\[
\boxed{
\text{current}
\rightarrow
\text{direction}
}
\]

\[
\boxed{
\text{past}
\rightarrow
\text{target strength}
}
\]

\[
\boxed{
\text{future}
\rightarrow
\text{loss importance}.
}
\]

This preserves a pure OPD/JSD training framework while adding trajectory-aware supervision without introducing an unnecessary long-horizon normalization designed for much longer reasoning traces.
