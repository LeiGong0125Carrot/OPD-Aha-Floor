"""DG -- disagreement-gated teacher-trajectory distillation (OPD-Aha-Floor teacher_stability/disagreement_gate_plan.md v2).

Gate (label-free, before generation): g = TV(p_S0, p_T0) between the t=0 answer readouts of the two views, i.e. the
LM-head distribution at the last prompt position restricted to the question's option letters and renormalised.
  p_T0: frozen base teacher (crop view) -> precomputed offline once (extra_info.dg_pT0).
  p_S0: current student (red-box full image + hint) -> top-k logprobs of the FIRST generated position, returned by the
        vLLM rollout that samples the student's response anyway (no extra forward).
Per step the top-frac prompts by g (DG) or a matched random set (DR) get ONE of their n rollouts replaced by an offline
frozen-teacher trajectory (teacher view, majority answer, samples 4..7; extra_info.dg_traj); the loss is the usual JSD.

Pure helpers here (CPU-testable); the trainer glue is RayPPOTrainer._maybe_apply_dg.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

LETTERS = "ABCDEF"
# crop-view wording only (the red box itself is named in BOTH views' text, so it is not a leak here); monitored, never masked
LEAK_RE = re.compile(r"zoom|close-up|closeup|crop|enlarg|magnif|inset|second image", re.I)


def opts_mask(opts) -> np.ndarray:
    """[6] bool over LETTERS for the option letters of the question ('ABCD' or ['A','B',...])."""
    m = np.zeros(len(LETTERS), dtype=bool)
    for o in opts:
        if o not in LETTERS:
            raise ValueError(f"option letter {o!r} not in {LETTERS}")
        m[LETTERS.index(o)] = True
    if not m.any():
        raise ValueError("question has no option letters")
    return m


def letter_dist_from_topk(topk: Optional[dict], letter_ids: list[int], opts) -> tuple[np.ndarray, dict]:
    """Option-letter distribution [6] from a {token_id: logprob} top-k of the first generated position, renormalised over
    the question's options. Options missing from the top-k get probability 0 (their true mass is below the k-th token);
    if NO option is in the top-k the readout is undefined -> uniform over the options (flagged in the info dict)."""
    m = opts_mask(opts)
    lp = np.full(len(LETTERS), -np.inf)
    if topk:
        for j, t in enumerate(letter_ids):
            if m[j] and t in topk:
                lp[j] = float(topk[t])
    found = np.isfinite(lp)
    info = {"covered": float(found[m].mean()), "no_letter": float(not found.any()),
            "letter_mass": float(np.exp(lp[found]).sum()) if found.any() else 0.0}
    if not found.any():
        return np.where(m, 1.0 / m.sum(), 0.0), info
    v = lp - lp[found].max()
    p = np.where(found, np.exp(v), 0.0)
    return p / p.sum(), info


def letter_dist_from_logp(letter_logp, opts) -> np.ndarray:
    """Offline readout (full-vocab log_softmax at the letter tokens, [6]) -> renormalised over the options (as probe_disagree.t0_dist)."""
    m = opts_mask(opts)
    v = np.where(m, np.asarray(letter_logp, dtype=np.float64), -np.inf)
    v = v - np.logaddexp.reduce(v[m])
    return np.where(m, np.exp(v), 0.0)


def tv(p, q) -> float:
    return float(0.5 * np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum())


def n_select(n_prompts: int, frac: float) -> int:
    """Prompts to replace per step: round(frac * prompts in the batch), at least 1 when frac > 0."""
    if not (0.0 <= frac <= 1.0):
        raise ValueError(f"dg_frac must be in [0,1], got {frac}")
    return 0 if frac == 0 else max(1, int(round(frac * n_prompts)))


def select_prompts(g: np.ndarray, eligible: np.ndarray, frac: float, mode: str, seed: int, step: int) -> np.ndarray:
    """Indices (into the prompt list) to replace. Both modes pick the SAME number m = min(n_select(N, frac), #eligible):
    gate   -> the m eligible prompts with the largest g (ties in a fixed random order);
    random -> m eligible prompts uniformly at random (g is not looked at). rng depends on (seed, step) only."""
    g = np.asarray(g, dtype=np.float64); eligible = np.asarray(eligible, dtype=bool)
    if g.shape != eligible.shape:
        raise ValueError("g / eligible shape mismatch")
    el = np.flatnonzero(eligible)
    m = min(n_select(len(g), frac), len(el))
    if m == 0:
        return np.zeros(0, dtype=int)
    rng = np.random.default_rng([int(seed), int(step), 7])
    if mode == "gate":
        tie = rng.permutation(len(el))
        order = np.lexsort((tie, -g[el]))          # primary: g descending; secondary: fixed random tie-break
        return np.sort(el[order[:m]])
    if mode == "random":
        return np.sort(rng.choice(el, size=m, replace=False))
    raise ValueError(f"dg_mode must be gate|random, got {mode!r}")


def majority(letters) -> Optional[str]:
    """Unique most frequent parsed letter (None on a tie for first or when nothing parsed); = probe_hidden_answer.majority."""
    import collections
    c = collections.Counter(x for x in letters if x is not None).most_common()
    if not c or (len(c) > 1 and c[0][1] == c[1][1]):
        return None
    return c[0][0]


def choose_teacher_traj(letters, texts, row: int, seed: int = 42, pool=(4, 5, 6, 7)) -> Optional[int]:
    """Sample index of the offline teacher trajectory for one question: uniformly among the samples in `pool` whose parsed
    answer equals the teacher's 8-sample majority letter (teacher self-consistency, no GT). None when there is no unique
    majority or no pool sample carries it."""
    maj = majority(letters)
    if maj is None:
        return None
    cand = [s for s in pool if s < len(letters) and letters[s] == maj and texts[s] and texts[s].strip()]
    if not cand:
        return None
    return int(np.random.default_rng([int(seed), int(row), 11]).choice(cand))


def first_topk_from_vllm(position_logprobs) -> dict:
    """vLLM per-position logprobs entry (mapping token_id -> Logprob) -> {int token_id: float logprob}."""
    return {int(t): float(getattr(v, "logprob", v)) for t, v in position_logprobs.items()}
