# Copyright 2026 Vision-OPD / OPD-Aha-Floor authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Layer-prior probe: which internal read-out of ONE frozen privileged-teacher forward approximates
the visual-null distribution p^0, and does the resulting reconstructed target approximate Q_A?

Spec: docs/negative_history/ir_layer_prior_probe_plan_2026-10-05.md (v0.2, `05c3eab`).

Everything here is pure torch and CPU-testable. GPU / model loading lives in scripts/probe_layer_prior.py.

Notation (plan §2.3-§2.6):
  z+           anchor logits of the real forward, already / tau            [n, V]
  h^l          residual stream at block boundary l (l = 0..L), l = L is the final pre-norm state
  D_final      diag((1 + w_norm) * s_t),  s_t = (mean_d (h^L)^2 + eps)^-1/2          (zero-centred RMSNorm)
  D_own(l)     same with the RMS of h^l instead of h^L                                 (candidate B)
  p_l^final    softmax(tau^-1 W D_final h^l)                                           (candidate A)
  r_j          tau^-1 W D_final (h^{j+1} - h^j)                                         (block contribution, candidate C)
  C_K(p)       coarsen a full-vocab distribution to the student's top-K + one tail bucket
  Q_A          softmax(log P+ + beta * (log P+ - log P0))  on K+1 (aggregate first, then reconstruct = original A)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.teacher_residual import find_text_model_and_head

__all__ = [
    "BoundaryCollector",
    "final_scale_vec",
    "own_scale_vec",
    "readout_logits",
    "coarsen_log_probs",
    "coarsen_logits",
    "aha_target",
    "tv_from_log",
    "centered",
    "RegressionAccumulator",
    "corr_pieces",
    "find_answer_token_positions",
    "weighted_stats",
    "paired_bootstrap",
]


# ----------------------------------------------------------------------------------------------
# 1. read-only multi-boundary collector
# ----------------------------------------------------------------------------------------------
class BoundaryCollector:
    """Capture the residual stream at EVERY block boundary h^0 .. h^L of the text model, restricted to
    the positions selected by a boolean mask over the flat token axis, from a single forward.

    h^l for l < L is the input of block l (forward pre-hook); h^L is the input of the final norm.
    Hooks return None (read-only) and are removed on exit, also on exceptions. The lm_head is NOT
    hooked: the anchor logits must come from the model's own output (plan §2.4).
    """

    def __init__(self, model: nn.Module, mask_flat: Optional[torch.Tensor] = None) -> None:
        self.text_model, self.norm, self.lm_head = find_text_model_and_head(model)
        self.L = len(self.text_model.layers)
        self.mask = mask_flat
        self._handles: list = []
        self.reset()

    def reset(self) -> None:
        self.h: list[Optional[torch.Tensor]] = [None] * (self.L + 1)
        self.calls = [0] * (self.L + 1)

    def set_mask(self, mask_flat: torch.Tensor) -> None:
        self.mask = mask_flat

    @staticmethod
    def _as_hidden(x) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x):
            raise TypeError(f"layer-prior probe: expected a tensor hidden state, got {type(x)}")
        return x

    def _select(self, hid: torch.Tensor) -> torch.Tensor:
        flat = hid.reshape(-1, hid.shape[-1])
        if self.mask is None:
            return flat.detach().float()
        if self.mask.numel() != flat.shape[0]:
            raise ValueError(
                f"layer-prior probe: mask has {self.mask.numel()} entries but the hidden state has "
                f"{flat.shape[0]} tokens"
            )
        return flat[self.mask.to(flat.device)].detach().float()

    def _make_pre_hook(self, l: int):
        def hook(module, args, kwargs=None):
            self.calls[l] += 1
            self.h[l] = self._select(self._as_hidden(args[0] if args else kwargs["hidden_states"]))
        return hook

    def _norm_pre_hook(self, module, args, kwargs=None):
        self.calls[self.L] += 1
        self.h[self.L] = self._select(self._as_hidden(args[0] if args else kwargs["hidden_states"]))

    def __enter__(self):
        layers = self.text_model.layers
        self._handles = [layers[l].register_forward_pre_hook(self._make_pre_hook(l), with_kwargs=True) for l in range(self.L)]
        self._handles.append(self.norm.register_forward_pre_hook(self._norm_pre_hook, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc, tb):
        for hd in self._handles:
            hd.remove()
        self._handles = []
        return False

    def check_single_forward(self) -> None:
        if any(c != 1 for c in self.calls):
            raise RuntimeError(f"layer-prior probe: every boundary hook must fire exactly once, got {self.calls}")
        if any(x is None for x in self.h):
            raise RuntimeError("layer-prior probe: a boundary state is missing")

    def stacked(self) -> torch.Tensor:
        """[L+1, n, d] fp32 (h^0 .. h^L)."""
        self.check_single_forward()
        return torch.stack(self.h, dim=0)

    @property
    def norm_eps(self) -> float:
        return float(getattr(self.norm, "eps", getattr(self.norm, "variance_epsilon", 1e-6)))

    @property
    def norm_weight(self) -> torch.Tensor:
        return self.norm.weight.detach().float()

    @property
    def head_weight(self) -> torch.Tensor:
        return self.lm_head.weight.detach()

    @property
    def head_bias(self) -> Optional[torch.Tensor]:
        b = getattr(self.lm_head, "bias", None)
        return None if b is None else b.detach().float()


# ----------------------------------------------------------------------------------------------
# 2. read-out math
# ----------------------------------------------------------------------------------------------
def final_scale_vec(h_last: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    """D_final as a per-position vector [n, d]: (1 + w) * rsqrt(mean(h_L^2) + eps)."""
    s = torch.rsqrt(h_last.float().pow(2).mean(-1, keepdim=True) + eps)
    return s * (1.0 + norm_weight.float())


def own_scale_vec(h_l: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Candidate B: the same learned (1 + w) but the RMS of h^l itself."""
    return final_scale_vec(h_l, norm_weight, eps)


def readout_logits(
    x: torch.Tensor, head_weight: torch.Tensor, tau: float, bias: Optional[torch.Tensor] = None, vocab_chunk: int = 32768
) -> torch.Tensor:
    """tau^-1 (W x + b) for an already-scaled hidden x [n, d] -> [n, V] fp32, chunked over the vocab."""
    if not (tau > 0):
        raise ValueError(f"tau must be > 0, got {tau}")
    n, V = x.shape[0], head_weight.shape[0]
    out = torch.empty(n, V, dtype=torch.float32, device=x.device)
    xf = x.float()
    for i in range(0, V, vocab_chunk):
        out[:, i : i + vocab_chunk] = F.linear(xf, head_weight[i : i + vocab_chunk].float())
    if bias is not None:
        out.add_(bias.float())
    out.div_(tau)
    return out


# ----------------------------------------------------------------------------------------------
# 3. coarsening to the shared support K + tail, A target, distances
# ----------------------------------------------------------------------------------------------
def _tail_from_topk_log(log_k: torch.Tensor) -> torch.Tensor:
    # identical to core_algos.add_tail: log(1 - sum p_K) via clamp(-1e-7) + log(-expm1)
    log_s = torch.logsumexp(log_k, dim=-1, keepdim=True).clamp(max=-1e-7)
    return torch.log(-torch.expm1(log_s))


def coarsen_log_probs(log_p_full: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    """Full-vocab log-probs [n, V] (already normalised over V) -> [n, K+1] = (log p_K ..., log p_tail)."""
    log_k = torch.gather(log_p_full, dim=-1, index=topk_idx)
    return torch.cat([log_k, _tail_from_topk_log(log_k)], dim=-1)


def coarsen_logits(logits_full: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    """Full-vocab logits [n, V] -> normalise over V -> coarsen. Never renormalises inside K."""
    return coarsen_log_probs(torch.log_softmax(logits_full.float(), dim=-1), topk_idx)


def aha_target(log_p_plus: torch.Tensor, log_p_ref: torch.Tensor, beta: float) -> torch.Tensor:
    """Original A order: both inputs are ALREADY coarsened [n, K+1]; Q = softmax(log P+ + beta (log P+ - log Pref)).
    The tail column keeps its real log-ratio (plan §2.3)."""
    if not math.isfinite(beta):
        raise ValueError(f"beta must be finite, got {beta}")
    u = log_p_plus - log_p_ref
    if not torch.isfinite(u).all():
        raise FloatingPointError("aha_target: non-finite log-ratio")
    return torch.log_softmax(log_p_plus + beta * u, dim=-1)


def tv_from_log(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Total variation per position over the last axis: 0.5 * sum |p - q| -> [n]."""
    return 0.5 * (log_p.exp() - log_q.exp()).abs().sum(-1)


# ----------------------------------------------------------------------------------------------
# 4. omega-centred regression and correlation (plan §3.2-§3.3)
# ----------------------------------------------------------------------------------------------
def omega_from_log_p_plus(log_p_plus_k: torch.Tensor) -> torch.Tensor:
    """omega_{t,v} = p+(v) / sum_{k in K} p+(k) on the explicit top-K columns [n, K]."""
    p = log_p_plus_k.float().exp()
    return p / p.sum(-1, keepdim=True).clamp(min=1e-30)


def centered(f: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Z_t(f) = f - sum_k omega_k f_k, broadcast over trailing dims: f [n, K] or [n, K, L], omega [n, K]."""
    if f.dim() == 2:
        return f - (omega * f).sum(-1, keepdim=True)
    if f.dim() == 3:
        return f - (omega.unsqueeze(-1) * f).sum(1, keepdim=True)
    raise ValueError(f"centered: unsupported shape {tuple(f.shape)}")


class RegressionAccumulator:
    """Sufficient statistics for the global block-weight fit  w = (H/N + eta I)^-1 (b/N):
    H = sum X^T Omega X, b = sum X^T Omega u, S_uu = sum u^T Omega u (all centred), fp64, L x L only.
    Also evaluates R^2_corr of a FIXED w on any split (fit-split in-sample value is derived from the stats)."""

    def __init__(self, L: int) -> None:
        self.L = L
        self.H = torch.zeros(L, L, dtype=torch.float64)
        self.b = torch.zeros(L, dtype=torch.float64)
        self.S_uu = 0.0
        self.N = 0

    def add(self, X: torch.Tensor, u: torch.Tensor, omega: torch.Tensor) -> None:
        """X [n, K, L] centred block contributions, u [n, K] centred target, omega [n, K]."""
        if X.shape[-1] != self.L:
            raise ValueError(f"RegressionAccumulator: expected L={self.L}, got {X.shape[-1]}")
        Xd, ud, od = X.double().cpu(), u.double().cpu(), omega.double().cpu()
        XO = Xd * od.unsqueeze(-1)                                   # [n, K, L]
        self.H += torch.einsum("nkl,nkm->lm", XO, Xd)
        self.b += torch.einsum("nkl,nk->l", XO, ud)
        self.S_uu += float((od * ud * ud).sum())
        self.N += int(X.shape[0])

    def solve(self, eta_rel: float = 1e-4) -> dict:
        if self.N == 0 or self.S_uu <= 0.0:
            return {"fittable": False, "reason": "no positions or zero target energy", "N": self.N}
        Hn, bn = self.H / self.N, self.b / self.N
        eta = eta_rel * float(torch.trace(Hn)) / self.L + 1e-12
        A = Hn + eta * torch.eye(self.L, dtype=torch.float64)
        w = torch.linalg.solve(A, bn)
        evals = torch.linalg.eigvalsh(Hn)
        rank = int((evals > evals.max().clamp(min=1e-300) * 1e-10).sum())
        cond = float(evals.max() / evals.min().clamp(min=1e-300))
        # in-sample R^2 from the sufficient statistics: residual = S_uu - 2 w.b + w^T H w
        resid = self.S_uu - 2.0 * float(w @ self.b) + float(w @ self.H @ w)
        return {
            "fittable": True,
            "w": w.tolist(),
            "eta": eta,
            "rank": rank,
            "cond": cond,
            "eig_min": float(evals.min()),
            "eig_max": float(evals.max()),
            "N": self.N,
            "r2_in_sample": 1.0 - resid / self.S_uu,
        }

    @staticmethod
    def r2_pieces(X: torch.Tensor, u: torch.Tensor, omega: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Per-position (residual energy, target energy) [n, 2] for a FIXED w; any subset's R^2 = 1 - sum/sum."""
        pred = torch.einsum("nkl,l->nk", X.double(), w.double())
        od, ud = omega.double(), u.double()
        return torch.stack([(od * (ud - pred) ** 2).sum(-1), (od * ud * ud).sum(-1)], dim=-1).float()

    @staticmethod
    def r2_fixed(X: torch.Tensor, u: torch.Tensor, omega: torch.Tensor, w: torch.Tensor) -> tuple[float, float]:
        """Return (residual energy, target energy) summed over one batch; caller sums over a split."""
        p = RegressionAccumulator.r2_pieces(X, u, omega, w).double().sum(0)
        return float(p[0]), float(p[1])


def corr_pieces(u_hat_k: torch.Tensor, u_k: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Per-position pieces of Corr_omega: [n, 3] = (<Z u_hat, Z u>_omega, ||Z u_hat||^2, ||Z u||^2)."""
    a, b = centered(u_hat_k.float(), omega), centered(u_k.float(), omega)
    return torch.stack([(omega * a * b).sum(-1), (omega * a * a).sum(-1), (omega * b * b).sum(-1)], dim=-1)


def pooled_corr(pieces: torch.Tensor) -> Optional[float]:
    """pieces [N, 3] -> pooled Corr_omega, or None when either energy is ~0 (undefined)."""
    s = pieces.double().sum(0)
    if s[1] <= 1e-12 or s[2] <= 1e-12:
        return None
    return float(s[0] / torch.sqrt(s[1] * s[2]))


# ----------------------------------------------------------------------------------------------
# 5. answer-position parsing (plan §3.5)
# ----------------------------------------------------------------------------------------------
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)
_LEADING_LETTER = re.compile(r"^\s*\(?([A-D])\)?(?=[\.\)\s:]|$)")
_LAST_LINE_LETTER = re.compile(
    r"^[\s\*>#\u2705]*(?:(?:correct|final)?\s*answer\s*(?:is)?\s*:?[\s\*]*)?\(?([A-D])\)?(?=[\.\):\*\s]|$)", re.I)
_SPECIAL = re.compile(r"<\|[a-z_]+\|>")


def find_answer_token_positions(response_ids: list[int], decode_prefix) -> dict:
    """Locate the final-answer tokens inside a response.

    Rule (pre-registered before the full run): (1) the content of the FIRST <answer>...</answer> pair, tags
    excluded; (2) otherwise, when the response STARTS with an option letter (optionally in parentheses /
    followed by '.', ')' or ':'), that letter; (3) otherwise, when the LAST non-empty line of the response
    (special tokens such as <|im_end|> removed) is a final-answer line, i.e. after optional markdown
    (`*`, `>`, `#`, check mark) and an optional "[Correct|Final] answer [is]:" prefix it starts with an
    option letter ("A. blue", "**B**", "Correct answer: **D. cable**", "Final Answer: **D**"), that
    letter; (4) otherwise missing. A letter inside the reasoning text is never taken as the answer.
    `decode_prefix(k)` must return the decoded string of response_ids[:k]. Token i covers the
    character span [len(decode(ids[:i])), len(decode(ids[:i+1]))).
    """
    T = len(response_ids)
    text = decode_prefix(T)
    m = _ANSWER_TAG.search(text)
    if m:
        c0, c1, src = m.start(1), m.end(1), "answer_tag"
        inner = text[c0:c1]
        lead = inner.strip()
        # strip whitespace inside the tag from the span
        c0 += len(inner) - len(inner.lstrip())
        c1 -= len(inner) - len(inner.rstrip())
        if not lead:
            return {"found": False, "source": "answer_tag_empty", "positions": []}
    else:
        m2 = _LEADING_LETTER.match(text)
        if m2:
            c0, c1, src = m2.start(1), m2.end(1), "leading_letter"
        else:
            body = _SPECIAL.sub(lambda mm: " " * len(mm.group(0)), text)      # keep char offsets, drop special tokens
            lines = [(mm.start(), mm.group(0)) for mm in re.finditer(r"[^\n]+", body) if mm.group(0).strip()]
            if not lines:
                return {"found": False, "source": "missing", "positions": []}
            off, last = lines[-1]
            m3 = _LAST_LINE_LETTER.match(last)
            if not m3:
                return {"found": False, "source": "missing", "positions": []}
            c0, c1, src = off + m3.start(1), off + m3.end(1), "last_line_letter"
    prefixes = [decode_prefix(k) for k in range(T + 1)]
    if any(not text.startswith(pfx) for pfx in prefixes):           # byte-level BPE can split a UTF-8 char: prefixes then drift
        return {"found": False, "source": "unmappable", "positions": [], "char_span": [c0, c1]}
    ends = [len(pfx) for pfx in prefixes]                           # ends[k] = chars covered by first k tokens
    positions = [i for i in range(T) if ends[i] < c1 and ends[i + 1] > c0]
    return {"found": bool(positions), "source": src, "positions": positions, "char_span": [c0, c1]}


# ----------------------------------------------------------------------------------------------
# 6. aggregation helpers (token-weighted, per-question, bootstrap over question groups)
# ----------------------------------------------------------------------------------------------
def weighted_stats(x: torch.Tensor, qid: torch.Tensor) -> dict:
    """x [N] per-position values, qid [N] question ids -> token mean/median/p90, per-question mean, counts."""
    if x.numel() == 0:
        return {"n_tokens": 0, "n_questions": 0, "token_mean": None, "token_median": None, "token_p90": None, "question_mean": None}
    xd = x.double()
    uq, inv = torch.unique(qid, return_inverse=True)
    per_q = torch.zeros(len(uq), dtype=torch.float64).index_add_(0, inv, xd) / torch.bincount(inv, minlength=len(uq)).double()
    return {
        "n_tokens": int(x.numel()),
        "n_questions": int(len(uq)),
        "token_mean": float(xd.mean()),
        "token_median": float(xd.median()),
        "token_p90": float(torch.quantile(xd, 0.9)) if x.numel() > 1 else float(xd[0]),
        "question_mean": float(per_q.mean()),
    }


def paired_bootstrap(diff: torch.Tensor, group: torch.Tensor, n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired difference per position, resampled over GROUPS (questions / images), token-weighted mean.
    Returns mean and a 95% interval; `significant` only when the interval excludes 0."""
    if diff.numel() == 0:
        return {"mean": None, "ci95": None, "significant": False, "n_groups": 0}
    ug, inv = torch.unique(group, return_inverse=True)
    G = len(ug)
    sums = torch.zeros(G, dtype=torch.float64).index_add_(0, inv, diff.double())
    cnts = torch.bincount(inv, minlength=G).double()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, G, (n_boot, G), generator=g)
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))
    return {"mean": float(sums.sum() / cnts.sum()), "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0), "n_groups": int(G)}


@dataclass
class Thresholds:
    m3_median_tv: float = 0.10
    m3_corr: float = 0.70
    m5_min_answer_tokens: int = 30      # below this the answer subset is "待补充", never a pass
