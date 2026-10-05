# Copyright 2026 Vision-OPD / OPD-Aha-Floor authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Route 1 -- internal residual reconstruction from a SINGLE frozen-teacher forward.

Design: docs/negative_history/teacher_internal_residual_reconstruction_plan.md (v0.1).

For a contiguous block interval [a, b) of the frozen privileged teacher we read, from the same
forward that produces the anchor logits z+, the interval's actual contribution to the final
logits through the model's own read-out path:

    r_t = tau^-1 * W_out * D_t * (h_t^b - h_t^a),      D_t = diag((1 + w_norm) * s_t),
    s_t = (mean_i (h_t^L)_i^2 + eps)^-1/2               (Qwen3.5 zero-centered RMSNorm)

and the target is  q_t ∝ p_t^+ * exp(lambda * r_t)  (no null forward, no student in the target).

Everything here is read-only: hooks return None, never mutate activations, and are removed on
exit (also on exceptions). Residual states are only kept at the response positions that predict
y_t (P + t - 1 in the packed layout), selected by a caller-supplied boolean mask.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["find_text_model_and_head", "ResidualCapture", "build_internal_targets"]


def _unwrap(m: nn.Module) -> nn.Module:
    # FSDP1 wraps as .module / ._fsdp_wrapped_module; FSDP2 leaves the module in place.
    for attr in ("_fsdp_wrapped_module", "module"):
        inner = getattr(m, attr, None)
        if isinstance(inner, nn.Module) and inner is not m:
            return _unwrap(inner)
    return m


def find_text_model_and_head(model: nn.Module) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Return (text_model, final_norm, lm_head) for a (possibly FSDP-wrapped) Qwen3.5-style VLM.

    text_model must expose `.layers` (ModuleList of decoder blocks) and `.norm` (final RMSNorm);
    lm_head is the root model's `lm_head`. Raises if the structure is not recognised so that we
    never silently hook the wrong module.
    """
    root = _unwrap(model)
    lm_head = getattr(root, "lm_head", None)
    if lm_head is None:
        raise ValueError("internal-residual capture: root model has no `lm_head`")
    text_model = None
    for name, m in root.named_modules():
        mm = _unwrap(m)
        if hasattr(mm, "layers") and isinstance(getattr(mm, "layers"), nn.ModuleList) and hasattr(mm, "norm"):
            # the language model (skip the vision tower, whose blocks are named differently)
            if name.endswith("language_model") or name.endswith("model") or text_model is None:
                text_model = mm
                if name.endswith("language_model"):
                    break
    if text_model is None:
        raise ValueError("internal-residual capture: could not locate a text model with `.layers` and `.norm`")
    norm = _unwrap(text_model.norm)
    if not hasattr(norm, "weight"):
        raise ValueError("internal-residual capture: final norm has no `.weight`")
    return text_model, norm, _unwrap(lm_head)


class ResidualCapture:
    """Context manager that captures h^a, h^b (residual stream at block boundaries), the final
    pre-norm state and computes r at the masked (response) positions inside the lm_head hook,
    where FSDP has the head weight materialised.

    `response_mask_packed` is a bool tensor over the packed token axis (rmpad layout, batch dim 1);
    only those rows are kept.
    """

    def __init__(
        self,
        model: nn.Module,
        start_block: int,
        end_block_exclusive: int,
        temperature: float,
        response_mask_packed: Optional[torch.Tensor] = None,
        vocab_chunk: int = 32768,
        max_positions: int = 4096,
    ) -> None:
        self.vocab_chunk = int(vocab_chunk)
        self.max_positions = int(max_positions)
        self.text_model, self.norm, self.lm_head = find_text_model_and_head(model)
        n_layers = len(self.text_model.layers)
        a, b = int(start_block), int(end_block_exclusive)
        if not (0 <= a < b <= n_layers):
            raise ValueError(f"internal-residual: need 0 <= a < b <= L={n_layers}, got [a,b)=[{a},{b})")
        if not (temperature > 0):
            raise ValueError(f"internal-residual: temperature must be > 0, got {temperature}")
        self.a, self.b, self.tau = a, b, float(temperature)
        self.mask = response_mask_packed
        self._handles: list = []
        self.reset()

    # ---- state ----
    def reset(self) -> None:
        self.h_a: Optional[torch.Tensor] = None
        self.h_b: Optional[torch.Tensor] = None
        self.h_last: Optional[torch.Tensor] = None
        self.r: Optional[torch.Tensor] = None          # [n_resp, V] fp32, already / tau
        self.calls = {"a": 0, "b": 0, "norm": 0, "head": 0}

    def set_response_mask(self, mask: torch.Tensor) -> None:
        self.mask = mask

    # ---- helpers ----
    @staticmethod
    def _as_hidden(x) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x):
            raise TypeError(f"internal-residual: expected a tensor hidden state, got {type(x)}")
        return x

    def _select(self, h: torch.Tensor) -> torch.Tensor:
        # h: [1, N, d] (rmpad) or [B, S, d]; flatten the token axes, apply the packed mask
        flat = h.reshape(-1, h.shape[-1])
        if self.mask is None:
            return flat.detach().float()
        if self.mask.numel() != flat.shape[0]:
            raise ValueError(
                f"internal-residual: response mask has {self.mask.numel()} entries but the hidden state "
                f"has {flat.shape[0]} tokens (packed layout mismatch)"
            )
        return flat[self.mask.to(flat.device)].detach().float()

    # ---- hooks (read-only, return None) ----
    def _pre_hook_a(self, module, args, kwargs=None):
        self.calls["a"] += 1
        self.h_a = self._select(self._as_hidden(args[0] if args else kwargs["hidden_states"]))

    def _hook_b(self, module, args, output):
        self.calls["b"] += 1
        self.h_b = self._select(self._as_hidden(output))

    def _hook_norm(self, module, args, output):
        self.calls["norm"] += 1
        self.h_last = self._select(self._as_hidden(args[0]))

    def _hook_head(self, module, args, output):
        self.calls["head"] += 1
        if self.h_a is None or self.h_b is None or self.h_last is None:
            raise RuntimeError("internal-residual: lm_head reached before block/norm hooks fired")
        # D_t = (1 + w) * s_t  -- Qwen3.5 zero-centered RMSNorm, computed in fp32 like the model does
        eps = float(getattr(self.norm, "eps", getattr(self.norm, "variance_epsilon", 1e-6)))
        s = torch.rsqrt(self.h_last.pow(2).mean(-1, keepdim=True) + eps)            # [n, 1]
        w_eff = 1.0 + self.norm.weight.detach().float()                            # [d]
        dh = (self.h_b - self.h_a) * s * w_eff                                     # [n, d] fp32
        # NOTE: an output bias (absent in Qwen3.5) belongs to the anchor only and never enters r.
        n = dh.shape[0]
        if n > self.max_positions:
            raise ValueError(
                f"internal-residual: {n} response positions in one micro-batch exceeds max_positions="
                f"{self.max_positions} (r is [n, V] fp32); lower the micro-batch or raise the bound explicitly"
            )
        weight = module.weight.detach()                                            # [V, d], materialised by FSDP for this forward
        V = weight.shape[0]
        r = torch.empty(n, V, dtype=torch.float32, device=dh.device)
        for i in range(0, V, self.vocab_chunk):                                    # chunked fp32 head: ~335 MB transient instead of 2.5 GB
            r[:, i:i + self.vocab_chunk] = F.linear(dh, weight[i:i + self.vocab_chunk].float())
        r.div_(self.tau)
        if not torch.isfinite(r).all():
            raise FloatingPointError("internal-residual: non-finite r")
        self.r = r

    # ---- context ----
    def __enter__(self):
        layers = self.text_model.layers
        self._handles = [
            layers[self.a].register_forward_pre_hook(self._pre_hook_a, with_kwargs=True),
            layers[self.b - 1].register_forward_hook(self._hook_b),
            self.norm.register_forward_hook(self._hook_norm),
            self.lm_head.register_forward_hook(self._hook_head),
        ]
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles = []
        # drop the big intermediates but keep r for the caller
        self.h_a = self.h_b = self.h_last = None
        return False

    def check_single_forward(self) -> None:
        if any(v != 1 for v in self.calls.values()):
            raise RuntimeError(f"internal-residual: hooks must fire exactly once per forward, got {self.calls}")


def build_internal_targets(
    logits_resp: torch.Tensor,
    r_resp: torch.Tensor,
    lam: float,
    topk_idx_resp: torch.Tensor,
    tail_policy: str,
) -> dict[str, torch.Tensor]:
    """Build the internal-residual target on the student's top-k + tail support.

    logits_resp: [n, V] anchor logits ALREADY divided by the scoring temperature (fp32).
    r_resp:      [n, V] contribution read-out, already / tau (fp32).
    topk_idx_resp: [n, K] student top-k token ids.
    tail_policy:
      "full": q^V = softmax(z/tau + lam*r) over the FULL vocabulary, then coarsen to K + tail
              (plan §6.1). Returns `target_log_probs` [n, K+1] (log q_K ..., log q_tail).
      "aha":  only r at the explicit top-k is returned (`r_topk` [n, K]); the target is formed
              in the loss by log_softmax(add_tail(log p+_K) + lam*[r_K, 0]) -- tail r := 0,
              i.e. the OPD-Aha coarsen-then-reconstruct order (declared variant, plan §6.2).
    `r_topk` is returned in both modes for diagnostics.
    """
    if tail_policy not in ("full", "aha"):
        raise ValueError(f"internal-residual: tail_policy must be 'full' or 'aha', got {tail_policy!r}")
    if not (math.isfinite(lam) and lam >= 0.0):
        raise ValueError(f"internal-residual: strength (lambda) must be finite and >= 0, got {lam}")
    if logits_resp.shape != r_resp.shape:
        raise ValueError(f"internal-residual: logits {tuple(logits_resp.shape)} vs r {tuple(r_resp.shape)}")
    r_topk = torch.gather(r_resp, dim=-1, index=topk_idx_resp).float()
    out = {"r_topk": r_topk}
    if tail_policy == "aha":
        return out
    # one [n, V] fp32 working tensor: s = z/tau + lam * r  (in place on a copy of z)
    s = logits_resp.float().clone()
    s.add_(r_resp, alpha=lam)
    lse_all = torch.logsumexp(s, dim=-1, keepdim=True)                       # [n,1]
    s_k = torch.gather(s, dim=-1, index=topk_idx_resp)                       # [n,K]
    del s
    log_q_k = s_k - lse_all
    lse_k = torch.logsumexp(s_k, dim=-1, keepdim=True)
    # log q_tail = log(1 - exp(lse_k - lse_all)) computed stably; clamp the argument so that a
    # K == V support (tail exactly empty) gives a finite, very negative tail instead of -inf.
    gap = (lse_k - lse_all).clamp(max=-1e-7)
    log_q_tail = torch.log(-torch.expm1(gap))
    target = torch.cat([log_q_k, log_q_tail], dim=-1)
    if not torch.isfinite(target).all():
        raise FloatingPointError("internal-residual: non-finite target")
    out["target_log_probs"] = target
    return out
