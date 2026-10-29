"""OPRD-Vanilla (On-Policy Representation Distillation, arXiv 2606.06021) ported to the VLM OPD setting (user 10-26).

Reference: the paper-exact `main` branch of https://github.com/<OPRD repo> (local clone refs_tsw/OPRD, `git show main:...`,
commit e1f7d22), verl/verl/utils/rep_distillation.py:
  * layers  (get_rep_distillation_hidden_state_indices): HF ``hidden_states`` index 0 = embeddings, 1..L = block outputs;
    "all" = 1..L, "last" = L, "even"/"odd" = parity subsets of the blocks.
  * positions (build_rep_distillation_position_mask): over the RESPONSE token positions (the hidden state AT y_t, i.e.
    ``hidden_states[:, -response_len:]`` in their right-aligned layout); "all" | "last" | "last_k" | "first_k" with a
    per-sample k = min(k, valid length).
  * loss (normalized_mse_loss on main = NO normalisation, then multi_layer_normalized_mse_loss): per token the mean over the
    hidden dim of (h_S - sg(h_T))^2, token-mean over the selected positions (agg_loss "token-mean" of the micro-batch), then
    the plain mean over layers. Rep-only by default (REP_DISTILLATION_ONLY=True, coef 1).
The VLM difference handled here: the teacher sees a different prompt (crop view) so its response starts at its own
``response_start_idx``; positions are always taken as start + t of each view.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

VALID_LAYERS = ("all", "last", "even", "odd")
VALID_POSITIONS = ("all", "last", "last_k", "first_k", "first_frac")


def layer_indices(num_hidden_states: int, layers: str) -> list[int]:
    """= OPRD get_rep_distillation_hidden_state_indices (0 = embeddings is never used)."""
    if layers not in VALID_LAYERS:
        raise ValueError(f"oprd_layers must be one of {VALID_LAYERS}, got {layers!r}")
    if num_hidden_states < 2:
        raise ValueError(f"expected embeddings + >=1 block in hidden_states, got {num_hidden_states}")
    L = num_hidden_states - 1
    if layers == "last":
        return [L]
    if layers == "all":
        return list(range(1, L + 1))
    if layers == "even":
        return [1 + i for i in range(0, L, 2)]
    return [1 + i for i in range(1, L, 2)]


def position_mask(response_mask: torch.Tensor, positions: str, last_k: int = 2000, first_k: int = 2000,
                  frac: float = 0.25) -> torch.Tensor:
    """(B, R) float mask of the supervised response positions (= OPRD build_rep_distillation_position_mask; response
    tokens are left-aligned inside the response block, valid length = response_mask.sum).
    "first_frac" (our token-level ablation, user 10-28; not in OPRD): the first ceil(frac * n) response tokens of each
    sample (n = its valid length), i.e. probe_oprd_segments' first25 segment without the pre-generation state t=0."""
    if positions not in VALID_POSITIONS:
        raise ValueError(f"oprd_positions must be one of {VALID_POSITIONS}, got {positions!r}")
    rm = response_mask.float()
    if positions == "all":
        return rm
    B, R = rm.shape
    pos = torch.arange(R, device=rm.device).unsqueeze(0)
    n = rm.long().sum(1, keepdim=True)
    if positions == "last":
        return rm * (pos == (n - 1).clamp(min=0)).float()
    if positions == "first_k":
        k = torch.minimum(n, torch.full_like(n, int(first_k)))
        return rm * (pos < k).float()
    if positions == "first_frac":
        if not 0.0 < float(frac) <= 1.0:
            raise ValueError(f"oprd_frac must be in (0, 1], got {frac}")
        k = torch.ceil(n.double() * float(frac) - 1e-9).long()   # -1e-9: 100*0.07 = 7.000000000000001 -> 7, not 8
        return rm * (pos < k).float()
    k = torch.minimum(n, torch.full_like(n, int(last_k)))
    return rm * ((pos >= (n - k).clamp(min=0)) & (pos < n)).float()


def with_t0(pm: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Prepend the pre-generation state t=0 (the hidden state AT the last prompt token, which predicts y_0) to a (B, R)
    response position mask -> (B, R+1); column 0 = t=0, supervised for every row with >= 1 valid response token
    (user 10-29 ablation "first_frac + t=0" = probe_oprd_segments' full first25 segment). Pairs with
    DataParallelPPOActor._oprd_response_token_positions(..., include_t0=True), whose column 0 is response_start - 1 of each view."""
    has = (response_mask.float().sum(1, keepdim=True) > 0).to(pm.dtype)
    return torch.cat([has, pm], 1)


def gather_response_hidden(hidden_states, idx: list[int], abs_pos: torch.Tensor, valid: torch.Tensor,
                           packed_index: torch.Tensor | None = None) -> torch.Tensor:
    """Stack the selected layers' hidden states at absolute sequence positions -> (B, L_sel, R, D) float32.

    hidden_states: HF tuple; each entry (B, S, D) (padded layout) or (1, nnz, D) (remove-padding layout).
    abs_pos: (B, R) absolute position of response token t in each row (start + t); valid: (B, R) bool.
    packed_index: (B*S,) map from a padded flat index b*S+s to its row in the packed layout (-1 = padding), required for
    the remove-padding layout. Invalid positions are filled with 0 (they are masked out of the loss)."""
    B, R = abs_pos.shape
    out = []
    for i in idx:
        h = hidden_states[i]
        if packed_index is not None:
            h = h.squeeze(0)                                    # (nnz, D)
            S = packed_index.numel() // B
            flat = (torch.arange(B, device=abs_pos.device).unsqueeze(1) * S + abs_pos.clamp(0, S - 1)).reshape(-1)
            rows = packed_index[flat]
            ok = (rows >= 0) & valid.reshape(-1)
            g = h[rows.clamp(min=0)] * ok.unsqueeze(-1).to(h.dtype)
            out.append(g.reshape(B, R, -1))
        else:
            S = h.shape[1]
            g = torch.gather(h, 1, abs_pos.clamp(0, S - 1).unsqueeze(-1).expand(-1, -1, h.shape[-1]))
            out.append(g * valid.unsqueeze(-1).to(g.dtype))
    return torch.stack(out, 1).float()


def mse_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """OPRD multi-layer loss on (B, L, R, D) reprs and a (B, R) mask: per layer token-mean (over the masked positions of
    the micro-batch) of the per-token mean-over-D squared error, then the mean over layers. normalize=True = the
    OPRD-Bridge branch variant (L2-normalise both vectors first); main / the paper = False. Teacher is detached."""
    t = teacher.detach()
    s = student
    if normalize:
        s, t = F.normalize(s, p=2, dim=-1), F.normalize(t, p=2, dim=-1)
    per_tok = ((s - t) ** 2).mean(-1)                           # (B, L, R)
    m = mask.unsqueeze(1).to(per_tok.dtype)                      # (B, 1, R)
    per_layer = (per_tok * m).sum((0, 2)) / m.sum((0, 2)).clamp(min=1.0)   # (L,)
    return per_layer.mean()


@torch.no_grad()
def cosine(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean cosine similarity per layer -> (L,) (monitoring)."""
    c = F.cosine_similarity(student.float(), teacher.float(), dim=-1)    # (B, L, R)
    m = mask.unsqueeze(1).to(c.dtype)
    return (c * m).sum((0, 2)) / m.sum((0, 2)).clamp(min=1.0)
