"""CPU-testable math for the candidate visual-risk policy objective.

This is NOT a VLM trainer. It implements only the numerical contracts documented
in the accompanying Markdown files. Model/sampler/distributed adapters are not
included. Teacher-derived costs, returns, baselines and advantages are detached.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

CostKind = Literal['probability_gap', 'negative_log_ratio']
Reduction = Literal['token_mean', 'trajectory_sum_mean']


def _check_bt(x: Tensor, mask: Tensor, name: str) -> None:
    if x.ndim != 2 or mask.shape != x.shape or mask.dtype != torch.bool:
        raise ValueError(f'{name}: expected floating [B,T] and boolean mask of same shape')
    if not x.is_floating_point():
        raise TypeError(f'{name}: expected floating tensor')
    if x.device != mask.device:
        raise ValueError(f'{name}: tensor and mask must share device')
    if not torch.isfinite(x[mask]).all():
        raise FloatingPointError(f'{name}: nonfinite value at a valid response position')
    # Response-aligned masks must be contiguous prefixes of True followed by pad.
    if mask.shape[1] > 1 and (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError('mask contains a hole; packing must be unpacked per response first')


def _work_dtype(x: Tensor) -> Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _check_logp(x: Tensor, mask: Tensor, name: str) -> None:
    _check_bt(x, mask, name)
    if (x[mask] > 1e-6).any():
        raise ValueError(f'{name}: log probabilities must be <= 0')


def reduce_positions(values: Tensor, mask: Tensor, reduction: Reduction) -> Tensor:
    _check_bt(values, mask, 'values')
    if not mask.any():
        raise ValueError('cannot reduce a training batch with no valid response tokens')
    safe = torch.where(mask, values, torch.zeros_like(values))
    if reduction == 'token_mean':
        return safe.sum() / mask.sum()
    if reduction == 'trajectory_sum_mean':
        nonempty = mask.any(dim=1)
        return safe.sum(dim=1)[nonempty].mean()
    raise ValueError(f'unknown reduction: {reduction}')


@torch.no_grad()
def visual_cost(
    logp_plus: Tensor, logp_null: Tensor, mask: Tensor,
    kind: CostKind = 'probability_gap',
) -> dict[str, Tensor]:
    """Gathered *full-vocabulary* teacher log probabilities for realized tokens.

    logp_plus/logp_null must use identical teacher scoring temperature/processors.
    They must not be top-k-renormalized token probabilities or tail-bucket values.
    """
    _check_logp(logp_plus, mask, 'logp_plus')
    _check_logp(logp_null, mask, 'logp_null')
    lp = torch.where(mask, _work_dtype(logp_plus.detach()), 0.0)
    ln = torch.where(mask, _work_dtype(logp_null.detach()), 0.0)
    u = lp - ln
    if kind == 'probability_gap':
        # p0 - p+ = p0 * (1-exp(u)); stable when u is just below zero.
        cost = ln.exp() * (-torch.expm1(u.clamp(max=0)))
    elif kind == 'negative_log_ratio':
        cost = (-u).clamp(min=0)
    else:
        raise ValueError(f'unknown cost kind: {kind}')
    zero = torch.zeros_like(cost)
    return {'u': torch.where(mask, u, zero), 'cost': torch.where(mask, cost, zero)}


@torch.no_grad()
def history_and_risk(
    cost: Tensor, mask: Tensor, risk: Literal['linear', 'bounded_slope'] = 'bounded_slope',
) -> dict[str, Tensor]:
    """H_t is exclusive prefix sum; d_t is incremental risk, not repeated history.

    bounded_slope: Phi(H)=2H-log(1+H), hence d=2c-log1p(c/(1+H)).
    """
    _check_bt(cost, mask, 'cost')
    if (cost[mask] < 0).any():
        raise ValueError('visual cost must be nonnegative')
    c = torch.where(mask, _work_dtype(cost.detach()), 0.0)
    inclusive = c.cumsum(dim=1)
    history = torch.cat([torch.zeros_like(c[:, :1]), inclusive[:, :-1]], dim=1)
    if risk == 'linear':
        d = c
    elif risk == 'bounded_slope':
        d = 2 * c - torch.log1p(c / (1 + history))
    else:
        raise ValueError(f'unknown risk transform: {risk}')
    return {'history': torch.where(mask, history, 0.0),
            'increment': torch.where(mask, d, 0.0)}


@torch.no_grad()
def discounted_cost_to_go(increment: Tensor, mask: Tensor, gamma: float) -> Tensor:
    """Includes the current increment. True termination/pad has zero continuation.

    This implements *observed-prefix* returns at a length cutoff, not a bootstrap
    estimate of the unobserved continuation. See termination contract in docs.
    """
    _check_bt(increment, mask, 'increment')
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('gamma must be finite and in [0,1]')
    d = torch.where(mask, _work_dtype(increment.detach()), 0.0)
    out = torch.zeros_like(d)
    acc = torch.zeros(d.shape[0], device=d.device, dtype=d.dtype)
    for t in range(d.shape[1] - 1, -1, -1):
        acc = torch.where(mask[:, t], d[:, t] + gamma * acc, 0.0)
        out[:, t] = acc
    return out


@torch.no_grad()
def leave_one_out(
    returns: Tensor, mask: Tensor, group_ids: Tensor,
) -> dict[str, Tensor]:
    """Fixed-size independent rollout groups; terminated peers contribute zero.

    All group members MUST be assembled before calling, even across GPU ranks.
    This function cannot check statistical independence of generation.
    """
    _check_bt(returns, mask, 'returns')
    if group_ids.shape != (returns.shape[0],) or group_ids.device != returns.device:
        raise ValueError('group_ids must be [B] on the same device')
    r = torch.where(mask, _work_dtype(returns.detach()), 0.0)
    b = torch.zeros_like(r)
    for group in torch.unique(group_ids):
        rows = torch.where(group_ids == group)[0]
        n = rows.numel()
        if n < 2:
            raise ValueError('leave-one-out requires >=2 independently sampled rollouts per group')
        part = r[rows]
        b[rows] = (part.sum(dim=0, keepdim=True) - part) / (n - 1)
    a = torch.where(mask, b - r, 0.0)
    return {'baseline': torch.where(mask, b, 0.0), 'advantage': a}


def ppo_clip_loss(
    new_logp: Tensor, old_logp: Tensor, advantage: Tensor, mask: Tensor,
    clip_epsilon: float, reduction: Reduction = 'token_mean',
) -> dict[str, Tensor]:
    """new/old logp must refer to the SAME actual sampling-policy definition.

    No silent clamp on log-ratios: nonfinite ratios are treated as an error.
    Clipping limits a sample's optimization incentive, not the policy itself.
    """
    _check_logp(new_logp, mask, 'new_logp')
    _check_logp(old_logp, mask, 'old_logp')
    _check_bt(advantage, mask, 'advantage')
    if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError('clip_epsilon must be finite and in (0,1)')
    new = _work_dtype(new_logp)
    old = _work_dtype(old_logp.detach())
    a = torch.where(mask, _work_dtype(advantage.detach()), 0.0)
    log_ratio = torch.where(mask, new - old, 0.0)
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio[mask]).all():
        raise FloatingPointError('nonfinite PPO ratio; audit sampler/logp consistency')
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    position_loss = -torch.minimum(ratio * a, clipped * a)
    clipped_incentive = ((a > 0) & (ratio > 1 + clip_epsilon)) | ((a < 0) & (ratio < 1 - clip_epsilon))
    return {
        'loss': reduce_positions(position_loss, mask, reduction),
        'ratio': ratio.detach(),
        'position_loss': torch.where(mask, position_loss, 0.0),
        'active_clip_fraction': clipped_incentive[mask].float().mean().detach(),
    }


def jsd_from_logits(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor) -> Tensor:
    """Returns [B,T] JSD. Full finite logits only; teacher is stop-gradient.

    A production top-k+tail adapter must assemble common-support distributions
    correctly. This simple core intentionally does not guess that adapter.
    """
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError('JSD logits must have the same [B,T,V] shape')
    if mask.shape != student_logits.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('JSD mask must have shape [B,T]')
    if not torch.isfinite(student_logits[mask]).all() or not torch.isfinite(teacher_logits[mask]).all():
        raise FloatingPointError('full-support JSD logits must be finite at valid positions')
    s = torch.where(mask.unsqueeze(-1), _work_dtype(student_logits), 0.0)
    t = torch.where(mask.unsqueeze(-1), _work_dtype(teacher_logits.detach()), 0.0)
    log_s = s.log_softmax(dim=-1)
    log_t = t.log_softmax(dim=-1)
    log_m = torch.logaddexp(log_s, log_t) - math.log(2)
    # m must remain in the computational graph: student depends on both terms.
    value = 0.5 * ((log_s.exp() * (log_s - log_m)).sum(-1)
                   + (log_t.exp() * (log_t - log_m)).sum(-1))
    return torch.where(mask, value, 0.0)


def gu_loss(
    student_logp_raw: Tensor, gate: Tensor, mask: Tensor,
    reduction: Reduction = 'token_mean',
) -> Tensor:
    """Alternative local objective only. Do not silently add to temporal PPO."""
    _check_logp(student_logp_raw, mask, 'student_logp_raw')
    _check_bt(gate, mask, 'gate')
    if (gate[mask] < 0).any():
        raise ValueError('GU gate must be nonnegative')
    lp = torch.where(mask, _work_dtype(student_logp_raw), 0.0)
    g = torch.where(mask, _work_dtype(gate.detach()), 0.0)
    return reduce_positions(g / (2 - lp.exp()), mask, reduction)
