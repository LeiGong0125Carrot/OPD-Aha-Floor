# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

import math
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.utils.metric import AggregationType, Metric
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]

        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = False,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n, )]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        if loss_scale_factor is None:
            loss_scale_factor = loss_mask.shape[-1]
        loss = torch.sum(seq_losses) / loss_scale_factor
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_config: Any,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_null_all_log_probs: Optional[torch.Tensor] = None,
    teacher_null_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_null_log_probs: Optional[torch.Tensor] = None,
    teacher_internal_target_log_probs: Optional[torch.Tensor] = None,
    teacher_internal_r_topk: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
    pbd_loss_mask: Optional[torch.Tensor] = None,
    pbd_is_branch: Optional[torch.Tensor] = None,
    pbd_scale: float = 1.0,          # V0 rows: N_total / N_v0
    pbd_scale_b: Optional[float] = None,   # branch rows: N_total / N_branch (defaults to pbd_scale)
) -> tuple[torch.Tensor, dict[str, Any]]:

    metrics = {}

    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
    # P5 PBD: branch rows only learn on t >= t* minus leak tokens; Replace parents lose t >= t*.
    # response_mask itself stays 1 on real tokens (it doubles as the teacher attention mask).
    _pbd = pbd_is_branch is not None
    if pbd_loss_mask is not None:
        loss_mask = loss_mask * pbd_loss_mask.to(loss_mask.dtype)
    if _pbd and getattr(self_distillation_config, "counterfactual_null_mode", None) is not None:
        raise ValueError("PBD rows present but counterfactual_null_mode is set (PBD v0.1 is V0-based)")

    counterfactual_null_mode = getattr(self_distillation_config, "counterfactual_null_mode", None)
    # Reference distribution for the tilt u = log p_real - log p_ref (candidate S,
    # docs/negative_history/candidate_s_null_free_implementation_plan.md §11):
    #   "null"    -> p_ref = teacher on the matched visual-null view (OPD-Aha; needs the null forward)
    #   "student" -> p_ref = the CURRENT student's own distribution, detached (no null forward at all)
    #   "teacher" -> ROLES SWAPPED (candidate S2): base = sg(p_student), p_ref = the single teacher forward
    #                (fed e.g. the full image with the GT box region blanked), u = log sg(p_S) - log p_T.
    #                No privileged teacher and no null forward.
    # Reconstruction on/off stays keyed on counterfactual_null_mode; this field only picks the reference.
    counterfactual_reference = str(
        getattr(self_distillation_config, "counterfactual_reference", "null") or "null"
    )
    # Candidate S-tail (candidate_s_results §5 (a)): the student reference is only trusted where the
    # student actually places mass -- u is defined on the explicit top-k set and forced to 0 on the
    # tail bucket, so the aggregated tail can neither be boosted nor suppressed by the tilt.
    counterfactual_reference_tail_u_zero = bool(
        getattr(self_distillation_config, "counterfactual_reference_tail_u_zero", False)
    )
    counterfactual_extrapolation_beta = float(
        getattr(self_distillation_config, "counterfactual_extrapolation_beta", 1.0)
    )
    counterfactual_u_clip_pos = bool(getattr(self_distillation_config, "counterfactual_u_clip_pos", False))
    counterfactual_st_enable = bool(getattr(self_distillation_config, "counterfactual_st_enable", False))
    counterfactual_st_tau = float(getattr(self_distillation_config, "counterfactual_st_tau", 0.10) or 0.10)
    counterfactual_st_head_ratio = float(
        getattr(self_distillation_config, "counterfactual_st_head_ratio", 0.10) or 0.10
    )
    counterfactual_st_alpha_max = float(
        getattr(self_distillation_config, "counterfactual_st_alpha_max", 0.50) or 0.50
    )
    counterfactual_st_eps = float(getattr(self_distillation_config, "counterfactual_st_eps", 0.30) or 0.30)
    counterfactual_floor_alpha = float(
        getattr(self_distillation_config, "counterfactual_floor_alpha", 0.0) or 0.0
    )
    counterfactual_tanh_scale = float(
        getattr(self_distillation_config, "counterfactual_tanh_scale", 0.0) or 0.0
    )
    _gamma_raw = getattr(self_distillation_config, "counterfactual_target_gamma", None)
    counterfactual_target_gamma = 1.0 if _gamma_raw is None else float(_gamma_raw)
    # Runtime guard, NOT a dataclass guard: on this repo's training path main_ppo.py forces
    # the legacy worker impl, so the actor config stays a plain DictConfig and
    # SelfDistillationConfig.__post_init__ never runs -- every validation living there is
    # dead code in production. Without this check, TARGET_GAMMA=-1 would silently train an
    # INVERTED target (mass on the least likely tokens) for hours with no error, and
    # TARGET_GAMMA=0 would silently degrade to the identity.
    if counterfactual_target_gamma <= 0.0:
        raise ValueError(
            "self_distillation.counterfactual_target_gamma must be > 0, got "
            f"{counterfactual_target_gamma} (gamma<=0 inverts or flattens the target)"
        )
    if counterfactual_tanh_scale < 0.0:
        raise ValueError(
            f"self_distillation.counterfactual_tanh_scale must be >= 0, got {counterfactual_tanh_scale}"
        )
    # P1 (Counterfactual-Flip Distillation, docs/proposals_2026-10-06/01 + 06):
    #   target_mode "tilt"   -> every existing path, bit-identical;
    #   "flip"   -> on the flip set F = {t>=1: argmax p+ != argmax p0, argmax p+ not in tail} the target is the
    #               gamma-sharpened p+ (explicit columns, tail mass preserved), elsewhere plain p+ (V0);
    #   "flip_u" -> on F the A target softmax(log p+ + beta u), elsewhere plain p+ (isolates the off-flip tilt).
    #   Positions in F get loss weight 1 + flip_lambda. t = 0 (no student prefix) is never in F by definition.
    counterfactual_target_mode = str(getattr(self_distillation_config, "counterfactual_target_mode", "tilt") or "tilt")
    _fg = getattr(self_distillation_config, "flip_gamma", 50.0)
    flip_gamma = 50.0 if _fg is None else float(_fg)
    _fl = getattr(self_distillation_config, "flip_lambda", 1.0)
    flip_lambda = 1.0 if _fl is None else float(_fl)
    flip_tail_policy = str(getattr(self_distillation_config, "flip_tail_policy", "exclude") or "exclude")
    # A-t0 control (06 §1.1): the tilt's beta is 0 at the first response token, everything else = A.
    counterfactual_t0_beta_zero = bool(getattr(self_distillation_config, "counterfactual_t0_beta_zero", False))
    if counterfactual_target_mode not in ("tilt", "flip", "flip_u"):
        raise ValueError(f"self_distillation.counterfactual_target_mode must be tilt|flip|flip_u, got {counterfactual_target_mode!r}")
    _flip = counterfactual_target_mode in ("flip", "flip_u")
    if _flip:
        if counterfactual_null_mode is None or counterfactual_reference != "null":
            raise ValueError("counterfactual_target_mode=flip/flip_u requires counterfactual_null_mode set and counterfactual_reference='null'")
        if not (flip_gamma > 0.0):
            raise ValueError(f"self_distillation.flip_gamma must be > 0, got {flip_gamma}")
        if flip_lambda < 0.0:
            raise ValueError(f"self_distillation.flip_lambda must be >= 0, got {flip_lambda}")
        if flip_tail_policy not in ("exclude", "include"):
            raise ValueError(f"self_distillation.flip_tail_policy must be exclude|include, got {flip_tail_policy!r}")
        if counterfactual_target_gamma != 1.0 or counterfactual_tanh_scale > 0.0 or counterfactual_floor_alpha > 0.0 \
                or counterfactual_u_clip_pos or counterfactual_st_enable or counterfactual_t0_beta_zero:
            raise ValueError("counterfactual_target_mode=flip/flip_u is exclusive with target_gamma!=1, tanh, floor, u_clip_pos, st and t0_beta_zero")
    if counterfactual_t0_beta_zero and counterfactual_null_mode is None:
        raise ValueError("counterfactual_t0_beta_zero requires counterfactual_null_mode (it modifies the tilt)")
    if counterfactual_t0_beta_zero and (counterfactual_st_enable or counterfactual_reference != "null"):
        raise ValueError("counterfactual_t0_beta_zero is defined for the plain tilt path only (not with st or a student/teacher reference)")
    counterfactual_hist_adaptive_beta = bool(
        getattr(self_distillation_config, "counterfactual_hist_adaptive_beta", False)
    )
    if _flip and counterfactual_hist_adaptive_beta:
        raise ValueError("counterfactual_target_mode=flip/flip_u is exclusive with counterfactual_hist_adaptive_beta")
    if counterfactual_t0_beta_zero and counterfactual_hist_adaptive_beta:
        raise ValueError("counterfactual_t0_beta_zero is exclusive with counterfactual_hist_adaptive_beta")
    counterfactual_hist_alpha = float(
        getattr(self_distillation_config, "counterfactual_hist_alpha", 1.0) or 1.0
    )
    counterfactual_hist_shuffle = bool(
        getattr(self_distillation_config, "counterfactual_hist_shuffle", False)
    )
    counterfactual_future_weight = bool(
        getattr(self_distillation_config, "counterfactual_future_weight", False)
    )
    counterfactual_future_alpha = float(
        getattr(self_distillation_config, "counterfactual_future_alpha", 1.0) or 1.0
    )
    if (_flip or counterfactual_t0_beta_zero) and counterfactual_future_weight:
        raise ValueError("counterfactual_target_mode=flip/flip_u and t0_beta_zero are exclusive with counterfactual_future_weight")
    # History aggregation (arm C uses "cumsum", 04 §4). "mean" replaces the cumulative N_t by the
    # prefix MEAN conflict Nbar_t = N_t / (#valid tokens before t) and maps it with a reference
    # kappa: s_t = Nbar_t / (Nbar_t + kappa). Motivation (docs/negative_history/nhC_r1_casestudy.md):
    # the cumulative N_t saturates within a few dozen tokens (c_t ~ 0.10/token), so beta_t became
    # an almost-constant ~6.3 -- arm C measured a DOSE change, not temporal adaptation. The mean
    # stays O(c) and moves with the trajectory; kappa ~= the measured average c (0.103) puts an
    # average prefix at s=0.5 (beta_t ~= 6), i.e. dose-matched to arm C, so any difference from C
    # is attributable to the temporal variation itself.
    counterfactual_hist_mode = str(
        getattr(self_distillation_config, "counterfactual_hist_mode", "cumsum") or "cumsum"
    )
    _kappa_raw = getattr(self_distillation_config, "counterfactual_hist_kappa", None)
    counterfactual_hist_kappa = 0.10 if _kappa_raw is None else float(_kappa_raw)
    # Which half of the visual contrast the trajectory state modulates (X1, 10-03):
    #   "neg": suppression of visually opposed tokens (arms C / Ahm / Ahf),
    #   "pos": re-grounding -- the history state raises the PROMOTION of visually supported
    #          tokens, q ∝ p+ · exp(beta_t·u⁺ + beta·u⁻); the negative half keeps base beta.
    counterfactual_hist_half = str(
        getattr(self_distillation_config, "counterfactual_hist_half", "neg") or "neg"
    )
    # ---- Route 1: internal residual target (single frozen-teacher forward) ----
    teacher_target_mode = str(getattr(self_distillation_config, "teacher_target_mode", "legacy") or "legacy")
    if teacher_target_mode not in ("legacy", "internal_residual"):
        raise ValueError(f"teacher_target_mode must be 'legacy' or 'internal_residual', got {teacher_target_mode!r}")
    _internal = teacher_target_mode == "internal_residual"
    _ti = getattr(self_distillation_config, "teacher_internal", None)
    def _tiget(k, d=None):
        if _ti is None:
            return d
        v = _ti.get(k, d) if hasattr(_ti, "get") else getattr(_ti, k, d)
        return d if v is None else v
    internal_lambda = _tiget("strength")
    internal_tail = str(_tiget("tail_policy", "full"))
    if _internal:
        # Runtime guards. (The dataclass __post_init__ DOES run at worker start-up via
        # omega_conf_to_dataclass -- reviewer-verified 10-05 -- but the loss is also called from
        # tests and probes with plain dicts, so the guards are repeated here.)
        if internal_lambda is None or not math.isfinite(float(internal_lambda)) or float(internal_lambda) < 0.0:
            raise ValueError(f"internal_residual: teacher_internal.strength must be set, finite and >= 0, got {internal_lambda!r}")
        internal_lambda = float(internal_lambda)
        if internal_tail not in ("full", "aha"):
            raise ValueError(f"internal_residual: tail_policy must be 'full' or 'aha', got {internal_tail!r}")
        if counterfactual_null_mode is not None or counterfactual_reference != "null":
            raise ValueError("internal_residual is null-free and student-free: counterfactual_null_mode must be None "
                             "and counterfactual_reference must be 'null' (the legacy placeholder)")
        if (counterfactual_u_clip_pos or counterfactual_st_enable or counterfactual_floor_alpha > 0.0
                or counterfactual_tanh_scale > 0.0 or counterfactual_target_gamma != 1.0
                or counterfactual_hist_adaptive_beta or counterfactual_future_weight or counterfactual_hist_shuffle):
            raise ValueError("internal_residual is mutually exclusive with u_clip_pos / st / floor / tanh / gamma!=1 / "
                             "hist / future / hist_shuffle (single-variable discipline)")
        if (teacher_null_topk_log_probs is not None or teacher_null_all_log_probs is not None
                or teacher_null_log_probs is not None):
            raise ValueError("internal_residual: null-teacher tensors were passed although no null forward may run")
        if not self_distillation_config.full_logit_distillation or getattr(self_distillation_config, "distillation_topk", None) is None \
                or not bool(getattr(self_distillation_config, "distillation_add_tail", True)):
            raise ValueError("internal_residual requires full_logit_distillation with a top-k support and distillation_add_tail=True")
        if internal_tail == "full" and teacher_internal_target_log_probs is None:
            raise ValueError("internal_residual(full): teacher_internal_target_log_probs [B,T,K+1] is required")
        if teacher_internal_r_topk is None:
            raise ValueError("internal_residual: teacher_internal_r_topk [B,T,K] is required")
    internal_r_abs_per_token = internal_r_sq_per_token = None
    if counterfactual_reference not in ("null", "student", "teacher"):
        raise ValueError(
            f"counterfactual_reference must be 'null', 'student' or 'teacher', got {counterfactual_reference!r}"
        )
    _ref_student = counterfactual_reference in ("student", "teacher")   # both are null-free modes
    _swap_sides = counterfactual_reference == "teacher"
    if _ref_student:
        # Runtime guards (dataclass __post_init__ is dead code on this training path).
        if counterfactual_null_mode is None:
            raise ValueError(
                "counterfactual_reference='student' requires counterfactual_null_mode to be set "
                "(it gates target reconstruction); with null_mode=None this would silently be V0."
            )
        if not self_distillation_config.full_logit_distillation or getattr(
                self_distillation_config, "distillation_topk", None) is None:
            raise ValueError(
                "counterfactual_reference='student' requires full_logit_distillation=True with a top-k "
                "support (the student-reference target is defined on the student's top-k + tail)."
            )
        if (counterfactual_u_clip_pos or counterfactual_st_enable or counterfactual_floor_alpha > 0.0
                or counterfactual_tanh_scale > 0.0 or counterfactual_target_gamma != 1.0
                or counterfactual_hist_adaptive_beta or counterfactual_future_weight
                or counterfactual_hist_shuffle):
            raise ValueError(
                "counterfactual_reference='student' (arm S, first round) is mutually exclusive with "
                "u_clip_pos / st_enable / floor_alpha / tanh_scale / target_gamma!=1 / hist / future / "
                "hist_shuffle (single-variable discipline, candidate S plan §11.2)."
            )
        if (teacher_null_topk_log_probs is not None or teacher_null_all_log_probs is not None
                or teacher_null_log_probs is not None):
            raise ValueError(
                "counterfactual_reference='student' but null-teacher log-probs were passed: the null "
                "forward must be skipped end-to-end, not silently ignored (config contradiction)."
            )
    if counterfactual_reference_tail_u_zero:
        if not _ref_student:
            raise ValueError("counterfactual_reference_tail_u_zero requires counterfactual_reference='student'")
        if not bool(getattr(self_distillation_config, "distillation_add_tail", True)):
            raise ValueError("counterfactual_reference_tail_u_zero requires distillation_add_tail=True (there is no tail column otherwise)")
    _nh_any = counterfactual_hist_adaptive_beta or counterfactual_future_weight
    if counterfactual_hist_adaptive_beta:
        if counterfactual_hist_half not in ("neg", "pos"):
            raise ValueError(
                f"counterfactual_hist_half must be 'neg' or 'pos', got {counterfactual_hist_half!r}"
            )
        if counterfactual_hist_half == "pos" and counterfactual_u_clip_pos:
            raise ValueError(
                "counterfactual_hist_half='pos' modulates u⁺, which u_clip_pos removes -- "
                "the combination would be a silent no-op."
            )
    if counterfactual_hist_adaptive_beta:
        if counterfactual_hist_mode not in ("cumsum", "mean", "hf"):
            raise ValueError(
                "counterfactual_hist_mode must be 'cumsum', 'mean' or 'hf', "
                f"got {counterfactual_hist_mode!r}"
            )
        if counterfactual_hist_mode in ("mean", "hf") and counterfactual_hist_kappa <= 0.0:
            raise ValueError(f"counterfactual_hist_kappa must be > 0, got {counterfactual_hist_kappa}")
    if _nh_any:
        # Runtime guards (the dataclass __post_init__ never runs on this training path --
        # main_ppo forces the legacy worker impl, so the config stays a plain DictConfig).
        # hist_adaptive_beta may run on the full A tilt: beta_t then modulates ONLY the negative
        # half, q ∝ p+ · exp(beta·u⁺ + beta_t·u⁻) (positive half keeps the base beta). On the
        # suppression-only base (u_clip_pos) u⁺ ≡ 0, so this reduces bit-identically to arm C.
        # future_weight was only designed/validated on the negative-only base: keep that guard.
        if counterfactual_future_weight and not counterfactual_u_clip_pos:
            raise ValueError(
                "counterfactual_future_weight requires counterfactual_u_clip_pos=True: the design "
                "builds on the negative-only tilt q ∝ p+ · exp(beta · min(u, 0)) "
                "(docs/negative_history/03)."
            )
        if teacher_null_log_probs is None:
            raise ValueError(
                "negative-history gates require teacher_null_log_probs (realized-token null "
                "log-probs, [B,T]) -- plumbed from dp_actor's teacher_null forward."
            )
        if counterfactual_st_enable or counterfactual_floor_alpha > 0.0 \
                or counterfactual_tanh_scale > 0.0 or counterfactual_target_gamma != 1.0:
            raise ValueError(
                "negative-history gates are mutually exclusive with st_enable / floor_alpha / "
                "tanh_scale / target_gamma (single-variable discipline)."
            )
    if counterfactual_hist_shuffle and not counterfactual_hist_adaptive_beta:
        raise ValueError(
            "counterfactual_hist_shuffle is the misalign CONTROL for hist_adaptive_beta and "
            "requires it to be enabled."
        )
    counterfactual_target_tv_per_token = None
    sup_frac_u_pos_per_token = None
    cf_u_abs_mean_per_token = cf_u_frac_pos_per_token = None
    cf_u_p90 = None
    teacher_student_kl_per_token = teacher_tail_mass_per_token = None
    target_tail_mass_per_token = target_argmax_tail_frac_per_token = None
    floor_clamped_frac_per_token = None
    tanh_compressed_frac_per_token = None
    target_max_prob_per_token = None
    target_entropy_per_token = None
    st_delta_per_token = st_cov_per_token = st_donor_frac_per_token = None
    st_mass_d_per_token = st_mass_r_per_token = None

    # ---- Negative-history trajectory branch (docs/negative_history/02-05) ----
    # Two parallel views of the same visual signal: the vocabulary-level u_t(v) reshapes
    # the target (existing code below); the realized-token u_t(y_t) describes the sampled
    # trajectory's state and is computed here from exact [B,T] realized log-probs
    # (immune to the sampled token falling outside the top-k support).
    nh_beta_t = None          # [B,T] adaptive suppression strength (arm C/E)
    nh_weight = None          # [B,T] future-persistence JSD weight  (arm D/E)
    nh_c_mean = nh_c_frac_pos = nh_N_last_mean = None
    nh_s_mean = nh_beta_t_mean = nh_w_mean = nh_w_p90 = None
    nh_h_mean = nh_f_mean = nh_eta_row_frac = nh_s_within_var = nh_s_rowmean_sq = None
    if _nh_any:
        with torch.no_grad():
            # c_t = [-u_t(y_t)]_+  (02 §6); padding never enters (02 §11)
            u_realized = teacher_log_probs - teacher_null_log_probs
            # torch.where, not "* loss_mask": a non-finite realized log-prob at a padded
            # position gives inf*0 = NaN, and the reverse cumsum for the future weight would
            # then smear that NaN backwards onto every valid position of the row.
            nh_c = torch.where(loss_mask > 0, torch.relu(-u_realized), torch.zeros_like(u_realized))
            nh_c_mean = (verl_F.masked_sum(nh_c, loss_mask) / loss_mask.sum().clamp(min=1.0)).item()
            nh_c_frac_pos = (
                verl_F.masked_sum((nh_c > 0).float(), loss_mask) / loss_mask.sum().clamp(min=1.0)
            ).item()
            if counterfactual_hist_adaptive_beta:
                c_hist = nh_c
                if counterfactual_hist_shuffle:
                    # Misalign CONTROL: permute c within each row's valid span -- preserves
                    # the marginal distribution, destroys temporal alignment. If arm C's gain
                    # survives this shuffle, it is a perturbation-magnitude artifact
                    # (the C-temporal lesson), not temporal credit.
                    c_hist = nh_c.clone()
                    for _b in range(c_hist.shape[0]):
                        _idx = loss_mask[_b].nonzero(as_tuple=True)[0]
                        if _idx.numel() > 1:
                            _perm = _idx[torch.randperm(_idx.numel(), device=_idx.device)]
                            c_hist[_b, _idx] = nh_c[_b, _perm]
                # N_t = sum_{k<t} c_k (EXCLUSIVE, 04 §4); s = N/(1+N); beta_t = beta(1+a_H s)
                nh_N = torch.cumsum(c_hist, dim=1) - c_hist
                if counterfactual_hist_mode == "mean":
                    # exclusive count of valid prefix tokens; t with an empty prefix -> s = 0
                    _n_prev = torch.cumsum(loss_mask, dim=1) - loss_mask
                    nh_Nbar = nh_N / _n_prev.clamp(min=1.0)
                    nh_s = nh_Nbar / (nh_Nbar + counterfactual_hist_kappa)
                elif counterfactual_hist_mode == "hf":
                    # History-future residual (aha_history_future_residual_design.md §06):
                    #   h_t = Hbar/(Hbar+k), f_t = Fbar/(Fbar+k), s_t = h_t f_t eta_t
                    # so Delta_beta_t = beta*a_H*h_t*f_t*eta_t on the negative half only.
                    # Hbar: exclusive prefix mean; Fbar: STRICT suffix mean (k>t), 0 if no suffix.
                    # eta: the rollout ended normally (not length-truncated) -- same convention
                    # as verl's response_length/clip_ratio (valid length == padded width).
                    # Truncated rows fall back to the plain A target, but keep their supervision.
                    _n_prev = torch.cumsum(loss_mask, dim=1) - loss_mask
                    nh_Hbar = nh_N / _n_prev.clamp(min=1.0)
                    nh_h = nh_Hbar / (nh_Hbar + counterfactual_hist_kappa)
                    _rev_h = lambda x: x.flip(1).cumsum(dim=1).flip(1)
                    _Fs = _rev_h(c_hist) - c_hist
                    _Kn = _rev_h(loss_mask) - loss_mask
                    nh_Fbar_hf = _Fs / _Kn.clamp(min=1.0)
                    nh_f = nh_Fbar_hf / (nh_Fbar_hf + counterfactual_hist_kappa)
                    _resp_len = response_mask.sum(dim=1)
                    _eta_row = (_resp_len < response_mask.shape[1]).to(nh_f.dtype)   # [B]
                    nh_s = nh_h * nh_f * _eta_row.unsqueeze(1)
                    _den = loss_mask.sum().clamp(min=1.0)
                    nh_h_mean = (verl_F.masked_sum(nh_h, loss_mask) / _den).item()
                    nh_f_mean = (verl_F.masked_sum(nh_f, loss_mask) / _den).item()
                    _has = loss_mask.sum(dim=1) > 0
                    nh_eta_row_frac = _eta_row[_has].mean().item() if _has.any() else 1.0
                else:
                    nh_s = nh_N / (1.0 + nh_N)
                nh_beta_t = counterfactual_extrapolation_beta * (
                    1.0 + counterfactual_hist_alpha * nh_s
                )
                if not torch.isfinite(nh_beta_t[loss_mask > 0]).all():
                    raise FloatingPointError(
                        "negative-history: non-finite beta_t on a valid position "
                        "(would silently poison the reconstructed target)")
                _lens = loss_mask.sum(dim=1)
                _last = _lens.clamp(min=1).long() - 1
                _rows = _lens > 0          # rows fully masked by self_distillation_mask excluded
                nh_N_last_mean = (
                    nh_N.gather(1, _last.unsqueeze(1)).squeeze(1)[_rows].mean().item()
                    if _rows.any() else 0.0
                )
                nh_s_mean = (verl_F.masked_sum(nh_s, loss_mask) / loss_mask.sum().clamp(min=1.0)).item()
                # Is s_t temporal or a per-rollout dose? Micro-batches usually hold ONE rollout,
                # so report additive pieces and combine after the cross-micro-batch mean:
                #   within-rollout var  = mean(nh_s_within_var)
                #   between-rollout var = mean(nh_s_rowmean_sq) - mean(nh_s_mean)^2
                _cnt = loss_mask.sum(dim=1)
                _ok = _cnt > 0
                if _ok.any():
                    _rm = verl_F.masked_sum(nh_s, loss_mask, axis=1)[_ok] / _cnt[_ok]
                    _dev = (nh_s[_ok] - _rm.unsqueeze(1)) ** 2
                    nh_s_within_var = ((_dev * loss_mask[_ok]).sum() / _cnt[_ok].sum()).item()
                    nh_s_rowmean_sq = (_rm ** 2).mean().item()
                nh_beta_t_mean = (
                    verl_F.masked_sum(nh_beta_t, loss_mask) / loss_mask.sum().clamp(min=1.0)
                ).item()
            if counterfactual_future_weight:
                # Exclusive suffix mean (05 §5-§6): Fbar_t = (sum_{k>t} c_k) / K_t, K_t=0 -> w=1
                _rev = lambda x: x.flip(1).cumsum(dim=1).flip(1)
                nh_F = _rev(nh_c) - nh_c
                nh_K = _rev(loss_mask) - loss_mask
                nh_Fbar = nh_F / nh_K.clamp(min=1.0)
                nh_r = nh_Fbar / (1.0 + nh_Fbar)
                nh_weight = (1.0 + counterfactual_future_alpha * nh_r) * loss_mask
                if not torch.isfinite(nh_weight[loss_mask > 0]).all():
                    raise FloatingPointError(
                        "negative-history: non-finite future weight on a valid position "
                        "(would silently poison the JSD gradient)")
                _wv = nh_weight[loss_mask > 0]
                nh_w_mean = _wv.mean().item() if _wv.numel() else 1.0
                nh_w_p90 = _wv.quantile(0.9).item() if _wv.numel() else 1.0

    if self_distillation_config.full_logit_distillation:
        use_topk = self_distillation_config.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError("top-k distillation requires student_topk_log_probs and teacher_topk_log_probs.")

            def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                # Compute tail log-probability using logsumexp for numerical stability
                # log(1 - sum(p_i)) = log(1 - exp(log_sum_exp(log(p_i))))
                log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)  # Clamp to avoid log_s >= 0 (which implies sum(probs) >= 1)
                tail_log = torch.log(-torch.expm1(log_s))  # We use the identity: 1 - exp(x) = -(exp(x) - 1); torch.expm1(x) computes (e^x - 1) with high precision for small x.
                return torch.cat([log_probs, tail_log], dim=-1)

            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ

            student_distill_log_probs = student_topk_log_probs
            teacher_real_distill_log_probs = teacher_topk_log_probs
            teacher_null_distill_log_probs = teacher_null_topk_log_probs
            if self_distillation_config.distillation_add_tail:
                student_distill_log_probs = add_tail(student_distill_log_probs)
                teacher_real_distill_log_probs = add_tail(teacher_real_distill_log_probs)
                if teacher_null_distill_log_probs is not None:
                    teacher_null_distill_log_probs = add_tail(teacher_null_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                teacher_real_distill_log_probs = renorm_topk_log_probs(teacher_real_distill_log_probs)
                if teacher_null_distill_log_probs is not None:
                    teacher_null_distill_log_probs = renorm_topk_log_probs(teacher_null_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError("full_logit_distillation requires student_all_log_probs and teacher_all_log_probs.")
            student_distill_log_probs = student_all_log_probs
            teacher_real_distill_log_probs = teacher_all_log_probs
            teacher_null_distill_log_probs = teacher_null_all_log_probs

        if counterfactual_null_mode is not None:
            _true_teacher_distill_log_probs = teacher_real_distill_log_probs   # for diagnostics (pre-swap)
            if _swap_sides:
                # Candidate S2: base = sg(p_student); reference = the teacher forward (hidebox view).
                # Reuse the same code path by swapping the two tensors; both are no-grad.
                with torch.no_grad():
                    teacher_null_distill_log_probs = teacher_real_distill_log_probs
                    teacher_real_distill_log_probs = student_distill_log_probs.detach()
            elif _ref_student:
                # Candidate S: the reference is the student's OWN distribution on the same
                # support (already add_tail'ed / renormalised above), detached so that no
                # gradient flows through the target. Not an in-place op on the student tensor.
                with torch.no_grad():
                    teacher_null_distill_log_probs = student_distill_log_probs.detach()
            if teacher_null_distill_log_probs is None:
                raise ValueError(
                    "Visual-counterfactual target reconstruction requires null-teacher log probabilities."
                )
            # Configurable residual extrapolation in log-probability space:
            # q = softmax(log p_real + beta * (log p_real - log p_ref)).
            u_term = teacher_real_distill_log_probs - teacher_null_distill_log_probs
            if counterfactual_reference_tail_u_zero:
                # S-tail: last column is the tail bucket (add_tail); zero its u. Not in-place on a
                # tensor that aliases the student (u_term is a fresh no-grad tensor).
                u_term = torch.cat([u_term[..., :-1], torch.zeros_like(u_term[..., -1:])], dim=-1)
            if not torch.isfinite(u_term[loss_mask > 0]).all():
                raise FloatingPointError(
                    "non-finite u = log p_real - log p_ref on a valid position "
                    f"(reference={counterfactual_reference}); refusing to silently reshape the target"
                )
            with torch.no_grad():
                # Reference-agnostic diagnostics (candidate S plan §15): how far the reference is
                # from the teacher, and how much teacher mass sits outside the student's top-k.
                _u_abs = u_term.abs()
                cf_u_abs_mean_per_token = _u_abs.mean(dim=-1)
                cf_u_frac_pos_per_token = (u_term > 0).float().mean(dim=-1)
                _u_valid = _u_abs[loss_mask > 0]
                cf_u_p90 = _u_valid.float().quantile(0.9).item() if _u_valid.numel() else 0.0
                del _u_valid
                # always w.r.t. the ACTUAL teacher forward (in S2 the "real" slot holds the student base)
                _p_real = _true_teacher_distill_log_probs.exp()
                teacher_student_kl_per_token = (
                    _p_real * (_true_teacher_distill_log_probs - student_distill_log_probs.detach())
                ).sum(dim=-1)
                if use_topk and self_distillation_config.distillation_add_tail:
                    teacher_tail_mass_per_token = _p_real[..., -1]
                del _p_real
            if _flip:
                with torch.no_grad():
                    _real = teacher_real_distill_log_probs
                    _null = teacher_null_distill_log_probs
                    _a_real = _real.argmax(dim=-1)
                    _a_null = _null.argmax(dim=-1)
                    _has_tail = bool(use_topk and self_distillation_config.distillation_add_tail)
                    _tail_idx = _real.shape[-1] - 1
                    flip_raw = (_a_real != _a_null) & (loss_mask > 0)
                    flip_tail_excl = flip_raw & (_a_real == _tail_idx) if _has_tail else torch.zeros_like(flip_raw)
                    flip_set = flip_raw.clone()
                    if _has_tail and flip_tail_policy == "exclude":
                        flip_set = flip_set & ~flip_tail_excl
                    flip_t0 = flip_raw[:, 0].clone()            # raw t=0 flips (before tail exclusion), reported as flip_t0_raw_sum
                    flip_set[:, 0] = False                      # t = 0: no student prefix -> outside the method's domain
                    if counterfactual_target_mode == "flip":
                        # gamma-sharpen the EXPLICIT columns of p+, keep the tail bucket's mass (same rule as target_gamma)
                        _explicit = torch.ones_like(_real, dtype=torch.bool)
                        if _has_tail:
                            _explicit[..., -1] = False
                        _neg_inf = torch.finfo(_real.dtype).min
                        if bool(_explicit.all()):
                            _sharp = F.log_softmax(flip_gamma * _real, dim=-1)
                        else:
                            _mass = torch.logsumexp(_real.masked_fill(~_explicit, _neg_inf), dim=-1, keepdim=True)
                            _scaled = flip_gamma * _real
                            _z = torch.logsumexp(_scaled.masked_fill(~_explicit, _neg_inf), dim=-1, keepdim=True)
                            _sharp = torch.where(_explicit, _scaled - _z + _mass, _real)
                    else:  # flip_u: the A target, only on F
                        _sharp = F.log_softmax(_real + counterfactual_extrapolation_beta * u_term, dim=-1)
                    teacher_distill_log_probs = torch.where(flip_set.unsqueeze(-1), _sharp, _real).detach()
                    # loss weight 1 + lambda on F, reused through the weighted-aggregation path (05 §14)
                    nh_weight = (1.0 + flip_lambda * flip_set.float()) * loss_mask
                    _Tn = loss_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    _rel = torch.arange(loss_mask.shape[1], device=loss_mask.device, dtype=torch.float32).unsqueeze(0) / (_Tn - 1.0).clamp(min=1.0)
                    flip_pos_rel_sum = (_rel * flip_set.float()).sum().item()
                    flip_count = flip_set.float().sum().item()
                    flip_raw_count = flip_raw.float().sum().item()
                    flip_tail_excl_count = flip_tail_excl.float().sum().item()
                    flip_t0_count = flip_t0.float().sum().item()
                    _q_f = teacher_distill_log_probs.exp()
                    flip_tv_sum = (0.5 * (_q_f - _real.exp()).abs().sum(dim=-1) * flip_set.float()).sum().item()
                    _s_arg = student_distill_log_probs.detach().argmax(dim=-1)
                    flip_student_agree_sum = ((_s_arg == _a_real).float() * flip_set.float()).sum().item()
                    del _q_f, _s_arg
            elif counterfactual_st_enable:
                # ST (selective suppression + conserved transfer), replaces the tilt:
                # donors D = {u < -tau} ∩ Head(a; rho) ∩ Head(s; rho) lose
                # delta = min(eps, alpha_max * mass(D)) of probability mass; recipients
                # {u >= -tau} receive it proportionally to a; everything else (incl.
                # the tail bucket) keeps a. Conserved: TV(q, a) = delta exactly.
                with torch.no_grad():
                    a_prob = teacher_real_distill_log_probs.exp()
                    s_prob = student_distill_log_probs.detach().exp()
                    explicit = torch.ones_like(a_prob, dtype=torch.bool)
                    if use_topk and self_distillation_config.distillation_add_tail:
                        explicit[..., -1] = False
                    a_exp = a_prob.masked_fill(~explicit, 0.0)
                    s_exp = s_prob.masked_fill(~explicit, 0.0)
                    head_a = a_exp >= counterfactual_st_head_ratio * a_exp.amax(dim=-1, keepdim=True)
                    head_s = s_exp >= counterfactual_st_head_ratio * s_exp.amax(dim=-1, keepdim=True)
                    donors = (u_term < -counterfactual_st_tau) & head_a & head_s & explicit
                    recipients = (u_term >= -counterfactual_st_tau) & explicit
                    st_mass_d_per_token = (a_prob * donors).sum(dim=-1)
                    st_mass_r_per_token = (a_prob * recipients).sum(dim=-1)
                    st_valid = (st_mass_d_per_token > 0) & (st_mass_r_per_token > 0)
                    st_delta_per_token = (
                        torch.minimum(
                            torch.full_like(st_mass_d_per_token, counterfactual_st_eps),
                            counterfactual_st_alpha_max * st_mass_d_per_token,
                        )
                        * st_valid.float()
                    )
                    _tiny = 1e-12
                    scale_d = (1.0 - st_delta_per_token / st_mass_d_per_token.clamp_min(_tiny)).unsqueeze(-1)
                    scale_r = (1.0 + st_delta_per_token / st_mass_r_per_token.clamp_min(_tiny)).unsqueeze(-1)
                    q_st = a_prob * torch.where(
                        donors, scale_d, torch.where(recipients, scale_r, torch.ones_like(a_prob))
                    )
                    q_st = q_st / q_st.sum(dim=-1, keepdim=True).clamp_min(_tiny)
                    teacher_distill_log_probs = (q_st + _tiny).log()
                    st_cov_per_token = st_valid.float()
                    st_donor_frac_per_token = donors.float().sum(dim=-1) / explicit.float().sum(dim=-1).clamp(min=1.0)
            else:
                if counterfactual_tanh_scale > 0.0:
                    # Bounded tilt: u <- tau * tanh(u / tau), so exp(beta*u) is confined to
                    # [exp(-beta*tau), exp(+beta*tau)] instead of being unbounded.
                    #
                    # Why: with beta=4 the exponential tilt is dominated by whichever token
                    # happens to have the largest u -- on a six-token worked example a single
                    # u=+1.92 token takes 99.76% of the target mass, making q a near-one-hot
                    # that ignores the rest of the visual contrast. tanh keeps the small-|u|
                    # regime linear (tau*tanh(u/tau) = u + O(u^3/tau^2)) and only compresses
                    # the extremes, i.e. it redistributes the tilt rather than removing it.
                    #
                    # tanh is odd and strictly increasing, so it preserves sign and order of u.
                    # That makes it commute with both downstream gates: the floor's (u < 0)
                    # mask and sup's clamp(max=0) select the same tokens before and after.
                    # Applying it first is therefore purely a presentation choice.
                    with torch.no_grad():
                        # |u| > 2*tau  <=>  tanh has compressed this token by >=3.6%
                        # (tanh(2) = 0.964). NOT the same as saturation: float32 tanh
                        # returns exactly +-1 only around |u| ~ 9*tau, and only there is
                        # the ORDER between extreme tokens destroyed. Read this metric as
                        # "how much of the distribution is in the nonlinear regime", not
                        # "how much is pinned at the bound".
                        tanh_compressed_frac_per_token = (
                            u_term.abs() > 2.0 * counterfactual_tanh_scale
                        ).float().mean(dim=-1)
                    u_term = counterfactual_tanh_scale * torch.tanh(u_term / counterfactual_tanh_scale)
                if counterfactual_floor_alpha > 0.0:
                    # Plausibility floor (Ren ICLR25 anti-squeeze), ported byte-for-byte from
                    # Vision-OPD-OPSA reconstruct_aha_target: negative u is clamped to 0 on the
                    # valley (p_real < alpha * max p_real), so the negative force lands only on
                    # the head. Positive u is untouched everywhere. The tail column belongs to
                    # the valley by construction (same rule), matching the OPSA arm.
                    with torch.no_grad():
                        valley = teacher_real_distill_log_probs < (
                            math.log(counterfactual_floor_alpha)
                            + teacher_real_distill_log_probs.amax(dim=-1, keepdim=True)
                        )
                        clamp_mask = valley & (u_term < 0)
                        floor_clamped_frac_per_token = clamp_mask.float().mean(dim=-1)
                    u_term = torch.where(clamp_mask, torch.zeros_like(u_term), u_term)
                elif counterfactual_u_clip_pos:
                    # Suppression-only tilt: q ∝ p_real * exp(beta * min(u, 0)).
                    # Only the negative half of the visual contrast enters the target;
                    # tokens the real image supports (u > 0) keep their p_real mass and
                    # receive the suppressed mass proportionally via renormalization.
                    with torch.no_grad():
                        sup_frac_u_pos_per_token = (u_term > 0).float().mean(dim=-1)
                    u_term = torch.clamp(u_term, max=0.0)
                if nh_beta_t is not None and counterfactual_hist_half == "pos":
                    # X1 re-grounding: beta_t acts on the POSITIVE half only.
                    _tilt = (
                        nh_beta_t.unsqueeze(-1) * torch.clamp(u_term, min=0.0)
                        + counterfactual_extrapolation_beta * torch.clamp(u_term, max=0.0)
                    )
                elif nh_beta_t is not None:
                    # beta_t acts on the negative half only; the positive half keeps base beta.
                    # With u_clip_pos (arm C) clamp(u,min=0) == 0 and clamp(u,max=0) == u, so this is
                    # bit-identical to the previous beta_t * u_term.
                    _tilt = (
                        counterfactual_extrapolation_beta * torch.clamp(u_term, min=0.0)
                        + nh_beta_t.unsqueeze(-1) * torch.clamp(u_term, max=0.0)
                    )
                else:
                    _tilt = counterfactual_extrapolation_beta * u_term
                if counterfactual_t0_beta_zero:
                    # A-t0 (06 §1.1): no tilt at the first response token (no student prefix there)
                    _t0_mask = torch.ones(1, _tilt.shape[1], 1, device=_tilt.device, dtype=_tilt.dtype)
                    _t0_mask[:, 0, :] = 0.0
                    _tilt = _tilt * _t0_mask
                teacher_distill_log_probs = F.log_softmax(
                    teacher_real_distill_log_probs + _tilt,
                    dim=-1,
                ).detach()  # the target is a constant for the student (value-identical for the null path)
            if counterfactual_target_gamma != 1.0 and counterfactual_st_enable:
                raise ValueError(
                    "counterfactual_target_gamma is incompatible with counterfactual_st_enable "
                    "(ST defines the target's shape by construction; re-sharpening destroys "
                    "its conservation property)"
                )
            if counterfactual_target_gamma != 1.0:
                # Target sharpness knob, applied AFTER every other target-construction
                # branch (tilt / floor / sup / ST) has produced q:
                #
                #     log q_gamma = log_softmax(gamma * log q)
                #
                # gamma = 1 is the exact identity (A arm, bit-identical). Large gamma
                # drives q to onehot(argmax q) -- with the measured q_max ~= 0.98,
                # gamma=50 gives a ratio of ~1e85 between the top token and the runner-up,
                # i.e. numerically a hard label. gamma < 1 flattens q instead.
                #
                # Purpose: q already carries ~98% of its mass on a single token
                # (metric target_max_prob, measured 0.9824 on the A-style target), so the
                # question this knob answers is whether the remaining ~2% of soft mass
                # does any work at all. See docs/gamma_sharpening_plan.md.
                #
                # Safety: log_softmax subtracts the max, so gamma*log q down at -2500 is
                # fine; and JSD against a one-hot q stays bounded by log 2 (no log(0)).
                # Sharpen the EXPLICIT columns only, preserving the tail bucket's mass.
                # The tail is the aggregated mass outside the student's top-k -- a truncation
                # artifact, not a token. Sharpening it as if it were one would, at gamma<1,
                # inflate its share by orders of magnitude (with q ~= [0.93, ..., tail=1e-5],
                # gamma=0.5 lifts tail/argmax from ~1e-5 to ~3e-3), so the "flatter target"
                # arm would be telling the student to move mass OUTSIDE its top-k rather than
                # amplifying the teacher's soft mass -- the exact quantity under test. At
                # large gamma it would make a hard label on "not in the top-k" wherever the
                # tail happens to be the argmax.
                _explicit = torch.ones_like(teacher_distill_log_probs, dtype=torch.bool)
                if use_topk and self_distillation_config.distillation_add_tail:
                    _explicit[..., -1] = False
                _neg_inf = torch.finfo(teacher_distill_log_probs.dtype).min
                if bool(_explicit.all()):
                    teacher_distill_log_probs = F.log_softmax(
                        counterfactual_target_gamma * teacher_distill_log_probs, dim=-1
                    )
                else:
                    _mass = torch.logsumexp(
                        teacher_distill_log_probs.masked_fill(~_explicit, _neg_inf), dim=-1, keepdim=True
                    )
                    _scaled = counterfactual_target_gamma * teacher_distill_log_probs
                    _z = torch.logsumexp(_scaled.masked_fill(~_explicit, _neg_inf), dim=-1, keepdim=True)
                    teacher_distill_log_probs = torch.where(
                        _explicit, _scaled - _z + _mass, teacher_distill_log_probs
                    )
            with torch.no_grad():
                q_prob = teacher_distill_log_probs.exp()
                target_max_prob_per_token = q_prob.amax(dim=-1)
                target_entropy_per_token = -(
                    q_prob * teacher_distill_log_probs.clamp(min=-1e30)
                ).sum(dim=-1)
                if use_topk and self_distillation_config.distillation_add_tail:
                    # tail = the collective mass outside the explicit top-k set, not a token
                    target_tail_mass_per_token = q_prob[..., -1]
                    target_argmax_tail_frac_per_token = (
                        q_prob.argmax(dim=-1) == q_prob.shape[-1] - 1
                    ).float()
            counterfactual_target_tv_per_token = 0.5 * (
                q_prob - teacher_real_distill_log_probs.exp()
            ).abs().sum(dim=-1)
            del q_prob
        elif _internal:
            # Route 1: q ∝ p+ · exp(λ r). The target is a constant for the student (frozen teacher
            # only; r read from the same forward that produced p+). Tail policies, plan §6:
            #   full -> q already built on the full vocab and coarsened to K+tail in dp_actor;
            #   aha  -> coarsen first (add_tail'ed p+), tilt only the explicit top-k, tail r := 0.
            with torch.no_grad():
                K = teacher_internal_r_topk.shape[-1]
                if teacher_real_distill_log_probs.shape[-1] != K + 1:
                    raise ValueError(f"internal_residual: support width {teacher_real_distill_log_probs.shape[-1]} != K+1={K+1}")
                if internal_tail == "full":
                    q_log = teacher_internal_target_log_probs.detach().float()
                    if q_log.shape != teacher_real_distill_log_probs.shape:
                        raise ValueError(f"internal_residual: target shape {tuple(q_log.shape)} != support {tuple(teacher_real_distill_log_probs.shape)}")
                    _norm_err = torch.logsumexp(q_log[loss_mask > 0], dim=-1).abs().max().item() if (loss_mask > 0).any() else 0.0
                    if not torch.isfinite(q_log[loss_mask > 0]).all() or _norm_err > 1e-3:
                        raise FloatingPointError(f"internal_residual: target not finite/normalised (max |logsumexp| = {_norm_err:.3e})")
                else:
                    _tilt_k = internal_lambda * teacher_internal_r_topk.detach().float()
                    _tilt = torch.cat([_tilt_k, torch.zeros_like(_tilt_k[..., :1])], dim=-1)
                    q_log = F.log_softmax(teacher_real_distill_log_probs.float() + _tilt, dim=-1)
                teacher_distill_log_probs = q_log.to(teacher_real_distill_log_probs.dtype).detach()
                q_prob = teacher_distill_log_probs.exp()
                target_max_prob_per_token = q_prob.amax(dim=-1)
                target_entropy_per_token = -(q_prob * teacher_distill_log_probs.clamp(min=-1e30)).sum(dim=-1)
                target_tail_mass_per_token = q_prob[..., -1]
                target_argmax_tail_frac_per_token = (q_prob.argmax(dim=-1) == q_prob.shape[-1] - 1).float()
                counterfactual_target_tv_per_token = 0.5 * (q_prob - teacher_real_distill_log_probs.exp()).abs().sum(dim=-1)
                _r = teacher_internal_r_topk.detach().float()
                internal_r_abs_per_token = _r.abs().mean(dim=-1)
                internal_r_sq_per_token = (_r ** 2).mean(dim=-1)
                del q_prob
        else:
            teacher_distill_log_probs = teacher_real_distill_log_probs

        if self_distillation_config.alpha == 0.0:
            kl_loss = F.kl_div(
                student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True
            )
        elif self_distillation_config.alpha == 1.0:
            kl_loss = F.kl_div(
                teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True
            )
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            alpha = torch.tensor(
                self_distillation_config.alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_distill_log_probs + torch.log(1 - alpha), teacher_distill_log_probs + torch.log(alpha)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha)  # Compute the Generalized Jensen-Shannon Divergence

        raw_per_token_loss = kl_loss.sum(-1)
    else:
        assert self_distillation_config.alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
        log_ratio = student_log_probs - teacher_log_probs
        raw_per_token_loss = log_ratio.detach() * student_log_probs

    weighted_per_token_loss = raw_per_token_loss

    is_clip = self_distillation_config.is_clip
    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")

        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        weighted_per_token_loss = weighted_per_token_loss * ratio

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        weighted_per_token_loss = weighted_per_token_loss * rollout_is_weights
    if _pbd:
        # 12 §3.4 / 13 §19: L = mean_{V0 rows}(JSD) + lambda * mean_{branch rows}(JSD). Rows are token-mean
        # aggregated by the caller over ALL rows (micro-batch = row), so weight V0 rows by
        # scale = N_total/N_v0 and branch rows by lambda*scale (scale passed from the trainer).
        _lam = float(getattr(self_distillation_config, "pbd_lambda", 0.5))
        _isb = pbd_is_branch.to(weighted_per_token_loss.dtype).unsqueeze(1)
        _sb = float(pbd_scale if pbd_scale_b is None else pbd_scale_b)
        # V0 rows: N/N_v0 ; branch rows: lambda * N/N_b  ->  mean over all rows = mean_v0 + lambda * mean_branch
        _w = pbd_scale * (1.0 - _isb) + _lam * _sb * _isb
        # Replace parents keep only t < t*: weight the row by its kept-token fraction so the V0 term stays a
        # token-level average over M^S (12 §3.4) instead of promoting a 10-token prefix to a full row (Keep: 1).
        if pbd_loss_mask is not None:
            _kept = (response_mask * pbd_loss_mask.to(response_mask.dtype)).sum(-1, keepdim=True)
            _tot = response_mask.sum(-1, keepdim=True).clamp(min=1.0)
            _frac = (_kept / _tot).to(weighted_per_token_loss.dtype)
            _w = _w * ((1.0 - _isb) * _frac + _isb)
        weighted_per_token_loss = weighted_per_token_loss * _w

    valid_token_count = loss_mask.sum().clamp(min=1.0)
    if batch_num_tokens is None:
        batch_num_tokens = valid_token_count
    metrics["self_distillation/raw_jsd_token_mean"] = (
        verl_F.masked_sum(raw_per_token_loss, loss_mask) / valid_token_count
    ).detach().item()
    metrics["self_distillation/weighted_jsd_token_mean"] = (
        verl_F.masked_sum(weighted_per_token_loss, loss_mask) / valid_token_count
    ).detach().item()
    if self_distillation_mask is None:
        metrics["self_distillation/self_distillation_mask.mean()"] = 1.0
    else:
        metrics["self_distillation/self_distillation_mask.mean()"] = self_distillation_mask.float().mean().detach().item()
    metrics["self_distillation/num_distill_tokens"] = loss_mask.sum().detach().item()
    if _pbd:
        _isb_row = pbd_is_branch.to(loss_mask.dtype).unsqueeze(1)
        _m_b = loss_mask * _isb_row
        _m_v = loss_mask * (1.0 - _isb_row)
        # tail-8 of each row's loss positions (13 §16.2 J^tail)
        _rev = torch.flip(torch.cumsum(torch.flip(loss_mask, dims=[1]), dim=1), dims=[1])
        _tail = loss_mask * (_rev <= 8).to(loss_mask.dtype)
        for _name, _mm in (("pbd_jsd_branch", _m_b), ("pbd_jsd_v0", _m_v),
                           ("pbd_jsd_tail8_branch", _tail * _isb_row), ("pbd_jsd_tail8_v0", _tail * (1.0 - _isb_row))):
            metrics[f"self_distillation/{_name}_sum"] = verl_F.masked_sum(raw_per_token_loss, _mm).detach().item()
            metrics[f"self_distillation/{_name}_cnt"] = _mm.sum().detach().item()
        metrics["self_distillation/pbd_branch_rows"] = pbd_is_branch.float().sum().detach().item()
        metrics["self_distillation/pbd_rows"] = float(pbd_is_branch.numel())
        metrics["self_distillation/pbd_scale"] = float(pbd_scale)
        metrics["self_distillation/pbd_scale_b"] = float(pbd_scale if pbd_scale_b is None else pbd_scale_b)
    metrics["self_distillation/counterfactual_target_enabled"] = float(counterfactual_null_mode is not None)
    metrics["self_distillation/counterfactual_extrapolation_beta"] = (
        counterfactual_extrapolation_beta
        if (counterfactual_null_mode is not None and counterfactual_target_mode != "flip") else 0.0
    )
    if counterfactual_target_tv_per_token is not None:
        metrics["self_distillation/counterfactual_target_tv"] = (
            verl_F.masked_sum(counterfactual_target_tv_per_token, loss_mask) / valid_token_count
        ).detach().item()
    metrics["self_distillation/reference_is_student"] = float(_ref_student)
    metrics["self_distillation/reference_swap_sides"] = float(_swap_sides)
    metrics["self_distillation/teacher_target_mode_internal"] = float(_internal)
    metrics["self_distillation/counterfactual_target_mode_flip"] = float(counterfactual_target_mode == "flip")
    metrics["self_distillation/counterfactual_target_mode_flip_u"] = float(counterfactual_target_mode == "flip_u")
    metrics["self_distillation/t0_beta_zero"] = float(counterfactual_t0_beta_zero)
    if _flip:
        # count-based (sum) metrics: global rates = sum / num_distill_tokens across micro-batches (no max/min substrings)
        metrics["self_distillation/flip_lambda"] = flip_lambda
        metrics["self_distillation/flip_gamma"] = flip_gamma if counterfactual_target_mode == "flip" else 0.0
        metrics["self_distillation/flip_count_sum"] = flip_count
        metrics["self_distillation/flip_raw_count_sum"] = flip_raw_count
        metrics["self_distillation/flip_tail_excluded_sum"] = flip_tail_excl_count
        metrics["self_distillation/flip_t0_raw_sum"] = flip_t0_count
        metrics["self_distillation/flip_frac"] = flip_count / max(loss_mask.sum().item(), 1.0)
        metrics["self_distillation/flip_pos_rel_sum"] = flip_pos_rel_sum
        metrics["self_distillation/flip_pos_rel_mean"] = flip_pos_rel_sum / max(flip_count, 1.0)
        metrics["self_distillation/flip_target_tv_sum"] = flip_tv_sum
        metrics["self_distillation/flip_target_tv_mean"] = flip_tv_sum / max(flip_count, 1.0)
        metrics["self_distillation/flip_student_agree_sum"] = flip_student_agree_sum
        metrics["self_distillation/flip_student_agree_frac"] = flip_student_agree_sum / max(flip_count, 1.0)
    if _internal:
        metrics["self_distillation/internal_lambda"] = internal_lambda
        metrics["self_distillation/internal_tail_is_full"] = float(internal_tail == "full")
        for _k, _t in (("internal_r_abs", internal_r_abs_per_token), ("internal_r_sq", internal_r_sq_per_token)):
            if _t is not None:
                metrics[f"self_distillation/{_k}_sum"] = verl_F.masked_sum(_t, loss_mask).detach().item()
                metrics[f"self_distillation/{_k}_mean"] = (verl_F.masked_sum(_t, loss_mask) / valid_token_count).detach().item()
    for _k, _t in (
        ("counterfactual_u_abs_mean", cf_u_abs_mean_per_token),
        ("counterfactual_u_frac_pos", cf_u_frac_pos_per_token),
        ("teacher_student_kl", teacher_student_kl_per_token),
        ("teacher_tail_mass", teacher_tail_mass_per_token),
    ):
        if _t is not None:
            metrics[f"self_distillation/{_k}"] = (
                verl_F.masked_sum(_t, loss_mask) / valid_token_count
            ).detach().item()
    if cf_u_p90 is not None:
        metrics["self_distillation/counterfactual_u_p90"] = cf_u_p90
    if sup_frac_u_pos_per_token is not None:
        metrics["self_distillation/sup_frac_u_pos"] = (
            verl_F.masked_sum(sup_frac_u_pos_per_token, loss_mask) / valid_token_count
        ).detach().item()
    if nh_c_mean is not None:
        metrics["self_distillation/nh_c_mean"] = nh_c_mean
        metrics["self_distillation/nh_c_frac_pos"] = nh_c_frac_pos
    if nh_N_last_mean is not None:
        metrics["self_distillation/nh_N_last_mean"] = nh_N_last_mean
        metrics["self_distillation/nh_s_mean"] = nh_s_mean
        metrics["self_distillation/nh_beta_t_mean"] = nh_beta_t_mean
    if nh_s_within_var is not None:
        metrics["self_distillation/nh_s_within_var"] = nh_s_within_var
        metrics["self_distillation/nh_s_rowmean_sq"] = nh_s_rowmean_sq
    if nh_h_mean is not None:
        metrics["self_distillation/nh_h_mean"] = nh_h_mean
        metrics["self_distillation/nh_f_mean"] = nh_f_mean
        metrics["self_distillation/nh_eta_row_frac"] = nh_eta_row_frac
    if nh_w_mean is not None:
        metrics["self_distillation/nh_w_mean"] = nh_w_mean
        metrics["self_distillation/nh_w_p90"] = nh_w_p90
    if tanh_compressed_frac_per_token is not None:
        metrics["self_distillation/tanh_compressed_frac"] = (
            verl_F.masked_sum(tanh_compressed_frac_per_token, loss_mask) / valid_token_count
        ).detach().item()
    # ---- Target-shape statistics (count-based; candidate_s_results §7.6) ----
    # verl's reduce_metrics() picks np.max / np.min for any plain key whose NAME contains "max"/"min"
    # (verl/utils/metric/utils.py). The former keys target_max_prob and target_argmax_tail_frac were
    # therefore max-reduced across micro-batches (= worst-rollout envelopes), not global means.
    # Rules now: (1) no "max"/"min" substring in any plain key; (2) global ratios are reported as a
    # masked SUM plus the shared valid-position COUNT, so that mean(sum)/mean(count) after the
    # equal-weight micro-batch / worker reduction equals the token-weighted global value;
    # (3) the worst micro-batch is a separate, explicitly MAX-aggregated Metric object.
    _n_valid = loss_mask.sum().detach().item()
    _tgt_stats = (
        ("target_entropy", target_entropy_per_token),
        ("target_top1_prob", target_max_prob_per_token),
        ("target_tail_mass", target_tail_mass_per_token),
        ("target_tail_top1", target_argmax_tail_frac_per_token),
    )
    if any(_t is not None for _, _t in _tgt_stats):
        metrics["self_distillation/target_stat_count"] = _n_valid
    for _k, _t in _tgt_stats:
        if _t is None:
            continue
        _msum = verl_F.masked_sum(_t, loss_mask)
        metrics[f"self_distillation/{_k}_sum"] = _msum.detach().item()
        # equal-weight micro-batch mean (kept for dashboards; NOT token-weighted across micro-batches);
        # computed exactly as before so the pre-existing keys stay bit-identical.
        metrics[f"self_distillation/{_k}{'_frac' if _k == 'target_tail_top1' else ''}"] = (
            _msum / valid_token_count
        ).detach().item()
    if target_max_prob_per_token is not None:
        metrics["self_distillation/target_top1_prob_worst_mb"] = Metric(
            aggregation=AggregationType.MAX,
            value=(verl_F.masked_sum(target_max_prob_per_token, loss_mask) / valid_token_count).detach().item(),
        )
    if floor_clamped_frac_per_token is not None:
        metrics["self_distillation/floor_clamped_frac"] = (
            verl_F.masked_sum(floor_clamped_frac_per_token, loss_mask) / valid_token_count
        ).detach().item()
    if st_delta_per_token is not None:
        for _name, _t in (
            ("st_delta_tv", st_delta_per_token),
            ("st_donor_coverage", st_cov_per_token),
            ("st_donor_tokens_frac", st_donor_frac_per_token),
            ("st_mass_donor", st_mass_d_per_token),
            ("st_mass_recipient", st_mass_r_per_token),
        ):
            metrics[f"self_distillation/{_name}"] = (
                verl_F.masked_sum(_t, loss_mask) / valid_token_count
            ).detach().item()

    agg_mask = loss_mask
    agg_num_tokens = batch_num_tokens
    if nh_weight is not None:
        # Weighted JSD (05 §14): L = sum(M w l) / sum(M w). masked_sum multiplies by the
        # mask, so passing M*w as the mask gives the weighted numerator; the denominator
        # must be the WEIGHTED count -- batch_num_tokens was already defaulted to the
        # unweighted loss_mask.sum() above, so override it here.
        # Rescale rather than replace batch_num_tokens: if a GLOBAL token count was passed
        # in, replacing it with a local weighted sum would change the loss scale by dp_size.
        agg_mask = loss_mask * nh_weight
        agg_num_tokens = batch_num_tokens * (
            agg_mask.sum() / loss_mask.sum().clamp(min=1.0)
        ).clamp(min=1e-6)
    loss = agg_loss(
        loss_mat=weighted_per_token_loss,
        loss_mask=agg_mask,
        loss_agg_mode=loss_agg_mode,
        batch_num_tokens=agg_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=loss_scale_factor,
    )
    return loss, metrics


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator,
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+.
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics
