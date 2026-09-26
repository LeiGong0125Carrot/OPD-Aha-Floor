#!/usr/bin/env bash
# Bounded-tilt OPD-Aha: q ∝ p⁺ · exp(β · τ·tanh(u/τ)), u = log p⁺ − log p⁰.
#
# Single-variable ablation vs A (pair_ahaA, plain exp(β·u)): same pair data, same
# recipe, same seed42, same β=4. The only change is that u is squashed through
# τ·tanh(·/τ) before the exponential, so the tilt factor lives in
# [e^{-βτ}, e^{+βτ}] instead of being unbounded.
#
# Motivation (docs/trajectory_measurements_verdict.md §五): with β=4 the target is
# frequently a near-one-hot decided by a single extreme u — on a six-token worked
# example one u=+1.92 token takes 99.76% of the target mass (TV=0.83). tanh keeps
# the small-|u| regime linear and only compresses the extremes, i.e. it redistributes
# the tilt across tokens rather than weakening it. Direction = "less extreme, more
# diffuse", which is what the ST arm taught us suppression needs.
#
# τ=0.5 with β=4 gives a tilt range [e^{-2}, e^{2}] = [0.135, 7.39].
# 用法: TANH_SCALE=0.5 bash scripts/train_pair_tanh.sh [hydra override...]
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-pair_tanhn2_6karmA}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet}"

export TEACHER_MODEL_SOURCE=legacy TEACHER_REGULARIZATION=frozen TEACHER_UPDATE_RATE=0.0
export COUNTERFACTUAL_NULL_MODE=mean_color COUNTERFACTUAL_EXTRAPOLATION_BETA="${TANH_BETA:-4.0}"
export ALPHA=0.5 LR="${LR:-2e-6}" MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=1024 DATA_SEED=42
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}" PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-48}" ROLLOUT_N="${ROLLOUT_N:-2}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-3}" TRAINER_NNODES=1
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-5}" TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-51}"
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export ACTOR_USE_DYNAMIC_BSZ=False

echo "${EXPERIMENT_NAME_OVERRIDE}: bounded tilt u<-tau*tanh(u/tau), tau=${TANH_SCALE:-0.5}, β=${COUNTERFACTUAL_EXTRAPOLATION_BETA} (seed42, pair)"
echo "  并行配置: n_gpus=${TRAINER_N_GPUS_PER_NODE} ulysses_sp=${ULYSSES_SP:-1} rollout_n=${ROLLOUT_N} lr=${LR}  <- 复现跑必须逐项相同"
exec "${PROJECT_ROOT}/scripts/run_visual_counterfactual_unit.sh" \
    actor_rollout_ref.actor.self_distillation.counterfactual_null_scope="${NULL_SCOPE:-last}" \
    actor_rollout_ref.actor.self_distillation.counterfactual_tanh_scale="${TANH_SCALE:-0.5}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "$@"
