#!/usr/bin/env bash
# A + history-adaptive suppression (mean-history): full OPD-Aha tilt, but the NEGATIVE half
# is scaled by a per-token beta_t from the prefix-mean realized visual conflict:
#     q ∝ p⁺ · exp(β·u⁺ + β_t·u⁻),   β_t = β(1 + s_t),   s_t = N̄_t / (N̄_t + κ)
# N̄_t = mean of c_k = [−u_k(y_k)]₊ over the valid prefix k<t; κ ≈ measured mean c (0.10),
# so an average prefix gives s=0.5 (β_t≈6) -- dose-matched to arm C (β_t≈6.3), isolating
# the temporal variation. Positive half keeps β=4, i.e. the A base is untouched.
# Rationale: docs/negative_history/nhC_r1_casestudy.md (cumsum history saturates -> dose only;
# sup base itself costs ~1.5 TB vs A).
# HIST_MODE=cumsum gives "A + arm-C-style history" for comparison.
# 用法: [HIST_MODE=mean] [HIST_KAPPA=0.10] [RUN_SUFFIX=r2] bash scripts/train_pair_ahist.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIST_MODE="${HIST_MODE:-mean}"; HIST_KAPPA="${HIST_KAPPA:-0.10}"
HIST_HALF="${HIST_HALF:-neg}"
TAG=Ahm; [ "$HIST_MODE" = cumsum ] && TAG=Ahc; [ "$HIST_MODE" = hf ] && TAG=Ahf
[ "$HIST_HALF" = pos ] && TAG="X1${TAG#Ah}"     # X1m / X1f / X1c: history-triggered re-grounding

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-pair_${TAG}${RUN_SUFFIX:-}_6karmA}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet}"

export TEACHER_MODEL_SOURCE=legacy TEACHER_REGULARIZATION=frozen TEACHER_UPDATE_RATE=0.0
export COUNTERFACTUAL_NULL_MODE=mean_color COUNTERFACTUAL_EXTRAPOLATION_BETA="${AHA_BETA:-4.0}"
export ALPHA=0.5 LR="${LR:-2e-6}" MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=1024 DATA_SEED=42
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}" PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-48}" ROLLOUT_N="${ROLLOUT_N:-2}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-3}" TRAINER_NNODES=1
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-5}" TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-51}"
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export ACTOR_USE_DYNAMIC_BSZ=False

echo "${EXPERIMENT_NAME_OVERRIDE}: A + history-adaptive u⁻ (mode=${HIST_MODE}, κ=${HIST_KAPPA}, half=${HIST_HALF}) β=${COUNTERFACTUAL_EXTRAPOLATION_BETA} (seed42, pair)"
echo "  并行配置: n_gpus=${TRAINER_N_GPUS_PER_NODE} ulysses_sp=${ULYSSES_SP:-1} rollout_n=${ROLLOUT_N} lr=${LR}  <- 复现跑必须逐项相同"
exec "${PROJECT_ROOT}/scripts/run_visual_counterfactual_unit.sh" \
    actor_rollout_ref.actor.self_distillation.counterfactual_null_scope=last \
    actor_rollout_ref.actor.self_distillation.counterfactual_hist_adaptive_beta=True \
    actor_rollout_ref.actor.self_distillation.counterfactual_hist_mode="${HIST_MODE}" \
    actor_rollout_ref.actor.self_distillation.counterfactual_hist_kappa="${HIST_KAPPA}" \
    actor_rollout_ref.actor.self_distillation.counterfactual_hist_half="${HIST_HALF}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "$@"
