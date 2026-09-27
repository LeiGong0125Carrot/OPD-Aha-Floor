#!/usr/bin/env bash
# Target-sharpness OPD-Aha: log q <- log_softmax(gamma * log q), applied after the
# normal tilt q ∝ p⁺·exp(β·u) has been built.
#
# Single-variable ablation vs A (pair_ahaAn2 / rr2): same pair data, same recipe, same
# seed42, same β=4, **same 2-GPU/SP=2 parallel config**. The only change is how sharp
# the distillation target is.
#
# Why (docs/gamma_sharpening_plan.md): the measured target already puts ~98% of its mass
# on one token (self_distillation/target_max_prob = 0.9824 on the A-style target), so the
# open question is whether the remaining ~2% of soft mass does any work.
#   TARGET_GAMMA=50  -> numerically a hard label at argmax q  (the decisive test)
#   TARGET_GAMMA=1   -> exact identity, i.e. the A arm
#   TARGET_GAMMA=0.5 -> flatter target (the opposite direction)
#
# 用法: TARGET_GAMMA=50 bash scripts/train_pair_gamma.sh [hydra override...]
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-pair_gam${TARGET_GAMMA:-50}_6karmA}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet}"

export TEACHER_MODEL_SOURCE=legacy TEACHER_REGULARIZATION=frozen TEACHER_UPDATE_RATE=0.0
export COUNTERFACTUAL_NULL_MODE=mean_color COUNTERFACTUAL_EXTRAPOLATION_BETA="${BETA:-4.0}"
export ALPHA=0.5 LR="${LR:-2e-6}" MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=1024 DATA_SEED=42
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}" PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-48}" ROLLOUT_N="${ROLLOUT_N:-2}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-2}" TRAINER_NNODES=1
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-5}" TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-51}"
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export ACTOR_USE_DYNAMIC_BSZ=False

echo "${EXPERIMENT_NAME_OVERRIDE}: target sharpness gamma=${TARGET_GAMMA:-50}, β=${COUNTERFACTUAL_EXTRAPOLATION_BETA} (seed42, pair)"
echo "  并行配置: n_gpus=${TRAINER_N_GPUS_PER_NODE} ulysses_sp=${ULYSSES_SP:-2} rollout_n=${ROLLOUT_N} lr=${LR}  <- 复现跑必须逐项相同"
exec "${PROJECT_ROOT}/scripts/run_visual_counterfactual_unit.sh" \
    actor_rollout_ref.actor.self_distillation.counterfactual_null_scope="${NULL_SCOPE:-last}" \
    actor_rollout_ref.actor.self_distillation.counterfactual_target_gamma="${TARGET_GAMMA:-50}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP:-2}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP:-2}" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "$@"
