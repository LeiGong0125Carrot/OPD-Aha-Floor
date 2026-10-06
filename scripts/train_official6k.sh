#!/usr/bin/env bash
# Official-recipe replication on the FULL Vision-OPD-6K (6241 q, official student/teacher views, null_scope=all):
#   OFF_MODE=A   -> OPD-Aha as in scripts/train_beta4.sh (beta=4, mean_color null)                 tag off6k_A
#   OFF_MODE=Ahm -> + history-adaptive beta on the negative half (hist_mode=mean, kappa=0.10)      tag off6k_Ahm
#   OFF_MODE=Ahf -> + history+future-adaptive beta on the negative half (hist_mode=hf, kappa=0.10) tag off6k_Ahf
# Hardware adaptation (3 x RTX Pro 6000, driver cgroup 1T): TRAIN_BATCH_SIZE 48 (paper 96), ROLLOUT_N default 8 (paper 8),
# steps default 130 (~1 epoch at batch 48; paper 70 steps x 96 = 1.08 epoch), save every 10. Override with env.
# 用法: OFF_MODE=A|Ahm|Ahf [ROLLOUT_N=8] [TRAINER_TOTAL_TRAINING_STEPS=130] [RUN_SUFFIX=r2] bash scripts/train_official6k.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFF_MODE="${OFF_MODE:-A}"
case "$OFF_MODE" in A) TAG=off6k_A; HIST=False; HMODE=mean;; Ahm) TAG=off6k_Ahm; HIST=True; HMODE=mean;; Ahf) TAG=off6k_Ahf; HIST=True; HMODE=hf;; *) echo "OFF_MODE must be A|Ahm|Ahf"; exit 1;; esac
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-${TAG}${RUN_SUFFIX:-}}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/scratch/nkw3mr/Vision-OPD/data/train.parquet}"
export TEACHER_MODEL_SOURCE=legacy TEACHER_REGULARIZATION=frozen TEACHER_UPDATE_RATE=0.0
export COUNTERFACTUAL_NULL_MODE=mean_color COUNTERFACTUAL_EXTRAPOLATION_BETA="${AHA_BETA:-4.0}"
export ALPHA=0.5 LR="${LR:-2e-6}" MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=1024 DATA_SEED=42
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}" PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-48}" ROLLOUT_N="${ROLLOUT_N:-8}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-3}" TRAINER_NNODES=1
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}" TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-2}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-130}"
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export ACTOR_USE_DYNAMIC_BSZ=False
echo "${EXPERIMENT_NAME_OVERRIDE}: official Vision-OPD-6K recipe, mode=${OFF_MODE} (hist=${HIST}, hist_mode=${HMODE}), beta=${COUNTERFACTUAL_EXTRAPOLATION_BETA}, null_scope=all, batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} steps=${TRAINER_TOTAL_TRAINING_STEPS} save=${TRAINER_SAVE_FREQ}"
echo "  并行配置: n_gpus=${TRAINER_N_GPUS_PER_NODE} ulysses_sp=${ULYSSES_SP:-1} rollout_n=${ROLLOUT_N} lr=${LR}"
exec "${PROJECT_ROOT}/scripts/run_visual_counterfactual_unit.sh" \
    actor_rollout_ref.actor.self_distillation.counterfactual_null_scope=all \
    actor_rollout_ref.actor.self_distillation.counterfactual_hist_adaptive_beta="${HIST}" \
    actor_rollout_ref.actor.self_distillation.counterfactual_hist_mode="${HMODE}" \
    actor_rollout_ref.actor.self_distillation.counterfactual_hist_kappa=0.10 \
    actor_rollout_ref.actor.self_distillation.counterfactual_hist_half=neg \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "$@"
