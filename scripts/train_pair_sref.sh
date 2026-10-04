#!/usr/bin/env bash
# Candidate S: OPD-Aha with the STUDENT's own distribution as the reference -- no visual-null
# forward anywhere.  u_S = log p+ - log sg(p_S),  q ∝ p+ · exp(β·u_S),  β=4, everything else = A.
#   docs/negative_history/candidate_s_null_free_implementation_plan.md (v0.1)
# counterfactual_null_mode stays mean_color because it gates target reconstruction; with
# counterfactual_reference=student ray_trainer builds NO null view and dp_actor runs NO null forward.
# VOPD_FORBID_NULL=1 makes any null construction/forward raise -- kept on for every S run as an audit.
# First round: single variable (no hist/future/sup/floor/ST/tanh/gamma).
# 用法: [RUN_SUFFIX=r2] bash scripts/train_pair_sref.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-pair_S${RUN_SUFFIX:-}_6karmA}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet}"

export TEACHER_MODEL_SOURCE=legacy TEACHER_REGULARIZATION=frozen TEACHER_UPDATE_RATE=0.0
export COUNTERFACTUAL_NULL_MODE=mean_color COUNTERFACTUAL_EXTRAPOLATION_BETA="${AHA_BETA:-4.0}"
export ALPHA=0.5 LR="${LR:-2e-6}" MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=1024 DATA_SEED=42
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}" PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-48}" ROLLOUT_N="${ROLLOUT_N:-2}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-3}" TRAINER_NNODES=1
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-5}" TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-51}"
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export ACTOR_USE_DYNAMIC_BSZ=False
export VOPD_FORBID_NULL="${VOPD_FORBID_NULL:-1}"

echo "${EXPERIMENT_NAME_OVERRIDE}: candidate S -- A tilt with STUDENT reference (no null forward), β=${COUNTERFACTUAL_EXTRAPOLATION_BETA} (seed42, pair), VOPD_FORBID_NULL=${VOPD_FORBID_NULL}"
echo "  并行配置: n_gpus=${TRAINER_N_GPUS_PER_NODE} ulysses_sp=${ULYSSES_SP:-1} rollout_n=${ROLLOUT_N} lr=${LR}  <- 复现跑必须逐项相同"
exec "${PROJECT_ROOT}/scripts/run_visual_counterfactual_unit.sh" \
    actor_rollout_ref.actor.self_distillation.counterfactual_null_scope=last \
    actor_rollout_ref.actor.self_distillation.counterfactual_reference=student \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "$@"
