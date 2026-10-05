#!/usr/bin/env bash
# Arm Amis: OPD-Aha (A) with a MISMATCHED-CROP visual null -- null view = [full image, crop of another prompt
# in the same batch] instead of [full image, mean-colour block]. Same beta / JSD / support / frozen teacher as A.
# Rationale: cancel the "a real zoom image is present" style channel of u, keep the crop-content evidence channel
# (docs/negative_history/opd_aha_mechanism_conclusion_2026-10-05.md, mismatch plan). verl/utils/mismatch_null.py.
# Options:  MIS_SUP=1   -> suppression-only (counterfactual_u_clip_pos=True), tag pair_Amiss*
#           MIS_VIEWS=2 -> two mismatched nulls averaged in log space,        tag pair_Amis2v*
# 用法: [MIS_SUP=0|1] [MIS_VIEWS=1|2] [RUN_SUFFIX=r2] bash scripts/train_pair_mis.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIS_SUP="${MIS_SUP:-0}"; MIS_VIEWS="${MIS_VIEWS:-1}"
TAG=Amis; [ "$MIS_SUP" = "1" ] && TAG=Amiss; [ "$MIS_VIEWS" = "2" ] && TAG="${TAG}2v"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-pair_${TAG}${RUN_SUFFIX:-}_6karmA}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet}"
export TEACHER_MODEL_SOURCE=legacy TEACHER_REGULARIZATION=frozen TEACHER_UPDATE_RATE=0.0
export COUNTERFACTUAL_NULL_MODE=mismatch_crop COUNTERFACTUAL_EXTRAPOLATION_BETA="${AHA_BETA:-4.0}"
export ALPHA=0.5 LR="${LR:-2e-6}" MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=1024 DATA_SEED=42
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}" PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-48}" ROLLOUT_N="${ROLLOUT_N:-2}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-3}" TRAINER_NNODES=1
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-5}" TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-51}"
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export ACTOR_USE_DYNAMIC_BSZ=False
SUP_FLAG=False; [ "$MIS_SUP" = "1" ] && SUP_FLAG=True
echo "${EXPERIMENT_NAME_OVERRIDE}: Amis -- A with mismatched-crop null (β=${COUNTERFACTUAL_EXTRAPOLATION_BETA}, sup_only=${SUP_FLAG}, null_views=${MIS_VIEWS}; seed42, pair, n=${ROLLOUT_N})"
echo "  并行配置: n_gpus=${TRAINER_N_GPUS_PER_NODE} ulysses_sp=${ULYSSES_SP:-1} rollout_n=${ROLLOUT_N} lr=${LR}  <- 复现跑必须逐项相同"
exec "${PROJECT_ROOT}/scripts/run_visual_counterfactual_unit.sh" \
    actor_rollout_ref.actor.self_distillation.counterfactual_null_scope=last \
    actor_rollout_ref.actor.self_distillation.counterfactual_null_num_views="${MIS_VIEWS}" \
    actor_rollout_ref.actor.self_distillation.counterfactual_u_clip_pos="${SUP_FLAG}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "$@"
