#!/usr/bin/env bash
# Route 1 (IR): internal residual reconstruction from a SINGLE frozen-teacher forward.
#   q ∝ p+ · exp(λ·r),  r = τ⁻¹ W_out D_t (h^b − h^a) read out through the teacher's own final norm + head.
#   docs/negative_history/teacher_internal_residual_reconstruction_plan.md
# Teacher sees the privileged pair view [full image, crop] (same data as A); NO null forward
# (counterfactual_null_mode is unset; VOPD_FORBID_NULL=1 audits it). Student unchanged.
# Pre-registered (10-05): [a,b) = [12,20) (mid-depth, contains full-attention blocks 15 & 19), λ = 1.0.
# Tail policy: IR_TAIL=full (full-vocab reconstruct then coarsen, tag pair_IRf*) or
#              IR_TAIL=aha  (coarsen first, tail r := 0, tag pair_IRa*).
# 用法: [IR_TAIL=full|aha] [IR_A=12 IR_B=20 IR_LAMBDA=1.0] [RUN_SUFFIX=r2] bash scripts/train_pair_ir.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IR_TAIL="${IR_TAIL:-full}"; IR_A="${IR_A:-12}"; IR_B="${IR_B:-20}"; IR_LAMBDA="${IR_LAMBDA:-1.0}"
case "$IR_TAIL" in full) TAG=IRf;; aha) TAG=IRa;; *) echo "IR_TAIL must be full|aha"; exit 1;; esac
# tag carries non-default λ / interval so runs never share a checkpoint dir (resume_mode=auto)
[ "$IR_LAMBDA" != "1.0" ] && TAG="${TAG}l$(echo "$IR_LAMBDA" | tr -d '.')"
[ "$IR_A$IR_B" != "1220" ] && TAG="${TAG}b${IR_A}_${IR_B}"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-pair_${TAG}${RUN_SUFFIX:-}_6karmA}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet}"

export TEACHER_MODEL_SOURCE=legacy TEACHER_REGULARIZATION=frozen TEACHER_UPDATE_RATE=0.0
export COUNTERFACTUAL_NULL_MODE=null COUNTERFACTUAL_EXTRAPOLATION_BETA=0.0   # beta unused in IR mode
export ALPHA=0.5 LR="${LR:-2e-6}" MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=1024 DATA_SEED=42
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}" PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-48}" ROLLOUT_N="${ROLLOUT_N:-2}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-3}" TRAINER_NNODES=1
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-5}" TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-51}"
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export ACTOR_USE_DYNAMIC_BSZ=False
export VOPD_FORBID_NULL="${VOPD_FORBID_NULL:-1}"

echo "${EXPERIMENT_NAME_OVERRIDE}: IR -- internal residual [a,b)=[${IR_A},${IR_B}) λ=${IR_LAMBDA} tail=${IR_TAIL} (single frozen-teacher forward, no null; seed42, pair), VOPD_FORBID_NULL=${VOPD_FORBID_NULL}"
echo "  并行配置: n_gpus=${TRAINER_N_GPUS_PER_NODE} ulysses_sp=${ULYSSES_SP:-1} rollout_n=${ROLLOUT_N} lr=${LR}  <- 复现跑必须逐项相同"
exec "${PROJECT_ROOT}/scripts/run_visual_counterfactual_unit.sh" \
    actor_rollout_ref.actor.self_distillation.teacher_target_mode=internal_residual \
    actor_rollout_ref.actor.self_distillation.teacher_internal.start_block="${IR_A}" \
    actor_rollout_ref.actor.self_distillation.teacher_internal.end_block_exclusive="${IR_B}" \
    actor_rollout_ref.actor.self_distillation.teacher_internal.strength="${IR_LAMBDA}" \
    actor_rollout_ref.actor.self_distillation.teacher_internal.tail_policy="${IR_TAIL}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "$@"
