#!/usr/bin/env bash
# P1 Counterfactual-Flip Distillation (CFD) arms, docs/proposals_2026-10-06/01 + 06.
#   CFD_MODE=flip    -> target = gamma-sharpened p+ on the flip set, p+ elsewhere; loss weight 1+lambda on F   (tag pair_CFD)
#   CFD_MODE=flip_u  -> A's target only on the flip set, p+ elsewhere, lambda=0                                   (tag pair_CFDu)
#   CFD_MODE=flip    with CFD_LAMBDA=0                                                                              (tag pair_CFDl0)
#   CFD_MODE=t0      -> A with beta=0 at the first response token (A-t0 control)                                    (tag pair_At0)
# Everything else = A (pair view, mean-colour null, beta=4, JSD 0.5, top-100+tail, frozen teacher, seed 42, n=2, 3 GPU/SP1).
# 用法: CFD_MODE=flip|flip_u|t0 [CFD_GAMMA=50] [CFD_LAMBDA=1] [RUN_SUFFIX=r2] bash scripts/train_pair_cfd.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFD_MODE="${CFD_MODE:-flip}"; CFD_GAMMA="${CFD_GAMMA:-50.0}"; CFD_LAMBDA="${CFD_LAMBDA:-1.0}"
case "$CFD_MODE" in
  flip)   TAG=CFD; if awk "BEGIN{exit !($CFD_LAMBDA==0)}"; then TAG=CFDl0; fi; TM=flip; T0=False; LAM=$CFD_LAMBDA ;;
  flip_u) TAG=CFDu; TM=flip_u; T0=False; LAM=0.0 ;;
  t0)     TAG=At0;  TM=tilt;   T0=True;  LAM=0.0 ;;
  *) echo "CFD_MODE must be flip|flip_u|t0"; exit 1 ;;
esac
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
echo "${EXPERIMENT_NAME_OVERRIDE}: CFD mode=${CFD_MODE} target_mode=${TM} gamma=${CFD_GAMMA} lambda=${LAM} t0_beta_zero=${T0} (A otherwise; seed42, pair, n=${ROLLOUT_N})"
echo "  并行配置: n_gpus=${TRAINER_N_GPUS_PER_NODE} ulysses_sp=${ULYSSES_SP:-1} rollout_n=${ROLLOUT_N} lr=${LR}  <- 复现跑必须逐项相同"
exec "${PROJECT_ROOT}/scripts/run_visual_counterfactual_unit.sh" \
    actor_rollout_ref.actor.self_distillation.counterfactual_null_scope=last \
    actor_rollout_ref.actor.self_distillation.counterfactual_target_mode="${TM}" \
    actor_rollout_ref.actor.self_distillation.flip_gamma="${CFD_GAMMA}" \
    actor_rollout_ref.actor.self_distillation.flip_lambda="${LAM}" \
    actor_rollout_ref.actor.self_distillation.flip_tail_policy=exclude \
    actor_rollout_ref.actor.self_distillation.counterfactual_t0_beta_zero="${T0}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "$@"
