#!/usr/bin/env bash
# P5 PBD -- privileged-view branch distillation on the V0 base (docs/proposals_2026-10-06/12, 13 §20).
#   PBD_MODE=keep    -> tag pair_PBDk_6karmA  (main arm: parent rows keep their full loss mask)
#   PBD_MODE=replace -> tag pair_PBDr_6karmA  (control: parent rows lose t >= t*)
# Frozen rows: ratio 10% (snapped to word boundary), coverage 50% random, lambda 0.5, continuation cap 256,
# y^T generated in the rollout stage by the current weights under the crop view (no hint), scoring teacher =
# frozen initial model with the usual teacher_prompt. No null forward (V0). Leak words monitored, not masked.
# 用法: PBD_MODE=keep [PBD_RATIO=0.10] [PBD_COVERAGE=0.5] [RUN_SUFFIX=r2] bash scripts/train_pair_pbd.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PBD_MODE="${PBD_MODE:-keep}"
PBD_VIEW="${PBD_VIEW:-hide}"   # hide: ONE hide image as the privileged view (train_6k_armA_hide.parquet, teacher_prompt '<image>\n{q}', no hint)
                               # crop: pair data, continuation under [full][crop] (crop_append); scoring teacher keeps teacher_prompt
case "$PBD_MODE" in keep) T=PBDk;; replace) T=PBDr;; *) echo "PBD_MODE must be keep|replace"; exit 1;; esac
case "$PBD_VIEW" in
  hide) T="${T}H"; DEF_FILE=/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6k_armA_hide.parquet; CONT_VIEW=teacher_prompt;;
  crop) DEF_FILE=/sfs/weka/scratch/nkw3mr/Vision-OPD-OPSA/data/TreeVGR-RL-37K/train_6karmA_pair.parquet; CONT_VIEW=crop_append;;
  *) echo "PBD_VIEW must be hide|crop"; exit 1;;
esac
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-pair_${T}${RUN_SUFFIX:-}_6karmA}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-$DEF_FILE}"
export TEACHER_MODEL_SOURCE=legacy TEACHER_REGULARIZATION=frozen TEACHER_UPDATE_RATE=0.0
export COUNTERFACTUAL_NULL_MODE=null COUNTERFACTUAL_EXTRAPOLATION_BETA=0.0
export ALPHA=0.5 LR="${LR:-2e-6}" MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=1024 DATA_SEED=42
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}" PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-48}" ROLLOUT_N="${ROLLOUT_N:-2}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-3}" TRAINER_NNODES=1
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-5}" TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-51}"
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export VOPD_FORBID_NULL=1   # V0 base: any null-view construction is a bug
export ACTOR_USE_DYNAMIC_BSZ=False
PBD_RATIO="${PBD_RATIO:-0.10}"; PBD_COVERAGE="${PBD_COVERAGE:-0.5}"; PBD_LAMBDA="${PBD_LAMBDA:-0.5}"; PBD_MAXCONT="${PBD_MAXCONT:-256}"; PBD_LEAK_MASK="${PBD_LEAK_MASK:-False}"
echo "${EXPERIMENT_NAME_OVERRIDE}: PBD view=${PBD_VIEW} (cont_view=${CONT_VIEW}, data=$(basename $TASK_TRAIN_FILE)) mode=${PBD_MODE} ratio=${PBD_RATIO} coverage=${PBD_COVERAGE} lambda=${PBD_LAMBDA} maxcont=${PBD_MAXCONT} leak_mask=${PBD_LEAK_MASK} (V0 base, no null; seed42, pair)"
echo "  并行配置: n_gpus=${TRAINER_N_GPUS_PER_NODE} ulysses_sp=${ULYSSES_SP:-1} rollout_n=${ROLLOUT_N} lr=${LR}"
exec "${PROJECT_ROOT}/scripts/run_visual_counterfactual_unit.sh" \
    actor_rollout_ref.actor.self_distillation.pbd_enable=True \
    actor_rollout_ref.actor.self_distillation.pbd_mode="${PBD_MODE}" \
    actor_rollout_ref.actor.self_distillation.pbd_ratio="${PBD_RATIO}" \
    actor_rollout_ref.actor.self_distillation.pbd_coverage="${PBD_COVERAGE}" \
    actor_rollout_ref.actor.self_distillation.pbd_lambda="${PBD_LAMBDA}" \
    actor_rollout_ref.actor.self_distillation.pbd_max_cont_len="${PBD_MAXCONT}" \
    actor_rollout_ref.actor.self_distillation.pbd_leak_mask="${PBD_LEAK_MASK}" \
    actor_rollout_ref.actor.self_distillation.pbd_cont_view="${CONT_VIEW}" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/scripts/pbd_agent_loop.yaml" \
    +actor_rollout_ref.rollout.limit_images=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "$@"
