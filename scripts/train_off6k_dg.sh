#!/usr/bin/env bash
# DG (OPD-Aha-Floor teacher_stability/disagreement_gate_plan.md v2) on the off6k V0 base:
#   DG_ARM=DG -> off6k_DGn2  (per step the top 25% prompts by g = TV(p_S0, p_T0) get one of n=2 rollouts replaced by a
#                             fresh teacher-view generation of the rollout engine)
#   DG_ARM=DR -> off6k_DRn2  (same number of prompts, chosen at random)
# seed 42, rollout_n 2, lr 2e-6, 130 steps, save 10, all checkpoints kept (as TSW).
# 用法: DG_ARM=DG|DR [RUN_SUFFIX=r2] [DG_FRAC=0.25] bash scripts/train_off6k_dg.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DG_ARM="${DG_ARM:?DG_ARM=DG|DR}"
case "$DG_ARM" in
  DG) MODE=gate;;
  DR) MODE=random;;
  *) echo "DG_ARM must be DG|DR"; exit 1;;
esac
export OFF_MODE=V0 ROLLOUT_N="${ROLLOUT_N:-2}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/scratch/nkw3mr/Vision-OPD/data/train_dg.parquet}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-off6k_${DG_ARM}n2${RUN_SUFFIX:-}}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-130}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP=null
[ -s "$TASK_TRAIN_FILE" ] || { echo "missing $TASK_TRAIN_FILE (scripts/build_dg_parquet.py)"; exit 1; }
echo "DG arm=${DG_ARM}: dg_mode=${MODE} dg_frac=${DG_FRAC:-0.25} data=$(basename "$TASK_TRAIN_FILE")"
exec "${PROJECT_ROOT}/scripts/train_official6k.sh" \
    actor_rollout_ref.actor.self_distillation.dg_enable=True \
    actor_rollout_ref.actor.self_distillation.dg_mode="${MODE}" \
    actor_rollout_ref.actor.self_distillation.dg_frac="${DG_FRAC:-0.25}" \
    actor_rollout_ref.actor.self_distillation.dg_topk=20 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/scripts/pbd_agent_loop.yaml" \
    "$@"
