#!/usr/bin/env bash
# TSW round 1 (teacher_stability/tsw_initial_training_plan.md §4.3): question-level distillation weights on the off6k V0 base.
#   TSW_ARM=V0 -> off6k_TSWv0n2  (equal weights; tsw off, same parquet)
#   TSW_ARM=PS -> off6k_TSWpsn2  (probe-lowest 25%, lambda 0.5)
#   TSW_ARM=PH -> off6k_TSWphn2  (probe-lowest 25%, lambda 0)
#   TSW_ARM=RS -> off6k_TSWrsn2  (fixed random 25%, lambda 0.5)
#   TSW_ARM=RH -> off6k_TSWrhn2  (same random 25%, lambda 0)
# Weights are normalised to row-mean 1 over each step batch in the trainer. seed 42, rollout_n 2, lr 2e-6, 130 steps, save 10.
# 用法: TSW_ARM=V0|PS|PH|RS|RH [RUN_SUFFIX=r2] bash scripts/train_off6k_tsw.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TSW_ARM="${TSW_ARM:?TSW_ARM=V0|PS|PH|RS|RH}"
case "$TSW_ARM" in
  V0) EN=False; KEY=tsw_low25;  LAM=1.0;;
  PS) EN=True;  KEY=tsw_low25;  LAM=0.5;;
  PH) EN=True;  KEY=tsw_low25;  LAM=0.0;;
  RS) EN=True;  KEY=tsw_rand25; LAM=0.5;;
  RH) EN=True;  KEY=tsw_rand25; LAM=0.0;;
  *) echo "TSW_ARM must be V0|PS|PH|RS|RH"; exit 1;;
esac
export OFF_MODE=V0 ROLLOUT_N="${ROLLOUT_N:-2}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/scratch/nkw3mr/Vision-OPD/data/train_tsw.parquet}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-off6k_TSW$(echo "$TSW_ARM" | tr 'A-Z' 'a-z')n2${RUN_SUFFIX:-}}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-130}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP=null   # user 10-24: keep every checkpoint on this line
[ -s "$TASK_TRAIN_FILE" ] || { echo "missing $TASK_TRAIN_FILE (scripts/build_tsw_parquet.py)"; exit 1; }
echo "TSW arm=${TSW_ARM}: tsw_enable=${EN} key=${KEY} lambda=${LAM} data=$(basename "$TASK_TRAIN_FILE")"
exec "${PROJECT_ROOT}/scripts/train_official6k.sh" \
    actor_rollout_ref.actor.self_distillation.tsw_enable="${EN}" \
    actor_rollout_ref.actor.self_distillation.tsw_key="${KEY}" \
    actor_rollout_ref.actor.self_distillation.tsw_lambda="${LAM}" \
    "$@"
