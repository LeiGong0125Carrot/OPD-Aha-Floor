#!/usr/bin/env bash
# OPRD-Vanilla (arXiv 2606.06021, paper-exact `main` branch) on the off6k V0 base (user 10-26): the student (red-box full
# image + hint) is trained to match the frozen base teacher's (crop view) hidden states at its own response tokens --
# plain MSE, all transformer layers, last 2000 response tokens (= every response token here, R = 1024), rep-only
# (the V0 JSD is still computed and logged, not trained). verl/utils/oprd.py.
#   tag off6k_OPRDvn2 ; seed 42, rollout_n 2, lr 2e-6 (project default; the paper uses 1e-5), 130 steps, save 10.
# 用法: [RUN_SUFFIX=..] [OPRD_ONLY=True] [OPRD_LAYERS=all] [OPRD_POSITIONS=last_k] bash scripts/train_off6k_oprd.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OFF_MODE=V0 ROLLOUT_N="${ROLLOUT_N:-2}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/scratch/nkw3mr/Vision-OPD/data/train.parquet}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-off6k_OPRDvn2${RUN_SUFFIX:-}}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-130}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP=null
echo "OPRD-Vanilla: only=${OPRD_ONLY:-True} layers=${OPRD_LAYERS:-all} positions=${OPRD_POSITIONS:-last_k}/${OPRD_LAST_K:-2000} coef=${OPRD_COEF:-1.0} normalize=False"
exec "${PROJECT_ROOT}/scripts/train_official6k.sh" \
    actor_rollout_ref.actor.self_distillation.oprd_enable=True \
    actor_rollout_ref.actor.self_distillation.oprd_only="${OPRD_ONLY:-True}" \
    actor_rollout_ref.actor.self_distillation.oprd_coef="${OPRD_COEF:-1.0}" \
    actor_rollout_ref.actor.self_distillation.oprd_layers="${OPRD_LAYERS:-all}" \
    actor_rollout_ref.actor.self_distillation.oprd_positions="${OPRD_POSITIONS:-last_k}" \
    actor_rollout_ref.actor.self_distillation.oprd_last_k="${OPRD_LAST_K:-2000}" \
    actor_rollout_ref.actor.self_distillation.oprd_normalize=False \
    "$@"
