#!/usr/bin/env bash
# t=0 arms on the off6k V0 base (user 10-31: "在 vision-opd 的基础上加入 t=0 是否会更好; 单独 t=0 能否拿到大部分收益"):
#   T0_ARM=V0T0H -> off6k_V0t0hn2 : V0 JSD on every response token (unchanged) + T0_COEF x OPRD hidden MSE at t=0 only
#                  (the last prompt token of each view = the pre-generation state; all layers, plain MSE; oprd_positions=none
#                  + oprd_include_t0, oprd_only=False)
#   T0_ARM=Y0    -> off6k_V0y0n2  : control "distil t=0 only" in token-distribution space: the V0 JSD on y_0 only
#                  (self_distillation.sd_first_token_only; token-mean over the y_0 tokens, so each y_0 weighs ~mean-length x V0)
# Everything else = V0 (seed 42, rollout_n 2, lr 2e-6, 130 steps, save 10).
# 用法: T0_ARM=V0T0H|Y0 [T0_COEF=1.0] [RUN_SUFFIX=..] bash scripts/train_off6k_t0.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OFF_MODE=V0 ROLLOUT_N="${ROLLOUT_N:-2}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/scratch/nkw3mr/Vision-OPD/data/train.parquet}"
case "${T0_ARM:-}" in
    V0T0H) _TAG=off6k_V0t0hn2
           _OV=(actor_rollout_ref.actor.self_distillation.oprd_enable=True
                actor_rollout_ref.actor.self_distillation.oprd_only=False
                actor_rollout_ref.actor.self_distillation.oprd_coef="${T0_COEF:-1.0}"
                actor_rollout_ref.actor.self_distillation.oprd_layers=all
                actor_rollout_ref.actor.self_distillation.oprd_positions=none
                actor_rollout_ref.actor.self_distillation.oprd_include_t0=True
                actor_rollout_ref.actor.self_distillation.oprd_normalize=False) ;;
    Y0)    _TAG=off6k_V0y0n2
           _OV=(actor_rollout_ref.actor.self_distillation.sd_first_token_only=True) ;;
    *) echo "T0_ARM must be V0T0H or Y0, got '${T0_ARM:-}'" >&2; exit 1 ;;
esac
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-${_TAG}${RUN_SUFFIX:-}}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-130}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP=null
echo "t0 arm ${T0_ARM}: tag=${EXPERIMENT_NAME_OVERRIDE} overrides: ${_OV[*]}"
exec "${PROJECT_ROOT}/scripts/train_official6k.sh" "${_OV[@]}" "$@"
