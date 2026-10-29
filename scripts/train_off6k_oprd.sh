#!/usr/bin/env bash
# OPRD-Vanilla (arXiv 2606.06021, paper-exact `main` branch) on the off6k V0 base (user 10-26): the student (red-box full
# image + hint) is trained to match the frozen base teacher's (crop view) hidden states at its own response tokens --
# plain MSE, all transformer layers, last 2000 response tokens (= every response token here, R = 1024), rep-only
# (the V0 JSD is still computed and logged, not trained). verl/utils/oprd.py.
#   tag off6k_OPRDvn2 ; seed 42, rollout_n 2, lr 2e-6 (project default; the paper uses 1e-5), 130 steps, save 10.
# Token-level ablation (user 10-28): OPRD_POSITIONS=first_frac OPRD_FRAC=0.25 -> only the first ceil(0.25 n) response
# tokens (t=0 / the last prompt token is never included), tag off6k_OPRDf25n2; everything else identical to Vanilla.
# + t=0 (user 10-29, f25 matched Vanilla): OPRD_T0=1 also supervises the last prompt token's hidden state (pre-generation
# state) -> tag suffix "t0", e.g. off6k_OPRDf25t0n2.
# 用法: [RUN_SUFFIX=..] [OPRD_T0=0|1] [OPRD_ONLY=True] [OPRD_LAYERS=all] [OPRD_POSITIONS=last_k|first_frac] [OPRD_FRAC=0.25] bash scripts/train_off6k_oprd.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OFF_MODE=V0 ROLLOUT_N="${ROLLOUT_N:-2}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-/scratch/nkw3mr/Vision-OPD/data/train.parquet}"
case "${OPRD_POSITIONS:-last_k}" in
    last_k) _TAG=off6k_OPRDvn2 ;;
    first_frac) _f="${OPRD_FRAC:-0.25}"; _TAG="off6k_OPRDf$(python3 -c "print(round(100*float('${_f}')))")n2" ;;
    *) _TAG="off6k_OPRD${OPRD_POSITIONS}n2" ;;
esac
case "${OPRD_T0:-0}" in
    0) _T0=False ;;
    1) _T0=True; _TAG="${_TAG%n2}t0n2" ;;
    *) echo "OPRD_T0 must be 0 or 1, got ${OPRD_T0}" >&2; exit 1 ;;
esac
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-${_TAG}${RUN_SUFFIX:-}}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}" TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-130}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP=null
echo "OPRD-Vanilla: only=${OPRD_ONLY:-True} layers=${OPRD_LAYERS:-all} positions=${OPRD_POSITIONS:-last_k} last_k=${OPRD_LAST_K:-2000} frac=${OPRD_FRAC:-0.25} (frac only used by first_frac) coef=${OPRD_COEF:-1.0} include_t0=${_T0} normalize=False tag=${EXPERIMENT_NAME_OVERRIDE}"
exec "${PROJECT_ROOT}/scripts/train_official6k.sh" \
    actor_rollout_ref.actor.self_distillation.oprd_enable=True \
    actor_rollout_ref.actor.self_distillation.oprd_only="${OPRD_ONLY:-True}" \
    actor_rollout_ref.actor.self_distillation.oprd_coef="${OPRD_COEF:-1.0}" \
    actor_rollout_ref.actor.self_distillation.oprd_layers="${OPRD_LAYERS:-all}" \
    actor_rollout_ref.actor.self_distillation.oprd_positions="${OPRD_POSITIONS:-last_k}" \
    actor_rollout_ref.actor.self_distillation.oprd_last_k="${OPRD_LAST_K:-2000}" \
    actor_rollout_ref.actor.self_distillation.oprd_frac="${OPRD_FRAC:-0.25}" \
    actor_rollout_ref.actor.self_distillation.oprd_normalize=False \
    actor_rollout_ref.actor.self_distillation.oprd_include_t0="${_T0}" \
    "$@"
