#!/usr/bin/env bash
# One-step real-GPU PBD integration smoke (14_P5_pre_run_integration_test_plan.md §3): Keep then Replace,
# 12 prompts x n=2 (B=24: divisible by the 3-GPU world size AND the 8 agent-loop workers); Keep uses coverage 1.0 (every parent branches),
# Replace uses coverage 0.5 (N_branch=3 != N_v0=6 exercises the two-scale normalisation). Dumps under
# eval/pbd_smoke/<mode>, then scripts/test_pbd_integration.py checks the invariants. Production code path.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
PYBIN="${VOPD_PY:-python}"
rc_all=0
for MODE in ${MODES:-keep replace}; do
  DUMP="$PROJECT_ROOT/eval/pbd_smoke/${PBD_VIEW:-hide}_$MODE"; rm -rf "$DUMP"; mkdir -p "$DUMP"
  echo "================ PBD smoke mode=$MODE $(date)"
  rm -rf "$PROJECT_ROOT/checkpoints/pair_PBDsmoke${PBD_VIEW:-hide}${MODE}_6karmA"
  COV=1.0; [ "$MODE" = replace ] && COV=0.5
  PBD_SMOKE_DUMP="$DUMP" PBD_VIEW="${PBD_VIEW:-hide}" PBD_MODE=$MODE PBD_COVERAGE=$COV TRAIN_BATCH_SIZE=12 PPO_MIMI_BATCH_SIZE=12 ROLLOUT_N=2 \
    TRAINER_TOTAL_TRAINING_STEPS=1 TRAINER_SAVE_FREQ=1000 EXPERIMENT_NAME_OVERRIDE=pair_PBDsmoke${PBD_VIEW:-hide}${MODE}_6karmA \
    TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-3}" ULYSSES_SP=1 \
    bash scripts/train_pair_pbd.sh trainer.resume_mode=disable > "$DUMP/train.log" 2>&1; rc=$?
  echo "train exit=$rc; dumps: $(ls $DUMP | wc -l) files"
  rm -rf "$PROJECT_ROOT/checkpoints/pair_PBDsmoke${PBD_VIEW:-hide}${MODE}_6karmA"
  [ $rc -eq 0 ] || { echo "!! training step failed (mode=$MODE); tail:"; grep -aE "Error|Traceback|raise" "$DUMP/train.log" | grep -v codecache | tail -5; rc_all=1; continue; }
  "$PYBIN" scripts/test_pbd_integration.py --dump "$DUMP" --mode "$MODE" | tee "$DUMP/check.log"; r=${PIPESTATUS[0]}
  [ $r -eq 0 ] || rc_all=1
done
echo "PBD_SMOKE_EXIT=$rc_all"; exit $rc_all
