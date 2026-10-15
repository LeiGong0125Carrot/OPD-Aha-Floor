#!/bin/bash
#SBATCH -p standard -c 1 --time=4-00:00:00 -J vopd-o6PBDk
#SBATCH -o /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/orch_o6PBDk.out
# 官方 Vision-OPD-6K 数据上的 PBDk (用户 10-07: PBD Keep vs 同数据 V0), hold 21014446:
#   冒烟 2 步 -> 130 步 (48 x n8 ~ 1 epoch, save 10, resume_mode=auto) -> eval V* + HR-4K/8K (双口径) -> 按三基准峰值裁剪
set -uo pipefail
for v in $(env | grep -oE "^SLURM_[A-Z_]+"); do unset "$v"; done
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup; REPO=/scratch/nkw3mr/Vision-OPD/OPD-Aha-sup
NAME=PBDk; J=21014446; TAG=off6k_PBDk; SN=off6kPBDk
TRAINCMD="PBD_MODE=keep PBD_VIEW=off6k PBD_RATIO=0.10 PBD_COVERAGE=0.5 PBD_LAMBDA=0.5 bash scripts/train_pair_pbd.sh"
STOP=$SETUP/orch_o6$NAME.stop; DONE=$SETUP/orch_o6$NAME.train_done
say() { echo "[o6$NAME $(date '+%m-%d %H:%M:%S')] $*"; }
run_on() { local log=$1; shift
  srun --jobid=$J --ntasks=1 --cpus-per-task=8 bash -c "
    source $SETUP/00_env.sh; export RAY_TMPDIR=/dev/shm/ray_o6 TMPDIR=/dev/shm/tmp_o6; mkdir -p \"\$RAY_TMPDIR\" \"\$TMPDIR\"
    unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
    $*
  " >> "$log" 2>&1; }
train_run() { local tag=$1 steps=$2 sfx=$3
  run_on "$SETUP/logs/train_$sfx.log" "cd $REPO && echo 'TRAIN ${sfx^^} START' \$(date) && EXPERIMENT_NAME_OVERRIDE=$tag TRAINER_TOTAL_TRAINING_STEPS=$steps TRAINER_SAVE_FREQ=10 \
     TRAINER_N_GPUS_PER_NODE=3 ULYSSES_SP=1 TRAIN_BATCH_SIZE=48 PPO_MIMI_BATCH_SIZE=48 ROLLOUT_N=8 TRAINER_TOTAL_EPOCHS=2 $TRAINCMD trainer.resume_mode=auto; echo 'TRAIN ${sfx^^} EXIT='\$? \$(date)"
  tail -5 "$SETUP/logs/train_$sfx.log" | grep -aq "TRAIN ${sfx^^} EXIT=0"; }
mech() { tr '\r' '\n' < "$SETUP/logs/train_$1.log" | grep -aoE '(pbd/branch_rows|pbd/yT_len_mean|pbd/leak_row_frac|teacher_null_forward_frac|response_length/mean|timing_s/step|timing_s/pbd_gen|self_distillation/pbd_jsd_branch_cnt):[0-9.e-]+' | awk -F: '{s[$1]+=$2;n[$1]++} END{for(k in s) printf "%s=%.4f ", k, s[k]/n[k]}'; }
ckpt_cleaner() { local tag=$1
  while true; do sleep 600
    local a=$(df -BG --output=avail /sfs/weka/scratch/nkw3mr | tail -1 | tr -dc '0-9'); [ "${a:-0}" -ge 500 ] && continue
    local d="$REPO/checkpoints/$tag"; [ -d "$d" ] || continue; local latest=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null)
    for p in $(ls -d "$d"/global_step_* 2>/dev/null | sort -V); do local n=${p##*_}; [ -z "$latest" ] && continue; [ "$n" -lt "$latest" ] || continue
      rm -rf "$p" && say "  cleaner(磁盘 ${a}G<500G): 删 $tag/global_step_$n" && break; done; done; }
[ -f "$STOP" ] && { say stop; exit 0; }
say "等待 hold $J"; while ! squeue -j $J -h -o %T 2>/dev/null | grep -q RUNNING; do [ -f "$STOP" ] && { say stop; exit 0; }; sleep 120; done
g=""; for try in 1 2 3; do g=$(timeout 150 srun --jobid=$J --ntasks=1 bash -c 'echo $SLURM_GPUS_ON_NODE' 2>/dev/null | grep -E '^[0-9]+$' | tail -1); [ "${g:-0}" -ge 3 ] && break; sleep 300; done
say "hold $J GPU=${g:-?}"; [ "${g:-0}" -ge 3 ] || { say "!! 拿不到 3 卡"; exit 1; }
if [ "1" = 1 ] && [ ! -f "$DONE" ] && [ ! -d "$REPO/checkpoints/$TAG" ]; then
  say "一步真机集成测试 (14 号计划, PBD_VIEW=off6k, keep)"
  run_on "$SETUP/logs/pbd_smoke_integration_o6$NAME.log" "cd $REPO && PYTHONPATH=$REPO MODES=keep PBD_VIEW=off6k bash scripts/run_pbd_smoke.sh"
  grep -aq "PBD_SMOKE_EXIT=0" "$SETUP/logs/pbd_smoke_integration_o6$NAME.log" || { say "!! 集成测试未通过: $(grep -a 'RESULT' $SETUP/logs/pbd_smoke_integration_o6$NAME.log | tail -1)"; exit 1; }
  say "集成测试通过: $(grep -a 'RESULT' $SETUP/logs/pbd_smoke_integration_o6$NAME.log | tail -1) $(grep -a 'abs_diff' $SETUP/logs/pbd_smoke_integration_o6$NAME.log | tail -1)"
fi
if [ ! -f "$DONE" ] && [ ! -d "$REPO/checkpoints/$TAG" ]; then
  say "冒烟 2 步"; rm -rf "$REPO/checkpoints/${TAG}_smoke"; : > "$SETUP/logs/train_o6smoke$NAME.log"
  train_run ${TAG}_smoke 2 o6smoke$NAME || { say "!! 冒烟失败: $(grep -aE 'Error|Killed|OOM' $SETUP/logs/train_o6smoke$NAME.log | grep -v codecache | tail -2 | cut -c1-200)"; exit 1; }
  say "  冒烟指标: $(mech o6smoke$NAME)"; rm -rf "$REPO/checkpoints/${TAG}_smoke"; say "冒烟通过"
fi
if [ ! -f "$DONE" ]; then
  say "训练 $TAG 130 步"; ckpt_cleaner "$TAG" & cl=$!
  train_run "$TAG" 130 "$TAG"; rc=$?; kill $cl 2>/dev/null; wait $cl 2>/dev/null
  [ $rc -eq 0 ] || { say "!! 训练失败 (重投可续跑)"; exit 1; }; touch "$DONE"; say "  训练指标: $(mech $TAG)"
fi
say "评测 $TAG 全部 ckpt (V* + HR-4K/8K, 双口径)"
run_on "$SETUP/logs/eval_$TAG.log" "TAG=$TAG SN=$SN BENCHES='vstar hrbench-4k hrbench-8k' bash $SETUP/eval_off6k.sh"
grep -aq "EVAL_OFF6K EXIT=0" "$SETUP/logs/eval_$TAG.log" || { say "!! 评测未正常结束"; exit 1; }
sed -n '/^== 汇总/,/EVAL_OFF6K EXIT/p' "$SETUP/logs/eval_$TAG.log" | sed 's/^/  /'
SN=$SN TRIM_BENCHES="vstar hrbench-4k hrbench-8k" bash "$SETUP/trim_off6k.sh" --apply "$TAG" 2>&1 | sed 's/^/  /'
say "释放 hold $J"; scancel $J; say "=== 完成 ==="
