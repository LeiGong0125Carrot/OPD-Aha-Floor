#!/bin/bash
#SBATCH -p standard -c 1 --time=4-00:00:00 -J vopd-watchdog
#SBATCH -o /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/logs/watchdog_pbd.out
# 夜间看护 (用户 10-08): 每 5 分钟
#   - 磁盘 < 600G: 删已训完臂全部 ckpt 的优化器状态, 训练中臂非最新 ckpt 的优化器状态 (只留模型权重, 不影响评测/续训)
#   - 编排器日志出现 "完成" -> 释放对应 hold
#   - 编排器不在且未完成 -> 重投 (resume_mode=auto / 幂等评测), 每条最多 3 次
#   - hold 消失但未完成 -> 新申 hold, 生成换 hold 号的编排器副本并重投 (最多 2 次)
#   - 全部完成 -> 退出
set -uo pipefail
for v in $(env | grep -oE "^SLURM_[A-Z_]+"); do unset "$v"; done
S=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup; H3=/sfs/weka/scratch/nkw3mr/Vision-OPD/hold_gpu3.sbatch
CK=/scratch/nkw3mr/Vision-OPD/OPD-Aha-sup/checkpoints
say() { echo "[watchdog $(date '+%m-%d %H:%M:%S')] $*"; }
# name|orch_file|log|jobname|hold|ckpt_tag|train_done_marker
ENTRIES="replace|orch_pbdreplace.sbatch|orch_pbdreplace.out|vopd-pbdreplace|20971535|pair_PBDrH_6karmA|orch_pbdreplace.train_done
keepr2|orch_pbdkeepr2.sbatch|orch_pbdkeepr2.out|vopd-pbdkeepr2|20971905|pair_PBDkHr2_6karmA|orch_pbdkeepr2.train_done
p05|orch_pbdkp05.sbatch|orch_pbdkp05.out|vopd-pbdkp05|21012723|pair_PBDkHp05_6karmA|orch_pbdkeepp05.train_done
p20|orch_pbdkp20.sbatch|orch_pbdkp20.out|vopd-pbdkp20|21012724|pair_PBDkHp20_6karmA|orch_pbdkeepp20.train_done
o6|orch_o6PBDk.sbatch|orch_o6PBDk.out|vopd-o6PBDk|21014446|off6k_PBDk|orch_o6PBDk.train_done"
ST=$S/watchdog_state; mkdir -p $ST
while IFS='|' read -r n o l jn h t m; do [ -f $ST/$n.hold ] || echo $h > $ST/$n.hold; done <<< "$ENTRIES"
hold_state() { squeue -j "$1" -h -o %T 2>/dev/null | head -1; }
disk_guard() {
  local a=$(df -BG --output=avail /sfs/weka/scratch/nkw3mr | tail -1 | tr -dc '0-9'); [ "${a:-0}" -ge 600 ] && return
  while IFS='|' read -r n o l jn h t m; do
    local d=$CK/$t; [ -d $d ] || continue
    if [ -f $S/$m ]; then rm -f $d/global_step_*/actor/optim_world_size_*.pt
    else local L=$(cat $d/latest_checkpointed_iteration.txt 2>/dev/null); [ -z "$L" ] && continue
      for g in $d/global_step_*; do [ "${g##*_}" -lt "$L" ] && rm -f $g/actor/optim_world_size_*.pt; done; fi
  done <<< "$ENTRIES"
  say "磁盘 ${a}G<600G: 已清理优化器状态, 现在 $(df -BG --output=avail /sfs/weka/scratch/nkw3mr | tail -1 | tr -dc '0-9')G"
}
gpu_check() {   # 每轮对每张 RUNNING 的 hold 采样; 整张 hold 连续 3 轮 (15 min) 全部 GPU 空闲且其编排器不在 -> 释放
  for hf in $ST/*.hold; do
    local n=$(basename $hf .hold); [ -f $ST/$n.done ] && continue; local h=$(cat $hf)
    [ "$(hold_state $h)" = RUNNING ] || continue
    local u=$(timeout 90 srun --jobid=$h --ntasks=1 --overlap bash -c "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr '\n' ';'" 2>/dev/null)
    [ -z "$u" ] && continue
    local busy=$(echo "$u" | tr ';' '\n' | awk -F', ' 'NF==2 && ($1>5 || $2>2000){c++} END{print c+0}')
    local tot=$(echo "$u" | tr ';' '\n' | awk 'NF{c++} END{print c+0}')
    echo "$(date '+%m-%d %H:%M') $n hold=$h busy=$busy/$tot [$u]" >> $ST/gpu_usage.log
    if [ "$busy" = 0 ]; then
      local c=$(( $(cat $ST/$n.idle 2>/dev/null || echo 0) + 1 )); echo $c > $ST/$n.idle
      [ $c -ge 3 ] && say "!! $n hold $h 已连续 $c 轮 (≥15 min) 全部 GPU 空闲 [$u]"
    else
      echo 0 > $ST/$n.idle
    fi
  done
}
say "启动 (含 GPU 空闲监视)"
it=0
while true; do
  disk_guard
  gpu_check
  all_done=1
  while IFS='|' read -r n o l jn h0 t m; do
    [ -f $ST/$n.done ] && continue
    all_done=0; h=$(cat $ST/$n.hold)
    if tail -n 3 "$S/logs/$l" 2>/dev/null | grep -q "完成"; then
      hs=$(hold_state $h); [ -n "$hs" ] && { scancel $h; say "$n 完成 -> 释放 hold $h ($hs)"; } || say "$n 完成 (hold $h 已不在)"
      touch $ST/$n.done; continue
    fi
    oj=$(squeue -u $USER -h -n "$jn" -o %i 2>/dev/null | head -1)
    hs=$(hold_state $h)
    if [ -z "$hs" ]; then
      r=$(cat $ST/$n.rehold 2>/dev/null || echo 0); [ "$r" -ge 2 ] && { say "!! $n hold 已重申 $r 次, 放弃"; touch $ST/$n.done; continue; }
      nh=$(sbatch --parsable $H3); echo $nh > $ST/$n.hold; echo $((r+1)) > $ST/$n.rehold
      [ -n "$oj" ] && scancel $oj
      nf=$S/${o%.sbatch}_h$nh.sbatch
      sed -E "s/^(ARM=[^;]+; |NAME=[^;]+; )J=[0-9]+;/\1J=$nh;/" $S/$o > $nf
      if grep -q "J=$nh;" $nf; then sbatch $nf >/dev/null && say "$n: hold $h 消失 -> 新 hold $nh, 重投 $(basename $nf)"; else say "!! $n: 生成新编排器失败"; fi
      continue
    fi
    if [ -z "$oj" ]; then
      k=$(cat $ST/$n.retry 2>/dev/null || echo 0); [ "$k" -ge 3 ] && { say "!! $n 编排器已重投 $k 次仍退出, 释放 hold $h 并放弃"; scancel $h; touch $ST/$n.done; continue; }
      f=$S/$o; [ -f $S/${o%.sbatch}_h$h.sbatch ] && f=$S/${o%.sbatch}_h$h.sbatch
      sbatch $f >/dev/null && echo $((k+1)) > $ST/$n.retry && say "$n 编排器不在且未完成 (末行: $(tail -n1 $S/logs/$l | cut -c1-70)) -> 重投 $(basename $f) (第 $((k+1)) 次)"
    fi
  done <<< "$ENTRIES"
  [ $all_done = 1 ] && { say "=== 全部完成, 退出 ==="; exit 0; }
  sleep 300
done
