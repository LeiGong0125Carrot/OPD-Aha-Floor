#!/bin/bash
# 用户 10-07 规则: 所有 ckpt 测全 V* / ZoomBench / HR-4K / HR-8K (pure-llm judge 齐全) 后,
#   1) 写 run_records/<TAG>.yaml (配置 + 每个 ckpt 各基准双口径分数 + 峰值 + 保留的 ckpt);
#   2) 只保留 ZoomBench (pure-llm) 最好的 ckpt, 其余 ckpt 与 merged 删除。
# 未测全则跳过 (不删)。TRIM_BENCHES 已废弃 (保留变量只为兼容旧调用)。
# 用法: [SN=..] [CK_ROOT=..] bash trim_off6k.sh [--apply] TAG
set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
CK="${CK_ROOT:-/scratch/nkw3mr/Vision-OPD/OPD-Aha-sup/checkpoints}"; EV=/sfs/weka/scratch/nkw3mr/Vision-OPD/eval; MG=/sfs/weka/scratch/nkw3mr/Vision-OPD/merged
PYB=/sfs/weka/scratch/nkw3mr/envs/vision-opd/bin/python
APPLY=0; [ "${1:-}" = "--apply" ] && { APPLY=1; shift; }
nq() { case "$1" in vstar) echo 191;; zoombench) echo 845;; hrbench-4k|hrbench-8k) echo 800;; esac; }
score() { "$PYB" -c "
import json,sys
f=sys.argv[1]; n=int(sys.argv[2])
try: r=json.load(open(f))
except Exception: r=[json.loads(l) for l in open(f)]
print('%.2f'%(100*sum(str(x.get('judge','')).strip().lower().startswith('yes') for x in r)/len(r)) if len(r)==n else 'BAD')" "$1" "$2" 2>/dev/null || echo BAD; }
for tag in "$@"; do
  d="$CK/$tag"; [ -d "$d" ] || { echo "SKIP $tag: 无 ckpt 目录 $d"; continue; }
  sn="${SN:-${tag/off6k_/off6k}}"
  steps=$(ls "$d" | grep -oE 'global_step_[0-9]+' | sed 's/global_step_//' | sort -n | tr '\n' ' '); [ -z "$steps" ] && continue
  ok=1; zb=""
  for s in $steps; do for b in vstar zoombench hrbench-4k hrbench-8k; do
    f="$EV/judge/pure-llm/$b/${sn}-step${s}_seed42_answer.jsonl"
    a=BAD; [ -f "$f" ] && a=$(score "$f" "$(nq $b)")
    [ "$a" = BAD ] && { ok=0; echo "  未齐: $b step $s"; continue; }
    [ "$b" = zoombench ] && zb="$zb $s:$a"
  done; done
  [ $ok = 1 ] || { echo "SKIP $tag: 四个基准未测全, 不写记录不删除"; continue; }
  keep=$(echo $zb | tr ' ' '\n' | sort -t: -k2 -nr -k1 -n | head -1 | cut -d: -f1)
  rec=$("$PYB" $SETUP/record_run.py --tag "$tag" --sn "$sn" --ckpt_dir "$d" --kept "$keep") || { echo "!! $tag: 记录写入失败, 不删除"; continue; }
  del=""; for s in $steps; do [ "$s" = "$keep" ] || del="$del $s"; done
  echo "$tag: ZoomBench[$zb] 保留 step $keep; 记录 $rec; del=[$del]"
  [ $APPLY -eq 1 ] && for s in $del; do rm -rf "$d/global_step_$s" "$MG/$tag-step$s" && echo "  删 $tag/global_step_$s (+merged)"; done
done
