#!/bin/bash
# 多卡并行版评测 (用户 10-09: 评测阶段不许 2/3 卡空转): 把 STEPS 轮流分给每张可见 GPU, 每个 worker 独占一张卡,
# 用自己的被测服务端口 (8811+i) 与 judge 端口 (8913+i), 跑 eval_off6k_w.sh (= eval_off6k.sh + 端口参数化) 的完整流程
# (merge -> serve -> infer -> gpt-oss judge 双口径)。step 集合互不相交, 不会写同一个答案文件。全部结束后汇总一次。
# 用法同 eval_off6k.sh: TAG=.. SN=.. [CK=..] [STEPS=..] [EXTRA_NAMES=..] [EXTRA_SERVE=..] bash eval_off6k_par.sh
set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
TAG="${TAG:?}"; SN="${SN:-${TAG/off6k_/off6k}}"
CK="${CK:-/scratch/nkw3mr/Vision-OPD/OPD-Aha-sup/checkpoints/$TAG}"
STEPS="${STEPS:-$(ls "$CK" 2>/dev/null | grep -oE 'global_step_[0-9]+' | sed 's/global_step_//' | sort -n | tr '\n' ' ')}"
GPU_LIST="${GPU_LIST:-$(seq -s ' ' 0 $(( $(nvidia-smi -L 2>/dev/null | wc -l) - 1 )))}"; read -ra GL <<< "$GPU_LIST"; NG=${#GL[@]}; [ "$NG" -ge 1 ] || { GL=(0); NG=1; }
read -ra ST <<< "$STEPS"; [ ${#ST[@]} -lt $NG ] && NG=${#ST[@]}; [ $NG -ge 1 ] || NG=1
echo "================ eval_off6k_par $TAG ($SN) steps=[$STEPS] gpus=$NG $(hostname) $(date)"
pids=()
for ((i=0; i<NG; i++)); do
  sub=""; for ((k=i; k<${#ST[@]}; k+=NG)); do sub="$sub ${ST[$k]}"; done
  ex=""; xs=""; [ $i -eq 0 ] && { ex="${EXTRA_NAMES:-}"; xs="${EXTRA_SERVE:-}"; }
  echo ">>> worker $i GPU ${GL[$i]} steps=[$sub]"
  ( CUDA_VISIBLE_DEVICES=${GL[$i]} PORT=$((8811+i)) JP=$((8913+i)) JUDGE_PID_KILL=1 TAG=$TAG SN=$SN CK=$CK STEPS="$sub" EXTRA_NAMES="$ex" EXTRA_SERVE="$xs" \
      BENCHES="${BENCHES:-vstar zoombench hrbench-4k hrbench-8k}" bash $SETUP/eval_off6k_w.sh > $SETUP/logs/eval_${TAG}_w$i.log 2>&1 ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for ((i=0; i<NG; i++)); do grep -aq "EVAL_OFF6K EXIT=0" $SETUP/logs/eval_${TAG}_w$i.log || { echo "!! worker $i 未正常结束: $(tail -n2 $SETUP/logs/eval_${TAG}_w$i.log | tr '\n' ' ' | cut -c1-200)"; rc=1; }; done
[ $rc -eq 0 ] || { echo "!! 并行评测有 worker 失败"; exit 1; }
# 汇总 (复用 worker 版脚本的汇总段: 所有答案/judge 已齐, 只打印表)
"$VOPD_PY" - "$VOPD_ROOT/eval" "$SN" "$STEPS" <<'PY'
import json, os, sys
ev, sn, steps = sys.argv[1], sys.argv[2], sys.argv[3].split()
B = {"vstar": 191, "zoombench": 845, "hrbench-4k": 800, "hrbench-8k": 800}
def acc(f, n):
    if not os.path.exists(f): return None
    try: r = json.load(open(f))
    except Exception: r = [json.loads(l) for l in open(f)]
    return 100*sum(str(x.get('judge','')).strip().lower().startswith('yes') for x in r)/len(r) if len(r) == n else None
print("== 汇总 (judge 口径 %; rule-first | pure-llm)")
print(f"{'model':24s}" + "".join(f"{b+'/'+p:>24s}" for b in B for p in ("rule-first", "pure-llm")))
for s in steps:
    n = f"{sn}-step{s}"; row = f"{n:24s}"
    for b, q in B.items():
        for p in ("rule-first", "pure-llm"):
            a = acc(f"{ev}/judge/{p}/{b}/{n}_seed42_answer.jsonl", q); row += f"{'-':>24s}" if a is None else f"{a:19.2f}(n={q})"
    print(row)
PY
echo "EVAL_OFF6K EXIT=0 $TAG $(date)"
