#!/bin/bash
# 官方 Vision-OPD-6K 复现臂评测 (用户 10-06): 所有已存 ckpt -> merge -> vLLM serve -> OPD-Aha 的 eval/infer.py
# (vstar, zoombench) -> gpt-oss-120b judge, 两种 protocol (rule-first / pure-llm) 分目录 judge/<protocol>/<bench>/ -> cal_acc
# TreeBench 不测 (用户 10-06)。幂等: 答案/judge 文件齐全就跳过。
# 用法: TAG=off6k_A [STEPS="10 20"] [BENCHES="vstar zoombench"] [EXTRA_NAMES="base ahaAn2-step50 ..."] bash eval_off6k.sh
set -uo pipefail
SETUP=/sfs/weka/scratch/nkw3mr/Vision-OPD-setup
source "$SETUP/00_env.sh"
REPO=/scratch/nkw3mr/Vision-OPD/OPD-Aha-sup      # OPD-Aha 评测代码 (eval/infer.py, judge_qwenlm.py, cal_acc.py)
EVAL=$VOPD_ROOT/eval                              # 基准 json / 图片 / 答案 / judge 目录
TAG="${TAG:?TAG=off6k_A|off6k_Ahm|off6k_Ahf}"
SN="${SN:-${TAG/off6k_/off6k}}"
CK="${CK:-$REPO/checkpoints/$TAG}"; MERGE_ROOT=$VOPD_ROOT/merged
STEPS="${STEPS:-$(ls "$CK" 2>/dev/null | grep -oE 'global_step_[0-9]+' | sed 's/global_step_//' | sort -n | tr '\n' ' ')}"
BENCHES="${BENCHES:-vstar zoombench}"
# 用户 10-07 规则: 所有 ckpt 一次性测全 V* / ZoomBench / HR-Bench 4K / 8K (调用方传的 BENCHES 只会被扩充, 不会被缩小)
for _b in vstar zoombench hrbench-4k hrbench-8k; do case " $BENCHES " in *" $_b "*) ;; *) BENCHES="$BENCHES $_b";; esac; done
EXTRA_NAMES="${EXTRA_NAMES:-}"
PROTOCOLS="${PROTOCOLS:-rule-first pure-llm}"
export PORT="${PORT:-8811}"; JP=8813
nq() { case "$1" in vstar) echo 191;; zoombench) echo 845;; hrbench-4k|hrbench-8k) echo 800;; *) echo 0;; esac; }
bjson() { case "$1" in vstar) echo vstar.json;; zoombench) echo zoombench.json;; hrbench-4k) echo hr_bench_4k.json;; hrbench-8k) echo hr_bench_8k.json;; esac; }
ans_ok() { [ -f "$1" ] && [ "$(wc -l < "$1")" -ge "$2" ] && ! grep -q '\[API_ERROR\]\|\[FUTURE_ERROR\]' "$1"; }
EXTRA_SERVE="${EXTRA_SERVE:-}"   # "name|model_path ..." 对照模型: 缺答案的基准补推理 (base / 官方 ckpt)
echo "================ eval_off6k $TAG ($SN) steps=[$STEPS] benches=[$BENCHES] $(hostname) $(date)"
NAMES=""
# ---------- 1. merge + serve + OPD-Aha infer.py ----------
for s in $STEPS; do
  name="$SN-step$s"; NAMES="$NAMES $name"; DST="$MERGE_ROOT/$TAG-step$s"
  need=0; for b in $BENCHES; do ans_ok "$EVAL/model_answer/$b/${name}_seed42_answer.jsonl" "$(nq $b)" || need=1; done
  [ $need = 0 ] && { echo ">>> $name: 答案齐, 跳过推理"; continue; }
  if ! { [ -f "$DST/config.json" ] && ls "$DST"/*.safetensors >/dev/null 2>&1; }; then
    SRC="$CK/global_step_$s/actor"; [ -d "$SRC" ] || { echo "!! 缺 $SRC, 跳过"; continue; }
    echo ">>> merge $name $(date)"; mkdir -p "$DST"
    ( cd "$REPO" && PYTHONPATH="$REPO" "$VOPD_PY" -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$DST" ) \
      || { rm -rf "$DST"; echo "!! merge 失败 $name"; continue; }
  fi
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 5
  echo ">>> serve $name $(date)"
  MODEL="$DST" SERVED_NAME="$name" bash "$SETUP/06_serve.sh" start || { echo "!! serve $name 失败, 跳过"; continue; }
  for b in $BENCHES; do
    ans="$EVAL/model_answer/$b/${name}_seed42_answer.jsonl"; ans_ok "$ans" "$(nq $b)" && continue
    echo ">>> infer $b $name $(date)"
    ( cd "$REPO/eval" && "$VOPD_PY" infer.py --benchmark "$b" --benchmark_json "$EVAL/$(bjson $b)" --out_dir "$EVAL/model_answer" \
        --model_name "${name}_seed42" --seed 42 --api_base "http://localhost:$PORT/v1/" --api_key EMPTY --model_id "$name" \
        --max_tokens 4096 --max_retries 3 --parallel_workers 32 ) || echo "!! infer $b $name 失败"
  done
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 5
done
# ---------- 1b. 对照模型补推理 ----------
for spec in $EXTRA_SERVE; do
  name="${spec%%|*}"; path="${spec##*|}"
  need=0; for b in $BENCHES; do ans_ok "$EVAL/model_answer/$b/${name}_seed42_answer.jsonl" "$(nq $b)" || need=1; done
  [ $need = 0 ] && { echo ">>> $name: 对照答案齐, 跳过"; continue; }
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 5
  MODEL="$path" SERVED_NAME="$name" bash "$SETUP/06_serve.sh" start || { echo "!! serve $name 失败, 跳过"; continue; }
  for b in $BENCHES; do
    ans="$EVAL/model_answer/$b/${name}_seed42_answer.jsonl"; ans_ok "$ans" "$(nq $b)" && continue
    echo ">>> infer $b $name $(date)"
    ( cd "$REPO/eval" && "$VOPD_PY" infer.py --benchmark "$b" --benchmark_json "$EVAL/$(bjson $b)" --out_dir "$EVAL/model_answer" \
        --model_name "${name}_seed42" --seed 42 --api_base "http://localhost:$PORT/v1/" --api_key EMPTY --model_id "$name" \
        --max_tokens 4096 --max_retries 3 --parallel_workers 32 ) || echo "!! infer $b $name 失败"
  done
  bash "$SETUP/06_serve.sh" stop >/dev/null 2>&1; sleep 5
done
# ---------- 2. gpt-oss-120b judge, 两种 protocol ----------
if ! curl -sf "http://localhost:$JP/v1/models" >/dev/null; then
  VLLM_BIN="$(dirname "$VOPD_PY")/vllm"; mkdir -p /dev/shm/tmp_off6k
  echo ">>> 启动 gpt-oss-120b judge $(date)"
  "$VLLM_BIN" serve openai/gpt-oss-120b --port $JP --gpu-memory-utilization 0.9 --max-model-len 8192 > /dev/shm/tmp_off6k/gptoss.log 2>&1 &
  for i in $(seq 1 150); do sleep 10; curl -sf "http://localhost:$JP/v1/models" >/dev/null 2>&1 && break; done
  curl -sf "http://localhost:$JP/v1/models" >/dev/null || { echo "!! judge 服务 25 min 未就绪"; tail -20 /dev/shm/tmp_off6k/gptoss.log; exit 1; }
fi
cd "$REPO/eval"
for name in $NAMES $EXTRA_NAMES; do for b in $BENCHES; do
  ans="$EVAL/model_answer/$b/${name}_seed42_answer.jsonl"; ans_ok "$ans" "$(nq $b)" || { echo "!! $name $b 答案缺/不全, 不判"; continue; }
  for P in $PROTOCOLS; do
    jf="$EVAL/judge/$P/$b/${name}_seed42_answer.jsonl"
    if [ -f "$jf" ] && "$VOPD_PY" - "$jf" "$(nq $b)" <<'PY'
import json,sys
try: r=json.load(open(sys.argv[1]))
except Exception: r=[json.loads(l) for l in open(sys.argv[1])]
sys.exit(0 if len(r)==int(sys.argv[2]) else 1)
PY
    then echo ">>> judge $P $b $name 已有, 跳过"; continue; fi
    echo ">>> judge $P $b $name $(date)"
    "$VOPD_PY" judge_qwenlm.py --benchmark "$b" --model "${name}_seed42" --protocol "$P" --answer_path "$ans" --save_path "$jf" \
        --api_base "http://localhost:$JP/v1" --judge_model openai/gpt-oss-120b 2>&1 | tail -2
    "$VOPD_PY" cal_acc.py --benchmark "$b" --judge_json "$jf" --benchmark_json "$EVAL/$(bjson $b)" 2>&1 | tail -3 | sed 's/^/    /'
  done
done; done
# ---------- 3. 汇总 ----------
echo "== 汇总 $TAG (judge 口径 %; rule-first | pure-llm)"
"$VOPD_PY" - "$EVAL" "$BENCHES" "$PROTOCOLS" $NAMES $EXTRA_NAMES <<'PY'
import json, os, sys
ev, benches, protos, names = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4:]
def acc(f):
    if not os.path.exists(f): return None
    try: r = json.load(open(f))
    except Exception: r = [json.loads(l) for l in open(f)]
    return 100*sum(str(x.get('judge','')).lower().startswith('yes') for x in r)/len(r), len(r)
print(f"{'model':24s}" + "".join(f"{b+'/'+p:>22s}" for b in benches for p in protos))
for n in names:
    row = f"{n:24s}"
    for b in benches:
        for p in protos:
            a = acc(f"{ev}/judge/{p}/{b}/{n}_seed42_answer.jsonl")
            row += f"{'-':>22s}" if a is None else f"{a[0]:17.2f}(n={a[1]})"
    print(row)
PY
echo "EVAL_OFF6K EXIT=0 $TAG $(date)"
