#!/bin/bash
# 在 GPU hold 内跑 negative-history 全套验证 (新套件 + 四套回归)
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
cd /scratch/nkw3mr/Vision-OPD/OPD-Aha-sup
echo "========== test_neghist =========="
CUDA_VISIBLE_DEVICES="" "$VOPD_PY" scripts/test_neghist.py; r1=$?
echo "========== 回归: sup/floor/tanh/gamma =========="
for t in test_sup test_floor test_tanh test_gamma; do
  CUDA_VISIBLE_DEVICES="" "$VOPD_PY" scripts/$t.py > /tmp/$t.out 2>&1
  echo "$t: $(tail -1 /tmp/$t.out)"
done
exit $r1
