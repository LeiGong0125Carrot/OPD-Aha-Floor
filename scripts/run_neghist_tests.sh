#!/bin/bash
# 在 GPU hold 内跑 negative-history 全套验证 (新套件 + 四套回归)
set -uo pipefail
source /sfs/weka/scratch/nkw3mr/Vision-OPD-setup/00_env.sh
cd /scratch/nkw3mr/Vision-OPD/OPD-Aha-sup
echo "========== test_neghist =========="
CUDA_VISIBLE_DEVICES="" "$VOPD_PY" scripts/test_neghist.py; r1=$?
echo "========== test_sref (candidate S) =========="
CUDA_VISIBLE_DEVICES="" "$VOPD_PY" scripts/test_sref.py; r2=$?
echo "========== test_internal_residual (route 1) =========="
CUDA_VISIBLE_DEVICES="" "$VOPD_PY" scripts/test_internal_residual.py; r3=$?
echo "========== test_layer_prior_probe (layer-prior probe, offline) =========="
CUDA_VISIBLE_DEVICES="" PROBE_TEST_STRICT=1 "$VOPD_PY" scripts/test_layer_prior_probe.py; r4=$?
echo "========== test_mismatch_null (arm Amis) =========="
CUDA_VISIBLE_DEVICES="" PROBE_TEST_STRICT=1 "$VOPD_PY" scripts/test_mismatch_null.py; r5=$?
echo "========== test_flip (P1 CFD) =========="
CUDA_VISIBLE_DEVICES="" "$VOPD_PY" scripts/test_flip.py; r6=$?
echo "========== test_pbd (P5 PBD) =========="
CUDA_VISIBLE_DEVICES="" PROBE_TEST_STRICT=1 "$VOPD_PY" scripts/test_pbd.py; r7=$?
echo "========== 回归: sup/floor/tanh/gamma =========="
for t in test_sup test_floor test_tanh test_gamma; do
  CUDA_VISIBLE_DEVICES="" "$VOPD_PY" scripts/$t.py > /tmp/$t.out 2>&1
  echo "$t: $(tail -1 /tmp/$t.out)"
done
exit $(( r1 | r2 | r3 | r4 | r5 | r6 | r7 ))
