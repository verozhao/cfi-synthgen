#!/bin/bash
# usage: fast_products.sh <gpu> <eval dir> <sku list file | a,b,c> <run name> [run_unitex_fast.py flags]
# The two GPU steps of the texture pipeline on one 24 GB GPU, with the speed options of run_unitex_fast.py:
#   1. views at 1024 px (crop layout tokens, delight at 512 with --delight-shading 4), no bake  -> <run name>
#   2. photo_front.py (auto registration, whole text lines, 75 deg, 4096 px texture), one bake with the
#      sampling tree height for large point sets                                                -> <run name>_pf
# The options that leave the views unchanged are always on. Lossy ones are passed through, e.g. the faster setting
# that kept every front of the 28 eval products (DESIGN.md, step 9):
#   fast_products.sh 1 $R/unitex_eval_28 skus.txt fast20 --texture-steps 20 --delight-steps 8
# Each step starts only when the GPU is free (gpu_ok.sh) and waits otherwise. Step 2 runs on every product that
# has views, also when step 1 failed on some. Each step prints its own "Done: N ok, M errors".
# Env: UNITEX_ROOT (server work folder), LORA (texture LoRA), REPO and UX (this repo and the patched UniTEX checkout
# there, defaults below).
source ${UNITEX_ROOT:?set UNITEX_ROOT to the server work folder}/env.sh
GPU=${1:?gpu}; EVAL=${2:?eval dir}; SKUS=${3:?skus}; NAME=${4:?run name}; shift 4
REPO=${REPO:-$R/repos/cfi-synthgen}
UX=${UX:-$R/repos/train/UniTEX}
LORA=${LORA:?set LORA to the texture LoRA (pytorch_lora_weights.safetensors)}
export CFI_ROOT=$REPO PYTHONPATH=$REPO
wait_gpu() { local n=0; until bash $REPO/unitex/runs/gpu_ok.sh $GPU; do [ $((n % 20)) -eq 0 ] && echo "GPU $GPU busy, waiting"; n=$((n + 1)); sleep 30; done; }
cd $REPO
wait_gpu
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU $R/venv/bin/python unitex/run_unitex_fast.py --unitex-root $UX --eval-dir $EVAL \
  --run-name $NAME --seed 63 --skus $SKUS --texture-lora $LORA --view-res 1024 --pos-scale 0.5 --shift-mu 2.19 \
  --delight-res 512 --glyph-json-name text_reg.json --glyph-layout --glyph-layout-mode tokens \
  --glyph-layout-kind crop --delight-shading 4 --no-bake "$@"
rc1=$?
t1=$(date +%s)
wait_gpu
t2=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU UNITEX_BAKE_ANGLE=105 UNITEX_BAKE_TEX=4096 $R/venv/bin/python unitex/photo_front.py \
  --unitex-root $UX --eval-dir $EVAL --src-run $NAME --run-name ${NAME}_pf --skus $SKUS --register auto \
  --flow-below 0.8 --flow-min 0.9 --text-json text_reg.json --text-pad 1.0 --text-min-ncc 0.8 --one-bake \
  --fps-height 9
rc2=$?
t3=$(date +%s)
echo "views $((t1 - t0)) s (exit $rc1), photo front and bake $((t3 - t2)) s (exit $rc2), both with their model loading"
exit $((rc1 | rc2))
