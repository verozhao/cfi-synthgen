#!/bin/bash
# usage: fast_products.sh <gpu> <eval dir> <sku list file | a,b,c> <run name> [run_unitex_fast.py flags]
# The texture pipeline on one 24 GB GPU, from prepared product folders to the finished models and a check of them:
#   1. views at 1024 px (crop layout tokens, delight at 512, its shading as brightness only), no bake  -> <run name>
#      (or at 512 px, see VIEW_RES)
#   2. photo_front.py: the real photo on the front (auto registration, whole text lines, 75 deg),
#      one bake at 4096 px, roughness 0.6                                                               -> <run name>_pf
#   3. check_products.py: one row per product with its flags          -> <eval dir>/check_<run name>_pf.md and .csv
# <run name>_pf/textured_mesh.glb is the finished model.
# The speed options that leave the views unchanged are always on. Lossy ones are passed through to step 1, e.g. the
# faster setting that kept every front of the 28 eval products (DESIGN.md, step 9):
#   fast_products.sh 1 $R/unitex_eval_28 skus.txt fast20 --texture-steps 20 --delight-steps 8
# Each step starts only when the GPU is free (gpu_ok.sh) and waits otherwise. Step 2 runs on every product that has
# views, also when step 1 failed on some. Each step prints its own "Done: N ok, M errors".
# Env: UNITEX_ROOT (server work folder), LORA (texture LoRA), REPO and UX (this repo and the patched UniTEX checkout
# there, defaults below). Settings, also env:
#   VIEW_RES=512  the smaller option (default 1024): FLUX runs at 512 px per view with the same LoRA and inputs, and
#                 photo_front.py --view-res 1024 enlarges the views, so the photo still goes on at 1024 px. About 3 min
#                 per product against 11, with large made-up words on backs and sides (DESIGN.md, step 11)
#   LUMA=0        the delight shading with one ratio per colour channel, as before --delight-luma (default 1)
#   ROUGHNESS=R   roughness of the models (default 0.6, the current Gemini pipeline's. UniTEX itself writes 1.0)
#   FINISH=1      off by default. Also runs back_panel.py on the photo front (a typeset back panel from product
#                 records, plain sides) -> <run name>_final, the bake then happens there. Needs OFF=<folder of
#                 <sku>.json records>. In table scenes these plain sides and blurred backs looked worse than the
#                 generated ones (DESIGN.md, step 10), so the pipeline's model is the one of step 2.
source ${UNITEX_ROOT:?set UNITEX_ROOT to the server work folder}/env.sh
GPU=${1:?gpu}; EVAL=${2:?eval dir}; SKUS=${3:?skus}; NAME=${4:?run name}; shift 4
REPO=${REPO:-$R/repos/cfi-synthgen}
UX=${UX:-$R/repos/train/UniTEX}
LORA=${LORA:?set LORA to the texture LoRA (pytorch_lora_weights.safetensors)}
ROUGHNESS=${ROUGHNESS:-0.6}
case ${VIEW_RES:-1024} in
  1024) RES_FLAGS="--view-res 1024 --pos-scale 0.5 --shift-mu 2.19 --delight-res 512"; PF_RES=;;
  512) RES_FLAGS="--view-res 512"; PF_RES="--view-res 1024";;
  *) echo "VIEW_RES must be 1024 or 512"; exit 2;;
esac
[ "${LUMA:-1}" = 1 ] && LUMA_FLAG=--delight-luma || LUMA_FLAG=
[ "${FINISH:-0}" = 1 ] && : ${OFF:?FINISH=1 needs OFF, the folder of product records}
export CFI_ROOT=$REPO PYTHONPATH=$REPO
wait_gpu() { local n=0; until bash $REPO/unitex/runs/gpu_ok.sh $GPU; do [ $((n % 20)) -eq 0 ] && echo "GPU $GPU busy, waiting"; n=$((n + 1)); sleep 30; done; }
cd $REPO
wait_gpu
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU $R/venv/bin/python unitex/run_unitex_fast.py --unitex-root $UX --eval-dir $EVAL \
  --run-name $NAME --seed 63 --skus $SKUS --texture-lora $LORA $RES_FLAGS \
  --glyph-json-name text_reg.json --glyph-layout --glyph-layout-mode tokens \
  --glyph-layout-kind crop --delight-shading 4 $LUMA_FLAG --no-bake "$@"
rc1=$?
t1=$(date +%s)
wait_gpu
t2=$(date +%s)
if [ "${FINISH:-0}" = 1 ]; then PF_BAKE=--no-bake; else PF_BAKE="--one-bake --fps-height 9 --roughness $ROUGHNESS"; fi
CUDA_VISIBLE_DEVICES=$GPU UNITEX_BAKE_ANGLE=105 UNITEX_BAKE_TEX=4096 $R/venv/bin/python unitex/photo_front.py \
  --unitex-root $UX --eval-dir $EVAL --src-run $NAME --run-name ${NAME}_pf --skus $SKUS --register auto \
  --text-json text_reg.json --text-pad 1.0 --text-min-ncc 0.8 $PF_RES $PF_BAKE
rc2=$?
rc3=0
FINAL=
if [ "${FINISH:-0}" = 1 ]; then
  CUDA_VISIBLE_DEVICES=$GPU UNITEX_BAKE_ANGLE=105 UNITEX_BAKE_TEX=4096 $R/venv/bin/python unitex/back_panel.py \
    --unitex-root $UX --eval-dir $EVAL --src-run ${NAME}_pf --run-name ${NAME}_final --skus $SKUS --off-dir $OFF \
    --clean-views 1,3,4,5 --sides-from-edge --clean-front-edges --color-match lut --one-bake --fps-height 9 \
    --roughness $ROUGHNESS
  rc3=$?
  FINAL="--final-run ${NAME}_final"
fi
t3=$(date +%s)
CUDA_VISIBLE_DEVICES="" $R/venv/bin/python unitex/check_products.py --eval-dir $EVAL --run $NAME --skus $SKUS $FINAL
echo "views $((t1 - t0)) s (exit $rc1), photo front and bake $((t3 - t2)) s (exit $((rc2 | rc3))), both with their model loading"
exit $((rc1 | rc2 | rc3))
