#!/bin/bash
# Extra early look: checkpoint 250 of the 1024 run on the six progress SKUs, GPU 0, while GPU 0 would
# otherwise wait for checkpoint 500. Same settings as orchestrate_1024.sh, separate page.
source /mnt/nvme1n1/veronica_unitex/env.sh
cd $R
RUN=trainAB_1024_pos05; N=250; run=g1024_ckpt$N
REPO=$R/repos/cfi-synthgen
LOG=$R/logs/orchestrate_1024.log
LORA=$R/runs/$RUN/checkpoint-$N/pytorch_lora_weights.safetensors
INFER="--view-res 1024 --pos-scale 0.5 --shift-mu 2.19 --delight-res 512 --glyph-json-name text.json --glyph-mode center"
log() { echo "$(date '+%F %T') [ckpt250] $*" >> $LOG; }

until [ -f $LORA ]; do sleep 120; done
sleep 60
until $R/tmp/gpu_ok.sh 0; do sleep 30; done
log "eval start (GPU 0)"
for E in unitex_eval_28:$R/tmp/sub28.txt unitex_eval_uvfix:$R/tmp/subfix.txt; do
  D=${E%%:*}; L=${E#*:}
  CUDA_VISIBLE_DEVICES=0 $R/venv/bin/python $REPO/unitex/run_unitex_lowmem.py --unitex-root $R/repos/train/UniTEX \
    --eval-dir $R/$D --run-name $run --seed 63 --resume --skus $L --texture-lora $LORA $INFER \
    >> $R/logs/eval_${run}_$D.log 2>&1 < /dev/null
  CUDA_VISIBLE_DEVICES=0 $R/venv/bin/python $REPO/unitex/photo_front.py --unitex-root $R/repos/train/UniTEX \
    --eval-dir $R/$D --src-run $run --run-name ${run}_photofront --skus $L >> $R/logs/photofront_${run}.log 2>&1 < /dev/null
done
log "eval: $(grep -h 'Done:' $R/logs/eval_${run}_*.log | tr '\n' ' ')"
for r in $run ${run}_photofront; do
  lst=$R/tmp/views_$r.txt; : > $lst
  for D in unitex_eval_28 unitex_eval_uvfix; do
    for g in $R/$D/*/$r/textured_mesh.glb; do [ -f $g ] && echo "$(basename $(dirname $(dirname $g))) $g" >> $lst; done
  done
  mkdir -p $R/tmp/mvviews_$r
  (cd $REPO && TMPDIR=$R/tmp/mvviews_$r CUDA_VISIBLE_DEVICES="" $R/venv_bpy/bin/python mvgen.py --mode views \
     --glb-list $lst --views 0,3,2,1 --res 768 --device cpu --out $R/compare/views/$r > $R/logs/views_$r.log 2>&1)
done
(cd $REPO && $R/venv/bin/python unitex/compare_runs.py --out $R/compare/UniTEX_1024_ckpt250.html \
   --photos $R/compare/views/photos --titles $R/compare/titles --views 0,3,1 \
   --skus 016000233164,079200049836,078742371627,072360002031,017082883896,611269818994 \
   --title "UniTEX text at 1024 px per view: checkpoint 250 (early look)" \
   --note "Glyph LoRA at 1024 px per view after 250 of 2000 steps (3 samples per step, position scale 0.5, seeded from the 512 A+B LoRA). Delight pass at 512 with the 1024 detail added back. Photo front: the product photo's detail on the front view before baking." \
   --model "Pre-summer pipeline=$R/compare/views/presummer" --model "UniTEX (stock)=$R/compare/views/unitex" \
   --model "Glyph LoRA A+B, 512=$R/compare/views/glyph_trainAB" \
   --model "Glyph LoRA A+B, 512 + photo front=$R/compare/views/glyph_trainAB_photofront" \
   --model "1024, checkpoint 250=$R/compare/views/$run" \
   --model "1024, checkpoint 250 + photo front=$R/compare/views/${run}_photofront" >> $LOG 2>&1)
log "page written: compare/UniTEX_1024_ckpt250.html"
