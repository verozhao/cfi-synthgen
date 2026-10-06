#!/bin/bash
# Unattended run of the rest of step 2 on thanos6. Writes only inside $R. Starts a GPU job only on
# a GPU that is completely free (no process of anyone, under 30 MiB used), and waits otherwise.
source /mnt/nvme1n1/veronica_unitex/env.sh
cd $R
LOG=$R/logs/orchestrate.log
REPO=$R/repos/cfi-synthgen
FIX="611269818994 078742369594 012000064500 603084571239"
STEPS=1000; ACC=4; CKPT=250

log() { echo "$(date '+%F %T') $*" >> $LOG; }

gpu_free() {
  local g=$1 used procs
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  procs=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)
  [ "$used" -lt 30 ] && [ "$procs" -eq 0 ]
}

wait_free() {
  local g=$1 n=0
  until gpu_free $g; do
    [ $((n % 20)) -eq 0 ] && log "GPU $g not free, waiting ($(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader))"
    n=$((n + 1)); sleep 30
  done
  log "GPU $g free"
}

eval_run() {   # eval_run <gpu> <run_name> <lora or "">
  local g=$1 run=$2 lora=$3 extra=""
  [ -n "$lora" ] && extra="--texture-lora $lora"
  for E in unitex_eval_28:$R/tmp/eval_24.txt unitex_eval_uvfix:$R/unitex_eval_uvfix/skus.txt; do
    local D=${E%%:*} L=${E#*:}
    wait_free $g
    log "eval $run on $D (GPU $g)"
    CUDA_VISIBLE_DEVICES=$g $R/venv/bin/python $REPO/unitex/run_unitex_lowmem.py --unitex-root $R/repos/train/UniTEX \
      --eval-dir $R/$D --run-name $run --seed 63 --resume --skus $L --glyph-json-name text.json --glyph-mode center $extra \
      > $R/logs/eval_${run}_$D.log 2>&1
    log "eval $run on $D finished: $(grep 'Done:' $R/logs/eval_${run}_$D.log | tail -1)"
  done
}

branch_A() {
  until ! pgrep -f "[t]ext_regions.py --root $R/data/mv_cfi" >/dev/null; do sleep 30; done
  log "re-verify A set"
  (cd $REPO && CUDA_VISIBLE_DEVICES="" PYTHONPATH=. $R/venv_ocr/bin/python text_regions.py --root $R/data/mv_cfi \
     --backend paddle --reverify >> $R/logs/reverify_A.log 2>&1)
  log "re-verify A done: $(grep 'reverify:' $R/logs/reverify_A.log | tail -1)"
  wait_free 1
  log "train A start (GPU 1)"
  bash $R/tmp/train_run.sh 1 trainA $R/data/train_A $STEPS $ACC $CKPT > $R/logs/train_A.log 2>&1
  log "train A exit $? : $(grep -o 'loss=[0-9.]*' $R/logs/train_A.log | tail -1)"
  if [ -f $R/runs/trainA/pytorch_lora_weights.safetensors ]; then
    eval_run 1 glyph_trainA $R/runs/trainA/pytorch_lora_weights.safetensors
  else
    log "train A produced no final LoRA, eval skipped"
  fi
  touch $R/runs/BRANCH_A_DONE
}

branch_AB() {
  until ! pgrep -f "[t]ext_regions.py --root $R/data/mv_gso" >/dev/null; do sleep 60; done
  log "GSO text regions done: $(ls $R/data/mv_gso/render/cfi/*/text.json | wc -l) text.json"
  until [ -f $R/logs/reverify_A.log ] && grep -q 'reverify:' $R/logs/reverify_A.log; do sleep 30; done
  wait_free 2
  log "train A+B start (GPU 2)"
  bash $R/tmp/train_run.sh 2 trainAB "$R/data/train_A,$R/data/mv_gso" $STEPS $ACC $CKPT > $R/logs/train_AB.log 2>&1
  log "train A+B exit $? : $(grep -o 'loss=[0-9.]*' $R/logs/train_AB.log | tail -1)"
  if [ -f $R/runs/trainAB/pytorch_lora_weights.safetensors ]; then
    eval_run 2 glyph_trainAB $R/runs/trainAB/pytorch_lora_weights.safetensors
  else
    log "train A+B produced no final LoRA, eval skipped"
  fi
  touch $R/runs/BRANCH_AB_DONE
}

log "orchestrator start (steps $STEPS, accumulation $ACC)"
branch_A & PA=$!
branch_AB & PB=$!
until grep -q CONTROL_DONE $R/logs/eval_control.log 2>/dev/null; do sleep 60; done
log "control eval (glyphs, released LoRA) done"
wait $PA $PB
log "all training and evals done, rendering views"

# views of every UniTEX run (same cameras as the earlier comparison), CPU so no GPU is needed
for run in glyph_released glyph_trainA glyph_trainAB; do
  : > $R/tmp/views_$run.txt
  for s in $(cat $R/tmp/eval_24.txt); do
    g=$R/unitex_eval_28/$s/$run/textured_mesh.glb; [ -f $g ] && echo "$s $g" >> $R/tmp/views_$run.txt
  done
  for s in $FIX; do
    g=$R/unitex_eval_uvfix/$s/$run/textured_mesh.glb; [ -f $g ] && echo "$s $g" >> $R/tmp/views_$run.txt
  done
  mkdir -p $R/tmp/mvviews_$run
  (cd $REPO && TMPDIR=$R/tmp/mvviews_$run CUDA_VISIBLE_DEVICES="" $R/venv_bpy/bin/python mvgen.py --mode views \
     --glb-list $R/tmp/views_$run.txt --views 0,3,2,1 --res 768 --device cpu --out $R/compare/views/$run \
     > $R/logs/views_$run.log 2>&1)
  log "views $run: $(ls $R/compare/views/$run 2>/dev/null | wc -l) skus"
done

(cd $REPO && $R/venv/bin/python unitex/compare_runs.py --out $R/compare/UniTEX_glyph_comparison.html \
   --photos $R/compare/views/photos --titles $R/compare/titles --views 0,3,1 \
   --title "UniTEX text: stock vs glyph conditioning" \
   --note "Glyph runs place the photo's OCR text as glyph tokens (center anchoring). Trained runs: 1000 steps x 4 samples, 4-bit FLUX base, warm start from the released UniTEX LoRA. A = 32 approved-bundle models (front-panel glyphs only, eval twins removed). A+B = A plus 336 Google Scanned Objects products with real printed text." \
   --model "Pre-summer pipeline=$R/compare/views/presummer" \
   --model "UniTEX (stock)=$R/compare/views/unitex" \
   --model "UniTEX + glyphs, no training=$R/compare/views/glyph_released" \
   --model "Glyph LoRA trained on A=$R/compare/views/glyph_trainA" \
   --model "Glyph LoRA trained on A+B (with GSO)=$R/compare/views/glyph_trainAB" \
   >> $LOG 2>&1)
log "ORCH_DONE"
touch $R/ORCH_DONE
