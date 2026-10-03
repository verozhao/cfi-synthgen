#!/bin/bash
# 1024 px run on thanos6, unattended. Writes only inside $R. GPUs 1 to 3 train (DDP), GPU 0 (shared
# with Kiran's 53 MiB web app, which is never touched) evaluates checkpoints. Every launch waits for
# tmp/gpu_ok.sh. Existing runs and eval outputs are never modified: everything new has a new name.
source /mnt/nvme1n1/veronica_unitex/env.sh
cd $R
LOG=$R/logs/orchestrate_1024.log
REPO=$R/repos/cfi-synthgen
RUN=trainAB_1024_pos05
STEPS=2000; CKPT=250; EVAL_AT="500 1000 1500"
INIT=$R/runs/trainAB/pytorch_lora_weights.safetensors
SUB28="016000233164 079200049836 078742371627 072360002031 017082883896"   # unitex_eval_28
SUBFIX="611269818994"                                                      # unitex_eval_uvfix (Red Bull)
INFER="--view-res 1024 --pos-scale 0.5 --shift-mu 2.19 --delight-res 512 --glyph-json-name text.json --glyph-mode center"

log() { echo "$(date '+%F %T') $*" >> $LOG; }
wait_gpu() { local g=$1 n=0; until $R/tmp/gpu_ok.sh $g; do [ $((n % 20)) -eq 0 ] && log "GPU $g busy, waiting"; n=$((n + 1)); sleep 30; done; }

eval_set() {   # eval_set <gpu> <run_name> <lora> <eval_dir> <sku_file>
  local g=$1 run=$2 lora=$3 D=$4 L=$5
  wait_gpu $g
  log "eval $run on $D ($(wc -l < $L) skus, GPU $g)"
  CUDA_VISIBLE_DEVICES=$g $R/venv/bin/python $REPO/unitex/run_unitex_lowmem.py --unitex-root $R/repos/train/UniTEX \
    --eval-dir $R/$D --run-name $run --seed 63 --resume --skus $L --texture-lora $lora $INFER \
    >> $R/logs/eval_${run}_$D.log 2>&1 < /dev/null
  log "eval $run on $D: $(grep 'Done:' $R/logs/eval_${run}_$D.log | tail -1)"
  CUDA_VISIBLE_DEVICES=$g $R/venv/bin/python $REPO/unitex/photo_front.py --unitex-root $R/repos/train/UniTEX \
    --eval-dir $R/$D --src-run $run --run-name ${run}_photofront --skus $L >> $R/logs/photofront_${run}.log 2>&1 < /dev/null
  log "photo front $run on $D: $(grep 'Done:' $R/logs/photofront_${run}.log | tail -1)"
}

views() {   # views <run_name>: front, left, back, right at 768 on the CPU, cameras of the earlier pages
  local run=$1 lst=$R/tmp/views_$1.txt
  : > $lst
  for D in unitex_eval_28 unitex_eval_uvfix; do
    for g in $R/$D/*/$run/textured_mesh.glb; do
      [ -f $g ] && echo "$(basename $(dirname $(dirname $g))) $g" >> $lst
    done
  done
  mkdir -p $R/tmp/mvviews_$run
  (cd $REPO && TMPDIR=$R/tmp/mvviews_$run CUDA_VISIBLE_DEVICES="" $R/venv_bpy/bin/python mvgen.py --mode views \
     --glb-list $lst --views 0,3,2,1 --res 768 --device cpu --out $R/compare/views/$run > $R/logs/views_$run.log 2>&1)
  log "views $run: $(ls $R/compare/views/$run 2>/dev/null | wc -l) skus"
}

page() {   # page <out.html> <title> <note> <skus or ""> <label=run>...
  local out=$1 title=$2 note=$3 skus=$4; shift 4
  local models=(--model "Pre-summer pipeline=$R/compare/views/presummer" --model "UniTEX (stock)=$R/compare/views/unitex")
  for m in "$@"; do models+=(--model "${m%%=*}=$R/compare/views/${m#*=}"); done
  (cd $REPO && $R/venv/bin/python unitex/compare_runs.py --out $out --photos $R/compare/views/photos \
     --titles $R/compare/titles --views 0,3,1 --title "$title" --note "$note" ${skus:+--skus $skus} "${models[@]}" >> $LOG 2>&1)
}

log "orchestrator start: $RUN, $STEPS steps, checkpoints every $CKPT, evals at $EVAL_AT"

# 1. data
until [ $(grep -l "rendered in" $R/logs/mv_gso_1024r_*.log 2>/dev/null | wc -l) -eq 3 ] || \
      ! pgrep -f "[m]vgen.py --mode train --out $R/data/mv_gso_1024" > /dev/null; do sleep 60; done
log "GSO 1024 renders done: $(grep -h 'rendered in' $R/logs/mv_gso_1024r_*.log | tr '\n' ' ')"
[ $(grep -l "rendered in" $R/logs/mv_gso_1024r_*.log | wc -l) -eq 3 ] || log "WARNING: a GSO shard ended without finishing, make_1024_sets lists the missing uids"
NAMES=mv_gso bash $R/tmp/make_1024_sets.sh >> $LOG 2>&1
(cd $REPO && CUDA_VISIBLE_DEVICES="" $R/venv/bin/python -m unitex.flux_dataset check --root $R/data/mv_gso_1024 \
   --view-res 1024 --json $R/tmp/dscheck_gso1024.json > $R/logs/dscheck_gso1024.log 2>&1; \
 echo "$(date '+%F %T') dataset check mv_gso_1024: $(tail -1 $R/logs/dscheck_gso1024.log)" >> $LOG) &

# 2. training on GPUs 1-3
for g in 1 2 3; do wait_gpu $g; done
log "train $RUN start (GPUs 1,2,3)"
POS_SCALE=0.5 nohup $R/tmp/train_run_1024.sh 1,2,3 $RUN "$R/data/train_A_1024,$R/data/mv_gso_1024" $STEPS 1 $CKPT $INIT 29521 \
  > $R/logs/train_$RUN.log 2>&1 < /dev/null &
TRAIN_PID=$!

# 3. checkpoint evals on GPU 0 (6 text-heavy SKUs)
echo $SUB28 | tr ' ' '\n' > $R/tmp/sub28.txt
echo $SUBFIX | tr ' ' '\n' > $R/tmp/subfix.txt
labels=("Glyph LoRA A+B, 512=glyph_trainAB" "Glyph LoRA A+B, 512 + photo front=glyph_trainAB_photofront")
for N in $EVAL_AT; do
  until [ -f $R/runs/$RUN/checkpoint-$N/pytorch_lora_weights.safetensors ] || ! kill -0 $TRAIN_PID 2>/dev/null; do sleep 120; done
  if [ ! -f $R/runs/$RUN/checkpoint-$N/pytorch_lora_weights.safetensors ]; then
    log "training exited before checkpoint-$N: $(tr '\r' '\n' < $R/logs/train_$RUN.log | grep -E 'Error|loss=' | tail -2)"
    break
  fi
  sleep 60          # let the checkpoint finish writing
  run=g1024_ckpt$N
  log "checkpoint-$N ready"
  eval_set 0 $run $R/runs/$RUN/checkpoint-$N/pytorch_lora_weights.safetensors unitex_eval_28 $R/tmp/sub28.txt
  eval_set 0 $run $R/runs/$RUN/checkpoint-$N/pytorch_lora_weights.safetensors unitex_eval_uvfix $R/tmp/subfix.txt
  views $run; views ${run}_photofront
  labels+=("1024, checkpoint $N=$run" "1024, checkpoint $N + photo front=${run}_photofront")
  page $R/compare/UniTEX_1024_progress.html "UniTEX text at 1024 px per view: training progress" \
    "Glyph LoRA trained at 1024 px per view (position scale 0.5, seeded from the 512 A+B LoRA, 3 GPUs x 1 sample per step). Delight pass at 512 with the 1024 detail added back. Photo front: the product photo's detail on the front view before baking (homography alignment, skipped when the photo does not register)." \
    "$(echo $SUB28 $SUBFIX | tr ' ' ',')" "${labels[@]}"
  log "progress page updated (checkpoint $N)"
done

# 4. final LoRA on all 28 SKUs, four GPUs
wait $TRAIN_PID
log "train $RUN exit: $(tr '\r' '\n' < $R/logs/train_$RUN.log | grep -o 'loss=[0-9.]*' | tail -1)"
if [ -f $R/runs/$RUN/pytorch_lora_weights.safetensors ]; then
  run=g1024_final
  for D in unitex_eval_28 unitex_eval_uvfix; do
    L=$R/tmp/eval_24.txt; [ $D = unitex_eval_uvfix ] && L=$R/unitex_eval_uvfix/skus.txt
    split -n r/4 -d $L $R/tmp/final_${D}_
  done
  for k in 0 1 2 3; do
    ( for D in unitex_eval_28 unitex_eval_uvfix; do
        f=$R/tmp/final_${D}_0$k; [ -s $f ] && eval_set $k $run $R/runs/$RUN/pytorch_lora_weights.safetensors $D $f
      done ) &
  done
  wait
  views $run; views ${run}_photofront
  page $R/compare/UniTEX_1024_final.html "UniTEX text: 1024 px per view and photo front" \
    "Glyph LoRA at 1024 px per view: $STEPS steps x 3 samples (position scale 0.5, 4-bit FLUX base, seeded from the 512 A+B LoRA: 1000 steps x 4 samples). Delight pass at 512 with the 1024 detail added back. Photo front: the product photo's detail on the front view before baking (homography alignment, skipped when the photo does not register). Inference for the trained rows also places the photo's OCR text as glyph tokens." "" \
    "Glyph LoRA A+B, 512=glyph_trainAB" "Glyph LoRA A+B, 512 + photo front=glyph_trainAB_photofront" \
    "Glyph LoRA A+B, 1024=$run" "Glyph LoRA A+B, 1024 + photo front=${run}_photofront"
  log "final page written"
else
  log "no final LoRA, final eval skipped"
fi
log "ORCH1024_DONE"
touch $R/ORCH1024_DONE
