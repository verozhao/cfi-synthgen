#!/bin/bash
# Text placement test on thanos6 with the final 1024 LoRA, no retraining. Writes only new files
# (<sku>/text_reg.json, <sku>/front_registered.png, <sku>/text_check.json) and new run dirs
# (<sku>/pos_<variant>) inside $R. Every launch waits for a GPU with no processes on it.
#
#   UNITEX_ROOT=<work folder> bash positions_pilot.sh <eval dir name> <sku list file> <gpus> <variant>...
#
# variants (glyph placement x glyph kind, everything else as in the final 1024 eval g1024_final):
#   reg_fixed  registered placement, fixed font renders     (placement alone)
#   old_box    bbox-fit placement, text drawn to its box    (size alone)
#   reg_box    registered placement, text drawn to its box
#   reg_gt     registered placement, crops of the registered photo on the front view
# Registered placement: anchors.py --register-run g1024_final_photofront (bbox fit, then the
# photo_front SIFT homography). SKUs whose photo did not register keep the bbox fit.
set -u
R=${UNITEX_ROOT:?set UNITEX_ROOT to the UniTEX work folder}
source $R/env.sh
EVAL=$1; SKUS=$2; GPUS=$3; shift 3; VARIANTS="$*"
REPO=${REPO:-$R/repos/cfi-synthgen-positions}
ANCHOR_RUN=${ANCHOR_RUN:-unitex_s63}
REG_RUN=${REG_RUN:-g1024_final_photofront}
LORA=$R/runs/trainAB_1024_pos05/pytorch_lora_weights.safetensors
INFER="--view-res 1024 --pos-scale 0.5 --shift-mu 2.19 --delight-res 512 --glyph-mode center"
export CFI_ROOT=$REPO PYTHONPATH=$REPO
PY=$R/venv/bin/python
LOG=$R/logs/positions_pilot.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }

# the runner appends its repo to sys.path, so an older checkout on PYTHONPATH would win
got=$(cd /tmp && $PY -c "import unitex.glyph_tokens as g; print(g.__file__)")
case "$got" in $REPO/*) ;; *) log "ABORT: unitex resolves to $got, not $REPO"; exit 1 ;; esac

gpu_free() {   # no compute process at all on GPU $1
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$1 '$1 == g {print $2}')
  [ -n "$uuid" ] && ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"
}

# 1. layouts on the CPU. text_check.json re-lifts without registration and must equal text.json,
#    which proves these anchors arguments are the ones text.json was made with.
while read -r sku; do
  d=$R/$EVAL/$sku
  [ -f $d/text_reg.json ] && continue
  (cd $REPO && $PY -m unitex.anchors --eval-dir $R/$EVAL --geometry unitex --run-name $ANCHOR_RUN --sku $sku \
     --out $d/text_check.json) >> $LOG 2>&1 < /dev/null
  if ! $PY - "$d/text.json" "$d/text_check.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
sys.exit(0 if (a["res"], a["items"]) == (b["res"], b["items"]) else 1)     # what the glyph tokens read
EOF
  then log "ABORT: $sku text_check.json differs from text.json, check ANCHOR_RUN and the anchors arguments"; exit 1; fi
  (cd $REPO && $PY -m unitex.anchors --eval-dir $R/$EVAL --geometry unitex --run-name $ANCHOR_RUN --sku $sku \
     --register-run $REG_RUN --front-image front_registered.png --out $d/text_reg.json \
     --debug-png $d/text_reg_debug.png) >> $LOG 2>&1 < /dev/null || { log "ABORT: anchors failed for $sku"; exit 1; }
  log "$sku: text_reg.json ($($PY -c "import json; print(json.load(open('$d/text_reg.json'))['lift']['view0_map'])" < /dev/null))"
done < $SKUS

# 2. one variant per free GPU, in parallel, all resumable
glyph_args() {
  case $1 in
    reg_fixed) echo "--glyph-json-name text_reg.json" ;;
    old_box)   echo "--glyph-json-name text.json --glyph-kind box" ;;
    reg_box)   echo "--glyph-json-name text_reg.json --glyph-kind box" ;;
    reg_gt)    echo "--glyph-json-name text_reg.json --glyph-kind gt" ;;
    *) return 1 ;;
  esac
}
for v in $VARIANTS; do glyph_args $v > /dev/null || { log "ABORT: unknown variant $v"; exit 1; }; done
pids=()
for v in $VARIANTS; do
  g=""
  until [ -n "$g" ]; do
    for c in ${GPUS//,/ }; do
      in_use=0
      for p in "${pids[@]+"${pids[@]}"}"; do [ "${p%%:*}" = "$c" ] && kill -0 "${p#*:}" 2>/dev/null && in_use=1; done
      [ $in_use -eq 0 ] && gpu_free $c && { g=$c; break; }
    done
    [ -n "$g" ] || sleep 60
  done
  log "variant $v on GPU $g ($(wc -l < $SKUS) skus)"
  CUDA_VISIBLE_DEVICES=$g $PY $REPO/unitex/run_unitex_lowmem.py --unitex-root $R/repos/train/UniTEX \
    --eval-dir $R/$EVAL --run-name pos_$v --seed 63 --resume --skus $SKUS --texture-lora $LORA $INFER \
    $(glyph_args $v) >> $R/logs/pos_${v}_$EVAL.log 2>&1 < /dev/null &
  pids+=("$g:$!")
  sleep 120     # let the run load its models before the next GPU check
done
wait
for v in $VARIANTS; do log "variant $v: $(grep 'Done:' $R/logs/pos_${v}_$EVAL.log | tail -1)"; done
log "POSITIONS_PILOT_DONE $EVAL $VARIANTS"
