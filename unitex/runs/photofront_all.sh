#!/bin/bash
# usage: photofront_all.sh <gpu>
# Photo front on the 28 eval SKUs for stock UniTEX and the A+B glyph LoRA (new run dirs, the
# source runs are read only), views with the cameras of the earlier pages, comparison page.
source /mnt/nvme1n1/veronica_unitex/env.sh
GPU=${1:-0}
REPO=$R/repos/cfi-synthgen
cd $REPO
for src in unitex_s63 glyph_trainAB; do
  dst=${src}_photofront
  for E in unitex_eval_28:$R/tmp/eval_24.txt unitex_eval_uvfix:$R/unitex_eval_uvfix/skus.txt; do
    D=${E%%:*}; L=${E#*:}
    CUDA_VISIBLE_DEVICES=$GPU $R/venv/bin/python unitex/photo_front.py --unitex-root $R/repos/train/UniTEX \
      --eval-dir $R/$D --src-run $src --run-name $dst --skus $L >> $R/logs/photofront_$dst.log 2>&1
  done
  : > $R/tmp/views_$dst.txt
  for s in $(cat $R/tmp/eval_24.txt); do
    g=$R/unitex_eval_28/$s/$dst/textured_mesh.glb; [ -f $g ] && echo "$s $g" >> $R/tmp/views_$dst.txt
  done
  for s in $(cat $R/unitex_eval_uvfix/skus.txt); do
    g=$R/unitex_eval_uvfix/$s/$dst/textured_mesh.glb; [ -f $g ] && echo "$s $g" >> $R/tmp/views_$dst.txt
  done
  mkdir -p $R/tmp/mvviews_$dst
  TMPDIR=$R/tmp/mvviews_$dst CUDA_VISIBLE_DEVICES="" $R/venv_bpy/bin/python mvgen.py --mode views \
    --glb-list $R/tmp/views_$dst.txt --views 0,3,2,1 --res 768 --device cpu --out $R/compare/views/$dst \
    > $R/logs/views_$dst.log 2>&1
  echo "$dst: $(ls $R/compare/views/$dst 2>/dev/null | wc -l) skus with views"
done
$R/venv/bin/python unitex/compare_runs.py --out $R/compare/UniTEX_photo_front.html \
  --photos $R/compare/views/photos --titles $R/compare/titles --views 0,3,1 \
  --title "UniTEX text: real photo on the front view before baking" \
  --note "Photo front: the product photo is aligned to UniTEX's generated front view (SIFT homography) and its fine detail replaces that view where the surface faces the camera, then UniTEX bakes the texture as usual. Inference only, nothing is retrained. Glyph LoRA A+B: 1000 steps at 512 px per view, 4-bit FLUX base, 32 lab products plus 336 Google Scanned Objects." \
  --model "Pre-summer pipeline=$R/compare/views/presummer" \
  --model "UniTEX (stock)=$R/compare/views/unitex" \
  --model "UniTEX (stock) + photo front=$R/compare/views/unitex_s63_photofront" \
  --model "Glyph LoRA A+B=$R/compare/views/glyph_trainAB" \
  --model "Glyph LoRA A+B + photo front=$R/compare/views/glyph_trainAB_photofront"
echo PHOTOFRONT_DONE
