#!/bin/bash
# Views of the baked meshes before LTM (cache/wo_LTM) and after (cache/w_LTM) for the walkthrough page.
# New folder walkthrough/renders/, existing outputs untouched. CPU only.
source /mnt/nvme1n1/veronica_unitex/env.sh
OUT=$R/walkthrough/renders
mkdir -p $OUT
cd $R/repos/cfi-synthgen
for run in unitex_s63 g1024_final_photofront; do
  for stage in wo_LTM w_LTM; do
    lst=$OUT/list_${run}_$stage.txt
    : > $lst
    for s in 016000233164 072360002031; do
      echo "$s $R/unitex_eval_28/$s/$run/cache/$stage/textured_mesh.glb" >> $lst
    done
    mkdir -p $R/tmp/mvviews_wt_${run}_$stage
    TMPDIR=$R/tmp/mvviews_wt_${run}_$stage CUDA_VISIBLE_DEVICES="" nice -n 10 $R/venv_bpy/bin/python mvgen.py \
      --mode views --glb-list $lst --views 0,3,2,1,4,5 --res 768 --device cpu --out $OUT/${run}_$stage \
      > $R/logs/walkthrough_${run}_$stage.log 2>&1 &
  done
done
wait
for d in $OUT/*/; do echo "$(basename $d): $(find $d -name 'view_*.png' | wc -l) views"; done
echo WT_RENDERS_DONE
