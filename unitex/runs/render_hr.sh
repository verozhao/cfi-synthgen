#!/bin/bash
# High-resolution front views for the zoomable comparison page. Writes only to compare/views_hr/
# (new folder); compare/views/ and every existing page stay as they are. Same mvgen views command as
# the existing pages (same cameras, unlit), front camera only, 1536 px instead of 768. CPU only.
source /mnt/nvme1n1/veronica_unitex/env.sh
OUT=$R/compare/views_hr
LST=$R/tmp/views_hr
mkdir -p $OUT $LST
cd $R/repos/cfi-synthgen
list() {   # list <model>: "sku glb" lines
  local m=$1
  for d in $R/unitex_compare/assets/unitex/*/; do
    s=$(basename $d)
    case $m in
      presummer|unitex) g=$R/unitex_compare/assets/$m/$s/textured.glb ;;
      *) g=$R/unitex_eval_28/$s/$m/textured_mesh.glb
         [ -f $g ] || g=$R/unitex_eval_uvfix/$s/$m/textured_mesh.glb ;;
    esac
    [ -f $g ] && echo "$s $g"
  done
}
render() {
  local m=$1
  list $m > $LST/$m.txt
  mkdir -p $R/tmp/mvviews_hr_$m
  TMPDIR=$R/tmp/mvviews_hr_$m CUDA_VISIBLE_DEVICES="" nice -n 10 $R/venv_bpy/bin/python mvgen.py --mode views \
    --glb-list $LST/$m.txt --views 0 --res 1536 --device cpu --out $OUT/$m > $R/logs/views_hr_$m.log 2>&1
  echo "$m: $(wc -l < $LST/$m.txt) listed, $(ls $OUT/$m 2>/dev/null | wc -l) rendered"
}
set -- presummer unitex unitex_s63_photofront glyph_trainAB glyph_trainAB_photofront g1024_final \
       g1024_final_photofront g1024_ckpt250 g1024_ckpt500 g1024_ckpt1000 g1024_ckpt1500
while [ $# -gt 0 ]; do
  render $1 & [ $# -gt 1 ] && render $2 & [ $# -gt 2 ] && render $3 & wait
  shift $(( $# < 3 ? $# : 3 ))
done
echo VIEWS_HR_DONE
