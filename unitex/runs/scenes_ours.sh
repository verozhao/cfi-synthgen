#!/bin/bash
# Table scenes for our final pipeline (1024 glyph LoRA + photo front), same settings as tmp/scenes.sh
# (seed 0, 3 scenes x 6 products, 4 cameras, 1536 px, studio HDRI, table textures). CPU only: the
# GPUs run Anudeep's ShapeGen servers and are not ours to use. Low priority.
source /mnt/nvme1n1/veronica_unitex/env.sh
M=g1024_final_photofront
A=$R/unitex_compare/assets/$M
mkdir -p $A
for d in $R/unitex_compare/assets/unitex/*/; do
  s=$(basename $d)
  g=$R/unitex_eval_28/$s/$M/textured_mesh.glb
  [ -f $g ] || g=$R/unitex_eval_uvfix/$s/$M/textured_mesh.glb
  [ -f $g ] || { echo "missing $s"; continue; }
  mkdir -p $A/$s
  cp $g $A/$s/textured.glb
  cp $d/manifest_entry.json $A/$s/
done
echo "assets: $(ls $A | wc -l)"
cd $R/repos/cfi-synthgen
run() {
  local p=$1
  mkdir -p $R/tmp/scene_${M}_$p
  CUDA_VISIBLE_DEVICES="" TMPDIR=$R/tmp/scene_${M}_$p nice -n 10 $R/venv_bpy/bin/python synthgen.py --glbs $A \
    --out $R/unitex_compare/scenes/$M/$p --scenes 3 --products-per-scene 6 --cameras-json cameras_4cam.json \
    --resolution 1536 --hdri hdri/studio.exr --backgrounds textures --placement $p --seed 0 \
    > $R/logs/scene_${M}_$p.log 2>&1
  echo "$p: $(ls $R/unitex_compare/scenes/$M/$p/images 2>/dev/null | wc -l) images"
}
run scatter & run cluster_mid & wait
run cluster_tight & run stacking & wait
echo SCENES_OURS_DONE
