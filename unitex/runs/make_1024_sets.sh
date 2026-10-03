#!/bin/bash
# 1024 px training sets with the same uids and text.json as the 512 runs. Cameras are the same
# (mvgen, same yaw policy), so a 512 text.json is valid for the 1024 render: it records res 512 and
# unitex.glyph_tokens rescales it to the training view_res.
source /mnt/nvme1n1/veronica_unitex/env.sh
set -e
for name in ${NAMES:-mv_cfi mv_gso}; do
  src=$R/data/$name; dst=$R/data/${name}_1024
  n=0; miss=0
  for u in $(python3 -c "import json; print(' '.join(x.split('/')[1] for x in json.load(open('$src/training_uid.json'))))"); do
    if [ ! -f $dst/render/cfi/$u/0005_rgb.png ] || [ ! -f $dst/render_random/cfi/$u/metadata.json ]; then
      echo "missing 1024 render: $name $u"; miss=$((miss + 1)); continue
    fi
    if [ -f $src/render/cfi/$u/text.json ]; then
      cp $src/render/cfi/$u/text.json $dst/render/cfi/$u/text.json; n=$((n + 1))
    fi
  done
  [ -f $dst/training_uid.mvgen.json ] || cp $dst/training_uid.json $dst/training_uid.mvgen.json
  cp $src/training_uid.json $dst/training_uid.json
  echo "$name: $n text.json copied, $miss uids without a complete 1024 render"
done
S=$R/data/train_A_1024
mkdir -p $S/render/cfi $S/render_random/cfi $S/caption/cfi
for u in $(python3 -c "import json; print(' '.join(x.split('/')[1] for x in json.load(open('$R/data/train_A/training_uid.json'))))"); do
  for sub in render render_random caption; do
    ln -sfn $R/data/mv_cfi_1024/$sub/cfi/$u $S/$sub/cfi/$u
  done
done
cp $R/data/train_A/training_uid.json $S/training_uid.json
echo "train_A_1024: $(ls $S/render/cfi | wc -l) uids, text.json in $(ls $S/render/cfi/*/text.json | wc -l)"
python3 -c "from PIL import Image; import glob; p = sorted(glob.glob('$R/data/mv_gso_1024/render/cfi/*/0000_rgb.png'))[0]; q = sorted(glob.glob('$R/data/mv_gso_1024/render_random/cfi/*/0000_rgb.png'))[0]; print('sizes', Image.open(p).size, Image.open(q).size)"
