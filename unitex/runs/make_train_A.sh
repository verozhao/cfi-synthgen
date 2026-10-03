R=/mnt/nvme1n1/veronica_unitex
# twins of eval products (same product line / label design) found by leak_check.py, kept out of training
cat > $R/data/eval_siblings.txt <<EOF
002e90258258
b5852b274b7e
013409351505
31a80a5fcfe2
4bf510bace29
016000459007
017082600738
021cde956d37
611269001280
611269002034
611269235685
611269331240
611269546019
6221c1446513
7ee47af280cb
048001213487
048001354500
051500243220
051500255162
273553df4004
661cfa90a821
eaea257ffbe4
EOF
S=$R/data/train_A
rm -rf $S; mkdir -p $S/render/cfi $S/render_random/cfi $S/caption/cfi
: > $S/uids.txt
for d in $R/data/mv_cfi/render/cfi/*/; do
  u=$(basename $d)
  grep -qx $u $R/data/eval_siblings.txt && continue
  ln -s $d $S/render/cfi/$u
  ln -s $R/data/mv_cfi/render_random/cfi/$u $S/render_random/cfi/$u
  ln -s $R/data/mv_cfi/caption/cfi/$u $S/caption/cfi/$u
  echo "cfi/$u" >> $S/uids.txt
done
python3 -c "import json; json.dump([l.strip() for l in open('$S/uids.txt')], open('$S/training_uid.json','w'))"
echo "train_A uids: $(wc -l < $S/uids.txt)"
