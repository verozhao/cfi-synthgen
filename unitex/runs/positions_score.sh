#!/bin/bash
# Mac side of positions_pilot.sh: pull what the text scores need from thanos6 into a new local
# folder, score the front view of every run with Apple Vision, and run the position audit.
# Reads from the server only. Nothing in the existing local eval folders is touched.
#
#   UNITEX_ROOT=<server work folder> bash positions_score.sh <eval dir name> <sku list file> <run>...
#
# Every run is scored with the same crops: --align homography from the photo_front registration
# of g1024_final_photofront, so a line's crop never depends on where a run placed its glyphs.
set -u
SRV=${SRV:-thanos-6}
R=${UNITEX_ROOT:?set UNITEX_ROOT to the server work folder}
EVAL=$1; SKUS=$2; shift 2; RUNS="$*"
LOCAL=${LOCAL:-$HOME/CyLab/unitex_positions}
REG_RUN=${REG_RUN:-g1024_final_photofront}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python3}
mkdir -p $LOCAL/$EVAL

files=(ref.png ref_meta.json gt_text.txt gt_ocr.json manifest_entry.json text.json text_reg.json front_registered.png
       text_reg_debug.png "$REG_RUN/run_info.json" "$REG_RUN/photo_front.png")
for run in $RUNS; do
  files+=("$run/run_info.json" "$run/rmbg_mask_1024.png")
  for f in mv_rgb_w_light.png mv_rgb_delit_512.png mv_rgb.png mv_rgb_generated.png mv_alpha.png glyph_tokens.json \
           processed_image.png rembg_image.png; do
    files+=("$run/cache/$f")
  done
done
while read -r sku; do
  [ -n "$sku" ] || continue
  list=$(mktemp)
  for f in "${files[@]}"; do echo "$sku/$f"; done > $list
  # macOS rsync 2.6.9 has no --ignore-missing-args: a missing optional file is exit code 23
  rsync -a --files-from=$list "$SRV:$R/$EVAL/" "$LOCAL/$EVAL/" < /dev/null 2> $list.err
  rc=$?
  rm -f $list
  if [ $rc -ne 0 ] && [ $rc -ne 23 ]; then cat $list.err >&2; echo "rsync failed for $sku ($rc)" >&2; exit 1; fi
  rm -f $list.err
done < $SKUS

# each variant must have used the glyph kind it is named after
for run in $RUNS; do
  $PY - "$LOCAL/$EVAL" "$run" "$SKUS" <<'EOF'
import collections, json, pathlib, sys
ev, run, skus = pathlib.Path(sys.argv[1]), sys.argv[2], [s.strip() for s in open(sys.argv[3]) if s.strip()]
kinds, files = collections.Counter(), collections.Counter()
for sku in skus:
    p = ev / sku / run / "cache" / "glyph_tokens.json"
    if p.exists():
        g = json.load(open(p))
        kinds.update(i["kind"] for i in g["instances"])
        files[pathlib.Path(g.get("text_json") or "?").name] += 1
print(f"{run}: glyph kinds {dict(kinds)}, layouts {dict(files)}")
EOF
done

for run in $RUNS; do
  (cd $REPO && $PY -m unitex.eval_text --eval-dir $LOCAL/$EVAL --run-name $run --skus $SKUS \
     --stages lit,delit512,delit --backend vision --align homography --register-run $REG_RUN \
     --out $LOCAL/$EVAL/results_${run}_hom)
done
first=${RUNS%% *}
(cd $REPO && $PY -m unitex.position_audit --eval-dir $LOCAL/$EVAL --run $first \
   --scores $LOCAL/$EVAL/results_${first}_hom --stage delit --out $LOCAL/$EVAL/audit_$first)
