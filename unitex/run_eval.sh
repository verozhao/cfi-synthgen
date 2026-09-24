#!/usr/bin/env bash
# UniTEX step 1 text-fidelity evaluation, end to end:
#
#   prepare -> [GPU] run_unitex -> vae_ceiling -> renders -> eval_text -> report
#
# Usage (from anywhere):
#   unitex/run_eval.sh                          # all steps
#   STEPS="eval report" unitex/run_eval.sh      # only some steps
#   DRY_RUN=1 unitex/run_eval.sh                # fake UniTEX outputs, no GPU (plumbing test)
#
# Environments. Three pythons can be involved, set them to what the machine has:
#   PY         numpy + pillow (+ rapidfuzz), the OCR engine (paddleocr on the server, Apple Vision
#              needs only macOS), torch + diffusers for the VAE step. The UniTEX env works.
#   UNITEX_PY  the UniTEX env (torch, diffusers==0.32.2, nvdiffrast, slangtorch, ...), GPU >= 40 GB.
#   BPY        a python with bpy 4.2 for mvgen.py renders (this repo's .venv).
# Splitting machines: run "prepare" anywhere, rsync $EVAL_DIR to the server, run "unitex"
# (and "vae" on GPU) there, rsync back or run the rest there too. Paths inside the eval dir are
# relative to it except run_log.jsonl host info.
#
# OCR. For numbers to report use paddle (what GlyphAnchor's InfoTextBench uses) for BOTH the
# ground truth (prepare) and the stages (eval), ideally after correcting gt_text.txt by hand
# (set its status line to "# status: manual"). vision is for local triage.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$HERE")}"

EVAL_DIR="${EVAL_DIR:-$REPO/../unitex_eval}"                   # eval inputs and all outputs
BUNDLE_V2="${BUNDLE_V2:-/Users/test/CFI-3DGen/approved_bundle_v2}"
SKUS="${SKUS:-$HERE/eval_skus.txt}"                             # file or comma list
UNITEX_ROOT="${UNITEX_ROOT:-$HOME/UniTEX}"                      # UniTEX checkout (cwd for run_unitex)
RUN="${RUN:-unitex_s63}"                                        # per-SKU output subdir, results_$RUN
SEED="${SEED:-63}"
OCR="${OCR:-paddle}"                                            # paddle | vision
PY="${PY:-python}"
UNITEX_PY="${UNITEX_PY:-python}"
BPY="${BPY:-${CFI_BPY_PYTHON:-$REPO/.venv/bin/python}}"
RENDER_DEVICE="${RENDER_DEVICE:-}"                              # mvgen --device (empty: mvgen default)
# VAE source: a FLUX.1-dev diffusers dir or repo id, or a single file plus config dir
FLUX_DIR="${FLUX_DIR:-black-forest-labs/FLUX.1-dev}"
VAE_FILE="${VAE_FILE:-}"                                        # e.g. .../ae.safetensors
VAE_CONFIG="${VAE_CONFIG:-}"                                    # e.g. .../vaecfg
VAE_DEVICE="${VAE_DEVICE:-}"                                    # cuda | cpu (empty: cuda when available)
EXTRA_LORA="${EXTRA_LORA:-}"                                    # "path1 path2" our LoRAs (optional)
EXTRA_LORA_WEIGHTS="${EXTRA_LORA_WEIGHTS:-}"                    # "1.0 1.0"
DRY_RUN="${DRY_RUN:-}"
STEPS="${STEPS:-prepare unitex vae render eval report}"

has_step() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# ── 1. prepare: mesh.glb, square ref.png, GT OCR of the photo, gt_text.txt, baseline.glb ─────
if has_step prepare; then
  echo "== prepare -> $EVAL_DIR"
  "$PY" -m unitex.prepare_eval --out "$EVAL_DIR" --bundle-v2 "$BUNDLE_V2" --skus "$SKUS" --backend "$OCR"
  echo "   review and correct $EVAL_DIR/<sku>/gt_text.txt, then set '# status: manual'"
fi

# ── 2. [GPU] stock UniTEX over every SKU (resumable, per-SKU errors are logged) ─────────────
if has_step unitex; then
  echo "== run_unitex ($RUN, seed $SEED${DRY_RUN:+, dry run})"
  args=(--eval-dir "$(cd "$EVAL_DIR" && pwd)" --run-name "$RUN" --seed "$SEED" --resume)
  [[ -n "$DRY_RUN" ]] && args+=(--dry-run)
  if [[ -n "$EXTRA_LORA" ]]; then
    # shellcheck disable=SC2206
    args+=(--add-lora-path $EXTRA_LORA --add-lora-weights $EXTRA_LORA_WEIGHTS)
  fi
  if [[ -n "$DRY_RUN" ]]; then
    "$UNITEX_PY" "$HERE/run_unitex.py" "${args[@]}"
  else
    (cd "$UNITEX_ROOT" && "$UNITEX_PY" "$HERE/run_unitex.py" --unitex-root . "${args[@]}")
  fi
fi

# ── 3. FLUX VAE ceiling of the framed reference (512 and 1024) ──────────────────────────────
if has_step vae; then
  echo "== vae_ceiling"
  vae_args=(--eval-dir "$EVAL_DIR" --run-name "$RUN")
  if [[ -n "$VAE_FILE" ]]; then
    vae_args+=(--vae-file "$VAE_FILE" --vae-config "$VAE_CONFIG")
  else
    vae_args+=(--flux-dir "$FLUX_DIR")
  fi
  [[ -n "$VAE_DEVICE" ]] && vae_args+=(--device "$VAE_DEVICE")
  "$PY" -m unitex.vae_ceiling "${vae_args[@]}"
fi

# ── 4. front renders at 1024: UniTEX bake and the current pipeline's textured.glb ────────────
if has_step render; then
  echo "== renders (mvgen.py views, front, 1024)"
  dev_args=()
  [[ -n "$RENDER_DEVICE" ]] && dev_args=(--device "$RENDER_DEVICE")
  # ${a[@]+"${a[@]}"}: an empty array under set -u is an error in bash < 4.4 (macOS /bin/bash 3.2).
  # </dev/null keeps the renderer from reading the rest of skus.txt off the loop's stdin.
  while read -r sku; do
    d="$EVAL_DIR/$sku"
    [[ -d "$d" ]] || continue
    if [[ -f "$d/$RUN/textured_mesh.glb" && ! -f "$d/$RUN/dry_run.json" && ! -f "$d/$RUN/renders/baked/view_00.png" ]]; then
      "$BPY" "$REPO/mvgen.py" --mode views --glb "$d/$RUN/textured_mesh.glb" --views 0 --res 1024 \
        --out "$d/$RUN/renders/baked" ${dev_args[@]+"${dev_args[@]}"} </dev/null
    fi
    if [[ -f "$d/baseline.glb" && ! -f "$d/renders/baseline/view_00.png" ]]; then
      "$BPY" "$REPO/mvgen.py" --mode views --glb "$d/baseline.glb" --views 0 --res 1024 \
        --yaw-policy cfi3dgen --out "$d/renders/baseline" ${dev_args[@]+"${dev_args[@]}"} </dev/null
    fi
  done < "$EVAL_DIR/skus.txt"
fi

# ── 5. score every stage against the photo GT (renders still missing are made lazily) ───────
if has_step eval; then
  echo "== eval_text"
  eval_args=(--eval-dir "$EVAL_DIR" --run-name "$RUN" --backend "$OCR" --blender-python "$BPY")
  [[ -n "$RENDER_DEVICE" ]] && eval_args+=(--render-device "$RENDER_DEVICE")
  "$PY" -m unitex.eval_text "${eval_args[@]}"
fi

# ── 6. static HTML review sheet ─────────────────────────────────────────────────────────────
if has_step report; then
  echo "== report"
  "$PY" -m unitex.report --results "$EVAL_DIR/results_$RUN"
  echo "   open $EVAL_DIR/results_$RUN/index.html"
fi
