#!/usr/bin/env bash
# Clone UniTEX (inference) and UniTEX-FLUX (LoRA training) at the pinned commits and apply our
# patches from unitex/patches/ (GlyphAnchor tokens, fp32 position ids, zero text embeddings,
# per-view resolution, cfi dataset). Idempotent: a repo already at the pinned commit is reused and
# a patch that is already applied is skipped.
#
# Usage:
#   unitex/setup_unitex.sh [TARGET_DIR]                 # default $HOME/unitex_repos
#   UNITEX_URL=/path/UniTEX UNITEX_FLUX_URL=/path/UniTEX-FLUX unitex/setup_unitex.sh /scratch/repos
#                                                       # clone from local mirrors (offline)
#   FORCE=1 unitex/setup_unitex.sh ...                  # reset repos with local changes to the pin
#
# Result:
#   TARGET_DIR/UniTEX        affa1e2 + unitex/patches/UniTEX.patch (when that file exists)
#   TARGET_DIR/UniTEX-FLUX   036a736 + unitex/patches/UniTEX-FLUX.patch
# The patches stay uncommitted, so `git -C TARGET_DIR/UniTEX-FLUX diff --stat` shows them.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
TARGET="${1:-${TARGET:-$HOME/unitex_repos}}"
UNITEX_URL="${UNITEX_URL:-https://github.com/YixunLiang/UniTEX.git}"
UNITEX_FLUX_URL="${UNITEX_FLUX_URL:-https://github.com/lightillusions/UniTEX-FLUX.git}"
UNITEX_COMMIT="affa1e29e665670dbdfd13ee4a3a68a45942df47"
UNITEX_FLUX_COMMIT="036a73699bf9c9948ab81cbba3143bcaf8886006"
FORCE="${FORCE:-0}"

log() { printf '[setup_unitex] %s\n' "$*"; }
die() { printf '[setup_unitex] ERROR: %s\n' "$*" >&2; exit 1; }

# ────────────────────────────────────────────────────────────────────────────
# clone at a pinned commit
# ────────────────────────────────────────────────────────────────────────────

checkout_pinned() {   # url dir commit patch
  local url="$1" dir="$2" commit="$3" patch="$4"
  if [ -d "$dir/.git" ]; then
    local head
    head="$(git -C "$dir" rev-parse HEAD)"
    if [ "$head" != "$commit" ]; then
      [ "$FORCE" = 1 ] || die "$dir is at $head, not $commit (FORCE=1 resets it)"
      git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || git -C "$dir" fetch --quiet origin
      git -C "$dir" checkout --quiet --force "$commit"
    fi
    if ! git -C "$dir" diff --quiet; then
      if [ -f "$patch" ] && git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        log "$(basename "$dir"): at ${commit:0:7}, $(basename "$patch") already applied"
        return 0
      fi
      [ "$FORCE" = 1 ] || die "$dir has local changes that are not $(basename "$patch") (FORCE=1 discards them)"
      git -C "$dir" checkout --quiet --force "$commit"
      git -C "$dir" clean --quiet -fd
    fi
  else
    [ -e "$dir" ] && die "$dir exists and is not a git checkout"
    log "cloning $url -> $dir"
    git clone --quiet "$url" "$dir"
    git -C "$dir" checkout --quiet "$commit" || die "commit $commit not found in $url"
  fi
  log "$(basename "$dir"): at $(git -C "$dir" rev-parse --short HEAD)"
  if [ -f "$patch" ]; then
    git -C "$dir" apply --check "$patch" || die "$(basename "$patch") does not apply to $dir"
    git -C "$dir" apply "$patch"
    log "$(basename "$dir"): applied $(basename "$patch") ($(git -C "$dir" diff --shortstat))"
  else
    log "$(basename "$dir"): no $(basename "$patch"), left as upstream"
  fi
}

mkdir -p "$TARGET"
TARGET="$(cd "$TARGET" && pwd)"
checkout_pinned "$UNITEX_URL" "$TARGET/UniTEX" "$UNITEX_COMMIT" "$HERE/patches/UniTEX.patch"
checkout_pinned "$UNITEX_FLUX_URL" "$TARGET/UniTEX-FLUX" "$UNITEX_FLUX_COMMIT" "$HERE/patches/UniTEX-FLUX.patch"

# ────────────────────────────────────────────────────────────────────────────
# environment notes
# ────────────────────────────────────────────────────────────────────────────

cat <<EOF

Done. Repos in $TARGET. Environment (one conda env serves inference and training):

  conda create -n unitex python=3.10 -y && conda activate unitex
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
  pip install diffusers==0.32.2 peft==0.15.2 transformers==4.52.4 accelerate deepspeed prodigyopt \\
      tensorboard lmdb omegaconf pandas matplotlib trimesh sentencepiece safetensors pillow scipy
  # bitsandbytes==0.45 only for training WITHOUT --zero_text_embeds (NF4 T5). The full UniTEX
  # inference stack (kaolin, nvdiffrast, xformers, ...) is in $TARGET/UniTEX/env.sh

  diffusers must stay 0.32.2: 0.33 breaks UniTEX LoRA loading, and the patched trainer was tested
  against 0.32.2 + peft 0.15.2.

  export CFI_ROOT=$REPO              # launch.py --cfi_root defaults to this (unitex.* imports)
  export PYTHONPATH=\$CFI_ROOT:\$PYTHONPATH

Weights:
  FLUX.1-dev (gated on the Hub): huggingface-cli download black-forest-labs/FLUX.1-dev \\
      --local-dir pretrain_models/black-forest-labs/FLUX.1-dev
  Released UniTEX texture LoRA (warm start): huggingface-cli download lyxun/UniTEX \\
      mv_lora_weights.safetensors --local-dir pretrain_models/unitex
  then copy it to <output_dir>/checkpoint-0/pytorch_lora_weights.safetensors and pass
  --resume_from_checkpoint latest (see $HERE/TRAINING.md).

Check the data before a run:
  python -m unitex.flux_dataset check --root <mvgen root> [--view-res 1024]
EOF
