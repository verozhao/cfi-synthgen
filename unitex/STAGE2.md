# Stage 2: GlyphAnchor inside UniTEX, integration status

Step 2 of the advisor's plan: put GlyphAnchor glyph tokens into UniTEX-FLUX LoRA training and into
UniTEX inference. Every piece is built, and the whole chain ran end to end on a CPU laptop with a
tiny random FLUX (2026-10-01). Nothing has run on a GPU or with real FLUX.1-dev weights yet.
File formats are in `DESIGN.md`. The training runbook with every flag is in `TRAINING.md`.

## What is built

| file | role |
|---|---|
| `mvgen.py` | Blender exporter. `--mode train` writes training renders, `--mode views --with-geometry` writes nocs / normal / alpha of any GLB |
| `text_regions.py` | training layout: UV-space OCR of the GLB texture to `render/<uid>/text.json` (`source: uv-ocr`) |
| `pack_lmdb.py` | PNG dirs to LMDB envs, which UniTEX-FLUX reads |
| `unitex/flux_dataset.py` | training dataset (bit-identical to UniTEX-FLUX's loaders) plus the `check` CLI |
| `unitex/glyph.py`, `unitex/glyph_ids.py` | glyph patches and fp32 position ids anchored to text boxes on axis 0 = 1 |
| `unitex/glyph_tokens.py` | glyphs at any per-view resolution (multiples of 512), VAE encode, packing, config parsing |
| `unitex/anchors.py` | inference layout: photo OCR lifted onto the mesh and projected into the six views (`source: photo-lift`) |
| `unitex/patches/UniTEX-FLUX.patch` | trainer and launcher changes against UniTEX-FLUX `036a736` |
| `unitex/patches/UniTEX.patch` | inference changes against UniTEX `affa1e2` (glyph tokens, `view_res`, fp32 ids) |
| `unitex/setup_unitex.sh` | clones both repos at the pinned commits and applies the patches |
| `unitex/run_unitex.py`, `unitex/run_eval.sh` | eval runner with glyph flags and `--texture-lora` |
| `unitex/glyph_debug.py` | draws glyph token footprints on a strip |

## Run it on the GPU server

### 1. Repos and environment

```bash
git clone <cfi-synthgen> /work/cfi-synthgen && cd /work/cfi-synthgen
unitex/setup_unitex.sh /work/unitex_repos      # prints the conda / pip lines and weight downloads
export CFI_ROOT=/work/cfi-synthgen PYTHONPATH=/work/cfi-synthgen:$PYTHONPATH
```

### 2. Training data

The eval SKUs and the rotated GLBs must stay out of training. An explicit `--exclude` replaces
mvgen's default list, so pass both lists in one file:

```bash
BPY=/work/cfi-synthgen/.venv/bin/python
DATA=/data/cfi_mv_512
cat unitex/bad_orientation_skus.txt unitex/eval_skus.txt > /data/train_exclude.txt
$BPY mvgen.py --mode train --out $DATA --glbs /data/approved_bundle --res 512 --samples 64 \
    --random-views 20 --device optix --exclude /data/train_exclude.txt --shard 0/4   # one shard per GPU
$BPY mvgen.py --mode index --out $DATA
python text_regions.py --root $DATA --bundle-v2 /data/approved_bundle_v2 --backend paddle --verify-view-ocr
python pack_lmdb.py --root $DATA
python -m unitex.flux_dataset check --root $DATA --json $DATA/check.json      # exit 1 on any broken uid
```

The data audit also lists product families (Red Bull x7, Jif x5 and others). Excluding the eval SKU
alone leaves its siblings in training, so decide on a family split before step 3.

### 3. Training

Write `glyph_train.json` (example in `TRAINING.md`), then from the patched UniTEX-FLUX:

```bash
cd /work/unitex_repos/UniTEX-FLUX
OUT=output/cfi_glyph_512 && mkdir -p $OUT/checkpoint-0
cp pretrain_models/unitex/mv_lora_weights.safetensors $OUT/checkpoint-0/pytorch_lora_weights.safetensors
accelerate launch --config_file configs/acc_8gpus.yaml launch.py \
    --pretrained_model_name_or_path pretrain_models/black-forest-labs/FLUX.1-dev \
    --dataset_impl cfi --cfi_root $CFI_ROOT --dataset_name_list /data/cfi_mv_512 \
    --view_resolution 512 --resolution 512 3072 --n_rows 1 --n_cols 6 \
    --use_complex_dataset --six_views_or_four_views --dual_image --control_image --both_ccm_normal_condition \
    --mixed_precision bf16 --lora_rank 16 --lora_alpha 16 --optimizer prodigy --learning_rate 1.0 \
    --guidance_scale 1.0 --train_batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing \
    --random_drop_noise --random_drop_noise_probability 0.75 \
    --random_drop_condition --random_drop_condition_probability 0.25 \
    --max_train_steps 8000 --checkpointing_steps 500 --validation_steps 10000000 --dataloader_num_workers 4 \
    --zero_text_embeds --glyph --glyph_config $CFI_ROOT/unitex/glyph_train.json --text_keep_boost 0.3 \
    --resume_from_checkpoint latest --output_dir $OUT --report_to tensorboard --tasks texturing --seed 666
```

The 1024 px per view variant and the reasoning behind each flag are in `TRAINING.md`.

### 4. Inference and evaluation with our checkpoint

The trained file is a full texture LoRA (warm-started from `mv_lora_weights`), so it must replace
the released texture adapter, not stack on it. Use `--texture-lora` (or `TEXTURE_LORA=` in
run_eval.sh), not `--add-lora-path`:

```bash
cd /work/cfi-synthgen
# stock baseline first (its cache grids give the anchors geometry)
EVAL_DIR=/data/unitex_eval UNITEX_ROOT=/work/unitex_repos/UniTEX UNITEX_PY=python \
    RUN=unitex_s63 unitex/run_eval.sh
# photo-lift text.json, then the glyph run with our LoRA
EVAL_DIR=/data/unitex_eval UNITEX_ROOT=/work/unitex_repos/UniTEX UNITEX_PY=python \
    GLYPH=1 TEXTURE_LORA=/work/unitex_repos/UniTEX-FLUX/output/cfi_glyph_512/pytorch_lora_weights.safetensors \
    RUN=glyph_center_s63 STEPS="anchors unitex vae render eval report" unitex/run_eval.sh
```

Or run_unitex directly:

```bash
python -m unitex.anchors --eval-dir /data/unitex_eval --skus all --geometry unitex --run-name unitex_s63 \
    --out-dir /data/anchors      # or one SKU at a time with --out <eval>/<sku>/text.json
cd /work/unitex_repos/UniTEX && python /work/cfi-synthgen/unitex/run_unitex.py --unitex-root . \
    --eval-dir /data/unitex_eval --run-name glyph_center_s63 --seed 63 --resume \
    --glyph-json-name text.json --glyph-mode center \
    --texture-lora /work/unitex_repos/UniTEX-FLUX/output/cfi_glyph_512/pytorch_lora_weights.safetensors
```

Run glyphs with the released LoRA too (no `--texture-lora`). That control shows whether any gain
comes from training or from the extra tokens alone.

## What was tested on CPU (2026-10-01)

Scratch outputs: `/Users/test/.claude/jobs/5ee54a5d/tmp/stage2_integ/`.

### Training path, 2 SKUs (016000263192 Tuna Helper box, 611269818994 Red Bull can)

| stage | result |
|---|---|
| `mvgen.py --mode train --res 512 --samples 8 --random-views 20 --no-denoise` | 2/2 rendered in 57.6 s |
| `text_regions.py --backend vision --bundle-v2 --verify-view-ocr` | Tuna 114 items (8 front, 106 generated), front view-OCR NED median 1.0. Red Bull 11 items (8 front, 3 generated), template leaks LEFT / RIGHT / BACK, front median 0.0 (wordmark 1.0, ENERGY DRINK 0.92, fine print 0 to 0.28). 27.8 s |
| `pack_lmdb.py` | 4/4 envs, 164 images, 18.2 MiB |
| `flux_dataset check` (LMDB) | 0 failed, 0 warnings. Glyph dry run: Tuna 7 instances / 201 tokens, Red Bull 14 / 455 |
| patches | both apply with `patch -p1` to fresh GitHub tarballs at `affa1e2` and `036a736`, no fuzz, no offsets |
| patched `launch.py`, tiny FLUX, `--dataset_impl cfi --glyph --zero_text_embeds`, prodigy, 2 steps | step 1: loss 1.5229, Ng 93 (Red Bull, 14 instances, gt + box). step 2: loss 1.5612, Ng 128 (Tuna, 5 instances, gt + fixed) |

Transformer input per step, read from a forward hook: sequence 7087 = 6994 + 93 and 7011 = 6883 + 128
(the non-glyph part shrinks with the random token drop). Ids are fp32. The last Ng ids all have axis 0 = 1.
Glyph rows / cols fall in 0..29 / 11..147 and 2..27 / 7..26, inside the target strip (rows 0..31,
cols 0..191). The conditions use rows 32..63 and the reference cols 192..223. Checkpoints 1, 2 and
the final LoRA were written (30 tensors for the tiny model, no `x_embedder` keys).

I looked at the glyph_debug PNG of both batches. Red Bull: gt crops of the wordmark, ENERGY DRINK
and the small print sit on their text in the front, left, right and top views. Tuna: HELPER, TUNA,
TETRAZZINI (fixed render), ROTISSERIE CHICKEN, 5 SERVINGS sit on the front panel.

### Inference path, same 2 SKUs from the eval set (copied, originals untouched)

| stage | result |
|---|---|
| `mvgen.py --mode views --with-geometry` of `mesh.glb` (= `generated_mesh.glb`, byte-identical) | 2.7 s and 2.8 s |
| `unitex.anchors --geometry mvgen` | Tuna silhouette IoU 0.9413 to 0.9698, 8 items, all in view 0. Red Bull 0.9794 to 0.9863, 9 items, views {0: 9, 1: 6, 3: 6} |
| patched `infer_mv` (stubbed renderer, real tiny bf16 PBRFluxPipeline, 2 steps, R = 512) with the real control strip from the mvgen geometry (glTF CCM + world normals, UniTEX grid order) and the photo | Tuna: 7 instances, Ng 201, texture pass sequence 13513 = 6144 + 6144 + 1024 + 201, delight pass 12288 (no glyphs). Red Bull: 14 instances, Ng 458 (view 0: 258, raw 3: 161, raw 1: 39), sequence 13770. Outputs 512 x 3072 strip and 1024 x 1536 grid |

In both SKUs the glyph ids the transformer saw equal the ids rebuilt from text.json, they are fp32
with axis 0 = 1, the glyph latents are the clean encoded patches, and every instance's ids sit in its
own view's strip slot (0 outside). The debug PNG on the control strip shows raw view 3 glyphs in slot 1
(left) and raw view 1 in slot 2 (right), as `FULL_INDEX` requires.

Also tested:

- `run_unitex.py --dry-run`: `--glyph-mode warp --glyph-set token_budget=3072` gives 201 / 435 tokens.
  `--view-res 1024` (center) gives 804 / 1832. glyph_tokens.json and run_log.jsonl record mode,
  budget and per-view counts.
- The LoRA from the training run loads into the UniTEX PBRFluxPipeline with `load_lora_weights` as
  an extra adapter. Weight 1 changes the transformer output (by 5.0e-6 after 2 steps), weight 0
  reproduces the base output exactly.
- `R = 256` with glyphs is rejected (`glyph_tokens` needs a multiple of 512, and run_unitex's CLI
  already says so).

### Unit tests

`python -m pytest unitex/tests -q`: **209 passed** in 450 s (first run: 203 passed, 4 failed, fixed below).
There are no tests at the repo root.

## Changes made during integration

1. **`--texture-lora` in `run_unitex.py`, `TEXTURE_LORA` in `run_eval.sh`.** UniTEX's
   `build_pipeline` loads our files only through `add_lora_path`, which adds them with the given
   weight to BOTH passes on top of the released texture LoRA. A checkpoint warm-started from
   `mv_lora_weights` already contains the released weights, so that applied them twice and also put
   the texture LoRA into the delight pass. `use_texture_lora` loads ours as adapter `cfi_texture`
   and sets texture weights `[0, 0, 1]`, delight `[0, 1, 0]`. The docstring example that used
   `--add-lora-path` now uses `--texture-lora`. Tested by `unitex/tests/test_stage2_integration.py`
   (the stubbed infer_mv sees exactly those two set_adapters calls, the dry-run log records the path,
   and the stock key set is unchanged without the flag).
2. **`unitex/tests/test_flux_train.py::_import_pristine_datasets`.** Stock `data/datasets.py:168`
   parses `PIL.ImageColor.colormap` as hex strings at import, and Pillow's `getrgb()` caches parsed
   tuples into that dict. After other test files drew with named colours, 4 tests failed with
   `TypeError: int() can't convert non-string with explicit base`. The helper now restores the
   strings. Production is not affected: `launch.py:61` imports `data.datasets` before any
   `unitex` module.

## UNTESTED

- Anything on a GPU: real FLUX.1-dev, DeepSpeed ZeRO-2, multi-GPU data sharding, bf16 on CUDA,
  memory and step time (512 and 1024 per view), nvdiffrast baking, LTM.
- The released `mv_lora_weights.safetensors` as `checkpoint-0`, and `--texture-lora` against the
  real `build_pipeline` (tested on the stubbed UniTEX classes and with diffusers adapters on the tiny
  pipeline only).
- `setup_unitex.sh` itself in this run. The sandbox here blocks git commands outside the worktree,
  so the patches were applied with `patch -p1` to GitHub tarballs of the pinned commits instead.
  The patch files are also applied with `git apply` by the test fixtures on every pytest run. The
  script's clone, pin and idempotency logic was not exercised here.
- `text_regions.py --backend paddle` (only Apple Vision ran here).
- UniTEX's real preprocessing (RMBG-2.0 framing) of the reference photo. The CPU run framed the photo
  with a simple alpha-bbox stand-in.
- Whether glyph tokens improve text at all. That is what the GPU training run is for.

## Known data issues seen during the run

- Red Bull's `textured.glb` has template words (LEFT, RIGHT, BACK) painted on the can. They appear in
  the training targets. text_regions flags them `template_leak` and glyphs skip them, but the model
  still sees them as target pixels. The data audit counted 23 SKUs with leaks. Excluding or
  repainting them before step 3 is a data decision.
- Center-mode fixed renders (24 px font) are often wider than the photo box, so glyph footprints
  overlap neighbouring lines and spill past the can's silhouette. That is the paper's default. The
  smaller fixed size (`fixed_font_px=12 fixed_line_px=16 fixed_margin_px=0`) is the documented
  alternative.

## Known limitations from review (not fixed)

- 1024 px per view: `glyph.items_of` rejects text.json with `res` 1024, and `eval_text.py` mis-scales
  front boxes at 1024. Both need fixing before a 1024 experiment.
- `anchors.py --ocr-json` assumes quads in original photo pixels. Quads from OCR of the padded
  `ref.png` would be shifted by the pad.
- Small train/inference differences in coverage counting and in quads for partly visible items.
- 23 training SKUs have template words (LEFT/RIGHT/BACK...) painted into their textures. Glyphs skip
  those items, but the pixels stay in the training targets. Exclude or repaint those SKUs.
