# Training UniTEX-FLUX with GlyphAnchor glyph tokens (step 2 to 4)

This is the runbook for training our own UniTEX texture LoRA with GlyphAnchor glyph tokens on a
GPU server. File formats are in `unitex/DESIGN.md`. Everything here was run on a CPU laptop with
a tiny random FLUX (see "What was tested"). Nothing has run on a GPU or with real FLUX.1-dev
weights yet.

## Pieces

| file | role |
|---|---|
| `mvgen.py` | Blender exporter: `render/`, `render_random/`, `caption/`, `training_uid.json` |
| `text_regions.py` | writes `render/<uid>/text.json` from UV-space OCR of the GLB texture (separate component) |
| `pack_lmdb.py` | PNG dirs to LMDB envs (what UniTEX-FLUX reads in production) |
| `unitex/flux_dataset.py` | drop-in training dataset plus `check` CLI |
| `unitex/glyph.py`, `unitex/glyph_ids.py` | glyph patches and anchored fp32 position ids (512 px geometry) |
| `unitex/glyph_tokens.py` | glyph instances at any per-view resolution, VAE encode, packing, text token mask, GlyphConfig parsing |
| `unitex/patches/UniTEX-FLUX.patch` | trainer / launcher changes against UniTEX-FLUX `036a736` |
| `unitex/setup_unitex.sh` | clones UniTEX `affa1e2` and UniTEX-FLUX `036a736`, applies the patches |

## 1. Set up the repos and the environment

```bash
git clone <cfi-synthgen> /work/cfi-synthgen && cd /work/cfi-synthgen
unitex/setup_unitex.sh /work/unitex_repos        # prints the pip lines and weight downloads
export CFI_ROOT=/work/cfi-synthgen
```

Pins: torch 2.4.1 (cu118), `diffusers==0.32.2`, `peft==0.15.2`, transformers 4.52.4, plus
accelerate, deepspeed, prodigyopt, tensorboard, lmdb, omegaconf, pandas, matplotlib, trimesh,
sentencepiece. bitsandbytes is only needed when training without `--zero_text_embeds`.
The tests ran with torch 2.2.2 CPU, diffusers 0.32.2, peft 0.15.2, transformers 4.46.3,
accelerate 1.15.0.

Weights: FLUX.1-dev into `pretrain_models/black-forest-labs/FLUX.1-dev` (subfolders `scheduler`,
`vae`, `transformer`, plus `tokenizer*` / `text_encoder*` if you train with real captions), and
the released texture LoRA `lyxun/UniTEX` `mv_lora_weights.safetensors` for the warm start.

## 2. Generate the training data

```bash
BPY=/work/cfi-synthgen/.venv/bin/python        # python with bpy 4.2
DATA=/data/cfi_mv_512

# renders: 6 fixed ortho views + 20 reference candidates per SKU (the loader needs >= 20)
$BPY mvgen.py --mode train --out $DATA --glbs /data/approved_bundle --res 512 \
    --samples 64 --random-views 20 --device optix --shard 0/4     # one shard per GPU, then merge
$BPY mvgen.py --mode index --out $DATA                             # rebuild training_uid.json

# text regions: render/<uid>/text.json (DESIGN.md). A uid without text.json trains without glyphs.
python text_regions.py --root $DATA --bundle-v2 /data/approved_bundle_v2 --backend paddle --verify-view-ocr

# LMDB (in place, keeps metadata.json and text.json as plain files)
python pack_lmdb.py --root $DATA

# check every uid before spending GPU hours (UniTEX-FLUX itself silently swaps broken samples)
python -m unitex.flux_dataset check --root $DATA --json $DATA/check.json
```

For the 1024 px experiment render at that size, including the references:
`mvgen.py ... --res 1024 --ref-res 1024`, then `check --root $DATA_1024 --view-res 1024`.
The check warns when renders are smaller than `--view-res` (they would be upsampled).

`check` loads every uid and verifies: every map decodes (an RGB `_rgb.png` without alpha is an
error, not a skip), shapes and ranges at `--view-res`, camera rotations equal `common.RAW_C2W`,
nocs mask vs rgb alpha IoU per view, CCM extent near 0.95, unit world normals, all 20 reference
candidates decode, reference weights are not all zero, the prompt exists, and `text.json` parses
and builds glyphs. Exit code 1 on any failure, `--strict` also fails on warnings, `--export DIR`
writes strip PNGs for a visual check.

## 3. Launch training

From the patched UniTEX-FLUX checkout. 8 GPUs (DeepSpeed ZeRO-2 config shipped with the repo, its
`gradient_accumulation_steps: 8` must equal the CLI value):

```bash
cd /work/unitex_repos/UniTEX-FLUX
OUT=output/cfi_glyph_512
mkdir -p $OUT/checkpoint-0
cp pretrain_models/unitex/mv_lora_weights.safetensors $OUT/checkpoint-0/pytorch_lora_weights.safetensors

accelerate launch --config_file configs/acc_8gpus.yaml launch.py \
    --pretrained_model_name_or_path pretrain_models/black-forest-labs/FLUX.1-dev \
    --dataset_impl cfi --cfi_root $CFI_ROOT --dataset_name_list /data/cfi_mv_512 \
    --view_resolution 512 --resolution 512 3072 --n_rows 1 --n_cols 6 \
    --use_complex_dataset --six_views_or_four_views \
    --dual_image --control_image --both_ccm_normal_condition \
    --mixed_precision bf16 --lora_rank 16 --lora_alpha 16 \
    --optimizer prodigy --learning_rate 1.0 --guidance_scale 1.0 \
    --train_batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing \
    --random_drop_noise --random_drop_noise_probability 0.75 \
    --random_drop_condition --random_drop_condition_probability 0.25 \
    --max_train_steps 8000 --checkpointing_steps 500 --validation_steps 10000000 \
    --dataloader_num_workers 4 \
    --zero_text_embeds \
    --glyph --glyph_config $CFI_ROOT/unitex/glyph_train.json \
    --text_keep_boost 0.3 \
    --resume_from_checkpoint latest --output_dir $OUT \
    --report_to tensorboard --tasks texturing --seed 666
```

4 GPUs: `--config_file configs/acc_4gpus.yaml`. That file pins `gpu_ids: 4,5,6,7` and port 6681,
edit it to the GPUs you have. Keep `--gradient_accumulation_steps 8` (it must match the yaml),
which halves the effective batch to 32.

`glyph_train.json` does not exist in the repo, write it next to the run. Example (these are the
GlyphConfig defaults written out, any field of `unitex.glyph.GlyphConfig` can go in):

```json
{
  "anchor_mode": "center",
  "stages": [[2000, 0.6, 0.2, 0.2], [6000, 0.3, 0.3, 0.4], ["inf", 0.05, 0.25, 0.7]],
  "token_budget": 1536,
  "p_all_drop": 0.1,
  "p_item_drop": 0.15
}
```

Single fields can be overridden on the command line: `--glyph_set anchor_mode=warp
fixed_font_px=12 fixed_line_px=16 fixed_margin_px=0`. The trainer writes the resolved config to
`$OUT/glyph_config.json`.

### New flags (all in `launch.py`, defaults reproduce the stock trainer)

| flag | default | effect |
|---|---|---|
| `--dataset_impl {unitex,cfi}` | `unitex` | `cfi` builds `unitex.flux_dataset` over `--dataset_name` / `--dataset_name_list` |
| `--cfi_root` | `$CFI_ROOT` | cfi-synthgen checkout, prepended to `sys.path` |
| `--view_resolution` | 512 | pixels per view. Strip R x 6R, reference R x R, ids follow the latent size |
| `--cfi_image_ext {auto,.png,.mdb}` | auto | LMDB when `data.mdb` exists, else PNG |
| `--cfi_alpha_source {rgb,mask}` | rgb | `alphas` from the `_rgb` alpha or the `_nocs` hard mask |
| `--cfi_skip_broken` | off | log a broken sample and use the next uid (default: raise with the uid) |
| `--glyph` | off | append glyph tokens (needs `--dataset_impl cfi`, `--train_batch_size 1`) |
| `--glyph_config` | none | GlyphConfig JSON file or inline JSON object |
| `--glyph_set k=v ...` | none | GlyphConfig overrides, values parsed as JSON |
| `--zero_text_embeds` | off | zero T5 `[B, 512, 4096]` and CLIP `[B, 768]` like UniTEX inference, text encoders not loaded |
| `--train_x_embedder` | off | put `x_embedder` back in `modules_to_save` and make it trainable |
| `--text_keep_boost b` | 0 | noise drop keeps text-box target tokens with `p + b(1 - p)` (p = 0.25 at the default drop) |
| `--glyph_layout` | off | every text line drawn at its place and size in each view (`unitex/glyph_layout.py`), VAE-encoded, on 64 extra input channels of the target tokens through a zero-initialised projection next to `x_embedder` |
| `--glyph_layout_config` / `--glyph_layout_set k=v ...` | none | LayoutConfig (which lines, drops, share of edge-map lines) |
| `--glyph_layout_probe_every N` | 0 | every N steps one more forward without the layout, logs the drawn small-text loss with and without it |
| `--glyph_layout_lr lr` | 0 | a separate AdamW with this learning rate for the layout projection (0 = it stays in the main optimizer) |
| `--glyph_layout_mode {channels,tokens}` | channels | channels: the layout on extra input channels through the projection. tokens: the layout latent under the drawn lines appended as tokens on the glyph plane at those target tokens' (row, col), no projection |
| `--glyph_layout_max_tokens N` | 4096 | tokens mode: at most N layout tokens, the lines with the most tokens left out |
| `--ocr_loss_weight w` | 0 | adds w times the OCR loss (`unitex/ocr_loss.py`): at low noise a TrOCR model reads the small lines of the predicted clean image |
| `--ocr_loss_config` / `--ocr_loss_set k=v ...` | none | OcrConfig (recognizer, noise limit, line heights and count, label check) |
| `--ocr_loss_log_every N` | 0 | every N steps the predicted lines are read too and logged next to the clean readings in `ocr_rank<r>.jsonl` |
| `--ocr_loss_grad_every N` | 0 | diagnostic, one process: every N steps the gradient norms of the flow loss and of the unweighted OCR term go to `ocr_rank<r>.jsonl` |

Other changes in the patch: position ids are built in fp32 everywhere (trainer call sites and
`_prepare_latent_image_ids` in both task pipelines), the loss slice is `model_pred[:, :n_noise]`,
`--lora_layers` now adds an adapter (the stock code never did), `cond_image_type` is mapped to
`rgb_postfix` (it was a `TypeError`), the final `args.validation_config` access no longer crashes,
`get_sigmas` gets the device (it defaulted to `cuda:0`), and `data/datasets.py` imports `zipfile`
(it raised `NameError` on import without `CAUGHT_ALL_EXCEPTIONS=1`).

## 4. Recommendations

- **Glyph layout instead of glyph tokens.** Appended glyph tokens reach the target only through
  attention, placed by their position ids, and the trained LoRAs ignore them (same output with and
  without glyphs). In training they were jittered by up to 2 tokens at 1024 and fixed renders are
  2 to 5 times the height of small text, so a patch almost never lies on its letters at their size.
  `--glyph_layout` gives target token (r, c) the latent of the layout pixels it covers. Train it with
  `GLYPH_ARGS=""` (no glyph tokens), the reference text blur and `--text_loss_weight`, which then
  only covers the lines drawn in that sample's layout. The projection is saved as
  `glyph_layout_proj.safetensors` next to every `pytorch_lora_weights.safetensors`, and
  `run_unitex.py --glyph-layout` loads it. `layout_probe_rank<r>.jsonl` in the output dir has the probe.
  Give the projection its own learning rate with `--glyph_layout_lr`. On a warm start Prodigy moves
  each weight about 1e-5 per step, and after 250 steps the projection was 1% of `x_embedder`'s size
  with under 1% less small-text loss in the probe.
- **Layout tokens over layout channels.** With its own AdamW (lr 3e-4) the projection grows, but the
  probe's small-text loss with the layout is 2 to 5% higher at noise levels up to 0.44 for the first
  100 steps: the frozen base reads the projected layout as part of the noisy latent. Use
  `--glyph_layout_mode tokens`: the same layout latent goes in as tokens on the glyph plane at the
  text's own target tokens, and the model reads them through attention. Inference:
  `run_unitex.py --glyph-layout --glyph-layout-mode tokens` (no projection file, no GlyphAnchor tokens).
- **AdamW for fine-tunes.** Add `--optimizer AdamW --learning_rate 1e-4 --lr_scheduler
  constant_with_warmup --lr_warmup_steps 20`. Prodigy restarts its step size estimate on a warm start,
  and a fine-tune of a trained LoRA then barely moves it (0.04% of the effective weights in 100 steps).
- **Crops for the layout.** `--glyph_layout_set p_edges=0.0 p_crop=0.7 crop_shift=1.0 crop_blur=0.75
  crop_color=0.15` draws 70% of the training lines as degraded crops of the target view and the rest
  as font. Inference: `--glyph-layout-kind crop` with a registered text json (`front_image`), so
  front-view lines get crops of the photo and other views fall back to font.
  `--glyph_layout_oracle` (diagnostic) feeds the clean target latents instead of the drawn layout,
  which shows how fast aligned layout tokens can be learned at all.
- **Edge maps the same way.** `--glyph_layout_set p_edges=0.7 p_crop=0.0 crop_shift=1.0 crop_blur=0.75
  crop_color=0.15` draws 70% of the training lines as Canny edges of the target view, found on the pixels
  after the crop degradations, and the rest as font. Inference: `--glyph-layout-kind edges`. In the eval
  the edge maps added nothing, so crops stay the layout to use (DESIGN.md, step 5). Adding `edge_degrade=shift`
  finds the edges after the shift only. On a training sample the blur and contrast changes left 67 of 267
  small lines without edges. After the shift only, 22 had none, the same as on clean pixels.
- **Edges and font together.** `--glyph_layout_set p_edges=0.0 p_crop=0.0 p_both=0.7 crop_shift=1.0
  crop_blur=0.75 crop_color=0.15` gives 70% of the training lines both inputs: the font on the glyph plane
  and the edges on a second plane at the same target tokens. The rest are font. Tokens mode only.
  Inference: `--glyph-layout-kind both`. In the eval it read as many lines as crops with the sharp photo
  and about half as many with the photo's text blurred, so crops stay the layout to use (DESIGN.md, step 5).
- **OCR loss.** `--ocr_loss_weight w` adds a recognizer's loss on the text itself. At steps with noise
  level up to `max_sigma` (0.4) it takes up to 6 lines drawn in the layout, 12 to 64 px tall at 1024 and
  at most 24 times as long as tall. Every target token of their boxes is kept in the noise drop (the flow
  loss still sees only the tokens the drop kept), the predicted clean latent of each box goes through the
  VAE decoder, and TrOCR's cross entropy of its reading of the clean target box is the loss. Lines TrOCR
  cannot read in the clean target are left out. `ocr_rank<r>.jsonl` holds every step's lines, readings and
  losses. The weight 0.007 comes from a pilot with `--ocr_loss_grad_every` (DESIGN.md, step 7). In the
  eval it did not raise the lines read, on the test products or on all 28 (DESIGN.md, step 7).

- **Warm start from the released LoRA** through `checkpoint-0` as above. It needs `--lora_rank 16
  --lora_alpha 16` and the default target modules (no `--lora_layers`), which match the 684 LoRA
  tensors of `mv_lora_weights.safetensors`. Its two `x_embedder` tensors are dropped with a
  warning unless `--train_x_embedder` is set (see below). Resume restores LoRA weights and the
  step count only: Prodigy's step size estimate starts over.
- **`--zero_text_embeds`**. UniTEX inference feeds zero text embeddings, so training on real
  captions only adds a train / test mismatch, an NF4 T5 on every GPU and a bitsandbytes
  dependency. The glyph tokens carry the text.
- **Leave `x_embedder` untrained.** With the pinned peft 0.15.2, the stock `modules_to_save`
  entry never made `x_embedder` trainable (measured: `requires_grad` stays False and the saved copy
  equals the base weight), which explains why the released LoRAs hold an `x_embedder` identical
  to base FLUX. diffusers 0.32.2 drops it at load anyway. Without the flag, warm-start files that
  carry `x_embedder` keys would otherwise be copied over the frozen base weight by peft (measured),
  so the patch drops them.
- **Glyph stages** follow GlyphAnchor's staged SFT: gt crops of the training target first, then
  box-size renders, then mostly fixed-size renders (the inference kind). The stage boundaries are
  optimizer steps (`global_step`), and with 8 GPUs x accumulation 8 one step is 64 samples. With
  a warm start from `checkpoint-0` the schedule starts at stage 1. A stage applies while
  `global_step < until_step`, so with the default stages the last stage (70% fixed renders, the
  kind `build_glyphs_infer` uses) only starts at step 6000: `--max_train_steps` must be well
  above 6000 (the command above uses 8000), or shorten the stages. Glyphs are rebuilt every
  sample from `text.json`, so gt crops always come from the target the trainer picked (lit rgb or
  albedo, 50/50).
- **Generated text is off by default** (`GlyphConfig.use_generated=False`). With text_regions.py
  output, most items on our GLBs are `provenance: generated` (Tuna Helper: 106 of 114, Red Bull: 3
  of 11), so the default trains glyphs on the front-photo text only (7 and 14 instances on those two
  SKUs). The generated text is still rendered in the targets, so `--glyph_set use_generated=true`
  gives many more glyph examples. Which one trains better is an experiment.
- **Id jitter scales with the resolution.** Instances are built at 512 px geometry and rebuilt at R,
  so `jitter_tokens=1` means +-1 token at 512 and +-2 tokens at 1024 (the same +-16 px at 512 of the
  object). Set `--glyph_set jitter_tokens=0` for an unjittered 1024 run.
- **`--text_keep_boost`** counters the 75% target token drop, which often leaves small text with
  no loss in a step. 0.3 keeps text tokens with probability 0.475 instead of 0.25. The value is a
  guess, not tuned.
- **1024 px per view** (`--view_resolution 1024 --resolution 1024 6144`, data rendered at 1024) is
  the experiment for the VAE ceiling (step 1 measured about 0.54 of front-text words recoverable
  through the FLUX VAE at 512 px per view and 0.86 at 1024). With the default drops the image
  sequence grows from about 6.9k tokens plus up to 1536 glyph tokens at 512 to about 27.6k plus up
  to 6144 at 1024 (512 text tokens on top of both), because the glyph patches and the budget scale
  by 4. Expect several times the memory and step time (not measured). Lower `--glyph_set token_budget=768` or raise
  `--random_drop_condition_probability` if it does not fit. The released LoRA was trained at 512,
  and UniTEX inference hardcodes 512 (`infer_mv`, `export_condition`), so a 1024 LoRA needs the
  patched inference.
- **Data loader**: stock `train.sh` uses `--dataloader_num_workers 0`. Use 4 or more. The glyph
  build itself runs in the main process (it needs the step and the target): 0.2 to 0.36 s per
  sample at 512 and 0.46 to 0.92 s at 1024 for an 88-item SKU on the laptop CPU.

Logs: tensorboard gets `loss`, `lr` and `glyph_tokens` (glyph tokens of the last micro-batch).

## 5. What was tested (CPU, tiny random FLUX)

`python -m pytest unitex/tests/test_flux_train.py` (29 tests, 265 s on a 4-core laptop). The tiny
FLUX has 1 double and 1 single block, 2 heads of 16, rope axes (4, 6, 6). The tiny VAE has 16
latent channels and FLUX's scaling and shift.

- The dataset returns bit-identical tensors to UniTEX-FLUX's own `MVDataset` /
  `ReconstructDataset` / `PBRTextureGenerationDataset` (launch.py config) on synthetic data (PNG
  and LMDB) and on two real mvgen SKUs, including the reference view `random.choices` picks.
- Glyph off: one patched trainer step reproduces the stock trainer's loss exactly, fp32 (1.86876)
  and bf16 (1.84661), with the same trainable parameter list.
- Glyph on (real Tuna Helper renders, 88 OCR items, gt stage): the step runs, the sequence grows
  by exactly Ng (498), glyph ids are fp32 with axis0 = 1, the LoRA weights change, and overwriting
  the glyph token outputs with 1e4 leaves the loss unchanged (they are outside the loss).
- 1024 px per view: 1024 x 6144 strip, 53248 image ids with target cols up to 383 and reference
  cols up to 447, all unique in fp32 (bf16 would merge 10112 of them), glyph ids inside their
  64-token slots. At 1024 a gt glyph token holds the same pixels as the target token at its id in
  center, stretch and warp modes.
- `accelerate launch launch.py` (single CPU process, fp32, gradient checkpointing, prodigy, LMDB
  data, `--glyph --zero_text_embeds --text_keep_boost 0.5`) ran 2 steps and wrote checkpoints and
  the final LoRA (no `x_embedder` keys). A second run warm-started from `checkpoint-0` with
  accumulation 2.
- `check` passes on real renders (2 SKUs at 512, 1 SKU rendered at 1024 with a res-1024 text.json,
  no upsampling warning) and names each broken uid (missing map, RGB without alpha, fewer than 20
  reference cameras).
- The glyph step, keep boost, real-data equivalence and launch tests also pass with text.json
  files written by `text_regions.py` (vision backend, `--bundle-v2`): +117 glyph tokens in the
  step test, 128 and 115 in the two launch runs.

## 6. Untested

- Anything on GPU: DeepSpeed ZeRO-2, multi-process data sharding, bf16 autocast on CUDA, memory
  and speed with real FLUX.1-dev (12B), the 1024 px memory estimate above.
- The real released LoRA as `checkpoint-0` (tested with a LoRA of the same key format from the tiny
  model).
- Training with real captions (text encoders, NF4 T5), whose code the patch leaves unchanged.
- The delight LoRA trainer (`tasks/delight_mv`) only gets the fp32 id change in its pipeline.
- The stale validation hook (`--validation_prompt`), unchanged apart from the 512 constant.
- `text_regions.py` with the paddle backend and `--verify-view-ocr` (the tests used its vision
  backend output, synthetic text.json files and a quick per-view OCR one).
- Whether the glyph tokens improve text at all. That is what the training run is for.
