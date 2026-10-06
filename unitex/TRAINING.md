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

Other changes in the patch: position ids are built in fp32 everywhere (trainer call sites and
`_prepare_latent_image_ids` in both task pipelines), the loss slice is `model_pred[:, :n_noise]`,
`--lora_layers` now adds an adapter (the stock code never did), `cond_image_type` is mapped to
`rgb_postfix` (it was a `TypeError`), the final `args.validation_config` access no longer crashes,
`get_sigmas` gets the device (it defaulted to `cuda:0`), and `data/datasets.py` imports `zipfile`
(it raised `NameError` on import without `CAUGHT_ALL_EXCEPTIONS=1`).

## 4. Recommendations

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
