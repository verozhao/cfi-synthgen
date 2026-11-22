# cfi-synthgen
A synthetic dataset generation pipeline that consumes glb files from CFI-3DGen and emits a COCO detection dataset to train detector on.

## How to generate a dataset (sample commands):
```bash
uv sync
uv run python synthgen.py \
--glbs ./approved_bundle \
--out ./dataset \
--scenes 2 \
--products-per-scene 5 \
--cameras-per-scene 3 \
--resolution 2048 \
--hdri hdri/studio.exr \
--backgrounds ./textures \
# --cameras-json overrides --cameras-per-scene 
--cameras-json cameras_4cam.json  \ 
# placement options: 
# 1. scatter (products spread loose, fills camera frame), 
# 2. cluster_mid (products at moderate density, gaps between objects), 
# 3. cluster_tight (products tightly piled, products touching),
# 4. stacking (box vertical stack + side products)
--placement stacking 
--seed 0 # optional

# To visualize labels
uv run python visualize.py --dataset ./dataset
```

## UniTEX texture pipeline (`unitex/`)

`unitex/` textures a product mesh from one product photo with [UniTEX](https://github.com/YixunLiang/UniTEX) (FLUX.1-dev with LoRAs). The work there is about keeping the printed text on the product readable. The texturing needs a GPU server. The dataset generator above does not need `unitex/`.

| file | what it does |
|---|---|
| `setup_unitex.sh` | clones UniTEX and UniTEX-FLUX at pinned commits and applies `patches/` |
| `prepare_eval.py`, `eval_text.py`, `report.py` | build the evaluation folder, score the printed text against the photo, write an HTML review sheet |
| `run_unitex.py` | runs UniTEX over the evaluation folder, with the glyph and layout options |
| `run_unitex_lowmem.py` | the same on a 24 GB GPU |
| `run_unitex_fast.py` | the same on a 24 GB GPU, about twice as fast with the same views |
| `photo_front.py` | puts the real product photo into the generated front view and bakes again |
| `back_panel.py` | typesets the nutrition facts and ingredients from the product's Open Food Facts record on the back and bakes again |
| `anchors.py`, `glyph*.py` | the text layout from the photo's OCR and the glyph tokens given to FLUX |
| `runs/` | the scripts that ran on the GPU server |
| `tests/` | unit tests: `python -m pytest unitex/tests -q` |

### Faster generation

Medians per product on one 24 GB RTX 4090, with 28 denoising steps per pass:

| | before | now |
|---|---|---|
| views at 1024 px, delight pass at 512 px | 21.2 min | 10.8 min |
| views at 512 px | 8.7 min | 2.8 min |
| views at 1024 px, then photo front | 22.7 min | 11.2 min |

- Before is `run_unitex_lowmem.py`. Now is `run_unitex_fast.py`. It sends FLUX's weights to the GPU a block at a time from page-locked memory and copies the next block while the current one computes.
- The generated views (`mv_rgb.png`) are pixel-identical between the two on every product compared.
- `photo_front.py --one-bake --fps-height 9` shortens the photo front step from about 90 s to 26 s.
- `--texture-steps 20 --delight-steps 8` brings the last row to 7.6 min. On the 28 eval products no front got worse and 4 backs or sides lost a detail.

`unitex/runs/fast_products.sh` runs both GPU steps. Each product's evaluation folder needs the photo and mesh from `prepare_eval.py`, and the registered text layout (`text_reg.json` and its front image) from `anchors.py --register-run --front-image`.

```bash
export UNITEX_ROOT=/path/to/work_folder                  # the server folder that holds env.sh
export LORA=/path/to/pytorch_lora_weights.safetensors    # the texture LoRA
# REPO and UX set the paths of this repo and the patched UniTEX checkout when they differ from the script's defaults
bash unitex/runs/fast_products.sh <gpu> <eval dir> <sku list file | a,b,c> <run name>
# with fewer steps
bash unitex/runs/fast_products.sh <gpu> <eval dir> skus.txt fast20 --texture-steps 20 --delight-steps 8
```

### Where to read more

- `unitex/DESIGN.md`: file formats, what was tried at each step and the results. The speed work is its last section, "Inference speed (step 9)".
- `unitex/TRAINING.md`: how to train the texture LoRA.
- `unitex/STAGE2.md`: the integration status and the first runs on the GPU server.

The models and images generated with `unitex/` are not stored in this repo.
