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
A texture pipeline that consumes the product photo and untextured mesh from CFI-3DGen and emits a textured glb, built on UniTEX (FLUX.1-dev) with a focus on readable label text.

Updated 2026-11-22: generation is about 2x faster with identical generated views (per product on one 24 GB GPU).
- 1024 px views: 21.2 to 10.8 min
- 512 px views: 8.7 to 2.8 min

### How to texture products (sample commands, on the GPU server):
```bash
export UNITEX_ROOT=/path/to/work_folder # the folder that holds env.sh
export LORA=/path/to/pytorch_lora_weights.safetensors # the texture LoRA
# full pipeline at 1024 px: about 11 min per product
bash unitex/runs/fast_products.sh <gpu> <eval dir> <sku list file | a,b,c> <run name>
# the models go to <eval dir>/<sku>/<run name>_pf/textured_mesh.glb, a check of every product to <eval dir>/check_<run name>_pf.md
# optional, fewer steps: 7.6 min per product, small losses on some backs
bash unitex/runs/fast_products.sh <gpu> <eval dir> <sku list file | a,b,c> <run name> --texture-steps 20 --delight-steps 8
# optional, 512 px views: about 3 min per product, larger made-up words on backs and sides
VIEW_RES=512 bash unitex/runs/fast_products.sh <gpu> <eval dir> <sku list file | a,b,c> <run name>
```

File formats, design and results: `unitex/DESIGN.md`. Training the texture LoRA: `unitex/TRAINING.md`.
