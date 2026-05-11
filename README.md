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