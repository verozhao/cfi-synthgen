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
--resolution 512 \
--hdri hdri/studio.exr \
--backgrounds ./textures \
# --cameras-json overrides --cameras-per-scene 
--cameras-json cameras_4cam.json  \ 
# options: scatter (products scattered), close_far (3 closeby 1 far away), stacking (boxes stacking together, only effective if the scene contains more than 2 boxes)
--placement stacking 
--seed 0 # optional

# To visualize labels
uv run python visualize.py --dataset ./dataset_hdri
```