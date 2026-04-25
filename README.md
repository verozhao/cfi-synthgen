# cfi-synthgen
A synthetic dataset generation pipeline that consumes glb files from CFI-3DGen and emits a COCO detection dataset to train detector on.

## How to generate a dataset (sample commands):
```bash
uv sync
uv run python synthgen.py \
--glbs ./approved_bundle \
--out ./dataset_hdri \
--scenes 2 \
--products-per-scene 5 \
--cameras-per-scene 3 \
--resolution 512 \
--hdri hdri/studio.exr \
--backgrounds ./textures \
--cameras-json cameras_lenovo_4cam.json

# To visualize labels:
uv run python visualize.py --dataset ./dataset_hdri
```
Note: ```--cameras-json``` overrides ```--cameras-per-scene```