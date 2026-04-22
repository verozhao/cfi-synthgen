# cfi-synthgen
A synthetic dataset generation pipeline that consumes glb files from CFI-3DGen and emits a COCO detection dataset to train detector on.

## How to generate a dataset (sample commands):
```bash
uv sync
uv run python synthgen.py --glbs ./glbs --out ./dataset --scenes 100 --products-per-scene 3 --cameras-per-scene 6 --resolution 640 --seed 0
uv run python synthgen.py --glbs ./glbs --out ./dataset --scenes 1 --products-per-scene 2 --cameras-per-scene 1 --resolution 256 
```