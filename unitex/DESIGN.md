# UniTEX text fidelity: design and file formats

This directory holds the step 1 evaluation of UniTEX on our products and the step 2 GlyphAnchor
integration. The Blender exporter for UniTEX-FLUX training data is `mvgen.py` at the repo root.
Everything shares the conventions in `unitex/common.py`.

## Conventions (see `unitex/common.py`)

- glTF / UniTEX frame: Y-up, front faces +Z. Blender frame: Z-up, `(x, y, z)_b = (x, -z, y)_g`.
- Normalization: bbox centre to the origin, longest half-extent 0.95.
- Six orthographic cameras, `ortho_scale = 2.0`, distance 2.8, 512 px per view.
  Raw order `front right back left top bottom` (`RAW_C2W`, Blender cam2world).
- FLUX strip (512 x 3072): slot i shows raw `FULL_INDEX[i]`, i.e. `front left right back top bottom`.
- UniTEX inference grids (`mv_ccm.png`, `mv_normal.png`, `mv_alpha.png`, `mv_rgb.png`): 2x3,
  tiles `f r t / b l d`, bottom tile rolled 180 degrees relative to raw view 5 (`GRID_TILE_TO_RAW`).
- Pixel coordinates: top-left origin, x right, y down, continuous (pixel centres at +0.5).
- Boxes are `[x0, y0, x1, y1]` in pixels unless a name says `_norm`.
- Quads are 4 points `[[x, y], ...]` ordered top-left, top-right, bottom-right, bottom-left
  in the text's reading frame (so a 90-degree rotated word still starts at its first letter).

## Training data layout written by `mvgen.py` (UniTEX-FLUX `MVDataset` compatible)

```
<out>/
  training_uid.json                      ["cfi/<sku>", ...]
  render/cfi/<sku>/
    0000_rgb.png ... 0005_rgb.png        RGBA, lit render, straight alpha
    0000_albedo.png ...                  RGB base colour (sRGB), composited on white
    0000_nocs.png ...                    RGBA, RGB = (p_blender + 1) / 2 (normalized frame), A = hard mask
    0000_normal.png ...                  RGB, (n_cam + 1) / 2, Blender camera frame (x right, y up, z to viewer)
    0000_bump.png, _metallic, _roughness 1x1 placeholders (required by the UniTEX-FLUX assert)
    0000_uv.npz ...                      float16 uv (H, W, 2) + bool mask, from the 1-sample pass (for text projection)
    metadata.json                        {"cam2world_matrixs": [6 x 4x4], "res", "ortho_scale", "normalize": {...}, "sku", "shape", "title", "source_glb", "yaw_deg"}
    text.json                            per-view text items (below), written by text_regions.py
  render_random/cfi/<sku>/
    0000_rgb.png ... 00NN_rgb.png        RGBA lit reference candidates, framed like UniTEX preprocess (tight bbox, 0.95, grey)
    0000_albedo.png ...                  RGB albedo of the same views
    metadata.json                        {"cam2world_matrixs": [...], "fov_deg": [...], "framing": [...]}
  caption/cfi/<sku>/prompt.txt           "[MVFLUX]" (UniTEX inference prompt)
```

`pack_lmdb.py` converts each `render/...` and `render_random/...` directory into an LMDB env
(`data.mdb`, keys are file basenames without extension, values are the PNG bytes), which is
what UniTEX-FLUX `launch.py` reads (`image_ext='.mdb'`). `metadata.json` and `text.json` stay
as plain files next to `data.mdb`.

## `text.json` (per uid)

```json
{
  "version": 1,
  "res": 512,
  "source": "uv-ocr" | "photo-lift" | "manual",
  "items": [
    {
      "id": 0,
      "text": "HELPER",
      "conf": 0.98,
      "provenance": "front" | "generated" | "photo" | "unknown",
      "flags": ["template_leak", "low_conf", "rotated"],
      "angle_deg": 0.0,
      "src_quad": [[x, y], [x, y], [x, y], [x, y]],
      "views": {
        "0": {
          "bbox": [x0, y0, x1, y1],
          "quad": [[x, y], [x, y], [x, y], [x, y]],
          "pixels": 812,
          "coverage": 0.97,
          "cos": 0.91,
          "height_px": 21.4,
          "grid": {"ns": 16, "nt": 4, "xy": [[[x, y] or null, ...], ...]}
        }
      }
    }
  ]
}
```

- `src_quad` is in texture pixels for `uv-ocr` and in photo pixels for `photo-lift`.
- `views` is keyed by raw view index (string). Only views where the item is visible appear.
- `coverage` is the fraction of the item's reading-direction extent visible in that view
  (16 bins along the baseline). `cos` is the mean |n . v| over its visible pixels.
- `grid` samples the item's own reading frame `(s, t) in [0,1]^2` (`ns` along the baseline,
  `nt` across) and stores where each cell centre lands in the view, `null` if not visible.
  It lets glyph tokens follow curved and foreshortened surfaces instead of an axis-aligned box.

## Evaluation (step 1)

Ground truth is the real product photo (`front_ref.png`), never our Gemini-painted GLB.
Only the front view has ground truth. Stages scored against the photo's OCR:

| stage | image | meaning |
|---|---|---|
| `ref512` | UniTEX `processed_image.png` | input ceiling: what FLUX can see |
| `vae512` | `ref512` after a FLUX VAE round trip | representational ceiling at 512 px |
| `lit` | `mv_rgb_w_light.png` slot 0 | texture LoRA output |
| `delit` | `mv_rgb.png` front tile | after the delight LoRA |
| `baked` | `textured_mesh.glb` rendered from the front at 1024 | final asset |
| `baseline` | our current `textured.glb` rendered from the front | current Gemini pipeline |

Metrics follow GlyphAnchor's InfoTextBench: NED (1 - normalized Levenshtein over the
reading-order concatenation), word precision / recall / F1 (multiset, case-folded, alphanumeric),
phrase hit rate (fraction of ground-truth lines found with line-level NED >= 0.8). Everything is
also bucketed by ground-truth text height measured in the 512 px front view, so large and small
text are reported separately.

## GlyphAnchor inside UniTEX-FLUX (step 2)

- Glyph patches: black text on white, VAE-encoded, packed into FLUX tokens, appended after the
  control and reference tokens as clean conditions (no loss, never dropped by UniTEX's random
  condition drop).
- Position ids: axis 0 = 1 (a "glyph plane" separate from the target), axes 1-2 = target strip
  token coordinates. Anchoring modes: `center` (paper: patch centred on the box, native 1-token
  spacing), `stretch` (patch tokens spread over the box), `warp` (patch tokens placed through
  the item's `grid`, following the surface). All ids are built in fp32.
- One patch copy per view in which the item is visible enough (coverage and cos thresholds),
  under a total glyph-token budget.
- Staged glyph source: ground-truth crops from the target view, box-size renders, fixed-size
  renders, with the sampling ratio moving towards fixed-size renders over training.
- Inference layout: OCR the photo, map boxes into the front view (affine fit of the photo's
  foreground bbox onto the mesh's front silhouette), lift through the front CCM onto the
  surface, re-project into all six views with a visibility test (`unitex/anchors.py`).
