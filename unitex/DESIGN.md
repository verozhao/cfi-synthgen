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
    0000_albedo.png ...                  RGB base colour (sRGB), straight colour, white only where alpha is 0
    0000_nocs.png ...                    RGBA, RGB = (p_blender + 1) / 2 (normalized frame), A = hard mask
    0000_normal.png ...                  RGB, (n_cam + 1) / 2, Blender camera frame (x right, y up, z to viewer)
    0000_bump.png, _metallic, _roughness 1x1 placeholders (required by the UniTEX-FLUX assert)
    0000_uv.npz ...                      float16 uv (H, W, 2) + bool mask, from the 1-sample pass (for text projection)
    metadata.json                        {"cam2world_matrixs": [6 x 4x4], "res", "ortho_scale", "normalize": {...}, "sku", "shape", "title", "source_glb", "yaw_deg"}
    text.json                            per-view text items (below), written by text_regions.py
  render_random/cfi/<sku>/
    0000_rgb.png ... 0019_rgb.png        RGBA lit reference candidates, framed like UniTEX preprocess
                                         (tight alpha bbox, longer side 0.95 of the canvas), transparent background.
                                         Keep at least 20: the UniTEX-FLUX loader indexes 0..19.
    0000_albedo.png ...                  RGB albedo of the same views (straight colour)
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
  Layout is `xy[it][is]`: `nt` rows across the text (t = 0 at the top of the letters), each
  holding `ns` cells along the baseline. It lets glyph tokens follow curved and foreshortened
  surfaces instead of an axis-aligned box.

## Evaluation (step 1)

Ground truth is the real product photo (`front_ref.png`), never our Gemini-painted GLB.
Only the front view has ground truth. Stages scored against the photo's OCR:

| stage | image | meaning |
|---|---|---|
| `ref512` | UniTEX `processed_image.png` | input ceiling: what FLUX can see |
| `vae512` | `ref512` after a FLUX VAE round trip | representational ceiling at 512 px |
| `lit` | `mv_rgb_w_light.png` slot 0 | texture LoRA output |
| `delit` | `mv_rgb.png` front tile | after the delight LoRA (with `--delight-shading`, its shading on `lit`) |
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
- Fixed-size renders default to the paper's scale (24 px font, 32 px lines), which makes every
  line 2 to 3 token rows. Most of our text is 15 to 32 px tall at 512 px per view, so center-mode
  footprints of small text overlap. `fixed_font_px=12, fixed_line_px=16, fixed_margin_px=0`
  gives one token row per line. Which works better is an experiment, not a settled choice.
- Inference layout: OCR the photo, map boxes into the front view (affine fit of the photo's
  foreground bbox onto the mesh's front silhouette), lift through the front CCM onto the
  surface, re-project into all six views with a visibility test (`unitex/anchors.py`).

## Glyph layout inside UniTEX-FLUX (step 5)

- The text of every line is drawn dark on white into its view quad, stretched to fill it, in every
  view where the line is visible enough: a layout strip at the target's size (`unitex/glyph_layout.py`).
  Kind `font` draws the OCR text. Kind `edges` draws Canny edges of the view's own pixels inside the
  quad (training: the target view, inference: the photo registered into the front view). Kind `crop`
  pastes those pixels themselves, in colour. Training degrades a crop like a registered photo: its
  content shifted (`crop_shift`), blurred (`crop_blur`), contrast and brightness scaled (`crop_color`).
  When these are set, training edges are found on the degraded pixels too. With `edge_degrade` set to
  `shift` they are found after the shift only. Blur and contrast changes erase the edges of low-contrast
  small text, which a sharp registered photo keeps at inference.
- The strip is VAE-encoded (posterior mode), packed like the target, and the packed latent of a white
  strip is subtracted. Token (r, c) of the target gets those 64 numbers as extra input channels, the
  condition and glyph tokens get zeros. `x_embedder` is wrapped: the first 64 channels go through the
  original layer, the extra 64 through a bias-free projection that starts at zero, so a warm-started
  LoRA gives exactly its old output until the projection learns.
- No position ids, no attention routing and no jitter: the layout sits on the target pixels it
  describes, at their size. This is how FLUX.1 Canny / Depth take their control image.
- Training lines: every text.json line passing the filters (generated text included by default),
  15% of lines and 10% of whole layouts dropped so the model keeps painting from the reference. Each
  line's kind is one draw: edges with `p_edges` (0.3 by default), crop with `p_crop`, else font. With
  `--text_loss_weight` only drawn lines are weighted.
- The projection trains in its own AdamW (`--glyph_layout_lr`). Prodigy picks one step size for all
  its weights from the LoRA, about 1e-5 per weight per step on a warm start, so a projection that
  starts at zero stays too small to matter within a fine-tune.
- Tokens mode (`--glyph_layout_mode tokens`) is the second way in. The layout strip is VAE-encoded
  without subtracting the blank strip, and the tokens under the drawn lines (at most 4096, the
  largest lines left out first) are appended last with ids (1, row, col): the glyph plane, exactly
  on the target tokens they describe. They are never dropped and are cut off before the loss. This
  is the GlyphAnchor glyph token without jitter and at the text's size, the way EasyText and
  OminiControl place spatially aligned conditions. The channels mode adds the projected layout to
  the target's own input embedding, which the frozen base reads as noisy latent. With a working
  learning rate that made small-text loss worse at low noise. At inference the layout tokens reach
  the texture pass as `glyph_latents` + `glyph_ids` (GLYPH_API 6).
- Fine-tunes use AdamW at 1e-4. On a warm start Prodigy restarts its step size estimate at 1e-6. In
  100 steps it moved the LoRA's effective weights by 0.04%, while AdamW moved them by 2.7%.
- Tokens mode works when the layout looks like the text. A diagnostic (`--glyph_layout_oracle`) puts
  the clean target latent of each layout token's own target token into it. Within 100 steps the
  small-text loss with those tokens fell to about a quarter of the loss without them. Font and edges
  ink gave no gain after 200 steps: it has to be turned into the label's own lettering and colours,
  which a short fine-tune does not learn. A crop is the closest a real photo gets to the oracle.
- Crop eval (`layouttokC_1024_ft500`, crops for 70% of training lines, 500 AdamW steps). Every line of
  the 6 test products is scored at its photo-registered spot (`eval_text --align homography`), 56
  lines in all.
  - With the photo's text blurred (`--ref-text-blur`) only the crops show the text. The final view
    then reads 28 of the 56 lines. Every earlier run with blurred text read at most 13 and no line
    under 16 px, with glyph tokens or without. So the glyph tokens never carried text, and the crops do.
  - With the sharp photo the crop run reads 34 lines in the final view and the same LoRA without a
    layout reads 30. Before the delight pass the crop run reads 39, the most of any run. The delight
    pass loses the difference (step 6).
- Edges eval (`layouttokE_1024_ft500`, edges for 70% of training lines, everything else as the crop
  run). These evals use `--delight-shading 4`, so the crop run's numbers here are those of step 6. The
  edge maps come from the same registered photo pixels as the crops, and their letters are clean.
  - With the sharp photo the edges run reads 37 lines in the final view and the crop run 39. The edges
    LoRA without a layout also reads 37, the very same lines. So the edge maps add nothing.
  - With the photo's text blurred the edges run reads 15 lines and the crop run 33. The crop run reads
    18 lines the edges run misses, and the edges run reads none that the crop run misses (p 8e-6). So
    the model takes no text from the edge maps.
  - The training probe agrees. It compares the small-text loss of a sample with and without its
    layout. Over the second half of training the layout lowered that loss by 3.8% with edges and by
    10.4% with crops.
- Longer edges run (`layouttokE2_1024_ft1500`). It finds the training edges after the shift only
  (`edge_degrade=shift`) and trains for 1500 steps instead of 500. Everything else is as in the edges
  eval. Final view, 56 lines:
  - With the sharp photo it reads 37 lines and the same LoRA without a layout 38. The crop run reads 39.
    So with the sharp photo the edge maps still add nothing.
  - With the photo's text blurred it reads 29 lines and the same LoRA without a layout 13. It reads 17
    lines that the run without a layout misses and misses 1 that it reads (p 1.5e-4). So after 1500
    steps the model does take text from the edge maps.
  - The crop run still reads more with the blurred photo, 33 lines after only 500 steps. Against the
    edges run that difference is not significant (p 0.34).
  - Lines read exactly, with the blurred photo: 18 with edges and 10 without a layout. The crop run
    reads 24 exactly.
  - The fix alone changes little. At step 500 the run reads 34 lines with the sharp photo and 16 with
    the blurred one. The 500-step edges run read 37 and 15.
  - The gain comes from the longer training. From step 500 to 1500 the blurred-photo run gains 13 lines
    and loses none (p 2.4e-4). Only 4 of the 13 are read exactly. The training probe agrees: the
    layout's gain on the small-text loss kept growing and reached 9.4% in the last third of training.
- Kind `both` gives a line the font and the edges at once, each as its own input. The font goes on the
  layout strip and the edges on a second strip below it. In tokens mode the line's target tokens get
  the font tokens on the glyph plane and the edge tokens on an edge plane (ids (2, row, col)). The
  target and the conditions keep axis 0 at 0, so the edge plane is free. Edge tokens count toward the
  token cap like the font tokens.
- Both eval (`layouttokB_1024_ft500`, kind both for 70% of training lines, everything else as the crop
  run). These evals use `--delight-shading 4`. As with crops, front-view lines get the font and the
  photo's edges, and lines in other views get the font alone. The edges are found on the sharp photo,
  so the layouts with and without `--ref-text-blur` are the same.
  - With the sharp photo the both run reads 39 lines in the final view, as many as the crop run. Each
    run reads one line the other misses. The same LoRA without a layout reads 36.
  - With the photo's text blurred the both run reads 17 lines and the crop run 33. The crop run reads
    16 lines the both run misses, and the both run reads none that the crop run misses (p 3e-5). The
    same LoRA given the font alone reads 15, as does the edges run. So font and edges together carry
    no more text than either one alone.
  - In the training probe the layout lowered the small-text loss by 6.6% over the second half. That is
    between edges (3.8%) and crops (10.4%). The model leans on the two inputs a little in training, but
    at inference they bring back almost none of the blurred text.

## Delight shading (step 6)

- At 1024 px per view the delight pass runs at half size (`--delight-res 512`), and `add_detail` puts
  back the texture pass's detail above that size. Below it the delight output is used as it is, and
  the delight LoRA redraws small letters there. In the crop run Gluten became Bluten, charms became
  Suleua and CONDENSED became (BOBURO).
- `--delight-shading S` takes only the shading from the delight pass: `mv_rgb = lit * G(delit) / G(lit)`.
  G is a Gaussian of S px per view at 512, applied inside each view tile. The ratio is clipped to
  [0.2, 5]. Letters and fine print come from the texture pass, and the raw delight output is also
  saved (`mv_rgb_delit_<res>.png`). The option needs GLYPH_API 7.
- The merge depends only on the two saved strips, so it was tested on finished runs. `add_detail` on
  `mv_rgb_w_light.png` and `mv_rgb_delit_512.png` rebuilds the saved `mv_rgb.png` exactly. Additive
  and multiplicative merges at three widths were tried on the crop run. The multiplicative one at
  S = 4 was kept and run on six more runs, with the pipeline's own `shading_detail`.
- Colours and broad shading stay those of the delight output. The difference to it is on letter and
  line edges.
- Lines read out of 56. The last column counts the 28 lines under 16 px:

| run | lit | delit now | delit with `--delight-shading 4` | under 16 px, now → with |
|---|---|---|---|---|
| crop layout, sharp photo | 39 | 34 | 39 | 10 → 14 |
| font layout, sharp photo | 37 | 34 | 39 | 9 → 13 |
| same LoRA, no layout | 34 | 30 | 34 | 8 → 9 |
| crop layout, photo text blurred | 32 | 28 | 33 | 5 → 8 |
| `rb_new_fixed_sharp` (reference text blur LoRA) | 37 | 35 | 36 | 9 → 11 |
| `tl_fixed_sharp` (text loss LoRA) | 37 | 33 | 37 | 9 → 11 |
| `g1024_final` (glyph tokens LoRA at 1024) | 36 | 31 | 35 | 8 → 11 |

- Over the seven runs the merge turns 31 missed lines into hits. It turns 3 hits into misses. Each
  run ends near its own lit score.
- With the merge the crop layout reads 39 lines and the same LoRA without a layout reads 34. The crop
  and font layouts tie overall, and both lead every run without a layout on small text.
- OCR noise: the stylised Lucky Charms logo reads charms or SulEUD in two images that differ by at
  most one grey level. A difference of one line between runs is noise.
- Lighting features finer than S, such as small glints, are kept from the lit view.

## Full eval (28 products)

- All 28 eval products, none of them a training product. They hold 277 ground-truth lines. Of these, 169
  are under 16 px. The view stages are scored at each line's photo-registered spot as in step 5. The
  baked asset and the baseline are front renders, scored through the refined box fit.
- The crop run uses `layouttokC_1024_ft500`, crops of the registered photo and `--delight-shading 4`.
  Every product got a layout, and all but one got photo crops. On the 6 test products the run reads
  exactly the lines the earlier crop run read, before and after the delight pass.
- Lines read out of 277. In brackets, those under 16 px:

| run | lit | final view | baked asset |
|---|---|---|---|
| `g1024_final` (glyph tokens LoRA at 1024) | 122 (35) | 108 (23) | 104 (22) |
| `g1024_final`, delight fix applied offline | 122 (35) | 126 (38) | |
| crop layout with `--delight-shading 4` | 154 (57) | 152 (58) | 146 (57) |
| the same with the OCR loss LoRA (step 7) | 151 (54) | 150 (53) | 140 (49) |
| current Gemini pipeline (`baseline`) | | | 157 (86) |

- In the final view the crop run reads 45 lines that `g1024_final` misses and misses 1 that it reads.
  The sign test gives p 1.3e-12. Most of them, 35, are under 16 px.
- The delight fix alone does not explain it. Against `g1024_final` with the same fix applied offline,
  the crop run still gains 28 lines and loses 2. The sign test gives p 9e-7.
- With the fix the delight pass keeps the letters. The crop run's final view gains 3 lines over its
  lit view and loses 5. The raw delight output at 512 reads only 114.
- Baking costs a few lines. The crop run's baked asset gains 8 lines over its final view and loses 14.
  That difference is not significant (p 0.29). `g1024_final` gains 7 and loses 11.
- Against the current Gemini pipeline the gap is small text. The baked crop run reads 146 lines and the
  baseline 157. Ours reads 45 lines that the baseline misses, and the baseline reads 56 that ours
  misses. That difference is not significant (p 0.32). On small lines the baseline reads 86 and ours 57.
  On the larger lines ours reads 89 and the baseline 71.
- So the remaining gap is small text, which the OCR loss (step 7) targets.

## OCR loss (step 7)

- `unitex/ocr_loss.py`, trainer flag `--ocr_loss_weight`. A pretrained text recognizer (TrOCR) reads the
  text lines of the predicted clean image during training. The loss is its cross entropy of the right
  reading, so the LoRA gets a signal on the letters themselves and not only on their pixels.
- Lines: at steps whose noise level is at most `max_sigma` (0.4), up to `max_lines` (6) lines drawn in the
  sample's layout, 12 to 64 px tall at 1024 and at most 24 times as long as tall. With the reference text
  blurred, a line that is not drawn cannot be known, so only drawn lines are read.
- Every target token of a line's box (its quad's bounding box plus 2 tokens) is kept in the noise drop, so
  the step predicts the whole line. Those extra tokens stay out of the flow matching loss.
- The predicted clean latent `x0 = x_t - sigma * v` of each box goes through the frozen VAE decoder with
  gradients (checkpointed per block). The line's quad, padded by 15% of its height, is resampled into a
  straight line image and resized for the recognizer.
- Label: TrOCR's greedy reading of the same box of the clean target latent, decoded the same way, as its
  own token ids. A line whose reading does not match its text.json text is left out (NED below 0.8, case
  folded and with spaces removed, since texture OCR often runs words together). The recognizer cannot read
  such a line even from the right latent. Taking the label after the VAE asks for nothing the VAE cannot
  show, and taking TrOCR's own tokens leaves no tokenizer or letter case convention to disagree with.
- Loss: the mean over the kept lines of the label's per-token cross entropy given the predicted line
  image (teacher forcing).
- Reader: `microsoft/trocr-small-printed`. Both printed TrOCR models read about a quarter of 5568
  VAE-decoded training lines. The base model reads 19 more of them at 9 times the time per line. Both
  rate a render with one wrong letter worse than the right one on every test line they read. Blur raises
  the loss on only some lines, so sharpness stays the flow loss's job. The stage 1 checkpoints were left
  out because their weights need torch 2.6 to load.
- Pilot: 60 steps on a single GPU at weight 1, with the gradient norms of both terms at every OCR step
  (`--ocr_loss_grad_every 1`). 114 of 161 lines passed the label check. TrOCR's loss on the predicted
  lines grows with the noise level. Its median rises from 0.17 at the lowest noise to 2.8 at the
  highest, and above sigma 0.4 none of the logged predicted lines read right. The OCR gradient grows the
  same way. It is tens of times the flow loss gradient around sigma 0.1 and hundreds of times above 0.4.
  A recognizer's gradient on a line it cannot read says little about the letters and would outweigh the
  flow loss, so `max_sigma` is 0.4. The run weight 0.007 puts the median OCR gradient below that limit
  at half the flow gradient. Memory peaked at 20 GB per GPU.
- Run (`ocrC_1024_ft500`): the crop run plus the OCR loss at weight 0.007, everything else the same.
  Each GPU used the loss at about a third of its steps. Of the lines it drew, 1834 of 2826 passed the
  label check. It added under 1% to the training time.
- Lines read on the 6 test products, out of 56, with `--delight-shading 4`. The last column counts the
  28 lines under 16 px in the final view:

| LoRA and layout | lit | final view | under 16 px |
|---|---|---|---|
| crop run, crop layout, sharp photo | 39 | 39 | 14 |
| OCR run, crop layout, sharp photo | 40 | 40 | 14 |
| crop run, crop layout, photo text blurred | 32 | 33 | 8 |
| OCR run, crop layout, photo text blurred | 31 | 31 | 7 |
| crop run, no layout | 34 | 34 | 9 |
| OCR run, no layout | 35 | 35 | 11 |

- With the sharp photo the OCR run reads one more line in the final view and loses none. That line is
  over 32 px, and both runs read the same small lines. With the photo's text blurred the OCR run reads
  2 fewer. Neither difference is significant.
- The OCR loss did change the LoRA. Its effective weights end 81% as far from the crop LoRA as the crop
  LoRA moved from its start. What the eval reads stayed the same.
- On all 28 products, scored as in the full eval above, the OCR run reads no more either. Its final view
  reads 150 lines and the crop run's 152. It reads 6 lines the crop run misses and misses 8 that it
  reads. The sign test gives p 0.79. On small lines it reads 53 and the crop run 58. The baked asset
  reads 140 lines against 146. The sign test gives p 0.18. Both runs used the same layouts, so only the
  LoRA differs.
- So at weight 0.007 the OCR loss does not close the small-text gap to the current pipeline.
