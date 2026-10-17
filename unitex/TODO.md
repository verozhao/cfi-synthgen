# TODO

Steps in order. Finish each step before starting the next.

## Now: step 1, evaluate stock UniTEX

- [ ] Run stock UniTEX (generated_mesh.glb + front_ref.png) on the eval SKUs on the GPU server.
- [ ] Review the text manually (HTML report), score it per stage, share the results.

## Next: step 2, only if step 1 shows the text is bad

- [ ] Integrate GlyphAnchor-style glyph conditioning into UniTEX-FLUX LoRA training.
- [ ] Generate training data with mvgen.py (target about 10k models).

## Later (after the steps above are done)

- [ ] Test EasyText (arXiv 2505.24417, FLUX LoRA, public code) as the glyph plug-in instead of
      reimplementing GlyphAnchor. Compare against the GlyphAnchor integration on the same eval.
      Verified 2026-09-24: the released EasyText code cannot edit an existing image. It always
      starts from noise, a custom `latents` init crashes (tuple unpack), and output is fixed at
      1024x1024 by hardcoded token counts. Reuse its position-mapping mechanism inside the UniTEX
      trainer, or modify its pipeline (SDEdit init or masked blending) before a per-view test.
