# TODO

Order set by the advisor (meeting 2026-09-23). Finish each step before starting the next.

## Now: step 1, evaluate stock UniTEX

- [ ] Run stock UniTEX (generated_mesh.glb + front_ref.png) on the eval SKUs on the GPU server.
- [ ] Review the text manually (HTML report), score it per stage, send results to the advisor.

## Next: step 2, only if step 1 shows the text is bad

- [ ] Integrate GlyphAnchor-style glyph conditioning into UniTEX-FLUX LoRA training.
- [ ] Generate training data with mvgen.py (target about 10k models).

## Later (after the advisor's plan is done)

- [ ] Test EasyText (arXiv 2505.24417, FLUX LoRA, public code) as the glyph plug-in instead of
      reimplementing GlyphAnchor. Compare against the GlyphAnchor integration on the same eval.
