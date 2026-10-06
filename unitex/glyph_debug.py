"""
Glyph token footprint viewer: text.json + 512 x 3072 strip -> debug PNG

Draws the strip with slot names and a faint 16 px token grid, outlines every kept glyph token
(the 16 x 16 cell at its (row, col) id, fractional ids drawn where they fall) in a per-instance
colour with an "#item v<raw> kind" label, dashes the item's view quad, and tiles the glyph patches
below the strip at 1:1.

Usage:
  python unitex/glyph_debug.py --text-json render/cfi/<sku>/text.json --strip strip.png \
      --out glyph_debug.png --mode warp --kind fixed
  python unitex/glyph_debug.py ... --train --step 500 --seed 3       (training draw)
"""

import argparse
import colorsys
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitex.common import STRIP_NAMES, TOKEN_PX, VIEW_RES, raw_to_slot, split_strip
from unitex import glyph as gl


def _colour(k):
    r, g, b = colorsys.hsv_to_rgb((k * 0.618034) % 1.0, 0.9, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _label(draw, xy, text, fill, font):
    l, t, r, b = draw.textbbox(xy, text, font=font)
    draw.rectangle((l - 2, t - 1, r + 2, b + 1), fill=(0, 0, 0, 190))
    draw.text(xy, text, fill=fill, font=font)


def _dashed(draw, pts, fill, dash=4):
    for a, b in zip(pts, pts[1:] + pts[:1]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.linalg.norm(b - a) / dash))
        for k in range(0, n, 2):
            p, q = a + (b - a) * k / n, a + (b - a) * min(k + 1, n) / n
            draw.line([tuple(p), tuple(q)], fill=fill, width=1)


# ────────────────────────────────────────────────────────────────────────────
# Drawing
# ────────────────────────────────────────────────────────────────────────────

def draw_debug(strip, instances, res=VIEW_RES, token_grid=True, title=None):
    """strip: (res, 6*res, 3|4) uint8 array or PIL image, or None for a grey canvas."""
    W = 6 * res
    if strip is None:
        base = Image.new("RGB", (W, res), (90, 90, 90))
    else:
        base = gl._on_white(strip) if not isinstance(strip, Image.Image) else strip.convert("RGB")
    base = base.convert("RGBA")
    font = gl.load_font(12)
    small = gl.load_font(10)

    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    if token_grid:
        for x in range(0, W, TOKEN_PX):
            d.line([(x, 0), (x, res)], fill=(255, 255, 255, 28))
        for y in range(0, res, TOKEN_PX):
            d.line([(0, y), (W, y)], fill=(255, 255, 255, 28))
    for s in range(6):
        d.line([(s * res, 0), (s * res, res)], fill=(255, 255, 0, 255), width=2)
        _label(d, (s * res + 4, res - 16), f"slot {s} {STRIP_NAMES[s]}", (255, 255, 0, 255), small)

    for k, inst in enumerate(instances):
        col = _colour(k)
        off = raw_to_slot(inst.raw_view) * res
        _dashed(d, [(x + off, y) for x, y in inst.quad], col + (255,))
        ids = inst.ids[inst.keep]
        ids = ids[np.isfinite(ids).all(axis=1)]
        for _, r, c in ids:
            x0, y0 = c * TOKEN_PX, r * TOKEN_PX
            d.rectangle((x0 + 1, y0 + 1, x0 + TOKEN_PX - 1, y0 + TOKEN_PX - 1),
                        fill=col + (45,), outline=col + (230,))
        if len(ids):
            lx, ly = ids[:, 2].min() * TOKEN_PX, max(ids[:, 1].min() * TOKEN_PX - 15, 0)
            _label(d, (lx, ly), f"#{inst.item_id} v{inst.raw_view} {inst.kind}", col + (255,), font)
    top = Image.alpha_composite(base, over).convert("RGB")

    # patches tiled below, wrapped to the strip width
    pad, cap = 8, 16
    rows, x, y, row_h = [], pad, pad, 0
    for k, inst in enumerate(instances):
        pw, ph = inst.patch.size
        if x + pw + pad > W and x > pad:
            x, y, row_h = pad, y + row_h + cap + pad, 0
        rows.append((k, x, y))
        x += max(pw, 150) + pad
        row_h = max(row_h, ph)
    tiles_h = (y + row_h + cap + pad) if instances else 0
    head = 22 if title else 0
    out = Image.new("RGB", (W, head + res + tiles_h), (30, 30, 30))
    out.paste(top, (0, head))
    d = ImageDraw.Draw(out)
    if title:
        d.text((6, 4), title, fill=(255, 255, 255), font=font)
    for k, x, y in rows:
        inst = instances[k]
        col = _colour(k)
        yy = head + res + y
        out.paste(inst.patch, (x, yy + cap))
        d.rectangle((x - 1, yy + cap - 1, x + inst.patch.width, yy + cap + inst.patch.height), outline=col)
        gh, gw = inst.token_hw
        d.text((x, yy), f"#{inst.item_id} v{inst.raw_view} {inst.kind} {gh}x{gw} keep {inst.n_keep}",
               fill=col, font=small)
    return out


def crop_slots(img, slots, res=VIEW_RES, head=0):
    """Keep only the listed strip slots (patch tiles are dropped)."""
    parts = [img.crop((s * res, 0, (s + 1) * res, head + res)) for s in slots]
    out = Image.new("RGB", (res * len(parts), head + res))
    for k, p in enumerate(parts):
        out.paste(p, (k * res, 0))
    return out


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-json", required=True)
    ap.add_argument("--strip", default=None, help="512 x 3072 strip png in FULL_INDEX order (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="center", choices=("center", "stretch", "warp"))
    ap.add_argument("--kind", default="fixed", choices=("fixed", "box", "gt"),
                    help="inference kind (ignored with --train)")
    ap.add_argument("--train", action="store_true", help="use build_glyphs with the stage schedule")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--use-generated", action="store_true")
    ap.add_argument("--font", default=None)
    ap.add_argument("--slots", default=None, help="comma list of strip slots to keep, e.g. 0,2")
    args = ap.parse_args(argv)

    cfg = gl.GlyphConfig(anchor_mode=args.mode, infer_kind=args.kind, font_path=args.font,
                         use_generated=args.use_generated)
    if args.budget is not None:
        cfg.token_budget = args.budget
    strip = np.asarray(Image.open(args.strip).convert("RGB")) if args.strip else None
    views = split_strip(strip) if strip is not None else None
    if args.train:
        cfg.p_all_drop = 0.0
        insts = gl.build_glyphs(args.text_json, views, args.step, args.seed, cfg)
        title = f"train step {args.step} seed {args.seed} mode {args.mode}"
    else:
        insts = gl.build_glyphs_infer(args.text_json, cfg, views=views)
        title = f"infer kind {args.kind} mode {args.mode}"
    title += f"  instances {len(insts)}  tokens {gl.total_tokens(insts)} / {cfg.token_budget}"
    img = draw_debug(strip, insts, title=title)
    if args.slots:
        img = crop_slots(img, [int(s) for s in args.slots.split(",")], head=22)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    img.save(args.out)
    for i in insts:
        print(f"  [#{i.item_id} v{i.raw_view}] {i.kind:5s} {i.token_hw[0]}x{i.token_hw[1]} "
              f"keep {i.n_keep:4d}  {i.text!r}")
    print(f"  [glyph_debug] {len(insts)} instances, {gl.total_tokens(insts)} tokens -> {args.out}")


if __name__ == "__main__":
    main()
