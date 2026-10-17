"""
Stage 4 (glyph tokens), step by step for one product, in one self-contained page built from the
real files: the photo's text lines, where they are placed on the front view, the line images, the
tokens, their positions, how they join the FLUX input, how the LoRA learned to use them, and the
geometry condition of the same pass.

Inputs: <eval>/<sku>/ref.png, ref_meta.json and gt_text.txt, a text.json layout (anchors.py), and
one UniTEX run's cache for the front view and geometry (mv_rgb.png or mv_rgb_generated.png,
mv_normal.png, mv_ccm.png). Optional: --text-reg (registered layout, anchors.py --register-run)
and --vae-json (vae_tokens.py: glyph tokens vs the photo's tokens).

Usage:
  python -m unitex.stage4_page --eval-dir E --sku 016000233164 --run g1024_final --text text.json \\
      --text-reg text_reg.json --out stage4_016000233164.html
"""

import argparse
import html
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from unitex import glyph as gl
from unitex import glyph_tokens as gt
from unitex.common import TOKEN_PX, split_grid
from unitex.walkthrough_figures import ACC, FONT, INK, LINE, MUTED, PANEL
from unitex.walkthrough_page import CSS, JS, control_strip, fig, on_grey, webp

BLACK, RED = (26, 26, 26), (140, 29, 24)
KIND_NAME = {"fixed": "fixed font (used in our runs)", "box": "drawn to its box", "gt": "crop of the photo"}


def load_photo(sku_dir):
    meta = json.load(open(sku_dir / "ref_meta.json"))
    px, py = meta["pad"]
    W, H = meta["orig_size"]
    return Image.open(sku_dir / "ref.png").convert("RGB").crop((px, py, px + W, py + H)), meta


def gt_status(sku_dir):
    p = sku_dir / "gt_text.txt"
    if p.exists():
        for line in open(p):
            if line.startswith("# status:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def front_of(run_dir, res):
    for name in ("mv_rgb_generated.png", "mv_rgb.png"):
        p = run_dir / "cache" / name
        if p.exists():
            g = np.asarray(Image.open(p).convert("RGB"))
            im = Image.fromarray(np.ascontiguousarray(split_grid(g, res=g.shape[1] // 3)[0]))
            return im.resize((res, res), Image.LANCZOS) if im.width != res else im
    raise FileNotFoundError(f"no mv_rgb.png in {run_dir}/cache")


def items_v0(doc):
    return [it for it in doc["items"] if "0" in (it.get("views") or {})]


def box_at(view, f):
    return [v * f for v in view["bbox"]]


def crop_union(im, boxes, pad):
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    return im.crop((max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad)),
                    min(im.width, int(max(xs) + pad)), min(im.height, int(max(ys) + pad))))


def draw_boxes(im, boxes, colour, width, labels=None):
    d = ImageDraw.Draw(im)
    font = gl.load_font(max(14, im.width // 40)) if labels else None
    for k, b in enumerate(boxes):
        d.rectangle(b, outline=colour, width=width)
        if labels:
            x, y = b[0], b[1] - font.size - 2
            d.text((x, max(0, y)), labels[k], fill=colour, font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    return im


def figw(src, caption, width, max_side=1600):
    """fig() at a fixed display width (fig's flex basis lets one image fill the row)."""
    return fig(src, caption, max_side).replace("<figure>", f'<figure style="flex:0 1 {width}px">', 1)


def drop_reason(item, view, cfg, res):
    """Why glyph selection skipped this line (glyph.expand_candidates filters, then the budget)."""
    if set(item.get("flags") or ()) & set(cfg.drop_flags):
        return "flagged " + ", ".join(sorted(set(item["flags"]) & set(cfg.drop_flags)))
    h = view.get("height_px")
    h512 = (h if h is not None else view["bbox"][3] - view["bbox"][1]) * 512 / res
    if h512 < cfg.min_height_px:
        return f"too small ({h512:.1f} px at 512, minimum {cfg.min_height_px:g})"
    if view.get("coverage", 1.0) < cfg.min_coverage or view.get("cos", 1.0) < cfg.min_cos:
        return "seen at too steep an angle"
    return "over the token budget"


def footprint(inst):
    ids = np.asarray(inst.ids)
    return [ids[:, 2].min() * TOKEN_PX, ids[:, 1].min() * TOKEN_PX,
            (ids[:, 2].max() + 1) * TOKEN_PX, (ids[:, 1].max() + 1) * TOKEN_PX]


def grid_overlay(patch):
    im = on_grey(patch).copy()
    d = ImageDraw.Draw(im)
    for x in range(0, im.width + 1, TOKEN_PX):
        d.line([(x, 0), (x, im.height)], fill=(200, 60, 50), width=1)
    for y in range(0, im.height + 1, TOKEN_PX):
        d.line([(0, y), (im.width, y)], fill=(200, 60, 50), width=1)
    return im


def sequence_svg(R, glyph_rows):
    """FLUX input of the texture pass: what is denoised and what only conditions it. Sizes: six
    target views and the six-view geometry strip of (R/16)^2 tokens per view, the reference photo
    of (R/16)^2 (at 512: 6144 + 6144 + 1024 + glyphs, STAGE2.md's patched infer_mv check)."""
    n_target = 6 * (R // TOKEN_PX) ** 2
    n_ref = (R // TOKEN_PX) ** 2
    W, y0, h = 1000, 40, 56

    def block(x, w, fill, stroke, title, sub):
        return (f'<rect x="{x}" y="{y0}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
                f'<text x="{x + w / 2}" y="{y0 + 24}" text-anchor="middle" font-size="14" fill="{INK}">{title}</text>'
                f'<text x="{x + w / 2}" y="{y0 + 43}" text-anchor="middle" font-size="12" fill="{MUTED}">{sub}</text>')

    parts = [block(0, 150, PANEL, LINE, "prompt", "text tokens (T5)"),
             block(160, 330, "#ffffff", INK, "six target views", f"{n_target:,} tokens, denoised"),
             block(500, 170, PANEL, LINE, "geometry", f"normal + CCM, {n_target:,}"),
             block(680, 130, PANEL, LINE, "reference", f"the photo, {n_ref:,}"),
             block(820, 180, "#ffffff", ACC, "glyph tokens", f"{sum(r[1] for r in glyph_rows):,} tokens, ours")]
    brace = (f'<path d="M500 {y0 + h + 8} v8 H1000 v-8" fill="none" stroke="{MUTED}"/>'
             f'<text x="750" y="{y0 + h + 32}" text-anchor="middle" font-size="12" fill="{MUTED}">'
             f'clean inputs: FLUX\'s output for these tokens is dropped</text>'
             f'<path d="M160 {y0 + h + 8} v8 H490 v-8" fill="none" stroke="{INK}"/>'
             f'<text x="325" y="{y0 + h + 32}" text-anchor="middle" font-size="12" fill="{INK}">'
             f'the only tokens FLUX predicts (and the only loss in training)</text>')
    head = (f'<text x="0" y="22" font-size="13" fill="{MUTED}">one sequence, every token attends to every other '
            f'token in every FLUX block (joint attention)</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 140" font-family="{FONT}" '
            f'style="width:100%;max-width:{W}px;background:#fff;border-radius:6px">{head}{"".join(parts)}{brace}</svg>')


def build(args):
    sku_dir = pathlib.Path(args.eval_dir) / args.sku
    R = args.view_res
    run_dir = sku_dir / args.run
    text_p = pathlib.Path(args.text) if pathlib.Path(args.text).is_absolute() else sku_dir / args.text
    doc = json.load(open(text_p))
    reg_p = None
    if args.text_reg:
        reg_p = pathlib.Path(args.text_reg) if pathlib.Path(args.text_reg).is_absolute() else sku_dir / args.text_reg
    reg = json.load(open(reg_p)) if reg_p and reg_p.exists() else None
    photo, meta = load_photo(sku_dir)
    front = front_of(run_dir, R)
    f = R / doc["res"]
    items = items_v0(doc)
    sec = []

    # 4.1 photo text
    ph = photo.copy()
    w = max(2, ph.width // 400)
    d = ImageDraw.Draw(ph)
    font = gl.load_font(max(14, ph.width // 40))
    for k, it in enumerate(items):
        q = [tuple(p) for p in it["src_quad"]]
        d.line(q + [q[0]], fill=BLACK, width=w)
        d.text((q[0][0], max(0, q[0][1] - font.size - 2)), str(k + 1), fill=RED, font=font, stroke_width=3,
               stroke_fill=(255, 255, 255))
    fg = meta.get("fg_bbox_photo") or [0, 0, photo.width, photo.height]
    ph = ph.crop(tuple(int(v) for v in fg))
    lines = "".join(f"<li>{html.escape(it['text'])}</li>" for it in items)
    sec.append(("4.1 Read the text from the photo",
                f"OCR finds each printed line and its box on the photo. These lines are the text the model is "
                f"told to write. Status of this product's line list: <b>{html.escape(gt_status(sku_dir))}</b> "
                f"(auto means straight from OCR, not checked by hand).",
                figw(ph, "OCR lines on the photo, numbered", 520) + f'<div class="col"><ol>{lines}</ol></div>'))

    # 4.2 placement on the front view
    al = doc["lift"]["align"]
    boxes = [box_at(it["views"]["0"], f) for it in items]
    pl = draw_boxes(front.copy(), boxes, BLACK, max(2, R // 400), [str(k + 1) for k in range(len(items))])
    figs = figw(pl, "front view, each line's box (black)", 520)
    note = ""
    if reg is not None:
        rmap = {it.get("src_id", it["id"]): it for it in items_v0(reg)}
        rb = [box_at(rmap[it.get("src_id", it["id"])]["views"]["0"], R / reg["res"]) for it in items
              if it.get("src_id", it["id"]) in rmap]
        both = draw_boxes(draw_boxes(front.copy(), boxes, BLACK, max(2, R // 400)), rb, RED, max(2, R // 400))
        figs += figw(both, "black: box fit (used so far), red: registered with the photo front homography", 520)
        note = (" The red boxes come from the photo front registration (stage 7): a homography that also corrects "
                "the tilt of the photo.")
    sec.append(("4.2 Place each line on the front view",
                f"The photo's product box is scaled and shifted onto the front view's silhouette box, then nudged "
                f"by up to 10% to overlap the silhouette best (overlap {al.get('iou_bbox')} before, "
                f"{al.get('iou')} after). Only scale and shift: a photo taken at an angle is not corrected, so "
                f"lines can land a little off.{note} Text that also shows on a side view is carried there through "
                f"the mesh depth.", figs))

    # 4.3 line images, 4.5 positions
    kinds = ["fixed", "box"] + (["gt"] if (reg or doc).get("front_image") else [])
    built = {}
    for kind in kinds:
        src = str(reg_p) if (kind == "gt" and reg is not None) else str(text_p)
        built[kind] = gt.build_infer_glyphs(src, gl.GlyphConfig(infer_kind=kind), view_res=R)
    rows, scale = [], 0.5
    by_id = {k: {i.item_id: i for i in v} for k, v in built.items()}
    reg_v0 = {i.get("src_id", i["id"]): i for i in items_v0(reg)} if reg is not None else {}
    for it in items:
        x0, y0, x1, y1 = box_at(it["views"]["0"], f)
        cells = [f"<td>{html.escape(it['text'])}</td><td>{round(y1 - y0)} x {round(x1 - x0)}</td>"]
        if reg is not None:
            ri = reg_v0.get(it.get("src_id", it["id"]))
            if ri is None:
                cells.append('<td class="stats">-</td>')
            else:
                a0, b0, a1, b1 = box_at(ri["views"]["0"], R / reg["res"])
                cells.append(f"<td>{round(b1 - b0)} x {round(a1 - a0)}</td>")
        for kind in kinds:
            inst = by_id[kind].get(it["id"])
            if inst is None:
                why = drop_reason(it, it["views"]["0"], gl.GlyphConfig(infer_kind=kind), doc["res"])
                cells.append(f'<td class="stats">not used: {html.escape(why)}</td>')
                continue
            pw, phh = inst.patch.width, inst.patch.height
            cells.append(f'<td><img src="{webp(inst.patch, 1024)}" style="width:{pw * scale:.0f}px;height:'
                         f'{phh * scale:.0f}px;cursor:default" data-caption="{html.escape(it["text"])}">'
                         f'<div class="stats">{phh} x {pw} px, {inst.token_hw[0]} x {inst.token_hw[1]} tokens</div></td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    head = "".join(f"<th>{KIND_NAME[k]}</th>" for k in kinds)
    reg_head = "<th>registered size</th>" if reg is not None else ""
    table = (f'<table class="pt"><tr><th>line</th><th>size on the front, box fit (h x w px)</th>{reg_head}{head}</tr>'
             f'{"".join(rows)}</table>')
    fx = [(i.text, i.patch.height / max(1e-6, (i.bbox[3] - i.bbox[1]))) for i in built["fixed"]]
    big = sorted(fx, key=lambda t: -t[1])[:3]
    font_name = pathlib.Path(gl.resolve_font_path() or "PIL default").stem
    sec.append(("4.3 Draw each line as a small image",
                f"Each line is drawn black on white (font: {html.escape(font_name)}). Our runs used the fixed font: "
                f"{gl.GlyphConfig.fixed_font_px} px "
                f"letters in {gl.GlyphConfig.fixed_line_px} px rows at 512 px per view, then enlarged 2x at "
                f"{R} px. So every line image has the same letter height, whatever the real text size. Small text "
                f"gets an image much larger than itself (" + ", ".join(f"{html.escape(t)}: {r:.1f}x taller"
                                                                       for t, r in big) +
                f"). The box kind draws each line to fill its own box, and the gt kind cuts the real letters out "
                f"of the registered photo. Images below are at half size, so their sizes compare directly.",
                table))

    ex = max(built["fixed"], key=lambda i: i.patch.width * i.patch.height) if built["fixed"] else None
    tok = ""
    if ex is not None:
        bx = by_id["box"].get(ex.item_id)
        tok = figw(grid_overlay(ex.patch), f"'{ex.text}', fixed: {ex.token_hw[0]} x {ex.token_hw[1]} = "
                   f"{ex.token_hw[0] * ex.token_hw[1]} tokens", ex.patch.width // 2, 1200)
        if bx is not None:
            tok += figw(grid_overlay(bx.patch), f"'{bx.text}', box: {bx.token_hw[0]} x {bx.token_hw[1]} = "
                        f"{bx.token_hw[0] * bx.token_hw[1]} tokens", bx.patch.width // 2, 1200)
    sec.append(("4.4 Turn each image into tokens",
                "The FLUX VAE (the same image encoder that encodes the views) shrinks the image 8x per side into "
                "16 numbers per pixel. Each 2 x 2 block of those becomes one token of 64 numbers, so one token "
                "covers 16 x 16 px of the line image (red grid). This is the same kind of token the six views "
                "are made of.", tok))

    fps = []
    for kind in ("fixed", "box"):
        im = draw_boxes(front.copy(), [list(i.bbox) for i in built[kind]], BLACK, max(2, R // 400))
        im = draw_boxes(im, [footprint(i) for i in built[kind]], RED, max(2, R // 512))
        allb = [list(i.bbox) for i in built[kind]] + [footprint(i) for i in built[kind]]
        fps.append(figw(crop_union(im, allb, 24), f"{KIND_NAME[kind]}: black = the text, red = where its tokens "
                        f"claim to be", 560))
    sec.append(("4.5 Give each token a position on the front view",
                "Every token carries a position (plane, row, column). The six views are plane 0, glyph tokens "
                "plane 1. Row and column say where on the front view the token belongs: each line's token grid is "
                "centred on its box. FLUX's position encoding (RoPE) makes the attention between two tokens "
                "depend on how far apart their positions are, which is how a glyph token is tied to one spot of "
                "the front view. With the fixed font, a small line's tokens cover far more than the line itself "
                "and overlap the lines next to it (overlapping positions are pushed apart by up to 2 tokens). "
                "With box or gt, they cover the line.", "".join(fps)))

    grow = [(i.text, int(i.n_keep)) for i in built["fixed"]]
    sec.append(("4.6 Add them to the FLUX input (appended, not added)",
                "The glyph tokens are extra tokens at the end of the sequence. Nothing is added on top of the "
                "view tokens: the views stay as they are, and the glyph tokens sit next to them as hints the "
                "model can look at. FLUX only predicts the six views, its output at the glyph positions is "
                f"thrown away. Here: {len(grow)} lines, {sum(n for _, n in grow):,} glyph tokens with the fixed "
                "font.", sequence_svg(R, grow)))

    c = gl.GlyphConfig()
    p_gt, p_box, p_fixed = c.stages[0][1:4]
    sec.append(("4.7 How the LoRA learned to use them",
                f"Training shows the model six target views of a textured product. The line boxes come from the "
                f"text in the product's own texture, projected into each view, so training positions are exact. "
                f"In the first {c.stages[0][0]:,} steps (all of our 1024 training) each line image is a crop of "
                f"the real text {p_gt:.0%} of the time, drawn to its box {p_box:.0%}, and in the fixed font "
                f"{p_fixed:.0%}. Positions are shifted by up to {c.jitter_tokens} token ({2 * c.jitter_tokens} at "
                f"1024), rendered lines are scaled by {c.aug_scale[0]}-{c.aug_scale[1]} and turned by up to "
                f"{c.aug_rot_deg:g} degrees, {c.p_item_drop:.0%} of lines and {c.p_all_drop:.0%} of all glyphs are "
                f"dropped, so the model also works without them. Only the LoRA (rank 16, on the attention and MLP "
                f"layers of every FLUX block) is trained, and the loss is only on the six views. At inference "
                f"our runs used the fixed font, the kind the model saw least.", ""))

    strip = control_strip(run_dir / "cache", Image.open(run_dir / "cache" / "mv_normal.png").height // 2)
    S = strip.height
    geo = "".join([figw(Image.open(run_dir / "cache" / "mv_normal.png").crop((0, 0, S, S)), "normals, front", 240),
                   figw(Image.open(run_dir / "cache" / "mv_ccm.png").crop((0, 0, S, S)),
                        "CCM: each point's 3D position as colour, front", 240),
                   figw(strip.crop((0, 0, S, S)), "what FLUX gets: the average of the two", 240),
                   figw(strip, "the whole geometry strip, six views", 1100, 2400)])
    sec.append(("4.8 Geometry input of the same pass",
                "The mesh is rendered from six orthographic cameras twice: as surface normals and as CCM (each "
                "surface point's 3D position stored as a colour). The two images are averaged 50/50 into one "
                "geometry image, encoded by the same VAE and given to FLUX as condition tokens, like the photo. "
                "This is UniTEX's own input, we did not change it.", geo))

    if args.vae_json and pathlib.Path(args.vae_json).exists():
        v = json.load(open(args.vae_json))
        sec.append(("VAE token check", v.get("summary_html", ""), v.get("figures_html", "")))

    n_small = sum(1 for _, r in fx if r >= 2.0)
    take = ("<ul>"
            "<li>Placement: each line is put on the front view by scaling and shifting the photo's product box. A "
            "photo taken at an angle (side panel or top in view) puts lines a little off (4.2).</li>"
            f"<li>Size: the fixed font draws every line at the same letter height, so {n_small} of "
            f"{len(fx)} lines here get an image at least 2x taller than the real text, and their tokens overlap "
            "neighbouring lines (4.3, 4.5).</li>"
            "<li>Training: the LoRA mostly saw crops of real text and lines drawn to their box, while our runs "
            "used the fixed font (4.7).</li>"
            "<li>Next test, no retraining needed: registered placement, then the box and gt kinds.</li>")
    if reg_v0:
        moves = []
        for it in items:
            ri = reg_v0.get(it.get("src_id", it["id"]))
            if ri is not None:
                a, b = box_at(it["views"]["0"], f), box_at(ri["views"]["0"], R / reg["res"])
                moves.append(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2) / TOKEN_PX)
        if moves:
            take += (f"<li>Measured here: with the photo registered, line centres move by {np.mean(moves):.1f} tokens "
                     f"on average (up to {max(moves):.1f}).</li>")
    if args.vae_json and pathlib.Path(args.vae_json).exists():
        vm = json.load(open(args.vae_json)).get("mean") or {}
        if all(k in vm for k in ("fixed", "box", "gt")):
            take += (f"<li>VAE tokens (last section): drawn lines share almost nothing with the photo's tokens at "
                     f"their spot (fixed {vm['fixed']['matched_c']:.2f}, box {vm['box']['matched_c']:.2f}), photo "
                     f"crops do ({vm['gt']['matched_c']:.2f}). Edge maps bring drawn lines closer "
                     f"({vm['box']['edges_matched_c']:.2f}).</li>")
    take += "</ul>"
    sec.insert(0, ("What this page shows", "The main points, each explained step by step below.", take))
    body = "".join(f'<div class="stage"><h3>{html.escape(t)}</h3><p class="sub">{p}</p>'
                   f'<div class="figs">{b}</div></div>' for t, p, b in sec)
    title = f"Stage 4, step by step: {html.escape(meta.get('title') or args.sku)}"
    extra = (".pt{border-collapse:collapse;margin-top:6px} .pt td,.pt th{border-top:1px solid var(--line);"
             "padding:6px 10px;text-align:left;vertical-align:middle;font-size:13px} .pt img{border-radius:2px;"
             "background:#fff;border:1px solid var(--line)}")
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-'
            f'width, initial-scale=1"><title>Stage 4 step by step</title><style>{CSS}{extra}</style></head><body>'
            f'<header><h1>{title}</h1><p class="sub">How the photo\'s text becomes glyph tokens, built from this '
            f'product\'s real files (run {html.escape(args.run)}, {R} px per view). Click an image to zoom.</p>'
            f'</header><main>{body}</main><div id="lb"><img><div></div></div><script>{JS}</script></body></html>')
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return page


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--sku", required=True)
    p.add_argument("--run", required=True, help="run under <eval>/<sku>/ for the front view and geometry")
    p.add_argument("--text", default="text.json", help="layout (relative to the SKU dir or absolute)")
    p.add_argument("--text-reg", default=None, help="registered layout, anchors.py --register-run")
    p.add_argument("--view-res", type=int, default=1024)
    p.add_argument("--vae-json", default=None, help="vae_tokens.py output")
    p.add_argument("--out", required=True)
    return build(p.parse_args(argv))


if __name__ == "__main__":
    main()
