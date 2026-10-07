"""
The two figures at the top of the UniTEX walkthrough page (walkthrough_page.py): the 10 stages with the
real output of each one for a product, and the pre-summer pipeline next to UniTEX (what replaces what,
which steps are paid API calls). Plain SVG in a neutral style, dark red only for our additions.

Usage:
  # thanos6, CPU only: thumbnails of every stage from one run cache, plus info.json
  python unitex/walkthrough_figures.py assets --eval-dir $R/unitex_eval_28 --sku 016000233164 \\
      --run g1024_final_photofront --out $R/walkthrough/fig
  # both figures on top of a copy of the walkthrough page (the input page is kept as is)
  python unitex/walkthrough_figures.py page --fig $R/walkthrough/fig --page UniTEX_walkthrough_v1.html \\
      --name "Lucky Charms" --glyphs "18.6,STRAWBERRY,FAMILY SIZE,MARSHMALLOWS,GALACTIC" \\
      --out UniTEX_walkthrough.html --svg-dir figures
"""

import argparse
import base64
import html
import json
import pathlib
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

INK, MUTED, LINE, PANEL = "#1a1a1a", "#555555", "#b9b9b9", "#f1f1f1"
ACC = "#8c1d18"                       # our additions
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
# first to last file written per product on thanos6 (one RTX 4090, models already loaded), 2026-10-06:
# unitex_s63 7.9 to 8.1 min, g1024_final 20.1 to 21.7 min, 24 products
GPU_TIME = ("UniTEX runs on our own GPUs, so no tokens. It costs GPU time instead: about 8 min per product at 512 px "
            "and about 21 min at 1024 px on one RTX 4090 (24 products, not counting model loading).")
OVERVIEW_SUB = ("The 10 stages, with the real output of each one for {name} (our final run). Stages outlined in red are "
                "our additions. (frozen) marks pretrained weights used as released, (trained) marks the part trained on "
                "our data. Details for every stage follow below.")
COMPARE_SUB = ("Left: the current pre-summer pipeline. Right: UniTEX. The dashed arrows show which UniTEX step takes "
               "over each pre-summer step. Grey boxes are paid API calls per product. Our additions are outlined in red.")
NAV = '<a href="#overview">Pipeline overview</a><a href="#compare">Pre-summer vs UniTEX</a>'


# ---- thumbnails from a run cache ---------------------------------------------------------------------------

def crop_fg(im, tol=16, margin=0.04):
    """Crop to everything that differs from the top-left pixel, plus a margin."""
    a = np.asarray(im.convert("RGB")).astype(int)
    fg = np.abs(a - a[0, 0]).max(-1) > tol
    if not fg.any():
        return im
    ys, xs = np.nonzero(fg)
    m = int(margin * max(im.size))
    return im.crop((max(0, xs.min() - m), max(0, ys.min() - m), min(im.width, xs.max() + 1 + m),
                    min(im.height, ys.max() + 1 + m)))


def save(im, path, max_side=520):
    im = im.copy()
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    im.save(path)


def view_rotation(yaw=-32, pitch=16):
    ry, rx = np.radians(yaw), np.radians(pitch)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    return Rx @ Ry


def render(mesh, yaw=-32, pitch=16, res=520, textured=True, base=(206, 208, 214)):
    """Orthographic z-buffer render of a mesh (or .glb) from a three-quarter view, Lambert shaded, RGBA.
    numpy only, so it needs no GPU and no ray-casting backend."""
    import trimesh
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.load(mesh)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry() if hasattr(mesh, "to_geometry") else mesh.dump(concatenate=True)
    vr = mesh.vertices @ view_rotation(yaw, pitch).T
    lo, hi = vr.min(0), vr.max(0)
    ext = (hi - lo)[:2].max() * 1.06
    cx, cy = (lo + hi)[:2] / 2
    P = np.stack([((vr[:, 0] - cx) / ext + 0.5) * res, (0.5 - (vr[:, 1] - cy) / ext) * res], -1)
    Z = vr[:, 2]
    F = mesh.faces
    fn = trimesh.Trimesh(vr, F, process=False).face_normals
    zbuf = np.full((res, res), -np.inf)
    fid = np.full((res, res), -1)
    bary = np.zeros((res, res, 3))
    for f, (a, b, c) in enumerate(F):
        if fn[f, 2] <= 0:
            continue                                    # back face
        pa, pb, pc = P[a], P[b], P[c]
        x0, x1 = int(max(0, np.floor(min(pa[0], pb[0], pc[0])))), int(min(res - 1, np.ceil(max(pa[0], pb[0], pc[0]))))
        y0, y1 = int(max(0, np.floor(min(pa[1], pb[1], pc[1])))), int(min(res - 1, np.ceil(max(pa[1], pb[1], pc[1]))))
        if x0 > x1 or y0 > y1:
            continue
        d = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(d) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((pb[1] - pc[1]) * (xs - pc[0]) + (pc[0] - pb[0]) * (ys - pc[1])) / d
        w1 = ((pc[1] - pa[1]) * (xs - pc[0]) + (pa[0] - pc[0]) * (ys - pc[1])) / d
        w2 = 1 - w0 - w1
        ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not ins.any():
            continue
        zz = (w0 * Z[a] + w1 * Z[b] + w2 * Z[c])[ins]
        yy = (ys[ins] - 0.5).astype(int)
        xx = (xs[ins] - 0.5).astype(int)
        k = zz > zbuf[yy, xx]
        yy, xx = yy[k], xx[k]
        zbuf[yy, xx] = zz[k]
        fid[yy, xx] = f
        bary[yy, xx] = np.stack([w0[ins][k], w1[ins][k], w2[ins][k]], -1)
    hit = fid >= 0
    light = np.array([-0.35, 0.55, 0.76])
    light /= np.linalg.norm(light)
    shade = 0.62 + 0.38 * np.clip(fn[fid[hit]] @ light, 0, 1)
    uv = getattr(mesh.visual, "uv", None)
    if textured and uv is not None:
        puv = (uv[F[fid[hit]]] * bary[hit][:, :, None]).sum(1)
        mat = mesh.visual.material
        tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        T = np.asarray(tex.convert("RGB"))
        H, W = T.shape[:2]
        px = np.clip((puv[:, 0] % 1.0) * (W - 1), 0, W - 1).astype(int)
        py = np.clip((1.0 - puv[:, 1] % 1.0) * (H - 1), 0, H - 1).astype(int)
        col = T[py, px].astype(np.float32)
    else:
        col = np.tile(np.array(base, np.float32), (hit.sum(), 1))
    out = np.zeros((res, res, 4), np.uint8)
    out[hit, :3] = np.clip(col * shade[:, None], 0, 255).astype(np.uint8)
    out[hit, 3] = 255
    return Image.fromarray(out, "RGBA")


def points_thumb(path, yaw=-32, pitch=16, res=520):
    """Surface points splatted from the same three-quarter view, nearer points darker and on top."""
    import trimesh
    p = np.asarray(trimesh.load(path, process=False).vertices) @ view_rotation(yaw, pitch).T
    lo, hi = p.min(0), p.max(0)
    ext = (hi - lo)[:2].max() * 1.06
    cx, cy = (lo + hi)[:2] / 2
    X = ((p[:, 0] - cx) / ext + 0.5) * (res - 3)
    Y = (0.5 - (p[:, 1] - cy) / ext) * (res - 3)
    z = (p[:, 2] - lo[2]) / max(1e-6, hi[2] - lo[2])
    img = np.zeros((res, res, 4), np.uint8)
    for i in np.argsort(p[:, 2]):
        g = int(190 - 140 * z[i])
        img[int(Y[i]):int(Y[i]) + 2, int(X[i]):int(X[i]) + 2] = (g, g, int(g + 18), 255)
    return Image.fromarray(img, "RGBA")


def make_assets(eval_dir, sku, run, out):
    """One thumbnail per stage from <eval>/<sku>/<run>/cache, plus info.json (sizes and glyph texts)."""
    import trimesh
    d = pathlib.Path(eval_dir) / sku
    c = d / run / "cache"
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    R = Image.open(c / "mv_rgb.png").width // 3                            # 2 x 3 grid of views
    save(render(d / "mesh.glb", textured=False), out / "mesh_34.png")
    save(render(c / "w_LTM" / "textured_mesh.glb"), out / "final_34.png")
    save(crop_fg(Image.open(d / "ref.png")), out / "photo.jpg")
    save(crop_fg(Image.open(c / "rembg_image.png").convert("RGB")), out / "rembg.jpg")
    n = np.asarray(Image.open(c / "mv_normal.png").convert("RGB")).astype(np.float32)
    g = np.asarray(Image.open(c / "mv_ccm.png").convert("RGB")).astype(np.float32)
    save(Image.fromarray((0.5 * n + 0.5 * g).astype(np.uint8)), out / "geometry.jpg", 900)
    lit = np.asarray(Image.open(c / "mv_rgb_w_light.png").convert("RGB"))  # 1 x 6 strip, shown as 2 x 3
    save(Image.fromarray(np.concatenate([lit[:, :3 * R], lit[:, 3 * R:]], 0)), out / "texture_pass.jpg", 900)
    delit = c / "mv_rgb_generated.png"                                      # photo_front.py keeps the views before it
    save(Image.open(delit if delit.exists() else c / "mv_rgb.png").convert("RGB"), out / "delight.jpg", 900)
    save(crop_fg(Image.open(c / "mv_rgb.png").convert("RGB").crop((0, 0, R, R))), out / "photo_front.jpg")
    uv = Image.open(c / "wo_LTM" / "completed_uv.png").convert("RGB")
    save(uv, out / "uv.jpg")
    save(points_thumb(c / "coarse_pcd_fps.ply"), out / "points.png")
    tok = c / "glyph_tokens.json"
    info = {"sku": sku, "run": run, "view_res": R, "uv_res": uv.width,
            "faces": len(trimesh.load(d / "mesh.glb", force="mesh").faces),
            "points": len(trimesh.load(c / "coarse_pcd_fps.ply", process=False).vertices),
            "glyphs": [i["text"] for i in json.load(open(tok))["instances"]] if tok.exists() else []}
    json.dump(info, open(out / "info.json", "w"), indent=1)
    return info


# ---- figures -----------------------------------------------------------------------------------------------

def overview_svg(fig, name, glyphs=None):
    """The 10 stages as cards, each with its real output for one product (fig: the assets folder). glyphs picks
    the glyph texts to show (only ones the run used, at most 5, default the first 5)."""
    fig = pathlib.Path(fig)
    info = json.load(open(fig / "info.json"))
    W, CW, CH, GAPX = 1600, 276, 360, 34
    X0 = (W - (5 * CW + 4 * GAPX)) // 2
    ROWY = [92, 548]
    PAD = 12
    cx = [X0 + i * (CW + GAPX) for i in range(5)]
    out = []

    def data_uri(fn):
        p = fig / fn
        mime = "image/png" if p.suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()

    def text(x, y, s, size=13, color=INK, weight=400, anchor="start", spacing=None, italic=False):
        sp = f' letter-spacing="{spacing}"' if spacing else ""
        it = ' font-style="italic"' if italic else ""
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
                   f'text-anchor="{anchor}"{sp}{it}>{escape(s)}</text>')

    def spans(x, y, parts, size=12.5):
        """One line of text from (string, colour, weight) parts."""
        t = "".join(f'<tspan fill="{c}" font-weight="{w}">{escape(s)}</tspan>' for s, c, w in parts)
        out.append(f'<text x="{x}" y="{y}" font-size="{size}">{t}</text>')

    def card(n, x, y, title, images, caption, module, ours=False, chips=None):
        stroke, sw = (ACC, 1.8) if ours else (LINE, 1.0)
        out.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="5" fill="#ffffff" stroke="{stroke}" '
                   f'stroke-width="{sw}"/>')
        out.append(f'<circle cx="{x + 22}" cy="{y + 24}" r="11" fill="{ACC if ours else INK}"/>')
        text(x + 22, y + 28.5, str(n), 12, "#ffffff", 700, "middle")
        text(x + 40, y + 29, title, 15.5, INK, 700)
        if ours:
            text(x + CW - 14, y + 29, "OURS", 11, ACC, 700, "end", "0.08em")
        ix, iy, iw, ih = x + 12, y + 46, CW - 24, 206
        out.append(f'<rect x="{ix}" y="{iy}" width="{iw}" height="{ih}" rx="2" fill="{PANEL}"/>')
        if chips:
            gy = iy + 22
            for g in chips:
                gw = 22 + 10.5 * len(g)
                gx = ix + (iw - gw) / 2
                out.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="28" rx="1.5" fill="#ffffff" stroke="#9a9a9a" '
                           f'stroke-width="0.8"/>')
                out.append(f'<text x="{gx + gw / 2}" y="{gy + 20}" font-size="16" fill="#000" text-anchor="middle">'
                           f'{escape(g)}</text>')
                gy += 36
        elif images:
            k, pad = len(images), 8
            cw = (iw - pad * (k + 1)) / k
            for j, fn in enumerate(images):
                out.append(f'<image href="{data_uri(fn)}" x="{ix + pad + j * (cw + pad)}" y="{iy + pad}" width="{cw}" '
                           f'height="{ih - 2 * pad}" preserveAspectRatio="xMidYMid meet"/>')
        ty = y + 276
        for line in caption:
            text(x + 13, ty, line, 12.5, "#333333")
            ty += 17
        out.append(f'<line x1="{x + 12}" y1="{y + CH - 34}" x2="{x + CW - 12}" y2="{y + CH - 34}" stroke="#e2e2e2"/>')
        if module:
            spans(x + 13, y + CH - 14, module, 12)

    def arrow_h(x1, x2, y):
        out.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#4d4d4d" stroke-width="1.4" '
                   f'marker-end="url(#ah3)"/>')

    def phase(i0, i1, row, label):
        x = cx[i0] - PAD
        w = cx[i1] + CW + PAD - x
        y = ROWY[row] - 38
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{CH + 38 + PAD}" rx="6" fill="none" stroke="#9c9c9c" '
                   f'stroke-width="1" stroke-dasharray="5 4"/>')
        text(x + 14, y + 23, label, 12, "#2b2b2b", 700, spacing="0.1em")

    phase(0, 2, 0, "I. PREPARATION")
    phase(3, 4, 0, "II. MULTI-VIEW GENERATION")
    phase(0, 1, 1, "II. CONTINUED")
    phase(2, 4, 1, "III. BAKING AND COMPLETION")
    show = [g for g in (glyphs or info["glyphs"]) if g in info["glyphs"]][:5]
    FZ = ("(frozen)", MUTED, 400)
    TR = ("(trained)", ACC, 700)
    cards = [
        (1, "Inputs", ["photo.jpg", "mesh_34.png"],
         ["Product photo and untextured mesh", f"({info['faces'] / 1000:.0f}k triangles, from ShapeGen)."],
         [("Input", MUTED, 400)], False, None),
        (2, "Preprocessing", ["rembg.jpg"], ["Background removed, product centred,", "mesh normalised."],
         [("RMBG-2.0 ", INK, 600), FZ], False, None),
        (3, "Geometry renders", ["geometry.jpg"], ["Six orthographic views of position and",
                                                   "normals: the shape condition for FLUX."],
         [("Rasteriser: nvdiffrast", MUTED, 400)], False, None),
        (4, "Glyph tokens", None, ["Photo text (OCR), drawn as images and", "placed where it sits on the front view."],
         [("OCR: Apple Vision", MUTED, 400)], True, show),
        (5, "Texture pass", ["texture_pass.jpg"], ["FLUX denoises all six views jointly,",
                                                   "conditioned on shape, photo and glyphs."],
         [("FLUX ", INK, 600), FZ, (" + texture LoRA ", INK, 600), TR], False, None),
        (6, "Delight pass", ["delight.jpg"], ["Lighting removed, leaving albedo", "(flat colour) views."],
         [("FLUX + delight LoRA ", INK, 600), FZ], False, None),
        (7, "Photo front", ["photo_front.jpg"], ["Photo registered to the front view",
                                                 "(homography), its detail transferred."],
         [("Image registration, no learning", MUTED, 400)], True, None),
        (8, "Bake to UV", ["uv.jpg"], ["Views back-projected onto the", f"{info['uv_res']} px UV texture."],
         [("Rasteriser: nvdiffrast", MUTED, 400)], False, None),
        (9, "Surface sampling", ["points.png"], [f"{info['points']:,} surface points", "(farthest-point sampling)."],
         [("Geometry only", MUTED, 400)], False, None),
        (10, "LTM and final bake", ["final_34.png"], ["LTM predicts colour where no view", "reached, then the final bake."],
         [("LTM ", INK, 600), FZ], False, None),
    ]
    for k, (n, title, ims, cap, mod, ours, chips) in enumerate(cards):
        row, col = divmod(k, 5)
        card(n, cx[col], ROWY[row], title, ims, cap, mod, ours, chips)
        if col < 4:
            arrow_h(cx[col] + CW + 5, cx[col + 1] - 6, ROWY[row] + 149)
    x5, x6 = cx[4] + CW / 2, cx[0] + CW - 50                              # card 5 down to card 6
    ymid = (ROWY[0] + CH + PAD + ROWY[1] - 38) / 2
    ybot = ROWY[1] - 6
    out.append(f'<path d="M{x5},{ROWY[0] + CH + 4} L{x5},{ymid - 10} Q{x5},{ymid} {x5 - 10},{ymid} L{x6 + 10},{ymid} '
               f'Q{x6},{ymid} {x6},{ymid + 10} L{x6},{ybot}" fill="none" stroke="#4d4d4d" stroke-width="1.4" '
               f'marker-end="url(#ah3)"/>')
    ly = ROWY[1] + CH + PAD + 34
    out.append(f'<rect x="{X0}" y="{ly - 13}" width="34" height="18" rx="3" fill="#fff" stroke="{ACC}" stroke-width="1.8"/>')
    text(X0 + 44, ly + 1, "our addition", 13, "#333")
    spans(X0 + 170, ly + 1, [("(frozen)", MUTED, 400), (" pretrained weights, used as released", "#333", 400)], 13)
    spans(X0 + 480, ly + 1, [("(trained)", ACC, 700), (" fine-tuned on our data", "#333", 400)], 13)
    text(W - X0, ly + 1, f"All images are real outputs for one product ({name}), final run at {info['view_res']} px "
         f"per view.", 12.5, MUTED, anchor="end", italic=True)
    H = ly + 24
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
            f'aria-label="UniTEX pipeline, 10 stages, with our additions" style="font-family:{FONT};display:block">'
            f'<defs><marker id="ah3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4d4d4d"/></marker></defs>'
            f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>' + "".join(out) + "</svg>")


def compare_svg():
    """The pre-summer pipeline next to UniTEX, row by row: what replaces what, the paid API calls, our additions."""
    W = 1400
    LX, RX, CW = 40, 840, 520                                              # left / right column x and width
    LC, RC = LX + CW // 2, RX + CW // 2
    LINE_H, PAD, GAP, ROWGAP = 19, 14, 12, 30
    neutral = dict(fill="#ffffff", stroke=LINE, sw=1.0, tag=INK)
    paid = dict(fill="#ececec", stroke="#8f8f8f", sw=1.0, tag=INK)
    ours = dict(fill="#ffffff", stroke=ACC, sw=1.8, tag=ACC)
    note = dict(fill="#ffffff", stroke="#9c9c9c", sw=1.0, tag=MUTED, dash=True, text=MUTED)
    out = []

    def h_of(lines):
        return len(lines) * LINE_H + 2 * PAD

    def box(x, y, w, lines, st, tag=None):
        h = h_of(lines) + (LINE_H if tag else 0)
        dash = ' stroke-dasharray="5 4"' if st.get("dash") else ""
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{st["fill"]}" stroke="{st["stroke"]}" '
                   f'stroke-width="{st["sw"]}"{dash}/>')
        ty = y + PAD + 14
        for ln in lines:
            out.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="14.5" '
                       f'fill="{st.get("text", INK)}">{escape(ln)}</text>')
            ty += LINE_H
        if tag:
            out.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="13" font-weight="700" '
                       f'fill="{st["tag"]}">{escape(tag)}</text>')
        return h

    def arrow(x1, y1, x2, y2, dashed=False):
        st = ' stroke="#a0a0a0" stroke-width="1.1" stroke-dasharray="5 5" marker-end="url(#ahd)"' if dashed else \
            ' stroke="#4d4d4d" stroke-width="1.3" marker-end="url(#ahc)"'
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{st}/>')

    # rows: (left boxes, middle label, right boxes), a box is (lines, style, tag), "NOTE" = nothing replaces it
    rows = [
        ([(["Product photo"], neutral, None)], "same", [(["Product photo"], neutral, None)]),
        ([(["Background removal"], neutral, None)], "same idea", [(["Background removal (RMBG)"], neutral, None)]),
        ([(["ShapeGen makes the mesh"], neutral, None)], "same mesh",
         [(["Same ShapeGen mesh", "(UniTEX does not make meshes)"], neutral, None)]),
        ([(["Gemini 2.5 Pro picks the shape type", "(box, can, bottle, bag)"], paid, "Paid API call")], "dropped",
         "NOTE"),
        ([(["Unwrap into a fixed template", "(box cross, can label strip)"], neutral, None)], "replaced by",
         [(["Automatic UV unwrap", "+ 6 camera renders of the shape"], neutral, None)]),
        ([(["Product name: BlueCart API,", "or Gemini 2.5 Flash reads it from the photo"], paid, "Paid API call"),
          (["Gemini 3.1 Flash Image paints the whole", "texture from the photo in one go"], paid,
           "Paid API call, the main cost")],
         "replaced by",
         [(["FLUX + texture LoRA paints 6 views", "from the photo and the shape"], ours,
           "Ours: retrained glyph LoRA, 1024 px, glyphs from OCR"),
          (["FLUX + delight LoRA removes the lighting"], neutral, None),
          (["Photo front: paste the photo's detail", "onto the front view"], ours, "Ours")]),
        ([(["Wrap the texture onto the mesh"], neutral, None)], "replaced by",
         [(["Bake the 6 views onto the texture"], neutral, None),
          (["LTM fills spots no camera saw"], neutral, None)]),
        ([(["Textured 3D model (GLB)"], neutral, None)], "same output", [(["Textured 3D model (GLB)"], neutral, None)]),
    ]

    def stack_h(bs):
        return sum(h_of(b[0]) + (LINE_H if b[2] else 0) for b in bs) + GAP * 2 * (len(bs) - 1)

    def column(boxes, x, c, top, prev):
        if prev is not None:
            arrow(c, prev, c, top - 4)
        yy = top
        for i, (lines, st, tag) in enumerate(boxes):
            if i:
                arrow(c, yy - 2 * GAP + 2, c, yy - 4)
            yy += box(x, yy, CW, lines, st, tag) + 2 * GAP
        return yy - 2 * GAP

    y = 74
    out.append(f'<text x="{LC}" y="40" text-anchor="middle" font-size="18" font-weight="700" fill="{INK}">'
               f'Pre-summer pipeline (CFI-3DGen)</text>')
    out.append(f'<text x="{RC}" y="40" text-anchor="middle" font-size="18" font-weight="700" fill="{INK}">'
               f'UniTEX (stock, and ours)</text>')
    out.append(f'<text x="{W / 2}" y="40" text-anchor="middle" font-size="13.5" fill="{MUTED}">what replaces what</text>')
    prev_l = prev_r = None
    for lb, mid, rb in rows:
        lh = stack_h(lb)
        rh = 2 * LINE_H + 2 * PAD if rb == "NOTE" else stack_h(rb)
        rowh = max(lh, rh)
        prev_l = column(lb, LX, LC, y + (rowh - lh) / 2, prev_l)
        if rb == "NOTE":
            box(RC + 70, y + (rowh - rh) / 2, CW / 2 - 70, ["Not needed:", "works on any shape"], note)
        else:
            prev_r = column(rb, RX, RC, y + (rowh - rh) / 2, prev_r)
        cy = y + rowh / 2
        arrow(LX + CW + 8, cy, RX - 10, cy, dashed=True)
        out.append(f'<rect x="{W / 2 - 58}" y="{cy - 13}" width="116" height="24" fill="#ffffff"/>')
        out.append(f'<text x="{W / 2}" y="{cy + 4}" text-anchor="middle" font-size="13.5" fill="{MUTED}" '
                   f'font-style="italic">{escape(mid)}</text>')
        y += rowh + ROWGAP

    y += 4                                                                 # legend
    x = LX
    for st, label in ((paid, "paid API call per product (tokens)"), (ours, "our addition"), (note, "not needed")):
        dash = ' stroke-dasharray="5 4"' if st.get("dash") else ""
        out.append(f'<rect x="{x}" y="{y}" width="34" height="18" rx="3" fill="{st["fill"]}" stroke="{st["stroke"]}" '
                   f'stroke-width="{st["sw"]}"{dash}/>')
        out.append(f'<text x="{x + 44}" y="{y + 14}" font-size="13.5" fill="#333333">{escape(label)}</text>')
        x += 70 + 7.6 * len(label)
    out.append(f'<text x="{LX}" y="{y + 46}" font-size="13.5" fill="#333333">{escape(GPU_TIME)}</text>')
    H = y + 70
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
            f'aria-label="Pre-summer pipeline versus UniTEX, step by step" style="font-family:{FONT};display:block">'
            f'<defs><marker id="ahc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4d4d4d"/></marker>'
            f'<marker id="ahd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#a0a0a0"/></marker></defs>'
            f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>' + "".join(out) + "</svg>")


# ---- page --------------------------------------------------------------------------------------------------

def add_figures(doc, overview, compare, name):
    """Both figures at the top of a walkthrough page, with nav links, the overview first. Figures already on
    the page are replaced, so running this on its own output gives the same page as on the original."""
    assert doc.count("<nav>") == 1 and doc.count("<main>") == 1
    for sid, label in (("overview", "Pipeline overview"), ("compare", "Pre-summer vs UniTEX")):
        doc = doc.replace(f'<a href="#{sid}">{label}</a>', "", 1)
        if f'<h2 id="{sid}">' in doc:
            a = doc.index(f'<h2 id="{sid}">')
            doc = doc[:a] + doc[doc.index("</svg></div>", a) + len("</svg></div>"):]
    sections = (f'<h2 id="overview">Pipeline overview</h2><p class="sub">{OVERVIEW_SUB.format(name=html.escape(name))}'
                f'</p><div class="stage">{overview}</div>'
                f'<h2 id="compare">Pre-summer vs UniTEX: what replaces what</h2><p class="sub">{COMPARE_SUB}</p>'
                f'<div class="stage">{compare}</div>')
    return doc.replace("<nav>", "<nav>" + NAV, 1).replace("<main>", "<main>" + sections, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assets", help="thumbnails of every stage from one run cache, plus info.json")
    a.add_argument("--eval-dir", required=True)
    a.add_argument("--sku", required=True)
    a.add_argument("--run", required=True)
    a.add_argument("--out", required=True)
    p = sub.add_parser("page", help="both figures on top of a copy of a walkthrough page")
    p.add_argument("--fig", required=True, help="output folder of the assets step")
    p.add_argument("--page", required=True, help="walkthrough_page.py output, kept as is")
    p.add_argument("--name", required=True, help="product name for the captions")
    p.add_argument("--glyphs", default="", help="comma list of glyph texts to show (default: the run's first 5)")
    p.add_argument("--out", required=True)
    p.add_argument("--svg-dir", help="also save both figures as .svg files here")
    args = ap.parse_args(argv)
    if args.cmd == "assets":
        info = make_assets(args.eval_dir, args.sku, args.run, args.out)
        print(f"Wrote {args.out} ({info['view_res']} px views, {info['faces']:,} faces, {len(info['glyphs'])} glyphs)")
        return
    out, page = pathlib.Path(args.out), pathlib.Path(args.page)
    if out.resolve() == page.resolve():
        raise SystemExit("--out must differ from --page: the input page is kept as is")
    overview = overview_svg(args.fig, args.name, [g for g in args.glyphs.split(",") if g])
    compare = compare_svg()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(add_figures(page.read_text(encoding="utf-8"), overview, compare, args.name), encoding="utf-8")
    if args.svg_dir:
        d = pathlib.Path(args.svg_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "pipeline_overview.svg").write_text(overview, encoding="utf-8")
        (d / "presummer_vs_unitex.svg").write_text(compare, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
