"""
UniTEX, stage by stage: the intermediate output of every stage for a few products, stock UniTEX next
to ours, in one self-contained page (every image embedded as WebP, click to zoom).

Everything comes from the run caches UniTEX writes (<eval>/<sku>/<run>/cache/, steps step_1_1 and
step_2_ablition of pipeline.py) plus views of the baked meshes before and after LTM
(<renders>/<run>_<wo_LTM|w_LTM>/<sku>/view_0X.png, mvgen.py --mode views). Derived images: the
geometry strip FLUX sees (computed like infer_mv), our glyph boxes and patches (unitex.glyph_tokens
on the same text.json), and point clouds drawn with an orthographic splat.

Usage (thanos6):
  python unitex/walkthrough_page.py --eval-dir $R/unitex_eval_28 --renders $R/walkthrough/renders \\
      --skus 016000233164,072360002031 --out $R/walkthrough/UniTEX_walkthrough.html
"""

import argparse
import base64
import html
import io
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

RUNS = [("unitex_s63", "Stock UniTEX", 512),
        ("g1024_final_photofront", "Ours: glyph LoRA at 1024 px + photo front", 1024)]
STRIP = ["front", "left", "right", "back", "top", "bottom"]       # texture-pass strip order
GRID = "2 x 3 grid: front, right, top / back, left, bottom"
RAW_TO_SLOT = {0: 0, 3: 1, 1: 2, 2: 3, 4: 4, 5: 5}                # common.FULL_INDEX inverted
VIEWS = [("0", "front"), ("3", "left"), ("2", "back"), ("1", "right"), ("4", "top"), ("5", "bottom")]
GREY = (128, 128, 128)

CSS = """
:root { --bg:#f6f6f4; --card:#fff; --ink:#1d1d1f; --muted:#6e6e73; --line:#e3e3e0; --accent:#b4441c; }
@media (prefers-color-scheme: dark) { :root { --bg:#121213; --card:#1c1c1e; --ink:#f2f2f2; --muted:#9a9aa0; --line:#2c2c2e; --accent:#ff8a5c; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 -apple-system,system-ui,sans-serif; }
header, main { max-width:1500px; margin:auto; padding:0 20px; } header { padding-top:24px; }
h1 { font-size:26px; margin:0 0 6px; } h2 { font-size:21px; margin:40px 0 4px; } h3 { font-size:17px; margin:0 0 4px; }
.sub { color:var(--muted); margin:2px 0 10px; } nav a { color:var(--accent); margin-right:14px; text-decoration:none; }
ol.steps { margin:8px 0 0 18px; padding:0; } ol.steps li { margin:2px 0; }
.stage { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin:14px 0; }
.ours { color:var(--accent); font-weight:600; }
.cols { display:grid; grid-template-columns: repeat(var(--c), minmax(0, 1fr)); gap:14px; margin-top:8px; }
.col h4 { margin:0 0 6px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.figs { display:flex; flex-wrap:wrap; gap:8px; } figure { margin:0; flex:1 1 var(--w, 260px); min-width:0; }
figcaption { font-size:12px; color:var(--muted); margin-top:3px; }
img { width:100%; border-radius:6px; cursor:zoom-in; display:block; background:#808080; }
.patches { display:flex; flex-wrap:wrap; gap:6px; align-items:flex-end; } .patches figure { flex:0 0 auto; }
.patches img { height:48px; width:auto; background:#fff; border:1px solid var(--line); }
.stats { font-size:13px; color:var(--muted); }
#lb { position:fixed; inset:0; background:rgba(0,0,0,.9); display:none; overflow:auto; text-align:center; z-index:9; }
#lb img { display:block; margin:3vh auto 0; max-width:96vw; max-height:90vh; width:auto; cursor:zoom-in; }
#lb.full img { max-width:none; max-height:none; cursor:zoom-out; } #lb div { color:#ddd; margin:8px; }
@media (max-width: 900px) { .cols { grid-template-columns: 1fr; } }
"""
JS = """
const lb=document.getElementById('lb'), li=lb.querySelector('img'), lc=lb.querySelector('div');
document.querySelectorAll('main img').forEach(im=>im.onclick=()=>{li.src=im.src;lc.textContent=im.dataset.caption+'  (click the image for full size, anywhere else to close)';lb.classList.remove('full');lb.style.display='block';lb.scrollTo(0,0);});
li.onclick=e=>{e.stopPropagation();lb.classList.toggle('full');};
lb.onclick=()=>lb.style.display='none'; document.onkeydown=e=>{if(e.key==='Escape')lb.style.display='none';};
"""


def on_grey(im):
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    if im.mode == "RGBA":
        im = Image.alpha_composite(Image.new("RGBA", im.size, GREY + (255,)), im)
    return im.convert("RGB")


def webp(im, max_side, quality=82):
    im = on_grey(im)
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=4)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def fig(src, caption, max_side=1600, width=None):
    if isinstance(src, (str, pathlib.Path)):
        if not pathlib.Path(src).exists():
            return f'<figure><div class="stats">missing: {html.escape(str(src))}</div></figure>'
        src = Image.open(src)
    style = f' style="--w:{width}px"' if width else ""
    return (f'<figure{style}><img src="{webp(src, max_side)}" data-caption="{html.escape(caption)}" loading="lazy">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def control_strip(cache, R):
    """The geometry condition of the texture pass, built like infer_mv: mean of the normal and CCM
    grids, bottom tile turned 180 degrees, tiles reordered into a 1 x 6 strip."""
    n = np.asarray(Image.open(cache / "mv_normal.png").convert("RGB")).reshape(2, R, 3, R, 3).astype(np.float32)
    c = np.asarray(Image.open(cache / "mv_ccm.png").convert("RGB")).reshape(2, R, 3, R, 3).astype(np.float32)
    t = (0.5 * n + 0.5 * c).astype(np.uint8)
    t[1, :, 2] = t[1, ::-1, 2, ::-1]
    return Image.fromarray(t.transpose(0, 2, 1, 3, 4).reshape(6, R, R, 3)[[0, 4, 1, 3, 2, 5]]
                           .transpose(1, 0, 2, 3).reshape(R, 6 * R, 3))


def glyph_boxes(strip, info, R):
    """Glyph instances of glyph_tokens.json drawn on the texture-pass strip."""
    im = strip.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w = max(2, R // 256)
    for g in info["instances"]:
        x0, y0, x1, y1 = g["bbox"]
        o = RAW_TO_SLOT[g["raw_view"]] * R
        d.rectangle((x0 + o, y0, x1 + o, y1), outline=(255, 30, 30), width=w)
        d.text((x0 + o + 2, max(0, y0 - 12)), g["text"][:30], fill=(255, 30, 30))
    return im


def glyph_patches(text_json, R):
    from unitex import glyph_tokens as gt
    return gt.build_infer_glyphs(gt.load_text(str(text_json)), None, view_res=R)


def points_image(path, res=900, view="front", context=None, gain=1.0, colour=None, size=2):
    """Orthographic splat of a point cloud (glTF frame, y up, front camera on +z), far points first."""
    import trimesh
    q = trimesh.load(path, process=False)
    v = np.asarray(q.vertices, dtype=np.float32)
    if colour is not None:
        c = np.tile(np.array(colour, np.uint8), (len(v), 1))
    else:
        c = np.asarray(q.visual.vertex_colors)[:, :3].astype(np.float32) * gain
        c = np.clip(c, 0, 255).astype(np.uint8)
    img = np.full((res, res, 3), 238, np.uint8)
    layers = [(context, np.array((205, 205, 205), np.uint8))] if context is not None else []
    for k, (pts, col) in enumerate(layers + [(v, c)]):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        if view == "back":
            x, z = -x, -z
        order = np.argsort(z)
        s = 2 if k < len(layers) else size
        px = np.clip(((x + 1) / 2 * (res - 1)).round().astype(int), 0, res - s)
        py = np.clip(((1 - (y + 1) / 2) * (res - 1)).round().astype(int), 0, res - s)
        for dy in range(s):
            for dx in range(s):
                img[py[order] + dy, px[order] + dx] = col[order] if col.ndim == 2 else col
    return Image.fromarray(img)


def card(n, name, text, cols):
    """cols: [(heading, html)] side by side."""
    inner = "".join(f'<div class="col"><h4>{html.escape(h)}</h4>{body}</div>' for h, body in cols)
    return (f'<div class="stage"><h3>{n}. {name}</h3><p class="sub">{text}</p>'
            f'<div class="cols" style="--c:{len(cols)}">{inner}</div></div>')


def product(eval_dir, renders, sku):
    import trimesh
    d = eval_dir / sku
    title = json.load(open(d / "manifest_entry.json")).get("title", "") if (d / "manifest_entry.json").exists() else ""
    caches = {run: d / run / "cache" for run, _, _ in RUNS}
    s_run, o_run = RUNS[0][0], RUNS[1][0]
    s_cache, o_cache = caches[s_run], caches[o_run]
    mesh_in = trimesh.load(d / "mesh.glb", force="mesh")
    mesh_pp = trimesh.load(s_cache / "processed_mesh.obj", force="mesh", process=False)
    parts = [f'<h2 id="p{sku}">{html.escape(title[:100])} <span class="stats">{sku}</span></h2>']
    n = 0

    def nxt():
        nonlocal n
        n += 1
        return n

    # 1. inputs
    normal_front = Image.open(o_cache / "mv_normal.png")
    R = RUNS[1][2]
    parts.append(card(nxt(), "Inputs", "UniTEX takes one product photo and an untextured mesh (here the CFI-3DGen mesh that the pre-summer pipeline also uses, "
                      f"{len(mesh_in.faces):,} faces). Nothing else.",
                      [("Shared", '<div class="figs">' + fig(d / "ref.png", "product photo (input)", 1400)
                        + fig(normal_front.crop((0, 0, R, R)), "input mesh, untextured (front, shaded by normals)", 1024)
                        + "</div>")]))
    # 2. preprocessing
    parts.append(card(
        nxt(), "Preprocessing (step_1_1: preprocess_blank_mesh, preprocess_reference_image)",
        "The mesh is normalised to fill 95% of the unit cube and remeshed only when it has fewer than 20k or more than "
        f"200k faces (here {len(mesh_in.faces):,} faces in, {len(mesh_pp.faces):,} out). The photo goes through an RMBG "
        "background-removal model, is centred at 95% of a 1024 px square on grey, then resized to the view resolution.",
        [("Shared", '<div class="figs">' + fig(s_cache / "rembg_image.png", "background removed, centred (rembg_image.png, 1024 px)", 1024)
          + "</div>")]
        + [(label, '<div class="figs">' + fig(caches[run] / "processed_image.png", f"processed_image.png, {res} px", 1024)
            + "</div>") for run, label, res in RUNS]))
    # 3. geometry renders
    geo = '<div class="figs">' + "".join(fig(o_cache / f, f"{f} ({GRID})", 2048, 380)
                                          for f in ("mv_alpha.png", "mv_ccm.png", "mv_normal.png")) + "</div>"
    strip = control_strip(o_cache, R)
    parts.append(card(
        nxt(), "Geometry renders (step_1_1: render_geometry_images)",
        "The mesh is rendered from 6 orthographic cameras: silhouette (alpha), canonical coordinate map (CCM, the 3D "
        "position of each pixel as a colour) and normals. The mean of the normal and CCM images, laid out as one 1 x 6 "
        "strip, is the geometry condition FLUX sees. The same cameras are used later to bake the texture. Stock UniTEX "
        f"renders these at 512 px per view, ours at 1024.",
        [("Renders (ours, 1024 px per view)", geo
          + '<div class="figs">' + fig(strip, "geometry condition for FLUX: (normal + CCM) / 2, strip " + ", ".join(STRIP), 3072)
          + "</div>")]))
    # 4. glyphs (ours)
    gi = json.load(open(o_cache / "glyph_tokens.json")) if (o_cache / "glyph_tokens.json").exists() else None
    if gi:
        insts = glyph_patches(d / "text.json", R)
        patches = '<div class="patches">' + "".join(fig(i.patch, i.text[:40], 512) for i in insts) + "</div>"
        lit = Image.open(o_cache / "mv_rgb_w_light.png")
        parts.append(card(
            nxt(), '<span class="ours">Ours:</span> glyph tokens',
            f"Text read from the photo (OCR), mapped onto the 6 views (text.json, {gi['n_items']} lines). Each line is "
            f"rendered black on white, VAE-encoded and appended to FLUX's input at the position of that text in its "
            f"view (center anchoring). Here {gi['n_instances']} glyphs, {gi['n_tokens']} tokens. Not used by stock UniTEX.",
            [("Glyph patches", patches),
             ("Where they are placed (boxes on the texture-pass output)", '<div class="figs">'
              + fig(glyph_boxes(lit, gi, R), "glyph boxes, strip " + ", ".join(STRIP), 3072) + "</div>")]))
    # 5. texture pass
    parts.append(card(
        nxt(), "Texture pass (step_1_1: infer_mv, texture LoRA)",
        "FLUX.1-dev with UniTEX's texture LoRA paints all 6 views at once as one image, conditioned on the geometry strip "
        "and the photo (the photo enters as extra tokens). 28 steps, guidance 3.5. The output still has lighting "
        "(mv_rgb_w_light.png). Ours uses our glyph LoRA, 1024 px per view and the glyph tokens.",
        [(label, '<div class="figs">' + fig(caches[run] / "mv_rgb_w_light.png", f"mv_rgb_w_light.png, {res} px per view, "
                                             "strip " + ", ".join(STRIP), 3072) + "</div>") for run, label, res in RUNS]))
    # 6. delight pass
    ours_delight = ('<div class="figs">' + fig(o_cache / "mv_rgb_delit_512.png", "delight output at 512 px per view "
                                               "(mv_rgb_delit_512.png)", 3072)
                    + fig(o_cache / "mv_rgb_generated.png", "1024 detail added back, before photo front "
                          f"({GRID})", 3072) + "</div>")
    parts.append(card(
        nxt(), "Delight pass (step_1_1: infer_mv, delight LoRA)",
        "The same FLUX with UniTEX's delight LoRA takes the lit strip and removes lighting, giving albedo views, then "
        "they are arranged in the 2 x 3 grid used for baking (mv_rgb.png). The released delight LoRA is a 512 model, "
        "so ours runs it at 512 and adds the texture pass's 1024 detail back.",
        [(RUNS[0][1], '<div class="figs">' + fig(s_cache / "mv_rgb.png", f"mv_rgb.png ({GRID})", 3072) + "</div>"),
         (RUNS[1][1], ours_delight)]))
    # 7. photo front (ours)
    pf = d / o_run / "photo_front.png"
    if pf.exists():
        info = json.load(open(d / o_run / "run_info.json"))
        reg = info.get("register", {})
        parts.append(card(
            nxt(), '<span class="ours">Ours:</span> photo front',
            "The photo is aligned to the generated front view (box fit, then a SIFT homography, correlation "
            f"{reg.get('ncc_final')}), and its detail replaces the front view where the surface faces the camera and the "
            "two agree locally. Not used by stock UniTEX.",
            [("Generated front | aligned photo | weight | result", '<div class="figs">' + fig(pf, "photo_front.png", 4096) + "</div>"),
             ("Grid that gets baked", '<div class="figs">' + fig(o_cache / "mv_rgb.png", f"mv_rgb.png ({GRID})", 3072) + "</div>")]))
    # 8. bake without LTM
    def renders_html(run, stage, caption):
        return '<div class="figs">' + "".join(fig(renders / f"{run}_{stage}" / sku / f"view_0{v}.png", f"{caption}, {name}", 768, 150)
                                              for v, name in VIEWS) + "</div>"
    parts.append(card(
        nxt(), "Bake (step_2_ablition: reproject_and_query_field, no inpainting)",
        "The 6 views are projected back onto the mesh's 2048 px UV texture with the render cameras (inverse rendering, "
        "nvdiffrast). White in the mask: texels a view covers. The space between UV islands is only padding. Surface "
        "that no view covers is left for the LTM.",
        [(label, '<div class="figs">' + fig(caches[run] / "wo_LTM" / "completed_uv.png", "UV texture (wo_LTM/completed_uv.png)", 1400, 300)
          + fig(caches[run] / "wo_LTM" / "visable_uv_mask.png", "texels seen by a view (visable_uv_mask.png)", 1400, 300) + "</div>"
          + renders_html(run, "wo_LTM", "baked, before LTM")) for run, label, _ in RUNS]))
    # 9. surface sampling
    parts.append(card(
        nxt(), "Surface sampling (step_2_ablition: sampling_on_mesh)",
        "200,000 points are sampled on the surface, then 32,768 kept by farthest-point sampling, once on sharp edges and "
        "once on the whole surface. They describe the geometry for the LTM.",
        [("Shared", '<div class="figs">'
          + fig(points_image(s_cache / "coarse_pcd_fps.ply", view="front", colour=(70, 70, 70)), "coarse_pcd_fps.ply (32,768 points), front", 900, 300)
          + fig(points_image(s_cache / "sharp_pcd_fps.ply", view="front", colour=(70, 70, 70)), "sharp_pcd_fps.ply (32,768 points), front", 900, 300)
          + "</div>")]))
    # 10. LTM
    def ltm_html(run):
        c = caches[run] / "w_LTM"
        import trimesh as _t
        n_in = len(_t.load(c / "pcd_input.ply", process=False).vertices)
        n_out = len(_t.load(c / "pcd_output.ply", process=False).vertices)
        ctx = np.asarray(_t.load(s_cache / "coarse_pcd_fps.ply", process=False).vertices, dtype=np.float32)
        share = 100.0 * n_out / max(1, n_in + n_out)
        figs = (fig(points_image(c / "pcd_input.ply", view="front"), f"points coloured from the views, front ({n_in:,})", 900, 220)
                + fig(points_image(c / "pcd_input.ply", view="back"), "same, back", 900, 220)
                + fig(points_image(c / "pcd_output.ply", view="front", context=ctx, size=4),
                      f"points no view saw, coloured by LTM, front ({n_out:,}, {share:.1f}% of the surface points)", 900, 220)
                + fig(points_image(c / "pcd_output.ply", view="back", context=ctx, size=4), "LTM points, back", 900, 220))
        return ('<div class="figs">' + figs + "</div><div class=\"figs\">"
                + fig(c / "completed_uv.png", "final UV texture (w_LTM/completed_uv.png)", 1400, 300)
                + fig(c / "visable_uv_mask.png", "w_LTM/visable_uv_mask.png", 1400, 300) + "</div>"
                + renders_html(run, "w_LTM", "final"))
    parts.append(card(
        nxt(), "LTM completion and final bake (step_2_ablition: infer_field, reproject_and_query_field with inpainting)",
        "UniTEX's Large Texturing Model (an RGB field VAE) encodes the 6 albedo views, CCMs, silhouettes and 32,768 surface "
        "points (visible ones carry their colour from the views), then predicts a colour for every surface point no view "
        "saw. Those colours fill the UV texture where the views did not reach, and the mesh is baked again. This is the "
        "final textured mesh. For these two products the 6 views already cover almost the whole surface, so LTM only "
        "fills thin edges and rims and the before and after views look nearly the same.",
        [(label, ltm_html(run)) for run, label, _ in RUNS]))
    return "".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--skus", required=True, help="comma list")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    eval_dir, renders = pathlib.Path(args.eval_dir), pathlib.Path(args.renders)
    skus = [s for s in args.skus.split(",") if s]
    titles = {s: json.load(open(eval_dir / s / "manifest_entry.json")).get("title", s)[:40]
              if (eval_dir / s / "manifest_entry.json").exists() else s for s in skus}
    nav = "<nav>" + "".join(f'<a href="#p{s}">{html.escape(titles[s])}</a>' for s in skus) + "</nav>"
    steps = ("<ol class=\"steps\"><li>Inputs: photo + untextured mesh</li><li>Preprocessing of mesh and photo</li>"
             "<li>Geometry renders from 6 cameras (the condition for FLUX)</li>"
             "<li><span class=\"ours\">Ours:</span> glyph tokens from the photo's text</li>"
             "<li>Texture pass: FLUX + texture LoRA paints the 6 views</li><li>Delight pass: FLUX + delight LoRA removes lighting</li>"
             "<li><span class=\"ours\">Ours:</span> photo front</li><li>Bake the views onto the UV texture</li>"
             "<li>Surface sampling</li><li>LTM fills what no view saw, final bake</li></ol>")
    body = "".join(product(eval_dir, renders, s) for s in skus)
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
           f'initial-scale=1"><title>UniTEX, stage by stage</title><style>{CSS}</style></head><body><header>'
           f'<h1>UniTEX, stage by stage</h1><p class="sub">The real intermediate output of every stage for '
           f'{len(skus)} products, stock UniTEX next to ours (glyph LoRA at 1024 px + photo front). Stages marked '
           f'"Ours" are our additions. Click any image to open it, click again for full size.</p>{steps}{nav}</header>'
           f'<main>{body}</main><div id="lb"><img><div></div></div><script>{JS}</script></body></html>')
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(skus)} products)")


if __name__ == "__main__":
    main()
