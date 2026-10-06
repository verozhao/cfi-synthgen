"""
One self-contained page with every UniTEX text comparison (manual, no metrics). All images are
embedded, so the single .html opens anywhere.

  1. per product: real photo, then front / left / right views of each model (mvgen.py --mode views,
     same cameras for every model)
  2. training progress: front views of the progress products across checkpoints
  3. table scenes: synthgen.py scenes with the same seed, products, placements, cameras, lighting and
     table texture for every model, so only the product textures differ

Usage (thanos6, R=/mnt/nvme1n1/veronica_unitex):
  python unitex/combined_page.py --views $R/compare/views --scenes $R/unitex_compare/scenes \\
      --titles $R/compare/titles --out $R/compare/UniTEX_text_comparison.html
"""

import argparse
import base64
import html
import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from compare_runs import CSS, JS, VIEW_NAMES, cell  # noqa: E402

MODELS = [
    ("presummer", "Pre-summer pipeline (CFI-3DGen)"),
    ("unitex", "UniTEX (stock)"),
    ("unitex_s63_photofront", "UniTEX (stock) + photo front"),
    ("glyph_trainAB", "Glyph LoRA, 512"),
    ("glyph_trainAB_photofront", "Glyph LoRA, 512 + photo front"),
    ("g1024_final", "Glyph LoRA, 1024"),
    ("g1024_final_photofront", "Glyph LoRA, 1024 + photo front"),
]
PROGRESS = [("glyph_trainAB", "512, 1000 steps"), ("g1024_ckpt250", "1024, step 250"),
            ("g1024_ckpt500", "1024, step 500"), ("g1024_ckpt1000", "1024, step 1000"),
            ("g1024_ckpt1500", "1024, step 1500"), ("g1024_final", "1024, step 2000")]
PROGRESS_SKUS = ["016000233164", "079200049836", "078742371627", "072360002031", "017082883896", "611269818994"]
SCENE_SETS = [("presummer", "Pre-summer pipeline"), ("unitex", "UniTEX (stock)"),
              ("g1024_final_photofront", "Glyph LoRA, 1024 + photo front")]
PLACEMENTS = ["scatter", "cluster_mid", "cluster_tight", "stacking"]

EXTRA_CSS = """
h2 { margin:34px 0 6px; font-size:20px; } nav a { color:var(--accent); margin-right:14px; text-decoration:none; }
.prog { display:grid; grid-template-columns: 150px repeat(var(--n), minmax(0, 1fr)); gap:6px; align-items:end; }
.prog h4 { margin:0 0 4px; font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.scene { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:8px; }
.scene h4 { margin:0 0 4px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
@media (max-width: 900px) { .scene, .prog { grid-template-columns: 1fr; } }
"""


def title_of(titles, sku):
    p = pathlib.Path(titles) / sku / "manifest_entry.json"
    return json.load(open(p)).get("title", "") if p.exists() else ""


def per_product(views, titles, skus, view_ids, thumb):
    cards = []
    for sku in skus:
        photo = (f'<div class="model"><h4>Real photo</h4>'
                 f'{cell(views / "photos" / f"{sku}.png", f"{sku} photo", thumb)}</div>')
        rows = []
        for key, label in MODELS:
            vs = "".join(cell(views / key / sku / f"view_0{v}.png", f"{label} · {VIEW_NAMES[v]}", thumb)
                         for v in view_ids)
            rows.append(f'<div class="row model"><h4>{html.escape(label)}</h4>{vs}</div>')
        cards.append(f'<div class="card"><h3>{html.escape(title_of(titles, sku)[:100])} <span>{sku}</span></h3>'
                     f'<div class="grid" style="--v:{len(view_ids)}">{photo}<div class="rows">{"".join(rows)}</div>'
                     f'</div></div>')
    return "".join(cards)


def progress(views, titles, thumb):
    cards = []
    head = "".join(f"<h4>{html.escape(label)}</h4>" for _, label in PROGRESS)
    for sku in PROGRESS_SKUS:
        cells = "".join(cell(views / key / sku / "view_00.png", f"{label} · front", thumb) for key, label in PROGRESS)
        cards.append(f'<div class="card"><h3>{html.escape(title_of(titles, sku)[:100])} <span>{sku}</span></h3>'
                     f'<div class="prog" style="--n:{len(PROGRESS)}"><h4>Real photo</h4>{head}'
                     f'{cell(views / "photos" / f"{sku}.png", f"{sku} photo", thumb)}{cells}</div></div>')
    return "".join(cards)


def scenes(root, thumb):
    parts = []
    for p in PLACEMENTS:
        names = sorted(im.name for im in (root / SCENE_SETS[-1][0] / p / "images").glob("*.png"))
        parts.append(f'<h3 id="{p}">{p.replace("_", " ")}</h3>')
        for n in names:
            cols = "".join(f'<div><h4>{html.escape(label)}</h4>'
                           f'{cell(root / key / p / "images" / n, f"{label} · {p} / {n}", thumb)}</div>'
                           for key, label in SCENE_SETS)
            parts.append(f'<div class="card scene">{cols}</div>')
    return "".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# High resolution (--hires-views): large front views cropped to the product, zoomable
# ────────────────────────────────────────────────────────────────────────────

HI_CSS = """
.tiles { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:8px; }
.tiles figure { margin:0; } .tiles figcaption { font-size:12px; color:var(--muted); margin-top:3px; }
.tiles img { aspect-ratio:1; object-fit:contain; }
details { margin-top:10px; } summary { cursor:pointer; color:var(--accent); font-size:13px; }
#lb { overflow:auto; text-align:center; }
#lb img { display:block; margin:3vh auto 0; max-width:96vw; max-height:90vh; width:auto; cursor:zoom-in; }
#lb.full img { max-width:none; max-height:none; cursor:zoom-out; }
@media (max-width: 900px) { .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
"""
HI_JS = """
const lb=document.getElementById('lb'), li=lb.querySelector('img'), lc=lb.querySelector('div');
document.querySelectorAll('main img').forEach(im=>im.onclick=()=>{li.src=im.src;lc.textContent=im.dataset.caption+'  (click the image for full size, anywhere else to close)';lb.classList.remove('full');lb.style.display='block';lb.scrollTo(0,0);});
li.onclick=e=>{e.stopPropagation();lb.classList.toggle('full');};
lb.onclick=()=>lb.style.display='none'; document.onkeydown=e=>{if(e.key==='Escape')lb.style.display='none';};
"""


def product_crop(im, margin=0.03):
    """Crop to the product: alpha when the image has transparency, else pixels away from the corner colour."""
    import numpy as np
    a = np.asarray(im)
    if im.mode == "RGBA" and (a[..., 3] < 250).any():
        fg = a[..., 3] > 8
    else:
        rgb = a[..., :3].astype(int)
        fg = np.abs(rgb - rgb[0, 0]).max(axis=-1) > 16
    ys, xs = np.nonzero(fg)
    if not len(xs):
        return im
    m = int(margin * max(im.size))
    return im.crop((max(0, xs.min() - m), max(0, ys.min() - m),
                    min(im.width, xs.max() + 1 + m), min(im.height, ys.max() + 1 + m)))


def embed_webp(path, max_side, crop=True, bg=(128, 128, 128), quality=80):
    from PIL import Image
    im = Image.open(path)
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    if crop:
        im = product_crop(im)
    if im.mode == "RGBA":
        im = Image.alpha_composite(Image.new("RGBA", im.size, bg + (255,)), im)
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=4)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def tile(path, caption, max_side, crop=True):
    if path is None or not pathlib.Path(path).exists():
        return f'<figure><div class="missing">missing</div><figcaption>{html.escape(caption)}</figcaption></figure>'
    return (f'<figure><img src="{embed_webp(path, max_side, crop)}" data-caption="{html.escape(caption)}" '
            f'loading="lazy"><figcaption>{html.escape(caption)}</figcaption></figure>')


def per_product_hires(views_hr, views, titles, skus, max_side, side_thumb):
    cards = []
    for sku in skus:
        tiles = [tile(views / "photos" / f"{sku}.png", "Real photo", max_side)]
        tiles += [tile(views_hr / key / sku / "view_00.png", label, max_side) for key, label in MODELS]
        rows = []
        for key, label in MODELS:
            vs = "".join(cell(views / key / sku / f"view_0{v}.png", f"{label} · {VIEW_NAMES[v]}", side_thumb)
                         for v in ("3", "1"))
            rows.append(f'<div class="row model"><h4>{html.escape(label)}</h4>{vs}</div>')
        cards.append(f'<div class="card"><h3>{html.escape(title_of(titles, sku)[:100])} <span>{sku}</span></h3>'
                     f'<div class="tiles">{"".join(tiles)}</div><details><summary>Left and right views</summary>'
                     f'<div class="rows" style="--v:2">{"".join(rows)}</div></details></div>')
    return "".join(cards)


def progress_hires(views_hr, views, titles, max_side):
    cards = []
    for sku in PROGRESS_SKUS:
        tiles = [tile(views / "photos" / f"{sku}.png", "Real photo", max_side)]
        tiles += [tile(views_hr / key / sku / "view_00.png", label, max_side) for key, label in PROGRESS]
        cards.append(f'<div class="card"><h3>{html.escape(title_of(titles, sku)[:100])} <span>{sku}</span></h3>'
                     f'<div class="tiles">{"".join(tiles)}</div></div>')
    return "".join(cards)


def scenes_hires(root, max_side):
    parts = []
    for p in PLACEMENTS:
        names = sorted(im.name for im in (root / SCENE_SETS[-1][0] / p / "images").glob("*.png"))
        parts.append(f'<h3 id="{p}">{p.replace("_", " ")}</h3>')
        for n in names:
            tiles = "".join(tile(root / key / p / "images" / n, f"{label} · {p} / {n}", max_side, crop=False)
                            for key, label in SCENE_SETS)
            parts.append(f'<div class="card"><div class="tiles" style="grid-template-columns: repeat(3, minmax(0, 1fr))">'
                         f'{tiles}</div></div>')
    return "".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", required=True, help="dir with <model>/<sku>/view_0X.png and photos/<sku>.png")
    ap.add_argument("--scenes", required=True, help="dir with <model>/<placement>/images/*.png")
    ap.add_argument("--titles", required=True, help="dir with <sku>/manifest_entry.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thumb", type=int, default=360, help="max side of per-product images")
    ap.add_argument("--scene-thumb", type=int, default=640, help="max side of scene images")
    ap.add_argument("--hires-views", default=None,
                    help="dir with <model>/<sku>/view_00.png rendered at high resolution: large front views cropped "
                         "to the product, click to zoom to full size (WebP). Without it the page is as before")
    ap.add_argument("--hires-max", type=int, default=1400, help="--hires-views: max side of front views and photos")
    ap.add_argument("--hires-scene-max", type=int, default=1280, help="--hires-views: max side of scene images")
    args = ap.parse_args(argv)
    views, sroot = pathlib.Path(args.views), pathlib.Path(args.scenes)
    skus = sorted(p.name for p in (views / MODELS[-1][0]).iterdir() if p.is_dir())
    nav = ('<nav><a href="#products">Per product</a><a href="#progress">Training progress</a>'
           '<a href="#scenes">Table scenes</a></nav>')
    intro = (
        f"{len(skus)} products from the approved bundle. Real photo on the left. Every model gets the same mesh "
        "and photo, and every row uses the same cameras, resolution and unlit renderer, so only the texture "
        "differs. Glyph LoRA: our UniTEX texture LoRA trained with glyph tokens (the photo's OCR text) on 32 "
        "lab products and 336 Google Scanned Objects, 1000 steps at 512 px per view, then 2000 more at 1024. "
        "Photo front: the product photo is aligned to the generated front view and its detail replaces that "
        "view before baking, where it lines up. It cannot fix text the photo does not show (back, sides). "
        "Click any image to enlarge.")
    scene_sub = ('<h2 id="scenes">Table scenes</h2><p class="sub">synthgen.py with the same seed, products, placements, '
                 'cameras, lighting and table texture. Only the product textures differ. Placements can shift slightly '
                 'where physics settles differently.</p>')
    if args.hires_views:
        hr = pathlib.Path(args.hires_views)
        intro = intro.replace("Click any image to enlarge.", "Front views are rendered at 1536 px and cropped to the "
                              "product. Click an image to open it, click it again for full size.")
        body = (
            f'<h2 id="products">Per product</h2><p class="sub">Front view of each model. Left and right views are '
            f'under each product.</p>{per_product_hires(hr, views, args.titles, skus, args.hires_max, args.thumb)}'
            f'<h2 id="progress">Training progress</h2><p class="sub">Front views of six text-heavy products, from the '
            f'512 LoRA through the 1024 checkpoints. No photo front.</p>'
            f'{progress_hires(hr, views, args.titles, args.hires_max)}'
            f'{scene_sub}{scenes_hires(sroot, args.hires_scene_max)}')
        style, script = CSS + EXTRA_CSS + HI_CSS, HI_JS
    else:
        body = (
            f'<h2 id="products">Per product</h2><p class="sub">Front, left and right views of each model.</p>'
            f'{per_product(views, args.titles, skus, ["0", "3", "1"], args.thumb)}'
            f'<h2 id="progress">Training progress</h2><p class="sub">Front views of six text-heavy products, from the '
            f'512 LoRA through the 1024 checkpoints. No photo front.</p>{progress(views, args.titles, args.thumb)}'
            f'{scene_sub}{scenes(sroot, args.scene_thumb)}')
        style, script = CSS + EXTRA_CSS, JS
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
           f'initial-scale=1"><title>UniTEX text comparison</title><style>{style}</style></head><body>'
           f'<header><h1>UniTEX text comparison</h1><p class="sub">{html.escape(intro)}</p>{nav}</header>'
           f'<main>{body}</main><div id="lb"><img><div></div></div><script>{script}</script></body></html>')
    out = pathlib.Path(args.out)
    out.write_text(doc)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(skus)} products x {len(MODELS)} models)")


if __name__ == "__main__":
    main()
