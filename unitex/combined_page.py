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
import html
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", required=True, help="dir with <model>/<sku>/view_0X.png and photos/<sku>.png")
    ap.add_argument("--scenes", required=True, help="dir with <model>/<placement>/images/*.png")
    ap.add_argument("--titles", required=True, help="dir with <sku>/manifest_entry.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thumb", type=int, default=360, help="max side of per-product images")
    ap.add_argument("--scene-thumb", type=int, default=640, help="max side of scene images")
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
    body = (
        f'<h2 id="products">Per product</h2><p class="sub">Front, left and right views of each model.</p>'
        f'{per_product(views, args.titles, skus, ["0", "3", "1"], args.thumb)}'
        f'<h2 id="progress">Training progress</h2><p class="sub">Front views of six text-heavy products, from the '
        f'512 LoRA through the 1024 checkpoints. No photo front.</p>{progress(views, args.titles, args.thumb)}'
        f'<h2 id="scenes">Table scenes</h2><p class="sub">synthgen.py with the same seed, products, placements, '
        f'cameras, lighting and table texture. Only the product textures differ. Placements can shift slightly '
        f'where physics settles differently.</p>{scenes(sroot, args.scene_thumb)}')
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
           f'initial-scale=1"><title>UniTEX text comparison</title><style>{CSS}{EXTRA_CSS}</style></head><body>'
           f'<header><h1>UniTEX text comparison</h1><p class="sub">{html.escape(intro)}</p>{nav}</header>'
           f'<main>{body}</main><div id="lb"><img><div></div></div><script>{JS}</script></body></html>')
    out = pathlib.Path(args.out)
    out.write_text(doc)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(skus)} products x {len(MODELS)} models)")


if __name__ == "__main__":
    main()
