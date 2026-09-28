"""
Side-by-side viewer: pre-summer pipeline (CFI-3DGen, Gemini-painted) vs stock UniTEX.

For manual comparison only, no metrics. Both columns use the same products, cameras, resolution and
renderer, so only the textures differ:
  1. per product: real photo, then front/left/back/right unlit views of each model (mvgen.py views)
  2. scenes: synthgen.py scenes with the same seed, products, placements and cameras, rendered
     once per asset set

Usage:
  python unitex/compare_page.py --root /Users/test/CyLab/unitex_compare --eval-dir /Users/test/CyLab/unitex_eval_28
"""

import argparse
import html
import json
import os
import pathlib
import shutil

VIEWS = [("0", "front"), ("3", "left"), ("2", "back"), ("1", "right")]
PLACEMENTS = ["scatter", "cluster_mid", "cluster_tight", "stacking"]
SETS = [("presummer", "Pre-summer pipeline (CFI-3DGen)"), ("unitex", "UniTEX (stock)")]


def img_tag(path, caption, root):
    path = pathlib.Path(path)
    if not path.exists():
        return f'<div class="missing">missing</div><div class="cap">{html.escape(caption)}</div>'
    rel = os.path.relpath(path, root)
    return (f'<img src="{html.escape(rel)}" data-full="{html.escape(rel)}" data-caption="{html.escape(caption)}" loading="lazy">'
            f'<div class="cap">{html.escape(caption)}</div>')


CSS = """
:root { --bg:#f6f6f4; --card:#ffffff; --ink:#1d1d1f; --muted:#6e6e73; --line:#e3e3e0; --accent:#b4441c; }
@media (prefers-color-scheme: dark) { :root { --bg:#121213; --card:#1c1c1e; --ink:#f2f2f2; --muted:#9a9aa0; --line:#2c2c2e; --accent:#ff8a5c; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system,system-ui,sans-serif; }
header { padding:28px 24px 8px; max-width:1500px; margin:auto; } h1 { margin:0 0 6px; font-size:26px; } h2 { margin:36px 0 10px; font-size:20px; }
.sub { color:var(--muted); } main { max-width:1500px; margin:auto; padding:0 24px 60px; }
nav a { color:var(--accent); margin-right:14px; text-decoration:none; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin:14px 0; }
.card h3 { margin:0 0 10px; font-size:16px; } .card h3 span { color:var(--muted); font-weight:400; }
.prod { display:grid; grid-template-columns: 180px 1fr 1fr; gap:14px; align-items:start; }
.side h4, .scene h4 { margin:0 0 6px; font-size:13px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.views { display:grid; grid-template-columns: repeat(4, 1fr); gap:6px; }
.scene { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
img { width:100%; object-fit:contain; border-radius:6px; cursor:zoom-in; display:block; background:#808080; }
.cap { font-size:12px; color:var(--muted); margin-top:3px; }
.missing { aspect-ratio:1; border:1px dashed var(--line); border-radius:6px; display:flex; align-items:center; justify-content:center; color:var(--muted); }
#lb { position:fixed; inset:0; background:rgba(0,0,0,.88); display:none; align-items:center; justify-content:center; flex-direction:column; z-index:9; }
#lb img { max-width:92vw; max-height:86vh; width:auto; cursor:zoom-out; } #lb div { color:#ddd; margin-top:8px; }
@media (max-width: 900px) { .prod { grid-template-columns: 1fr; } .scene { grid-template-columns: 1fr; } }
"""

JS = """
const lb=document.getElementById('lb'), li=lb.querySelector('img'), lc=lb.querySelector('div');
document.querySelectorAll('main img').forEach(im=>im.onclick=()=>{li.src=im.dataset.full;lc.textContent=im.dataset.caption;lb.style.display='flex';});
lb.onclick=()=>lb.style.display='none'; document.onkeydown=e=>{if(e.key==='Escape')lb.style.display='none';};
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="unitex_compare dir with assets/, views/, scenes/")
    ap.add_argument("--eval-dir", required=True, help="eval dir holding <sku>/ref.png (the real photo)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root, ev = pathlib.Path(args.root), pathlib.Path(args.eval_dir)
    out = pathlib.Path(args.out) if args.out else root / "comparison.html"

    (root / "photos").mkdir(exist_ok=True)
    for p in (root / "assets" / "unitex").iterdir():
        src = ev / p.name / "ref.png"
        if src.exists() and not (root / "photos" / f"{p.name}.png").exists():
            shutil.copy(src, root / "photos" / f"{p.name}.png")
    skus = sorted(p.name for p in (root / "assets" / "unitex").iterdir() if p.is_dir())
    parts = []
    parts.append('<h2 id="products">Per product</h2><p class="sub">Unlit texture views, same orthographic cameras '
                 'and resolution for both models. The photo is the real input image.</p>')
    for sku in skus:
        man = json.load(open(root / "assets" / "unitex" / sku / "manifest_entry.json"))
        title = man.get("title", "")
        row = [f'<div class="card"><h3>{html.escape(title[:90])} <span>{sku} · {man.get("shape","")}</span></h3>'
               '<div class="prod">']
        row.append(f'<div class="side"><h4>Real photo</h4>{img_tag(root / "photos" / f"{sku}.png", f"{sku} photo", root)}</div>')
        for key, label in SETS:
            cells = "".join(f'<div>{img_tag(root / "views" / key / sku / f"view_0{v}.png", f"{label} · {name}", root)}</div>'
                            for v, name in VIEWS)
            row.append(f'<div class="side"><h4>{label}</h4><div class="views">{cells}</div></div>')
        row.append("</div></div>")
        parts.append("".join(row))

    parts.append('<h2 id="scenes">Scenes</h2><p class="sub">synthgen.py with the same seed, products, placements, '
                 'cameras, lighting and table texture. Only the product textures differ. Placements can shift '
                 'slightly where physics settles differently.</p>')
    for p in PLACEMENTS:
        imgs = sorted((root / "scenes" / "unitex" / p / "images").glob("*.png"))
        parts.append(f'<h3 id="{p}">{p}</h3>')
        for im in imgs:
            pair = "".join(
                f'<div><h4>{label}</h4>{img_tag(root / "scenes" / key / p / "images" / im.name, f"{label} · {p} / {im.name}", root)}</div>'
                for key, label in SETS)
            parts.append(f'<div class="card scene">{pair}</div>')

    nav = '<nav><a href="#products">Per product</a>' + "".join(f'<a href="#{p}">{p}</a>' for p in PLACEMENTS) + "</nav>"
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Pre-summer vs UniTEX</title><style>{CSS}</style></head><body>'
           f'<header><h1>Pre-summer pipeline vs UniTEX</h1><p class="sub">{len(skus)} approved-bundle products. '
           f'Left: current CFI-3DGen output. Right: stock UniTEX (FLUX.1-dev + released LoRAs) from the same mesh '
           f'and photo. Click any image to enlarge.</p>{nav}</header><main>{"".join(parts)}</main>'
           f'<div id="lb"><img><div></div></div><script>{JS}</script></body></html>')
    out.write_text(doc)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
