"""
Self-contained side-by-side viewer for any number of texturing results (manual comparison, no
metrics). Every image is embedded, so the single .html file opens anywhere.

Each model is a directory of pre-rendered views, <dir>/<sku>/view_0X.png (mvgen.py --mode views
with the same cameras for every model). The real photo comes from <photos>/<sku>.png or from an
eval dir's <sku>/ref.png.

Usage:
  python unitex/compare_runs.py --out compare.html --photos /path/eval_28 \
      --model "Pre-summer pipeline=/path/views/presummer" --model "UniTEX (stock)=/path/views/unitex" \
      [--views 0,3,1] [--skus a,b] [--title ...] [--note ...]
"""

import argparse
import base64
import html
import io
import json
import pathlib

from PIL import Image

VIEW_NAMES = {"0": "front", "1": "right", "2": "back", "3": "left", "4": "top", "5": "bottom"}


def embed(path, max_side, bg=(128, 128, 128), quality=80):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        base = Image.new("RGBA", im.size, bg + (255,))
        im = Image.alpha_composite(base, im.convert("RGBA"))
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def cell(path, caption, max_side):
    if path is None or not pathlib.Path(path).exists():
        return f'<div><div class="missing">missing</div><div class="cap">{html.escape(caption)}</div></div>'
    return (f'<div><img src="{embed(path, max_side)}" data-caption="{html.escape(caption)}" loading="lazy">'
            f'<div class="cap">{html.escape(caption)}</div></div>')


def photo_path(photos, sku):
    p = pathlib.Path(photos)
    for c in (p / f"{sku}.png", p / sku / "ref.png", p / sku / "front_ref.png"):
        if c.exists():
            return c
    return None


CSS = """
:root { --bg:#f6f6f4; --card:#fff; --ink:#1d1d1f; --muted:#6e6e73; --line:#e3e3e0; --accent:#b4441c; }
@media (prefers-color-scheme: dark) { :root { --bg:#121213; --card:#1c1c1e; --ink:#f2f2f2; --muted:#9a9aa0; --line:#2c2c2e; --accent:#ff8a5c; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system,system-ui,sans-serif; }
header, main { max-width:1600px; margin:auto; padding:0 20px; } header { padding-top:24px; } h1 { font-size:24px; margin:0 0 6px; }
.sub { color:var(--muted); margin:4px 0 12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; margin:14px 0; }
.card h3 { margin:0 0 8px; font-size:15px; } .card h3 span { color:var(--muted); font-weight:400; }
.grid { display:grid; grid-template-columns: 200px 1fr; gap:14px; align-items:start; }
.rows { display:grid; gap:8px; }
.row { display:grid; grid-template-columns: 150px repeat(var(--v), minmax(0, 220px)); gap:6px; align-items:center; }
.row h4 { margin:0; }
.model h4 { margin:0 0 4px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.views { display:grid; grid-template-columns: repeat(var(--v), 1fr); gap:4px; }
img { width:100%; border-radius:5px; cursor:zoom-in; display:block; background:#808080; }
.cap { font-size:11px; color:var(--muted); } .missing { aspect-ratio:1; border:1px dashed var(--line); border-radius:5px; }
#lb { position:fixed; inset:0; background:rgba(0,0,0,.88); display:none; align-items:center; justify-content:center; flex-direction:column; z-index:9; }
#lb img { max-width:94vw; max-height:88vh; width:auto; cursor:zoom-out; } #lb div { color:#ddd; margin-top:8px; }
@media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
"""
JS = """
const lb=document.getElementById('lb'), li=lb.querySelector('img'), lc=lb.querySelector('div');
document.querySelectorAll('main img').forEach(im=>im.onclick=()=>{li.src=im.src;lc.textContent=im.dataset.caption;lb.style.display='flex';});
lb.onclick=()=>lb.style.display='none'; document.onkeydown=e=>{if(e.key==='Escape')lb.style.display='none';};
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--photos", required=True, help="dir with <sku>.png or eval dir with <sku>/ref.png")
    ap.add_argument("--model", action="append", required=True, help='"Label=views_dir" (repeatable, in column order)')
    ap.add_argument("--views", default="0,3,1", help="raw view ids to show per model")
    ap.add_argument("--skus", default=None, help="comma list (default: SKUs present in the first model)")
    ap.add_argument("--titles", default=None, help="dir with <sku>/manifest_entry.json for product titles")
    ap.add_argument("--title", default="Texturing comparison")
    ap.add_argument("--note", default="")
    ap.add_argument("--thumb", type=int, default=420)
    args = ap.parse_args(argv)

    models = []
    for m in args.model:
        label, _, d = m.partition("=")
        models.append((label.strip(), pathlib.Path(d.strip())))
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    skus = [s.strip() for s in args.skus.split(",")] if args.skus else \
        sorted(p.name for p in models[0][1].iterdir() if p.is_dir())

    cards = []
    for sku in skus:
        title = ""
        if args.titles:
            mp = pathlib.Path(args.titles) / sku / "manifest_entry.json"
            if mp.exists():
                title = json.load(open(mp)).get("title", "")
        photo = f'<div class="model"><h4>Real photo</h4>{cell(photo_path(args.photos, sku), f"{sku} photo", args.thumb)}</div>'
        rows = []
        for label, d in models:
            vs = "".join(cell(d / sku / f"view_0{v}.png", f"{label} · {VIEW_NAMES.get(v, v)}", args.thumb) for v in views)
            rows.append(f'<div class="row model"><h4>{html.escape(label)}</h4>{vs}</div>')
        cards.append(f'<div class="card"><h3>{html.escape(title[:100])} <span>{sku}</span></h3>'
                     f'<div class="grid" style="--v:{len(views)}">{photo}<div class="rows">{"".join(rows)}</div></div></div>')
    legend = " | ".join(html.escape(l) for l, _ in models)
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>{html.escape(args.title)}</title><style>{CSS}</style></head><body><header><h1>{html.escape(args.title)}</h1>'
           f'<p class="sub">{len(skus)} products. Real photo on the left, then one row per model: {legend}. Views: '
           f'{", ".join(VIEW_NAMES.get(v, v) for v in views)}, same cameras and resolution for every model, unlit. '
           f'Click an image to enlarge.</p><p class="sub">{html.escape(args.note)}</p></header>'
           f'<main>{"".join(cards)}</main><div id="lb"><img><div></div></div><script>{JS}</script></body></html>')
    out = pathlib.Path(args.out)
    out.write_text(doc)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(skus)} skus x {len(models)} models)")


if __name__ == "__main__":
    main()
