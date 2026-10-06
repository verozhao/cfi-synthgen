"""
Static HTML review sheet for an eval_text.py results dir (for manual review of printed text).

  python -m unitex.report --results /data/unitex_eval/results_unitex_s63

Writes <results>/index.html plus thumbnails in <results>/report_img/. All image paths are
relative, no external assets, so the results dir can be zipped or copied as is. Contents:
  - summary tables: stage x metric (mean over SKUs and pooled) and stage x text-height bucket
  - one card per SKU: photo, ref512, lit, delit, baked, baseline fronts (and the ceilings),
    each captioned with word recall / NED / phrase hit and the missed words (full list on hover),
    GT boxes toggled from the top bar (green = region hit, red = miss)
  - per SKU, the full lit strip and delit grid, and a line table with the rectified crop of every
    GT line in every stage next to what OCR read there
"""

import argparse
import html
import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitex import text_metrics as tm
from unitex.common import apply_affine

STAGE_INFO = {
    "photo":    "real catalog photo, the ground truth",
    "ref1024":  "UniTEX framing at 1024 px (input ceiling if views were 1024)",
    "vae1024":  "ref1024 after a FLUX VAE round trip",
    "ref512":   "processed_image.png, what FLUX sees",
    "vae512":   "ref512 after a FLUX VAE round trip, the 512 px ceiling",
    "lit":      "mv_rgb_w_light.png slot 0, texture LoRA",
    "delit":    "mv_rgb.png front tile, after the delight LoRA",
    "baked":    "textured_mesh.glb rendered from the front",
    "baseline": "current Gemini-painted textured.glb, front render",
}
MAIN_STAGES = ("ref512", "lit", "delit", "baked", "baseline")        # the pipeline, first row
CEILING_STAGES = ("ref1024", "vae1024", "vae512")                    # resolution / VAE ceilings, second row
CARD_STAGES = MAIN_STAGES + CEILING_STAGES
HIT, MISS, NOBOX, GT_BOX = (40, 170, 70), (220, 50, 50), (150, 150, 150), (47, 95, 179)

CSS = """
:root {
  --bg: #f6f6f3; --panel: #ffffff; --ink: #1b1d1f; --muted: #686d72; --line: #dcdcd6;
  --good: #1f8a4c; --mid: #b27a00; --bad: #c0392b; --accent: #2f5fb3; --chip: #eef0f3;
  --shadow: 0 1px 2px rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141517; --panel: #1d1f22; --ink: #e7e7e4; --muted: #9aa0a6; --line: #33363a;
    --good: #4cc27d; --mid: #e0a93b; --bad: #ff6b5e; --accent: #7aa7ff; --chip: #2a2d31;
    --shadow: none;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
header.top { position: sticky; top: 0; z-index: 5; background: var(--panel); border-bottom: 1px solid var(--line);
  padding: 10px 20px; display: flex; gap: 18px; align-items: baseline; flex-wrap: wrap; }
header.top h1 { font-size: 18px; margin: 0; }
header.top .meta { color: var(--muted); font-size: 13px; }
header.top label { margin-left: auto; font-size: 13px; cursor: pointer; user-select: none; }
main { padding: 16px 20px 60px; max-width: 1800px; margin: 0 auto; }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px;
  margin-bottom: 16px; box-shadow: var(--shadow); }
h2 { font-size: 16px; margin: 0 0 10px; }
h3 { font-size: 15px; margin: 0; }
.note { color: var(--muted); font-size: 12.5px; margin: 6px 0 0; }
.banner { border-left: 4px solid var(--mid); padding: 8px 12px; background: var(--chip); border-radius: 4px; margin-bottom: 10px; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 9px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th { font-weight: 600; color: var(--muted); font-size: 12.5px; }
td.l, th.l { text-align: left; }
tr.sub td { color: var(--muted); font-size: 12px; }
.good { color: var(--good); } .mid { color: var(--mid); } .bad { color: var(--bad); } .na { color: var(--muted); }
.scroll { overflow-x: auto; }
.card-head { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; margin-bottom: 10px; }
.chip { background: var(--chip); border-radius: 10px; padding: 1px 8px; font-size: 12px; color: var(--muted); }
.row { display: flex; flex-wrap: wrap; gap: 10px; padding-bottom: 6px; align-items: flex-start; }
.rowlabel { font-size: 12px; color: var(--muted); margin: 8px 0 4px; }
figure { margin: 0; flex: 0 0 auto; width: 228px; }
figure .imgbox { position: relative; width: 228px; height: 228px; background: var(--chip); border-radius: 6px; overflow: hidden; }
.row.small figure { width: 170px; }
.row.small figure .imgbox { width: 170px; height: 170px; }
header.top button { font: inherit; font-size: 13px; background: var(--chip); color: var(--ink); border: 1px solid var(--line);
  border-radius: 5px; padding: 2px 9px; cursor: pointer; }
figure img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
figure img.boxed { display: none; }
#boxes:checked ~ main figure img.boxed { display: block; }
#boxes:checked ~ main figure img.plain { visibility: hidden; }
figure.skipped .imgbox { display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 12.5px; padding: 12px; text-align: center; }
figcaption { font-size: 12.5px; margin-top: 4px; }
figcaption b { font-weight: 600; }
figcaption .miss { color: var(--muted); font-size: 11.5px; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--accent); font-size: 13px; }
.wide img { max-width: 100%; max-height: 360px; height: auto; border-radius: 6px; display: block; margin-top: 6px; background: var(--chip); }
table.lines td { vertical-align: top; text-align: left; white-space: normal; }
table.lines td.num { text-align: right; white-space: nowrap; }
table.lines img { display: block; max-height: 44px; max-width: 260px; background: #808080; border-radius: 3px; }
table.lines .pred { font-size: 11.5px; color: var(--muted); max-width: 260px; }
.toc a { color: var(--accent); text-decoration: none; margin-right: 10px; font-size: 13px; white-space: nowrap; }
"""


# ────────────────────────────────────────────────────────────────────────────
# Formatting
# ────────────────────────────────────────────────────────────────────────────

def cls(v):
    if v is None:
        return "na"
    return "good" if v >= 0.8 else "mid" if v >= 0.5 else "bad"


def num(v, digits=3):
    if v is None:
        return '<span class="na">-</span>'
    return f'<span class="{cls(v)}">{v:.{digits}f}</span>'


def esc(s):
    return html.escape(str(s), quote=True)


# ────────────────────────────────────────────────────────────────────────────
# Thumbnails
# ────────────────────────────────────────────────────────────────────────────

def thumb(src, dst, max_side=500):
    im = Image.open(src)
    im = im.convert("RGBA" if im.mode in ("RGBA", "LA") else "RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    im.save(dst)
    return dst


def boxed_thumb(img, quads_colors, dst, max_side=500, width=2):
    """Downscale `img` and draw each (quad in img px, rgb) on top."""
    im = img.convert("RGB")
    s = min(1.0, max_side / max(im.size))
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for q, col in quads_colors:
        pts = [(float(x) * s, float(y) * s) for x, y in q]
        d.polygon(pts, outline=col, width=width)
    im.save(dst)
    return dst


# ────────────────────────────────────────────────────────────────────────────
# Page
# ────────────────────────────────────────────────────────────────────────────

def summary_tables(summary):
    agg = summary["aggregate"]
    stages = [s for s in summary["stages"] if s in agg]
    keys = ("ned", "word_precision", "word_recall", "word_f1", "phrase_hit")
    head = ("NED", "word P", "word R", "word F1", "phrase hit")
    out = ['<section><h2>Stages</h2><div class="scroll"><table><tr><th class="l">stage</th><th class="l">meaning</th>'
           '<th>SKUs</th>' + "".join(f"<th>{h}</th>" for h in head)
           + "<th>region line NED</th><th>region hit</th><th>region word R</th><th>lines</th></tr>"]
    for s in stages:
        a = agg[s]
        if not a.get("n_skus"):
            out.append(f'<tr><td class="l">{s}</td><td class="l na">{esc(STAGE_INFO.get(s, ""))}</td>'
                       f'<td>0</td><td class="l na" colspan="9">skipped</td></tr>')
            continue
        g, p, r = a["global"]["mean"], a["global"]["pooled"], a["region"]
        out.append(f'<tr><td class="l"><b>{s}</b></td><td class="l na">{esc(STAGE_INFO.get(s, ""))}</td>'
                   f'<td>{a["n_skus"]}</td>' + "".join(f"<td>{num(g[k])}</td>" for k in keys)
                   + f'<td>{num(r["line_ned_mean_over_skus"])}</td><td>{num(r["hit_rate_pooled"])}</td>'
                   f'<td>{num(r["word_recall_pooled"])}</td><td>{r["n_lines"]}</td></tr>')
        out.append('<tr class="sub"><td></td><td class="l">pooled over SKUs</td><td></td>'
                   + "".join(f"<td>{num(p[k])}</td>" for k in keys)
                   + f'<td>{num(r["line_ned_pooled"])}</td><td colspan="3"></td></tr>')
    out.append('</table></div><p class="note">Global: OCR of the whole front image against all GT lines '
               '(NED over the reading-order concatenation, multiset word match, phrase hit = GT lines with '
               f'line NED &ge; {tm.PHRASE_THRESHOLD}). Region: each GT line is mapped from the photo into the stage '
               'with the foreground-bbox affine, cut out, upscaled and read alone. Means are over SKUs, '
               'the grey rows pool all words / lines.</p></section>')

    buckets = list(tm.BUCKET_NAMES) + ["unknown"]
    out.append('<section><h2>By text height (px in the 512 px front view)</h2><div class="scroll"><table>'
               '<tr><th class="l">stage</th>' + "".join(f'<th colspan="3">{b} px</th>' for b in buckets) + "</tr>"
               '<tr><th></th>' + "<th>lines</th><th>word R</th><th>region NED</th>" * len(buckets) + "</tr>")
    for s in stages:
        a = agg[s]
        if not a.get("n_skus"):
            continue
        row = f'<tr><td class="l"><b>{s}</b></td>'
        for b in buckets:
            gb = a["buckets"]["global"].get(b)
            rb = a["buckets"]["region"].get(b)
            if not gb:
                row += '<td class="na">-</td><td></td><td></td>'
                continue
            row += (f'<td class="na">{gb["n_lines"]}</td><td>{num(gb["word_recall"])}</td>'
                    f'<td>{num(rb["line_ned"]) if rb else num(None)}</td>')
        out.append(row + "</tr>")
    out.append('</table></div><p class="note">word R: global OCR word recall of the lines in the bucket (pooled). '
               'region NED: mean line NED of the per-line crops. "unknown": GT lines without a box.</p></section>')
    return "\n".join(out)


def figure_html(stage, st, img_plain, img_boxed, title_extra=""):
    if st is None or st.get("status") != "ok":
        reason = (st or {}).get("reason", "not run")
        return (f'<figure class="skipped"><div class="imgbox">{esc(stage)} skipped<br>{esc(reason)}</div>'
                f'<figcaption><b>{esc(stage)}</b></figcaption></figure>')
    g, r = st["global"], st["region"]
    missed = g.get("missed_words", [])
    tip = f"{stage}: {STAGE_INFO.get(stage, '')}\nmissed words ({len(missed)}): {' '.join(missed) or '-'}"
    tip += f"\nextra words: {' '.join(g.get('extra_words', [])) or '-'}\nOCR: {g.get('pred_concat', '')}"
    boxed = f'<img class="boxed" src="{esc(img_boxed)}" alt="">' if img_boxed else ""
    return (f'<figure title="{esc(tip)}"><div class="imgbox"><img class="plain" src="{esc(img_plain)}" alt="{esc(stage)}">'
            f'{boxed}</div><figcaption><b>{esc(stage)}</b> R {num(g["word_recall"], 2)} NED {num(g["ned"], 2)} '
            f'hit {num(g["phrase_hit"], 2)} | rNED {num(r["line_ned"], 2)}{title_extra}'
            f'<span class="miss">missed: {esc(" ".join(missed[:20])) or "-"}</span></figcaption></figure>')


def sku_card(res, results_dir, eval_dir, img_dir):
    sku = res["sku"]
    rel = lambda p: os.path.relpath(p, results_dir)
    d = img_dir / sku
    d.mkdir(parents=True, exist_ok=True)
    gt = res["gt_lines"]
    parts = []
    stages = res["stages"]
    ok_stages = [s for s in CARD_STAGES if s in stages]

    # photo with GT boxes (ref.png = photo + pad)
    photo = Image.open(eval_dir / res["photo"]).convert("RGB")
    px, py = res.get("photo_pad", [0, 0])
    pq = [([[x + px, y + py] for x, y in g["quad"]], GT_BOX) for g in gt if g["quad"] is not None]
    p_plain = boxed_thumb(photo, [], d / "photo.jpg")
    p_boxed = boxed_thumb(photo, pq, d / "photo_boxed.jpg")
    parts.append(f'<figure title="photo: {esc(STAGE_INFO["photo"])}"><div class="imgbox">'
                 f'<img class="plain" src="{esc(rel(p_plain))}" alt="photo"><img class="boxed" src="{esc(rel(p_boxed))}" alt="">'
                 f'</div><figcaption><b>photo</b> {len(gt)} GT lines ({esc(res["gt_source"])})'
                 f'<span class="miss">{esc(res.get("photo_box_source", ""))}</span></figcaption></figure>')

    ceil_parts = []
    for s in ok_stages:
        st = stages[s]
        dest = ceil_parts if s in CEILING_STAGES else parts
        if st.get("status") != "ok":
            dest.append(figure_html(s, st, None, None))
            continue
        img = Image.open(results_dir / st["image"]).convert("RGB")
        plain = boxed_thumb(img, [], d / f"{s}.jpg")
        quads = []
        if st.get("affine"):
            by_id = {l["id"]: l for l in st["lines"]}
            for g in gt:
                if g["quad"] is None:
                    continue
                l = by_id.get(g["id"], {})
                col = NOBOX if l.get("r_hit") is None else HIT if l["r_hit"] else MISS
                quads.append((apply_affine(np.asarray(g["quad"]), st["affine"]).tolist(), col))
        boxed = boxed_thumb(img, quads, d / f"{s}_boxed.jpg")
        dest.append(figure_html(s, st, rel(plain), rel(boxed)))

    head = (f'<div class="card-head"><h3 id="{esc(sku)}">{esc(sku)}</h3><span>{esc(res.get("title") or "")}</span>'
            f'<span class="chip">{esc(res.get("shape") or "")}</span><span class="chip">GT {esc(res["gt_source"])}</span>'
            + ('<span class="chip">dry run</span>' if res.get("dry_run") else "") + "</div>")
    body = [head, '<div class="row">' + "".join(parts) + "</div>"]
    if ceil_parts:
        body.append('<div class="rowlabel">ceilings: the same reference at 1024 px and after a FLUX VAE round trip</div>'
                    '<div class="row small">' + "".join(ceil_parts) + "</div>")
    if res.get("warnings"):
        body.append('<p class="note">' + "<br>".join(esc(w) for w in res["warnings"]) + "</p>")

    # all six views
    views = []
    for s, key in (("lit", "lit strip (f l r b t d)"), ("delit", "delit grid (f r t / b l d)")):
        st = stages.get(s)
        if st and st.get("status") == "ok" and st.get("src"):
            src = eval_dir / st["src"]
            if src.exists():
                t = thumb(src, d / f"{s}_all.jpg", 1536 if s == "lit" else 900)
                views.append(f'<div class="wide"><span class="note">{esc(key)}: {esc(st["src"])}</span>'
                             f'<img src="{esc(rel(t))}" alt="{esc(key)}"></div>')
    if views:
        body.append("<details open><summary>all six views</summary>" + "".join(views) + "</details>")

    # line table
    line_stages = [s for s in ok_stages if stages[s].get("status") == "ok"]
    if gt:
        rows = ['<tr><th class="l">#</th><th class="l">GT line</th><th>h512</th><th class="l">photo</th>'
                + "".join(f'<th class="l">{esc(s)}</th>' for s in line_stages) + "</tr>"]
        for g in gt:
            pc = results_dir / sku / "crops" / f"photo_{g['id']:03d}.png"
            h = g["height_front512"]
            h_txt = "" if h is None else f"{h:.1f}"
            p_img = f'<img src="{esc(rel(pc))}" alt="">' if pc.exists() else ""
            cells = [f'<td class="num">{g["id"]}</td><td>{esc(g["text"])}</td>'
                     f'<td class="num">{h_txt}</td><td>{p_img}</td>']
            for s in line_stages:
                l = next((x for x in stages[s]["lines"] if x["id"] == g["id"]), {})
                cp = results_dir / sku / "crops" / f"{s}_{g['id']:03d}.png"
                img = f'<img src="{esc(rel(cp))}" alt="">' if cp.exists() else ""
                cells.append(f'<td>{img}<div class="pred">{num(l.get("r_ned"), 2)} {esc(l.get("r_pred", ""))}</div></td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        body.append('<details><summary>line by line crops</summary><div class="scroll"><table class="lines">'
                    + "".join(rows) + "</table></div></details>")
    return "<section>" + "".join(body) + "</section>"


def build_report(results_dir, eval_dir=None, out=None, title=None):
    results_dir = pathlib.Path(results_dir).resolve()
    with open(results_dir / "summary.json") as f:
        summary = json.load(f)
    eval_dir = pathlib.Path(eval_dir or summary["eval_dir"]).resolve()
    img_dir = results_dir / "report_img"
    img_dir.mkdir(exist_ok=True)
    results = []
    for sku in summary["skus"]:
        p = results_dir / sku / "text_eval.json"
        if p.exists():
            with open(p) as f:
                results.append(json.load(f))

    title = title or f"UniTEX text fidelity: {summary['run_name']}"
    meta = (f"{len(results)} SKUs · OCR {esc(summary['backend'])} · GT "
            + ", ".join(f"{esc(k)} {v}" for k, v in summary.get("gt_sources", {}).items())
            + f" · {esc(summary.get('created', ''))}")
    banners = []
    if summary.get("dry_run_skus"):
        banners.append(f'<div class="banner">Dry run: {len(summary["dry_run_skus"])} SKUs have placeholder UniTEX '
                       'outputs from run_unitex.py --dry-run (lit and delit are the framed photo), so their '
                       'lit / delit numbers only test the plumbing.</div>')
    if summary.get("warnings"):
        banners.append('<div class="banner">' + "<br>".join(f"{esc(w)} (x{c})" for w, c in summary["warnings"].items())
                       + "</div>")
    toc = '<div class="toc">' + "".join(f'<a href="#{esc(r["sku"])}">{esc(r["sku"])}</a>' for r in results) + "</div>"
    page = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>",
        '<input type="checkbox" id="boxes" hidden>',
        f'<header class="top"><h1>{esc(title)}</h1><span class="meta">{meta}</span>'
        '<label for="boxes">&#9635; show GT boxes</label>'
        '<button type="button" onclick="document.querySelectorAll(\'details\').forEach(function (d) '
        '{ d.open = !this.dataset.open; }, this); this.dataset.open = this.dataset.open ? \'\' : \'1\';">'
        'expand / collapse all</button></header>',
        "<main>", "".join(banners), summary_tables(summary),
        f'<section><h2>Products</h2>{toc}<p class="note">Hover a stage image for its missed words, extra words '
        'and the full OCR text. Box colours: green = the line is read back in its crop (NED &ge; 0.8), red = not, '
        'grey = no region score.</p></section>',
    ]
    for r in results:
        print(f"  [{r['sku']}] report card")
        page.append(sku_card(r, results_dir, eval_dir, img_dir))
    page.append("</main></body></html>")
    out = pathlib.Path(out or results_dir / "index.html")
    out.write_text("\n".join(page))
    print(f"Wrote {out}")
    return out


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)      # progress shows up in redirected logs
    p = argparse.ArgumentParser(description="HTML review sheet for eval_text.py results.")
    p.add_argument("--results", required=True, help="eval_text.py --out dir (has summary.json)")
    p.add_argument("--eval-dir", default=None, help="default: the eval dir recorded in summary.json")
    p.add_argument("--out", default=None, help="default <results>/index.html")
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)
    build_report(args.results, args.eval_dir, args.out, args.title)


if __name__ == "__main__":
    main()
