"""
Where did the glyph tokens land, and are the garbled lines the misplaced ones?

For every text line of an eval SKU, compares its view-0 placement in two text.json layouts of
the same OCR lines (the bbox-fit lift, text.json, and the registered lift, anchors.py
--register-run) and joins the offset with the line's score from eval_text (region line NED of
one stage). Score the run with eval_text --align homography so a line's crop does not depend on
the bbox fit that placed its glyph tokens.

Offsets are measured at 512 px per view. One glyph token covers 16 px at 1024 px per view, so
8 px at 512: offset_tokens = offset_px512 / 8.

Outputs in --out:
  lines.csv          one row per line: offsets, height, line score
  summary.json       hit rate and mean offset per offset bucket and height bucket, Spearman rank
                     correlation between offset and line error (1 - NED)
  overlays/<sku>.png the run's front view, old placement in red, registered placement in black

Usage:
  python -m unitex.position_audit --eval-dir $R/unitex_eval_28 --run g1024_final \\
      --scores $R/unitex_eval_28/results_g1024_final_hom --stage generated --out $R/audit/positions
"""

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from unitex.common import split_grid
from unitex.ocr import quad_height

TOKEN_PX512 = 8.0                         # 16 px token at 1024 px per view
OFFSET_BUCKETS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf))
HEIGHT_BUCKETS = ((0.0, 16.0), (16.0, np.inf))


def centre(q):
    return np.asarray(q, np.float64).reshape(4, 2).mean(0)


def line_offsets(old_doc, new_doc):
    """Items of two layouts joined on src_id -> [{src_id, text, offsets in 512 px, height}]."""
    f_old, f_new = 512.0 / old_doc["res"], 512.0 / new_doc["res"]
    old = {it.get("src_id", it["id"]): it for it in old_doc["items"] if it.get("view0_quad")}
    rows = []
    for it in new_doc["items"]:
        sid = it.get("src_id", it["id"])
        if sid not in old or not it.get("view0_quad"):
            continue
        qo = np.asarray(old[sid]["view0_quad"], np.float64) * f_old
        qn = np.asarray(it["view0_quad"], np.float64) * f_new
        c = float(np.linalg.norm(centre(qn) - centre(qo)))
        h = float(quad_height(qn))
        rows.append({"src_id": sid, "text": it["text"], "height_px512": round(h, 2),
                     "centre_offset_px512": round(c, 2), "centre_offset_tokens": round(c / TOKEN_PX512, 3),
                     "max_corner_offset_px512": round(float(np.linalg.norm(qn - qo, axis=1).max()), 2),
                     "offset_over_height": round(c / h, 3) if h > 0 else None,
                     "old_quad": qo.round(2).tolist(), "new_quad": qn.round(2).tolist()})
    return rows


def rank(a):
    """Average ranks (ties share their mean rank)."""
    a = np.asarray(a, np.float64)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=np.float64)
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 4)


def _bucket(v, buckets):
    return next(f"{lo:g}-{hi:g}" for lo, hi in buckets if lo <= v < hi)


def summarize(rows):
    scored = [r for r in rows if r.get("line_ned") is not None]
    out = {"n_lines": len(rows), "n_scored": len(scored),
           "mean_offset_tokens": round(float(np.mean([r["centre_offset_tokens"] for r in rows])), 3) if rows else None,
           "spearman_offset_vs_error": spearman([r["centre_offset_tokens"] for r in scored],
                                                [1 - r["line_ned"] for r in scored])}
    for name, key, buckets in (("by_offset_tokens", "centre_offset_tokens", OFFSET_BUCKETS),
                               ("by_height_px512", "height_px512", HEIGHT_BUCKETS)):
        table = {}
        for r in scored:
            b = table.setdefault(_bucket(r[key], buckets), {"n": 0, "hits": 0, "ned": 0.0, "offset": 0.0})
            b["n"] += 1
            b["hits"] += int(bool(r["line_hit"]))
            b["ned"] += r["line_ned"]
            b["offset"] += r["centre_offset_tokens"]
        out[name] = {k: {"n": v["n"], "hit_rate": round(v["hits"] / v["n"], 3), "mean_ned": round(v["ned"] / v["n"], 3),
                         "mean_offset_tokens": round(v["offset"] / v["n"], 3)} for k, v in sorted(table.items())}
    # the same correlation inside each height bucket: small text is both harder and more offset-prone
    out["spearman_within_height"] = {
        _bucket(lo, HEIGHT_BUCKETS): spearman([r["centre_offset_tokens"] for r in scored if lo <= r["height_px512"] < hi],
                                              [1 - r["line_ned"] for r in scored if lo <= r["height_px512"] < hi])
        for lo, hi in HEIGHT_BUCKETS}
    return out


def front_view(run_dir):
    """The run's front view before photo_front when it exists, else mv_rgb.png's front tile."""
    for name in ("mv_rgb_generated.png", "mv_rgb.png"):
        p = run_dir / "cache" / name
        if p.exists():
            grid = np.asarray(Image.open(p).convert("RGB"))
            return Image.fromarray(np.ascontiguousarray(split_grid(grid, res=grid.shape[1] // 3)[0]))
    return None


def draw_overlay(img, rows, out_png, pad=24):
    f = img.width / 512.0
    im = img.copy()
    d = ImageDraw.Draw(im)
    w = max(1, round(f))
    pts = []
    for r in rows:
        for key, col in (("old_quad", (140, 29, 24)), ("new_quad", (26, 26, 26))):
            q = [(x * f, y * f) for x, y in r[key]]
            d.line(q + [q[0]], fill=col, width=w)
            pts += q
    if pts:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        im = im.crop((max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
                      min(im.width, int(max(xs)) + pad), min(im.height, int(max(ys)) + pad)))
    im.save(out_png)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", action="append", required=True, help="repeat for several eval dirs")
    p.add_argument("--run", required=True, help="run whose front view the overlays use")
    p.add_argument("--old", default="text.json")
    p.add_argument("--new", default="text_reg.json")
    p.add_argument("--scores", action="append", default=[],
                   help="eval_text --out dir per --eval-dir (same order), with <sku>/text_eval.json")
    p.add_argument("--stage", default="generated", help="eval_text stage whose region line scores are joined")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.scores and len(args.scores) != len(args.eval_dir):
        p.error("give one --scores per --eval-dir")
    out = pathlib.Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    all_rows, per_sku = [], {}
    for k, ev in enumerate(args.eval_dir):
        ev = pathlib.Path(ev)
        scores = pathlib.Path(args.scores[k]) if args.scores else None
        for sku_dir in sorted(d for d in ev.iterdir() if (d / args.old).exists() and (d / args.new).exists()):
            sku = sku_dir.name
            rows = line_offsets(json.load(open(sku_dir / args.old)), json.load(open(sku_dir / args.new)))
            te = scores / sku / "text_eval.json" if scores else None
            if te is not None and te.exists():
                st = json.load(open(te))["stages"].get(args.stage, {})
                by_id = {ln["id"]: ln for ln in st.get("lines", [])}
                for r in rows:
                    ln = by_id.get(r["src_id"])
                    if ln is not None and ln.get("r_ned") is not None:
                        r.update(line_ned=ln["r_ned"], line_hit=ln["r_hit"], line_pred=ln.get("r_pred"))
            for r in rows:
                r["sku"] = sku
            img = front_view(sku_dir / args.run)
            if img is not None and rows:
                draw_overlay(img, rows, out / "overlays" / f"{sku}.png")
            per_sku[sku] = summarize(rows)
            all_rows += rows
    cols = ["sku", "src_id", "text", "height_px512", "centre_offset_px512", "centre_offset_tokens",
            "max_corner_offset_px512", "offset_over_height", "line_ned", "line_hit", "line_pred"]
    with open(out / "lines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    summary = {"run": args.run, "stage": args.stage, "old": args.old, "new": args.new, "all": summarize(all_rows),
               "per_sku": per_sku}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    s = summary["all"]
    print(f"{s['n_lines']} lines ({s['n_scored']} scored), mean offset {s['mean_offset_tokens']} tokens, "
          f"Spearman offset vs error {s['spearman_offset_vs_error']}, within height {s['spearman_within_height']}")
    return summary


if __name__ == "__main__":
    main()
