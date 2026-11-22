"""
A check of every product a pipeline run made: one row per product with its numbers and plain flags, so a run of many
products can be sorted before anyone opens a model.

It reads what the steps already wrote (run_info.json of the views run and of the photo front run, photo_front.png, the
baked model's header). It renders nothing and needs no GPU. Flags:
  no_views          the views run has no finished views for this product
  no_model          there is no baked model
  front_generated   the photo did not register, so photo_front.py kept the generated front
  outline_differs   the photo's outline and the mesh's front outline overlap under IOU_MIN (a wrong shape, or a photo
                    that is not shot from the front)
  little_photo      the photo covers under PHOTO_MIN of the front
  colours_off       the front's coarse colours are over DE_MAX from the photo's (median CIELAB difference of the two
                    blurs where the photo was used)
  matte             the model's roughness is 1.0, UniTEX's export value
  no_record         with --final-run: the back panel had no product record
Notes, which are not problems: flow (aligned by optical flow), photo_layout (the photo kept its own layout),
weak_fit (the photo fits the generated front under NCC_NOTE, worth a look for doubled letters), no_text_lines (no
text line of the photo was taken whole, as with the flow).

  python unitex/check_products.py --eval-dir EVAL --run NAME [--pf-run NAME_pf] [--final-run NAME_final]
      [--skus a,b | file]
Writes EVAL/check_<model run>.csv and .md and prints the summary.
"""

import argparse
import csv
import json
import os
import pathlib
import statistics
import struct
import sys

import numpy as np
from PIL import Image

IOU_MIN = 0.9
PHOTO_MIN = 0.5
DE_MAX = 12.0
NCC_NOTE = 0.9
PROBLEMS = ("no_views", "no_model", "front_generated", "outline_differs", "little_photo", "colours_off", "matte",
            "no_record")
NOTES = ("flow", "photo_layout", "weak_fit", "no_text_lines")
COLUMNS = ("sku", "title", "flags", "notes", "register", "ncc", "iou", "photo_share", "text_lines", "colour_dE",
           "roughness", "views_s", "photo_front_s", "finish_s")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def glb_roughness(path):
    """roughnessFactor of the model's first material (1.0 when the file leaves it out), None without a model."""
    try:
        with open(path, "rb") as f:
            head = f.read(20)
            if len(head) < 20 or head[:4] != b"glTF":
                return None
            doc = json.loads(f.read(struct.unpack("<I", head[12:16])[0]).decode("utf-8"))
    except (OSError, ValueError):
        return None
    mats = doc.get("materials") or [{}]
    return float(mats[0].get("pbrMetallicRoughness", {}).get("roughnessFactor", 1.0))


def front_colour_difference(panel_path):
    """Median CIELAB difference between the coarse colours of the finished front and of the registered photo.

    photo_front.png holds four tiles: generated front, registered photo, weight, result. Both are blurred inside the
    area the photo was used (weight over 0.9), and near-black and near-white photo pixels are left out, since they are
    often shadow or glare. None when the photo was hardly used.
    """
    import cv2
    panel = np.asarray(Image.open(panel_path).convert("RGB")).astype(np.float32)
    res = panel.shape[0]
    if panel.shape[1] < 4 * res:
        return None
    photo, w, out = panel[:, res:2 * res], panel[:, 2 * res:3 * res, 0] / 255.0, panel[:, 3 * res:4 * res]
    core = (w > 0.9).astype(np.float32)
    if core.sum() < 0.05 * res * res:
        return None
    s = max(1.0, res / 32)
    den = np.maximum(cv2.GaussianBlur(core, (0, 0), s), 1e-6)[..., None]
    bp = cv2.GaussianBlur(photo * core[..., None], (0, 0), s) / den
    bo = cv2.GaussianBlur(out * core[..., None], (0, 0), s) / den
    lum = bp @ np.array([0.299, 0.587, 0.114], np.float32)
    use = (core > 0) & (lum > 12) & (lum < 243)
    if use.sum() < 100:
        return None
    lab_p = cv2.cvtColor(np.clip(bp / 255.0, 0, 1), cv2.COLOR_RGB2LAB)
    lab_o = cv2.cvtColor(np.clip(bo / 255.0, 0, 1), cv2.COLOR_RGB2LAB)
    return float(np.median(np.linalg.norm(lab_p - lab_o, axis=2)[use]))


def check_one(sku_dir, run, pf_run, final_run=None):
    """The row of one product: a dict with COLUMNS."""
    sku_dir = pathlib.Path(sku_dir)
    row = {c: "" for c in COLUMNS}
    row["sku"] = sku_dir.name
    flags, notes = [], []
    man = read_json(sku_dir / "manifest_entry.json") or {}
    row["title"] = str(man.get("title", ""))[:60]
    views = read_json(sku_dir / run / "run_info.json")
    if not views or views.get("status") != "ok" or not (sku_dir / run / "cache" / "mv_rgb.png").exists():
        flags.append("no_views")
    if views and views.get("wall_s") is not None:
        row["views_s"] = round(float(views["wall_s"]), 1)
    pf = read_json(sku_dir / pf_run / "run_info.json")
    if pf:
        reg, align = pf.get("register") or {}, pf.get("align") or {}
        row["register"] = f"{reg.get('method', '')}/{reg.get('layout', '')}"
        if reg.get("ncc_matched") is not None:
            row["ncc"] = round(float(reg["ncc_matched"]), 3)
        if align.get("iou") is not None:
            row["iou"] = round(float(align["iou"]), 3)
            if align["iou"] < IOU_MIN:
                flags.append("outline_differs")
        share = pf.get("replaced_frac_of_silhouette")
        if share is not None:
            row["photo_share"] = round(float(share), 2)
        if pf.get("skipped"):
            flags.append("front_generated")
        elif reg.get("layout") == "generated" and reg.get("ncc_matched") is not None and reg["ncc_matched"] < NCC_NOTE:
            notes.append("weak_fit")
        elif share is not None and share < PHOTO_MIN:
            flags.append("little_photo")
        if pf.get("text_json"):
            row["text_lines"] = pf.get("text_lines", 0) or 0
            if not row["text_lines"]:
                notes.append("no_text_lines")
        if reg.get("method") == "flow":
            notes.append("flow")
        if pf.get("mode") == "full":
            notes.append("photo_layout")
        if pf.get("wall_s") is not None:
            row["photo_front_s"] = round(float(pf["wall_s"]), 1)
        panel = sku_dir / pf_run / "photo_front.png"
        if panel.exists() and not pf.get("skipped"):
            de = front_colour_difference(panel)
            if de is not None:
                row["colour_dE"] = round(de, 1)
                if de > DE_MAX:
                    flags.append("colours_off")
    model_run = pf_run
    if final_run:
        fin = read_json(sku_dir / final_run / "run_info.json")
        model_run = final_run
        if fin:
            if fin.get("wall_s") is not None:
                row["finish_s"] = round(float(fin["wall_s"]), 1)
            if not fin.get("record"):
                flags.append("no_record")
    rough = glb_roughness(sku_dir / model_run / "textured_mesh.glb")
    if rough is None:
        flags.append("no_model")
    else:
        row["roughness"] = rough
        if rough >= 1.0:
            flags.append("matte")
    row["flags"] = " ".join(f for f in PROBLEMS if f in flags)
    row["notes"] = " ".join(n for n in NOTES if n in notes)
    return row


def summary(rows, model_run):
    """Markdown text: counts per flag, the flagged products, the timings."""
    flagged = [r for r in rows if r["flags"]]
    out = [f"# Check of {model_run}", "",
           f"{len(rows)} products, {len(flagged)} with a flag, {len(rows) - len(flagged)} without.", ""]
    counts = {f: sum(f in r["flags"].split() for r in rows) for f in PROBLEMS}
    if flagged:
        out += ["| flag | products |", "|---|---|"]
        out += [f"| {f} | {n} |" for f, n in counts.items() if n]
        out += ["", "| product | flags | register | NCC | IoU | photo share | colour dE |", "|---|---|---|---|---|---|---|"]
        out += [f"| {r['sku']} {r['title']} | {r['flags']} | {r['register']} | {r['ncc']} | {r['iou']} | "
                f"{r['photo_share']} | {r['colour_dE']} |" for r in flagged]
        out.append("")
    noted = [r for r in rows if r["notes"]]
    if noted:
        out += ["Notes: " + ", ".join(f"{r['sku']} ({r['notes']})" for r in noted) + ".", ""]
    for col, label in (("views_s", "views"), ("photo_front_s", "photo front"), ("finish_s", "finishing")):
        t = [float(r[col]) for r in rows if r[col] != ""]
        if t:
            out.append(f"Median seconds per product, {label}: {statistics.median(t):.1f} (max {max(t):.1f}, {len(t)} products).")
    de = [float(r["colour_dE"]) for r in rows if r["colour_dE"] != ""]
    if de:
        out.append(f"Colour difference to the photo, median over products: {statistics.median(de):.1f} (max {max(de):.1f}).")
    out += ["", f"Limits: outline IoU under {IOU_MIN}, photo share under {PHOTO_MIN}, colour difference over {DE_MAX}, "
                f"weak fit under NCC {NCC_NOTE}."]
    return "\n".join(out) + "\n"


def sku_list(spec):
    if os.path.isfile(spec):
        with open(spec) as f:
            skus = [line.split("#")[0].strip() for line in f]
    else:
        skus = [s.strip() for s in spec.split(",")]
    return [s for s in skus if s]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--run", required=True, help="the views run")
    p.add_argument("--pf-run", default=None, help="the photo front run (default: <run>_pf)")
    p.add_argument("--final-run", default=None, help="a back_panel.py run on the photo front run, when there is one")
    p.add_argument("--skus", default=None, help="file or comma list (default: every product with one of the runs)")
    args = p.parse_args(argv)
    eval_dir = pathlib.Path(args.eval_dir).resolve()
    pf_run = args.pf_run or f"{args.run}_pf"
    if args.skus:
        skus = sku_list(args.skus)
    else:
        skus = sorted(d.name for d in eval_dir.iterdir()
                      if d.is_dir() and ((d / args.run).exists() or (d / pf_run).exists()))
    rows = [check_one(eval_dir / sku, args.run, pf_run, args.final_run) for sku in skus]
    model_run = args.final_run or pf_run
    with open(eval_dir / f"check_{model_run}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    text = summary(rows, model_run)
    (eval_dir / f"check_{model_run}.md").write_text(text)
    print(text)
    print(f"Table: {eval_dir / f'check_{model_run}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
