"""
Build the step 1 evaluation input dir from CFI-3DGen approved_bundle_v2.

For every SKU in --skus (default unitex/eval_skus.txt) writes <out>/<sku>/:
  mesh.glb        copy of generated_mesh.glb (untextured Hunyuan mesh, UniTEX input)
  ref.png         front_ref.png padded to a square on white, so UniTEX's
                  .resize((1024, 1024)) does not squash the aspect ratio
                  (front_ref_despec.png when there is no front_ref.png, flagged in ref_meta)
  ref_meta.json   original size, padding offsets, source, title, shape, photo foreground bbox
  gt_ocr.json     OCR of the full-resolution photo (quads in original photo pixels)
  gt_text.txt     editable transcript, one GT line per row, prefilled from gt_ocr.json.
                  eval_text.py uses it only once its status line reads "# status: manual"
  baseline.glb    the current pipeline's textured.glb (baseline stage)
  manifest_entry.json  copy of the bundle manifest (mvgen.py reads the shape for its yaw policy)

Photo coordinates everywhere are pixels of the ORIGINAL photo (front_ref.png), top-left
origin. ref.png pixel = photo pixel + ref_meta["pad"].

Usage:
  python -m unitex.prepare_eval --out /data/unitex_eval --backend vision
  python -m unitex.prepare_eval --out /data/unitex_eval --backend paddle --skus 012000046445,016000263192
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitex import ocr as ocr_mod
from unitex.common import bbox_of_mask

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_BUNDLE_V2 = "/Users/test/CFI-3DGen/approved_bundle_v2"
DEFAULT_BASELINE_BUNDLE = "/Users/test/cfi-synthgen/approved_bundle"
DEFAULT_SKUS = str(HERE / "eval_skus.txt")
GREY = (128, 128, 128)            # PIL "grey", UniTEX's reference background
UNITEX_REF_RES = 1024
UNITEX_VIEW_RES = 512
UNITEX_SCALE = 0.95


# ────────────────────────────────────────────────────────────────────────────
# SKU lists
# ────────────────────────────────────────────────────────────────────────────

def read_sku_list(spec):
    """Comma list or a file with one SKU per line ("#" starts a comment)."""
    if spec and os.path.exists(spec):
        skus = []
        with open(spec) as f:
            for line in f:
                s = line.split("#", 1)[0].strip()
                if s:
                    skus.append(s.split()[0])
        return skus
    return [s.strip() for s in str(spec).split(",") if s.strip()]


# ────────────────────────────────────────────────────────────────────────────
# Photo foreground and padding
# ────────────────────────────────────────────────────────────────────────────

def border_color(arr, width=4):
    """Median colour of the outer `width` px frame (catalog photos: white, a few are coloured)."""
    a = np.asarray(arr)[..., :3]
    frame = np.concatenate([a[:width].reshape(-1, 3), a[-width:].reshape(-1, 3),
                            a[:, :width].reshape(-1, 3), a[:, -width:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(frame, axis=0))


def fg_mask_border(arr, bg=None, tol=12):
    """Foreground = pixels whose max channel difference from the border colour exceeds `tol`
    (tol 12 on white is close to the min-channel < 245 threshold used in the reader scratch)."""
    a = np.asarray(arr)[..., :3].astype(np.int16)
    bg = border_color(a) if bg is None else bg
    return np.abs(a - np.array(bg, dtype=np.int16)).max(-1) > tol


def fill_holes(mask):
    """Foreground plus every enclosed background region (white print inside a white-background
    product would otherwise be cut out of a stand-in mask). scipy when present, else PIL."""
    mask = np.asarray(mask, dtype=bool)
    try:
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(mask)
    except ImportError:
        from PIL import ImageDraw
        m = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8))
        W, H = m.size
        px = m.load()
        border = [(x, 0) for x in range(W)] + [(x, H - 1) for x in range(W)] + \
                 [(0, y) for y in range(H)] + [(W - 1, y) for y in range(H)]
        for xy in border:
            if px[xy] == 255:
                ImageDraw.floodfill(m, xy, 128)
        return np.asarray(m) != 128


def robust_bbox(mask, min_frac=0.002):
    """bbox (x0, y0, x1, y1, exclusive end) over rows / columns holding at least
    max(2, min_frac * extent) foreground pixels, so isolated JPEG specks do not grow it."""
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    rows = np.nonzero(mask.sum(1) >= max(2, min_frac * W))[0]
    cols = np.nonzero(mask.sum(0) >= max(2, min_frac * H))[0]
    if len(rows) == 0 or len(cols) == 0:
        return bbox_of_mask(mask)
    return [float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)]


def pad_to_square(img, color=(255, 255, 255)):
    """Centre `img` on a square canvas. Returns (square image, (left, top) offsets)."""
    W, H = img.size
    S = max(W, H)
    left, top = (S - W) // 2, (S - H) // 2
    out = Image.new(img.mode, (S, S), color if img.mode == "RGB" else color + (255,))
    out.paste(img, (left, top))
    return out, (left, top)


# ────────────────────────────────────────────────────────────────────────────
# UniTEX reference framing (TextureTools process_image.preprocess, replicated)
# ────────────────────────────────────────────────────────────────────────────

def unitex_bbox(alpha):
    """TextureTools get_bbox: (x1, y1, x2, y2) over alpha > 0 with x2, y2 the max INDEX.
    PIL crop(x1, y1, x2, y2) then drops the last row and column, as UniTEX does."""
    a = np.asarray(alpha) > 0
    rows = np.nonzero(a.sum(-1) > 0)[0]
    cols = np.nonzero(a.sum(-2) > 0)[0]
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def unitex_frame_boxes(alpha, H, W, scale=UNITEX_SCALE):
    """(source crop box, destination box) of UniTEX's bbox-fit framing, both xyxy floats.
    The pair is an exact axis-aligned map from alpha's image onto the framed canvas."""
    x1, y1, x2, y2 = unitex_bbox(alpha)
    dy, dx = y2 - y1, x2 - x1
    s = min(H * scale / dy, W * scale / dx)
    Ht, Wt = int(dy * s), int(dx * s)
    ox, oy = int((W - Wt) / 2), int((H - Ht) / 2)
    return [float(x1), float(y1), float(x2), float(y2)], [float(ox), float(oy), float(ox + Wt), float(oy + Ht)]


def unitex_frame(rgb, alpha, H, W, scale=UNITEX_SCALE, color=GREY):
    """process_image.preprocess(image, alpha, H, W, scale, color) for RGB + L inputs.
    Returns the RGBA framed image (what UniTEX saves as rembg_image.png at 1024)."""
    src, dst = unitex_frame_boxes(alpha, H, W, scale)
    x1, y1, x2, y2 = (int(v) for v in src)
    ox, oy, ex, ey = (int(v) for v in dst)
    Wt, Ht = ex - ox, ey - oy
    rgbc = rgb.crop((x1, y1, x2, y2)).resize((Wt, Ht))
    alphac = alpha.crop((x1, y1, x2, y2)).resize((Wt, Ht))
    alphat = Image.new("L", (W, H))
    alphat.paste(alphac, (ox, oy, ex, ey))
    out = Image.new("RGBA", (W, H), color)
    out.paste(rgbc, (ox, oy, ex, ey), alphac)
    out.putalpha(alphat)
    return out


def unitex_reference(ref_png, mask1024=None):
    """UniTEX preprocess_reference_image on our padded ref.png without RMBG-2.0.

    mask1024: the RMBG alpha on the 1024 input when a real run saved it, else the hole-filled
    border-colour foreground mask of the resized input stands in for it.
    Returns (rembg_image RGBA 1024, processed_image RGB 512, src_box_1024, dst_box_1024).
    """
    im = Image.open(ref_png).convert("RGB").resize((UNITEX_REF_RES, UNITEX_REF_RES))
    if mask1024 is None:
        mask1024 = Image.fromarray((fill_holes(fg_mask_border(np.asarray(im))) * 255).astype(np.uint8))
    rembg = unitex_frame(im, mask1024, UNITEX_REF_RES, UNITEX_REF_RES)
    processed = rembg.convert("RGB").resize((UNITEX_VIEW_RES, UNITEX_VIEW_RES))
    src, dst = unitex_frame_boxes(mask1024, UNITEX_REF_RES, UNITEX_REF_RES)
    return rembg, processed, src, dst


def ref1024_box_to_photo(box, meta):
    """xyxy box in the 1024 resized ref.png -> original photo pixels."""
    S, (px, py) = meta["square_size"], meta["pad"]
    f = S / UNITEX_REF_RES
    return [box[0] * f - px, box[1] * f - py, box[2] * f - px, box[3] * f - py]


# ────────────────────────────────────────────────────────────────────────────
# Ground-truth transcript (gt_text.txt)
# ────────────────────────────────────────────────────────────────────────────

GT_ROW = re.compile(r"^\s*([0-9]+(?:\s*\+\s*[0-9]+)*|-)\s*\|\s?(.*)$")
GT_STATUS = re.compile(r"^#\s*status:\s*(\w+)", re.I)


def gt_text_template(sku, title, lines, backend, min_conf):
    head = [
        f"# gt_text.txt for {sku}: {title}",
        f"# Prefilled from {backend} OCR of the original photo (gt_ocr.json). One printed text line per row,",
        "# in reading order, spelled exactly as printed on the product.",
        '# Row format "<ids> | <text>". <ids> are gt_ocr.json line ids that give the line\'s box:',
        '#   "3+4" merges two boxes, "-" (or no "|") marks a line without a box (scored globally only).',
        "# Rows starting with # are ignored. \"#? \" rows are OCR lines below the confidence cut:",
        "#   uncomment them to keep. Delete retailer overlays not printed on the pack (size badges).",
        '# When the transcript is correct, change the status line to "# status: manual".',
        "# status: auto",
    ]
    rows = []
    for i, ln in enumerate(lines):
        text = " ".join(ln["text"].split())
        keep = ln["conf"] >= min_conf and any(c.isalnum() for c in text)
        rows.append(f"{'' if keep else '#? '}{i} | {text}")
    return "\n".join(head + rows) + "\n"


def parse_gt_text(text):
    """-> (status, [{"ids": [int] or [], "text": str}]). Status is "manual" or "auto"."""
    status = "auto"
    rows = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            m = GT_STATUS.match(line.strip())
            if m:
                status = m.group(1).lower()
            continue
        m = GT_ROW.match(line)
        if m:
            ids = [] if m.group(1) == "-" else [int(x) for x in m.group(1).replace(" ", "").split("+")]
            t = m.group(2).strip()
        else:
            ids, t = [], line.strip()
        if t:
            rows.append({"ids": ids, "text": t})
    return status, rows


def merge_quads(quads):
    """Rectangle in the first quad's reading frame covering all corner points."""
    q0 = np.asarray(quads[0], dtype=np.float64)
    if len(quads) == 1:
        return q0.tolist()
    u = q0[1] - q0[0]
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.array([-u[1], u[0]])                 # 90 degrees clockwise in y-down image space
    pts = np.concatenate([np.asarray(q, dtype=np.float64) for q in quads])
    s, t = (pts - q0[0]) @ u, (pts - q0[0]) @ v
    o = q0[0]
    corners = [o + s.min() * u + t.min() * v, o + s.max() * u + t.min() * v,
               o + s.max() * u + t.max() * v, o + s.min() * u + t.max() * v]
    return [[float(x), float(y)] for x, y in corners]


def load_gt(sku_dir, min_conf=0.5):
    """GT lines [{"id", "text", "quad" (photo px) or None}] and their source ("manual" / "ocr").

    gt_text.txt is used only when its status is "manual". Otherwise the gt_ocr.json lines with
    conf >= min_conf and at least one alphanumeric character are the GT.
    """
    sku_dir = pathlib.Path(sku_dir)
    with open(sku_dir / "gt_ocr.json") as f:
        gt_ocr = json.load(f)
    ocr_lines = gt_ocr["lines"]
    txt = sku_dir / "gt_text.txt"
    warnings = []
    if txt.exists():
        content = txt.read_text()
        status, rows = parse_gt_text(content)
        if status == "manual":
            out = []
            for k, r in enumerate(rows):
                ids = [i for i in r["ids"] if 0 <= i < len(ocr_lines)]
                if len(ids) != len(r["ids"]):
                    warnings.append(f"gt_text.txt row {k}: unknown ids {r['ids']}")
                quad = merge_quads([ocr_lines[i]["quad"] for i in ids]) if ids else None
                out.append({"id": k, "text": r["text"], "quad": quad, "ocr_ids": ids})
            return out, "manual", warnings
        meta_p = sku_dir / "ref_meta.json"
        if meta_p.exists():
            with open(meta_p) as f:
                prefill = json.load(f).get("gt_text_prefill_sha1")
            if prefill and prefill != hashlib.sha1(content.encode()).hexdigest():
                warnings.append("gt_text.txt was edited but its status is not 'manual', using OCR GT")
    out = []
    for i, ln in enumerate(ocr_lines):
        if ln["conf"] >= min_conf and any(c.isalnum() for c in ln["text"]):
            out.append({"id": len(out), "text": " ".join(ln["text"].split()), "quad": ln["quad"], "ocr_ids": [i]})
    return out, "ocr", warnings


# ────────────────────────────────────────────────────────────────────────────
# Per-SKU preparation
# ────────────────────────────────────────────────────────────────────────────

def _copy(src, dst, link=False):
    dst = pathlib.Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def prepare_sku(sku, bundle_v2, baseline_bundle, out_dir, backend, ocr_kwargs, rotations,
                gt_min_conf=0.5, force_ocr=False, link=False):
    src_dir = pathlib.Path(bundle_v2) / sku
    d = pathlib.Path(out_dir) / sku
    d.mkdir(parents=True, exist_ok=True)
    with open(src_dir / "manifest_entry.json") as f:
        manifest = json.load(f)

    photo_p = src_dir / "front_ref.png"
    despec = False
    if not photo_p.exists():
        photo_p = src_dir / "front_ref_despec.png"
        despec = True
        print(f"  [{sku}] no front_ref.png, using front_ref_despec.png")
    photo = Image.open(photo_p).convert("RGB")
    arr = np.asarray(photo)
    bg = border_color(arr)
    fg_bbox = robust_bbox(fg_mask_border(arr, bg))

    ref, (px, py) = pad_to_square(photo)
    ref.save(d / "ref.png")
    _copy(src_dir / "generated_mesh.glb", d / "mesh.glb", link)
    # mvgen.py reads the shape (for --yaw-policy cfi3dgen) from manifest_entry.json beside the GLB
    with open(d / "manifest_entry.json", "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)

    baseline_src = src_dir / "textured.glb"
    if not baseline_src.exists():
        baseline_src = pathlib.Path(baseline_bundle) / sku / "textured.glb"
    if baseline_src.exists():
        _copy(baseline_src, d / "baseline.glb", link)
    else:
        print(f"  [{sku}] WARNING: no textured.glb for the baseline stage")

    # an edited transcript's "<ids> |" point into the current gt_ocr.json, so that file must stay
    txt_p = d / "gt_text.txt"
    meta_p = d / "ref_meta.json"
    old_meta = {}
    if meta_p.exists():
        with open(meta_p) as f:
            old_meta = json.load(f)
    prefill_sha1 = old_meta.get("gt_text_prefill_sha1")
    txt_status, txt_edited = None, False
    if txt_p.exists():
        cur = txt_p.read_text()
        txt_status, _ = parse_gt_text(cur)
        txt_edited = prefill_sha1 is not None and hashlib.sha1(cur.encode()).hexdigest() != prefill_sha1
    keep_txt = txt_status == "manual" or txt_edited

    # GT OCR of the full-resolution photo (resume unless forced or the backend changed)
    gt_p = d / "gt_ocr.json"
    gt = None
    if gt_p.exists() and not force_ocr:
        with open(gt_p) as f:
            gt = json.load(f)
        if gt.get("image") != photo_p.name:
            gt = None
        elif gt.get("backend") != backend:
            if keep_txt:
                print(f"  [{sku}] keeping {gt.get('backend')} gt_ocr.json: the edited gt_text.txt refers to its "
                      f"line ids (--force-ocr to replace both)")
            else:
                gt = None
    if gt is None and keep_txt and gt_p.exists():
        print(f"  [{sku}] WARNING re-running OCR under an edited gt_text.txt: check its line ids")
    if gt is None:
        items = ocr_mod.ocr_image(photo, backend, rotations=rotations, key=sku, backend_kwargs=ocr_kwargs)
        gt = {
            "backend": backend,
            "backend_kwargs": ocr_kwargs,
            "rotations": list(rotations),
            "image": photo_p.name,
            "size": list(photo.size),
            "coords": "original photo pixels, top-left origin, quad TL,TR,BR,BL in reading frame",
            "lines": [{"id": i, **it} for i, it in enumerate(items)],
        }
        with open(gt_p, "w") as f:
            json.dump(gt, f, indent=1, ensure_ascii=False)

    # transcript template: never overwrite a manual or hand-edited transcript
    template = gt_text_template(sku, manifest.get("title", ""), gt["lines"], gt["backend"], gt_min_conf)
    write_txt = not keep_txt
    if keep_txt:
        print(f"  [{sku}] keeping edited gt_text.txt (status {txt_status})")
    if write_txt:
        txt_p.write_text(template)
        prefill_sha1 = hashlib.sha1(template.encode()).hexdigest()

    meta = {
        "sku": sku,
        "title": manifest.get("title"),
        "shape": manifest.get("shape"),
        "source": photo_p.name,
        "source_path": str(photo_p),
        "despec_fallback": despec,
        "orig_size": list(photo.size),
        "square_size": ref.size[0],
        "pad": [px, py],
        "pad_color": [255, 255, 255],
        "bg_color": list(bg),
        "fg_bbox_photo": fg_bbox,
        "fg_bbox_source": "border-colour threshold (tol 12)",
        "baseline_source": str(baseline_src) if baseline_src.exists() else None,
        "ocr_backend": gt["backend"],
        "n_gt_ocr_lines": len(gt["lines"]),
        "n_gt_lines_auto": sum(1 for l in gt["lines"] if l["conf"] >= gt_min_conf),
        "gt_text_prefill_sha1": prefill_sha1,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"  [{sku}] {manifest.get('shape')} {photo.size[0]}x{photo.size[1]} pad={px},{py} "
          f"ocr lines={len(gt['lines'])} fg_bbox={[round(v) for v in fg_bbox]}")
    return meta


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)      # progress shows up in redirected logs
    p = argparse.ArgumentParser(description="Build the UniTEX text-fidelity eval input dir.")
    p.add_argument("--out", required=True)
    p.add_argument("--bundle-v2", default=DEFAULT_BUNDLE_V2)
    p.add_argument("--baseline-bundle", default=DEFAULT_BASELINE_BUNDLE,
                   help="fallback location of <sku>/textured.glb")
    p.add_argument("--skus", default=DEFAULT_SKUS, help="file (one SKU per line) or comma list")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--rotations", default="0,90,270")
    p.add_argument("--gt-min-conf", type=float, default=0.5,
                   help="OCR lines below this are commented out in gt_text.txt")
    p.add_argument("--force-ocr", action="store_true")
    p.add_argument("--link", action="store_true", help="symlink meshes instead of copying")
    ocr_mod.add_ocr_args(p)
    args = p.parse_args(argv)

    skus = read_sku_list(args.skus)
    if args.limit:
        skus = skus[:args.limit]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = ocr_mod.backend_kwargs_from_args(args)
    rot = ocr_mod.parse_rotations(args.rotations)
    print(f"Preparing {len(skus)} SKUs into {out} (OCR backend {args.backend})")
    done, failed = [], {}
    for sku in skus:
        try:
            prepare_sku(sku, args.bundle_v2, args.baseline_bundle, out, args.backend, kw, rot,
                        args.gt_min_conf, args.force_ocr, args.link)
            done.append(sku)
        except Exception as e:
            failed[sku] = f"{type(e).__name__}: {e}"
            print(f"  [{sku}] FAILED: {failed[sku]}")
    with open(out / "skus.txt", "w") as f:
        f.write("\n".join(done) + "\n")
    with open(out / "prepare_log.json", "w") as f:
        json.dump({"bundle_v2": args.bundle_v2, "backend": args.backend, "skus": done, "failed": failed,
                   "created": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=1)
    print(f"Done: {len(done)} prepared, {len(failed)} failed. SKU list: {out / 'skus.txt'}")


if __name__ == "__main__":
    main()
