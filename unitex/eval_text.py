"""
Score printed-text fidelity of UniTEX stages against the real product photo (step 1).

Only the front view has ground truth: the GT lines of <sku>/gt_text.txt (status manual) or
<sku>/gt_ocr.json (OCR of the full-resolution photo). Stages, in pipeline order:

  ref1024   <sku>/vae/ref1024.png or <run>/cache/rembg_image.png RGB   1024 px input ceiling
  vae1024   <sku>/vae/vae1024.png                                     1024 px after the FLUX VAE
  ref512    <run>/cache/processed_image.png                           what FLUX sees (512)
  vae512    <sku>/vae/vae512.png                                      512 px after the FLUX VAE
  lit       <run>/cache/mv_rgb_w_light.png slot 0                     texture LoRA output
  delit     <run>/cache/mv_rgb.png front tile (common.split_grid)     after the delight LoRA
  baked     front render of <run>/textured_mesh.glb at 1024           final asset
  baseline  front render of <sku>/baseline.glb at 1024 (--yaw-policy cfi3dgen)   current pipeline

baked / baseline renders come from "mvgen.py --mode views --glb G --views 0 --res 1024 --out D"
(D/view_00.png, RGBA) run with the Blender python, or from pre-rendered files at
--baked-pattern / --baseline-pattern. They are skipped with a warning when neither exists.

Scoring per stage:
  global  OCR of the whole stage image (upscaled, rotations merged) vs all GT lines:
          NED, word P / R / F1, phrase hit, per-line NED (unitex.text_metrics)
  region  every GT line quad is mapped from photo pixels into the stage with
          common.fit_box_affine(photo foreground bbox -> stage object bbox), cut out as a
          rectified crop with margin, upscaled to a fixed text height and OCR'd alone. Line NED,
          hit and word recall per line.
  Photo foreground bbox: from the RMBG mask saved by run_unitex.py (rmbg_mask_1024.png) when
  present, else the border-colour threshold from prepare_eval.py. With the mask, ref / vae stages
  use its alpha > 0 box (exact UniTEX framing), silhouette fits its alpha >= 128 box (the product
  without RMBG's faint specks and soft shadows).
  Stage object bbox: UniTEX framing box (ref / vae stages), mv_alpha.png tile 0 (lit / delit),
  render alpha (baked / baseline). For ref / vae stages framed from the saved RMBG mask the box
  pair is UniTEX's exact map. For the others the bbox fit is refined (--align refine) by a
  small search over the destination box maximizing gradient NCC between the warped photo and
  the stage, because catalog photos are not orthographic and generated labels drift.
  Every GT line is bucketed by its height in the 512 px front view (photo quad mapped onto the
  mv_alpha front silhouette, or the ref512 framing when there is no run): <8, 8-16, 16-32, >=32.

Outputs in --out (default <eval>/results_<run-name>/):
  <sku>/text_eval.json, <sku>/stage_<stage>.png, <sku>/crops/<stage>_<line>.png
  summary.json, summary_stages.csv, summary_buckets.csv, lines.csv

Usage:
  python -m unitex.eval_text --eval-dir /data/unitex_eval --run-name unitex_s63 --backend paddle
"""

import argparse
import csv
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitex import ocr as ocr_mod
from unitex import text_metrics as tm
from unitex.common import apply_affine, bbox_of_mask, fit_box_affine, split_grid
from unitex.ocr import quad_height, quad_length
from unitex.prepare_eval import (load_gt, read_sku_list, ref1024_box_to_photo, unitex_frame_boxes,
                                 unitex_reference)

REPO = pathlib.Path(__file__).resolve().parent.parent
STAGES = ("ref1024", "vae1024", "ref512", "vae512", "lit", "delit", "baked", "baseline")
GREY = (128, 128, 128)
DEFAULT_BPY = os.environ.get("CFI_BPY_PYTHON", "/Users/test/cfi-synthgen/.venv/bin/python")
DEFAULT_BAKED = "{eval}/{sku}/{run}/renders/baked/view_00.png"
DEFAULT_BASELINE = "{eval}/{sku}/renders/baseline/view_00.png"


# ────────────────────────────────────────────────────────────────────────────
# Geometry
# ────────────────────────────────────────────────────────────────────────────

def mask_box(mask):
    return bbox_of_mask(np.asarray(mask) > 0)


def nongrey_box(img, tol=6):
    """Object bbox of an image on UniTEX's grey (128) background."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    return bbox_of_mask(np.abs(a - 128).max(-1) > tol)


def map_quad(quad, aff):
    return apply_affine(np.asarray(quad, dtype=np.float64), aff)


def _blur(a, r=1):
    """Box blur of a 2D float array (integral image), edge padded."""
    if r <= 0:
        return a
    k = 2 * r + 1
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
    c = p.cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _gradmag(gray):
    g = _blur(np.asarray(gray, dtype=np.float64), 1)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1] = g[2:] - g[:-2]
    return _blur(np.hypot(gx, gy), 1)


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def refine_affine(photo, stage_img, photo_box, obj_box, work=192, steps=(0.03, 0.015, 0.0075),
                  max_shift=0.12, min_gain=0.01):
    """Refine the photo -> stage box affine by coordinate descent on the destination box
    (centre x / y, width, height), maximizing the NCC of blurred gradient magnitudes of the
    warped photo and the stage over the 5% expanded object box at ~`work` px object size.
    Gradients rather than colours, so lighting and delighting changes matter less.
    Needed because catalog photos are not orthographic (3/4 box shots show a side panel) and
    generated textures do not place the label exactly where the bbox fit says.
    Returns (affine, info). Falls back to the bbox fit when the gain is below `min_gain`."""
    aff0 = fit_box_affine(photo_box, obj_box)
    bw, bh = obj_box[2] - obj_box[0], obj_box[3] - obj_box[1]
    if bw < 4 or bh < 4:
        return aff0, {"method": "bbox", "reason": "object box too small"}
    r = work / max(bw, bh)
    small = stage_img.convert("L").resize((max(8, round(stage_img.width * r)), max(8, round(stage_img.height * r))),
                                          Image.LANCZOS)
    rx, ry = small.width / stage_img.width, small.height / stage_img.height
    gs = _gradmag(small)
    x0, y0 = max(0, int((obj_box[0] - 0.05 * bw) * rx)), max(0, int((obj_box[1] - 0.05 * bh) * ry))
    x1 = min(small.width, int(np.ceil((obj_box[2] + 0.05 * bw) * rx)))
    y1 = min(small.height, int(np.ceil((obj_box[3] + 0.05 * bh) * ry)))
    gs_r = gs[y0:y1, x0:x1]
    k = r * (aff0[0] + aff0[1]) / 2          # photo pre-scale close to the stage scale (no aliasing)
    ps = photo.convert("L").resize((max(8, round(photo.width * k)), max(8, round(photo.height * k))), Image.LANCZOS)
    kx, ky = ps.width / photo.width, ps.height / photo.height

    def box_of(c):
        return [c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2]

    def score(c):
        sx, sy, tx, ty = fit_box_affine(photo_box, box_of(c))
        data = (kx / (rx * sx), 0.0, -kx * tx / sx, 0.0, ky / (ry * sy), -ky * ty / sy)
        w = ps.transform(small.size, Image.AFFINE, data, resample=Image.BILINEAR, fillcolor=255)
        return _ncc(gs_r, _gradmag(w)[y0:y1, x0:x1])

    c0 = np.array([(obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2, bw, bh], dtype=np.float64)
    scale = np.array([bw, bh, bw, bh])
    cur, s0 = c0.copy(), score(c0)
    best, n_eval = s0, 1
    for step in steps:
        for _ in range(20):
            improved = False
            for i in range(4):
                for sgn in (-1.0, 1.0):
                    c2 = cur.copy()
                    c2[i] += sgn * step * scale[i]
                    if abs(c2[i] - c0[i]) > max_shift * scale[i]:
                        continue
                    sc = score(c2)
                    n_eval += 1
                    if sc > best + 1e-4:
                        best, cur, improved = sc, c2, True
            if not improved:
                break
    info = {"ncc_bbox": round(s0, 4), "ncc": round(best, 4), "n_eval": n_eval,
            "box": [round(v, 2) for v in box_of(cur)], "delta_frac": [round(float(v), 4) for v in (cur - c0) / scale]}
    if best - s0 < min_gain:
        return aff0, {"method": "bbox", **info, "reason": f"gain {best - s0:.4f} < {min_gain}"}
    return fit_box_affine(photo_box, box_of(cur)), {"method": "refined", **info}


def rectified_crop(img, quad, target_h=48.0, margin_h=0.35, margin_w=0.5, fill=GREY,
                   min_scale=0.5, max_scale=8.0, max_w=2048):
    """Cut the quad (plus margins in its own reading frame) out of `img` as an upright,
    rescaled rectangle whose text height is about `target_h` px."""
    q = np.asarray(quad, dtype=np.float64)
    h, L = quad_height(q), quad_length(q)
    if h < 0.5 or L < 0.5:
        return None
    u = (q[1] - q[0]) + (q[2] - q[3])
    u /= max(np.linalg.norm(u), 1e-9)
    v = (q[3] - q[0]) + (q[2] - q[1])
    v /= max(np.linalg.norm(v), 1e-9)
    mh, mw = max(2.0, margin_h * h), max(2.0, margin_w * h)
    tl = q[0] - u * mw - v * mh
    tr = q[1] + u * mw - v * mh
    br = q[2] + u * mw + v * mh
    bl = q[3] - u * mw + v * mh
    f = float(np.clip(target_h / h, min_scale, max_scale))
    w_out, h_out = (L + 2 * mw) * f, (h + 2 * mh) * f
    if w_out > max_w:
        f *= max_w / w_out
        w_out, h_out = max_w, h_out * max_w / w_out
    size = (max(8, int(round(w_out))), max(8, int(round(h_out))))
    data = tuple(float(c) for p in (tl, bl, br, tr) for c in p)     # PIL QUAD: UL, LL, LR, UR
    return img.transform(size, Image.QUAD, data, resample=Image.BICUBIC, fillcolor=fill)


# ────────────────────────────────────────────────────────────────────────────
# Stage construction
# ────────────────────────────────────────────────────────────────────────────

class Stage:
    def __init__(self, name, image, obj_box, photo_box, src, front512=1.0, exact=False):
        self.name = name
        self.image = image            # RGB PIL, what gets OCR'd
        self.obj_box = obj_box        # object bbox in stage pixels
        self.photo_box = photo_box    # matching photo foreground bbox (photo pixels)
        self.src = str(src)
        self.front512 = front512      # stage px -> 512 px front-view px
        self.exact = exact            # the box pair is UniTEX's own framing map, no refinement
        self.aff = None
        self.align = {"method": "exact" if exact else "bbox"}

    def affine(self):
        if self.aff is not None:
            return self.aff
        if self.obj_box is None or self.photo_box is None:
            return None
        return fit_box_affine(self.photo_box, self.obj_box)


def photo_framing(sku_dir, run_dir, meta):
    """Photo foreground bbox and the matching UniTEX reference boxes (1024 / 512)."""
    mask_p = run_dir / "rmbg_mask_1024.png"
    if mask_p.exists():
        mask = Image.open(mask_p).convert("L")
        src, dst = unitex_frame_boxes(mask, 1024, 1024)
        # photo_box (alpha > 0) is UniTEX's framing box and also spans RMBG's faint specks and soft
        # shadows. Fits onto a hard mesh silhouette use the product itself (alpha >= 128).
        hard = bbox_of_mask(np.asarray(mask) >= 128)
        return {"photo_box": ref1024_box_to_photo(src, meta), "ref_box_1024": dst,
                "ref_box_512": [v / 2 for v in dst], "source": "rmbg_mask_1024",
                "fg_box": ref1024_box_to_photo(hard or src, meta)}
    out = {"photo_box": meta["fg_bbox_photo"], "fg_box": meta["fg_bbox_photo"],
           "source": "border threshold (prepare_eval)"}
    rembg = run_dir / "cache" / "rembg_image.png"
    proc = run_dir / "cache" / "processed_image.png"
    if rembg.exists():
        b = mask_box(np.asarray(Image.open(rembg).getchannel("A")))
        out["ref_box_1024"] = b
        out["ref_box_512"] = [v / 2 for v in b] if b else None
    elif proc.exists():
        b = nongrey_box(Image.open(proc))
        out["ref_box_512"] = b
        out["ref_box_1024"] = [v * 2 for v in b] if b else None
    return out


def render_front(glb, png, res, bpy, mvgen, yaw_policy=None, mode="auto", device=None):
    """Front render via mvgen.py. Returns (path or None, reason when None).
    A failed GPU render is retried once with --device cpu."""
    png = pathlib.Path(png)
    if png.exists() and mode != "force":
        return png, None
    if mode == "never":
        return None, f"no pre-rendered {png} (--render never)"
    if not pathlib.Path(glb).exists():
        return None, f"missing {glb}"
    if not pathlib.Path(mvgen).exists():
        return None, f"mvgen.py not found at {mvgen}"
    if not pathlib.Path(bpy).exists():
        return None, f"Blender python not found at {bpy} (set --blender-python or CFI_BPY_PYTHON)"
    png.parent.mkdir(parents=True, exist_ok=True)
    for stale in (png, png.parent / "view_00.png"):
        if stale.exists():
            stale.unlink()
    cmd = [str(bpy), str(mvgen), "--mode", "views", "--glb", str(glb), "--views", "0",
           "--res", str(res), "--out", str(png.parent)]
    if yaw_policy:
        cmd += ["--yaw-policy", yaw_policy]
    r = None
    for dev in ([device] if device == "cpu" else [device, "cpu"]):
        r = subprocess.run(cmd + (["--device", dev] if dev else []), capture_output=True, text=True)
        if r.returncode == 0 and (png.parent / "view_00.png").exists():
            break
        print(f"    mvgen.py failed with --device {dev or 'default'} ({r.returncode})"
              + (", retrying on cpu" if dev != "cpu" else ""))
    if r.returncode != 0 or not (png.parent / "view_00.png").exists():
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return None, f"mvgen.py failed ({r.returncode}): {' | '.join(tail)}"
    out = png.parent / "view_00.png"
    if out != png:
        out.replace(png)
    return png, None


def _rgba_on_grey(p):
    im = Image.open(p)
    if im.mode in ("RGBA", "LA"):
        rgba = im.convert("RGBA")
        base = Image.new("RGBA", rgba.size, GREY + (255,))
        alpha = np.asarray(rgba.getchannel("A"))
        return Image.alpha_composite(base, rgba).convert("RGB"), alpha
    rgb = im.convert("RGB")
    return rgb, None


def build_stages(sku, eval_dir, run_name, names, args, warn):
    """Construct the requested stages that have inputs. Missing ones are reported via warn()."""
    sku_dir = eval_dir / sku
    run_dir = sku_dir / run_name
    cache = run_dir / "cache"
    with open(sku_dir / "ref_meta.json") as f:
        meta = json.load(f)
    fr = photo_framing(sku_dir, run_dir, meta)
    vae_meta = {}
    if (sku_dir / "vae" / "vae_meta.json").exists():
        with open(sku_dir / "vae" / "vae_meta.json") as f:
            vae_meta = json.load(f)
    # a local-source VAE ceiling carries its own framing boxes
    fr_exact = fr["source"] == "rmbg_mask_1024"
    vae_exact = bool(vae_meta.get("photo_box")) or fr_exact
    vae_photo = vae_meta.get("photo_box") or fr["photo_box"]
    vae_box = {1024: vae_meta.get("dst_box_1024") or fr.get("ref_box_1024"),
               512: vae_meta.get("dst_box_512") or fr.get("ref_box_512")}
    if vae_meta and vae_meta.get("run_name") not in (None, run_name) and str(vae_meta.get("source", "")).startswith("run"):
        warn(f"vae ceiling images were made from run {vae_meta.get('run_name')!r}")

    front_alpha = None
    if (cache / "mv_alpha.png").exists():
        grid_a = np.asarray(Image.open(cache / "mv_alpha.png").convert("L"))
        front_alpha = split_grid(grid_a, res=grid_a.shape[0] // 2)[0]
    front_box = mask_box(front_alpha > 127) if front_alpha is not None else None

    stages = {}
    for name in names:
        if name == "ref1024":
            p = sku_dir / "vae" / "ref1024.png"
            if p.exists():
                stages[name] = Stage(name, Image.open(p).convert("RGB"), vae_box[1024], vae_photo, p, 0.5, vae_exact)
            elif (cache / "rembg_image.png").exists():
                p = cache / "rembg_image.png"
                stages[name] = Stage(name, Image.open(p).convert("RGB"), fr.get("ref_box_1024"), fr["photo_box"], p, 0.5,
                                     fr_exact)
            else:
                warn(f"{name}: skipped (no vae/ref1024.png and no UniTEX run)")
        elif name in ("vae512", "vae1024"):
            r = int(name[3:])
            p = sku_dir / "vae" / f"{name}.png"
            if p.exists():
                stages[name] = Stage(name, Image.open(p).convert("RGB"), vae_box[r], vae_photo, p, 512 / r, vae_exact)
            else:
                warn(f"{name}: skipped (run vae_ceiling.py first)")
        elif name == "ref512":
            p = cache / "processed_image.png"
            if p.exists():
                stages[name] = Stage(name, Image.open(p).convert("RGB"), fr.get("ref_box_512"), fr["photo_box"], p,
                                     1.0, fr_exact)
            else:
                warn(f"{name}: skipped (no {p})")
        elif name == "lit":
            p = cache / "mv_rgb_w_light.png"
            if p.exists():
                strip = Image.open(p).convert("RGB")
                res = strip.height
                img = strip.crop((0, 0, res, res))        # strip slot 0 is the front view
                box = [v * res / 512 for v in front_box] if front_box is not None else nongrey_box(img)
                stages[name] = Stage(name, img, box, fr["fg_box"], p, 512 / res)
            else:
                warn(f"{name}: skipped (no {p})")
        elif name == "delit":
            p = cache / "mv_rgb.png" if (cache / "mv_rgb.png").exists() else run_dir / "mv_rgb.png"
            if p.exists():
                grid = np.asarray(Image.open(p).convert("RGB"))
                res = grid.shape[1] // 3
                img = Image.fromarray(np.ascontiguousarray(split_grid(grid, res=res)[0]))
                box = [v * res / 512 for v in front_box] if front_box is not None else nongrey_box(img)
                stages[name] = Stage(name, img, box, fr["fg_box"], p, 512 / res)
            else:
                warn(f"{name}: skipped (no mv_rgb.png)")
        elif name in ("baked", "baseline"):
            if name == "baked":
                glb = run_dir / "textured_mesh.glb"
                png = args.baked_pattern.format(eval=eval_dir, sku=sku, run=run_name)
                yaw = None
                if (run_dir / "dry_run.json").exists():
                    warn("baked: skipped (dry-run placeholder textured_mesh.glb is the untextured input mesh)")
                    continue
            else:
                glb = sku_dir / "baseline.glb"
                png = args.baseline_pattern.format(eval=eval_dir, sku=sku, run=run_name)
                yaw = "cfi3dgen"
            path, why = render_front(glb, png, args.render_res, args.blender_python, args.mvgen, yaw,
                                     args.render, args.render_device)
            if path is None:
                warn(f"{name}: skipped ({why})")
                continue
            img, alpha = _rgba_on_grey(path)
            box = mask_box(alpha > 127) if alpha is not None else nongrey_box(img)
            stages[name] = Stage(name, img, box, fr["fg_box"], path, 512 / img.width)
    return stages, fr, front_box, meta


# ────────────────────────────────────────────────────────────────────────────
# Scoring
# ────────────────────────────────────────────────────────────────────────────

def _bucket_rows(line_rows, key_ned, key_hit, key_nw, key_nm):
    out = {}
    for b in tm.BUCKET_NAMES + ("unknown",):
        rows = [r for r in line_rows if (r["bucket"] or "unknown") == b and r.get(key_ned) is not None]
        if not rows:
            continue
        nw = sum(r[key_nw] for r in rows)
        out[b] = {
            "n_lines": len(rows),
            "line_ned": round(sum(r[key_ned] for r in rows) / len(rows), 4),
            "hit_rate": round(sum(bool(r[key_hit]) for r in rows) / len(rows), 4),
            "n_words": nw,
            "n_words_matched": sum(r[key_nm] for r in rows),
            "word_recall": round(sum(r[key_nm] for r in rows) / nw, 4) if nw else None,
        }
    return out


def evaluate_sku(sku, eval_dir, out_dir, args, ocr_kwargs):
    t0 = time.time()
    warnings = []

    def warn(msg):
        warnings.append(msg)
        print(f"  [{sku}] WARNING {msg}")

    sku_dir = eval_dir / sku
    gt, gt_source, gt_warn = load_gt(sku_dir, args.gt_min_conf)
    for w in gt_warn:
        warn(w)
    stages, fr, front_box, meta = build_stages(sku, eval_dir, args.run_name, args.stages, args, warn)
    res_dir = out_dir / sku
    (res_dir / "crops").mkdir(parents=True, exist_ok=True)
    for old in list((res_dir / "crops").glob("*.png")) + list(res_dir.glob("stage_*.png")):
        old.unlink()                                  # stale files would show up in the report

    # GT heights in the 512 px front view
    if front_box is not None:
        h_aff, h_ref = fit_box_affine(fr["fg_box"], front_box), "mv_alpha front silhouette"
    elif fr.get("ref_box_512"):
        h_aff, h_ref = fit_box_affine(fr["photo_box"], fr["ref_box_512"]), "ref512 framing"
    else:
        # no UniTEX run yet: replicate its reference framing locally so heights stay comparable
        _, _, src, dst = unitex_reference(sku_dir / "ref.png")
        if src and dst:
            h_aff = fit_box_affine(ref1024_box_to_photo(src, meta), [v / 2 for v in dst])
            h_ref = "local ref512 framing"
        else:
            h_aff, h_ref = None, None
    gt_rows = []
    for g in gt:
        h512 = quad_height(map_quad(g["quad"], h_aff)) if (g["quad"] is not None and h_aff) else None
        gt_rows.append({"id": g["id"], "text": g["text"], "quad": g["quad"],
                        "height_photo": round(quad_height(g["quad"]), 2) if g["quad"] is not None else None,
                        "height_front512": round(h512, 2) if h512 is not None else None,
                        "bucket": tm.height_bucket(h512)})
    gt_texts = [g["text"] for g in gt]

    # photo -> stage alignment: exact for UniTEX's own framing, refined elsewhere
    ref = Image.open(sku_dir / "ref.png").convert("RGB")
    px, py = meta["pad"]
    W0, H0 = meta["orig_size"]
    photo = ref.crop((px, py, px + W0, py + H0))
    for st in stages.values():
        if st.exact or st.obj_box is None or st.photo_box is None or args.align == "bbox":
            continue
        st.aff, st.align = refine_affine(photo, st.image, st.photo_box, st.obj_box)

    # OCR: one batch for all global images, one for all region crops
    names = list(stages)
    for n in names:
        stages[n].image.save(res_dir / f"stage_{n}.png")
    cache_dir = out_dir / "ocr_cache"
    glob_res = ocr_mod.ocr_images([stages[n].image for n in names], args.backend,
                                  rotations=args.global_rotations, upscale_to=args.upscale_to,
                                  min_conf=args.min_conf, cache_dir=cache_dir, backend_kwargs=ocr_kwargs)
    crops, owners = [], []
    for n in names:
        st = stages[n]
        aff = st.affine()
        if aff is None:
            warn(f"{n}: no object bbox, region scoring skipped")
            continue
        for g in gt_rows:
            if g["quad"] is None:
                continue
            c = rectified_crop(st.image, map_quad(g["quad"], aff), target_h=args.crop_height)
            if c is None:
                continue
            crops.append(c)
            owners.append((n, g["id"]))
            if args.save_crops:
                c.save(res_dir / "crops" / f"{n}_{g['id']:03d}.png")
    if args.save_crops:
        for g in gt_rows:
            if g["quad"] is not None:
                c = rectified_crop(photo, g["quad"], target_h=args.crop_height, fill=(255, 255, 255))
                if c is not None:
                    c.save(res_dir / "crops" / f"photo_{g['id']:03d}.png")
    crop_res = ocr_mod.ocr_images(crops, args.backend, rotations=args.region_rotations, upscale_to=None,
                                  min_conf=args.min_conf, cache_dir=cache_dir,
                                  backend_kwargs=ocr_kwargs) if crops else []
    region = {}
    for (n, gid), items in zip(owners, crop_res):
        region.setdefault(n, {})[gid] = items

    out_stages = {}
    for n, items in zip(names, glob_res):
        st = stages[n]
        preds = [it["text"] for it in items]
        m = tm.compute_metrics(gt_texts, preds, fold_accents=args.fold_accents)
        line_rows = []
        for g, lm in zip(gt_rows, m.pop("lines")):
            row = {"id": g["id"], "bucket": g["bucket"], "height_front512": g["height_front512"],
                   "g_ned": lm["ned"], "g_hit": lm["hit"], "g_nw": lm["n_words"], "g_nm": lm["n_words_matched"]}
            ritems = region.get(n, {}).get(g["id"])
            if ritems is not None:
                rp = [it["text"] for it in ritems]
                score, _ = tm.line_score(g["text"], rp, args.fold_accents)
                gw = Counter(tm.tokenize(g["text"], args.fold_accents))
                pw = Counter(t for x in rp for t in tm.tokenize(x, args.fold_accents))
                row.update(r_ned=None if score is None else round(score, 4),
                           r_hit=None if score is None else bool(score >= tm.PHRASE_THRESHOLD),
                           r_pred=" / ".join(rp), r_nw=sum(gw.values()),
                           r_nm=sum(min(c, pw[w]) for w, c in gw.items()))
            line_rows.append(row)
        scored = [r for r in line_rows if r.get("r_ned") is not None]
        rw = sum(r["r_nw"] for r in scored)
        out_stages[n] = {
            "status": "ok",
            "image": os.path.relpath(res_dir / f"stage_{n}.png", out_dir),
            "src": os.path.relpath(st.src, eval_dir) if os.path.isabs(st.src) else st.src,
            "size": list(st.image.size),
            "front512_scale": st.front512,
            "obj_box": [round(v, 2) for v in st.obj_box] if st.obj_box else None,
            "photo_box": [round(v, 2) for v in st.photo_box] if st.photo_box else None,
            "affine": [round(v, 6) for v in st.affine()] if st.affine() else None,
            "align": st.align,
            "global": {**m, "pred_lines": [{"text": it["text"], "conf": round(it["conf"], 3), "quad": it["quad"]}
                                           for it in items]},
            "region": {
                "n_lines": len(scored),
                "line_ned": round(sum(r["r_ned"] for r in scored) / len(scored), 4) if scored else None,
                "hit_rate": round(sum(r["r_hit"] for r in scored) / len(scored), 4) if scored else None,
                "n_words": rw,
                "n_words_matched": sum(r["r_nm"] for r in scored),
                "word_recall": round(sum(r["r_nm"] for r in scored) / rw, 4) if rw else None,
            },
            "lines": line_rows,
            "buckets": {"global": _bucket_rows(line_rows, "g_ned", "g_hit", "g_nw", "g_nm"),
                        "region": _bucket_rows(scored, "r_ned", "r_hit", "r_nw", "r_nm")},
        }
    for n in args.stages:
        if n not in out_stages:
            reason = next((w.split("(", 1)[1].rstrip(")") for w in warnings if w.startswith(f"{n}:")), "not available")
            out_stages[n] = {"status": "skipped", "reason": reason}

    result = {
        "sku": sku, "title": meta.get("title"), "shape": meta.get("shape"), "run_name": args.run_name,
        "backend": args.backend, "backend_kwargs": ocr_kwargs,
        "global_rotations": list(args.global_rotations), "region_rotations": list(args.region_rotations),
        "gt_source": gt_source, "n_gt_lines": len(gt_rows), "gt_lines": gt_rows,
        "photo_box": fr["photo_box"], "photo_fg_box": fr["fg_box"], "photo_box_source": fr["source"],
        "height_reference": h_ref,
        "photo": os.path.relpath(sku_dir / "ref.png", eval_dir), "photo_pad": meta["pad"],
        "dry_run": (eval_dir / sku / args.run_name / "dry_run.json").exists(),
        "stages": out_stages, "warnings": warnings,
        "time_s": round(time.time() - t0, 1),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(res_dir / "text_eval.json", "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    summary = "  ".join(f"{n}: R={out_stages[n]['global']['word_recall']} rNED={out_stages[n]['region']['line_ned']}"
                        for n in names)
    print(f"  [{sku}] {len(gt_rows)} GT lines ({gt_source}) {result['time_s']} s  {summary}")
    return result


# ────────────────────────────────────────────────────────────────────────────
# Aggregation
# ────────────────────────────────────────────────────────────────────────────

def aggregate_results(results, stage_names):
    agg = {}
    for n in stage_names:
        ok = [r for r in results if r["stages"].get(n, {}).get("status") == "ok"]
        if not ok:
            agg[n] = {"n_skus": 0}
            continue
        g = tm.aggregate([r["stages"][n]["global"] for r in ok])
        lines = [l for r in ok for l in r["stages"][n]["lines"]]
        rs = [l for l in lines if l.get("r_ned") is not None]
        rw = sum(l["r_nw"] for l in rs)
        per_sku_r = [r["stages"][n]["region"]["line_ned"] for r in ok if r["stages"][n]["region"]["line_ned"] is not None]
        agg[n] = {
            "n_skus": len(ok),
            "skus": [r["sku"] for r in ok],
            "global": {"mean": g["mean"], "pooled": g["pooled"]},
            "region": {
                "n_lines": len(rs),
                "line_ned_mean_over_skus": round(sum(per_sku_r) / len(per_sku_r), 4) if per_sku_r else None,
                "line_ned_pooled": round(sum(l["r_ned"] for l in rs) / len(rs), 4) if rs else None,
                "hit_rate_pooled": round(sum(bool(l["r_hit"]) for l in rs) / len(rs), 4) if rs else None,
                "word_recall_pooled": round(sum(l["r_nm"] for l in rs) / rw, 4) if rw else None,
            },
            "buckets": {"global": _bucket_rows(lines, "g_ned", "g_hit", "g_nw", "g_nm"),
                        "region": _bucket_rows(rs, "r_ned", "r_hit", "r_nw", "r_nm")},
        }
    return agg


def write_tables(out_dir, results, agg, stage_names):
    with open(out_dir / "summary_stages.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = ("ned", "word_precision", "word_recall", "word_f1", "phrase_hit")
        w.writerow(["stage", "n_skus"] + [f"mean_{k}" for k in keys] + [f"pooled_{k}" for k in keys]
                   + ["region_n_lines", "region_line_ned_mean_over_skus", "region_line_ned_pooled",
                      "region_hit_rate_pooled", "region_word_recall_pooled"])
        for n in stage_names:
            a = agg[n]
            if not a["n_skus"]:
                w.writerow([n, 0])
                continue
            w.writerow([n, a["n_skus"]] + [a["global"]["mean"][k] for k in keys]
                       + [a["global"]["pooled"][k] for k in keys]
                       + [a["region"][k] for k in ("n_lines", "line_ned_mean_over_skus", "line_ned_pooled",
                                                   "hit_rate_pooled", "word_recall_pooled")])
    with open(out_dir / "summary_buckets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "bucket", "n_lines", "global_word_recall", "global_line_ned", "global_hit_rate",
                    "region_n_lines", "region_word_recall", "region_line_ned", "region_hit_rate"])
        for n in stage_names:
            a = agg[n]
            if not a["n_skus"]:
                continue
            for b in tm.BUCKET_NAMES + ("unknown",):
                gb = a["buckets"]["global"].get(b)
                rb = a["buckets"]["region"].get(b, {})
                if not gb:
                    continue
                w.writerow([n, b, gb["n_lines"], gb["word_recall"], gb["line_ned"], gb["hit_rate"],
                            rb.get("n_lines"), rb.get("word_recall"), rb.get("line_ned"), rb.get("hit_rate")])
    with open(out_dir / "lines.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sku", "stage", "line_id", "gt_text", "height_front512", "bucket",
                    "global_ned", "global_hit", "region_ned", "region_hit", "region_pred"])
        for r in results:
            gt = {g["id"]: g for g in r["gt_lines"]}
            for n in stage_names:
                st = r["stages"].get(n, {})
                if st.get("status") != "ok":
                    continue
                for l in st["lines"]:
                    w.writerow([r["sku"], n, l["id"], gt[l["id"]]["text"], l["height_front512"], l["bucket"],
                                l["g_ned"], l["g_hit"], l.get("r_ned"), l.get("r_hit"), l.get("r_pred")])


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)      # progress shows up in redirected logs
    p = argparse.ArgumentParser(description="Score UniTEX stage text fidelity against the product photo.")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--run-name", default="unitex")
    p.add_argument("--skus", default=None, help="file or comma list (default <eval>/skus.txt)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--out", default=None, help="default <eval>/results_<run-name>")
    ocr_mod.add_ocr_args(p)
    p.add_argument("--global-rotations", default="0,90,270")
    p.add_argument("--region-rotations", default="0", help="crops are already upright")
    p.add_argument("--upscale-to", type=int, default=2048, help="global OCR: upscale longest side to this")
    p.add_argument("--crop-height", type=float, default=48.0, help="region OCR: text height in the crop, px")
    p.add_argument("--min-conf", type=float, default=0.0, help="drop predicted lines below this")
    p.add_argument("--gt-min-conf", type=float, default=0.5, help="OCR GT lines kept at or above this")
    p.add_argument("--fold-accents", action="store_true", help="compare without diacritics")
    p.add_argument("--no-crops", dest="save_crops", action="store_false")
    p.add_argument("--align", choices=("refine", "bbox"), default="refine",
                   help="photo -> stage map for lit / delit / baked / baseline: bbox fit, or refined on image gradients")
    p.add_argument("--render", choices=("auto", "never", "force"), default="auto")
    p.add_argument("--render-res", type=int, default=1024)
    p.add_argument("--blender-python", default=DEFAULT_BPY)
    p.add_argument("--mvgen", default=str(REPO / "mvgen.py"))
    p.add_argument("--render-device", default=None, help="mvgen.py --device (default: mvgen's auto)")
    p.add_argument("--baked-pattern", default=DEFAULT_BAKED, help="pre-rendered baked front, {eval} {sku} {run}")
    p.add_argument("--baseline-pattern", default=DEFAULT_BASELINE, help="pre-rendered baseline front")
    args = p.parse_args(argv)
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in args.stages if s not in STAGES]
    if bad:
        p.error(f"unknown stages {bad}, choose from {STAGES}")
    args.global_rotations = ocr_mod.parse_rotations(args.global_rotations)
    args.region_rotations = ocr_mod.parse_rotations(args.region_rotations)

    eval_dir = pathlib.Path(args.eval_dir).resolve()
    out_dir = pathlib.Path(args.out or eval_dir / f"results_{args.run_name}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    skus = read_sku_list(args.skus or str(eval_dir / "skus.txt"))
    if args.limit:
        skus = skus[:args.limit]
    ocr_kwargs = ocr_mod.backend_kwargs_from_args(args)
    print(f"Scoring {len(skus)} SKUs, run {args.run_name!r}, stages {','.join(args.stages)}, OCR {args.backend}")

    results, failed = [], {}
    for sku in skus:
        try:
            results.append(evaluate_sku(sku, eval_dir, out_dir, args, ocr_kwargs))
        except Exception as e:
            import traceback
            failed[sku] = f"{type(e).__name__}: {e}"
            print(f"  [{sku}] FAILED {failed[sku]}")
            traceback.print_exc()

    agg = aggregate_results(results, args.stages)
    warn_counts = Counter(w.replace(r["sku"], "<sku>") for r in results for w in r["warnings"])
    summary = {
        "run_name": args.run_name, "eval_dir": str(eval_dir), "backend": args.backend,
        "backend_kwargs": ocr_kwargs, "stages": args.stages, "n_skus": len(results),
        "skus": [r["sku"] for r in results], "failed": failed,
        "gt_sources": dict(Counter(r["gt_source"] for r in results)),
        "dry_run_skus": [r["sku"] for r in results if r["dry_run"]],
        "warnings": dict(warn_counts),
        "buckets": list(tm.BUCKET_NAMES),
        "aggregate": agg,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    write_tables(out_dir, results, agg, args.stages)

    print(f"\n{'stage':<9} {'n':>3} {'NED':>6} {'wP':>6} {'wR':>6} {'wF1':>6} {'hit':>6} | {'rNED':>6} {'rHit':>6} {'rwR':>6}")
    for n in args.stages:
        a = agg[n]
        if not a["n_skus"]:
            print(f"{n:<9} {0:>3}  (skipped)")
            continue
        g, r = a["global"]["mean"], a["region"]
        fmt = lambda v: f"{v:6.3f}" if isinstance(v, (int, float)) else f"{'-':>6}"
        print(f"{n:<9} {a['n_skus']:>3} {fmt(g['ned'])} {fmt(g['word_precision'])} {fmt(g['word_recall'])} "
              f"{fmt(g['word_f1'])} {fmt(g['phrase_hit'])} | {fmt(r['line_ned_mean_over_skus'])} "
              f"{fmt(r['hit_rate_pooled'])} {fmt(r['word_recall_pooled'])}")
    for w, c in warn_counts.items():
        print(f"WARNING x{c}: {w}")
    print(f"Wrote {out_dir}/summary.json, summary_stages.csv, summary_buckets.csv, lines.csv")


if __name__ == "__main__":
    main()
