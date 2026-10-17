"""
Text regions of reference images, and the text blur that makes the texture LoRA read glyph tokens.

The texture pass sees the reference image (a render_random view in training, the product photo at
inference) next to the glyph tokens. Training references show every line sharply, so the model
copies letters from the reference and the glyph tokens add nothing at inference: the same LoRA with
no glyph tokens (run pos_noglyph) gets as many lines right as with them. Blurring small and medium
text in the reference, in training and at inference, leaves the glyph tokens as the only source of
those letters. Large text (logos) stays sharp, so brand lettering still comes from the reference.

Text is found with PaddleOCR's text detector (PP-OCRv5_mobile_det on the CPU with oneDNN off: the
paddle 3.3 CPU build fails inside its oneDNN executor). Polygons are stored in detection pixels and
scaled to the image they blur. A region is blurred when its text height (short side of the minimum
area rectangle) is at most max_height px at 1024: grown by pad_frac * height, Gaussian blur with
sigma_frac * height, blended in through a feathered mask.

  precompute (training references, resumable, one JSON per uid):
    python -m unitex.ref_text precompute --root <data_root> [--root ...] --out <dir> [--workers 7]
    -> <dir>/<uid>.json  {"uid", "res": [W, H], "detector", "views": {"<k>": [[[x, y], ...], ...]}}
  one image (debug):
    python -m unitex.ref_text blur --image in.png --out out.png
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DETECTOR = {"model_name": "PP-OCRv5_mobile_det", "limit_side_len": 1024, "limit_type": "max"}
BASE_RES = 1024         # max_height and the detection input are in px at this size
MAX_HEIGHT = 64         # 64 px at 1024 = 32 px at 512: every line height the eval gets wrong
N_REF_VIEWS = 20        # render_random views the dataset picks from (flux_dataset.N_REF_VIEWS)


def make_detector(cpu_threads=8):
    from paddleocr import TextDetection
    return TextDetection(device="cpu", enable_mkldnn=False, cpu_threads=cpu_threads, **DETECTOR)


def on_white(im):
    """PIL image -> uint8 RGB array, alpha composited on white (how the dataset shows references)."""
    a = np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:] + (1.0 - a[..., 3:])
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def detect(det, rgb):
    """uint8 RGB array -> text polygons [[x, y], ...] in its pixels."""
    res = det.predict(np.ascontiguousarray(rgb[..., ::-1]))
    if not res:
        return []
    return [[[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(p, np.float64).reshape(-1, 2)]
            for p in res[0]["dt_polys"]]


def poly_height(q):
    """Text height of a polygon: short side of its minimum area rectangle."""
    import cv2
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(q, np.float32).reshape(-1, 2))
    return float(min(w, h))


def blur_text(img, polys, scale=1.0, max_height=MAX_HEIGHT, sigma_frac=0.5, min_sigma=2.0, pad_frac=0.3):
    """Blur the text polygons of img whose height is at most max_height (px at BASE_RES).

    img: float32 [H, W, C]. polys: polygons in detection px, times scale = img px.
    Returns (blurred copy, float32 [H, W] blend weight of all regions).
    """
    import cv2
    H, W = img.shape[:2]
    lim = max_height * W / BASE_RES
    out = np.array(img, np.float32, copy=True)
    weight = np.zeros((H, W), np.float32)
    for p in polys:
        q = np.asarray(p, np.float32).reshape(-1, 2) * scale
        h = poly_height(q)
        if not 0 < h <= lim:
            continue
        sigma = max(min_sigma, sigma_frac * h)
        pad = max(1, int(round(pad_frac * h)))
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)))
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        r = int(np.ceil(3 * sigma))
        x0, x1 = max(0, xs.min() - r), min(W, xs.max() + r + 1)
        y0, y1 = max(0, ys.min() - r), min(H, ys.max() + r + 1)
        crop = out[y0:y1, x0:x1]
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        soft = cv2.GaussianBlur(m[y0:y1, x0:x1].astype(np.float32), (0, 0), sigmaX=max(1.0, pad / 2))
        soft = np.maximum(soft, m[y0:y1, x0:x1].astype(np.float32) * (soft > 0.5))   # full weight inside
        if crop.ndim == 3:
            out[y0:y1, x0:x1] = crop * (1 - soft[..., None]) + blurred.reshape(crop.shape) * soft[..., None]
        else:
            out[y0:y1, x0:x1] = crop * (1 - soft) + blurred * soft
        weight[y0:y1, x0:x1] = np.maximum(weight[y0:y1, x0:x1], soft)
    return out, weight


def n_blurred(polys, scale, width, max_height=MAX_HEIGHT):
    lim = max_height * width / BASE_RES
    return sum(0 < poly_height(np.asarray(p, np.float32).reshape(-1, 2) * scale) <= lim for p in polys)


def blur_tensor(x, polys, scale=1.0, **kw):
    """torch [C, H, W] float image -> blurred copy, same dtype and device."""
    import torch
    a = x.detach().permute(1, 2, 0).float().cpu().numpy()
    out, _ = blur_text(a, polys, scale, **kw)
    return torch.from_numpy(out).permute(2, 0, 1).to(x)


def blur_pil(im, polys, scale=1.0, **kw):
    """PIL image -> (blurred RGB PIL image, float32 [H, W] blend weight)."""
    a = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
    out, weight = blur_text(a, polys, scale, **kw)
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)), weight


# ────────────────────────────────────────────────────────────────────────────
# Training references
# ────────────────────────────────────────────────────────────────────────────

def ref_json_path(out_dir, uid):
    return os.path.join(out_dir, f"{uid}.json")


def load_ref_polys(ref_dir, uid, k):
    """Polygons of render_random view k of uid and their resolution (W, H)."""
    with open(ref_json_path(ref_dir, uid)) as f:
        d = json.load(f)
    return d["views"].get(str(k), []), tuple(d["res"])


_DET = None


def _init_worker(cpu_threads):
    global _DET
    _DET = make_detector(cpu_threads)


def detect_uid(root, uid, out_dir, n_views=N_REF_VIEWS, image_ext="auto", detect_fn=None):
    """Detect text in render_random views 0..n_views-1 of one uid and write its JSON (atomic)."""
    from unitex.flux_dataset import ImageDir
    out = ref_json_path(out_dir, uid)
    if os.path.exists(out):
        return "exists"
    detect_fn = detect_fn or (lambda a: detect(_DET, a))
    views, res = {}, None
    with ImageDir(os.path.join(root, "render_random", uid), image_ext) as src:
        for k in range(n_views):
            rgb = on_white(src.open(f"{k:04d}_rgb"))
            res = [int(rgb.shape[1]), int(rgb.shape[0])]
            views[str(k)] = detect_fn(rgb)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump({"uid": uid, "root": os.path.abspath(root), "res": res, "detector": DETECTOR, "views": views}, f)
    os.replace(out + ".tmp", out)
    return "ok"


def _job(args):
    t = time.time()
    try:
        return args[1], detect_uid(*args), time.time() - t
    except Exception as e:              # reported and counted, the other uids go on
        return args[1], f"error {type(e).__name__}: {e}", time.time() - t


def precompute(roots, out_dir, workers=7, cpu_threads=8, n_views=N_REF_VIEWS, image_ext="auto", detect_fn=None):
    """Every uid of every root's training_uid.json. detect_fn (tests) runs in this process."""
    jobs = []
    for root in roots:
        with open(os.path.join(root, "training_uid.json")) as f:
            jobs += [(root, uid, out_dir, n_views, image_ext) for uid in json.load(f)]
    todo = [j for j in jobs if not os.path.exists(ref_json_path(out_dir, j[1]))]
    print(f"ref_text precompute: {len(jobs)} uids, {len(todo)} to do, {workers} workers x {cpu_threads} threads")
    errors = []
    if detect_fn is not None or workers <= 1:
        if detect_fn is None:
            _init_worker(cpu_threads)
        results = (_job(j[:4] + (j[4], detect_fn)) for j in todo)
        for i, (uid, status, dt) in enumerate(results):
            errors += [(uid, status)] if status.startswith("error") else []
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(cpu_threads,)) as pool:
            t0 = time.time()
            for i, (uid, status, dt) in enumerate(pool.imap_unordered(_job, todo), 1):
                errors += [(uid, status)] if status.startswith("error") else []
                if i % 20 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} uids, {time.time() - t0:.0f} s, last {uid} {status} {dt:.1f} s", flush=True)
    for uid, status in errors:
        print(f"  {uid}: {status}", file=sys.stderr)
    print(f"ref_text precompute done: {len(todo) - len(errors)} written, {len(errors)} errors")
    return errors


# ────────────────────────────────────────────────────────────────────────────
# Inference reference
# ────────────────────────────────────────────────────────────────────────────

def blur_reference_file(path, detect_fn, max_height=MAX_HEIGHT):
    """Detect text in the reference image at path (at BASE_RES) and blur it in place.

    Next to it: <stem>_sharp.png (the original), ref_text_mask.png (blend weight), ref_text.json.
    """
    im = Image.open(path)
    rgb = np.asarray(im.convert("RGB"))
    W = rgb.shape[1]
    det_in = rgb if W == BASE_RES else np.asarray(Image.fromarray(rgb).resize((BASE_RES, BASE_RES), Image.BICUBIC))
    polys = detect_fn(det_in)
    scale = W / BASE_RES
    out, weight = blur_pil(im, polys, scale=scale, max_height=max_height)
    stem, _ = os.path.splitext(path)
    d = os.path.dirname(path)
    im.save(stem + "_sharp.png")
    out.save(path)
    Image.fromarray((weight * 255 + 0.5).astype(np.uint8)).save(os.path.join(d, "ref_text_mask.png"))
    info = {"n_regions": len(polys), "n_blurred": int(n_blurred(polys, scale, W, max_height)),
            "max_height": max_height, "detector": DETECTOR}
    with open(os.path.join(d, "ref_text.json"), "w") as f:
        json.dump({**info, "det_res": BASE_RES, "polys": polys}, f)
    return info


def wrap_reference_blur(pipe, max_height=MAX_HEIGHT, detect_fn=None, log=print):
    """Blur reference text right after UniTEX writes cache/processed_image.png. That file is the
    texture pass's dual image and only infer_mv reads it (the delight pass sees the lit strip)."""
    orig = pipe.preprocess_reference_image
    state = {"det": None, "last": None}

    def default_detect(a):
        if state["det"] is None:
            state["det"] = make_detector()
        return detect(state["det"], a)

    def wrapped(save_dir, input_image_path, *a, **kw):
        r = orig(save_dir, input_image_path, *a, **kw)
        state["last"] = blur_reference_file(os.path.join(save_dir, "processed_image.png"),
                                            detect_fn or default_detect, max_height)
        log(f"  [ref_text] blurred {state['last']['n_blurred']} of {state['last']['n_regions']} text regions "
            f"in the reference")
        return r

    pipe.preprocess_reference_image = wrapped
    pipe.ref_text_state = state
    return pipe


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("precompute")
    c.add_argument("--root", action="append", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=7)
    c.add_argument("--cpu-threads", type=int, default=8)
    c.add_argument("--views", type=int, default=N_REF_VIEWS)
    b = sub.add_parser("blur")
    b.add_argument("--image", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--max-height", type=int, default=MAX_HEIGHT)
    args = p.parse_args(argv)
    if args.cmd == "precompute":
        return 1 if precompute(args.root, args.out, args.workers, args.cpu_threads, args.views) else 0
    det = make_detector()
    im = Image.open(args.image)
    rgb = on_white(im)
    scale = 1.0
    if rgb.shape[1] != BASE_RES:
        scale = rgb.shape[1] / BASE_RES
        rgb = np.asarray(Image.fromarray(rgb).resize((BASE_RES, BASE_RES), Image.BICUBIC))
    polys = detect(det, rgb)
    out, _ = blur_pil(Image.fromarray(on_white(im)), polys, scale=scale, max_height=args.max_height)
    out.save(args.out)
    print(f"{len(polys)} text regions, {n_blurred(polys, scale, out.width, args.max_height)} blurred -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
