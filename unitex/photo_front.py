"""
Photo front: put the real product photo into UniTEX's generated front view, then re-bake.

UniTEX redraws the whole label through FLUX. Small print does not survive: Lucky Charms "18.6 oz"
comes out as "13.6" with or without glyph tokens, while the photo it was given shows "18.6"
clearly. This tool works on a finished UniTEX run (any LoRA, any view_res) and never modifies it:

  1. aligns the photo to view 0: the unitex/anchors.py fit (photo foreground bbox onto the view-0
     silhouette, refined for silhouette IoU), then a RANSAC homography from SIFT matches against
     the generated front view, which shows the photo's layout rectified to the mesh (--register),
  2. replaces view 0 of cache/mv_rgb.png where the surface faces the front camera, feathered:
       detail  (default) the photo's detail (photo minus its blur) on top of the generated view's
               blur, so colours and delit shading stay UniTEX's and the front meets the side views
               without a seam
       full    the photo's pixels as they are
       none    no change, a control: the re-bake alone should reproduce the source run
  3. re-runs only UniTEX's bake (step_2_ablition: reprojection + LTM inpainting) into a new run
     directory. FLUX is not loaded.

Outputs in <eval>/<sku>/<run-name>/: textured_mesh.glb, mv_rgb.png, photo_front.png (generated
front | warped photo | blend weight | result), run_info.json (alignment IoU, settings), and the
copied cache with cache/mv_rgb_generated.png (view 0 before the blend).

Usage (GPU server, UniTEX checkout with unitex/patches/UniTEX.patch):
  python unitex/photo_front.py --unitex-root /path/UniTEX --eval-dir EVAL --src-run glyph_trainAB \\
      --run-name glyph_trainAB_photofront [--skus a,b | file] [--mode detail|full|none] [--no-bake]
"""

import argparse
import datetime
import json
import os
import pathlib
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from unitex import anchors as A  # noqa: E402
from unitex import common as C  # noqa: E402

FRONT = 0                          # raw view index of the front camera
TO_FRONT_CAMERA = np.array([0.0, -1.0, 0.0])   # Blender frame, RAW_C2W[0] sits at -Y looking at the origin


def grid_tile(raw_idx):
    """(tile, rolled) of a raw view in UniTEX's 2x3 inference grid."""
    for tile, (r, rolled) in enumerate(C.GRID_TILE_TO_RAW):
        if r == raw_idx:
            return tile, rolled
    raise ValueError(f"raw view {raw_idx} is not in the grid")


def put_view(grid, raw_idx, img, res):
    """Inverse of common.split_grid for one view: write a raw-oriented image into its tile."""
    tile, rolled = grid_tile(raw_idx)
    r, c = divmod(tile, 3)
    grid[r * res:(r + 1) * res, c * res:(c + 1) * res] = img[::-1, ::-1] if rolled else img
    return grid


def warp_into_view(img, aff, res, resample=Image.LANCZOS):
    """PIL image in photo pixels -> (res, res[, C]) array in view pixels, x' = sx x + tx, y' = sy y + ty.

    Filtered for the downscale (anchors.warp_photo is nearest, for debug overlays). Outside the
    photo is 0.
    """
    sx, sy, tx, ty = aff
    W, H = img.size
    box = [(0 - tx) / sx, (0 - ty) / sy, (res - tx) / sx, (res - ty) / sy]
    pad = int(np.ceil(max(0.0, -box[0], -box[1], box[2] - W, box[3] - H))) + 2
    canvas = Image.new(img.mode, (W + 2 * pad, H + 2 * pad), 0)
    canvas.paste(img, (pad, pad))
    return np.asarray(canvas.resize((res, res), resample, box=tuple(v + pad for v in box)))


def masked_blur(img, mask, sigma):
    """Gaussian blur of img from the pixels where mask is set only (normalized convolution)."""
    import cv2
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-6)


def front_weight(mask0, normals0, photo_mask, res, facing=(0.35, 0.7), feather_frac=1 / 128):
    """Blend weight in view 0 and the trusted core it was built from.

    core: view-0 silhouette and warped photo mask, eroded by 2 * feather px so the feathered edge
    (Gaussian, sigma = feather) stays inside both. Weight: core feathered, times a smoothstep of how directly the surface
    faces the front camera (cos between facing[0] and facing[1]). Grazing surfaces keep the
    generated view, the side views see them better than the photo does.
    """
    import cv2
    feather = max(1.0, res * feather_frac)
    r = int(np.ceil(2 * feather))
    core = np.logical_and(mask0, photo_mask >= 0.5).astype(np.uint8)
    core = cv2.erode(core, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    cos = normals0 @ TO_FRONT_CAMERA
    lo, hi = facing
    f = np.clip((cos - lo) / (hi - lo), 0.0, 1.0)
    f = f * f * (3 - 2 * f)
    w = cv2.GaussianBlur(core.astype(np.float32), (0, 0), feather) * f
    w[~mask0] = 0.0
    return w.astype(np.float32), core


def local_agreement(gen0, photo, region, res, win_frac=1 / 16, blur_frac=1 / 64, lo=0.5, hi=0.8, flat_std=4.0):
    """Per-pixel trust in the aligned photo, in [0, 1].

    Local NCC (Gaussian window, sigma = win_frac * res, inside region) between the blurred grey
    generated view and the blurred grey photo, smoothstepped from lo to hi. The blur (blur_frac *
    res) hides the letters themselves, which are what differ, the layout around them should agree.
    A homography fits a box face exactly but not a tub or jar shot from above: Betty Crocker's
    "16 OZ" badge lands above the generated badge and both would show. Where the generated view
    is flat (local std under flat_std grey levels) nothing can conflict and the photo is trusted.
    """
    import cv2
    b = max(1.0, res * blur_frac)
    g = cv2.GaussianBlur(_gray(gen0).astype(np.float32), (0, 0), b)
    p = cv2.GaussianBlur(_gray(photo).astype(np.float32), (0, 0), b)
    m = region.astype(np.float32)
    s = max(1.0, res * win_frac)
    den = np.maximum(cv2.GaussianBlur(m, (0, 0), s), 1e-6)

    def avg(x):
        return cv2.GaussianBlur(x * m, (0, 0), s) / den

    mg, mp = avg(g), avg(p)
    vg = np.maximum(avg(g * g) - mg * mg, 0.0)
    vp = np.maximum(avg(p * p) - mp * mp, 0.0)
    ncc = (avg(g * p) - mg * mp) / np.sqrt(vg * vp + 1e-6)
    t = np.clip((ncc - lo) / (hi - lo), 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    trust = np.where(np.sqrt(vg) < flat_std, 1.0, t).astype(np.float32)
    return cv2.GaussianBlur(trust, (0, 0), s / 2)


def text_weight(text_json, mask0, normals0, photo_mask, res, pad=0.3, facing=0.15, feather_frac=1 / 128):
    """Weight 1 on the photo's own text lines, so each line comes from the photo whole.

    The lines are the view0_quad entries of a registered OCR file (text_reg.json), padded by pad times the
    line height. Without this, the facing ramp keeps the generated letters where a can turns away from the
    camera and the local trust keeps them where the generated text differs from the photo's, so a line can
    end up half photo, half made up ("Vitalizes" read "italizes" over a generated "Violines"). Surfaces
    facing the camera less than cos facing, and pixels outside the warped photo, keep weight 0.
    Returns (weight (res, res) float32, number of lines).
    """
    import cv2
    p = pathlib.Path(text_json)
    if not p.exists():
        return np.zeros((res, res), np.float32), 0
    d = json.loads(p.read_text())
    s = res / float(d.get("res", res))
    m = np.zeros((res, res), np.uint8)
    n = 0
    for it in d.get("items", []):
        q = it.get("view0_quad")
        if not q:
            continue
        q = np.asarray(q, np.float32) * s                      # top-left, top-right, bottom-right, bottom-left
        c = q.mean(axis=0)
        h = max(float(np.linalg.norm(q[3] - q[0])), float(np.linalg.norm(q[2] - q[1])), 1.0)
        wv = (q[1] - q[0]) / max(1e-6, float(np.linalg.norm(q[1] - q[0])))
        hv = (q[3] - q[0]) / max(1e-6, float(np.linalg.norm(q[3] - q[0])))
        out = np.sign((q - c) @ wv)[:, None] * wv + np.sign((q - c) @ hv)[:, None] * hv
        cv2.fillPoly(m, [np.round(q + out * pad * h).astype(np.int32)], 1)
        n += 1
    wt = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, res * feather_frac))
    g = np.clip((normals0 @ TO_FRONT_CAMERA - facing) / 0.1, 0.0, 1.0)
    wt = wt * g * (photo_mask >= 0.5) * mask0
    return wt.astype(np.float32), n


def blend(gen, photo, w, core, mode="detail", sigma=16.0, under=None):
    """gen, photo (res, res, 3) uint8, w (res, res) in [0, 1] -> uint8 view. under: what shows where w < 1
    (default gen)."""
    g = gen.astype(np.float32)
    p = photo.astype(np.float32)
    if mode == "none":
        return gen.copy()
    if mode == "detail":
        out = masked_blur(g, core, sigma) + (p - masked_blur(p, core, sigma))
    elif mode == "full":
        out = p
    else:
        raise ValueError(f"mode must be detail, full or none, got {mode}")
    u = g if under is None else under.astype(np.float32)
    res = w[..., None] * out + (1.0 - w[..., None]) * u
    return np.clip(np.rint(res), 0, 255).astype(np.uint8)


def own_layout_under(gen0, photo, mask0, pmask, sigma):
    """What shows where the photo fades out when it keeps its own layout: its own blur inside the warped photo,
    the generated view's blur outside it, the generated view outside the silhouette. With the generated blur
    alone the box of every whole text line showed where the facing ramp let the generated colours in (Betty
    Crocker's tub had a lighter rectangle round its lines). float32 (res, res, 3)."""
    import cv2
    m0 = mask0.astype(bool)
    pin = m0 & (pmask >= 0.5)
    pf = cv2.GaussianBlur(pin.astype(np.float32), (0, 0), max(1.0, sigma / 2))[..., None]
    u = pf * masked_blur(photo.astype(np.float32), pin, sigma) + (1.0 - pf) * masked_blur(gen0.astype(np.float32), m0, sigma)
    return np.where(m0[..., None], u, gen0.astype(np.float32)).astype(np.float32)


def _gray(a):
    import cv2
    return cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY)


def _ncc(a, b, mask):
    a, b = a[mask].astype(np.float64), b[mask].astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def register_photo(gen0, mask0, photo, pmask, aff, res, method="homography", scale=2, ratio=0.75,
                   min_inliers=12, max_corner_shift=0.5, flow_below=0.8, flow_min=0.9):
    """Photo -> view-0 pixels. Returns (photo (res, res, 3) uint8, photo mask (res, res) float, info).

    affine      the anchors.py fit only: photo foreground bbox onto the view-0 silhouette bbox,
                IoU-refined. Off when the photo is not frontal: a three-quarter shot of a box shows
                a side panel, and the bbox fit squeezes it onto the front face.
    homography  then a RANSAC homography from SIFT matches between the affine-warped photo and the
                generated front view, which shows the photo's layout rectified to the mesh's front
                face. Exact for a planar face. Kept only with min_inliers inliers, the silhouette
                bbox corners moving less than max_corner_shift of its size (a photo from above can
                need a large move: the bbox fit squeezes the lid in), and the blurred grey images
                agreeing at least as well as before (NCC). Otherwise the affine is used.
    flow        homography, then a smoothed DIS optical flow (curved labels)
    auto        flow only when the homography's NCC is under flow_below. The flow fixes a jar or tub
                shot from above (Jif 0.79 -> 0.96), but on a photo the homography already fits it shears
                letters ("Gluten" came out slanted), so it is not used there. A flow that still leaves the
                NCC under flow_min is not used either: the generated layout itself differs from the photo
                (Betty Crocker's ribbon is tilted and its badge sits lower, 0.74 -> 0.85), and bending the
                photo onto it mixed two copies of "Cream Cheese". info["layout"] is then "photo": the
                photo is placed by the affine silhouette fit and front_from_photo keeps its own layout.
                info["ncc_matched"] keeps the best fit's NCC, which still decides whether the photo shows
                the product's front at all (min_ncc).
    All work at scale * res and the result is area-downsampled, so small print is not aliased.
    """
    import cv2
    S = res * scale
    pre = np.ascontiguousarray(warp_into_view(photo, tuple(v * scale for v in aff), S)[..., :3])
    pre_m = warp_into_view(Image.fromarray(pmask.astype(np.uint8) * 255), tuple(v * scale for v in aff), S,
                           Image.BILINEAR).astype(np.float32) / 255.0
    info = {"method": "affine", "requested": method, "scale": scale}
    pre_aff, pre_m_aff = pre, pre_m
    g2 = cv2.resize(gen0, (S, S), interpolation=cv2.INTER_CUBIC)
    m2 = cv2.resize(mask0.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
    blur = lambda a: cv2.GaussianBlur(_gray(a), (0, 0), S / 128)       # noqa: E731
    region = m2 & (pre_m >= 0.5)
    info["ncc_affine"] = round(_ncc(blur(g2), blur(pre), region), 4)
    if method in ("homography", "flow", "auto"):
        sift = cv2.SIFT_create(nfeatures=5000)
        kg, dg = sift.detectAndCompute(_gray(g2), m2.astype(np.uint8) * 255)
        kp, dp = sift.detectAndCompute(_gray(pre), (pre_m >= 0.5).astype(np.uint8) * 255)
        good = []
        if dg is not None and dp is not None and len(kg) >= 2 and len(kp) >= 2:
            knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dp, dg, k=2)
            good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
        info.update(n_kp_gen=len(kg), n_kp_photo=len(kp), n_matches=len(good))
        H = None
        if len(good) >= min_inliers:
            src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kg[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0 * scale, maxIters=5000, confidence=0.999)
            n_in = int(inl.sum()) if inl is not None else 0
            info["n_inliers"] = n_in
            if H is None or n_in < min_inliers:
                H = None
        if H is not None:
            x0, y0, x1, y1 = C.bbox_of_mask(m2)
            corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
            shift = np.abs(cv2.perspectiveTransform(corners, H) - corners).reshape(-1, 2) / [x1 - x0, y1 - y0]
            info["corner_shift_frac"] = round(float(shift.max()), 4)
            warped = cv2.warpPerspective(pre, H, (S, S), flags=cv2.INTER_CUBIC, borderValue=0)
            warped_m = cv2.warpPerspective(pre_m, H, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
            ncc_h = _ncc(blur(g2), blur(warped), m2 & (warped_m >= 0.5))
            info["ncc_homography"] = round(ncc_h, 4)
            if shift.max() <= max_corner_shift and ncc_h >= info["ncc_affine"] - 0.01:
                pre, pre_m, info["method"] = warped, warped_m, "homography"
                info["homography"] = [[round(float(v), 6) for v in row] for row in H]
        if info["method"] == "homography" and (method == "flow" or (method == "auto"
                                                and info["ncc_homography"] < flow_below)):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            a = cv2.GaussianBlur(_gray(g2), (0, 0), 1.5 * scale)
            b = cv2.GaussianBlur(_gray(pre), (0, 0), 1.5 * scale)
            f = dis.calc(a, b, None)                                   # a(x) ~ b(x + f(x))
            reg = (m2 & (pre_m >= 0.5)).astype(np.float32)
            num = cv2.GaussianBlur(f * reg[..., None], (0, 0), S / 48)
            f = num / np.maximum(cv2.GaussianBlur(reg, (0, 0), S / 48)[..., None], 1e-6)
            mag = np.linalg.norm(f, axis=-1, keepdims=True)
            f = f * np.minimum(1.0, (S / 32) / np.maximum(mag, 1e-6))
            xx, yy = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
            mx, my = xx + f[..., 0], yy + f[..., 1]
            warped = cv2.remap(pre, mx, my, cv2.INTER_CUBIC, borderValue=0)
            warped_m = cv2.remap(pre_m, mx, my, cv2.INTER_LINEAR, borderValue=0)
            ncc_f = _ncc(blur(g2), blur(warped), m2 & (warped_m >= 0.5))
            info.update(ncc_flow=round(ncc_f, 4), flow_max_px=round(float(mag.max()) / scale, 2))
            if ncc_f >= info["ncc_homography"] - 0.01 and (method == "flow" or ncc_f >= flow_min):
                pre, pre_m, info["method"] = warped, warped_m, "flow"
    info["ncc_final"] = info[f"ncc_{info['method']}"]
    own = method == "auto" and info["method"] != "flow" and info["ncc_final"] < flow_below
    info["layout"] = "photo" if own else "generated"
    info["ncc_matched"] = info["ncc_final"]                 # best fit to the generated front, for the min_ncc gate
    if own and info["method"] != "affine":
        # a fit to a layout that differs only bends the photo (Betty Crocker's homography moved the silhouette
        # corners by 24 %), so the photo goes where the silhouettes put it
        pre, pre_m, info["method"], info["ncc_final"] = pre_aff, pre_m_aff, "affine", info["ncc_affine"]
    ph = cv2.resize(pre, (res, res), interpolation=cv2.INTER_AREA)
    pm = cv2.resize(pre_m, (res, res), interpolation=cv2.INTER_AREA)
    return ph, pm, info


def front_from_photo(sku_dir, src_run_dir, cache_dir, mode="detail", sigma_frac=1 / 32,
                     feather_frac=1 / 128, facing=(0.35, 0.7), register="homography", min_ncc=0.6,
                     trust=(0.5, 0.8), text_json=None, text_pad=0.3, text_min_ncc=0.8, flow_below=0.8,
                     flow_min=0.9):
    """Blend the photo into view 0 of cache_dir/mv_rgb.png. Returns (new grid, debug panel, info).

    The photo is used only when it registers: blurred grey NCC with the generated front view of at
    least min_ncc after alignment. Below that (Shin Ramyun cup shot from above: 0.04 after the
    bbox fit) pasting it would put the wrong part of the photo on the front, so view 0 is kept.
    With text_json (a registered OCR file in the SKU folder, e.g. text_reg.json) every photo text
    line gets weight 1 (text_weight), when the photo registers with NCC of at least text_min_ncc. Below
    that the photo sits a little off (a tub or jar shot from above), and forcing whole lines in would
    show them next to the generated ones (Betty Crocker's "16 OZ" twice).
    When --register auto finds that the generated layout differs from the photo (layout "photo", see
    register_photo), the photo replaces the facing part of view 0 as it is (mode full, no local trust,
    whole text lines), and where it fades out its own blur shows (own_layout_under), so no generated
    letters sit next to the photo's.
    """
    sku_dir, cache_dir = pathlib.Path(sku_dir), pathlib.Path(cache_dir)
    with open(sku_dir / "ref_meta.json") as f:
        meta = json.load(f)
    geo = A.load_unitex_cache(cache_dir)
    res = geo.res
    grid = np.array(Image.open(cache_dir / "mv_rgb.png").convert("RGB"))
    gen0 = np.ascontiguousarray(C.split_grid(grid, res)[FRONT])
    photo = A.load_photo(sku_dir, meta)
    pmask, fg_box, mask_src = A.photo_foreground(photo, meta, src_run_dir)
    aff, align = A.fit_photo_to_view0(pmask, fg_box, geo.masks[FRONT])
    ph, pm, reg = register_photo(gen0, geo.masks[FRONT], photo, pmask, aff, res, register, flow_below=flow_below,
                                 flow_min=flow_min)
    own_layout = reg["layout"] == "photo"
    w, core = front_weight(geo.masks[FRONT], geo.normals[FRONT], pm, res, facing, feather_frac)
    trust_ramp = tuple(trust)
    trust = (np.ones_like(w) if own_layout
             else local_agreement(gen0, ph, core, res, lo=trust_ramp[0], hi=trust_ramp[1]))
    w = w * trust
    n_text = 0
    ncc_gate = reg["ncc_homography"] if reg["method"] in ("homography", "flow") else reg["ncc_final"]
    if text_json and (own_layout or ncc_gate >= text_min_ncc):          # the fit before any flow
        wt, n_text = text_weight(sku_dir / text_json, geo.masks[FRONT], geo.normals[FRONT], pm, res, text_pad,
                                 feather_frac=feather_frac)
        w = np.maximum(w, wt)
    skipped = None
    if reg["ncc_matched"] < min_ncc:
        skipped = f"photo not registered: NCC {reg['ncc_matched']} < {min_ncc}, view 0 kept"
        w = np.zeros_like(w)
    sigma = max(1.0, res * sigma_frac)
    if own_layout and not skipped:
        mode = "full"
        new0 = blend(gen0, ph, w, core, mode, sigma, under=own_layout_under(gen0, ph, geo.masks[FRONT], pm, sigma))
    else:
        new0 = blend(gen0, ph, w, core, mode, sigma)
    out = put_view(grid.copy(), FRONT, new0, res)
    wv = np.repeat((w * 255).astype(np.uint8)[..., None], 3, axis=2)
    panel = np.concatenate([gen0, np.where(pm[..., None] >= 0.5, ph, 128).astype(np.uint8), wv, new0], axis=1)
    m0 = geo.masks[FRONT]
    info = {"view_res": res, "mode": mode, "sigma_px": round(sigma, 2), "feather_px": round(max(1.0, res * feather_frac), 2),
            "facing": list(facing), "photo_mask_source": mask_src, "align": align, "register": reg,
            "min_ncc": min_ncc, "skipped": skipped, "trust_ramp": list(trust_ramp),
            "text_json": text_json, "text_lines": n_text,
            "local_trust_mean_on_core": round(float(trust[core].mean()), 4) if core.any() else 0.0,
            "weight_mean_on_silhouette": round(float(w[m0].mean()), 4) if m0.any() else 0.0,
            "replaced_frac_of_silhouette": round(float((w[m0] >= 0.5).mean()), 4) if m0.any() else 0.0}
    return out, panel, info


def build_baker(unitex_root, view_res=512, seed=63):
    """UniTEX pipeline that runs only step_2_ablition (bake + LTM). FLUX and RMBG are not built."""
    root = os.path.abspath(unitex_root)
    if not os.path.exists(os.path.join(root, "pipeline.py")):
        raise SystemExit(f"{root} is not a UniTEX checkout (no pipeline.py)")
    os.chdir(root)                 # LTM/configs/... is opened relative to the cwd
    sys.path.insert(0, root)
    import pipeline as U
    U.build_pipeline = lambda **kw: (None, [1.0, 0.0], [0.0, 1.0], ["texture", "delight"])
    U.RMBG2 = lambda **kw: None
    extra = {"view_res": view_res} if view_res != 512 else {}
    pipe = U.CustomRGBTextureFullPipeline(super_resolutions=False, filt_gradient_points=False,
                                          filt_large_angle_points=True, seed=seed, **extra)
    pipe.step_seq = ["step_2_ablition"]
    pipe.export_video = lambda *a, **k: None     # two 120-frame orbit mp4s per SKU otherwise
    return pipe


def copy_cache(src_cache, dst_cache):
    if dst_cache.exists():
        shutil.rmtree(dst_cache)
    shutil.copytree(src_cache, dst_cache,
                    ignore=shutil.ignore_patterns("w_LTM", "wo_LTM", "textured_mesh.glb", "*.mp4"))


def run(args):
    eval_dir = pathlib.Path(args.eval_dir).resolve()
    if args.skus:
        p = pathlib.Path(args.skus)
        skus = [l.split("#")[0].strip() for l in open(p)] if p.is_file() else args.skus.split(",")
        skus = [s for s in skus if s]
    else:
        skus = sorted(d.name for d in eval_dir.iterdir() if (d / args.src_run / "cache" / "mv_rgb.png").exists())
    log_path = eval_dir / "photo_front_log.jsonl"
    baker = None
    n_ok = n_err = 0
    for i, sku in enumerate(skus):
        sku_dir = eval_dir / sku
        src, dst = sku_dir / args.src_run, sku_dir / args.run_name
        rec = {"sku": sku, "src_run": args.src_run, "run_name": args.run_name, "mode": args.mode,
               "time": datetime.datetime.now().isoformat(timespec="seconds")}
        if args.resume and (dst / "run_info.json").exists():
            print(f"  [{sku}] done before, skipped")
            continue
        if not (src / "cache" / "mv_rgb.png").exists():
            rec.update(status="missing_input", error=f"{src}/cache/mv_rgb.png")
            print(f"  [{sku}] no {args.src_run} run, skipped")
        else:
            t0 = time.time()
            try:
                dst.mkdir(parents=True, exist_ok=True)
                (dst / "run_info.json").unlink(missing_ok=True)
                copy_cache(src / "cache", dst / "cache")
                grid, panel, info = front_from_photo(sku_dir, src, dst / "cache", args.mode, args.sigma_frac,
                                                     args.feather_frac, tuple(args.facing), args.register,
                                                     args.min_ncc, tuple(args.trust), args.text_json, args.text_pad,
                                                     args.text_min_ncc, args.flow_below, args.flow_min)
                rec.update(info)
                shutil.copy(dst / "cache" / "mv_rgb.png", dst / "cache" / "mv_rgb_generated.png")
                Image.fromarray(grid).save(dst / "cache" / "mv_rgb.png")
                Image.fromarray(panel).save(dst / "photo_front.png")
                reg = info["register"]
                print(f"  [{sku}] ({i + 1}/{len(skus)}) aligned by {reg['method']}, {reg['layout']} layout "
                      f"(IoU {info['align'].get('iou')}, "
                      f"NCC affine {reg['ncc_affine']} homography {reg.get('ncc_homography')} flow {reg.get('ncc_flow')}, "
                      f"inliers {reg.get('n_inliers')}), "
                      + (info["skipped"] if info["skipped"] else
                         f"{100 * info['replaced_frac_of_silhouette']:.0f}% of the front silhouette from the photo"))
                if not args.no_bake:
                    if baker is None:
                        baker = build_baker(args.unitex_root, info["view_res"], args.seed)
                    elif getattr(baker, "view_res", 512) != info["view_res"]:
                        raise RuntimeError(f"view_res {info['view_res']} differs from the first SKU's, run them separately")
                    import torch
                    baker.generator = torch.Generator().manual_seed(args.seed)
                    torch.manual_seed(args.seed)
                    baker(str(dst), str(sku_dir / "ref.png"), str(sku_dir / "mesh.glb"))
                rec["status"] = "ok"
            except KeyboardInterrupt:
                raise
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
                print(f"  [{sku}] ERROR {rec['error']}")
            rec["wall_s"] = round(time.time() - t0, 1)
            if rec["status"] == "ok":
                with open(dst / "run_info.json", "w") as f:
                    json.dump(rec, f, indent=1)
                n_ok += 1
                print(f"  [{sku}] ok in {rec['wall_s']} s")
            else:
                n_err += 1
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"Done: {n_ok} ok, {n_err} errors. Log: {log_path}")
    return n_err


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unitex-root", default=".", help="UniTEX checkout (needed for the bake)")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--src-run", required=True, help="finished UniTEX run per SKU, read only")
    p.add_argument("--run-name", required=True, help="new per-SKU output dir")
    p.add_argument("--skus", default=None, help="file or comma list (default: SKUs with a --src-run cache)")
    p.add_argument("--mode", choices=("detail", "full", "none"), default="detail")
    p.add_argument("--register", choices=("affine", "homography", "flow", "auto"), default="homography",
                   help="photo to view 0: bbox affine only, + SIFT homography (default), + optical flow, "
                        "auto: flow only when the homography's NCC is under --flow-below")
    p.add_argument("--min-ncc", type=float, default=0.6,
                   help="use the photo only if it registers this well with the generated front (blurred grey NCC)")
    p.add_argument("--trust", type=float, nargs=2, default=(0.5, 0.8), metavar=("LO", "HI"),
                   help="local NCC (blurred grey, photo vs generated) where the photo starts and fully takes over")
    p.add_argument("--sigma-frac", type=float, default=1 / 32,
                   help="detail mode: blur sigma as a fraction of view_res (16 px at 512)")
    p.add_argument("--feather-frac", type=float, default=1 / 128, help="edge feather as a fraction of view_res")
    p.add_argument("--facing", type=float, nargs=2, default=(0.35, 0.7), metavar=("LO", "HI"),
                   help="cos to the front camera where the photo starts and fully takes over")
    p.add_argument("--text-json", default=None,
                   help="registered OCR file in the SKU folder (e.g. text_reg.json): its lines are taken from the "
                        "photo whole")
    p.add_argument("--text-pad", type=float, default=0.3, help="padding of each text line, times its height")
    p.add_argument("--text-min-ncc", type=float, default=0.8, help="registration NCC needed for --text-json")
    p.add_argument("--flow-below", type=float, default=0.8, help="--register auto: homography NCC under which flow is tried")
    p.add_argument("--flow-min", type=float, default=0.9,
                   help="--register auto: NCC the flow must reach, else the photo keeps its own layout")
    p.add_argument("--seed", type=int, default=63)
    p.add_argument("--resume", action="store_true", help="skip SKUs with run_info.json")
    p.add_argument("--no-bake", action="store_true", help="write mv_rgb.png and photo_front.png only (CPU)")
    args = p.parse_args(argv)
    if args.run_name == args.src_run:
        p.error("--run-name must differ from --src-run (the source run is read only)")
    sys.exit(1 if run(args) else 0)


if __name__ == "__main__":
    main()
