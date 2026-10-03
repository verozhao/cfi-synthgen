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


def blend(gen, photo, w, core, mode="detail", sigma=16.0):
    """gen, photo (res, res, 3) uint8, w (res, res) in [0, 1] -> uint8 view."""
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
    res = w[..., None] * out + (1.0 - w[..., None]) * g
    return np.clip(np.rint(res), 0, 255).astype(np.uint8)


def _gray(a):
    import cv2
    return cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY)


def _ncc(a, b, mask):
    a, b = a[mask].astype(np.float64), b[mask].astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def register_photo(gen0, mask0, photo, pmask, aff, res, method="homography", scale=2, ratio=0.75,
                   min_inliers=12, max_corner_shift=0.5):
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
    All work at scale * res and the result is area-downsampled, so small print is not aliased.
    """
    import cv2
    S = res * scale
    pre = np.ascontiguousarray(warp_into_view(photo, tuple(v * scale for v in aff), S)[..., :3])
    pre_m = warp_into_view(Image.fromarray(pmask.astype(np.uint8) * 255), tuple(v * scale for v in aff), S,
                           Image.BILINEAR).astype(np.float32) / 255.0
    info = {"method": "affine", "requested": method}
    g2 = cv2.resize(gen0, (S, S), interpolation=cv2.INTER_CUBIC)
    m2 = cv2.resize(mask0.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
    blur = lambda a: cv2.GaussianBlur(_gray(a), (0, 0), S / 128)       # noqa: E731
    region = m2 & (pre_m >= 0.5)
    info["ncc_affine"] = round(_ncc(blur(g2), blur(pre), region), 4)
    if method in ("homography", "flow"):
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
        if method == "flow" and info["method"] == "homography":
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
            if ncc_f >= info["ncc_homography"] - 0.01:
                pre, pre_m, info["method"] = warped, warped_m, "flow"
    info["ncc_final"] = info[f"ncc_{info['method']}"]
    ph = cv2.resize(pre, (res, res), interpolation=cv2.INTER_AREA)
    pm = cv2.resize(pre_m, (res, res), interpolation=cv2.INTER_AREA)
    return ph, pm, info


def front_from_photo(sku_dir, src_run_dir, cache_dir, mode="detail", sigma_frac=1 / 32,
                     feather_frac=1 / 128, facing=(0.35, 0.7), register="homography", min_ncc=0.6):
    """Blend the photo into view 0 of cache_dir/mv_rgb.png. Returns (new grid, debug panel, info).

    The photo is used only when it registers: blurred grey NCC with the generated front view of at
    least min_ncc after alignment. Below that (Shin Ramyun cup shot from above: 0.04 after the
    bbox fit) pasting it would put the wrong part of the photo on the front, so view 0 is kept.
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
    ph, pm, reg = register_photo(gen0, geo.masks[FRONT], photo, pmask, aff, res, register)
    w, core = front_weight(geo.masks[FRONT], geo.normals[FRONT], pm, res, facing, feather_frac)
    skipped = None
    if reg["ncc_final"] < min_ncc:
        skipped = f"photo not registered: NCC {reg['ncc_final']} < {min_ncc}, view 0 kept"
        w = np.zeros_like(w)
    sigma = max(1.0, res * sigma_frac)
    new0 = blend(gen0, ph, w, core, mode, sigma)
    out = put_view(grid.copy(), FRONT, new0, res)
    wv = np.repeat((w * 255).astype(np.uint8)[..., None], 3, axis=2)
    panel = np.concatenate([gen0, np.where(pm[..., None] >= 0.5, ph, 128).astype(np.uint8), wv, new0], axis=1)
    m0 = geo.masks[FRONT]
    info = {"view_res": res, "mode": mode, "sigma_px": round(sigma, 2), "feather_px": round(max(1.0, res * feather_frac), 2),
            "facing": list(facing), "photo_mask_source": mask_src, "align": align, "register": reg,
            "min_ncc": min_ncc, "skipped": skipped,
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
                                                     args.min_ncc)
                rec.update(info)
                shutil.copy(dst / "cache" / "mv_rgb.png", dst / "cache" / "mv_rgb_generated.png")
                Image.fromarray(grid).save(dst / "cache" / "mv_rgb.png")
                Image.fromarray(panel).save(dst / "photo_front.png")
                reg = info["register"]
                print(f"  [{sku}] ({i + 1}/{len(skus)}) aligned by {reg['method']} (IoU {info['align'].get('iou')}, "
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
    p.add_argument("--register", choices=("affine", "homography", "flow"), default="homography",
                   help="photo to view 0: bbox affine only, + SIFT homography (default), + optical flow")
    p.add_argument("--min-ncc", type=float, default=0.6,
                   help="use the photo only if it registers this well with the generated front (blurred grey NCC)")
    p.add_argument("--sigma-frac", type=float, default=1 / 32,
                   help="detail mode: blur sigma as a fraction of view_res (16 px at 512)")
    p.add_argument("--feather-frac", type=float, default=1 / 128, help="edge feather as a fraction of view_res")
    p.add_argument("--facing", type=float, nargs=2, default=(0.35, 0.7), metavar=("LO", "HI"),
                   help="cos to the front camera where the photo starts and fully takes over")
    p.add_argument("--seed", type=int, default=63)
    p.add_argument("--resume", action="store_true", help="skip SKUs with run_info.json")
    p.add_argument("--no-bake", action="store_true", help="write mv_rgb.png and photo_front.png only (CPU)")
    args = p.parse_args(argv)
    if args.run_name == args.src_run:
        p.error("--run-name must differ from --src-run (the source run is read only)")
    sys.exit(1 if run(args) else 0)


if __name__ == "__main__":
    main()
