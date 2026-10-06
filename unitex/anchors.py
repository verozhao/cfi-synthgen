"""
Inference-time text layout for GlyphAnchor: photo OCR -> per-view text items (text.json, "photo-lift")

Steps (lift / lift_sku):
  1. geometry of the untextured mesh in the six raw views, either mvgen.py views renders
     (view_XX_nocs.png / _normal.png / _alpha.png, normalized Blender frame) or UniTEX's cache
     grids (mv_ccm.png / mv_normal.png / mv_alpha.png, glTF frame p * 0.5 + 0.5, 2x3 f r t / b l d).
     Both writers truncate to 8 bits (2 / 255 world units = 2 px at 512 per bin), so the depth
     channel is de-quantized: half-bin offset, exact in-plane coordinates from the pixel centre,
     edge-preserving 5x5 smoothing of the depth only
  2. photo -> view 0: common.fit_box_affine of the photo foreground bbox onto the view-0
     silhouette bbox, then a coordinate search over the destination box (centre, size, +-10%)
     maximizing the IoU of the warped photo mask and the view-0 alpha
  3. per OCR item: map its quad into view 0, flag off_mesh when less than half of it lies on the
     silhouette (retailer badges beside the product), reparametrize its reading direction s by
     3D arc length on the surface (print is even on the product, the photo compresses it towards
     a can's limbs, and glyph patches are rendered evenly along s)
  4. per raw view k: push (s, t) samples from view 0 onto the surface and project them into view k
     with a visibility test (landing pixel inside view k's mask, depth within --depth-tol-bins
     8-bit bins of a 3x3 neighbour, geometric normal not facing away from camera k by more than
     --facing-tol) -> coverage (16 s-bins), grid {ns 16, nt 4, xy[it][is]}, quad of the visible
     s-range, height_px. Pull every masked pixel of view k back into view 0 with
     the same test and keep those inside the mapped quad -> pixels, bbox, cos. View 0 entries use
     the mapped quad itself. Other views need >= 4 pixels, coverage > 0 and cos >= 0.2 (edge-on
     slivers of flat faces and can walls seen from the top are dropped here).

cos is the mean |n . v| over the item's pixels in that view, n from the normal map (mvgen camera
normals or UniTEX world normals), or from central differences of the de-quantized CCM positions
(4 px baseline) when no normal map exists. The facing test always uses the CCM-gradient normals:
shading normals are vertex averages and can lean far over sharp edges of coarse meshes.

Geometry inputs
  mvgen    <python with bpy> mvgen.py --mode views --glb <eval>/<sku>/mesh.glb --views 0,1,2,3,4,5
           --with-geometry --albedo-samples 1 --out <dir>   (views mode yaw policy none = UniTEX frame)
  unitex   <eval>/<sku>/<run>/cache/ of a UniTEX run (run_unitex.py)

Usage:
  python -m unitex.anchors --eval-dir EVAL --sku 016000263192 --geometry mvgen \\
      --mesh-views /data/mesh_views/016000263192 --out text.json --debug-png anchors.png
  python -m unitex.anchors --eval-dir EVAL --skus all --geometry unitex --run-name unitex_s63 \\
      --out-dir /data/anchors --debug
"""

import argparse
import datetime
import json
import math
import os
import pathlib
import sys
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitex import common as C
from unitex.ocr import quad_bbox, quad_height

BIN = 2.0 / 255.0                  # world units per 8-bit CCM step
NS, NT = 16, 4                     # text.json grid
SUB_S, SUB_T = 4, 2                # push samples per grid cell (s, t)


@dataclass
class LiftConfig:
    depth_tol_bins: float = 3.0    # visibility: |depth - neighbour depth| in 8-bit CCM bins
    facing_tol: float = 0.3        # visibility also needs n . v_k > -facing_tol (thin pouches: the back
                                   # face is within depth_tol of the front, only the normal tells them apart)
    smooth_radius: int = 2         # depth de-quantization window (2 -> 5x5)
    cell_frac: float = 0.5         # a grid cell / s-bin is visible when this share of its samples is
    min_pixels: int = 4            # a view entry needs at least this many pulled pixels (views != 0)
    min_cos: float = 0.2           # views != 0: drop grazing entries (edge-on slivers), glyph filters again at 0.3
    min_on_mesh: float = 0.5       # below: off_mesh, no views
    partly_on_mesh: float = 0.9    # below: flag partly_off_mesh
    s_param: str = "arc"           # arc | photo
    min_conf: float = 0.5          # text-source ocr: lines below get the low_conf flag
    refine: bool = True
    refine_max: float = 0.1        # max change of the destination box (fraction of its size)

    def __post_init__(self):
        if self.s_param not in ("arc", "photo"):
            raise ValueError(f"s_param must be arc or photo, got {self.s_param}")


# ────────────────────────────────────────────────────────────────────────────
# Geometry: six raw views of the untextured mesh
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ViewGeometry:
    res: int
    masks: list                    # 6 x (res, res) bool
    depths: list                   # 6 x (res, res) de-quantized depth along the camera axis, NaN off mask
    filled: list                   # depths with the nearest masked value outside the mask
    normals: list                  # 6 x (res, res, 3) unit world normals (Blender frame), 0 off mask
    normal_source: str
    source: str
    path: str
    meta: dict
    images: list = None            # optional RGBA / RGB views for the debug overlay
    facing: list = None            # 6 x (res, res, 3) CCM-gradient (geometric) normals for the facing test


def decode_trunc(a):
    """uint8 written as trunc(255 * (v + 1) / 2) -> v in [-1, 1], centred in its bin."""
    return C.decode_ccm((np.asarray(a)[..., :3].astype(np.float64) + 0.5) / 255.0)


def camera_depth(pos, k):
    c2w = C.RAW_C2W[k]
    return -((np.asarray(pos, np.float64) - c2w[:3, 3]) @ c2w[:3, :3])[..., 2]


def pixel_centres(res):
    ys, xs = np.mgrid[0:res, 0:res].astype(np.float64) + 0.5
    return np.stack([xs, ys], axis=-1)


def unproject(xy, depth, k, res):
    """View-k pixel coordinates + depth -> Blender-frame points (inverse of common.project)."""
    c2w = C.RAW_C2W[k]
    half = C.ORTHO_SCALE / 2
    xy = np.asarray(xy, np.float64)
    q = np.stack([(2 * xy[..., 0] / res - 1) * half, (1 - 2 * xy[..., 1] / res) * half,
                  -np.asarray(depth, np.float64)], axis=-1)
    return q @ c2w[:3, :3].T + c2w[:3, 3]


def _shifted(a, dy, dx, fill=np.nan):
    """a shifted so that out[y, x] = a[y + dy, x + dx] (fill outside)."""
    H, W = a.shape[:2]
    out = np.full_like(a, fill)
    ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
    xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
    out[yd, xd] = a[ys, xs]
    return out


def smooth_depth(depth, mask, radius=2, max_diff=2.5 * BIN):
    """Edge-preserving box filter over masked neighbours within max_diff of the centre.

    Removes the 8-bit staircase on smooth surfaces, keeps depth steps between surfaces.
    """
    if radius <= 0:
        return np.where(mask, depth, np.nan)
    acc = np.zeros_like(depth)
    cnt = np.zeros_like(depth)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            n = _shifted(depth, dy, dx)
            ok = np.isfinite(n) & (np.abs(n - depth) <= max_diff)
            acc += np.where(ok, n, 0.0)
            cnt += ok
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), depth)
    return np.where(mask, out, np.nan)


def fill_nearest(depth, mask):
    """Depth with every off-mask pixel set to its nearest masked pixel's value (for interpolation)."""
    if not mask.any():
        return np.zeros_like(depth)
    try:
        from scipy.ndimage import distance_transform_edt
        iy, ix = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        return depth[iy, ix]
    except ImportError:
        out = np.where(mask, depth, np.nan)
        for _ in range(64):
            if np.isfinite(out).all():
                break
            best = out.copy()
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                n = _shifted(out, dy, dx)
                best = np.where(np.isnan(best), n, best)
            out = best
        return np.nan_to_num(out, nan=float(np.nanmean(depth[mask])))


def ccm_normals(filled, mask, k, res, step=2):
    """Unit world normals from central differences of the de-quantized positions, facing camera k."""
    P = unproject(pixel_centres(res), filled, k, res)
    dx = _shifted(P, 0, step) - _shifted(P, 0, -step)
    dy = _shifted(P, step, 0) - _shifted(P, -step, 0)
    n = np.cross(dx, dy)
    n = np.nan_to_num(n)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    v = C.view_dir(k)
    n *= np.where((n @ v) < 0, -1.0, 1.0)[..., None]
    bad = np.linalg.norm(n, axis=-1) < 0.5
    n[bad] = v
    return np.where(mask[..., None], n, 0.0)


def geometry_from_maps(positions, masks, normals=None, source="", path="", meta=None, images=None,
                       smooth_radius=2):
    """Per-view Blender-frame positions (CCM-decoded) + masks (+ world normals) -> ViewGeometry."""
    res = int(masks[0].shape[0])
    depths, filled = [], []
    for k in range(6):
        d = np.where(masks[k], camera_depth(positions[k], k), np.nan)
        d = smooth_depth(d, masks[k], smooth_radius)
        depths.append(d)
        filled.append(fill_nearest(d, masks[k]))
    facing = [ccm_normals(filled[k], masks[k], k, res) for k in range(6)]
    if normals is None:
        normals = facing
        nsrc = "ccm-gradient"
    else:
        normals = [np.where(masks[k][..., None],
                            normals[k] / np.maximum(np.linalg.norm(normals[k], axis=-1, keepdims=True), 1e-12), 0.0)
                   for k in range(6)]
        nsrc = "normal-map"
    return ViewGeometry(res=res, masks=list(masks), depths=depths, filled=filled, normals=normals,
                        normal_source=nsrc, source=source, path=str(path), meta=meta or {}, images=images,
                        facing=facing)


def load_mvgen_views(d, use_normals=True, smooth_radius=2):
    """mvgen.py --mode views --with-geometry output dir -> ViewGeometry (Blender frame)."""
    d = pathlib.Path(d)
    meta = {}
    if (d / "views.json").exists():
        with open(d / "views.json") as f:
            meta = json.load(f)
    use_normals = use_normals and all((d / f"view_{k:02d}_normal.png").exists() for k in range(6))
    pos, masks, normals, images = [], [], [], []
    for k in range(6):
        p = d / f"view_{k:02d}_nocs.png"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: render all six views with mvgen.py --mode views "
                                    f"--views 0,1,2,3,4,5 --with-geometry")
        nocs = np.asarray(Image.open(p).convert("RGBA"))
        ap = d / f"view_{k:02d}_alpha.png"
        masks.append((np.asarray(Image.open(ap).convert("L")) > 127) if ap.exists() else nocs[..., 3] > 127)
        pos.append(decode_trunc(nocs))
        if use_normals:
            n_cam = decode_trunc(np.asarray(Image.open(d / f"view_{k:02d}_normal.png").convert("RGB")))
            normals.append(n_cam @ C.RAW_C2W[k][:3, :3].T)      # camera -> world: n_w = R n_c
        ip = d / f"view_{k:02d}.png"
        images.append(np.asarray(Image.open(ip).convert("RGBA")) if ip.exists() else None)
    if meta.get("yaw_deg"):
        meta["warning"] = (f"views were rendered with yaw {meta['yaw_deg']} deg, but UniTEX does not yaw the mesh, "
                           f"so this layout only matches views rendered the same way")
    return geometry_from_maps(pos, masks, normals if use_normals else None, "mvgen", d, meta, images,
                              smooth_radius)


def load_unitex_cache(d, use_normals=True, smooth_radius=2):
    """UniTEX cache dir (mv_alpha.png, mv_ccm.png, mv_normal.png, glTF frame) -> ViewGeometry."""
    d = pathlib.Path(d)
    for name in ("mv_alpha.png", "mv_ccm.png"):
        if not (d / name).exists():
            raise FileNotFoundError(f"{d / name} missing (UniTEX render_geometry_images output)")
    alpha = np.asarray(Image.open(d / "mv_alpha.png").convert("L"))
    res = alpha.shape[0] // 2
    masks = [a > 127 for a in C.split_grid(alpha, res)]
    ccm = C.split_grid(np.asarray(Image.open(d / "mv_ccm.png").convert("RGB")), res)
    pos = [C.gltf_to_blender(decode_trunc(v)) for v in ccm]
    normals = None
    if use_normals and (d / "mv_normal.png").exists():
        nv = C.split_grid(np.asarray(Image.open(d / "mv_normal.png").convert("RGB")), res)
        normals = [C.gltf_to_blender(decode_trunc(v)) for v in nv]
    images = None
    if (d / "mv_rgb.png").exists():
        images = [np.ascontiguousarray(v) for v in C.split_grid(np.asarray(Image.open(d / "mv_rgb.png").convert("RGB")), res)]
    return geometry_from_maps(pos, masks, normals, "unitex", d, {"res": res}, images, smooth_radius)


# ────────────────────────────────────────────────────────────────────────────
# Sampling and visibility
# ────────────────────────────────────────────────────────────────────────────

def _floor_px(xy, res):
    ix = np.floor(xy[..., 0]).astype(np.int64)
    iy = np.floor(xy[..., 1]).astype(np.int64)
    inb = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)
    return np.clip(ix, 0, res - 1), np.clip(iy, 0, res - 1), inb


def sample_mask(mask, xy):
    ix, iy, inb = _floor_px(np.asarray(xy, np.float64), mask.shape[0])
    return inb & mask[iy, ix]


def sample_bilinear(a, xy):
    """Bilinear sample of a (H, W) array at continuous pixel coordinates (centres at +0.5)."""
    H, W = a.shape[:2]
    xy = np.asarray(xy, np.float64)
    u = np.clip(xy[..., 0] - 0.5, 0, W - 1)
    v = np.clip(xy[..., 1] - 0.5, 0, H - 1)
    x0 = np.minimum(np.floor(u).astype(np.int64), W - 2 if W > 1 else 0)
    y0 = np.minimum(np.floor(v).astype(np.int64), H - 2 if H > 1 else 0)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = u - x0, v - y0
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x1] * fx * (1 - fy)
            + a[y1, x0] * (1 - fx) * fy + a[y1, x1] * fx * fy)


def surface_points(xy0, geo):
    """View-0 pixel coordinates -> surface points (Blender frame) through the filled view-0 depth."""
    return unproject(xy0, sample_bilinear(geo.filled[0], xy0), 0, geo.res)


def visible_in(P, k, geo, tol):
    """(visible bool, view-k xy) for Blender-frame points: landing pixel masked in view k and the
    point's depth within tol of a masked pixel in the 3x3 window around it."""
    P = np.asarray(P, np.float64)
    xy, d = C.project(P, k, geo.res)
    ix, iy, inb = _floor_px(xy, geo.res)
    ok = inb & np.isfinite(d) & geo.masks[k][iy, ix]
    D = geo.depths[k]
    best = np.full(d.shape, np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            jx, jy = np.clip(ix + dx, 0, geo.res - 1), np.clip(iy + dy, 0, geo.res - 1)
            best = np.fmin(best, np.abs(D[jy, jx] - d))
    return ok & (best <= tol), xy


# ────────────────────────────────────────────────────────────────────────────
# Quads and the reading frame
# ────────────────────────────────────────────────────────────────────────────

def bilinear(quad, s, t):
    """Reading-frame (s, t) -> point of the quad TL, TR, BR, BL (broadcasts s and t)."""
    tl, tr, br, bl = [np.asarray(p, np.float64) for p in quad]
    s = np.asarray(s, np.float64)[..., None]
    t = np.asarray(t, np.float64)[..., None]
    return (1 - t) * ((1 - s) * tl + s * tr) + t * ((1 - s) * bl + s * br)


def inverse_bilinear(xy, quad, iters=8):
    """Points -> reading-frame (s, t) in the quad (Newton from the parallelogram solution)."""
    tl, tr, br, bl = [np.asarray(p, np.float64) for p in quad]
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    a, b, c = tr - tl, bl - tl, tl - tr + br - bl
    det0 = a[0] * b[1] - a[1] * b[0]
    if abs(det0) < 1e-12:
        return np.full(len(xy), np.nan), np.full(len(xy), np.nan)
    r = xy - tl
    s = (r[:, 0] * b[1] - r[:, 1] * b[0]) / det0
    t = (a[0] * r[:, 1] - a[1] * r[:, 0]) / det0
    for _ in range(iters):
        f = tl + s[:, None] * a + t[:, None] * b + (s * t)[:, None] * c - xy
        js = a + t[:, None] * c
        jt = b + s[:, None] * c
        det = js[:, 0] * jt[:, 1] - js[:, 1] * jt[:, 0]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        s = s - (f[:, 0] * jt[:, 1] - f[:, 1] * jt[:, 0]) / det
        t = t - (js[:, 0] * f[:, 1] - js[:, 1] * f[:, 0]) / det
    return s, t


def quad_orientation(quad):
    """Reading direction angle in degrees (clockwise, y down) and its 90-degree snap."""
    tl, tr, br, bl = [np.asarray(p, np.float64) for p in quad]
    d = (tr - tl) + (br - bl)
    ang = math.degrees(math.atan2(d[1], d[0]))
    return ang, int(round(ang / 90.0)) % 4 * 90


def arc_table(q0, geo, n=48, max_ratio=8.0):
    """Monotone table (s_photo, s_arc): arc length along the surface under the view-0 quad.

    Three lines (t = 0.25, 0.5, 0.75) are lifted through the view-0 depth. Each segment's 3D /
    in-plane length ratio is clipped to [1, max_ratio] (depth steps between surfaces would
    otherwise dominate), off-mesh segments take the line's median ratio.
    """
    s = np.linspace(0.0, 1.0, n + 1)
    lines = []
    for t in (0.25, 0.5, 0.75):
        xy = bilinear(q0, s, np.full_like(s, t))
        on = sample_mask(geo.masks[0], xy)
        P = surface_points(xy, geo)
        seg2 = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * (C.ORTHO_SCALE / geo.res)
        seg3 = np.linalg.norm(np.diff(P, axis=0), axis=-1)
        ok = on[1:] & on[:-1] & (seg2 > 1e-12)
        ratio = np.where(ok, np.clip(seg3 / np.maximum(seg2, 1e-12), 1.0, max_ratio), np.nan)
        med = np.nanmedian(ratio) if np.isfinite(ratio).any() else 1.0
        lines.append(seg2 * np.where(np.isfinite(ratio), ratio, med))
    seg = np.mean(lines, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return s, s.copy()
    return s, cum / cum[-1]


# ────────────────────────────────────────────────────────────────────────────
# Photo -> view 0
# ────────────────────────────────────────────────────────────────────────────

def warp_mask(photo_mask, aff, res):
    """Photo mask (H, W) resampled into view 0 through the axis-aligned affine (nearest)."""
    sx, sy, tx, ty = aff
    H, W = photo_mask.shape
    xs = np.floor((np.arange(res) + 0.5 - tx) / sx).astype(np.int64)
    ys = np.floor((np.arange(res) + 0.5 - ty) / sy).astype(np.int64)
    vx, vy = (xs >= 0) & (xs < W), (ys >= 0) & (ys < H)
    out = np.zeros((res, res), bool)
    out[np.ix_(vy, vx)] = photo_mask[np.ix_(ys[vy], xs[vx])]
    return out


def warp_photo(photo, aff, res, fill=(0, 0, 0)):
    """Photo (PIL or array) resampled into view 0 (nearest), for the debug overlay."""
    a = np.asarray(photo.convert("RGB") if isinstance(photo, Image.Image) else photo)[..., :3]
    sx, sy, tx, ty = aff
    H, W = a.shape[:2]
    xs = np.floor((np.arange(res) + 0.5 - tx) / sx).astype(np.int64)
    ys = np.floor((np.arange(res) + 0.5 - ty) / sy).astype(np.int64)
    vx, vy = (xs >= 0) & (xs < W), (ys >= 0) & (ys < H)
    out = np.empty((res, res, 3), np.uint8)
    out[:] = fill
    out[np.ix_(vy, vx)] = a[np.ix_(ys[vy], xs[vx])]
    return out


def mask_iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def fit_photo_to_view0(photo_mask, fg_box, mask0, refine=True, max_frac=0.1,
                       steps=(0.02, 0.01, 0.005, 0.0025)):
    """Photo foreground bbox -> view-0 silhouette bbox, refined for silhouette IoU.

    Coordinate descent over the destination box (centre x, centre y, width, height), each within
    +-max_frac of the silhouette bbox. Returns (affine (sx, sy, tx, ty), info).
    """
    res = mask0.shape[0]
    dst0 = C.bbox_of_mask(mask0)
    if dst0 is None:
        raise ValueError("view 0 silhouette is empty")
    aff0 = C.fit_box_affine(fg_box, dst0)
    iou0 = mask_iou(warp_mask(photo_mask, aff0, res), mask0) if photo_mask is not None else None
    info = {"fg_box_photo": [round(float(v), 2) for v in fg_box], "view0_box": dst0,
            "iou_bbox": None if iou0 is None else round(iou0, 4), "method": "bbox"}
    if not refine or photo_mask is None:
        info["iou"] = info["iou_bbox"]
        return aff0, info

    def box_of(c):
        return [c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2]

    def score(c):
        return mask_iou(warp_mask(photo_mask, C.fit_box_affine(fg_box, box_of(c)), res), mask0)

    bw, bh = dst0[2] - dst0[0], dst0[3] - dst0[1]
    c0 = np.array([(dst0[0] + dst0[2]) / 2, (dst0[1] + dst0[3]) / 2, bw, bh])
    scale = np.array([bw, bh, bw, bh])
    cur, best, n_eval = c0.copy(), iou0, 1
    for step in steps:
        for _ in range(30):
            improved = False
            for i in range(4):
                for sgn in (-1.0, 1.0):
                    c2 = cur.copy()
                    c2[i] += sgn * step * scale[i]
                    if abs(c2[i] - c0[i]) > max_frac * scale[i] + 1e-9:
                        continue
                    sc = score(c2)
                    n_eval += 1
                    if sc > best + 1e-6:
                        best, cur, improved = sc, c2, True
            if not improved:
                break
    info.update(method="iou-refined" if best > iou0 else "bbox", iou=round(best, 4), n_eval=n_eval,
                box=[round(v, 2) for v in box_of(cur)],
                delta_frac=[round(float(v), 4) for v in (cur - c0) / scale])
    return C.fit_box_affine(fg_box, box_of(cur)), info


# ────────────────────────────────────────────────────────────────────────────
# Lift
# ────────────────────────────────────────────────────────────────────────────

def _r(v, n=2):
    return round(float(v), n)


def _bbox_px(ixy):
    return [float(ixy[:, 0].min()), float(ixy[:, 1].min()), float(ixy[:, 0].max() + 1), float(ixy[:, 1].max() + 1)]


def _grid_json(G, vis):
    xy = [[[_r(G[it, i, 0]), _r(G[it, i, 1])] if vis[it, i] else None for i in range(G.shape[1])]
          for it in range(G.shape[0])]
    return {"ns": int(G.shape[1]), "nt": int(G.shape[0]), "xy": xy}


class _Pull:
    """Masked pixels of views 1..5 whose surface point is visible in view 0, with their view-0 xy."""

    def __init__(self, geo, tol, facing_tol=0.3):
        self.views = {}
        C0 = pixel_centres(geo.res)
        for k in range(1, 6):
            m = geo.masks[k]
            iy, ix = np.nonzero(m)
            xy = C0[iy, ix]
            P = unproject(xy, geo.depths[k][iy, ix], k, geo.res)
            vis0, xy0 = visible_in(P, 0, geo, tol)
            vis0 &= (geo.facing[k][iy, ix] @ C.view_dir(0)) > -facing_tol
            self.views[k] = {"ixy": np.stack([ix, iy], -1)[vis0], "xy0": xy0[vis0],
                             "n": geo.normals[k][iy, ix][vis0]}

    def members(self, k, q0):
        v = self.views[k]
        if len(v["xy0"]) == 0:
            return np.zeros(0, bool)
        x0, y0, x1, y1 = quad_bbox(q0)
        sel = ((v["xy0"][:, 0] >= x0 - 1) & (v["xy0"][:, 0] <= x1 + 1)
               & (v["xy0"][:, 1] >= y0 - 1) & (v["xy0"][:, 1] <= y1 + 1))
        out = np.zeros(len(sel), bool)
        if sel.any():
            s, t = inverse_bilinear(v["xy0"][sel], q0)
            out[sel] = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
        return out


def _view0_pixels(q0, geo):
    """View-0 pixel indices inside the quad and the mask."""
    x0, y0, x1, y1 = quad_bbox(q0)
    xa, xb = max(int(math.floor(x0)), 0), min(int(math.ceil(x1)), geo.res)
    ya, yb = max(int(math.floor(y0)), 0), min(int(math.ceil(y1)), geo.res)
    if xa >= xb or ya >= yb:
        return np.zeros((0, 2), np.int64)
    ys, xs = np.mgrid[ya:yb, xa:xb]
    s, t = inverse_bilinear(np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], -1), q0)
    inside = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    ixy = np.stack([xs.ravel(), ys.ravel()], -1)[inside]
    return ixy[geo.masks[0][ixy[:, 1], ixy[:, 0]]]


def lift_quad(q0, geo, cfg, pull):
    """Mapped view-0 quad -> (views dict, info) with text.json view entries keyed by raw index."""
    tol = cfg.depth_tol_bins * BIN
    ns_s, nt_s = NS * SUB_S, NT * SUB_T
    s_arc = (np.arange(ns_s) + 0.5) / ns_s
    t_smp = (np.arange(nt_s) + 0.5) / nt_s
    if cfg.s_param == "arc":
        tab_p, tab_a = arc_table(q0, geo)
    else:
        tab_p = tab_a = np.linspace(0, 1, 2)

    def to_photo(sa):
        return np.interp(sa, tab_a, tab_p)

    sp = to_photo(s_arc)
    S, T = np.meshgrid(sp, t_smp)                       # (nt_s, ns_s)
    xy0 = bilinear(q0, S, T)
    on0 = sample_mask(geo.masks[0], xy0)
    P = surface_points(xy0, geo)
    ix0, iy0, _ = _floor_px(xy0, geo.res)
    n0 = geo.facing[0][iy0, ix0]
    sc = to_photo((np.arange(NS) + 0.5) / NS)
    Sc, Tc = np.meshgrid(sc, (np.arange(NT) + 0.5) / NT)
    xyc0 = bilinear(q0, Sc, Tc)
    Pc = surface_points(xyc0, geo)
    Pe = [surface_points(bilinear(q0, sp, np.full_like(sp, t)), geo) for t in (0.0, 1.0)]
    info = {"on_mesh": _r(on0.mean(), 4),
            "s_arc_at_photo_mid": _r(np.interp(0.5, tab_p, tab_a), 4)}

    views = {}
    for k in range(6):
        if k == 0:
            vis = on0
        else:
            vis, _ = visible_in(P, k, geo, tol)
            vis &= on0 & ((n0 @ C.view_dir(k)) > -cfg.facing_tol)
        cell = vis.reshape(NT, SUB_T, NS, SUB_S).mean(axis=(1, 3))
        cell_vis = cell >= cfg.cell_frac
        bins = vis.reshape(nt_s, NS, SUB_S).mean(axis=(0, 2))
        coverage = float((bins >= cfg.cell_frac).mean())
        if not cell_vis.any():
            continue
        cols = np.nonzero(vis.any(axis=0))[0]
        s_lo, s_hi = cols.min() / ns_s, (cols.max() + 1) / ns_s
        if k == 0:
            G = xyc0
            ixy = _view0_pixels(q0, geo)
            n = geo.normals[0][ixy[:, 1], ixy[:, 0]] if len(ixy) else np.zeros((0, 3))
            quad = np.asarray(q0, np.float64)
            bbox = quad_bbox(quad)
            height = quad_height(quad)
        else:
            G, _ = C.project(Pc, k, geo.res)
            mem = pull.members(k, q0)
            if mem.sum() < cfg.min_pixels or coverage <= 0:
                continue
            ixy = pull.views[k]["ixy"][mem]
            n = pull.views[k]["n"][mem]
            if float(np.abs(n @ C.view_dir(k)).mean()) < cfg.min_cos:
                info["n_grazing_dropped"] = info.get("n_grazing_dropped", 0) + 1
                continue
            corners = []
            for sa, t in ((s_lo, 0.0), (s_hi, 0.0), (s_hi, 1.0), (s_lo, 1.0)):
                Pq = surface_points(bilinear(q0, to_photo(sa), t), geo)
                corners.append(C.project(Pq, k, geo.res)[0])
            quad = np.asarray(corners)
            bbox = _bbox_px(ixy)
            e0, _ = C.project(Pe[0], k, geo.res)
            e1, _ = C.project(Pe[1], k, geo.res)
            colv = vis.any(axis=0)
            height = float(np.linalg.norm(e1 - e0, axis=-1)[colv].mean())
        cos = float(np.abs(n @ C.view_dir(k)).mean()) if len(n) else float("nan")
        views[str(k)] = {
            "bbox": [_r(v) for v in bbox],
            "quad": [[_r(x), _r(y)] for x, y in quad],
            "pixels": int(len(ixy)),
            "coverage": _r(coverage, 4),
            "cos": None if not np.isfinite(cos) else _r(cos, 4),
            "height_px": _r(height),
            "grid": _grid_json(G, cell_vis),
            "s_range": [_r(s_lo, 4), _r(s_hi, 4)],
        }
    return views, info


def lift(items, photo_mask, fg_box, geo, cfg=None, photo_size=None):
    """OCR items (photo px quads) + photo foreground + mesh views -> text.json dict (photo-lift).

    items: [{"text", "conf", "quad" [[x, y] x 4] TL, TR, BR, BL in reading order, optional "id",
    "flags"}]. photo_mask: bool (H, W) photo foreground for the IoU refinement (None: bbox fit only).
    """
    cfg = cfg or LiftConfig()
    aff, align = fit_photo_to_view0(photo_mask, fg_box, geo.masks[0], cfg.refine, cfg.refine_max)
    pull = _Pull(geo, cfg.depth_tol_bins * BIN, cfg.facing_tol)
    out_items = []
    for k, it in enumerate(items):
        q = np.asarray(it["quad"], np.float64).reshape(4, 2)
        flags = list(it.get("flags") or [])
        conf = float(it.get("conf", 1.0))
        if conf < cfg.min_conf and "low_conf" not in flags:
            flags.append("low_conf")
        ang, snap = quad_orientation(q)
        if snap != 0:
            flags.append("rotated")
        q0 = C.apply_affine(q, aff)
        rec = {"id": len(out_items), "text": " ".join(str(it["text"]).split()), "conf": _r(conf, 4),
               "provenance": "photo", "flags": flags, "angle_deg": _r(ang, 1),
               "src_quad": [[_r(x), _r(y)] for x, y in q], "src_id": it.get("id", k),
               "view0_quad": [[_r(x), _r(y)] for x, y in q0]}
        if quad_height(q0) < 0.5 or np.linalg.norm(q0[1] - q0[0]) < 0.5:
            rec.update(flags=flags + ["degenerate"], on_mesh=0.0, views={})
            out_items.append(rec)
            continue
        views, info = lift_quad(q0, geo, cfg, pull)
        if info["on_mesh"] < cfg.min_on_mesh:
            flags.append("off_mesh")
            views = {}
        elif info["on_mesh"] < cfg.partly_on_mesh:
            flags.append("partly_off_mesh")
        rec.update(flags=flags, views=views, **info)
        out_items.append(rec)
    per_view = {str(k): sum(str(k) in it["views"] for it in out_items) for k in range(6)}
    doc = {
        "version": 1,
        "res": geo.res,
        "source": "photo-lift",
        "items": out_items,
        "lift": {
            "geometry": geo.source, "geometry_path": geo.path, "normal_source": geo.normal_source,
            "geometry_meta_warning": geo.meta.get("warning"),
            "photo_size": list(photo_size) if photo_size else None,
            "affine_photo_to_view0": [round(float(v), 6) for v in aff], "align": align,
            "config": asdict(cfg), "n_items": len(out_items),
            "n_off_mesh": sum("off_mesh" in it["flags"] for it in out_items),
            "items_per_view": per_view,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    }
    return doc


def rescale_doc(doc, res):
    """text.json pixel coordinates -> another per-view resolution (pixels scale with area)."""
    f = res / float(doc["res"])
    out = json.loads(json.dumps(doc))
    out["res"] = int(res)
    for it in out["items"]:
        if "view0_quad" in it:
            it["view0_quad"] = [[_r(x * f), _r(y * f)] for x, y in it["view0_quad"]]
        for v in it["views"].values():
            v["bbox"] = [_r(b * f) for b in v["bbox"]]
            v["quad"] = [[_r(x * f), _r(y * f)] for x, y in v["quad"]]
            v["pixels"] = int(round(v["pixels"] * f * f))
            v["height_px"] = _r(v["height_px"] * f)
            v["grid"]["xy"] = [[None if p is None else [_r(p[0] * f), _r(p[1] * f)] for p in row]
                               for row in v["grid"]["xy"]]
    return out


# ────────────────────────────────────────────────────────────────────────────
# Eval-dir inputs
# ────────────────────────────────────────────────────────────────────────────

def load_photo(sku_dir, meta):
    """Original photo (ref.png minus its square padding) as RGB PIL."""
    ref = Image.open(pathlib.Path(sku_dir) / "ref.png").convert("RGB")
    px, py = meta.get("pad", [0, 0])
    W0, H0 = meta.get("orig_size", ref.size)
    return ref.crop((px, py, px + W0, py + H0))


def photo_foreground(photo, meta, run_dir=None):
    """(mask, fg_box, source) in photo pixels.

    RMBG mask saved by run_unitex.py (rmbg_mask_1024.png, alpha >= 128) when present, else the
    border-colour threshold of prepare_eval.py with enclosed holes filled, and the fg box from
    ref_meta.json fg_bbox_photo in that case (prepare_eval's robust bbox).
    """
    from unitex import prepare_eval as pe
    if run_dir is not None and (pathlib.Path(run_dir) / "rmbg_mask_1024.png").exists():
        m = Image.open(pathlib.Path(run_dir) / "rmbg_mask_1024.png").convert("L")
        S = int(meta.get("square_size", max(photo.size)))
        px, py = meta.get("pad", [0, 0])
        m = np.asarray(m.resize((S, S), Image.BILINEAR))[py:py + photo.height, px:px + photo.width] >= 128
        box = C.bbox_of_mask(m)
        if box is not None:
            return m, box, "rmbg_mask_1024"
    a = np.asarray(photo)
    bg = tuple(meta["bg_color"]) if meta.get("bg_color") else None
    mask = pe.fill_holes(pe.fg_mask_border(a, bg))
    box = meta.get("fg_bbox_photo") or pe.robust_bbox(mask)
    return mask, [float(v) for v in box], "border threshold (prepare_eval)"


def parse_ocr_json(path):
    """unitex.ocr output (item list, {"lines"/"items": [...]}, or the CLI's {"results": {img: [...]}})."""
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        if "results" in d:
            d = next(iter(d["results"].values()))
        else:
            d = d.get("lines") or d.get("items") or []
    return [{"id": it.get("id", i), "text": it["text"], "conf": float(it.get("conf", 1.0)), "quad": it["quad"]}
            for i, it in enumerate(d) if it.get("quad") is not None and str(it.get("text", "")).strip()]


def load_text_items(sku_dir, text_source="gt", ocr_json=None, min_conf=0.5):
    """Photo text items for one SKU.

    gt   prepare_eval.load_gt: the manual gt_text.txt when its status is manual, else the gt_ocr.json
         lines with conf >= min_conf. conf is the lowest conf of the row's OCR lines. Rows without
         a box ("-" in a manual transcript) are skipped, they have no quad to lift
    ocr  every gt_ocr.json line, low_conf flagged by lift()
    """
    sku_dir = pathlib.Path(sku_dir)
    if ocr_json:
        return parse_ocr_json(ocr_json), f"ocr-json {ocr_json}"
    with open(sku_dir / "gt_ocr.json") as f:
        lines = json.load(f)["lines"]
    if text_source == "ocr":
        return [{"id": ln.get("id", i), "text": ln["text"], "conf": ln["conf"], "quad": ln["quad"]}
                for i, ln in enumerate(lines) if str(ln["text"]).strip()], "gt_ocr.json"
    from unitex import prepare_eval as pe
    gt, src, _ = pe.load_gt(sku_dir, min_conf)
    out = []
    for g in gt:
        if g["quad"] is None:
            continue
        confs = [lines[i]["conf"] for i in g.get("ocr_ids", []) if 0 <= i < len(lines)]
        out.append({"id": g["id"], "text": g["text"], "conf": min(confs) if confs else 1.0, "quad": g["quad"]})
    return out, f"gt ({src})"


def _fmt(pattern, eval_dir, sku, run):
    return pattern.format(eval=eval_dir, sku=sku, run=run) if pattern else None


def load_geometry(kind, mesh_views=None, unitex_cache=None, smooth_radius=2):
    if kind == "mvgen":
        return load_mvgen_views(mesh_views, smooth_radius=smooth_radius)
    if kind == "unitex":
        return load_unitex_cache(unitex_cache, smooth_radius=smooth_radius)
    raise ValueError(f"geometry must be mvgen or unitex, got {kind}")


def lift_sku(eval_dir, sku, geometry="mvgen", run_name=None, mesh_views=None, unitex_cache=None,
             text_source="gt", ocr_json=None, cfg=None, debug_png=None):
    """One eval-dir SKU -> text.json dict. mesh_views / unitex_cache / ocr_json accept {eval} {sku} {run}."""
    cfg = cfg or LiftConfig()
    eval_dir = pathlib.Path(eval_dir)
    sku_dir = eval_dir / sku
    with open(sku_dir / "ref_meta.json") as f:
        meta = json.load(f)
    run_dir = sku_dir / run_name if run_name else None
    mv = _fmt(mesh_views or "{eval}/{sku}/mesh_views", eval_dir, sku, run_name)
    uc = _fmt(unitex_cache or "{eval}/{sku}/{run}/cache", eval_dir, sku, run_name or "unitex")
    geo = load_geometry(geometry, mv, uc, cfg.smooth_radius)
    photo = load_photo(sku_dir, meta)
    mask, fg_box, mask_src = photo_foreground(photo, meta, run_dir)
    items, text_src = load_text_items(sku_dir, text_source, _fmt(ocr_json, eval_dir, sku, run_name), cfg.min_conf)
    doc = lift(items, mask, fg_box, geo, cfg, photo.size)
    doc["lift"].update(sku=sku, photo=str(sku_dir / "ref.png"), photo_pad=meta.get("pad"),
                       photo_mask_source=mask_src, text_source=text_src)
    if debug_png:
        draw_debug(doc, geo, photo, mask, debug_png)
    return doc


# ────────────────────────────────────────────────────────────────────────────
# Debug overlay
# ────────────────────────────────────────────────────────────────────────────

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
           (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180)]


def _font(size=13):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _edge(mask):
    m = np.asarray(mask, bool)
    inner = m.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        inner &= _shifted(m.astype(np.float64), dy, dx, 0.0) > 0.5
    return m & ~inner


def _view_background(geo, k):
    m = geo.masks[k]
    img = geo.images[k] if geo.images else None
    if img is not None:
        rgb = np.asarray(img)[..., :3].astype(np.float64)
        if m.any() and rgb[m].std() > 6:
            out = np.where(m[..., None], rgb, 40.0)
            return out.astype(np.uint8)
    v = C.view_dir(k)
    up = np.array([0.0, 0.0, 1.0]) if k not in (4, 5) else np.array([0.0, 1.0, 0.0])
    light = v + 0.6 * up
    light /= np.linalg.norm(light)
    sh = 0.3 + 0.7 * np.clip(geo.normals[k] @ light, 0, 1)
    g = np.where(m, 60 + 150 * sh, 40.0)
    return np.repeat(g[..., None], 3, axis=-1).astype(np.uint8)


def draw_debug(doc, geo, photo, photo_mask, path, panel=512):
    """2 x 4 panels: photo with OCR quads | view 0 (photo blended in) | raw views 1..5 | legend."""
    res = geo.res
    font = _font(13)
    aff = doc["lift"]["affine_photo_to_view0"]
    canvas = Image.new("RGB", (4 * panel, 2 * panel), (25, 25, 25))
    f = panel / max(photo.size)
    ph = photo.convert("RGB").resize((max(1, round(photo.width * f)), max(1, round(photo.height * f))), Image.LANCZOS)
    canvas.paste(ph, (0, 0))
    d = ImageDraw.Draw(canvas)
    for it in doc["items"]:
        col = PALETTE[it["id"] % len(PALETTE)]
        q = [(x * f, y * f) for x, y in it["src_quad"]]
        off = "off_mesh" in it["flags"]
        d.polygon(q, outline=(255, 0, 0) if off else col, width=3 if off else 2)
        d.text((q[0][0], max(0, q[0][1] - 14)), f"{it['id']}{'x' if off else ''}", fill=(255, 0, 0) if off else col,
               font=font)
    d.text((6, panel - 18), "photo + OCR quads (red = off_mesh)", fill=(255, 255, 255), font=font)

    slots = {0: (1, 0), 1: (2, 0), 2: (3, 0), 3: (0, 1), 4: (1, 1), 5: (2, 1)}
    for k in range(6):
        bg = _view_background(geo, k).astype(np.float64)
        if k == 0:
            wp = warp_photo(photo, aff, res).astype(np.float64)
            m = geo.masks[0][..., None]
            bg = np.where(m, 0.45 * bg + 0.55 * wp, 0.5 * bg + 0.5 * wp)
            bg[_edge(geo.masks[0])] = (255, 255, 0)
            if photo_mask is not None:
                bg[_edge(warp_mask(photo_mask, aff, res))] = (0, 255, 255)
        tile = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
        s = panel / res
        if s != 1:
            tile = tile.resize((panel, panel), Image.NEAREST)
        td = ImageDraw.Draw(tile)
        for it in doc["items"]:
            v = it["views"].get(str(k))
            if not v:
                continue
            col = PALETTE[it["id"] % len(PALETTE)]
            q = [(x * s, y * s) for x, y in v["quad"]]
            td.polygon(q, outline=col, width=2)
            td.line([q[0], q[3]], fill=(255, 255, 255), width=2)          # start edge of the reading frame
            x0, y0, x1, y1 = [b * s for b in v["bbox"]]
            td.rectangle([x0, y0, x1, y1], outline=tuple(int(c * 0.6) for c in col), width=1)
            for row in v["grid"]["xy"]:
                for p in row:
                    if p is not None:
                        td.ellipse([p[0] * s - 1.5, p[1] * s - 1.5, p[0] * s + 1.5, p[1] * s + 1.5], fill=col)
            td.text((q[0][0], max(0, q[0][1] - 14)), str(it["id"]), fill=col, font=font)
        name = C.RAW_VIEW_NAMES[k]
        n = sum(str(k) in it["views"] for it in doc["items"])
        td.text((6, 6), f"raw {k} {name}: {n} items", fill=(255, 255, 255), font=font)
        cx, cy = slots[k]
        canvas.paste(tile, (cx * panel, cy * panel))

    L = doc["lift"]
    al = L["align"]
    lines = [f"{L.get('sku', '')}  geometry {L['geometry']} ({L['normal_source']})",
             f"silhouette IoU bbox {al.get('iou_bbox')} -> {al.get('iou')} ({al.get('method')})",
             f"items {L['n_items']}, off_mesh {L['n_off_mesh']}, per view {L['items_per_view']}",
             "yellow: mesh silhouette, cyan: photo mask", ""]
    for it in doc["items"][:40]:
        vs = ",".join(sorted(it["views"]))
        lines.append(f"{it['id']:>2} [{vs or '-'}] {it['text'][:38]}" + (f"  {','.join(it['flags'])}" if it["flags"] else ""))
    y = panel + 8
    for i, ln in enumerate(lines):
        col = (255, 255, 255) if i < 5 else PALETTE[doc["items"][i - 5]["id"] % len(PALETTE)]
        d.text((3 * panel + 8, y), ln, fill=col, font=font)
        y += 16
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def _sku_list(eval_dir, spec):
    if spec == "all":
        spec = str(pathlib.Path(eval_dir) / "skus.txt")
    from unitex.prepare_eval import read_sku_list
    return read_sku_list(spec)


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description="Lift photo OCR onto the six UniTEX views (text.json, photo-lift).")
    p.add_argument("--eval-dir", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sku")
    g.add_argument("--skus", help="batch: comma list, file, or 'all' (<eval>/skus.txt)")
    p.add_argument("--geometry", choices=("mvgen", "unitex"), default="mvgen")
    p.add_argument("--run-name", default=None,
                   help="UniTEX run under <eval>/<sku>/: its cache for --geometry unitex, its rmbg_mask_1024.png "
                        "for the photo mask")
    p.add_argument("--mesh-views", default=None,
                   help="mvgen views dir ({eval} {sku} {run} expand), default {eval}/{sku}/mesh_views")
    p.add_argument("--unitex-cache", default=None, help="default {eval}/{sku}/{run}/cache")
    p.add_argument("--text-source", choices=("gt", "ocr"), default="gt")
    p.add_argument("--ocr-json", default=None, help="unitex.ocr output to lift instead ({sku} expands)")
    p.add_argument("--out", default=None, help="single SKU: text.json path")
    p.add_argument("--out-dir", default=None, help="batch: <out-dir>/<sku>/text.json + anchors_summary.json")
    p.add_argument("--debug-png", default=None, help="single SKU: overlay PNG")
    p.add_argument("--debug", action="store_true", help="batch: <out-dir>/<sku>/anchors_debug.png")
    p.add_argument("--out-res", type=int, default=None, help="rescale the written text.json to this per-view res")
    p.add_argument("--no-refine", action="store_true", help="bbox fit only, no silhouette IoU search")
    p.add_argument("--s-param", choices=("arc", "photo"), default="arc")
    p.add_argument("--depth-tol-bins", type=float, default=3.0)
    p.add_argument("--facing-tol", type=float, default=0.3, help="visible needs n . v_k > -this (geometric n)")
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--min-on-mesh", type=float, default=0.5)
    args = p.parse_args(argv)
    cfg = LiftConfig(depth_tol_bins=args.depth_tol_bins, facing_tol=args.facing_tol, s_param=args.s_param, min_conf=args.min_conf,
                     min_on_mesh=args.min_on_mesh, refine=not args.no_refine)
    if args.geometry == "unitex" and not (args.run_name or args.unitex_cache):
        p.error("--geometry unitex needs --run-name or --unitex-cache")
    skus = [args.sku] if args.sku else _sku_list(args.eval_dir, args.skus)
    if args.skus and not args.out_dir:
        p.error("batch mode needs --out-dir")
    if args.sku and not (args.out or args.out_dir):
        p.error("--sku needs --out or --out-dir")

    summary, failed = {}, {}
    for sku in skus:
        out = pathlib.Path(args.out) if (args.sku and args.out) else pathlib.Path(args.out_dir) / sku / "text.json"
        dbg = args.debug_png if args.sku else None
        if args.debug and not dbg:
            dbg = str(out.parent / "anchors_debug.png")
        try:
            doc = lift_sku(args.eval_dir, sku, args.geometry, args.run_name, args.mesh_views, args.unitex_cache,
                           args.text_source, args.ocr_json, cfg, dbg)
        except Exception as e:
            failed[sku] = f"{type(e).__name__}: {e}"
            print(f"  [{sku}] FAILED {failed[sku]}")
            continue
        if args.out_res and args.out_res != doc["res"]:
            doc = rescale_doc(doc, args.out_res)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
        L = doc["lift"]
        summary[sku] = {"iou_bbox": L["align"]["iou_bbox"], "iou": L["align"]["iou"], "n_items": L["n_items"],
                        "n_off_mesh": L["n_off_mesh"], "items_per_view": L["items_per_view"],
                        "warning": L["geometry_meta_warning"], "text_json": str(out)}
        print(f"  [{sku}] IoU {L['align']['iou_bbox']} -> {L['align']['iou']}  items {L['n_items']} "
              f"(off_mesh {L['n_off_mesh']})  per view {L['items_per_view']} -> {out}"
              + (f"  WARNING {L['geometry_meta_warning']}" if L["geometry_meta_warning"] else ""))
        if L["align"]["iou"] is not None and L["align"]["iou"] < 0.8:
            print(f"  [{sku}] WARNING silhouette IoU {L['align']['iou']} < 0.8: check the mesh orientation")
    if args.out_dir:
        pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        with open(pathlib.Path(args.out_dir) / "anchors_summary.json", "w") as f:
            json.dump({"eval_dir": str(args.eval_dir), "geometry": args.geometry, "run_name": args.run_name,
                       "config": asdict(cfg), "skus": summary, "failed": failed}, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
