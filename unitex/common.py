"""
Shared conventions for UniTEX-style 6-view data (numpy only, no bpy / torch).

Every script in this repo that touches UniTEX views imports its cameras, frames and
strip layout from here, so the Blender exporter (mvgen.py), the training loader, the
text-region projector and the evaluation harness agree by construction.

Frames:
  glTF / UniTEX world : Y-up, product front faces +Z (CFI-3DGen generated_mesh.glb frame).
  Blender world       : Z-up. The Blender glTF importer bakes G2B into the vertices.
  Normalized frame    : bbox centre at the origin, longest half-extent = 0.95
                        (TextureTools scale_to_bbox(0.95)).

Views:
  "raw" index  : UniTEX-FLUX MVDataset order on disk (render/<uid>/000i_*), Blender cameras.
  "strip" slot : position in the 512 x 3072 FLUX canvas, slot i shows raw FULL_INDEX[i].
  "grid" tile  : UniTEX inference 2x3 grid (mv_ccm.png, mv_normal.png, mv_rgb.png),
                 export order f r t / b l d, bottom tile NOT rolled.
"""

import numpy as np

RADIUS = 2.8
ORTHO_SCALE = 2.0          # image spans [-1, 1] camera units (TextureTools fx = fy = 1)
GEOMETRY_SCALE = 0.95      # longest half-extent after normalization
VIEW_RES = 512             # UniTEX per-view resolution (training and inference)
TOKEN_PX = 16              # FLUX: VAE 8x, then 2x2 packing -> one token per 16x16 px

# glTF (Y-up) -> Blender (Z-up): (x, y, z) -> (x, -z, y). B2G is UniTEX-FLUX's c2w_post.
G2B = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
B2G = G2B.T.copy()

RAW_VIEW_NAMES = ("front", "right", "back", "left", "top", "bottom")

# Blender cam2world per raw view. Cameras look down their local -Z, image right = local +X,
# image up = local +Y. Verified against TextureTools generate_box_views_c2ws + UniTEX
# infer_mv + UniTEX-FLUX c2w_post / full_index (max |diff| 0.002 after 8-bit quantization).
R = RADIUS
RAW_C2W = np.array([
    [[1, 0, 0, 0], [0, 0, -1, -R], [0, 1, 0, 0], [0, 0, 0, 1]],     # 0 front  cam at -Y_b (= +Z_g)
    [[0, 0, 1, R], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],       # 1 right  cam at +X
    [[-1, 0, 0, 0], [0, 0, 1, R], [0, 1, 0, 0], [0, 0, 0, 1]],      # 2 back   cam at +Y_b (= -Z_g)
    [[0, 0, -1, -R], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],    # 3 left   cam at -X
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, R], [0, 0, 0, 1]],       # 4 top    cam at +Z_b (= +Y_g)
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, -R], [0, 0, 0, 1]],    # 5 bottom cam at -Z_b, Rx(180)
], dtype=np.float64)
del R

# UniTEX-FLUX launch.py full_index: strip slot i <- raw view FULL_INDEX[i].
FULL_INDEX = (0, 3, 1, 2, 4, 5)
STRIP_NAMES = tuple(RAW_VIEW_NAMES[i] for i in FULL_INDEX)   # front left right back top bottom

# UniTEX inference 2x3 grid (export order frtbld, row-major): tile -> (raw index, rolled_180).
# "rolled_180" means the tile must be rotated 180 degrees to equal the raw view image.
GRID_TILE_TO_RAW = ((0, False), (1, False), (4, False), (2, False), (3, False), (5, True))


def raw_to_slot(raw_idx):
    return FULL_INDEX.index(raw_idx)


def slot_to_raw(slot):
    return FULL_INDEX[slot]


# ────────────────────────────────────────────────────────────────────────────
# Frame helpers
# ────────────────────────────────────────────────────────────────────────────

def gltf_to_blender(p):
    """(..., 3) glTF points or directions -> Blender frame."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def blender_to_gltf(p):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def normalize_params(verts, scale=GEOMETRY_SCALE):
    """Centre (bbox centre) and factor so that (v - centre) * factor has max half-extent `scale`."""
    verts = np.asarray(verts, dtype=np.float64)
    lo, hi = verts.min(0), verts.max(0)
    centre = (lo + hi) / 2
    half = (hi - lo).max() / 2
    return centre, scale / max(half, 1e-12)


# ────────────────────────────────────────────────────────────────────────────
# Orthographic projection (normalized Blender frame, raw cameras)
# ────────────────────────────────────────────────────────────────────────────

def project(p_blender, raw_idx, res=VIEW_RES):
    """Blender-frame points (..., 3) -> (pixel xy (..., 2), depth (...)).

    Pixel origin is the top-left corner of the view image, x right, y down, continuous
    coordinates (pixel centres at +0.5). Depth is the distance in front of the camera.
    """
    c2w = RAW_C2W[raw_idx]
    q = (np.asarray(p_blender, dtype=np.float64) - c2w[:3, 3]) @ c2w[:3, :3]   # world -> cam
    half = ORTHO_SCALE / 2
    x = (q[..., 0] / half + 1) * 0.5 * res
    y = (1 - q[..., 1] / half) * 0.5 * res
    return np.stack([x, y], axis=-1), -q[..., 2]


def view_dir(raw_idx):
    """Unit vector from the object towards the camera (Blender frame)."""
    return RAW_C2W[raw_idx][:3, 2].copy()


def decode_ccm(ccm_uint8):
    """CCM image (H, W, 3) uint8 or float in [0,1] -> positions in [-1, 1] (same frame as stored)."""
    a = np.asarray(ccm_uint8)
    a = a.astype(np.float64) / 255.0 if a.dtype == np.uint8 else a.astype(np.float64)
    return a[..., :3] * 2 - 1


def encode_ccm(p):
    """Positions in [-1, 1] -> float CCM in [0, 1]."""
    return (np.asarray(p, dtype=np.float64) + 1) / 2


# ────────────────────────────────────────────────────────────────────────────
# Strip / grid layout
# ────────────────────────────────────────────────────────────────────────────

def raw_box_to_strip(box_xyxy, raw_idx, res=VIEW_RES):
    """Pixel box in raw view `raw_idx` -> pixel box in the 1x6 strip canvas."""
    x0, y0, x1, y1 = box_xyxy
    off = raw_to_slot(raw_idx) * res
    return [x0 + off, y0, x1 + off, y1]


def raw_xy_to_strip(xy, raw_idx, res=VIEW_RES):
    xy = np.asarray(xy, dtype=np.float64).copy()
    xy[..., 0] += raw_to_slot(raw_idx) * res
    return xy


def split_strip(strip, res=VIEW_RES):
    """(H, 6*res, C) strip image -> list of 6 raw-ordered view images (bottom kept as in strip = raw)."""
    strip = np.asarray(strip)
    views = [None] * 6
    for slot in range(6):
        views[slot_to_raw(slot)] = strip[:, slot * res:(slot + 1) * res]
    return views


def stack_strip(raw_views):
    """6 raw-ordered view images -> (H, 6*res, C) strip in FULL_INDEX order."""
    return np.concatenate([raw_views[slot_to_raw(s)] for s in range(6)], axis=1)


def split_grid(grid, res=VIEW_RES):
    """UniTEX inference 2x3 grid (2*res, 3*res, C) -> list of 6 raw-ordered view images.

    The bottom tile is rolled 180 degrees so every returned image matches its raw camera.
    """
    grid = np.asarray(grid)
    views = [None] * 6
    for tile, (raw_idx, rolled) in enumerate(GRID_TILE_TO_RAW):
        r, c = divmod(tile, 3)
        img = grid[r * res:(r + 1) * res, c * res:(c + 1) * res]
        views[raw_idx] = img[::-1, ::-1] if rolled else img
    return views


def gltf_ccm_views_to_blender(views_ccm_uint8):
    """UniTEX inference CCM (glTF frame, p*0.5+0.5) -> Blender-frame positions per view."""
    return [gltf_to_blender(decode_ccm(v)) for v in views_ccm_uint8]


# ────────────────────────────────────────────────────────────────────────────
# Reference-image framing (UniTEX preprocess_reference_image)
# ────────────────────────────────────────────────────────────────────────────

def bbox_of_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def fit_box_affine(src_box, dst_box, uniform=False):
    """Axis-aligned affine (sx, sy, tx, ty) mapping src_box onto dst_box (xyxy)."""
    sw, sh = src_box[2] - src_box[0], src_box[3] - src_box[1]
    dw, dh = dst_box[2] - dst_box[0], dst_box[3] - dst_box[1]
    sx, sy = dw / sw, dh / sh
    if uniform:
        sx = sy = (sx + sy) / 2
    scx, scy = (src_box[0] + src_box[2]) / 2, (src_box[1] + src_box[3]) / 2
    dcx, dcy = (dst_box[0] + dst_box[2]) / 2, (dst_box[1] + dst_box[3]) / 2
    return sx, sy, dcx - sx * scx, dcy - sy * scy


def apply_affine(xy, aff):
    sx, sy, tx, ty = aff
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([xy[..., 0] * sx + tx, xy[..., 1] * sy + ty], axis=-1)
