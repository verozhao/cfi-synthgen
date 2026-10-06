"""
GlyphAnchor position ids for glyph patch tokens on the UniTEX 1x6 FLUX strip (numpy, fp32).

FLUX img_ids columns are (axis0, row, col). The 512 x 3072 target strip packs into a 32 x 192
token grid on axis0 = 0, and target token (r, c) covers strip pixels [16r, 16r+16) x [16c, 16c+16),
so a continuous strip pixel p sits at id p / 16 - 0.5. Glyph tokens live on their own plane
(axis0 = frame, default 1, GlyphAnchor's "virtual glyph plane") and reuse the target's (row, col).

Anchor modes (one id triple per packed patch token, row-major like UniTEX _pack_latents):
  center  : the paper (sec. 3.3). Native 1-token spacing, patch centre on the box centre,
            optionally rounded to integers, clamped inside the item's own view slot.
  stretch : gh x gw tokens spread evenly over the box.
  warp    : our 3D-anchored variant. Token centres (s, t) in the item's reading frame are mapped
            through its per-view grid (text.json "grid"), so tokens follow curved and foreshortened
            surfaces. Tokens that land on invisible cells get keep=False and NaN ids; the caller
            drops the matching latent tokens.

Ids are computed in float64 and returned as float32. Never build them in bf16: UniTEX does, and
bf16 rounds 257 -> 256 and 150.3 -> 150 (critic report B1).
"""

import warnings

import numpy as np

try:
    from unitex.common import TOKEN_PX, VIEW_RES, raw_to_slot
except ImportError:   # imported with unitex/ itself on sys.path
    from common import TOKEN_PX, VIEW_RES, raw_to_slot

ANCHOR_MODES = ("center", "stretch", "warp")
GLYPH_FRAME = 1


class PatchTooLarge(ValueError):
    """A center-anchored patch has more tokens than its view slot along some axis."""


# ────────────────────────────────────────────────────────────────────────────
# Strip token coordinates
# ────────────────────────────────────────────────────────────────────────────

def view_tokens(res=VIEW_RES):
    return res // TOKEN_PX


def px_to_id(px):
    """Continuous strip (or view) pixel coordinate -> token id coordinate (float64)."""
    return np.asarray(px, dtype=np.float64) / TOKEN_PX - 0.5


def slot_bounds(raw_view, res=VIEW_RES):
    """Inclusive token-centre id range (row_lo, row_hi, col_lo, col_hi) of a raw view's slot."""
    n = view_tokens(res)
    c0 = raw_to_slot(int(raw_view)) * n
    return 0.0, float(n - 1), float(c0), float(c0 + n - 1)


def clamp_to_slot(ids, raw_view, res=VIEW_RES):
    """Clip rows / cols into the slot's token-centre range (NaN rows stay NaN).

    Token centres, not pixel edges: a glyph token at col slot*32 - 0.5 would sit exactly between
    two views and attend to both equally.
    """
    ids = np.array(ids, dtype=np.float64)
    rlo, rhi, clo, chi = slot_bounds(raw_view, res)
    ids[:, 1] = np.clip(ids[:, 1], rlo, rhi)
    ids[:, 2] = np.clip(ids[:, 2], clo, chi)
    return ids


def _pack(rows, cols, frame):
    """Row ids (gh,) x col ids (gw,) -> [gh*gw, 3] float32, row-major like _pack_latents."""
    rr, cc = np.meshgrid(np.asarray(rows, np.float64), np.asarray(cols, np.float64), indexing="ij")
    ids = np.stack([np.full(rr.shape, float(frame)), rr, cc], axis=-1).reshape(-1, 3)
    return ids.astype(np.float32)


def downscale_factor(token_hw, res=VIEW_RES):
    """1.0 if a gh x gw patch fits in one view slot, else the factor to shrink it by."""
    n = view_tokens(res)
    return min(1.0, n / max(int(token_hw[0]), int(token_hw[1])))


# ────────────────────────────────────────────────────────────────────────────
# center / stretch
# ────────────────────────────────────────────────────────────────────────────

def center_ids(box, raw_view, token_hw, frame=GLYPH_FRAME, quantize=True, res=VIEW_RES):
    """Paper anchoring: patch tokens at native spacing, centred on the box centre.

    box is [x0, y0, x1, y1] in raw-view pixels. Raises PatchTooLarge when the patch has more
    tokens than the view along an axis; the caller must downscale the patch first.
    """
    gh, gw = int(token_hw[0]), int(token_hw[1])
    n = view_tokens(res)
    if gh > n or gw > n:
        raise PatchTooLarge(f"patch {gh}x{gw} tokens exceeds the {n}x{n} view slot, downscale it")
    x0, y0, x1, y1 = [float(v) for v in box]
    off = raw_to_slot(int(raw_view)) * res
    r0 = float(px_to_id((y0 + y1) / 2)) - (gh - 1) / 2
    c0 = float(px_to_id((x0 + x1) / 2 + off)) - (gw - 1) / 2
    if quantize:
        r0, c0 = np.floor(r0 + 0.5), np.floor(c0 + 0.5)
    rlo, rhi, clo, chi = slot_bounds(raw_view, res)
    r0 = min(max(r0, rlo), rhi - (gh - 1))
    c0 = min(max(c0, clo), chi - (gw - 1))
    return _pack(r0 + np.arange(gh), c0 + np.arange(gw), frame)


def stretch_ids(box, raw_view, token_hw, frame=GLYPH_FRAME, res=VIEW_RES):
    """gh x gw tokens spread over the box: row_i = (y0 + (i+0.5)(y1-y0)/gh)/16 - 0.5, same for cols."""
    gh, gw = int(token_hw[0]), int(token_hw[1])
    x0, y0, x1, y1 = [float(v) for v in box]
    off = raw_to_slot(int(raw_view)) * res
    rows = px_to_id(y0 + (np.arange(gh) + 0.5) * (y1 - y0) / gh)
    cols = px_to_id(x0 + off + (np.arange(gw) + 0.5) * (x1 - x0) / gw)
    return clamp_to_slot(_pack(rows, cols, frame), raw_view, res).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# warp: per-view (s, t) grid
# ────────────────────────────────────────────────────────────────────────────

def parse_grid(grid):
    """text.json grid -> (nt, ns, 2) float64 view-pixel array, NaN where a cell is not visible.

    Convention: xy[it][is], i.e. nt rows across the text (t = 0 at the top of the letters) of
    ns cells along the baseline (s = 0 at the first letter), cell centres at ((is+0.5)/ns,
    (it+0.5)/nt). A transposed [ns][nt] list is accepted when ns != nt. An ndarray of shape
    (nt, ns, 2) passes through.
    """
    if isinstance(grid, np.ndarray):
        return grid.astype(np.float64)
    ns, nt = int(grid["ns"]), int(grid["nt"])
    xy = grid["xy"]
    n0 = len(xy)
    n1 = len(xy[0]) if n0 else 0
    if (n0, n1) == (nt, ns):
        transpose = False
    elif (n0, n1) == (ns, nt):
        transpose = True
    else:
        raise ValueError(f"grid xy is {n0}x{n1}, expected nt x ns = {nt}x{ns}")
    arr = np.full((n0, n1, 2), np.nan)
    for i, row in enumerate(xy):
        for j, p in enumerate(row):
            if p is not None:
                arr[i, j] = (float(p[0]), float(p[1]))
    return arr.transpose(1, 0, 2).copy() if transpose else arr


def grid_from_quad(quad, ns=16, nt=4):
    """Planar grid (bilinear over the reading-order quad TL, TR, BR, BL) for items without one."""
    tl, tr, br, bl = [np.asarray(p, np.float64) for p in quad]
    s = (np.arange(ns) + 0.5) / ns
    t = (np.arange(nt) + 0.5) / nt
    S, T = np.meshgrid(s, t)
    S, T = S[..., None], T[..., None]
    top = tl + (tr - tl) * S
    bot = bl + (br - bl) * S
    return top + (bot - top) * T


def grid_extent(G):
    """Projected (length_px, height_px) of the whole item from its grid, NaN if undefined.

    Uses the mean spacing of adjacent visible cells, so a partially visible item gets its full
    length at the visible pixel density (the invisible part is extrapolated).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nt, ns = G.shape[:2]
        ds = np.linalg.norm(G[:, 1:] - G[:, :-1], axis=-1) if ns > 1 else np.full((1,), np.nan)
        dt = np.linalg.norm(G[1:] - G[:-1], axis=-1) if nt > 1 else np.full((1,), np.nan)
        length = float(np.nanmean(ds)) * ns if np.isfinite(ds).any() else float("nan")
        height = float(np.nanmean(dt)) * nt if np.isfinite(dt).any() else float("nan")
    return length, height


def sample_grid(G, s, t):
    """Map reading-frame points (s, t) in [0,1]^2 through the grid to view pixels.

    Returns (xy [N, 2] float64, keep [N] bool). keep is False where the cell containing (s, t) is
    null. Positions use bilinear interpolation between cell centres (linear extrapolation in the
    border half cells); when some of the 4 corners are null the valid ones are renormalised, and a
    point with no valid corner falls back to the nearest valid cell.
    """
    nt, ns = G.shape[:2]
    s = np.asarray(s, np.float64).ravel()
    t = np.asarray(t, np.float64).ravel()
    valid = np.isfinite(G[..., 0]) & np.isfinite(G[..., 1])
    ci = np.clip(np.floor(s * ns).astype(int), 0, ns - 1)
    ri = np.clip(np.floor(t * nt).astype(int), 0, nt - 1)
    keep = valid[ri, ci]

    u, v = s * ns - 0.5, t * nt - 0.5
    i0 = np.clip(np.floor(u), 0, max(ns - 2, 0)).astype(int)
    j0 = np.clip(np.floor(v), 0, max(nt - 2, 0)).astype(int)
    i1, j1 = np.minimum(i0 + 1, ns - 1), np.minimum(j0 + 1, nt - 1)
    fu = u - i0 if ns > 1 else np.zeros_like(u)
    fv = v - j0 if nt > 1 else np.zeros_like(v)
    corners = ((j0, i0), (j0, i1), (j1, i0), (j1, i1))
    P = np.stack([np.nan_to_num(G[r, c]) for r, c in corners], axis=1)    # [N, 4, 2]
    V = np.stack([valid[r, c] for r, c in corners], axis=1)               # [N, 4]

    def weights(a, b):
        return np.stack([(1 - b) * (1 - a), (1 - b) * a, b * (1 - a), b * a], axis=1)

    out = np.full((len(s), 2), np.nan)
    full = V.all(axis=1)
    w = weights(fu, fv)
    out[full] = np.einsum("nk,nkd->nd", w[full], P[full])
    wc = weights(np.clip(fu, 0, 1), np.clip(fv, 0, 1)) * V
    sw = wc.sum(axis=1)
    part = ~full & (sw > 1e-9)
    out[part] = np.einsum("nk,nkd->nd", wc[part], P[part]) / sw[part, None]
    rest = ~full & ~part
    if rest.any() and valid.any():
        # nearest valid cell centres, ties averaged (a hole's 4 neighbours give its centre)
        vr, vc = np.nonzero(valid)
        d = (vc[None, :] - u[rest, None]) ** 2 + (vr[None, :] - v[rest, None]) ** 2
        near = d <= d.min(axis=1, keepdims=True) + 1e-9
        out[rest] = (near[..., None] * G[vr, vc][None]).sum(axis=1) / near.sum(axis=1, keepdims=True)
    return out, keep


def warp_ids(grid, raw_view, token_hw, frame=GLYPH_FRAME, mask_null=True, res=VIEW_RES):
    """Warp anchoring. Returns (ids [gh*gw, 3] float32, keep [gh*gw] bool).

    Patch token (i, j) has reading-frame centre (s, t) = ((j+0.5)/gw, (i+0.5)/gh). Masked tokens
    get NaN rows / cols so a forgotten mask fails loudly. With mask_null=False, tokens on null
    cells are kept and placed by the nearest-valid fallback instead.
    """
    G = parse_grid(grid)
    gh, gw = int(token_hw[0]), int(token_hw[1])
    S, T = np.meshgrid((np.arange(gw) + 0.5) / gw, (np.arange(gh) + 0.5) / gh)
    xy, keep = sample_grid(G, S.ravel(), T.ravel())
    if not mask_null:
        keep = np.isfinite(xy[:, 0])
    off = raw_to_slot(int(raw_view)) * res
    ids = np.empty((gh * gw, 3), np.float64)
    ids[:, 0] = float(frame)
    ids[:, 1] = px_to_id(xy[:, 1])
    ids[:, 2] = px_to_id(xy[:, 0] + off)
    ids[~keep, 1:] = np.nan
    return clamp_to_slot(ids, raw_view, res).astype(np.float32), keep


def warp_token_count(grid, token_hw):
    """Number of tokens warp_ids would keep (for budget estimates)."""
    G = parse_grid(grid)
    gh, gw = int(token_hw[0]), int(token_hw[1])
    S, T = np.meshgrid((np.arange(gw) + 0.5) / gw, (np.arange(gh) + 0.5) / gh)
    _, keep = sample_grid(G, S.ravel(), T.ravel())
    return int(keep.sum())


# ────────────────────────────────────────────────────────────────────────────
# Jitter
# ────────────────────────────────────────────────────────────────────────────

def shift_ids(ids, dr, dc):
    ids = np.array(ids, dtype=np.float32)
    ids[:, 1] += np.float32(dr)
    ids[:, 2] += np.float32(dc)
    return ids


def _shift_range(ids, keep, raw_view, res=VIEW_RES):
    """Integer (dr, dc) ranges that keep every kept token inside the slot."""
    k = ids[keep] if keep is not None else ids
    k = k[np.isfinite(k[:, 1])]
    if len(k) == 0:
        return (0, 0), (0, 0)
    rlo, rhi, clo, chi = slot_bounds(raw_view, res)
    r = (int(np.ceil(rlo - k[:, 1].min())), int(np.floor(rhi - k[:, 1].max())))
    c = (int(np.ceil(clo - k[:, 2].min())), int(np.floor(chi - k[:, 2].max())))
    return (min(r[0], 0), max(r[1], 0)), (min(c[0], 0), max(c[1], 0))


def jitter_ids(ids, keep, raw_view, rng, max_tokens=1, res=VIEW_RES):
    """Uniform integer translation in [-max_tokens, max_tokens]^2 per instance, kept inside the slot.

    Returns (ids float32, (dr, dc)). The shift is clipped rather than the ids, so the footprint
    keeps its shape at the slot border.
    """
    dr, dc = (int(v) for v in rng.integers(-max_tokens, max_tokens + 1, size=2))
    (rl, rh), (cl, ch) = _shift_range(ids, keep, raw_view, res)
    dr, dc = min(max(dr, rl), rh), min(max(dc, cl), ch)
    return shift_ids(ids, dr, dc), (dr, dc)


# ────────────────────────────────────────────────────────────────────────────
# Concatenation and collision checks
# ────────────────────────────────────────────────────────────────────────────

def _ids_keep(inst):
    if isinstance(inst, (tuple, list)):
        ids, keep = inst[0], inst[1]
    else:
        ids, keep = inst.ids, inst.keep
    ids = np.asarray(ids, np.float32)
    keep = np.ones(len(ids), bool) if keep is None else np.asarray(keep, bool)
    return ids, keep


def concat_ids(instances, kept_only=True):
    """Stack instance ids. Returns (ids [N, 3] float32, [slice per instance]).

    instances are GlyphInstance-like objects (.ids, .keep) or (ids, keep) pairs. With kept_only
    the masked warp tokens are left out, so each slice lines up with that instance's latent
    tokens after latents[:, keep].
    """
    parts, slices, n = [], [], 0
    for inst in instances:
        ids, keep = _ids_keep(inst)
        a = ids[keep] if kept_only else ids
        parts.append(a)
        slices.append(slice(n, n + len(a)))
        n += len(a)
    ids = np.concatenate(parts, 0) if parts else np.zeros((0, 3), np.float32)
    return ids.astype(np.float32), slices


def _keys(ids):
    """float32 id triples -> set of hashable byte keys (NaN rows skipped, -0.0 folded into 0.0)."""
    a = np.ascontiguousarray(np.asarray(ids, np.float32) + np.float32(0.0))
    a = a[np.isfinite(a).all(axis=1)]
    return {r.tobytes() for r in a}


def count_collisions(ids, slices):
    """Number of glyph tokens whose exact id triple also occurs in a DIFFERENT instance.

    Tokens repeated inside one instance (a fixed patch squeezed into a small box) are allowed.
    """
    owner = {}
    n = 0
    for k, sl in enumerate(slices):
        for key in _keys(ids[sl]):
            if key in owner and owner[key] != k:
                n += 1
            else:
                owner.setdefault(key, k)
    return n


def verify_unique(instances, nudge=False, max_shift=2, warn=True, res=VIEW_RES):
    """Check that no two instances share an id triple; optionally nudge later ones apart.

    instances are in priority order and need .ids, .keep and .raw_view. With nudge, each
    colliding instance is moved by the smallest integer (dr, dc) within +-max_shift that stays
    in its slot and minimises collisions with the instances before it (the paper's "overlap
    adjustment"). Mutates .ids. Returns the number of colliding tokens left.
    """
    occupied = set()
    if nudge:
        cand = sorted(((dr, dc) for dr in range(-max_shift, max_shift + 1)
                       for dc in range(-max_shift, max_shift + 1)),
                      key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[0]), d))
        for inst in instances:
            ids, keep = _ids_keep(inst)
            keys = _keys(ids[keep])
            if keys & occupied:
                (rl, rh), (cl, ch) = _shift_range(ids, keep, inst.raw_view, res)
                best, best_n = (0, 0), len(keys & occupied)
                for dr, dc in cand:
                    if not (rl <= dr <= rh and cl <= dc <= ch):
                        continue
                    n = len(_keys(shift_ids(ids[keep], dr, dc)) & occupied)
                    if n < best_n:
                        best, best_n = (dr, dc), n
                    if n == 0:
                        break
                if best != (0, 0):
                    inst.ids = shift_ids(ids, *best)
                    keys = _keys(inst.ids[keep])
            occupied |= keys
    ids, slices = concat_ids(instances)
    n = count_collisions(ids, slices)
    if n and warn:
        warnings.warn(f"{n} glyph tokens share an id triple with another glyph instance")
    return n
