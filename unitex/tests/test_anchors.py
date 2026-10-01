"""
Tests for unitex/anchors.py on analytic scenes ray-cast into the six raw UniTEX cameras.

Scenes (normalized Blender frame, 512 px per view, 8-bit truncated CCM like mvgen / UniTEX):
  box   half extents (0.45, 0.3, 0.8), yaw 30 deg: the front face is seen by raw 0 (cos 0.87) and
        raw 1 (cos 0.5), so front text must land on a known box in the right view
  can   vertical cylinder r 0.55: a label spanning -40..70 deg around the axis wraps onto raw 1
        and raw 3 with non-uniform (arc-length) spacing, plus a vertical item reading downwards,
        an item beside the can (off_mesh) and one half past the limb (partly_off_mesh)
Ground truth per view is the set of pixel centres whose analytic surface point lies in the text
region. The "photo" is view 0 upscaled 2x on a padded canvas, so the affine fit is exercised.

Run: python -m pytest unitex/tests/test_anchors.py  (ANCHORS_DEBUG_OUT=<dir> keeps debug PNGs)
"""

import json
import math
import os
import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from unitex import anchors as an
from unitex import common as C
from unitex import glyph as gl
from unitex import glyph_ids as gid

RES = C.VIEW_RES
PAD = (37, 21)
UP = 2


# ────────────────────────────────────────────────────────────────────────────
# Analytic ray casting
# ────────────────────────────────────────────────────────────────────────────

def _rays(k):
    xy = an.pixel_centres(RES)
    o = an.unproject(xy, np.zeros(xy.shape[:2]), k, RES)          # on the camera plane
    d = np.broadcast_to(-C.RAW_C2W[k][:3, 2], o.shape)
    return o, d


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


class Box:
    def __init__(self, half=(0.45, 0.3, 0.8), yaw=30.0):
        self.h = np.asarray(half, np.float64)
        self.R = _rz(yaw)

    def hit(self, o, d):
        ol, dl = o @ self.R, d @ self.R                            # world -> box frame (R^T x)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-self.h - ol) / dl, (self.h - ol) / dl
        tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
        tn = np.where(np.isfinite(tn), tn, -np.inf)
        tf = np.where(np.isfinite(tf), tf, np.inf)
        t_near, axis = tn.max(-1), tn.argmax(-1)
        ok = (t_near <= tf.min(-1)) & (t_near > 0)
        nl = np.zeros(o.shape)
        np.put_along_axis(nl, axis[..., None], -np.sign(np.take_along_axis(dl, axis[..., None], -1)), -1)
        P = o + t_near[..., None] * d
        local = P @ self.R
        face = np.where(ok, axis * 2 + (np.take_along_axis(nl, axis[..., None], -1)[..., 0] > 0), -1)
        return ok, t_near, P, nl @ self.R.T, {"face": face, "u": local[..., 0], "z": local[..., 2]}

    def point(self, u, z):
        """Front face (box frame y = -h_y) point."""
        u, z = np.broadcast_arrays(np.asarray(u, np.float64), np.asarray(z, np.float64))
        return np.stack([u, np.full(u.shape, -self.h[1]), z], -1) @ self.R.T

    def normal(self, u, z):
        return np.broadcast_to(np.array([0.0, -1.0, 0.0]) @ self.R.T, np.shape(u) + (3,))

    def in_region(self, g, reg):
        return (g["face"] == 2) & (g["u"] >= reg["u"][0]) & (g["u"] <= reg["u"][1]) \
            & (g["z"] >= reg["z"][0]) & (g["z"] <= reg["z"][1])


class Can:
    def __init__(self, r=0.55, hz=0.9):
        self.r, self.hz = r, hz

    def hit(self, o, d):
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        b = 2 * (o[..., 0] * d[..., 0] + o[..., 1] * d[..., 1])
        c = o[..., 0] ** 2 + o[..., 1] ** 2 - self.r ** 2
        disc = b * b - 4 * a * c
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
        zs = o[..., 2] + ts * d[..., 2]
        side = (a > 1e-12) & (disc >= 0) & (ts > 0) & (np.abs(zs) <= self.hz)
        t_side = np.where(side, ts, np.inf)
        t_cap = np.full(o.shape[:-1], np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            for zc in (self.hz, -self.hz):
                tc = (zc - o[..., 2]) / d[..., 2]
                pc = o + tc[..., None] * d
                okc = np.isfinite(tc) & (tc > 0) & (pc[..., 0] ** 2 + pc[..., 1] ** 2 <= self.r ** 2)
                t_cap = np.where(okc & (tc < t_cap), tc, t_cap)
        t = np.minimum(t_side, t_cap)
        ok = np.isfinite(t)
        t = np.where(ok, t, 0.0)
        P = o + t[..., None] * d
        is_side = ok & (t_side <= t_cap)
        n = np.where(is_side[..., None], np.stack([P[..., 0], P[..., 1], np.zeros_like(t)], -1) / self.r,
                     np.stack([np.zeros_like(t), np.zeros_like(t), np.sign(P[..., 2])], -1))
        phi = np.degrees(np.arctan2(P[..., 0], -P[..., 1]))
        return ok, t, P, n, {"side": is_side, "phi": phi, "z": P[..., 2]}

    def point(self, phi, z):
        phi, z = np.broadcast_arrays(np.radians(np.asarray(phi, np.float64)), np.asarray(z, np.float64))
        return np.stack([self.r * np.sin(phi), -self.r * np.cos(phi), z], -1)

    def normal(self, phi, z):
        p = self.point(phi, z)
        return np.stack([p[..., 0], p[..., 1], np.zeros(p.shape[:-1])], -1) / self.r

    def in_region(self, g, reg):
        return g["side"] & (g["phi"] >= reg["u"][0]) & (g["phi"] <= reg["u"][1]) \
            & (g["z"] >= reg["z"][0]) & (g["z"] <= reg["z"][1])


def render(shape):
    views = []
    for k in range(6):
        o, d = _rays(k)
        ok, t, P, n, g = shape.hit(o, d)
        views.append({"mask": ok, "pos": np.where(ok[..., None], P, 0.0), "normal": np.where(ok[..., None], n, 0.0),
                      "g": g})
    return views


def _trunc(v):
    return (np.clip((np.asarray(v) + 1) / 2, 0, 1) * 255).astype(np.uint8)


def write_mvgen(d, views, normals=True):
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for k, v in enumerate(views):
        rgb = _trunc(v["pos"])
        rgb[~v["mask"]] = 0
        Image.fromarray(np.dstack([rgb, v["mask"].astype(np.uint8) * 255]), "RGBA").save(d / f"view_{k:02d}_nocs.png")
        Image.fromarray(v["mask"].astype(np.uint8) * 255, "L").save(d / f"view_{k:02d}_alpha.png")
        if normals:
            n = _trunc(v["normal"] @ C.RAW_C2W[k][:3, :3])
            n[~v["mask"]] = 255
            Image.fromarray(n, "RGB").save(d / f"view_{k:02d}_normal.png")
    with open(d / "views.json", "w") as f:
        json.dump({"views": list(range(6)), "res": RES, "yaw_deg": 0.0, "with_geometry": True}, f)
    return d


def write_unitex(d, views):
    """UniTEX export_condition layout: glTF frame, grey background, 2x3 f r t / b l d, bottom rolled."""
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    grids = {n: np.zeros((2 * RES, 3 * RES, c), np.uint8) for n, c in (("alpha", 1), ("ccm", 3), ("normal", 3))}
    for tile, (raw, rolled) in enumerate(C.GRID_TILE_TO_RAW):
        v = views[raw]
        m = v["mask"]
        tiles = {"alpha": (m.astype(np.uint8) * 255)[..., None],
                 "ccm": np.where(m[..., None], _trunc(C.blender_to_gltf(v["pos"])), 128).astype(np.uint8),
                 "normal": np.where(m[..., None], _trunc(C.blender_to_gltf(v["normal"])), 128).astype(np.uint8)}
        r, c = divmod(tile, 3)
        for n, img in tiles.items():
            grids[n][r * RES:(r + 1) * RES, c * RES:(c + 1) * RES] = img[::-1, ::-1] if rolled else img
    Image.fromarray(grids["alpha"][..., 0], "L").save(d / "mv_alpha.png")
    Image.fromarray(grids["ccm"], "RGB").save(d / "mv_ccm.png")
    Image.fromarray(grids["normal"], "RGB").save(d / "mv_normal.png")
    return d


# ────────────────────────────────────────────────────────────────────────────
# Scenes and ground truth
# ────────────────────────────────────────────────────────────────────────────

def view0_quad(shape, reg, orient=0):
    """Reading-order quad in view 0 of a (u, z) region. orient 90: reads downwards (top of letters to +x)."""
    (u0, u1), (z0, z1) = reg["u"], reg["z"]
    if orient == 0:
        corners = [(u0, z1), (u1, z1), (u1, z0), (u0, z0)]
    else:
        corners = [(u1, z1), (u1, z0), (u0, z0), (u0, z1)]
    return np.array([C.project(shape.point(u, z), 0, RES)[0] for u, z in corners])


def gt_view(shape, views, k, reg):
    m = shape.in_region(views[k]["g"], reg) & views[k]["mask"]
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None, 0
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], int(m.sum())


def gt_grid(shape, reg, k, orient=0):
    """Analytic (nt, ns, 2) projection of the text.json cell centres (arc-length s) and visibility."""
    (u0, u1), (z0, z1) = reg["u"], reg["z"]
    S, T = np.meshgrid((np.arange(an.NS) + 0.5) / an.NS, (np.arange(an.NT) + 0.5) / an.NT)
    if orient == 0:
        u, z = u0 + S * (u1 - u0), z1 - T * (z1 - z0)
    else:
        u, z = u1 - T * (u1 - u0), z1 - S * (z1 - z0)
    P = shape.point(u, z)
    xy, _ = C.project(P, k, RES)
    vis = (shape.normal(u, z) @ C.view_dir(k)) > 1e-6
    return xy, vis


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def centre_err(a, b):
    return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2))


def photo_of(views):
    """View-0 silhouette upscaled 2x on a padded canvas, and the matching quad map."""
    m = np.kron(views[0]["mask"], np.ones((UP, UP), bool))
    canvas = np.zeros((m.shape[0] + 2 * PAD[1], m.shape[1] + 2 * PAD[0]), bool)
    canvas[PAD[1]:PAD[1] + m.shape[0], PAD[0]:PAD[0] + m.shape[1]] = m
    return canvas, C.bbox_of_mask(canvas)


def to_photo(q):
    return (np.asarray(q) * UP + np.array(PAD)).tolist()


BOX_ITEMS = [
    {"name": "front", "reg": {"u": (-0.1, 0.4), "z": (0.2, 0.35)}, "orient": 0},
    {"name": "vertical", "reg": {"u": (-0.35, -0.25), "z": (-0.6, 0.1)}, "orient": 90},
]
CAN_ITEMS = [
    {"name": "wrap", "reg": {"u": (-40.0, 70.0), "z": (0.1, 0.25)}, "orient": 0},
    {"name": "vertical", "reg": {"u": (50.0, 60.0), "z": (-0.5, 0.2)}, "orient": 90},
]


def scene_items(shape, specs):
    items = []
    for i, sp in enumerate(specs):
        q = view0_quad(shape, sp["reg"], sp["orient"])
        items.append({"id": i, "text": f"ITEM {sp['name'].upper()}", "conf": 0.99, "quad": to_photo(q)})
    return items


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    root = tmp_path_factory.mktemp("anchors")
    out = {}
    for name, shape, specs in (("box", Box(), BOX_ITEMS), ("can", Can(), CAN_ITEMS)):
        views = render(shape)
        mv = write_mvgen(root / f"{name}_mvgen", views)
        ux = write_unitex(root / f"{name}_unitex", views)
        nn = write_mvgen(root / f"{name}_nonormal", views, normals=False)
        pmask, fg = photo_of(views)
        out[name] = dict(shape=shape, views=views, specs=specs, mvgen=mv, unitex=ux, nonormal=nn,
                         photo_mask=pmask, fg_box=fg, items=scene_items(shape, specs), root=root)
    return out


def _geo(sc, geometry):
    cache = sc.setdefault("geo", {})
    if geometry not in cache:
        cache[geometry] = (an.load_mvgen_views(sc[geometry]) if geometry in ("mvgen", "nonormal")
                           else an.load_unitex_cache(sc["unitex"]))
    return cache[geometry]


def _lift(sc, geometry="mvgen", cfg=None, items=None, **kw):
    geo = _geo(sc, geometry)
    doc = an.lift(items or sc["items"], sc["photo_mask"], sc["fg_box"], geo, cfg or an.LiftConfig(**kw),
                  photo_size=sc["photo_mask"].shape[::-1])
    dbg = os.environ.get("ANCHORS_DEBUG_OUT")
    if dbg:
        tag = f"{geometry}_{len(doc['items'])}_{doc['lift']['config']['s_param']}"
        an.draw_debug(doc, geo, Image.fromarray(sc["photo_mask"].astype(np.uint8) * 200).convert("RGB"),
                      sc["photo_mask"], pathlib.Path(dbg) / f"anchors_{tag}_{id(sc) % 997}.png")
    return doc, geo


# ────────────────────────────────────────────────────────────────────────────
# Unit helpers
# ────────────────────────────────────────────────────────────────────────────

def test_inverse_bilinear_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = np.array([[0, 0], [100, 0], [100, 30], [0, 30]], np.float64) + rng.normal(0, 6, (4, 2))
        s, t = rng.random(200), rng.random(200)
        xy = an.bilinear(q, s, t)
        s2, t2 = an.inverse_bilinear(xy, q)
        assert np.abs(s2 - s).max() < 1e-9 and np.abs(t2 - t).max() < 1e-9


def test_unproject_inverts_project():
    rng = np.random.default_rng(1)
    P = rng.uniform(-0.9, 0.9, (100, 3))
    for k in range(6):
        xy, d = C.project(P, k, RES)
        assert np.abs(an.unproject(xy, d, k, RES) - P).max() < 1e-12


def test_depth_dequantization(scenes):
    """Truncated 8-bit CCM depth after the half-bin offset and smoothing, vs the analytic depth."""
    for name in ("box", "can"):
        sc = scenes[name]
        geo = _geo(sc, "mvgen")
        for k in (0, 1):
            v = sc["views"][k]
            true = an.camera_depth(v["pos"], k)
            m = v["mask"].copy()
            for dy in range(-3, 4):                                # away from silhouettes and the box edges
                for dx in range(-3, 4):
                    m &= an._shifted(v["mask"].astype(float), dy, dx, 0.0) > 0.5
            if name == "box":
                m &= v["g"]["face"] == v["g"]["face"][m].min()
            raw = an.camera_depth(an.decode_trunc(_trunc(v["pos"])), k)
            err_raw = np.abs(raw - true)[m] / an.BIN
            err = np.abs(geo.depths[k] - true)[m] / an.BIN
            assert err.mean() < 0.6 * err_raw.mean() + 0.02, (name, k, err.mean(), err_raw.mean())
            assert err.mean() < 0.2, (name, k, err.mean())


# ────────────────────────────────────────────────────────────────────────────
# Lift: box, can, rotated, off mesh
# ────────────────────────────────────────────────────────────────────────────

def _check_views(sc, doc, item_idx, expect_views, iou_min=0.8, centre_max=2.0, grid_max=2.0):
    spec = sc["specs"][item_idx]
    it = doc["items"][item_idx]
    assert set(it["views"]) == set(expect_views), (spec["name"], sorted(it["views"]))
    report = {}
    for k in expect_views:
        v = it["views"][k]
        gt_box, gt_n = gt_view(sc["shape"], sc["views"], int(k), spec["reg"])
        iou, ce = box_iou(v["bbox"], gt_box), centre_err(v["bbox"], gt_box)
        G = gid.parse_grid(v["grid"])
        Gt, vis = gt_grid(sc["shape"], spec["reg"], int(k), spec["orient"])
        both = np.isfinite(G[..., 0]) & vis
        gerr = np.linalg.norm(G - Gt, axis=-1)[both]
        report[k] = dict(iou=round(iou, 4), centre_err=round(ce, 3), pixels=v["pixels"], gt_pixels=gt_n,
                         grid_err_mean=round(float(gerr.mean()), 3), grid_err_max=round(float(gerr.max()), 3),
                         cells=int(both.sum()), coverage=v["coverage"], gt_coverage=float(vis.any(0).mean()),
                         cos=v["cos"])
        assert iou > iou_min and ce < centre_max, (spec["name"], k, report[k])
        assert gerr.mean() < grid_max, (spec["name"], k, report[k])
        # null cells only where the analytic surface faces away (one cell of slack at the limb)
        assert (np.isfinite(G[..., 0]) & ~vis).sum() <= an.NT, (spec["name"], k)
    print(spec["name"], json.dumps(report))
    return report


def test_box_front_text_lands_on_right_view(scenes):
    sc = scenes["box"]
    doc, _ = _lift(sc)
    assert doc["lift"]["align"]["iou_bbox"] > 0.995
    rep = _check_views(sc, doc, 0, ["0", "1"])
    v1 = doc["items"][0]["views"]["1"]
    assert abs(v1["cos"] - 0.5) < 0.02 and abs(doc["items"][0]["views"]["0"]["cos"] - math.cos(math.radians(30))) < 0.02
    assert rep["1"]["coverage"] == 1.0
    # view 0 entries are the mapped quad itself
    q0 = doc["items"][0]["views"]["0"]["quad"]
    assert np.abs(np.asarray(q0) - view0_quad(sc["shape"], BOX_ITEMS[0]["reg"])).max() < 0.02


def test_can_label_wraps_and_bends(scenes):
    sc = scenes["can"]
    doc, _ = _lift(sc)
    rep = _check_views(sc, doc, 0, ["0", "1", "3"])
    it = doc["items"][0]
    # arc length: the photo's midpoint is not the label's midpoint on a can
    phi0, phi1 = CAN_ITEMS[0]["reg"]["u"]
    x0, x1 = math.sin(math.radians(phi0)), math.sin(math.radians(phi1))
    phi_mid = math.degrees(math.asin((x0 + x1) / 2))
    assert abs(it["s_arc_at_photo_mid"] - (phi_mid - phi0) / (phi1 - phi0)) < 0.01
    # s range seen by each side camera: right sees phi > 0, left phi < 0
    split = (0 - phi0) / (phi1 - phi0)
    assert abs(it["views"]["1"]["s_range"][0] - split) < 0.04 and it["views"]["1"]["s_range"][1] == 1.0
    assert it["views"]["3"]["s_range"][0] == 0.0 and abs(it["views"]["3"]["s_range"][1] - split) < 0.04
    for k in ("1", "3"):
        assert abs(rep[k]["coverage"] - rep[k]["gt_coverage"]) <= 1.5 / an.NS
    # grid spacing along s is non-uniform in view 1 (foreshortened towards the limb): the lift bends
    G = gid.parse_grid(it["views"]["1"]["grid"])
    ds = np.diff(G[1, :, 0])
    ds = ds[np.isfinite(ds)]
    assert ds.max() > 3 * ds.min() > 0


def test_arc_param_beats_photo_param_on_can(scenes):
    sc = scenes["can"]
    doc_a, _ = _lift(sc, s_param="arc")
    doc_p, _ = _lift(sc, s_param="photo")
    Gt, vis = gt_grid(sc["shape"], CAN_ITEMS[0]["reg"], 1)
    errs = {}
    for tag, doc in (("arc", doc_a), ("photo", doc_p)):
        G = gid.parse_grid(doc["items"][0]["views"]["1"]["grid"])
        both = np.isfinite(G[..., 0]) & vis
        errs[tag] = float(np.linalg.norm(G - Gt, axis=-1)[both].mean())
    print("can grid error in view 1 (px):", errs)
    assert errs["arc"] < 1.5 and errs["photo"] > 4 * errs["arc"]


def test_rotated_items(scenes):
    for name in ("box", "can"):
        sc = scenes[name]
        doc, _ = _lift(sc)
        it = doc["items"][1]
        assert "rotated" in it["flags"] and abs(it["angle_deg"] - 90) < 1
        _check_views(sc, doc, 1, ["0", "1"])
        for k in ("0", "1"):
            assert gl._view_orientation(it["views"][k]) == 90, (name, k)


def test_off_mesh_and_partial(scenes):
    sc = scenes["can"]
    r = C.project(sc["shape"].point(90, 0), 0, RES)[0][0]        # right limb x in view 0
    beside = [[r + 20, 200], [r + 70, 200], [r + 70, 215], [r + 20, 215]]
    straddle = [[r - 30, 300], [r + 15, 300], [r + 15, 312], [r - 30, 312]]    # 2/3 on the can
    items = sc["items"] + [{"text": "BADGE", "conf": 0.9, "quad": to_photo(beside)},
                           {"text": "EDGE", "conf": 0.3, "quad": to_photo(straddle)}]
    doc, _ = _lift(sc, items=items)
    badge, edge = doc["items"][2], doc["items"][3]
    assert "off_mesh" in badge["flags"] and badge["views"] == {} and badge["on_mesh"] == 0.0
    assert "partly_off_mesh" in edge["flags"] and "low_conf" in edge["flags"] and "0" in edge["views"]
    assert 0.55 < edge["on_mesh"] < 0.8
    assert doc["lift"]["n_off_mesh"] == 1


def test_unitex_cache_matches_mvgen(scenes):
    for name in ("box", "can"):
        sc = scenes[name]
        a, _ = _lift(sc, "mvgen")
        b, _ = _lift(sc, "unitex")
        for ia, ib in zip(a["items"], b["items"]):
            assert set(ia["views"]) == set(ib["views"])
            for k in ia["views"]:
                va, vb = ia["views"][k], ib["views"][k]
                assert va["bbox"] == vb["bbox"] and va["pixels"] == vb["pixels"], (name, k)
                assert abs(va["cos"] - vb["cos"]) < 0.01
                Ga, Gb = gid.parse_grid(va["grid"]), gid.parse_grid(vb["grid"])
                assert np.array_equal(np.isfinite(Ga), np.isfinite(Gb))
                assert np.nanmax(np.abs(Ga - Gb)) < 0.05


def test_ccm_normals_fallback(scenes):
    for name in ("box", "can"):
        sc = scenes[name]
        a, _ = _lift(sc, "mvgen")
        b, geo = _lift(sc, "nonormal")
        assert geo.normal_source == "ccm-gradient"
        for ia, ib in zip(a["items"], b["items"]):
            for k in ia["views"]:
                assert abs(ia["views"][k]["cos"] - ib["views"][k]["cos"]) < 0.05, (name, k)


def test_iou_refinement_recovers_perturbed_box(scenes):
    sc = scenes["can"]
    fg = np.asarray(sc["fg_box"], np.float64)
    w, h = fg[2] - fg[0], fg[3] - fg[1]
    bad = [fg[0] + 0.05 * w, fg[1] - 0.03 * h, fg[2] + 0.02 * w, fg[3] - 0.04 * h]
    mask0 = sc["views"][0]["mask"]
    aff0, info0 = an.fit_photo_to_view0(sc["photo_mask"], bad, mask0, refine=False)
    aff1, info1 = an.fit_photo_to_view0(sc["photo_mask"], bad, mask0, refine=True)
    print("iou bbox", info0["iou"], "refined", info1["iou"])
    assert info1["iou"] > info0["iou"] + 0.03 and info1["iou"] > 0.97
    # the refined map puts a photo point back near its true view-0 location
    true = C.apply_affine([[400.0, 500.0]], C.fit_box_affine(sc["fg_box"], C.bbox_of_mask(mask0)))
    assert np.linalg.norm(C.apply_affine([[400.0, 500.0]], aff1) - true) < \
        np.linalg.norm(C.apply_affine([[400.0, 500.0]], aff0) - true)


def test_rescale_doc(scenes):
    doc, _ = _lift(scenes["box"])
    d2 = an.rescale_doc(doc, 1024)
    assert d2["res"] == 1024
    v, v2 = doc["items"][0]["views"]["1"], d2["items"][0]["views"]["1"]
    assert np.allclose(np.asarray(v2["bbox"]), 2 * np.asarray(v["bbox"]), atol=0.02)
    assert v2["pixels"] == 4 * v["pixels"] and abs(v2["height_px"] - 2 * v["height_px"]) < 0.02


# ────────────────────────────────────────────────────────────────────────────
# Glyph conditions from a lifted layout
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", gid.ANCHOR_MODES)
def test_glyph_infer_ids_stay_in_slots(scenes, mode):
    doc, _ = _lift(scenes["can"])
    cfg = gl.GlyphConfig(anchor_mode=mode, min_height_px=4.0, warn_collisions=False)
    inst = gl.build_glyphs_infer(doc, cfg)
    assert inst, mode
    assert {i.raw_view for i in inst} >= {0, 1}
    n = gid.view_tokens()
    for i in inst:
        ids = i.ids[i.keep]
        assert np.isfinite(ids).all() and (ids[:, 0] == cfg.frame).all()
        rlo, rhi, clo, chi = gid.slot_bounds(i.raw_view)
        assert ids[:, 1].min() >= rlo and ids[:, 1].max() <= rhi
        assert ids[:, 2].min() >= clo and ids[:, 2].max() <= chi, (mode, i.raw_view)
        assert len(i.ids) == i.token_hw[0] * i.token_hw[1] and i.patch.size == (i.token_hw[1] * 16, i.token_hw[0] * 16)
    ids, slices = gid.concat_ids(inst)
    assert ids.dtype == np.float32 and ids[:, 2].max() < 6 * n


# ────────────────────────────────────────────────────────────────────────────
# CLI on a synthetic eval dir
# ────────────────────────────────────────────────────────────────────────────

def test_cli_eval_dir(scenes, tmp_path):
    sc = scenes["can"]
    ev = tmp_path / "eval"
    sku = "000000000001"
    d = ev / sku
    d.mkdir(parents=True)
    H, W = sc["photo_mask"].shape
    photo = np.full((H, W, 3), 255, np.uint8)
    photo[sc["photo_mask"]] = (30, 60, 160)
    S = max(H, W)
    ref = np.full((S, S, 3), 255, np.uint8)
    pad = ((S - W) // 2, (S - H) // 2)
    ref[pad[1]:pad[1] + H, pad[0]:pad[0] + W] = photo
    Image.fromarray(ref).save(d / "ref.png")
    meta = {"sku": sku, "orig_size": [W, H], "square_size": S, "pad": list(pad), "bg_color": [255, 255, 255],
            "fg_bbox_photo": sc["fg_box"]}
    (d / "ref_meta.json").write_text(json.dumps(meta))
    lines = [{"id": i, "text": it["text"], "conf": it["conf"], "quad": it["quad"]} for i, it in enumerate(sc["items"])]
    lines.append({"id": len(lines), "text": "faint", "conf": 0.2, "quad": sc["items"][0]["quad"]})
    (d / "gt_ocr.json").write_text(json.dumps({"lines": lines}))
    (ev / "skus.txt").write_text(sku + "\n")
    out = tmp_path / "out"
    rc = an.main(["--eval-dir", str(ev), "--skus", "all", "--geometry", "mvgen",
                  "--mesh-views", str(sc["mvgen"]), "--out-dir", str(out), "--debug"])
    assert rc == 0
    doc = json.loads((out / sku / "text.json").read_text())
    assert doc["source"] == "photo-lift" and doc["res"] == RES
    assert [it["text"] for it in doc["items"]] == [it["text"] for it in sc["items"]]   # gt source drops conf 0.2
    assert doc["lift"]["align"]["iou"] > 0.99 and doc["lift"]["photo_mask_source"].startswith("border")
    assert (out / sku / "anchors_debug.png").exists()
    summ = json.loads((out / "anchors_summary.json").read_text())
    assert summ["skus"][sku]["items_per_view"]["1"] == 2
    # ocr source keeps the low-confidence line, flagged (single-SKU form with --out)
    rc = an.main(["--eval-dir", str(ev), "--sku", sku, "--mesh-views", str(sc["mvgen"]),
                  "--text-source", "ocr", "--out", str(tmp_path / "t.json")])
    doc2 = json.loads((tmp_path / "t.json").read_text())
    assert rc == 0 and "low_conf" in doc2["items"][-1]["flags"]
    gl.items_of(doc2)                                  # glyph accepts the file (res check)


def test_parse_ocr_json_formats(tmp_path):
    q = [[1, 2], [11, 2], [11, 8], [1, 8]]
    items = [{"text": "PURE", "conf": 0.9, "quad": q}, {"text": " ", "conf": 0.9, "quad": q},
             {"text": "X", "quad": None}]
    for i, d in enumerate((items, {"lines": items}, {"items": items}, {"backend": "vision", "results": {"a.png": items}})):
        p = tmp_path / f"o{i}.json"
        p.write_text(json.dumps(d))
        out = an.parse_ocr_json(p)
        assert [o["text"] for o in out] == ["PURE"] and out[0]["quad"] == q and out[0]["conf"] == 0.9


# ────────────────────────────────────────────────────────────────────────────
# Adversarial: independent UniTEX cameras, thin products, edges
# ────────────────────────────────────────────────────────────────────────────

# TextureTools generate_box_views_c2ws(radius=2.8), copied verbatim (glTF frame, order f r b l t d).
_TT_R = 2.8
TT_BOX_C2WS = np.array([
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, _TT_R], [0, 0, 0, 1]],
    [[0, 0, 1, _TT_R], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -_TT_R], [0, 0, 0, 1]],
    [[0, 0, -1, -_TT_R], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    [[1, 0, 0, 0], [0, 0, 1, _TT_R], [0, -1, 0, 0], [0, 0, 0, 1]],
    [[-1, 0, 0, 0], [0, 0, -1, -_TT_R], [0, -1, 0, 0], [0, 0, 0, 1]],
], np.float64)


def _render_texturetools(shape, d):
    """UniTEX export_condition re-implemented from TextureTools alone (no unitex.common cameras):
    ortho intrinsics scale 1, intr_to_proj with the nvdiffrast y flip (row 0 = camera +y), c2ws
    reordered frbltd -> frtbld, 2x3 grid without any roll, grey background, trunc to uint8."""
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    ys, xs = np.mgrid[0:RES, 0:RES].astype(np.float64) + 0.5
    x_cam, y_cam = 2 * xs / RES - 1, -(2 * ys / RES - 1)
    tiles = {"alpha": [], "ccm": [], "normal": []}
    for tt in (0, 1, 4, 2, 3, 5):
        c2w = TT_BOX_C2WS[tt]
        o_g = np.stack([x_cam, y_cam, np.zeros_like(x_cam)], -1) @ c2w[:3, :3].T + c2w[:3, 3]
        d_g = np.broadcast_to(-c2w[:3, 2], o_g.shape)
        ok, _, P_b, n_b, _ = shape.hit(C.gltf_to_blender(o_g), C.gltf_to_blender(d_g))
        P_g, n_g = C.blender_to_gltf(P_b), C.blender_to_gltf(n_b)
        tiles["alpha"].append((ok.astype(np.uint8) * 255)[..., None])
        tiles["ccm"].append(np.where(ok[..., None], _trunc(P_g), 127).astype(np.uint8))
        tiles["normal"].append(np.where(ok[..., None], _trunc(n_g), 127).astype(np.uint8))
    for name, fn in (("alpha", "mv_alpha.png"), ("ccm", "mv_ccm.png"), ("normal", "mv_normal.png")):
        t = np.asarray(tiles[name])                                     # (6, H, W, c) row-major 2x3
        g = t.reshape(2, 3, RES, RES, -1).transpose(0, 2, 1, 3, 4).reshape(2 * RES, 3 * RES, -1)
        Image.fromarray(g[..., 0] if name == "alpha" else g, "L" if name == "alpha" else "RGB").save(d / fn)
    return d


def test_unitex_frames_from_texturetools_cameras(scenes, tmp_path):
    """The UniTEX cache path must agree with the raw Blender cameras without sharing their code:
    masks equal and depths within one 8-bit bin in every raw view (catches a wrong tile order,
    a missing or extra 180-degree roll of the bottom tile, or a glTF / Blender axis slip)."""
    for name in ("box", "can"):
        sc = scenes[name]
        geo_tt = an.load_unitex_cache(_render_texturetools(sc["shape"], tmp_path / name))
        geo_mv = _geo(sc, "mvgen")
        for k in range(6):
            m_tt, m_mv = geo_tt.masks[k], geo_mv.masks[k]
            assert (m_tt != m_mv).sum() == 0, (name, k, int((m_tt != m_mv).sum()))
            dd = np.abs(geo_tt.depths[k] - geo_mv.depths[k])[m_mv]
            assert dd.max() < 1.01 * an.BIN, (name, k, float(dd.max()))
            cosd = np.abs((geo_tt.normals[k] * geo_mv.normals[k]).sum(-1))[m_mv]
            assert cosd.min() > 0.99, (name, k, float(cosd.min()))
        a, _ = _lift(sc, "mvgen")
        b = an.lift(sc["items"], sc["photo_mask"], sc["fg_box"], geo_tt, an.LiftConfig())
        for ia, ib in zip(a["items"], b["items"]):
            assert set(ia["views"]) == set(ib["views"]), (name, ia["text"])
            for k in ia["views"]:
                assert ia["views"][k]["bbox"] == ib["views"][k]["bbox"], (name, k)


def test_thin_product_front_text_not_on_back(tmp_path):
    """A flat pouch / card: front and back faces closer than the depth tolerance (3 bins = 0.0235).
    Front text must not be lifted onto the back view (it would be painted mirrored there), and
    the other views must still agree with the analytic ground truth."""
    slab = Box(half=(0.6, 0.008, 0.9), yaw=20.0)
    views = render(slab)
    sc = dict(shape=slab, views=views, specs=[{"name": "front", "reg": {"u": (-0.3, 0.3), "z": (0.1, 0.3)},
                                               "orient": 0}])
    pmask, fg = photo_of(views)
    sc.update(photo_mask=pmask, fg_box=fg, items=scene_items(slab, sc["specs"]),
              mvgen=write_mvgen(tmp_path / "slab", views))
    doc, _ = _lift(sc)
    it = doc["items"][0]
    print("thin slab views:", {k: (v["pixels"], v["cos"]) for k, v in it["views"].items()})
    assert "2" not in it["views"], it["views"]["2"]["bbox"]
    _check_views(sc, doc, 0, ["0", "1"])


def test_inward_normal_maps_do_not_change_the_lift(scenes, tmp_path):
    """A mesh with inverted winding renders inward vertex normals (UniTEX world_normal, Blender
    normal pass). Visibility must not depend on the normal map's sign, cos is |n . v|."""
    sc = scenes["can"]
    flipped = [dict(v, normal=-v["normal"]) for v in sc["views"]]
    geo_f = an.load_mvgen_views(write_mvgen(tmp_path / "flipped", flipped))
    a, _ = _lift(sc)
    b = an.lift(sc["items"], sc["photo_mask"], sc["fg_box"], geo_f, an.LiftConfig())
    for ia, ib in zip(a["items"], b["items"]):
        assert set(ia["views"]) == set(ib["views"]), ia["text"]
        for k in ia["views"]:
            assert ia["views"][k]["bbox"] == ib["views"][k]["bbox"]
            assert abs(ia["views"][k]["cos"] - ib["views"][k]["cos"]) < 0.01


def test_text_at_box_top_edge_not_on_top_view(scenes):
    """Front text touching the top edge of a box front face: the push lands on the rim of the top
    face and the pull finds top-face pixels within the depth tolerance of the front edge. Neither
    may create a raw 4 (top) entry, since the front face is edge-on from above."""
    sc = scenes["box"]
    reg = {"u": (-0.2, 0.3), "z": (0.7, 0.8)}                    # top of the front face (h_z 0.8)
    q = view0_quad(sc["shape"], reg)
    items = [{"id": 0, "text": "TOP EDGE", "conf": 0.99, "quad": to_photo(q)}]
    doc, _ = _lift(sc, items=items)
    it = doc["items"][0]
    print("top edge views:", {k: (v["pixels"], v["cos"], v["height_px"]) for k, v in it["views"].items()})
    assert "4" not in it["views"], it["views"]["4"]
    assert set(it["views"]) == {"0", "1"}
