"""
Tests for text_regions.py on synthetic renders whose per-view boxes are known analytically.

Scenes (512 px views, mvgen file layout, uv stored float16 with Blender's v-up convention):
  flat      raw 0 shows a planar label through an affine view -> texture map (with shear):
            a horizontal item, a vertical item reading upwards, an item cut by the silhouette,
            a tiny item and a small item nested inside a big one
  cylinder  raw 0 (front) and raw 1 (right) look at a vertical cylinder whose texture u is the
            azimuth: one item fully visible in front, one wrapping past the front limb that the
            right view sees whole
  twin      the same texels rendered twice in one view (a copy far from the label), which the
            component filter must drop

The end-to-end tests build a GLB with the texture in its BIN chunk and run process_sku / main
with a scripted OCR backend or the json backend. One test uses Apple Vision (macOS only).

Run: python -m pytest unitex/tests/test_text_regions.py
"""

import io
import json
import math
import os
import platform
import shutil
import struct
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import text_regions as tr
from unitex import glyph as gl
from unitex.common import VIEW_RES

RES = VIEW_RES


# ────────────────────────────────────────────────────────────────────────────
# Synthetic renders
# ────────────────────────────────────────────────────────────────────────────

def _pix_centres(res=RES):
    yy, xx = np.mgrid[0:res, 0:res]
    return xx + 0.5, yy + 0.5


def _empty_view(res=RES):
    return {"texel": np.zeros((res, res, 2)), "mask": np.zeros((res, res), bool),
            "normal": np.zeros((res, res, 3))}


def _normal_png(n, mask):
    rgb = (np.clip((n + 1.0) / 2.0, 0, 1) * 255.0).astype(np.uint8)   # mvgen quantize_data
    rgb[~mask] = 255
    return rgb


def write_render(d, views, tex_rgb, glb_path, sku="synth", flip_uv=False):
    """views: 6 dicts {texel (H, W, 2) continuous texture px, mask, normal n_cam}."""
    d.mkdir(parents=True, exist_ok=True)
    Ht, Wt = tex_rgb.shape[:2]
    for i, v in enumerate(views):
        tx, ty = v["texel"][..., 0], v["texel"][..., 1]
        uv = np.stack([tx / Wt, (ty / Ht) if flip_uv else 1.0 - ty / Ht], -1)
        uv[~v["mask"]] = 0.0
        np.savez_compressed(d / f"{i:04d}_uv.npz", uv=uv.astype(np.float16), mask=v["mask"])
        Image.fromarray(_normal_png(v["normal"], v["mask"])).save(d / f"{i:04d}_normal.png")
        alb = np.full((RES, RES, 3), 255, np.uint8)
        ix = np.clip(np.floor(tx).astype(int), 0, Wt - 1)
        iy = np.clip(np.floor(ty).astype(int), 0, Ht - 1)
        alb[v["mask"]] = tex_rgb[iy[v["mask"]], ix[v["mask"]]]
        Image.fromarray(alb).save(d / f"{i:04d}_albedo.png")
    with open(d / "metadata.json", "w") as f:
        json.dump({"res": RES, "sku": sku, "source_glb": str(glb_path), "yaw_deg": 0.0}, f)
    return d


def write_glb(path, tex_rgb, prefix=b"\x07" * 20, extra=None):
    """Minimal GLB: one material -> texture -> image -> bufferView at a non-zero offset."""
    buf = io.BytesIO()
    Image.fromarray(tex_rgb).save(buf, "PNG")
    png = buf.getvalue()
    binc = prefix + png
    binc += b"\x00" * (-len(binc) % 4)
    gltf = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binc)}],
            "bufferViews": [{"buffer": 0, "byteOffset": len(prefix), "byteLength": len(png)}],
            "images": [{"bufferView": 0, "mimeType": "image/png"}],
            "textures": [{"source": 0}],
            "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}]}
    if extra:
        extra(gltf)
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binc)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<II", len(js), tr.CHUNK_JSON) + js)
        f.write(struct.pack("<II", len(binc), tr.CHUNK_BIN) + binc)
    return path


def _texture(W, H, seed=0):
    """Blocky random texture: every flip or offset of the uv lookup changes most pixels."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (H // 16 + 1, W // 16 + 1, 3), dtype=np.uint8)
    return np.repeat(np.repeat(small, 16, 0), 16, 1)[:H, :W].copy()


# flat scene: texture px = A (xy - 60) + 40 inside the view square [60, 452)^2
FLAT_A = np.array([[1.8, 0.3], [0.25, 1.7]])
FLAT_TEX = (1024, 1024)
FLAT_ITEMS = {
    "HELPER": [[200, 150], [500, 150], [500, 210], [200, 210]],
    "TETRAZZINI": [[700, 600], [700, 300], [760, 300], [760, 600]],     # reads upwards
    "PARTIAL": [[600, 400], [900, 400], [900, 440], [600, 440]],        # cut near X 795
    "TINY": [[100, 700], [103, 700], [103, 703], [100, 703]],
    "INNER": [[300, 160], [340, 160], [340, 200], [300, 200]],           # inside HELPER
}
FLAT_NZ = np.array([0.36, 0.0, 0.933])


def flat_view_xy(tex_pts):
    """Texture px -> view px, the inverse of the flat scene map."""
    p = np.asarray(tex_pts, np.float64) - 40.0
    return p @ np.linalg.inv(FLAT_A).T + 60.0


def flat_views():
    x, y = _pix_centres()
    mask = (x > 60) & (x < 452) & (y > 60) & (y < 452)
    texel = np.stack([x - 60, y - 60], -1) @ FLAT_A.T + 40.0
    n = np.broadcast_to(FLAT_NZ / np.linalg.norm(FLAT_NZ), (RES, RES, 3)).copy()
    views = [_empty_view() for _ in range(6)]
    views[0] = {"texel": texel, "mask": mask, "normal": n}
    return views


# cylinder scene: radius R px around x = CX, texture X = (phi + 180) / 360 * W, Y = 40 + 1.5 (y - 60)
CX, CR = 256.0, 180.0
CYL_TEX = (2048, 1024)
CYL_K = 1.5


def cyl_X(phi_deg):
    return (np.asarray(phi_deg, np.float64) + 180.0) / 360.0 * CYL_TEX[0]


CYL_ITEMS = {
    "FRONTWORD": [[cyl_X(-40), 300], [cyl_X(50), 300], [cyl_X(50), 360], [cyl_X(-40), 360]],
    "WRAPWORD": [[cyl_X(40), 500], [cyl_X(130), 500], [cyl_X(130), 560], [cyl_X(40), 560]],
}


def cyl_view(alpha_deg):
    x, y = _pix_centres()
    sn = (x - CX) / CR
    mask = (np.abs(sn) < 1.0) & (y > 60) & (y < 452)
    theta = np.degrees(np.arcsin(np.clip(sn, -1, 1)))
    texel = np.stack([cyl_X(alpha_deg + theta), 40.0 + CYL_K * (y - 60.0)], -1)
    th = np.radians(theta)
    n = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], -1)
    return {"texel": texel, "mask": mask, "normal": n}


def cyl_truth(item, alpha_deg, s, t):
    """View xy of reading-frame point (s, t) of a cylinder item, NaN when behind the limb."""
    q = np.asarray(CYL_ITEMS[item])
    X = q[0, 0] + s * (q[1, 0] - q[0, 0])
    Y = q[0, 1] + t * (q[3, 1] - q[0, 1])
    phi = X / CYL_TEX[0] * 360.0 - 180.0
    th = np.radians(phi - alpha_deg)
    x = np.where(np.abs(th) < np.pi / 2, CX + CR * np.sin(th), np.nan)
    return np.stack([x, 60.0 + (Y - 40.0) / CYL_K + 0 * x], -1)


def _lines(items):
    return [{"text": k, "conf": 0.95, "quad": v} for k, v in items.items()]


def _project(items_dict, views, tex_wh, min_pixels=8):
    items = tr.build_items(_lines(items_dict))
    idmap = tr.instance_map([it["src_quad"] for it in items], tex_wh)
    vv = []
    for v in views:
        W, H = tex_wh
        uv = np.stack([v["texel"][..., 0] / W, 1.0 - v["texel"][..., 1] / H], -1).astype(np.float16)
        vv.append((uv.astype(np.float64), v["mask"], tr.normal_cos(_normal_png(v["normal"], v["mask"])), None))
    counts = tr.project_items(items, idmap, tex_wh, (tr.WRAP_REPEAT, tr.WRAP_REPEAT), vv, min_pixels)
    return {it["text"]: it for it in items}, counts


def _grid(view):
    return gl.gid.parse_grid(view["grid"])


# ────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ────────────────────────────────────────────────────────────────────────────

def test_inverse_bilinear_roundtrip():
    quad = [[10.0, 20.0], [300.0, 5.0], [320.0, 90.0], [-5.0, 70.0]]      # not a parallelogram
    rng = np.random.default_rng(1)
    st = rng.random((500, 2))
    p = tr.bilerp(quad, st[:, 0], st[:, 1])
    back = tr.inverse_bilinear(p, quad)
    assert np.abs(back - st).max() < 1e-9
    par = [[0.0, 0.0], [100.0, 10.0], [100.0, 50.0], [0.0, 40.0]]       # parallelogram (linear branch)
    back = tr.inverse_bilinear(tr.bilerp(par, st[:, 0], st[:, 1]), par)
    assert np.abs(back - st).max() < 1e-9


def test_instance_map_small_quad_wins():
    idmap = tr.instance_map([[[10, 10], [60, 10], [60, 40], [10, 40]], [[20, 15], [30, 15], [30, 25], [20, 25]]],
                            (80, 50))
    assert idmap[20, 25] == 1 and idmap[12, 12] == 0 and idmap[45, 70] == -1
    assert (idmap == 1).sum() == 100            # 10 x 10 texels, pixel-centre convention


def test_component_labelling_matches_scipy():
    pytest.importorskip("scipy")
    from scipy.ndimage import label
    rng = np.random.default_rng(3)
    mask = rng.random((40, 60)) < 0.45
    ours, n_ours = tr._label_numpy(mask)
    ref, n_ref = label(mask, structure=np.ones((3, 3), int))
    assert n_ours == n_ref
    # same partition: a bijection between label ids
    pairs = set(zip(ours[mask].tolist(), ref[mask].tolist()))
    assert len(pairs) == n_ref


def test_flags_and_leak_words():
    leak = lambda s: "template_leak" in tr.item_flags(s, 0.9, 0.0)
    assert leak("LEFT") and leak("left side") and leak("BACK.") and leak("RIGHT_PAD") and leak("Top")
    assert leak("BоTтоM")                       # Cyrillic look-alikes (a real Vision reading)
    assert not leak("Leftover") and not leak("BOX TOPS") and not leak("BACKPACK")
    assert tr.item_flags("x", 0.49, 0.0) == ["low_conf"]
    assert tr.item_flags("x", 0.5, 9.0) == []
    assert tr.item_flags("x", 0.9, -90.0) == ["rotated"]
    assert tr.reading_angle(FLAT_ITEMS["TETRAZZINI"]) == pytest.approx(-90.0)


# ────────────────────────────────────────────────────────────────────────────
# Flat label
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def flat():
    return _project(FLAT_ITEMS, flat_views(), FLAT_TEX)


def test_flat_fully_visible_item(flat):
    items, counts = flat
    v = items["HELPER"]["views"]["0"]
    q_true = flat_view_xy(FLAT_ITEMS["HELPER"])
    assert np.abs(np.asarray(v["quad"]) - q_true).max() < 0.5
    tl, tr_, br, bl = q_true
    assert v["height_px"] == pytest.approx(np.linalg.norm(bl - tl), abs=0.3)
    assert v["coverage"] == 1.0
    assert v["st_range"] == [0.0, 1.0, 0.0, 1.0]
    assert v["cos"] == pytest.approx(0.933 / np.linalg.norm(FLAT_NZ), abs=0.006)
    lo, hi = q_true.min(0), q_true.max(0)
    assert np.abs(np.asarray(v["bbox"]) - [lo[0], lo[1], hi[0], hi[1]]).max() < 1.0
    # grid cell centres: every cell, including the ones INNER took the texels of (filled holes)
    G = _grid(v)
    assert np.isfinite(G).all()
    S, T = np.meshgrid((np.arange(16) + 0.5) / 16, (np.arange(4) + 0.5) / 4)
    truth = flat_view_xy(tr.bilerp(FLAT_ITEMS["HELPER"], S, T))
    assert np.abs(G - truth).max() < 0.35
    assert set(counts) == {"0", "1", "2", "3", "4", "5"} and counts["1"] == 0


def test_flat_nested_item_has_its_own_box(flat):
    items, _ = flat
    v = items["INNER"]["views"]["0"]
    assert np.abs(np.asarray(v["quad"]) - flat_view_xy(FLAT_ITEMS["INNER"])).max() < 0.5


def test_flat_vertical_item_keeps_reading_order(flat):
    items, _ = flat
    v = items["TETRAZZINI"]["views"]["0"]
    q = np.asarray(v["quad"])
    assert np.abs(q - flat_view_xy(FLAT_ITEMS["TETRAZZINI"])).max() < 0.5
    assert q[1, 1] < q[0, 1] - 100               # TL -> TR runs up the view
    assert gl._view_orientation(v) == 270        # glyph.py reads it as reading upwards
    assert "rotated" in items["TETRAZZINI"]["flags"]


def test_flat_partial_item(flat):
    items, _ = flat
    v = items["PARTIAL"]["views"]["0"]
    # the silhouette x = 452 crosses the item at texture X 791.8 (top) .. 798.9 (bottom)
    s_cut = (np.array([791.8, 798.9]) - 600.0) / 300.0
    assert s_cut.min() - 0.02 < v["st_range"][1] < s_cut.max() + 0.02
    assert v["coverage"] == pytest.approx(11 / 16)
    G = _grid(v)
    assert np.isfinite(G[:, :10]).all() and not np.isfinite(G[:, 11:]).any()
    assert v["bbox"][2] <= 452.0 + 1e-6


def test_flat_tiny_item_dropped(flat):
    items, _ = flat
    assert items["TINY"]["views"] == {}          # about 3 pixels < --min-pixels 8


# ────────────────────────────────────────────────────────────────────────────
# Cylinder
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cyl():
    views = [_empty_view() for _ in range(6)]
    views[0], views[1] = cyl_view(0.0), cyl_view(90.0)
    return _project(CYL_ITEMS, views, CYL_TEX)


def _cell_truth(item, alpha):
    S, T = np.meshgrid((np.arange(16) + 0.5) / 16, (np.arange(4) + 0.5) / 4)
    return cyl_truth(item, alpha, S, T), S


def test_cylinder_front_item(cyl):
    items, _ = cyl
    v = items["FRONTWORD"]["views"]["0"]
    truth, _ = _cell_truth("FRONTWORD", 0.0)
    G = _grid(v)
    assert np.isfinite(G).all()
    assert np.abs(G - truth).max() < 0.6
    corners = cyl_truth("FRONTWORD", 0.0, np.array([0, 1, 1, 0.0]), np.array([0, 0, 1, 1.0]))
    assert np.abs(np.asarray(v["quad"]) - corners).max() < 1.0
    assert v["coverage"] == 1.0
    assert v["height_px"] == pytest.approx(60 / CYL_K, abs=0.3)
    xs = np.linspace(CX + CR * math.sin(math.radians(-40)), CX + CR * math.sin(math.radians(50)), 2001)
    assert v["cos"] == pytest.approx(np.sqrt(1 - ((xs - CX) / CR) ** 2).mean(), abs=0.01)
    # baseline spacing follows the surface: cells shrink towards the limb at +50 deg
    ds = np.diff(G[0, :, 0])
    assert ds[0] > 1.1 * ds[-1]


def test_cylinder_wrapping_item_front_and_side(cyl):
    items, _ = cyl
    front = items["WRAPWORD"]["views"]["0"]
    # visible up to the limb at 90 deg: s < (90 - 40) / 90 = 0.556
    assert 7 / 16 <= front["coverage"] <= 9 / 16
    # the last 4 deg before the limb fall inside the last half pixel, so s_hi stops near 0.51
    assert 0.5 <= front["st_range"][1] <= 0.56
    G = _grid(front)
    truth, S = _cell_truth("WRAPWORD", 0.0)
    assert not np.isfinite(G[:, 9:]).any()
    near = 40 + S * 90 < 75                                   # away from the limb
    ok = near & np.isfinite(G[..., 0])
    assert ok.sum() >= 4 * 5 and np.abs(G[ok] - truth[ok]).max() < 1.0
    side = items["WRAPWORD"]["views"]["1"]
    assert side["coverage"] == 1.0 and np.isfinite(_grid(side)).all()
    truth1, _ = _cell_truth("WRAPWORD", 90.0)
    assert np.abs(_grid(side) - truth1).max() < 0.6
    assert side["cos"] > front["cos"]
    # FRONTWORD from the right camera: theta in [-130, -40], visible for s >= 0.444
    fw = items["FRONTWORD"]["views"]["1"]
    assert 40 / 90 <= fw["st_range"][0] <= 0.5
    assert 8 / 16 <= fw["coverage"] <= 10 / 16


# ────────────────────────────────────────────────────────────────────────────
# Component filter (the same texels rendered twice)
# ────────────────────────────────────────────────────────────────────────────

def test_duplicate_texels_elsewhere_are_dropped():
    views = flat_views()
    v = views[0]
    # copy a strip of HELPER's top rows (texture Y 150..165) to a distant patch of the view
    x, y = _pix_centres()
    patch = (x > 400) & (x < 440) & (y > 420) & (y < 440)
    v["mask"] = v["mask"] | patch
    v["texel"][patch] = np.stack([300 + (x[patch] - 400) * 5, 150 + (y[patch] - 420) * 0.7], -1)
    items, _ = _project({"HELPER": FLAT_ITEMS["HELPER"]}, views, FLAT_TEX)
    e = items["HELPER"]["views"]["0"]
    assert e["dropped_px"] > 300
    assert np.abs(np.asarray(e["quad"]) - flat_view_xy(FLAT_ITEMS["HELPER"])).max() < 0.5
    assert e["bbox"][3] < 300


def test_measure_view_split_by_gap_keeps_both_parts():
    # a flat item with a vertical band of pixels removed: both halves agree with one map
    st = np.stack(np.meshgrid(np.linspace(0.005, 0.995, 200), np.linspace(0.01, 0.99, 20)), -1).reshape(-1, 2)
    xy = np.stack([100 + st[:, 0] * 200, 50 + st[:, 1] * 20], -1)
    keep = (st[:, 0] < 0.4) | (st[:, 0] > 0.55)
    xy = np.floor(xy[keep]) + 0.5
    e = tr.measure_view(st[keep], xy, np.ones(keep.sum()), min_pixels=8)
    assert e["dropped_px"] == 0
    assert e["coverage"] == 15 / 16                               # bin 7 lies inside the gap
    assert np.isfinite(gl.gid.parse_grid(e["grid"])).all()      # the gap cells are filled


# ────────────────────────────────────────────────────────────────────────────
# GLB texture, template provenance
# ────────────────────────────────────────────────────────────────────────────

def test_glb_texture_extraction(tmp_path):
    tex = _texture(96, 64)
    glb = write_glb(tmp_path / "a.glb", tex)
    img, info = tr.base_color_texture(glb)
    assert np.array_equal(np.asarray(img.convert("RGB")), tex)
    assert info["size"] == [96, 64] and info["wrap"] == [tr.WRAP_REPEAT, tr.WRAP_REPEAT]

    # plain .gltf with a data-uri image
    buf = io.BytesIO()
    Image.fromarray(tex).save(buf, "PNG")
    import base64
    gltf = {"asset": {"version": "2.0"}, "images": [{"uri": "data:image/png;base64," +
                                                     base64.b64encode(buf.getvalue()).decode()}],
            "textures": [{"source": 0, "sampler": 0}], "samplers": [{"wrapS": 33071, "wrapT": 33648}],
            "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            "meshes": [{"primitives": [{"attributes": {}, "material": 0}]}]}
    (tmp_path / "b.gltf").write_text(json.dumps(gltf))
    img, info = tr.base_color_texture(tmp_path / "b.gltf")
    assert np.array_equal(np.asarray(img.convert("RGB")), tex) and info["wrap"] == [33071, 33648]

    def two_images(g):
        g["images"].append(dict(g["images"][0]))
        g["textures"].append({"source": 1})
        g["materials"].append({"pbrMetallicRoughness": {"baseColorTexture": {"index": 1}}})
        g["meshes"][0]["primitives"].append({"attributes": {}, "material": 1})
    with pytest.raises(ValueError, match="2 different baseColor images"):
        tr.base_color_texture(write_glb(tmp_path / "c.glb", tex, extra=two_images))

    def texcoord1(g):
        g["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"] = 1
    with pytest.raises(ValueError, match="TEXCOORD_1"):
        tr.base_color_texture(write_glb(tmp_path / "d.glb", tex, extra=texcoord1))


def test_uv_wrap_modes():
    uv = np.array([[1.25, 0.25], [-0.25, 1.5]])
    np.testing.assert_allclose(tr.uv_to_texel(uv, (100, 10)), [[25, 7.5], [75, 5.0]])
    np.testing.assert_allclose(tr.uv_to_texel(uv, (100, 10), (tr.WRAP_MIRROR, tr.WRAP_CLAMP))[:, 0], [75, 25])


def test_template_provenance(tmp_path):
    meta = {"layout": "box", "canvas_w": 1000, "canvas_h": 800, "islands": [
        {"name": "front", "bbox": [100, 100, 300, 400], "shape": "rect"},
        {"name": "left", "bbox": [20, 100, 60, 400], "shape": "rect"},
        {"name": "top", "bbox": [700, 600, 200, 200], "shape": "circle"}]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(meta))
    tpl = tr.load_template(p, (2000, 1600))                 # texture = 2x the canvas
    assert tpl["front"] == [200, 200, 800, 1000]
    assert tr.provenance_of((500, 600), tpl) == "front" and tr.island_of((500, 600), tpl) == "front"
    assert tr.provenance_of((100, 600), tpl) == "generated" and tr.island_of((100, 600), tpl) == "left"
    # circle bbox (x, y) is the centre: texture centre (1400, 1200), radius 200
    assert tr.island_of((1400, 1390), tpl) == "top" and tr.island_of((1590, 1390), tpl) is None
    assert tr.provenance_of((1900, 100), tpl) == "generated"
    assert tr.provenance_of((500, 600), None) == "unknown"

    can = {"layout": "can", "canvas_w": 2048, "canvas_h": 1638, "islands": [
        {"name": "body", "bbox": [40, 40, 1968, 616], "shape": "rect"}]}
    p.write_text(json.dumps(can))
    tpl = tr.load_template(p, (2304, 1856))
    sx = 2304 / 2048
    np.testing.assert_allclose(tpl["front"][0::2], [(40 + 492) * sx, (40 + 984) * sx])
    assert tr.provenance_of(((40 + 700) * sx, 300), tpl) == "front"          # 2nd quarter
    assert tr.provenance_of(((40 + 1500) * sx, 300), tpl) == "generated"


# ────────────────────────────────────────────────────────────────────────────
# End to end: process_sku, verify, CLI, glyph.py
# ────────────────────────────────────────────────────────────────────────────

class ScriptedOCR:
    """Returns fixed lines for the texture, and for every other image (a verify crop) one line
    inside the text band plus a neighbour line above it that the band filter must drop."""
    name = "scripted"

    def __init__(self, tex_lines, tex_size):
        self.tex_lines, self.tex_size, self.crops = tex_lines, tex_size, 0

    def signature(self):
        return {"backend": self.name}

    def recognize_batch(self, images):
        out = []
        for im in images:
            if im.size == self.tex_size:
                out.append([(ln["text"], ln["conf"], np.asarray(ln["quad"], float)) for ln in self.tex_lines])
                continue
            self.crops += 1
            w, h = im.size
            mid = [[0.1 * w, 0.35 * h], [0.9 * w, 0.35 * h], [0.9 * w, 0.65 * h], [0.1 * w, 0.65 * h]]
            top = [[0.1 * w, 0.0], [0.9 * w, 0.0], [0.9 * w, 0.08 * h], [0.1 * w, 0.08 * h]]
            out.append([("HELPER", 0.9, np.array(mid)), ("NEIGHBOUR", 0.9, np.array(top))])
        return out


@pytest.fixture()
def flat_sku(tmp_path):
    tex = _texture(*FLAT_TEX)
    glb = write_glb(tmp_path / "textured.glb", tex)
    d = write_render(tmp_path / "out" / "render" / "cfi" / "synth", flat_views(), tex, glb)
    return tmp_path, d, tex, glb


def test_process_sku_end_to_end(flat_sku):
    tmp, d, tex, glb = flat_sku
    ocr = ScriptedOCR(_lines({"HELPER": FLAT_ITEMS["HELPER"], "LEFT": FLAT_ITEMS["TETRAZZINI"]}), FLAT_TEX)
    cfg = tr.RegionConfig(backend=ocr, rotations=(0,), verify=True)
    doc = tr.process_sku(d, cfg)
    assert doc["source"] == "uv-ocr" and doc["res"] == RES and doc["version"] == 1
    assert doc["uv_check"]["mad"] == 0.0 and doc["uv_check"]["mad_vflip"] > 30
    by = {it["text"]: it for it in doc["items"]}
    assert by["LEFT"]["flags"] == ["template_leak", "rotated"]
    assert by["HELPER"]["provenance"] == "unknown"
    v = by["HELPER"]["views"]["0"]
    assert v["view_ocr_text"] == "HELPER" and v["view_ocr_ned"] == 1.0
    assert by["LEFT"]["views"]["0"]["view_ocr_ned"] < 0.5
    assert ocr.crops == 2
    assert doc["stats"]["template_leak"] == ["LEFT"] and doc["stats"]["items_per_view"]["0"] == 2
    # the written file parses and glyph.py builds instances from it in every anchor mode
    text = tr.dumps_text_json(doc)
    assert json.loads(text) == json.loads(json.dumps(doc))
    for mode in ("center", "stretch", "warp"):
        insts = gl.build_glyphs_infer(json.loads(text), gl.GlyphConfig(anchor_mode=mode, warn_collisions=False))
        assert [i.item_id for i in insts] == [by["HELPER"]["id"]]      # LEFT is dropped as a leak
        ids = insts[0].ids[insts[0].keep]
        assert (ids[:, 0] == 1).all() and np.isfinite(ids).all()
        q = np.asarray(v["quad"])
        assert q[:, 0].min() - 16 <= (ids[:, 2].min() + 0.5) * 16 and (ids[:, 2].max() + 0.5) * 16 <= q[:, 0].max() + 16
    albedo = np.asarray(Image.open(d / "0000_albedo.png").convert("RGB"))
    insts = gl.build_glyphs(json.loads(text), [albedo] + [None] * 5, 0, 0,
                            gl.GlyphConfig(anchor_mode="warp", p_all_drop=0.0, p_item_drop=0.0))
    assert len(insts) == 1
    img = tr.draw_debug(d, doc)
    assert img.size[0] == 6 * RES


def test_uv_flip_is_detected(tmp_path):
    tex = _texture(*FLAT_TEX)
    glb = write_glb(tmp_path / "t.glb", tex)
    d = write_render(tmp_path / "r", flat_views(), tex, glb, flip_uv=True)
    cfg = tr.RegionConfig(backend=ScriptedOCR([], FLAT_TEX), rotations=(0,))
    with pytest.raises(RuntimeError, match="uv convention"):
        tr.process_sku(d, cfg)


def test_cli_json_backend_skip_and_errors(flat_sku, capsys):
    tmp, d, tex, glb = flat_sku
    ocr_dir = tmp / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "synth.json").write_text(json.dumps({"w": FLAT_TEX[0], "h": FLAT_TEX[1], "items": [
        {"text": "HELPER", "conf": 0.3, "quad": FLAT_ITEMS["HELPER"]}]}))
    broken = tmp / "out" / "render" / "cfi" / "broken"
    broken.mkdir(parents=True)
    (broken / "metadata.json").write_text(json.dumps({"res": RES, "sku": "broken", "source_glb": "/nope.glb"}))
    root = str(tmp / "out")
    args = ["--root", root, "--backend", "json", "--ocr-json-dir", str(ocr_dir), "--rotations", "0",
            "--debug-dir", str(tmp / "dbg")]
    assert tr.main(args) == 1
    doc = json.loads((d / "text.json").read_text())
    assert doc["items"][0]["flags"] == ["low_conf"] and "0" in doc["items"][0]["views"]
    assert (tmp / "dbg" / "synth_views.png").exists() and (tmp / "dbg" / "synth_texture.png").exists()
    log = (tmp / "out" / "text_regions_errors.log").read_text()
    assert "broken" in log and "FileNotFoundError" in log
    mtime = os.path.getmtime(d / "text.json")
    assert tr.main(args + ["--skus", "synth"]) == 0
    assert os.path.getmtime(d / "text.json") == mtime and "exists, skipped" in capsys.readouterr().out
    # --glbs fallback when source_glb moved
    meta = json.loads((d / "metadata.json").read_text())
    meta["source_glb"] = "/moved/away.glb"
    (d / "metadata.json").write_text(json.dumps(meta))
    (tmp / "glbs" / "synth").mkdir(parents=True)
    shutil.copy(glb, tmp / "glbs" / "synth" / "textured.glb")
    assert tr.main(args + ["--skus", "synth", "--overwrite", "--glbs", str(tmp / "glbs")]) == 0


@pytest.mark.skipif(platform.system() != "Darwin" or not os.path.exists("/usr/bin/swiftc"),
                    reason="Apple Vision OCR needs macOS")
def test_vision_ocr_on_rendered_label(tmp_path):
    """Real OCR: draw words into the texture, render the flat scene, OCR texture and crops."""
    tex = np.full((FLAT_TEX[1], FLAT_TEX[0], 3), 235, np.uint8)
    img = Image.fromarray(tex)
    draw = ImageDraw.Draw(img)
    font = gl.load_font(64)
    draw.text((210, 140), "HELPER", fill=(20, 20, 20), font=font)
    up = Image.new("RGB", (330, 90), (235, 235, 235))
    ImageDraw.Draw(up).text((10, 5), "PASTA", fill=(20, 20, 20), font=font)
    img.paste(up.rotate(90, expand=True), (640, 300))                   # reads upwards
    tex = np.asarray(img)
    glb = write_glb(tmp_path / "t.glb", tex)
    d = write_render(tmp_path / "r", flat_views(), tex, glb)
    doc = tr.process_sku(d, tr.RegionConfig(backend="vision", rotations=(0, 90, 270), verify=True))
    by = {it["text"].upper(): it for it in doc["items"]}
    assert "HELPER" in by and "PASTA" in by, list(by)
    assert "rotated" in by["PASTA"]["flags"] and abs(by["PASTA"]["angle_deg"] + 90) < 5
    for word in ("HELPER", "PASTA"):
        v = by[word]["views"]["0"]
        assert v["view_ocr_ned"] >= 0.8, (word, v["view_ocr_text"])
    assert gl._view_orientation(by["PASTA"]["views"]["0"]) == 270


# ────────────────────────────────────────────────────────────────────────────
# Adversarial (review): winding, occlusion, uv precision, rotated OCR
# ────────────────────────────────────────────────────────────────────────────

def test_inverse_bilinear_random_convex_quads_and_instance_area():
    """Strong trapezoids (perspective-like OCR quads) in both windings: the root picked by t alone
    must still be the one with s in [0, 1], and instance_map must cover the polygon's area."""
    rng = np.random.default_rng(7)
    n_quads = 0
    while n_quads < 300:
        w, h = rng.uniform(5, 400), rng.uniform(5, 100)
        q = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        q = q + rng.normal(0, 1, (4, 2)) * [w, h] * rng.uniform(0, 0.45)
        if rng.random() < 0.5:
            q = q[[1, 0, 3, 2]]                       # mirrored reading frame, opposite winding
        c = [np.cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
        if not (all(v > 0 for v in c) or all(v < 0 for v in c)):
            continue
        n_quads += 1
        st = rng.random((200, 2))
        assert np.abs(tr.inverse_bilinear(tr.bilerp(q, st[:, 0], st[:, 1]), q) - st).max() < 1e-7
    trap = [[100.0, 100.0], [500.0, 160.0], [500.0, 200.0], [100.0, 300.0]]   # 5:1 taper
    idmap = tr.instance_map([trap], (600, 400))
    assert abs((idmap == 0).sum() - tr.quad_area(trap)) < 0.01 * tr.quad_area(trap)


def test_occluded_hole_cells_stay_null():
    """An occluder hides a band of a cylinder label. The two visible parts are one item, but the
    cells behind the occluder are not visible and must stay null (DESIGN.md), not be filled by
    extrapolating across the curved gap (7 px off before the fix)."""
    views = [_empty_view() for _ in range(6)]
    v = cyl_view(0.0)
    x, _ = _pix_centres()
    v["mask"] = v["mask"] & ~((x > 150) & (x < 300))
    views[0] = v
    items, _ = _project({"FRONTWORD": CYL_ITEMS["FRONTWORD"]}, views, CYL_TEX)
    e = items["FRONTWORD"]["views"]["0"]
    G = _grid(e)
    truth, _ = _cell_truth("FRONTWORD", 0.0)
    # cells about 16 px wide: centres well inside the gap are fully hidden, the edge cells are
    # partly visible (hit cells, extrapolated from their own pixels)
    hidden = (truth[..., 0] > 150 + 8) & (truth[..., 0] < 300 - 8)
    vis = (truth[..., 0] < 150) | (truth[..., 0] > 300)
    assert hidden.sum() == 4 * 8 and vis.sum() == 4 * 7
    assert not np.isfinite(G[hidden]).any()
    assert np.isfinite(G[vis]).all() and np.abs(G[vis] - truth[vis]).max() < 0.6
    edge = np.isfinite(G[..., 0]) & ~vis
    assert np.abs(G[edge] - truth[edge]).max() < 1.0
    assert e["coverage"] == pytest.approx(0.5, abs=1 / 16)
    # a small item nested in a big one still gets the big one's holes filled (the view shows
    # the big item's texels there, owned by the small one)
    flat_items, _ = _project(FLAT_ITEMS, flat_views(), FLAT_TEX)
    assert np.isfinite(_grid(flat_items["HELPER"]["views"]["0"])).all()


def test_float16_uv_on_4096_texture_small_text():
    """mvgen stores uv as float16 (spacing 2.4e-4 to 4.9e-4, about 1 to 2 texels at 4096).
    Text 8.4 px tall in the view must still get sub-pixel quads and an unbiased height."""
    W = H = 4096
    A = np.array([[7.5, 0.6], [0.4, 7.2]])
    x, y = _pix_centres()
    mask = (x > 20) & (x < 492) & (y > 20) & (y < 492)
    views = [_empty_view() for _ in range(6)]
    views[0] = {"texel": np.stack([x - 20, y - 20], -1) @ A.T + 30.0, "mask": mask,
                "normal": np.broadcast_to([0.0, 0.0, 1.0], (RES, RES, 3)).copy()}
    quads = {"FARCORNER": [[3000, 3500], [3600, 3500], [3600, 3560], [3000, 3560]],
             "NEARORIGIN": [[100, 100], [700, 100], [700, 160], [100, 160]]}
    items, _ = _project(quads, views, (W, H))
    S, T = np.meshgrid((np.arange(16) + 0.5) / 16, (np.arange(4) + 0.5) / 4)
    for k, q in quads.items():
        v = items[k]["views"]["0"]
        qt = (np.asarray(q, float) - 30.0) @ np.linalg.inv(A).T + 20.0
        assert np.abs(np.asarray(v["quad"]) - qt).max() < 0.2, k
        assert v["height_px"] == pytest.approx(np.linalg.norm(qt[3] - qt[0]), rel=0.01), k
        truth = (tr.bilerp(q, S, T) - 30.0) @ np.linalg.inv(A).T + 20.0
        assert np.abs(_grid(v) - truth).max() < 0.2, k


@pytest.mark.skipif(platform.system() != "Darwin" or not os.path.exists("/usr/bin/swiftc"),
                    reason="Apple Vision OCR needs macOS")
def test_vision_downward_and_upside_down_text(tmp_path):
    """Real OCR on words reading down (+90) and upside down (180). The verify crop is rectified
    in the item's reading frame, so a readable crop proves the quad corner order is right."""
    img = Image.fromarray(np.full((FLAT_TEX[1], FLAT_TEX[0], 3), 235, np.uint8))
    font = gl.load_font(64)
    ImageDraw.Draw(img).text((210, 140), "HELPER", fill=(20, 20, 20), font=font)
    for word, ang, pos in (("NOODLES", -90, (560, 280)), ("SAUCE", 180, (150, 420))):
        up = Image.new("RGB", (330, 90), (235, 235, 235))
        ImageDraw.Draw(up).text((10, 5), word, fill=(20, 20, 20), font=font)
        img.paste(up.rotate(ang, expand=True), pos)
    tex = np.asarray(img)
    glb = write_glb(tmp_path / "t.glb", tex)
    d = write_render(tmp_path / "r", flat_views(), tex, glb)
    doc = tr.process_sku(d, tr.RegionConfig(backend="vision", rotations=(0, 90, 270), verify=True))
    by = {it["text"].upper(): it for it in doc["items"]}
    assert "NOODLES" in by and "SAUCE" in by, list(by)
    assert abs(by["NOODLES"]["angle_deg"] - 90) < 5
    assert abs(abs(by["SAUCE"]["angle_deg"]) - 180) < 5
    for word in ("NOODLES", "SAUCE"):
        v = by[word]["views"]["0"]
        assert v["coverage"] == 1.0
        assert v["view_ocr_ned"] >= 0.8, (word, v["view_ocr_text"])
    # the flat view map is a shear close to identity, so view orientation = texture orientation
    assert gl._view_orientation(by["NOODLES"]["views"]["0"]) == 90
    assert gl._view_orientation(by["SAUCE"]["views"]["0"]) == 180
