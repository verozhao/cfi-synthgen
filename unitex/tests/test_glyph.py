"""
Tests for unitex/glyph.py, glyph_ids.py and glyph_debug.py on a synthetic 6-view scene.

Scene (raw view order front right back left top bottom):
  raw 0  flat front labels of three sizes (48, 28, 12 px)
  raw 1  vertical side label reading top to bottom (angle 90), plus a grazing-angle item
  raw 2  curved can label seen from a camera 25 degrees above the equator (baseline bends),
         partly past the limb (null grid cells), plus a "generated" item
  raw 3  the same can label from the side, 10/16 bins visible
  raw 4  template-leak item, raw 5 low-coverage and tiny items

Run: python -m pytest unitex/tests/test_glyph.py  (GLYPH_DEBUG_OUT=<dir> keeps the debug PNGs)
"""

import json
import math
import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from unitex import glyph as gl
from unitex import glyph_ids as gid
from unitex.common import TOKEN_PX, VIEW_RES, raw_to_slot, stack_strip

RES = VIEW_RES
LABEL_BG = (238, 226, 190)
INK = (120, 20, 20)


# ────────────────────────────────────────────────────────────────────────────
# Synthetic scene
# ────────────────────────────────────────────────────────────────────────────

def _grid_json(G):
    nt, ns = G.shape[:2]
    xy = [[None if not np.isfinite(G[i, j, 0]) else [float(G[i, j, 0]), float(G[i, j, 1])]
           for j in range(ns)] for i in range(nt)]
    return {"ns": ns, "nt": nt, "xy": xy}


def _label_image(text, w, h):
    """Upright label: dark text filling a cream w x h rectangle."""
    img = Image.new("RGB", (int(w), int(h)), LABEL_BG)
    font = gl._ink_font([text], w - 4, h - 4, None)
    l, t, r, b = font.getbbox(text)
    ImageDraw.Draw(img).text(((w - (r - l)) / 2 - l, (h - (b - t)) / 2 - t), text, fill=INK, font=font)
    return img


def _flat_view_entry(bbox, quad, height_px, cos=0.95, coverage=1.0):
    return {"bbox": list(map(float, bbox)), "quad": [list(map(float, p)) for p in quad],
            "pixels": int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) * 0.3), "coverage": coverage,
            "cos": cos, "height_px": float(height_px), "grid": _grid_json(gid.grid_from_quad(quad))}


def _flat(views, raw, text, bbox, orient=0, **kw):
    x0, y0, x1, y1 = bbox
    if orient == 0:
        img = _label_image(text, x1 - x0, y1 - y0)
        quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        h = y1 - y0
    else:   # reads top to bottom
        img = _label_image(text, y1 - y0, x1 - x0).transpose(Image.Transpose.ROTATE_270)
        quad = [[x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        h = x1 - x0
    v = Image.fromarray(views[raw])
    v.paste(img, (int(x0), int(y0)))
    views[raw][:] = np.asarray(v)
    return _flat_view_entry(bbox, quad, h, **kw)


CAN = dict(r=180.0, cx=256.0, cy=256.0, el=math.radians(25), phi0=math.radians(-60),
           phi1=math.radians(105), z_top=30.0, z_bot=0.0)


def _can_point(s, t, phic, c=CAN):
    phi = c["phi0"] + s * (c["phi1"] - c["phi0"])
    z = c["z_top"] - t * (c["z_top"] - c["z_bot"])
    a = phi - phic
    x = c["cx"] + c["r"] * np.sin(a)
    y = c["cy"] - z * np.cos(c["el"]) + c["r"] * np.cos(a) * np.sin(c["el"])
    return x, y, np.cos(a)


def _can(views, raw, text, phic, c=CAN):
    """Paint the curved label into a view by inverse mapping; return its text.json view entry."""
    L = c["r"] * (c["phi1"] - c["phi0"])
    Ht = c["z_top"] - c["z_bot"]
    lab = np.asarray(_label_image(text, int(L), int(Ht)), np.float32)
    ys, xs = np.mgrid[0:RES, 0:RES] + 0.5
    sinv = (xs - c["cx"]) / c["r"]
    inside = np.abs(sinv) < 1
    a = np.arcsin(np.clip(sinv, -1, 1))
    phi = phic + a
    z = (c["cy"] + c["r"] * np.cos(a) * np.sin(c["el"]) - ys) / np.cos(c["el"])
    s = (phi - c["phi0"]) / (c["phi1"] - c["phi0"])
    t = (c["z_top"] - z) / Ht
    m = inside & (s >= 0) & (s < 1) & (t >= 0) & (t < 1)
    v = views[raw]
    v[inside] = (170, 170, 185)                      # can body
    u = np.clip((s[m] * lab.shape[1]).astype(int), 0, lab.shape[1] - 1)
    w = np.clip((t[m] * lab.shape[0]).astype(int), 0, lab.shape[0] - 1)
    v[m] = lab[w, u].astype(np.uint8)

    ns, nt = 16, 4
    S, T = np.meshgrid((np.arange(ns) + 0.5) / ns, (np.arange(nt) + 0.5) / nt)
    x, y, ca = _can_point(S, T, phic, c)
    G = np.stack([x, y], -1)
    G[ca <= 0] = np.nan
    sd = np.linspace(0, 1, 400)
    xs0, ys0, ca0 = _can_point(sd, 0.0, phic, c)
    xs1, ys1, _ = _can_point(sd, 1.0, phic, c)
    vis = ca0 > 0
    s_lo, s_hi = sd[vis].min(), sd[vis].max()
    q = [_can_point(s_lo, 0, phic, c)[:2], _can_point(s_hi, 0, phic, c)[:2],
         _can_point(s_hi, 1, phic, c)[:2], _can_point(s_lo, 1, phic, c)[:2]]
    allx, ally = np.r_[xs0[vis], xs1[vis]], np.r_[ys0[vis], ys1[vis]]
    bins = _can_point((np.arange(16) + 0.5) / 16, 0.5, phic, c)[2]      # cos(a) per s bin
    return {"bbox": [float(allx.min()), float(ally.min()), float(allx.max()), float(ally.max())],
            "quad": [[float(px), float(py)] for px, py in q],
            "pixels": int(m.sum()), "coverage": float((bins > 0).mean()),
            "cos": float(ca0[vis].mean() * np.cos(c["el"])),
            "height_px": float(Ht * np.cos(c["el"])), "grid": _grid_json(G)}


def make_scene():
    views = [np.full((RES, RES, 3), 96, np.uint8) for _ in range(6)]
    views[0][40:480, 80:432] = (60, 110, 170)     # front box face
    views[1][60:460, 280:350] = (60, 110, 170)
    items = []

    def add(text, views_, **kw):
        items.append({"id": len(items), "text": text, "conf": kw.pop("conf", 0.97),
                      "provenance": kw.pop("provenance", "front"), "flags": kw.pop("flags", []),
                      "angle_deg": kw.pop("angle_deg", 0.0), "src_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                      "views": {str(k): v for k, v in views_.items()}})

    add("TUNA HELPER", {0: _flat(views, 0, "TUNA HELPER", [96, 60, 416, 108])})
    add("Cheeseburger Macaroni", {0: _flat(views, 0, "Cheeseburger Macaroni", [120, 130, 392, 158])})
    add("NET WT 5.5 OZ (156g)", {0: _flat(views, 0, "NET WT 5.5 OZ (156g)", [176, 440, 336, 452])})
    add("NUTRITION FACTS", {1: _flat(views, 1, "NUTRITION FACTS", [300, 96, 330, 416], orient=90)},
        angle_deg=90.0, flags=["rotated"])
    add("ORIGINAL RECIPE", {2: _can(views, 2, "ORIGINAL RECIPE", 0.0),
                            3: _can(views, 3, "ORIGINAL RECIPE", math.radians(90))})
    add("FRONT", {4: _flat(views, 4, "FRONT", [200, 200, 312, 240])}, flags=["template_leak"])
    add("GENERATED GIBBERISH", {2: _flat(views, 2, "GENERATED GIBBERISH", [140, 420, 372, 450])},
        provenance="generated")
    add("LOW COVERAGE", {5: _flat(views, 5, "LOW COVERAGE", [150, 100, 360, 130], coverage=0.3)})
    add("tiny", {5: _flat(views, 5, "tiny", [200, 300, 240, 305])})
    add("GRAZING", {1: _flat(views, 1, "GRAZING", [100, 470, 200, 490], cos=0.2)})
    text = {"version": 1, "res": RES, "source": "manual", "items": items}
    return views, text


@pytest.fixture(scope="module")
def scene():
    return make_scene()


@pytest.fixture(scope="module")
def outdir(tmp_path_factory):
    d = os.environ.get("GLYPH_DEBUG_OUT")
    if d:
        os.makedirs(d, exist_ok=True)
        return d
    return str(tmp_path_factory.mktemp("glyph_debug"))


def _cfg(**kw):
    kw.setdefault("warn_collisions", False)
    return gl.GlyphConfig(**kw)


def _check_instance(inst, cfg):
    gh, gw = inst.token_hw
    assert inst.patch.mode == "RGB"
    assert inst.patch.width % TOKEN_PX == 0 and inst.patch.height % TOKEN_PX == 0
    assert inst.patch.size == (gw * TOKEN_PX, gh * TOKEN_PX)
    assert inst.ids.dtype == np.float32 and inst.ids.shape == (gh * gw, 3)
    assert inst.keep.dtype == bool and inst.keep.shape == (gh * gw,)
    assert np.all(inst.ids[:, 0] == cfg.frame)
    k = inst.ids[inst.keep]
    assert np.isfinite(k).all()
    rlo, rhi, clo, chi = gid.slot_bounds(inst.raw_view)
    assert k[:, 1].min() >= rlo and k[:, 1].max() <= rhi
    assert k[:, 2].min() >= clo and k[:, 2].max() <= chi, (inst.item_id, inst.raw_view, k[:, 2])
    if cfg.anchor_mode == "center" and cfg.quantize_ids:
        assert np.all(k[:, 1:] == np.round(k[:, 1:]))
    if cfg.anchor_mode == "warp":
        assert np.isnan(inst.ids[~inst.keep][:, 1:]).all()


# ────────────────────────────────────────────────────────────────────────────
# Rendering
# ────────────────────────────────────────────────────────────────────────────

def test_render_sizes_multiple_of_16():
    for text, box in [("A", (5, 5)), ("TUNA HELPER", (300, 40)), ("NET WT 5.5 OZ", (150, 11)),
                      ("Cheeseburger Macaroni with real cheese sauce", (180, 160)), ("小王子的玫瑰", (60, 60))]:
        for lines in (None, 1, 2, 3):
            p = gl.render_fixed(text, box, lines=lines)
            assert p.width % 16 == 0 and p.height % 16 == 0
        p = gl.render_box(text, box)
        assert p.size == (gl.round16(box[0]), gl.round16(box[1]))
        p = gl.render_box(text, box, single_line=False, fit="contain")
        assert p.width % 16 == 0 and p.height % 16 == 0
        p = gl.render_box(text, box, fit="stretch")
        assert p.width % 16 == 0 and p.height % 16 == 0
    assert gl.render_box("HELPER", (100, 5)).size == (96, 16)       # min 16
    rng = np.random.default_rng(0)
    cfg = gl.GlyphConfig(p_aug_scale=1.0, p_aug_rot=1.0)
    base = gl.render_fixed("TUNA HELPER", (300, 40))
    for _ in range(40):
        a = gl.augment(base, rng, cfg)
        assert a.width % 16 == 0 and a.height % 16 == 0 and a.mode == "RGB"
        assert 0.6 * base.width <= a.width <= 1.45 * base.width


def test_render_fixed_wrap_follows_box_aspect():
    one = gl.render_fixed("Cheeseburger Macaroni", (400, 30))
    assert one.height == 1 * 32 + 16                                  # wide box -> one line
    rows, _ = gl.fixed_layout("Cheeseburger Macaroni with real cheese sauce and pasta", (200, 160))
    assert len(rows) >= 3                                             # squarish box -> wraps
    three = gl.render_fixed("Cheeseburger Macaroni", (400, 30), lines=3)   # override
    assert three.height == 2 * 32 + 16                                # clamped to 2 words
    rows, _ = gl.fixed_layout("A B C D E", (10, 400), lines=3)
    assert len(rows) == 3
    rows, (W, H) = gl.fixed_layout("x " * 60, (4000, 20), max_wh=(512, 512))
    assert W <= 512                                                   # prefers a count that fits


def test_font_logging_and_env(monkeypatch, capsys):
    f = gl.load_font(20)
    assert hasattr(f, "getbbox")
    monkeypatch.setenv(gl.FONT_ENV, "/nonexistent/font.ttf")
    gl.load_font(21)
    assert "not found" in capsys.readouterr().out


def test_font_fallback_without_system_fonts(monkeypatch):
    # a server without Noto / DejaVu / Arial still renders (PIL default font)
    monkeypatch.setattr(gl, "FONT_CANDIDATES", ())
    monkeypatch.delenv(gl.FONT_ENV, raising=False)
    gl._load_font_cached.cache_clear()
    try:
        assert gl.resolve_font_path() is None
        p = gl.render_fixed("TUNA HELPER", (300, 40))
        assert p.width % 16 == 0 and p.height == 48
        assert np.asarray(p).min() < 128                            # something was drawn
        assert gl.render_box("HELPER", (100, 20)).size == (96, 16)
    finally:
        gl._load_font_cached.cache_clear()


def test_crop_gt_axis_aligned_exact_and_rotated(scene):
    views, text = scene
    a = np.zeros((64, 64, 3), np.uint8)
    a[..., 0] = np.arange(64)[None, :] * 4
    a[..., 1] = np.arange(64)[:, None] * 4
    c = np.asarray(gl.crop_gt(a, [[16, 8], [48, 8], [48, 24], [16, 24]], (32, 16)))
    assert np.array_equal(c, a[8:24, 16:48])
    # the vertical side label, cropped with its reading-order quad, reads upright again
    v = text["items"][3]["views"]["1"]
    crop = np.asarray(gl.crop_gt(views[1], v["quad"], (320, 32)).convert("L"), np.float32)
    ref = np.asarray(_label_image("NUTRITION FACTS", 320, 30).convert("L").resize((320, 32)), np.float32)
    corr = np.corrcoef(crop.ravel(), ref.ravel())[0, 1]
    assert corr > 0.9, corr
    # same crop with the quad rotated the wrong way is much worse
    q = v["quad"]
    bad = np.asarray(gl.crop_gt(views[1], [q[2], q[3], q[0], q[1]], (320, 32)).convert("L"), np.float32)
    assert np.corrcoef(bad.ravel(), ref.ravel())[0, 1] < 0.5


def test_crop_gt_alpha_composites_white():
    a = np.zeros((32, 32, 4), np.uint8)
    a[..., 3] = 0
    a[8:24, 8:24] = (10, 20, 30, 255)
    c = np.asarray(gl.crop_gt(a, [[0, 0], [32, 0], [32, 32], [0, 32]], (32, 32)))
    assert (c[0, 0] == 255).all() and (c[16, 16] == (10, 20, 30)).all()


def test_crop_gt_grid_follows_curved_label(scene):
    views, text = scene
    v = text["items"][4]["views"]["2"]
    G, gh, gw = gl.warp_geometry(v, gl.GlyphConfig(anchor_mode="warp"))
    crop = np.asarray(gl.crop_gt_grid(views[2], G, (gw * 16, gh * 16)).convert("L"), np.float32)
    L, Ht = CAN["r"] * (CAN["phi1"] - CAN["phi0"]), CAN["z_top"] - CAN["z_bot"]
    ref = _label_image("ORIGINAL RECIPE", int(L), int(Ht)).convert("L").resize((gw * 16, gh * 16))
    ref = np.asarray(ref, np.float32)
    vis = int(gw * 16 * 0.88)                                      # s < ~0.9 is visible in raw 2
    corr = np.corrcoef(crop[:, :vis].ravel(), ref[:, :vis].ravel())[0, 1]
    assert corr > 0.75, corr
    assert crop[:, -8:].mean() > 250                               # past the limb -> white


# ────────────────────────────────────────────────────────────────────────────
# Selection
# ────────────────────────────────────────────────────────────────────────────

def _pairs(cands):
    return [(c.item_id, c.raw_view) for c in cands]


def test_select_filters_and_flags(scene):
    _, text = scene
    got = set(_pairs(gl.select_instances(text, _cfg())))
    assert got == {(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (4, 3)}
    got = set(_pairs(gl.select_instances(text, _cfg(use_generated=True))))
    assert (6, 2) in got
    got = set(_pairs(gl.select_instances(text, _cfg(min_coverage=0.7))))
    assert (4, 3) not in got and (4, 2) in got                      # 10/16 bins in raw 3
    got = set(_pairs(gl.select_instances(text, _cfg(drop_flags=()))))
    assert (5, 4) in got
    items = json.loads(json.dumps(text["items"]))
    items[0]["flags"] = ["low_conf"]
    assert (0, 0) not in set(_pairs(gl.select_instances(items, _cfg())))


def test_rank_front_first_then_height(scene):
    _, text = scene
    order = _pairs(gl.select_instances(text, _cfg()))
    assert [p for p in order[:3]] == [(0, 0), (1, 0), (2, 0)]       # front, 48 > 28 > 12 px
    assert order[3:] == [(3, 1), (4, 2), (4, 3)]                    # 30 px, then 27 px (cos .75 > .45)


def test_budget_overflow_and_greedy_skip(scene):
    _, text = scene
    cfg = _cfg(anchor_mode="center", infer_kind="box")
    full = gl.select_instances(text, cfg)
    sizes = [c.n_tokens for c in full]
    budget = sizes[0] + sizes[2] + 1                                 # item 1 cannot fit after item 0
    assert sizes[0] + sizes[1] > budget
    cfg.token_budget = budget
    kept = gl.select_instances(text, cfg)
    assert sum(c.n_tokens for c in kept) <= budget
    assert _pairs(kept)[:2] == [(0, 0), (2, 0)]                      # skipped 1, kept smaller 2
    # many items: training builds always respect the budget
    many = {"items": []}
    for k in range(40):
        y = 20 + 12 * k
        many["items"].append({"id": k, "text": f"LINE {k} OF SMALL PRINT", "provenance": "front",
                              "flags": [], "conf": 0.9,
                              "views": {"0": _flat_view_entry([60, y, 450, y + 10],
                                                              [[60, y], [450, y], [450, y + 10], [60, y + 10]], 10)}})
    for mode in gid.ANCHOR_MODES:
        cfg = _cfg(anchor_mode=mode, token_budget=300, p_all_drop=0.0)
        for seed in range(5):
            insts = gl.build_glyphs(many, None, 8000, seed, cfg)
            assert 0 < gl.total_tokens(insts) <= 300
            assert len(insts) < 40


def test_max_views_per_item():
    q = [[100, 100], [300, 100], [300, 130], [100, 130]]
    views = {str(k): _flat_view_entry([100, 100, 300, 130], q, 30 - k) for k in range(6)}
    items = [{"id": 0, "text": "EVERYWHERE", "provenance": "front", "flags": [], "views": views}]
    kept = gl.select_instances(items, _cfg())
    assert [c.raw_view for c in kept] == [0, 1, 2]


def test_estimates_match_built_tokens_at_inference(scene):
    views, text = scene
    for mode in gid.ANCHOR_MODES:
        for kind in gl.KINDS:
            cfg = _cfg(anchor_mode=mode, infer_kind=kind, nudge_collisions=False)
            est = {(c.item_id, c.raw_view): c.n_tokens for c in gl.select_instances(text, cfg)}
            got = {(i.item_id, i.raw_view): i.n_keep for i in gl.build_glyphs_infer(text, cfg, views)}
            assert est == got, (mode, kind)


# ────────────────────────────────────────────────────────────────────────────
# Position ids
# ────────────────────────────────────────────────────────────────────────────

def test_center_ids_centred_and_integer():
    ids = gid.center_ids([96, 60, 416, 108], 0, (3, 20))
    assert ids.dtype == np.float32
    assert sorted(set(ids[:, 1])) == [4.0, 5.0, 6.0]               # centre row 4.75 - 1 -> 4
    assert ids[:, 2].min() == 6.0 and ids[:, 2].max() == 25.0      # centre col 15.5 - 9.5
    ids = gid.center_ids([96, 60, 416, 108], 0, (3, 20), quantize=False)
    assert np.isclose(ids[:, 1].mean(), 84 / 16 - 0.5)
    # raw 2 (back) lives in strip slot 3
    ids = gid.center_ids([96, 60, 416, 108], 2, (3, 20))
    assert ids[:, 2].min() == 3 * 32 + 6.0


def test_center_ids_clamped_into_slot():
    # near the right edge of raw 3 (slot 1) and the top of the view
    ids = gid.center_ids([480, 0, 512, 10], 3, (3, 10))
    assert ids[:, 2].max() == 63 and ids[:, 2].min() == 54
    assert ids[:, 1].min() == 0
    ids = gid.center_ids([0, 500, 20, 512], 0, (4, 4))
    assert ids[:, 2].min() == 0 and ids[:, 1].max() == 31


def test_center_patch_too_large_is_downscaled():
    with pytest.raises(gid.PatchTooLarge):
        gid.center_ids([0, 0, 512, 20], 0, (3, 40))
    assert gid.downscale_factor((3, 40)) == pytest.approx(32 / 40)
    long = "SUPERCALIFRAGILISTICEXPIALIDOCIOUS" * 2
    q = [[10, 200], [500, 200], [500, 220], [10, 220]]
    items = [{"id": 0, "text": long, "provenance": "front", "flags": [],
              "views": {"0": _flat_view_entry([10, 200, 500, 220], q, 20)}}]
    cfg = _cfg(anchor_mode="center", infer_kind="fixed")
    rows, (W, H) = gl.fixed_layout(long, (490, 20))
    assert W > 512                                                   # one unbreakable word
    inst = gl.build_glyphs_infer(items, cfg)[0]
    assert max(inst.token_hw) <= 32
    _check_instance(inst, cfg)


def test_stretch_ids_formula():
    box = [100.0, 40.0, 260.0, 72.0]
    ids = gid.stretch_ids(box, 1, (2, 5))
    off = raw_to_slot(1) * RES
    rows = (40 + (np.arange(2) + 0.5) * 32 / 2) / 16 - 0.5
    cols = (100 + off + (np.arange(5) + 0.5) * 160 / 5) / 16 - 0.5
    assert np.allclose(ids[:, 1].reshape(2, 5), rows[:, None])
    assert np.allclose(ids[:, 2].reshape(2, 5), cols[None, :])
    assert ids.dtype == np.float32


def test_warp_equals_stretch_on_flat_label():
    box = [96.0, 60.0, 416.0, 108.0]
    quad = [[96, 60], [416, 60], [416, 108], [96, 108]]
    for hw in [(1, 1), (3, 20), (2, 7)]:
        w_ids, keep = gid.warp_ids(gid.grid_from_quad(quad), 0, hw)
        assert keep.all()
        assert np.allclose(w_ids, gid.stretch_ids(box, 0, hw), atol=1e-4)


def test_warp_curved_can_bends_and_masks(scene):
    _, text = scene
    v = text["items"][4]["views"]["2"]
    cfg = _cfg(anchor_mode="warp")
    G, gh, gw = gl.warp_geometry(v, cfg)
    # visible arc: x 109 -> 436 plus a 38 px bend over 14 of 16 cells, extrapolated to 16 cells
    length, _ = gid.grid_extent(G)
    assert 380 < length < 440
    assert gh == 2 and gw == int(length / 16 + 0.5)
    ids, keep = gid.warp_ids(v["grid"], 2, (gh, gw))
    assert 0 < keep.sum() < keep.size                              # tail past the limb is dropped
    assert not keep.reshape(gh, gw)[:, -1].any() and keep.reshape(gh, gw)[:, 0].all()
    assert np.isnan(ids[~keep][:, 1:]).all()
    k = ids.reshape(gh, gw, 3)
    top = k[0, keep.reshape(gh, gw)[0]]
    assert top[:, 2].min() >= 3 * 32 and top[:, 2].max() <= 3 * 32 + 31   # raw 2 -> slot 3
    bend = top[:, 1].max() - top[:, 1].min()
    assert bend > 1.5, bend                                         # baseline is a smile
    j_low = int(np.argmax(top[:, 1]))
    assert 0 < j_low < len(top) - 1                                 # lowest point is interior
    dc = np.diff(top[:, 2])
    assert dc[0] > 1.5 * dc[-1]                                     # foreshortened towards the limb
    # the side view keeps only the part past phi = 0
    v3 = text["items"][4]["views"]["3"]
    G3, gh3, gw3 = gl.warp_geometry(v3, cfg)
    ids3, keep3 = gid.warp_ids(v3["grid"], 3, (gh3, gw3))
    kk = keep3.reshape(gh3, gw3)
    assert not kk[:, 0].any() and kk[:, -1].all()
    assert np.nanmin(ids3[keep3][:, 2]) >= 32 and np.nanmax(ids3[keep3][:, 2]) <= 63


def test_vertical_label_orientation(scene):
    views, text = scene
    v = text["items"][3]["views"]["1"]
    assert gl._view_orientation(v) == 90
    for kind in gl.KINDS:
        cfg = _cfg(anchor_mode="center", infer_kind=kind)
        inst = [i for i in gl.build_glyphs_infer(text, cfg, views) if i.item_id == 3][0]
        gh, gw = inst.token_hw
        assert gh > gw, (kind, inst.token_hw)                        # view-oriented, tall
        _check_instance(inst, cfg)
    cfg = _cfg(anchor_mode="warp", infer_kind="box")
    inst = [i for i in gl.build_glyphs_infer(text, cfg, views) if i.item_id == 3][0]
    gh, gw = inst.token_hw
    assert gw > gh                                                   # reading frame, wide
    k = inst.ids.reshape(gh, gw, 3)
    assert np.all(np.diff(k[0, :, 1]) > 0)                           # reading goes down the view
    assert np.ptp(k[0, :, 2]) < 0.5                                  # at constant column
    _check_instance(inst, cfg)


def test_fixed_wrap_uses_reading_quad_not_bbox(scene):
    # the bent can line has a 327 x 90 bbox but a 27 px tall reading quad: stays one line
    views, text = scene
    for mode in ("center", "stretch"):
        cfg = _cfg(anchor_mode=mode, infer_kind="fixed")
        inst = [i for i in gl.build_glyphs_infer(text, cfg, views) if (i.item_id, i.raw_view) == (4, 2)][0]
        assert inst.token_hw[0] == (32 + 16) // 16, inst.token_hw


def test_grid_parse_layouts_and_nulls():
    G = np.arange(4 * 16 * 2, dtype=float).reshape(4, 16, 2)
    G[1, 3] = np.nan
    js = _grid_json(G)
    assert np.allclose(gid.parse_grid(js), G, equal_nan=True)
    tr = {"ns": 16, "nt": 4, "xy": [[js["xy"][i][j] for i in range(4)] for j in range(16)]}
    assert np.allclose(gid.parse_grid(tr), G, equal_nan=True)
    with pytest.raises(ValueError):
        gid.parse_grid({"ns": 5, "nt": 4, "xy": js["xy"]})
    # null cell -> masked; mask_null=False keeps it at an interpolated / nearest position
    s = np.array([(3 + 0.5) / 16])
    t = np.array([(1 + 0.5) / 4])
    xy, keep = gid.sample_grid(G, s, t)
    assert not keep[0] and np.isfinite(xy).all()
    P = gid.grid_from_quad([[0, 0], [160, 0], [160, 40], [0, 40]])
    P[1, 3] = np.nan
    ids, keep = gid.warp_ids(P, 0, (4, 16))
    assert keep.sum() == 63 and not keep.reshape(4, 16)[1, 3]
    ids2, keep2 = gid.warp_ids(P, 0, (4, 16), mask_null=False)
    assert keep2.all() and np.isfinite(ids2).all()
    assert np.allclose(ids2[1 * 16 + 3], gid.warp_ids(gid.grid_from_quad(
        [[0, 0], [160, 0], [160, 40], [0, 40]]), 0, (4, 16))[0][1 * 16 + 3], atol=1e-4)


def test_jitter_integer_and_inside_slot():
    rng = np.random.default_rng(1)
    ids = gid.center_ids([0, 0, 48, 16], 0, (1, 3))                 # touches top-left corner
    seen = set()
    for _ in range(100):
        j, (dr, dc) = gid.jitter_ids(ids, np.ones(3, bool), 0, rng, 1)
        assert dr in (0, 1) and dc in (0, 1)
        assert np.array_equal(j[:, 1:] - ids[:, 1:], np.tile([dr, dc], (3, 1)).astype(np.float32))
        seen.add((dr, dc))
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}
    ids = gid.center_ids([200, 200, 248, 216], 4, (1, 3))
    shifts = {gid.jitter_ids(ids, None, 4, rng, 1)[1] for _ in range(200)}
    assert len(shifts) == 9


def test_collisions_detected_and_nudged():
    class I:
        def __init__(self, ids, raw):
            self.ids, self.keep, self.raw_view = ids, np.ones(len(ids), bool), raw

    a = I(gid.center_ids([200, 200, 232, 216], 0, (1, 2)), 0)
    b = I(gid.center_ids([200, 200, 232, 216], 0, (1, 2)), 0)
    c = I(gid.center_ids([200, 200, 232, 216], 2, (1, 2)), 2)       # other slot, no clash
    ids, slices = gid.concat_ids([a, b, c])
    assert ids.dtype == np.float32 and [s.stop - s.start for s in slices] == [2, 2, 2]
    assert gid.count_collisions(ids, slices) == 2
    with pytest.warns(UserWarning):
        assert gid.verify_unique([a, b, c], nudge=False) == 2
    assert gid.verify_unique([a, b, c], nudge=True, warn=False) == 0
    assert np.array_equal(a.ids, gid.center_ids([200, 200, 232, 216], 0, (1, 2)))   # first stays
    # duplicates inside one instance are fine
    d = I(np.array([[1, 3, 4], [1, 3, 4]], np.float32), 0)
    ids, slices = gid.concat_ids([d])
    assert gid.count_collisions(ids, slices) == 0


# ────────────────────────────────────────────────────────────────────────────
# build_glyphs
# ────────────────────────────────────────────────────────────────────────────

def test_all_modes_fp32_slots_budget(scene):
    views, text = scene
    for mode in gid.ANCHOR_MODES:
        cfg = _cfg(anchor_mode=mode, p_all_drop=0.0)
        n_inst = 0
        for step in (0, 3000, 9000):
            for seed in range(8):
                insts = gl.build_glyphs(text, views, step, seed, cfg)
                assert gl.total_tokens(insts) <= cfg.token_budget
                for inst in insts:
                    _check_instance(inst, cfg)
                    assert inst.kind in gl.KINDS
                n_inst += len(insts)
                ids, slices = gid.concat_ids(insts)
                assert ids.dtype == np.float32 and len(ids) == gl.total_tokens(insts)
        assert n_inst > 50
    for mode in gid.ANCHOR_MODES:
        for kind in gl.KINDS:
            cfg = _cfg(anchor_mode=mode, infer_kind=kind)
            insts = gl.build_glyphs_infer(text, cfg, views)
            assert {i.kind for i in insts} == {kind}
            for inst in insts:
                _check_instance(inst, cfg)


def _signature(insts):
    return [(i.item_id, i.raw_view, i.kind, i.token_hw, i.ids.tobytes(), i.patch.tobytes()) for i in insts]


def test_determinism_with_seeded_rng(scene):
    views, text = scene
    for mode in gid.ANCHOR_MODES:
        cfg = _cfg(anchor_mode=mode)
        for seed in (0, 7):
            a = gl.build_glyphs(text, views, 1500, np.random.default_rng(seed), cfg)
            b = gl.build_glyphs(text, views, 1500, np.random.default_rng(seed), cfg)
            assert _signature(a) == _signature(b)
        sigs = {tuple(_signature(gl.build_glyphs(text, views, 1500, s, cfg))) for s in range(6)}
        assert len(sigs) > 1
        assert _signature(gl.build_glyphs_infer(text, cfg, views)) == \
            _signature(gl.build_glyphs_infer(text, cfg, views))


def test_stage_schedule_and_kind_frequencies(scene):
    views, text = scene
    cfg = _cfg()
    assert gl.stage_probs(cfg, 0) == pytest.approx((0.6, 0.2, 0.2))
    assert gl.stage_probs(cfg, 1999) == pytest.approx((0.6, 0.2, 0.2))
    assert gl.stage_probs(cfg, 2000) == pytest.approx((0.3, 0.3, 0.4))
    assert gl.stage_probs(cfg, 6000) == pytest.approx((0.05, 0.25, 0.7))
    assert gl.stage_probs(cfg, 10 ** 9) == pytest.approx((0.05, 0.25, 0.7))
    one = {"items": [text["items"][0]]}
    cfg = _cfg(p_all_drop=0.0, p_item_drop=0.0)
    for step, expect in ((0, (0.6, 0.2, 0.2)), (9000, (0.05, 0.25, 0.7))):
        kinds = [gl.build_glyphs(one, views, step, s, cfg)[0].kind for s in range(400)]
        freq = [kinds.count(k) / len(kinds) for k in gl.KINDS]
        assert np.allclose(freq, expect, atol=0.07), (step, freq)
    # without target views gt falls back to box
    kinds = {gl.build_glyphs(one, None, 0, s, cfg)[0].kind for s in range(40)}
    assert "gt" not in kinds


def test_dropout_rates(scene):
    views, text = scene
    one = {"items": [text["items"][2]]}
    cfg = _cfg(p_all_drop=0.1, p_item_drop=0.15)
    n = 600
    empty = sum(len(gl.build_glyphs(one, views, 9000, s, cfg)) == 0 for s in range(n)) / n
    assert abs(empty - (0.1 + 0.9 * 0.15)) < 0.05, empty
    cfg = _cfg(p_all_drop=1.0)
    assert gl.build_glyphs(text, views, 0, 0, cfg) == []


def test_debug_png(scene, outdir):
    from unitex import glyph_debug
    views, text = scene
    strip_path = os.path.join(outdir, "scene_strip.png")
    json_path = os.path.join(outdir, "scene_text.json")
    Image.fromarray(stack_strip(views)).save(strip_path)
    with open(json_path, "w") as f:
        json.dump(text, f)
    outs = []
    for mode in gid.ANCHOR_MODES:
        for extra, tag in (([], "infer_fixed"), (["--kind", "box"], "infer_box"),
                           (["--train", "--step", "0", "--seed", "3"], "train_s0")):
            out = os.path.join(outdir, f"debug_{mode}_{tag}.png")
            glyph_debug.main(["--text-json", json_path, "--strip", strip_path, "--out", out,
                              "--mode", mode] + extra)
            img = Image.open(out)
            assert img.width == 6 * RES and img.height > RES
            outs.append(out)
    out = os.path.join(outdir, "debug_warp_slots.png")
    glyph_debug.main(["--text-json", json_path, "--strip", strip_path, "--out", out,
                      "--mode", "warp", "--kind", "box", "--slots", "0,2,3"])
    assert Image.open(out).width == 3 * RES


# ────────────────────────────────────────────────────────────────────────────
# Review: adversarial checks against UniTEX-FLUX packing, orientation, clamping, inputs
# ────────────────────────────────────────────────────────────────────────────

UNITEX_FLUX_ROOT = os.environ.get("UNITEX_FLUX_ROOT", "/Users/test/.claude/jobs/5ee54a5d/tmp/UniTEX-FLUX")


def _np_pack_latents(lat):
    """Numpy copy of UniTEX-FLUX tasks/texturing/pipeline.py:241-248 (pixel_shuffle=True)."""
    b, c, h, w = lat.shape
    lat = lat.reshape(b, c, h // 2, 2, w // 2, 2).transpose(0, 2, 4, 1, 3, 5)
    return lat.reshape(b, (h // 2) * (w // 2), c * 4)


def _np_latent_image_ids(h, w):
    """Numpy copy of pipeline.py:268-275 with zero offsets: token n = (0, n // w, n % w)."""
    ids = np.zeros((h, w, 3))
    ids[..., 1] += np.arange(h)[:, None]
    ids[..., 2] += np.arange(w)[None, :]
    return ids.reshape(h * w, 3)


def _fake_vae_tokens(img):
    """8x8 average pool (the VAE's spatial footprint), 16 channels, then UniTEX packing -> [N, 64]."""
    a = np.asarray(img, np.float64)
    h, w = a.shape[0] // 8, a.shape[1] // 8
    lat = a.reshape(h, 8, w, 8, 3).mean(axis=(1, 3)).transpose(2, 0, 1)[None]
    lat = np.concatenate([lat] * 5 + [lat[:, :1]], axis=1)
    return _np_pack_latents(lat)[0]


def test_review_numpy_pack_matches_unitex_flux():
    """The numpy replicas used below equal the real UniTEX-FLUX static methods."""
    torch = pytest.importorskip("torch")
    path = os.path.join(UNITEX_FLUX_ROOT, "tasks", "texturing", "pipeline.py")
    if not os.path.exists(path):
        pytest.skip("UniTEX-FLUX checkout not found (set UNITEX_FLUX_ROOT)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_uf_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:     # diffusers / transformers missing or incompatible
        pytest.skip(f"cannot import UniTEX-FLUX pipeline: {e}")
    P = mod.PBRFluxPipeline
    lat = np.random.default_rng(0).standard_normal((1, 16, 8, 12)).astype(np.float32)
    ref = P._pack_latents(torch.from_numpy(lat), 1, 16, 8, 12).numpy()
    assert np.array_equal(_np_pack_latents(lat), ref)
    ids = P._prepare_latent_image_ids(1, 32, 192, "cpu", torch.float32).numpy()
    assert np.array_equal(_np_latent_image_ids(32, 192), ids)


def test_review_glyph_token_k_sits_on_matching_target_token():
    """End to end id math: for a 16-aligned gt crop, glyph token k holds the same pixels as the
    target strip token whose (row, col) equals ids[k], in every mode and every raw view."""
    rng = np.random.default_rng(0)
    views = [rng.integers(0, 256, (RES, RES, 3), dtype=np.uint8) for _ in range(6)]
    tgt = _fake_vae_tokens(stack_strip(views))
    lut = {(int(r), int(c)): n for n, (_, r, c) in enumerate(_np_latent_image_ids(32, 192))}
    for raw in range(6):
        for box in ([96, 64, 416, 112], [0, 0, 32, 512], [480, 496, 512, 512]):
            q = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
            items = [{"id": 0, "text": "X", "provenance": "front", "flags": [],
                      "views": {str(raw): _flat_view_entry(box, q, box[3] - box[1])}}]
            for mode in gid.ANCHOR_MODES:
                inst = gl.build_glyphs_infer(items, _cfg(anchor_mode=mode, infer_kind="gt"), views)[0]
                tok = _fake_vae_tokens(inst.patch)
                assert len(tok) == len(inst.ids) == inst.keep.sum()
                rc = inst.ids[:, 1:]
                assert np.allclose(rc, np.round(rc), atol=1e-4), (mode, raw, box)
                for n, (r, c) in enumerate(np.round(rc).astype(int)):
                    assert raw_to_slot(raw) * 32 <= c < raw_to_slot(raw) * 32 + 32
                    assert np.abs(tok[n] - tgt[lut[(r, c)]]).max() < 1e-6, (mode, raw, box, n)


def _paint_oriented(views, raw, text, bbox, orient):
    """Paint render_box(text) rotated into view orientation; the view entry has the reading quad."""
    x0, y0, x1, y1 = bbox
    rw, rh = (x1 - x0, y1 - y0) if orient in (0, 180) else (y1 - y0, x1 - x0)
    lab = gl._orient(gl.render_box(text, (rw, rh)), orient)
    assert lab.size == (x1 - x0, y1 - y0)
    im = Image.fromarray(views[raw])
    im.paste(lab, (x0, y0))
    views[raw][:] = np.asarray(im)
    c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    s = {0: 0, 90: 1, 180: 2, 270: 3}[orient]
    return _flat_view_entry(bbox, c[s:] + c[:s], rh), gl.render_box(text, (rw, rh))


def test_review_orientation_all_four_quadrants():
    """0 / 90 / 180 / 270 degree labels (the implementer tested 90 only): the view-oriented box
    render must equal the gt crop, and warp gt crops must read upright."""
    text = "BEST BEFORE 12"
    views = [np.full((RES, RES, 3), 96, np.uint8) for _ in range(6)]
    for orient, bbox in ((0, [96, 96, 400, 144]), (90, [448, 48, 496, 464]),
                         (180, [96, 304, 400, 352]), (270, [16, 48, 64, 464])):
        v, upright = _paint_oriented(views, 0, text, bbox, orient)
        assert gl._view_orientation(v) == orient
        items = [{"id": 0, "text": text, "provenance": "front", "flags": [], "views": {"0": v}}]
        for mode in ("center", "stretch"):
            gt = gl.build_glyphs_infer(items, _cfg(anchor_mode=mode, infer_kind="gt"), views)[0].patch
            bx = gl.build_glyphs_infer(items, _cfg(anchor_mode=mode, infer_kind="box"), views)[0].patch
            a = np.asarray(gt.convert("L"), np.float32).ravel()
            ok = np.corrcoef(a, np.asarray(bx.convert("L"), np.float32).ravel())[0, 1]
            flip = np.corrcoef(a, np.asarray(bx.rotate(180).convert("L"), np.float32).ravel())[0, 1]
            assert ok > 0.99 and flip < 0.6, (mode, orient, ok, flip)
        w = gl.build_glyphs_infer(items, _cfg(anchor_mode="warp", infer_kind="gt"), views)[0].patch
        assert w.width > w.height                                    # reading frame
        ref = np.asarray(upright.resize(w.size).convert("L"), np.float32).ravel()
        corr = np.corrcoef(np.asarray(w.convert("L"), np.float32).ravel(), ref)[0, 1]
        assert corr > 0.99, (orient, corr)


def test_review_offview_boxes_clamped_in_every_slot():
    """Boxes hanging off the view (projected text past the border) stay inside their own slot,
    for all six raw views, all modes, inference and jittered / nudged training draws."""
    for raw in range(6):
        clo = raw_to_slot(raw) * 32
        for box in ([-40, -10, 60, 30], [470, 490, 560, 530], [-5, 200, 520, 230]):
            q = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
            items = [{"id": 0, "text": "EDGE CASE TEXT", "provenance": "front", "flags": [],
                      "views": {str(raw): _flat_view_entry(box, q, box[3] - box[1])}}]
            for mode in gid.ANCHOR_MODES:
                insts = []
                for kind in ("fixed", "box"):
                    insts += gl.build_glyphs_infer(items, _cfg(anchor_mode=mode, infer_kind=kind))
                for seed in range(4):
                    insts += gl.build_glyphs(items, None, 0, seed,
                                             _cfg(anchor_mode=mode, p_all_drop=0.0, p_item_drop=0.0))
                assert insts
                for inst in insts:
                    k = inst.ids[inst.keep]
                    assert np.isfinite(k).all()
                    assert clo <= k[:, 2].min() and k[:, 2].max() <= clo + 31, (raw, box, mode)
                    assert 0 <= k[:, 1].min() and k[:, 1].max() <= 31, (raw, box, mode)


def test_review_warp_mask_matches_white_gt_pixels(scene):
    """Masked warp tokens (keep False) are the tokens whose centre falls on a null cell, which
    crop_gt_grid paints white; kept tokens on the visible can carry label pixels."""
    views, text = scene
    for raw in ("2", "3"):
        v = text["items"][4]["views"][raw]
        G, gh, gw = gl.warp_geometry(v, _cfg(anchor_mode="warp"))
        patch = np.asarray(gl.crop_gt_grid(views[int(raw)], G, (gw * 16, gh * 16)).convert("L"))
        _, keep = gid.warp_ids(v["grid"], int(raw), (gh, gw))
        keep = keep.reshape(gh, gw)
        centre = patch[8::16, 8::16]                                 # pixel at each token centre
        assert centre.shape == keep.shape
        assert (centre[~keep] == 255).all(), raw
        assert (centre[keep] < 250).mean() > 0.9, raw               # label background is cream
        # latent drop + concat stay aligned with the ids
        inst = [i for i in gl.build_glyphs_infer(text, _cfg(anchor_mode="warp", infer_kind="gt"), views)
                if (i.item_id, i.raw_view) == (4, int(raw))][0]
        tok = _fake_vae_tokens(inst.patch)[inst.keep]
        ids, sl = gid.concat_ids([inst])
        assert len(tok) == len(ids) == sl[0].stop and np.isfinite(ids).all()


def test_review_float_views_missing_views_and_nulls(scene):
    """Trainer tensors are float [0, 1]; the inference photo exists for the front only; JSON nulls."""
    views, text = scene
    cfg = _cfg(anchor_mode="center", infer_kind="gt")
    ref = [np.asarray(i.patch, np.int16) for i in gl.build_glyphs_infer(text, cfg, views)]
    f3 = [v.astype(np.float32) / 255.0 for v in views]
    f4 = [np.concatenate([v, np.ones(v.shape[:2] + (1,), np.float32)], -1) for v in f3]
    for fv in (f3, f4):
        got = [np.asarray(i.patch, np.int16) for i in gl.build_glyphs_infer(text, cfg, fv)]
        assert len(got) == len(ref) and all(np.abs(a - b).max() <= 1 for a, b in zip(ref, got))
    front_only = [views[0]] + [None] * 5
    for mode in gid.ANCHOR_MODES:
        insts = gl.build_glyphs_infer(text, _cfg(anchor_mode=mode, infer_kind="gt"), front_only)
        assert {i.kind for i in insts if i.raw_view == 0} == {"gt"}
        assert {i.kind for i in insts if i.raw_view != 0} == {"box"}
        tr = gl.build_glyphs(text, front_only, 0, 1, _cfg(anchor_mode=mode, p_all_drop=0.0))
        assert all(i.kind != "gt" for i in tr if i.raw_view != 0)
    t = json.loads(json.dumps(text))
    for it in t["items"]:
        it["conf"] = None
        for v in it["views"].values():
            v["coverage"] = v["cos"] = None
    got = set(_pairs(gl.select_instances(t, _cfg())))
    assert {(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (4, 3)} <= got     # nulls pass the filters
