"""
Tests for unitex/glyph_layout.py: lines drawn at their exact place and size, filters and training
draws, edges, the VAE encoding (text-free tokens are zero) and the zero-initialised x_embedder wrapper.
CPU only, a tiny random AutoencoderKL.

Run: python -m pytest unitex/tests/test_glyph_layout.py -v
"""

import json
import os
import sys
import types

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from unitex import glyph_layout as gla
from unitex import glyph_tokens as gt
from unitex.common import FULL_INDEX, TOKEN_PX

EVAL28 = os.environ.get("CFI_EVAL28", "/Users/test/CyLab/unitex_eval_28")
TINY_VAE = dict(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 8, 8),
                layers_per_block=1, latent_channels=16, norm_num_groups=4, scaling_factor=0.3611,
                shift_factor=0.1159, use_quant_conv=False, use_post_quant_conv=False,
                mid_block_add_attention=False)


def _quad(x0, y0, x1, y1, orient=0):
    """Reading-order quad of an axis-aligned box: orient 0 reads right, 90 reads down."""
    if orient == 90:
        return [[x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _text(items, res=512):
    return {"version": 1, "res": res, "items": items}


def _item(i, text, views, **kw):
    return {"id": i, "text": text, "conf": 0.99, "provenance": "front", "flags": [], "views": views, **kw}


def _view(quad, **kw):
    q = np.asarray(quad, float)
    return {"quad": quad, "bbox": [q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()],
            "coverage": 1.0, "cos": 1.0, **kw}


def _ink(strip, thresh=128):
    return np.asarray(strip)[..., 0] < thresh


def _slot_box(raw_view, R, box, margin=2):
    off = FULL_INDEX.index(raw_view) * R
    x0, y0, x1, y1 = box
    return slice(max(int(y0) - margin, 0), int(np.ceil(y1)) + margin), \
        slice(off + max(int(x0) - margin, 0), off + int(np.ceil(x1)) + margin)


# ────────────────────────────────────────────────────────────────────────────
# Drawing
# ────────────────────────────────────────────────────────────────────────────

def test_font_ink_stays_inside_its_quad_and_view_slot():
    R = 512
    a, b = (100, 200, 300, 232), (40, 50, 72, 250)
    text = _text([_item(0, "GLUTEN FREE", {"0": _view(_quad(*a))}),
                  _item(1, "SIDE PANEL", {"3": _view(_quad(*b, orient=90))})])
    strip, info = gla.layout_strip(text, R)
    assert strip.shape == (R, 6 * R, 3) and strip.dtype == np.uint8
    ink = _ink(strip)
    inside = np.zeros_like(ink)
    for rv, box in ((0, a), (3, b)):
        ys, xs = _slot_box(rv, R, box)
        assert ink[ys, xs].sum() > 200, f"no ink in view {rv}"
        inside[ys, xs] = True
    assert not (ink & ~inside).any(), "ink outside the line quads"
    assert info["n_lines"] == 2 and info["per_view"] == {"0": 1, "3": 1} and info["kinds"] == {"font": 2}


def test_rotated_line_reads_along_its_quad():
    R = 512
    box = (40, 50, 72, 450)
    strip, _ = gla.layout_strip(_text([_item(0, "VERTICAL TEXT", {"3": _view(_quad(*box, orient=90))})]), R)
    ys, xs = np.nonzero(_ink(strip))
    assert ys.max() - ys.min() > 4 * (xs.max() - xs.min())


@pytest.mark.parametrize("R,h", [(512, 8), (1024, 14), (1024, 40)])
def test_line_fills_its_box(R, h):
    """Ink spans the box in both directions, so the letters sit where the line's letters are."""
    x0, y0, L = 300.0, 400.0, 10.0 * h
    strip, _ = gla.layout_strip(_text([_item(0, "EST 1869 HEINZ", {"0": _view(_quad(x0, y0, x0 + L, y0 + h))})], R), R)
    ys, xs = np.nonzero(_ink(strip, 200))
    assert xs.min() >= x0 - 1 and xs.max() <= x0 + L + 1 and ys.min() >= y0 - 1 and ys.max() <= y0 + h + 1
    assert xs.max() - xs.min() >= 0.8 * L and ys.max() - ys.min() >= 0.6 * h


def test_text_json_at_another_resolution_is_rescaled():
    a = (100, 200, 300, 232)
    t512 = _text([_item(0, "ABC 123", {"0": _view(_quad(*a))})], res=512)
    s1024, _ = gla.layout_strip(t512, 1024)
    ys, xs = np.nonzero(_ink(s1024))
    assert xs.min() >= 2 * a[0] - 2 and xs.max() <= 2 * a[2] + 2 and ys.min() >= 2 * a[1] - 2 and ys.max() <= 2 * a[3] + 2


def test_empty_text_is_white():
    for text in (None, "", _text([]), _text([_item(0, "  ", {"0": _view(_quad(10, 10, 80, 30))})])):
        strip, info = gla.layout_strip(text, 512)
        assert (strip == 255).all() and info["n_lines"] == 0


# ────────────────────────────────────────────────────────────────────────────
# Filters and training draws
# ────────────────────────────────────────────────────────────────────────────

def _many(n=12):
    return _text([_item(i, f"LINE {i}", {"0": _view(_quad(20, 20 + 30 * i, 220, 44 + 30 * i))}) for i in range(n)])


def test_filters():
    q = _quad(20, 20, 220, 44)
    items = [_item(0, "KEEP", {"0": _view(q)}),
             _item(1, "GEN", {"0": _view(q)}, provenance="generated"),
             _item(2, "LEAK", {"0": _view(q)}, flags=["template_leak"]),
             _item(3, "TINY", {"0": _view(_quad(20, 20, 60, 24))}),
             _item(4, "BADOCR", {"0": _view(q, view_ocr_ned=0.2)}),
             _item(5, "SIDEWAYS", {"0": _view(q, cos=0.1)})]
    ids = lambda cfg: sorted(ln["item_id"] for ln in gla.layout_lines(_text(items), 512, cfg))
    assert ids(gla.LayoutConfig()) == [0, 1]
    assert ids(gla.LayoutConfig(use_generated=False)) == [0]
    assert ids(gla.LayoutConfig(min_height_px=2, drop_flags=())) == [0, 1, 2, 3]


def test_training_draws_and_determinism():
    text = _many()
    cfg = gla.LayoutConfig(p_item_drop=0.0, p_all_drop=0.0, p_edges=0.0)
    assert len(gla.layout_lines(text, 512, cfg, np.random.default_rng(0))) == 12
    assert gla.layout_lines(text, 512, gla.LayoutConfig(p_all_drop=1.0), np.random.default_rng(0)) == []
    assert gla.layout_lines(text, 512, gla.LayoutConfig(p_item_drop=1.0, p_all_drop=0.0), np.random.default_rng(0)) == []
    edges = gla.layout_lines(text, 512, gla.LayoutConfig(p_edges=1.0, p_item_drop=0, p_all_drop=0), np.random.default_rng(0))
    assert {ln["kind"] for ln in edges} == {"edges"}
    cfg = gla.LayoutConfig(p_item_drop=0.5, p_all_drop=0.0, p_edges=0.5)
    a = gla.layout_lines(text, 512, cfg, np.random.default_rng(7))
    b = gla.layout_lines(text, 512, cfg, np.random.default_rng(7))
    assert a == b and 0 < len(a) < 12
    # no rng: every line, no drops, the inference kind
    full = gla.layout_lines(text, 512, gla.LayoutConfig(p_item_drop=1.0, p_all_drop=1.0, infer_kind="edges"))
    assert len(full) == 12 and {ln["kind"] for ln in full} == {"edges"}


def test_config_parsing():
    cfg = gla.layout_config('{"use_generated": false}', ["p_edges=0.5", "drop_flags=[\"low_conf\"]"])
    assert cfg.use_generated is False and cfg.p_edges == 0.5 and cfg.drop_flags == ("low_conf",)
    with pytest.raises(ValueError):
        gla.layout_config(None, ["nope=1"])
    with pytest.raises(ValueError):
        gla.layout_config(None, ["infer_kind=glyph"])
    with pytest.raises(ValueError):
        gla.layout_config(None, ["p_edges=2"])
    assert json.loads(json.dumps(gla.layout_config_dict(cfg)))["p_edges"] == 0.5


# ────────────────────────────────────────────────────────────────────────────
# Edges
# ────────────────────────────────────────────────────────────────────────────

def test_edges_come_from_the_view_inside_the_quad_only():
    R = 512
    view = np.full((R, R, 3), 230, np.uint8)
    view[100:140, 100:300] = 20          # a dark bar crossing the quad border
    view[300:340, 50:90] = 20            # a dark block outside every quad
    box = (150, 90, 250, 150)
    text = _text([_item(0, "BAR", {"0": _view(_quad(*box))})])
    cfg = gla.LayoutConfig(infer_kind="edges")
    strip, info = gla.layout_strip(text, R, cfg, views=[view] + [None] * 5)
    ink = _ink(strip)
    assert info["kinds"] == {"edges": 1}
    ys, xs = np.nonzero(ink)
    assert len(ys) > 100 and ys.min() >= 90 and ys.max() <= 150 and xs.min() >= 150 and xs.max() <= 250
    # the bar's top and bottom edges, not its inside
    assert ink[99:102, 160:240].any() and ink[138:142, 160:240].any() and not ink[110:130, 160:240].any()
    # an edges line on a view without an image is drawn as font
    _, info2 = gla.layout_strip(text, R, cfg, views=None)
    assert info2["kinds"] == {"font": 1}
    with pytest.raises(ValueError):
        gla.layout_strip(text, R, cfg, views=[view[:256, :256]] + [None] * 5)


def test_training_edges_are_found_on_degraded_pixels():
    R = 512
    view = np.full((R, R, 3), 230, np.uint8)
    view[100:140, 100:300] = 20
    view[210:250, 100:300] = 20
    boxes = ((150, 90, 250, 150), (150, 200, 250, 260))
    text = _text([_item(i, f"BAR {i}", {"0": _view(_quad(*b))}) for i, b in enumerate(boxes)])
    views = [view] + [None] * 5
    base = dict(p_edges=1.0, p_crop=0.0, p_item_drop=0.0, p_all_drop=0.0)
    clean, _ = gla.layout_strip(text, R, gla.LayoutConfig(infer_kind="edges"), views=views)
    # no crop_* set: training edges are the clean ones and they use no rng number
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    plain, info = gla.layout_strip(text, R, gla.LayoutConfig(**base), rng=r1, views=views)
    gla.layout_lines(text, R, gla.LayoutConfig(**base), r2)
    assert info["kinds"] == {"edges": 2} and np.array_equal(plain, clean) and r1.random() == r2.random()
    # crop_* set: the edges follow the degraded pixels, repeat with the rng and stay inside the quads
    cfg = gla.LayoutConfig(**base, crop_shift=3.0, crop_blur=1.5, crop_color=0.2)
    strips = [gla.layout_strip(text, R, cfg, rng=np.random.default_rng(s), views=views)[0] for s in range(6)]
    assert np.array_equal(strips[0], gla.layout_strip(text, R, cfg, rng=np.random.default_rng(0), views=views)[0])
    assert len({s.tobytes() for s in strips}) > 1 and any(not np.array_equal(s, clean) for s in strips)
    off = FULL_INDEX.index(0) * R
    m = gla.quad_mask(_quad(*boxes[0]), (R, R)) | gla.quad_mask(_quad(*boxes[1]), (R, R))
    for s in strips:
        ink = _ink(s)[:, off:off + R]
        assert not (ink & ~m).any()
        ys = np.nonzero(ink[90:150])[0] + 90
        # the first bar's top and bottom edges, moved by at most the shift plus a blurred pixel
        assert len(ys) > 50 and np.abs(np.where(ys < 120, ys - 100, ys - 140)).max() <= 5


# ────────────────────────────────────────────────────────────────────────────
# Crops
# ────────────────────────────────────────────────────────────────────────────

def _colour_view(R):
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (R, R, 3), dtype=np.uint8)


def test_crop_copies_the_view_inside_the_quad_in_colour():
    R = 512
    view = _colour_view(R)
    box = (150, 90, 250, 150)
    text = _text([_item(0, "BAR", {"0": _view(_quad(*box))})])
    cfg = gla.LayoutConfig(infer_kind="crop")
    strip, info = gla.layout_strip(text, R, cfg, views=[view] + [None] * 5)
    assert info["kinds"] == {"crop": 1} and strip.shape == (R, 6 * R, 3)
    off = FULL_INDEX.index(0) * R
    m = gla.quad_mask(_quad(*box), (R, R))
    assert np.array_equal(strip[:, off:off + R][m], view[m])
    assert (strip[:, off:off + R][~m] == gla.WHITE).all()
    assert (np.delete(strip, np.s_[off:off + R], axis=1) == gla.WHITE).all()
    # a crop line on a view without an image is drawn as font
    _, info2 = gla.layout_strip(text, R, cfg, views=None)
    assert info2["kinds"] == {"font": 1}


def test_crop_degradations_stay_inside_the_quad_and_repeat_with_the_rng():
    R = 512
    view = np.full((R, R, 3), 200, np.uint8)
    view[110:130, 160:240] = (20, 40, 160)
    box = (150, 90, 250, 150)
    text = _text([_item(0, "BAR", {"0": _view(_quad(*box))})])
    cfg = gla.LayoutConfig(p_crop=1.0, p_edges=0.0, p_item_drop=0.0, p_all_drop=0.0,
                           crop_shift=3.0, crop_blur=1.5, crop_color=0.2)
    a, ia = gla.layout_strip(text, R, cfg, rng=np.random.default_rng(5), views=[view] + [None] * 5)
    b, _ = gla.layout_strip(text, R, cfg, rng=np.random.default_rng(5), views=[view] + [None] * 5)
    c, _ = gla.layout_strip(text, R, cfg, rng=np.random.default_rng(6), views=[view] + [None] * 5)
    assert ia["kinds"] == {"crop": 1} and np.array_equal(a, b) and not np.array_equal(a, c)
    off = FULL_INDEX.index(0) * R
    m = gla.quad_mask(_quad(*box), (R, R))
    assert (a[:, off:off + R][~m] == gla.WHITE).all()
    clean, _ = gla.layout_strip(text, R, gla.LayoutConfig(infer_kind="crop"), views=[view] + [None] * 5)
    assert not np.array_equal(a, clean)
    # still the blue bar, near where it was
    blue = a[:, off:off + R][..., 2].astype(int) - a[:, off:off + R][..., 0].astype(int) > 60
    ys, xs = np.nonzero(blue)
    assert len(ys) > 800 and abs(ys.mean() - 119.5) < 4 and abs(xs.mean() - 199.5) < 4


def test_one_kind_draw_per_item():
    text = _many()
    base = dict(p_item_drop=0.3, p_all_drop=0.0)
    e = gla.layout_lines(text, 512, gla.LayoutConfig(p_edges=0.5, **base), np.random.default_rng(11))
    c = gla.layout_lines(text, 512, gla.LayoutConfig(p_edges=0.0, p_crop=0.5, **base), np.random.default_rng(11))
    # the same lines, the same draws: edges with p_edges 0.5 are exactly crops with p_crop 0.5
    assert [ln["item_id"] for ln in e] == [ln["item_id"] for ln in c]
    assert [ln["kind"] == "edges" for ln in e] == [ln["kind"] == "crop" for ln in c]
    assert {ln["kind"] for ln in c} == {"font", "crop"}
    mix = gla.layout_lines(text, 512, gla.LayoutConfig(p_edges=0.5, p_crop=0.5, **base), np.random.default_rng(11))
    assert [ln["kind"] for ln in mix] == ["edges" if ln["kind"] == "edges" else "crop" for ln in e]
    with pytest.raises(ValueError):
        gla.layout_config(None, ["p_edges=0.6", "p_crop=0.6"])
    with pytest.raises(ValueError):
        gla.layout_config(None, ["crop_blur=-1"])
    assert gla.needs_views(gla.LayoutConfig(p_edges=0.0, p_crop=0.2))
    assert not gla.needs_views(gla.LayoutConfig(p_edges=0.0, p_crop=0.0))


def test_drawn_text_matches_the_text_token_mask():
    R = 1024
    text = _many(6)
    _, info = gla.layout_strip(text, R)
    assert torch.equal(gt.text_token_mask(info["drawn_text"], R), gt.text_token_mask(text, R))


# ────────────────────────────────────────────────────────────────────────────
# VAE encoding
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vae():
    from diffusers import AutoencoderKL
    torch.manual_seed(0)
    return AutoencoderKL(**TINY_VAE).eval()


def test_encode_layout_zero_without_text_and_largest_on_the_text(vae):
    R = 512
    white = np.full((R, 6 * R, 3), 255, np.uint8)
    z0 = gla.encode_layout(white, vae)
    assert z0.shape == (1, (R // 16) * (6 * R // 16), 64) and not z0.any()
    text = _text([_item(0, "GLUTEN FREE SIN GLUTEN", {"0": _view(_quad(64, 192, 448, 256))})])
    strip, info = gla.layout_strip(text, R)
    z = gla.encode_layout(strip, vae)
    m = gt.text_token_mask(info["drawn_text"], R)
    mag = z[0].abs().mean(-1)
    assert mag[m].mean() > 5 * mag[~m].mean()
    # the blank latent is cached per size and the result is deterministic (posterior mode)
    assert torch.equal(z, gla.encode_layout(strip, vae))


def test_layout_tokens_full():
    lay = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    full = gla.layout_tokens_full(lay, 9)
    assert full.shape == (2, 9, 3) and torch.equal(full[:, :5], lay) and not full[:, 5:].any()
    keep = torch.tensor([1, 3])
    sel = gla.layout_tokens_full(lay, 6, keep)
    assert torch.equal(sel[:, :2], lay[:, keep]) and not sel[:, 2:].any()
    with pytest.raises(ValueError):
        gla.layout_tokens_full(lay, 4)


# ────────────────────────────────────────────────────────────────────────────
# Layout tokens
# ────────────────────────────────────────────────────────────────────────────

def test_layout_tokens_sit_on_the_drawn_lines(vae):
    """The plain packed latent at the tokens under the drawn lines, with ids (1, row, col) of exactly
    those target tokens (the target's own ids are (0, row, col), row-major over the strip)."""
    R = 512
    text = _text([_item(0, "GLUTEN FREE", {"0": _view(_quad(64, 192, 448, 256))}),
                  _item(1, "NET WT 18.6 OZ", {"2": _view(_quad(100, 400, 300, 420)),
                                              "0": _view(_quad(40, 40, 120, 52))})])
    strip, info = gla.layout_strip(text, R)
    lat, ids, tinfo = gla.layout_tokens(strip, info, vae)
    m = gt.text_token_mask(info["drawn_text"], R)
    idx = torch.nonzero(m).flatten()
    assert lat.shape == (1, idx.numel(), 64) and ids.shape == (idx.numel(), 3) and ids.dtype == torch.float32
    z = gla.encode_layout(strip, vae, subtract_blank=False)
    assert torch.equal(lat, z[:, idx])
    nr, nc = R // TOKEN_PX, 6 * R // TOKEN_PX
    target_ids = torch.stack(torch.meshgrid(torch.arange(nr), torch.arange(nc), indexing="ij"), -1).view(-1, 2).float()
    assert torch.equal(ids[:, 0], torch.ones(idx.numel())) and torch.equal(ids[:, 1:], target_ids[idx])
    assert tinfo["n_layout_tokens"] == idx.numel() and tinfo["token_lines"] == info["n_lines"] == 3
    assert torch.equal(gt.text_token_mask(tinfo["drawn_text"], R), m)
    # the plain latent is the channels-mode latent plus the blank latent
    blank = gla.encode_layout(np.full_like(strip, 255), vae, subtract_blank=False)
    assert torch.allclose(z - blank, gla.encode_layout(strip, vae), atol=1e-5)


def test_layout_tokens_cap_leaves_out_the_largest_lines(vae):
    R = 512
    big = _item(0, "BIG", {"0": _view(_quad(20, 20, 480, 200))})
    small = [_item(i, f"S{i}", {"1": _view(_quad(20, 20 + 30 * i, 120, 34 + 30 * i))}) for i in range(1, 5)]
    strip, info = gla.layout_strip(_text([big] + small), R)
    _, kept = gla.layout_token_index(info)
    assert len(kept) == 5
    n_small = int(gt.text_token_mask(gla.lines_text([ln for ln in info["lines"] if ln["item_id"]], R), R).sum())
    idx, kept = gla.layout_token_index(info, max_tokens=n_small)
    assert sorted(ln["item_id"] for ln in kept) == [1, 2, 3, 4] and idx.numel() == n_small
    lat, ids, tinfo = gla.layout_tokens(strip, info, vae, max_tokens=n_small)
    assert lat.shape[1] == n_small and tinfo["token_lines"] == 4
    assert sorted(it["id"] for it in tinfo["drawn_text"]["items"]) == [1, 2, 3, 4]
    # nothing drawn: no tokens
    white = np.full((R, 6 * R, 3), 255, np.uint8)
    _, info0 = gla.layout_strip(_text([]), R)
    lat0, ids0, t0 = gla.layout_tokens(white, info0, vae)
    assert lat0.shape == (1, 0, 64) and ids0.shape == (0, 3) and t0["n_layout_tokens"] == 0


# ────────────────────────────────────────────────────────────────────────────
# x_embedder wrapper
# ────────────────────────────────────────────────────────────────────────────

def _fake_transformer(c=64, d=32):
    torch.manual_seed(1)
    t = torch.nn.Module()
    t.x_embedder = torch.nn.Linear(c, d)
    t.config = types.SimpleNamespace(in_channels=c)
    return t


def test_embedder_starts_at_zero_and_learns():
    t = _fake_transformer()
    base = t.x_embedder
    xe = gla.wrap_x_embedder(t)
    assert t.x_embedder is xe and gla.wrap_x_embedder(t) is xe
    x = torch.randn(1, 7, 128)
    assert torch.equal(xe(x), base(x[..., :64]))
    assert torch.equal(xe(x[..., :64]), base(x[..., :64]))
    with pytest.raises(ValueError):
        xe(torch.randn(1, 7, 96))
    xe(x).pow(2).sum().backward()
    assert xe.layout_proj.weight.grad.abs().sum() > 0
    # zero layout channels give the base output whatever the projection holds (no bias)
    with torch.no_grad():
        xe.layout_proj.weight.normal_()
    x0 = torch.cat([x[..., :64], torch.zeros(1, 7, 64)], -1)
    assert torch.allclose(xe(x0), base(x[..., :64]))
    assert not torch.allclose(xe(x), base(x[..., :64]))


def test_projection_save_load_roundtrip(tmp_path):
    t = _fake_transformer()
    xe = gla.wrap_x_embedder(t)
    with torch.no_grad():
        xe.layout_proj.weight.normal_()
    lora = tmp_path / "pytorch_lora_weights.safetensors"
    lora.write_bytes(b"")
    path = gla.save_layout_proj(xe, str(tmp_path))
    assert path == gla.layout_proj_path(str(lora)) == gla.layout_proj_path(str(tmp_path))
    for src in (str(lora), str(tmp_path), path):
        assert torch.equal(gla.load_layout_proj(src), xe.layout_proj.weight.detach())
    t2 = _fake_transformer()
    xe2 = gla.wrap_x_embedder(t2, gla.load_layout_proj(str(lora)))
    x = torch.randn(1, 3, 128)
    assert torch.allclose(xe2(x), xe(x))
    with pytest.raises(ValueError):
        gla.wrap_x_embedder(_fake_transformer(), torch.zeros(3, 3))
    with pytest.raises(FileNotFoundError):
        gla.load_layout_proj(str(tmp_path / "nowhere" / "pytorch_lora_weights.safetensors"))


# ────────────────────────────────────────────────────────────────────────────
# Real registered layout
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.isfile(os.path.join(EVAL28, "016000233164", "text_reg.json")),
                    reason="needs the eval set's registered layout")
def test_real_registered_layout_draws_inside_the_line_boxes():
    R = 1024
    text = gt.load_text(os.path.join(EVAL28, "016000233164", "text_reg.json"))
    strip, info = gla.layout_strip(text, R)
    assert info["n_lines"] >= 5 and "0" in info["per_view"]
    ink = _ink(strip)
    m = gt.text_token_mask(info["drawn_text"], R, pad_tokens=1).view(R // TOKEN_PX, 6 * R // TOKEN_PX)
    inside = m.repeat_interleave(TOKEN_PX, 0).repeat_interleave(TOKEN_PX, 1).numpy()
    assert ink.sum() > 1000 and not (ink & ~inside).any()
