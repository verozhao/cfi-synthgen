"""
Tests for the registered text placement path: anchors.py --front-image and glyph kind gt at
inference (the photo registered into view 0 as the gt view).

Run: python -m pytest unitex/tests/test_positions.py
"""

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from unitex import anchors as an
from unitex import glyph as gl
from unitex import glyph_ids as gid

INK = (200, 20, 20)


def _text_doc(front_image=None):
    """One front-view item at bbox (100, 200, 260, 232) in 512 px view-0 pixels."""
    quad = [[100.0, 200.0], [260.0, 200.0], [260.0, 232.0], [100.0, 232.0]]
    G = gid.grid_from_quad(quad)
    grid = {"ns": G.shape[1], "nt": G.shape[0],
            "xy": [[[float(G[i, j, 0]), float(G[i, j, 1])] for j in range(G.shape[1])] for i in range(G.shape[0])]}
    view = {"bbox": [100.0, 200.0, 260.0, 232.0], "quad": quad, "pixels": 5120, "coverage": 1.0, "cos": 1.0,
            "height_px": 32.0, "grid": grid}
    doc = {"version": 1, "res": 512, "items": [{"id": 0, "text": "GLUTEN FREE", "conf": 1.0, "flags": [],
                                                "views": {"0": view}}]}
    if front_image:
        doc["front_image"] = front_image
    return doc


def _front(path, res=1024):
    """Registered photo in view 0: the label region (bbox x 2) painted in INK."""
    im = Image.new("RGB", (res, res), (128, 128, 128))
    ImageDraw.Draw(im).rectangle([200, 400, 519, 463], fill=INK)
    im.save(path)


def test_gt_kind_crops_the_registered_front_image(tmp_path):
    gt = pytest.importorskip("unitex.glyph_tokens")             # needs torch
    _front(tmp_path / "front_registered.png")
    json.dump(_text_doc("front_registered.png"), open(tmp_path / "text_reg.json", "w"))
    cfg = gl.GlyphConfig(infer_kind="gt")
    insts = gt.build_infer_glyphs(str(tmp_path / "text_reg.json"), cfg, view_res=1024)
    assert [i.kind for i in insts] == ["gt"]
    assert insts[0].patch.size == (320, 64)                     # the 512 px bbox at 1024, re-cropped
    a = np.asarray(insts[0].patch.convert("RGB")).astype(int)
    assert (np.abs(a - INK).sum(-1) < 30).mean() > 0.95         # the photo's own pixels, not a render
    # the patched UniTEX pipeline loads the dict first: the relative path still resolves
    insts = gt.build_infer_glyphs(gt.load_text(str(tmp_path / "text_reg.json")), cfg, view_res=1024)
    assert [i.kind for i in insts] == ["gt"]


def test_gt_kind_without_a_front_image_renders_the_box(tmp_path):
    gt = pytest.importorskip("unitex.glyph_tokens")
    json.dump(_text_doc(), open(tmp_path / "text.json", "w"))
    insts = gt.build_infer_glyphs(str(tmp_path / "text.json"), gl.GlyphConfig(infer_kind="gt"), view_res=1024)
    assert [i.kind for i in insts] == ["box"] and insts[0].patch.size == (320, 64)
    fixed = gt.build_infer_glyphs(str(tmp_path / "text.json"), gl.GlyphConfig(), view_res=1024)
    assert [i.kind for i in fixed] == ["fixed"]


def test_save_front_image_takes_the_registered_photo_panel(tmp_path):
    R = 64
    panel = np.zeros((R, 4 * R, 3), np.uint8)
    for k, v in enumerate((10, 20, 30, 40)):
        panel[:, k * R:(k + 1) * R] = v
    Image.fromarray(panel).save(tmp_path / "photo_front.png")
    an.save_front_image(tmp_path / "photo_front.png", tmp_path / "front.png")
    out = np.asarray(Image.open(tmp_path / "front.png"))
    assert out.shape == (R, R, 3) and (out == 20).all()
    Image.fromarray(panel[:, :3 * R]).save(tmp_path / "bad.png")
    with pytest.raises(ValueError):
        an.save_front_image(tmp_path / "bad.png", tmp_path / "x.png")


# ────────────────────────────────────────────────────────────────────────────
# position_audit
# ────────────────────────────────────────────────────────────────────────────

def _layout(shifts, res=512, h=20.0):
    """Items with src_id k, a 100 x h box at (50, 50 + 40 k) shifted by shifts[k] (x, y)."""
    items = []
    for k, (dx, dy) in enumerate(shifts):
        x0, y0 = 50 + dx, 50 + 40 * k + dy
        q = [[x0, y0], [x0 + 100, y0], [x0 + 100, y0 + h], [x0, y0 + h]]
        items.append({"id": k, "src_id": k, "text": f"LINE {k}", "view0_quad": (np.asarray(q) * res / 512).tolist()})
    return {"res": res, "items": items}


def test_line_offsets_are_measured_in_512_px_and_tokens():
    from unitex import position_audit as pa
    rows = pa.line_offsets(_layout([(0, 0), (0, 0)]), _layout([(6, 8), (0, 0)], res=1024))
    assert [r["src_id"] for r in rows] == [0, 1]
    assert rows[0]["centre_offset_px512"] == 10.0 and rows[0]["centre_offset_tokens"] == 1.25
    assert rows[0]["offset_over_height"] == 0.5 and rows[1]["centre_offset_px512"] == 0.0


def test_spearman_handles_ties_and_constant_input():
    from unitex import position_audit as pa
    assert pa.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert pa.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert pa.spearman([1, 1, 2, 2], [0, 0, 1, 1]) == 1.0
    assert pa.spearman([1, 2, 3], [5, 5, 5]) is None and pa.spearman([1, 2], [1, 2]) is None


def test_audit_joins_line_scores_and_draws_overlays(tmp_path):
    from unitex import position_audit as pa
    ev = tmp_path / "eval"
    sku = ev / "123"
    (sku / "run" / "cache").mkdir(parents=True)
    json.dump(_layout([(0, 0)] * 4), open(sku / "text.json", "w"))
    json.dump(_layout([(0, 0), (4, 0), (12, 0), (24, 0)]), open(sku / "text_reg.json", "w"))
    Image.new("RGB", (3 * 512, 2 * 512), (200, 200, 200)).save(sku / "run" / "cache" / "mv_rgb.png")
    lines = [{"id": k, "r_ned": ned, "r_hit": ned >= 0.8, "r_pred": "x"} for k, ned in enumerate((1.0, 0.9, 0.5, 0.2))]
    (tmp_path / "scores" / "123").mkdir(parents=True)
    json.dump({"stages": {"generated": {"lines": lines}}}, open(tmp_path / "scores" / "123" / "text_eval.json", "w"))
    s = pa.main(["--eval-dir", str(ev), "--run", "run", "--scores", str(tmp_path / "scores"), "--out", str(tmp_path / "o")])
    a = s["all"]
    assert a["n_lines"] == 4 and a["n_scored"] == 4 and a["spearman_offset_vs_error"] == 1.0
    assert a["by_offset_tokens"]["0-0.5"]["n"] == 1 and a["by_offset_tokens"]["2-inf"]["hit_rate"] == 0.0
    assert (tmp_path / "o" / "overlays" / "123.png").exists()
    rows = list(open(tmp_path / "o" / "lines.csv"))
    assert len(rows) == 5 and rows[0].startswith("sku,src_id")
