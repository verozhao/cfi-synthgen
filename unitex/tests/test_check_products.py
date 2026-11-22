"""
Tests for unitex/check_products.py (the per-product check of a pipeline run), CPU only.

Run: python -m pytest unitex/tests/test_check_products.py -v
"""

import csv
import json
import os
import struct
import sys

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from unitex import check_products as K  # noqa: E402

RES = 64


def write_glb(path, roughness=None):
    pbr = {"metallicFactor": 0.0}
    if roughness is not None:
        pbr["roughnessFactor"] = roughness
    body = json.dumps({"asset": {"version": "2.0"}, "materials": [{"pbrMetallicRoughness": pbr}]}).encode()
    body += b" " * (-len(body) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 20 + len(body)) + struct.pack("<I4s", len(body), b"JSON") + body)


def panel(photo=(200, 60, 40), result=(200, 60, 40), used=True):
    """photo_front.png: generated front, registered photo, weight, result."""
    tiles = [np.full((RES, RES, 3), c, np.uint8) for c in ((90, 90, 90), photo, (0, 0, 0), result)]
    if used:
        tiles[2][8:56, 8:56] = 255
    return Image.fromarray(np.concatenate(tiles, axis=1))


def product(root, sku, views=True, pf=None, glb=0.6, photo=(200, 60, 40), result=(200, 60, 40), final=None):
    d = root / sku
    (d / "run" / "cache").mkdir(parents=True)
    (d / "manifest_entry.json").write_text(json.dumps({"sku": sku, "title": f"Product {sku}"}))
    if views:
        (d / "run" / "run_info.json").write_text(json.dumps({"status": "ok", "wall_s": 645.04}))
        (d / "run" / "cache" / "mv_rgb.png").write_bytes(b"png")
    if pf is not None:
        (d / "run_pf").mkdir()
        info = {"status": "ok", "wall_s": 26.2, "mode": "detail", "skipped": None, "text_json": "text_reg.json",
                "text_lines": 15, "replaced_frac_of_silhouette": 0.9577, "align": {"iou": 0.972},
                "register": {"method": "homography", "layout": "generated", "ncc_matched": 0.9749}}
        info.update(pf)
        (d / "run_pf" / "run_info.json").write_text(json.dumps(info))
        panel(photo, result, used=not info["skipped"]).save(d / "run_pf" / "photo_front.png")
        if glb is not None:
            write_glb(d / "run_pf" / "textured_mesh.glb", None if glb == "absent" else glb)
    if final is not None:
        (d / "run_final").mkdir()
        (d / "run_final" / "run_info.json").write_text(json.dumps({"status": "ok", "wall_s": 30.0, **final}))
        write_glb(d / "run_final" / "textured_mesh.glb", 0.6)
    return d


def test_glb_roughness_reads_the_factor_and_its_default(tmp_path):
    write_glb(tmp_path / "a.glb", 0.6)
    write_glb(tmp_path / "b.glb")
    (tmp_path / "c.glb").write_bytes(b"not a model")
    assert K.glb_roughness(tmp_path / "a.glb") == 0.6
    assert K.glb_roughness(tmp_path / "b.glb") == 1.0                 # glTF's default when the file leaves it out
    assert K.glb_roughness(tmp_path / "c.glb") is None and K.glb_roughness(tmp_path / "none.glb") is None


def test_front_colour_difference_is_the_lab_distance_of_the_coarse_colours(tmp_path):
    a, b = (200, 60, 40), (150, 200, 40)
    panel(a, a).save(tmp_path / "same.png")
    panel(a, b).save(tmp_path / "off.png")
    panel(a, b, used=False).save(tmp_path / "unused.png")
    lab = [cv2.cvtColor(np.float32([[c]]) / 255.0, cv2.COLOR_RGB2LAB)[0, 0] for c in (a, b)]
    assert K.front_colour_difference(tmp_path / "same.png") == pytest.approx(0.0, abs=0.01)
    assert K.front_colour_difference(tmp_path / "off.png") == pytest.approx(float(np.linalg.norm(lab[0] - lab[1])), rel=0.01)
    assert K.front_colour_difference(tmp_path / "unused.png") is None


def test_a_good_product_has_no_flag(tmp_path):
    row = K.check_one(product(tmp_path, "1", pf={}), "run", "run_pf")
    assert row["flags"] == "" and row["notes"] == ""
    assert (row["register"], row["ncc"], row["iou"], row["photo_share"], row["text_lines"]) == \
        ("homography/generated", 0.975, 0.972, 0.96, 15)
    assert (row["roughness"], row["views_s"], row["photo_front_s"], row["colour_dE"]) == (0.6, 645.0, 26.2, 0.0)
    assert row["title"] == "Product 1"


def test_each_problem_gets_its_flag(tmp_path):
    shin = product(tmp_path, "2", pf={"skipped": "photo not registered: NCC 0.1171 < 0.6, view 0 kept",
                                      "replaced_frac_of_silhouette": 0.0, "align": {"iou": 0.8448},
                                      "register": {"method": "affine", "layout": "photo", "ncc_matched": 0.1171}})
    row = K.check_one(shin, "run", "run_pf")
    assert row["flags"] == "front_generated outline_differs" and row["colour_dE"] == ""
    assert K.check_one(product(tmp_path, "3", pf={}, glb="absent"), "run", "run_pf")["flags"] == "matte"
    assert K.check_one(product(tmp_path, "4", pf={}, glb=None), "run", "run_pf")["flags"] == "no_model"
    assert K.check_one(product(tmp_path, "5", pf={}, result=(150, 200, 40)), "run", "run_pf")["flags"] == "colours_off"
    assert K.check_one(product(tmp_path, "6", pf={"replaced_frac_of_silhouette": 0.3}), "run", "run_pf")["flags"] == \
        "little_photo"
    assert K.check_one(product(tmp_path, "7", views=False, pf={}), "run", "run_pf")["flags"] == "no_views"
    assert K.check_one(product(tmp_path, "8"), "run", "run_pf")["flags"] == "no_model"      # no photo front run at all


def test_notes_are_not_flags(tmp_path):
    flow = product(tmp_path, "1", pf={"register": {"method": "flow", "layout": "generated", "ncc_matched": 0.956},
                                      "text_lines": 0})
    own = product(tmp_path, "2", pf={"mode": "full", "register": {"method": "affine", "layout": "photo",
                                                                  "ncc_matched": 0.742}})
    assert (lambda r: (r["flags"], r["notes"]))(K.check_one(flow, "run", "run_pf")) == ("", "flow no_text_lines")
    assert (lambda r: (r["flags"], r["notes"]))(K.check_one(own, "run", "run_pf")) == ("", "photo_layout")
    weak = product(tmp_path, "3", pf={"register": {"method": "homography", "layout": "generated", "ncc_matched": 0.8335}})
    assert (lambda r: (r["flags"], r["notes"]))(K.check_one(weak, "run", "run_pf")) == ("", "weak_fit")


def test_a_finished_run_is_checked_on_its_own_model_and_record(tmp_path):
    with_record = product(tmp_path, "1", pf={}, glb=None, final={"record": True})
    without = product(tmp_path, "2", pf={}, glb=None, final={"record": False})
    row = K.check_one(with_record, "run", "run_pf", "run_final")
    assert row["flags"] == "" and row["finish_s"] == 30.0 and row["roughness"] == 0.6
    assert K.check_one(without, "run", "run_pf", "run_final")["flags"] == "no_record"


def test_cli_writes_the_table_and_the_summary(tmp_path, capsys):
    product(tmp_path, "1", pf={})
    product(tmp_path, "2", pf={"skipped": "photo not registered", "replaced_frac_of_silhouette": 0.0,
                               "align": {"iou": 0.84}})
    product(tmp_path, "3", pf={}, glb=1.0)
    (tmp_path / "notes.txt").write_text("not a product")
    assert K.main(["--eval-dir", str(tmp_path), "--run", "run"]) == 0
    text = (tmp_path / "check_run_pf.md").read_text()
    assert "3 products, 2 with a flag, 1 without." in text
    assert "| front_generated | 1 |" in text and "| matte | 1 |" in text and "| 2 Product 2 | front_generated outline_differs |" in text
    assert "Median seconds per product, views: 645.0" in text
    assert text in capsys.readouterr().out
    rows = list(csv.DictReader(open(tmp_path / "check_run_pf.csv")))
    assert [r["sku"] for r in rows] == ["1", "2", "3"] and list(rows[0]) == list(K.COLUMNS)
    assert [r["flags"] for r in rows] == ["", "front_generated outline_differs", "matte"]
    assert K.main(["--eval-dir", str(tmp_path), "--run", "run", "--skus", "3"]) == 0
    assert "1 products, 1 with a flag" in (tmp_path / "check_run_pf.md").read_text()
