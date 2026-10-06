"""
Tests for the step 1 evaluation plumbing: OCR geometry and parsers, UniTEX framing and layout
replicas, GT transcripts, crops, alignment, and an end-to-end synthetic run
(prepare_eval -> run_unitex --dry-run -> eval_text -> report).

The end-to-end test needs a real OCR engine: Apple Vision on macOS, or set
UNITEX_TEST_OCR=paddle on a machine with PaddleOCR. Everything else is numpy / PIL only.
UNITEX_ROOT (default: the reader checkout) enables the byte-level TextureTools comparison.
"""

import json
import os
import pathlib
import platform
import subprocess
import sys
import types

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from unitex import ocr as ocr_mod
from unitex import prepare_eval as pe
from unitex import eval_text as et
from unitex.common import fit_box_affine, apply_affine, split_grid, split_strip, FULL_INDEX

UNITEX_ROOT = pathlib.Path(os.environ.get("UNITEX_ROOT", "/Users/test/.claude/jobs/5ee54a5d/tmp/UniTEX"))
PY = sys.executable


def _ocr_backend_for_tests():
    b = os.environ.get("UNITEX_TEST_OCR")
    if b:
        return b
    return "vision" if platform.system() == "Darwin" and os.path.exists("/usr/bin/swiftc") else None


# ────────────────────────────────────────────────────────────────────────────
# OCR geometry and parsers
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_unrotate_points_matches_rot90(rot):
    h, w = 30, 50
    img = np.zeros((h, w), np.uint8)
    img[7, 31] = 255                                  # pixel centre (31.5, 7.5)
    r = np.rot90(img, rot // 90)
    ys, xs = np.nonzero(r)
    back = ocr_mod.unrotate_points([[xs[0] + 0.5, ys[0] + 0.5]], rot, w, h)   # original size
    assert np.allclose(back[0], [31.5, 7.5])


def test_unrotate_quad_keeps_reading_frame():
    # a word read in the 90-degree CCW rotated image: its reading-frame TL maps back to the
    # original bottom-left region (text running bottom to top)
    w, h = 100, 200
    rq = [[20, 10], [80, 10], [80, 30], [20, 30]]    # horizontal in the rotated (200 x 100) image
    q = ocr_mod.unrotate_points(rq, 90, w, h)
    assert np.allclose(q[0], [w - 10, 20])           # x = w - y', y = x'
    assert q[0][1] < q[1][1]                         # baseline runs downwards in the original


def test_vision_quad_to_pixels():
    q = ocr_mod.vision_quad_to_pixels([[0.1, 0.9], [0.5, 0.9], [0.5, 0.8], [0.1, 0.8]], 200, 100)
    assert np.allclose(q, [[20, 10], [100, 10], [100, 20], [20, 20]])


def test_parse_paddle2_formats():
    quad = [[1, 2], [11, 2], [11, 8], [1, 8]]
    new = [[[quad, ("PURE", 0.9)], [quad, ("LEAF", 0.8)]]]
    old = [[quad, ("PURE", 0.9)]]
    assert [(t, c) for t, c, _ in ocr_mod.parse_paddle2(new)] == [("PURE", 0.9), ("LEAF", 0.8)]
    assert [(t, c) for t, c, _ in ocr_mod.parse_paddle2(old)] == [("PURE", 0.9)]
    assert ocr_mod.parse_paddle2([None]) == [] and ocr_mod.parse_paddle2(None) == []
    assert np.allclose(ocr_mod.parse_paddle2(new)[0][2], quad)


def test_parse_paddle3_formats():
    polys = [np.array([[0, 0], [10, 0], [10, 5], [0, 5]]), np.array([[0, 9], [9, 9], [9, 12], [0, 12]])]
    res = {"rec_texts": ["NO", "SUGAR"], "rec_scores": np.array([0.99, 0.7]), "rec_polys": polys}
    out = ocr_mod.parse_paddle3([res])
    assert [t for t, _, _ in out] == ["NO", "SUGAR"] and out[1][1] == pytest.approx(0.7)

    class R:                                  # OCRResult-like object exposing only .json
        json = {"res": {"rec_texts": ["X1"], "rec_scores": [0.5], "rec_polys": [],
                        "dt_polys": [], "rec_boxes": [[1, 2, 5, 9]]}}

        def __getitem__(self, k):
            raise KeyError(k)
    out = ocr_mod.parse_paddle3([R()])
    assert out[0][0] == "X1" and np.allclose(out[0][2], [[1, 2], [5, 2], [5, 9], [1, 9]])


def test_parse_json_result_formats():
    ours = {"lines": [{"text": "A", "conf": 1, "quad": [[0, 0], [4, 0], [4, 2], [0, 2]]}], "size": [100, 50]}
    raw_vision = {"w": 100, "h": 50, "items": [{"text": "B", "conf": 0.5,
                                                "quad_norm": [[0, 1], [0.5, 1], [0.5, 0.5], [0, 0.5]]}]}
    audit = {"w": 100, "h": 50, "items": [{"t": "C", "c": 0.3, "bb": [0.0, 0.5, 0.5, 0.5]}]}
    assert np.allclose(ocr_mod.parse_json_result(ours)[0][2], [[0, 0], [4, 0], [4, 2], [0, 2]])
    assert np.allclose(ocr_mod.parse_json_result(raw_vision)[0][2], [[0, 0], [50, 0], [50, 25], [0, 25]])
    assert np.allclose(ocr_mod.parse_json_result(audit)[0][2], [[0, 0], [50, 0], [50, 25], [0, 25]])
    # rescaled to a 2x image
    assert np.allclose(ocr_mod.parse_json_result(audit, (200, 100))[0][2], [[0, 0], [100, 0], [100, 50], [0, 50]])


def test_merge_rotations_votes_and_orders():
    q = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    items = [
        ocr_mod.make_item("ENERG DRINK", 0.5, q(10, 50, 110, 60), 0),
        ocr_mod.make_item("ENERGY DRINK", 0.5, q(11, 50, 111, 61), 90),
        ocr_mod.make_item("ENERGY DRINK", 0.5, q(10, 51, 110, 60), 270),
        ocr_mod.make_item("RED BULL", 1.0, q(10, 10, 110, 40), 0),
        ocr_mod.make_item("12 FL", 0.3, q(120, 12, 140, 22), 270),      # only found rotated
    ]
    merged = ocr_mod.sort_reading_order(ocr_mod.merge_rotations(items))
    assert [m["text"] for m in merged] == ["RED BULL", "12 FL", "ENERGY DRINK"]


def test_sort_reading_order_rows():
    q = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    items = [ocr_mod.make_item(t, 1, q(*b)) for t, b in
             [("C", (5, 40, 20, 50)), ("B", (60, 11, 90, 21)), ("A", (5, 10, 40, 20))]]
    assert [i["text"] for i in ocr_mod.sort_reading_order(items)] == ["A", "B", "C"]


# ────────────────────────────────────────────────────────────────────────────
# UniTEX layout and framing replicas
# ────────────────────────────────────────────────────────────────────────────

def test_front_tile_and_slot_match_unitex_reorder():
    """Replays UniTEX pipeline.infer_mv's numpy reorders (frtbld grid -> FLUX strip -> grid)
    on labelled views: strip slot 0 and grid tile 0 are the front, split_grid / split_strip agree."""
    names = ["f", "r", "t", "b", "l", "d"]                   # export_condition order (frtbld)
    views = []
    for k in range(6):
        v = np.full((512, 512, 3), k * 40, np.uint8)
        v[0, 0] = [k, 200, 0]                                 # corner marker to detect the 180 roll
        views.append(v)
    grid = np.concatenate([np.concatenate(views[:3], 1), np.concatenate(views[3:], 1)], 0)
    # pipeline.py:242-247
    img_temp = grid.reshape(2, 512, 3, 512, -1).copy()
    img_temp[1, :, 2] = img_temp[1, ::-1, 2, ::-1]
    strip = img_temp.transpose(0, 2, 1, 3, 4).reshape(6, 512, 512, -1)[[0, 4, 1, 3, 2, 5]].transpose(1, 0, 2, 3).reshape(512, 6 * 512, -1)
    slots = [strip[:, i * 512:(i + 1) * 512] for i in range(6)]
    assert [names[int(s[1, 1, 0]) // 40] for s in slots] == ["f", "l", "r", "b", "t", "d"]
    assert np.array_equal(slots[0], views[0])                # lit stage: strip slot 0 = front
    # pipeline.py:283-285 (delight output back to the grid)
    t2 = strip.reshape(512, 6, 512, -1).copy()
    t2[:, 5] = t2[::-1, 5, ::-1]
    grid2 = t2.transpose(1, 0, 2, 3)[[0, 2, 4, 3, 1, 5]].reshape(2, 3, 512, 512, -1).transpose(0, 2, 1, 3, 4).reshape(1024, 1536, -1)
    assert np.array_equal(grid2, grid)
    assert np.array_equal(split_grid(grid2)[0], views[0])    # delit stage: grid tile 0 = front
    raw = split_strip(strip)
    assert np.array_equal(raw[FULL_INDEX[0]], views[0])


@pytest.mark.skipif(not (UNITEX_ROOT / "TextureTools/texturetools/image/process_image.py").exists(),
                    reason="UniTEX checkout not found (set UNITEX_ROOT)")
def test_unitex_frame_matches_texturetools():
    src = (UNITEX_ROOT / "TextureTools/texturetools/image/process_image.py").read_text()
    fake = types.ModuleType("rembg")
    fake.sessions = types.SimpleNamespace(BaseSession=type("BaseSession", (), {}))
    fake.remove = None
    saved = {k: sys.modules.get(k) for k in ("rembg", "rembg.sessions")}
    sys.modules["rembg"] = fake
    sys.modules["rembg.sessions"] = fake.sessions
    try:
        ns = {}
        exec(compile(src, "process_image.py", "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    rng = np.random.default_rng(0)
    rgb = Image.fromarray(rng.integers(0, 255, (300, 220, 3), dtype=np.uint8))
    a = np.zeros((300, 220), np.uint8)
    a[40:260, 30:150] = 255
    a[35:40, 60:90] = 90                                      # soft edge
    alpha = Image.fromarray(a)
    ref = ns["preprocess"](rgb, alpha=alpha, H=1024, W=1024, scale=0.95, color="grey")
    ours = pe.unitex_frame(rgb, alpha, 1024, 1024)
    assert np.array_equal(np.asarray(ref), np.asarray(ours))


def test_framing_boxes_map_photo_into_ref512(tmp_path):
    photo = Image.new("RGB", (600, 900), (255, 255, 255))
    d = ImageDraw.Draw(photo)
    d.rectangle([100, 150, 499, 799], fill=(200, 30, 30))
    d.rectangle([200, 400, 399, 449], fill=(20, 20, 200))       # a "text line"
    sq, (px, py) = pe.pad_to_square(photo)
    assert (px, py) == (150, 0) and sq.size == (900, 900)
    sq.save(tmp_path / "ref.png")
    rembg, processed, src, dst = pe.unitex_reference(tmp_path / "ref.png")
    meta = {"square_size": 900, "pad": [px, py]}
    photo_box = pe.ref1024_box_to_photo(src, meta)
    assert np.allclose(photo_box, [100, 150, 500, 800], atol=2)
    aff = fit_box_affine(photo_box, [v / 2 for v in dst])
    cx, cy = apply_affine([[300, 425]], aff)[0]                 # centre of the blue line
    assert tuple(processed.getpixel((int(cx), int(cy)))) == pytest.approx((20, 20, 200), abs=12)


def test_fill_holes_keeps_white_print():
    m = np.zeros((50, 50), bool)
    m[10:40, 10:40] = True
    m[20:25, 15:35] = False                                     # white print inside the product
    f = pe.fill_holes(m)
    assert f[22, 25] and not f[5, 5]


# ────────────────────────────────────────────────────────────────────────────
# GT transcript
# ────────────────────────────────────────────────────────────────────────────

def test_gt_text_template_parse_and_manual(tmp_path):
    lines = [
        {"text": "PURE", "conf": 1.0, "quad": [[0, 0], [10, 0], [10, 5], [0, 5]]},
        {"text": "LEAF", "conf": 1.0, "quad": [[0, 6], [10, 6], [10, 11], [0, 11]]},
        {"text": "§?junk", "conf": 0.3, "quad": [[0, 20], [4, 20], [4, 22], [0, 22]]},
    ]
    tpl = pe.gt_text_template("sku1", "Title", lines, "vision", 0.5)
    status, rows = pe.parse_gt_text(tpl)
    assert status == "auto" and [r["text"] for r in rows] == ["PURE", "LEAF"]    # "#?" row ignored
    (tmp_path / "gt_ocr.json").write_text(json.dumps({"lines": lines}))
    (tmp_path / "gt_text.txt").write_text(tpl)
    gt, src, warn = pe.load_gt(tmp_path, 0.5)
    assert src == "ocr" and [g["text"] for g in gt] == ["PURE", "LEAF"] and not warn

    edited = tpl.replace("# status: auto", "# status: manual").replace("0 | PURE\n1 | LEAF\n", "0+1 | PURE LEAF\n")
    edited += "- | NO SUGAR\nfree text line | with bar\n"
    (tmp_path / "gt_text.txt").write_text(edited)
    gt, src, warn = pe.load_gt(tmp_path, 0.5)
    assert src == "manual"
    assert [g["text"] for g in gt] == ["PURE LEAF", "NO SUGAR", "free text line | with bar"]
    assert np.allclose(gt[0]["quad"], [[0, 0], [10, 0], [10, 11], [0, 11]])     # merged boxes
    assert gt[1]["quad"] is None and gt[2]["quad"] is None


def test_gt_text_edit_without_marker_warns(tmp_path):
    lines = [{"text": "PURE", "conf": 1.0, "quad": [[0, 0], [10, 0], [10, 5], [0, 5]]}]
    tpl = pe.gt_text_template("s", "t", lines, "vision", 0.5)
    (tmp_path / "gt_ocr.json").write_text(json.dumps({"lines": lines}))
    (tmp_path / "gt_text.txt").write_text(tpl + "- | ADDED\n")
    import hashlib
    (tmp_path / "ref_meta.json").write_text(json.dumps({"gt_text_prefill_sha1": hashlib.sha1(tpl.encode()).hexdigest()}))
    gt, src, warn = pe.load_gt(tmp_path, 0.5)
    assert src == "ocr" and warn and "not 'manual'" in warn[0]


# ────────────────────────────────────────────────────────────────────────────
# Crops and alignment
# ────────────────────────────────────────────────────────────────────────────

def test_rectified_crop_uprights_rotated_text():
    img = Image.new("RGB", (200, 200), (128, 128, 128))
    d = ImageDraw.Draw(img)
    d.rectangle([90, 40, 109, 159], fill=(0, 0, 0))            # a vertical "line", 20 wide, 120 tall
    # reading frame bottom -> top: TL is the bottom-left corner
    quad = [[90, 160], [90, 40], [110, 40], [110, 160]]
    c = et.rectified_crop(img, quad, target_h=40)
    a = np.asarray(c)
    assert c.width > c.height                                   # upright: long side horizontal
    mid = a[a.shape[0] // 2, a.shape[1] // 4: 3 * a.shape[1] // 4]
    assert mid.max() < 40                                       # the line fills the crop centre


def _textured(rng, w, h):
    blocks = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return Image.fromarray(blocks).resize((w, h), Image.NEAREST)


def test_refine_affine_recovers_offset():
    rng = np.random.default_rng(1)
    photo = Image.new("RGB", (400, 600), (255, 255, 255))
    photo.paste(_textured(rng, 300, 500), (50, 50))
    photo_box = [50, 50, 350, 550]
    true_box = [100, 20, 280, 320]                              # where the content really is
    sx, sy, tx, ty = fit_box_affine(photo_box, true_box)
    stage = photo.transform((384, 340), Image.AFFINE, (1 / sx, 0, -tx / sx, 0, 1 / sy, -ty / sy),
                            resample=Image.BILINEAR, fillcolor=(128, 128, 128))
    guess = [106, 26, 290, 330]                                 # bbox fit off by a few percent
    aff, info = et.refine_affine(photo, stage, photo_box, guess)
    assert info["method"] == "refined" and info["ncc"] > info["ncc_bbox"]
    got = apply_affine([[50, 50], [350, 550]], aff)
    assert np.abs(got - np.array([[100, 20], [280, 320]])).max() < 3.5


def test_refine_affine_keeps_exact_alignment():
    rng = np.random.default_rng(2)
    photo = Image.new("RGB", (300, 300), (255, 255, 255))
    photo.paste(_textured(rng, 200, 200), (50, 50))
    stage = photo.resize((150, 150), Image.LANCZOS)
    aff, info = et.refine_affine(photo, stage, [50, 50, 250, 250], [25, 25, 125, 125])
    assert info["method"] == "bbox"
    assert np.allclose(aff, fit_box_affine([50, 50, 250, 250], [25, 25, 125, 125]))


# ────────────────────────────────────────────────────────────────────────────
# run_unitex loop (dry run)
# ────────────────────────────────────────────────────────────────────────────

def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                         # Pillow < 10.1
        for p in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        pytest.skip("no scalable font available")


def _make_bundle(root, sku="000000000001"):
    """A fake approved_bundle_v2 SKU: synthetic product photo with printed lines, box mesh."""
    trimesh = pytest.importorskip("trimesh")
    d = root / sku
    d.mkdir(parents=True)
    photo = Image.new("RGB", (1200, 1500), (255, 255, 255))
    dr = ImageDraw.Draw(photo)
    dr.rectangle([250, 100, 949, 1399], fill=(235, 200, 60))
    dr.text((600, 400), "PURE LEAF", font=_font(130), fill=(20, 20, 20), anchor="mm")
    dr.text((600, 650), "GREEN TEA", font=_font(90), fill=(20, 20, 20), anchor="mm")
    dr.text((600, 900), "NET WT 12 OZ", font=_font(55), fill=(20, 20, 20), anchor="mm")
    photo.save(d / "front_ref.png")
    box = trimesh.creation.box(extents=(0.7, 1.3, 0.3))
    box.export(d / "generated_mesh.glb")
    box.export(d / "textured.glb")
    (d / "manifest_entry.json").write_text(json.dumps({"sku": sku, "shape": "box", "title": "Synthetic tea"}))
    return sku


def test_run_unitex_dry_run_resume_and_log(tmp_path):
    ev = tmp_path / "ev"
    s = ev / "a1"
    s.mkdir(parents=True)
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([100, 50, 299, 349], fill=(10, 120, 200))
    img.save(s / "ref.png")
    (s / "mesh.glb").write_bytes(b"glb placeholder")
    (ev / "skus.txt").write_text("a1\nmissing_sku\n")
    cmd = [PY, str(REPO / "unitex" / "run_unitex.py"), "--eval-dir", str(ev), "--dry-run", "--run-name", "r"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    assert "1 ok" in out
    for f in ("rembg_image.png", "mv_rgb.png", "textured_mesh.glb", "rmbg_mask_1024.png", "run_info.json",
              "dry_run.json", "cache/processed_image.png", "cache/mv_rgb_w_light.png", "cache/mv_rgb.png",
              "cache/mv_alpha.png", "cache/mv_ccm.png", "cache/mv_normal.png", "cache/camera_info.pth",
              "cache/processed_mesh.obj", "cache/wo_LTM/textured_mesh.glb", "cache/w_LTM/textured_mesh.glb",
              "cache/w_LTM/pcd_output.ply", "cache/textured_mesh.glb"):
        assert (s / "r" / f).exists(), f
    assert Image.open(s / "r" / "cache" / "mv_rgb_w_light.png").size == (3072, 512)
    assert Image.open(s / "r" / "cache" / "mv_rgb.png").size == (1536, 1024)
    assert Image.open(s / "r" / "cache" / "mv_alpha.png").mode == "L"
    lit0 = np.asarray(Image.open(s / "r" / "cache" / "mv_rgb_w_light.png"))[:, :512]
    assert np.array_equal(lit0, np.asarray(Image.open(s / "r" / "cache" / "processed_image.png")))
    log = [json.loads(l) for l in (ev / "run_log.jsonl").read_text().splitlines()]
    assert [(r["sku"], r["status"]) for r in log] == [("a1", "ok"), ("missing_sku", "missing_input")]
    out = subprocess.run(cmd + ["--resume"], capture_output=True, text=True, check=True).stdout
    assert "done, skipping" in out


# ────────────────────────────────────────────────────────────────────────────
# End to end with a real OCR engine
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(_ocr_backend_for_tests() is None, reason="needs Apple Vision or UNITEX_TEST_OCR=paddle")
def test_end_to_end_synthetic(tmp_path):
    backend = _ocr_backend_for_tests()
    if backend == "paddle":
        pytest.importorskip("paddleocr")
    bundle = tmp_path / "bundle"
    sku = _make_bundle(bundle)
    ev = tmp_path / "ev"
    env = dict(os.environ, PYTHONPATH=str(REPO))
    run = lambda *a: subprocess.run([PY, *a], capture_output=True, text=True, env=env, cwd=str(REPO))

    r = run("-m", "unitex.prepare_eval", "--out", str(ev), "--bundle-v2", str(bundle), "--skus", sku,
            "--backend", backend)
    assert r.returncode == 0, r.stderr
    gt = json.loads((ev / sku / "gt_ocr.json").read_text())
    texts = [l["text"].upper() for l in gt["lines"]]
    assert "PURE LEAF" in texts and "GREEN TEA" in texts
    meta = json.loads((ev / sku / "ref_meta.json").read_text())
    assert meta["pad"] == [150, 0] and np.allclose(meta["fg_bbox_photo"], [250, 100, 950, 1400], atol=1)

    r = run(str(REPO / "unitex" / "run_unitex.py"), "--eval-dir", str(ev), "--dry-run")
    assert r.returncode == 0, r.stderr

    out = ev / "res"
    r = run("-m", "unitex.eval_text", "--eval-dir", str(ev), "--backend", backend, "--out", str(out),
            "--stages", "ref1024,ref512,lit,delit,baked,baseline", "--render", "never")
    assert r.returncode == 0, r.stderr
    res = json.loads((out / sku / "text_eval.json").read_text())
    st = res["stages"]
    assert st["baked"]["status"] == "skipped" and "dry-run" in st["baked"]["reason"]
    assert st["baseline"]["status"] == "skipped"
    assert res["photo_box_source"] == "rmbg_mask_1024"
    # the dry-run lit / delit fronts are the framed photo: same image, same global numbers
    for k in ("ned", "word_recall", "word_precision", "phrase_hit"):
        assert st["lit"]["global"][k] == st["ref512"]["global"][k] == st["delit"]["global"][k]
    assert st["ref512"]["align"]["method"] == "exact"
    assert st["ref1024"]["global"]["word_recall"] >= 0.99
    big = next(l for l in st["ref512"]["lines"] if res["gt_lines"][l["id"]]["text"].upper() == "PURE LEAF")
    assert big["r_ned"] == 1.0 and big["bucket"] in (">=32", "16-32")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["aggregate"]["lit"]["n_skus"] == 1 and summary["aggregate"]["baked"]["n_skus"] == 0
    for f in ("summary_stages.csv", "summary_buckets.csv", "lines.csv"):
        assert (out / f).exists()

    r = run("-m", "unitex.report", "--results", str(out))
    assert r.returncode == 0, r.stderr
    html = (out / "index.html").read_text()
    assert sku in html and "prefers-color-scheme" in html
    import re
    srcs = re.findall(r'src="([^"]+)"', html)
    assert srcs and all((out / s).exists() for s in srcs)
