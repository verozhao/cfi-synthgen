"""
Tests for the reference text blur (unitex.ref_text): the blur itself, the training dataset option
(flux_dataset ref_text_blur with precomputed polygons) and the inference wrapper around UniTEX's
preprocess_reference_image. No PaddleOCR: detectors are stand-ins.

Run: python -m pytest unitex/tests/test_ref_text.py
"""

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unitex import flux_dataset as fd
from unitex import ref_text as rt

from test_flux_train import make_synth_root  # noqa: E402

LABEL = np.array([245, 240, 225], np.uint8)     # test_flux_train's label fill behind each text line


def _checker(res, cell=2):
    yy, xx = np.mgrid[0:res, 0:res]
    c = (((yy // cell) + (xx // cell)) % 2).astype(np.float32)
    return np.repeat(c[..., None], 3, axis=2)


def _quad(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _hf(a):
    """mean absolute difference between horizontal neighbours: high for the checker, ~0 when blurred"""
    return float(np.abs(np.diff(a, axis=1)).mean())


SMALL = _quad(100, 100, 300, 130)      # 30 px tall at 1024: blurred
LARGE = _quad(500, 500, 800, 620)      # 120 px tall at 1024: kept


def test_blur_text_blurs_small_text_only():
    img = _checker(1024)
    out, w = rt.blur_text(img, [SMALL, LARGE])
    assert _hf(img[100:130, 100:300]) > 0.4
    assert _hf(out[100:130, 100:300]) < 0.02
    assert np.array_equal(out[500:620, 500:800], img[500:620, 500:800])
    assert np.array_equal(out[800:, 800:], img[800:, 800:])
    assert w[100:130, 100:300].min() > 0.99 and w[800:, 800:].max() == 0


def test_blur_text_scales_polygons_and_height_limit():
    img = _checker(512)
    polys = [SMALL, LARGE]                     # detection px at 1024
    out, w = rt.blur_text(img, polys, scale=0.5)
    assert _hf(out[50:65, 50:150]) < 0.02                       # 15 px at 512 <= 32
    assert np.array_equal(out[250:310, 250:400], img[250:310, 250:400])    # 60 px at 512 > 32
    assert rt.n_blurred(polys, 0.5, 512) == 1
    out2, _ = rt.blur_text(img, polys, scale=0.5, max_height=128)
    assert _hf(out2[250:310, 250:400]) < 0.05


def _label_detector(rgb):
    """stand-in for PaddleOCR: the bbox of each test label (one per image at most)"""
    m = np.all(rgb == LABEL, axis=-1)
    if not m.any():
        return []
    ys, xs = np.nonzero(m)
    return [_quad(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))]


@pytest.fixture(scope="module")
def synth_ref(tmp_path_factory):
    root = make_synth_root(str(tmp_path_factory.mktemp("synth_ref")))
    out = str(tmp_path_factory.mktemp("ref_text"))
    errors = rt.precompute([root], out, detect_fn=_label_detector)
    assert errors == []
    return root, out


def test_precompute_writes_every_view(synth_ref):
    root, out = synth_ref
    d = json.load(open(rt.ref_json_path(out, "synth/a")))
    assert d["res"] == [512, 512] and sorted(map(int, d["views"])) == list(range(rt.N_REF_VIEWS))
    assert all(len(v) == 1 for v in d["views"].values())
    polys, res = rt.load_ref_polys(out, "synth/b", 3)
    assert res == (512, 512) and len(polys) == 1
    assert rt.precompute([root], out, detect_fn=_label_detector) == []      # resumable: nothing to do


def _sample(ds, index, seed=0):
    random.seed(seed)
    return ds.get(index)


def test_dataset_blurs_the_reference_text(synth_ref):
    root, out = synth_ref
    plain = fd.CFIFluxDataset(root, view_res=512)
    off = fd.CFIFluxDataset(root, view_res=512, ref_text_dir=out, ref_text_blur=0.0)
    on = fd.CFIFluxDataset(root, view_res=512, ref_text_dir=out, ref_text_blur=1.0, ref_text_max_height=128)
    a, b, c = _sample(plain, 0), _sample(off, 0), _sample(on, 0)
    assert a["ref_index"] == b["ref_index"] == c["ref_index"]
    assert b["ref_text_blurred"] == 0 and c["ref_text_blurred"] == 1
    assert np.array_equal(a["rgbs_ip"].numpy(), b["rgbs_ip"].numpy())
    (x0, y0), _, (x1, y1), _ = rt.load_ref_polys(out, "synth/a", a["ref_index"])[0][0]
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    before = a["rgbs_ip"].numpy().transpose(1, 2, 0)
    after = c["rgbs_ip"].numpy().transpose(1, 2, 0)
    assert _hf(after[y0:y1, x0:x1]) < 0.3 * _hf(before[y0:y1, x0:x1])
    assert np.array_equal(after[:max(0, y0 - 80)], before[:max(0, y0 - 80)])      # far above the label
    alb_b, alb_a = a["albedos_ip"].numpy(), c["albedos_ip"].numpy()
    assert not np.array_equal(alb_b, alb_a)
    # an 80 px limit at 1024 (40 at 512) blurs the 33 px side labels and keeps the 49 px front label
    dflt = fd.CFIFluxDataset(root, view_res=512, ref_text_dir=out, ref_text_blur=1.0, ref_text_max_height=80)
    n_big = n_small = 0
    for seed in range(40):
        x = _sample(dflt, 0, seed)
        h = rt.poly_height(np.asarray(rt.load_ref_polys(out, "synth/a", x["ref_index"])[0][0]))
        same = np.array_equal(x["rgbs_ip"].numpy(), _sample(plain, 0, seed)["rgbs_ip"].numpy())
        if h > 40:
            n_big += 1
            assert same and x["ref_text_blurred"] == 0
        else:
            n_small += 1
            assert not same and x["ref_text_blurred"] == 1
    assert n_big and n_small


def test_dataset_needs_polygons_for_every_uid(synth_ref, tmp_path):
    root, _ = synth_ref
    with pytest.raises(FileNotFoundError, match="2 of 2 uids"):
        fd.CFIFluxDataset(root, view_res=512, ref_text_dir=str(tmp_path), ref_text_blur=0.5)
    with pytest.raises(ValueError, match="needs ref_text_dir"):
        fd.CFIFluxDataset(root, view_res=512, ref_text_blur=0.5)


class _DummyPipe:
    def __init__(self, img):
        self.img = img

    def preprocess_reference_image(self, save_dir, input_image_path, scale=0.95, color="grey"):
        Image.fromarray(self.img).save(os.path.join(save_dir, "processed_image.png"))


def test_wrap_reference_blur(tmp_path):
    img = (_checker(1024) * 255).astype(np.uint8)
    pipe = _DummyPipe(img)
    rt.wrap_reference_blur(pipe, detect_fn=lambda a: [SMALL, LARGE], log=lambda *a: None)
    pipe.preprocess_reference_image(str(tmp_path), "photo.png")
    out = np.asarray(Image.open(tmp_path / "processed_image.png"))
    assert np.array_equal(np.asarray(Image.open(tmp_path / "processed_image_sharp.png")), img)
    assert _hf(out[100:130, 100:300].astype(np.float32) / 255) < 0.02
    assert np.array_equal(out[500:620, 500:800], img[500:620, 500:800])
    info = json.load(open(tmp_path / "ref_text.json"))
    assert info["n_regions"] == 2 and info["n_blurred"] == 1
    assert pipe.ref_text_state["last"]["n_blurred"] == 1
    assert (tmp_path / "ref_text_mask.png").exists()


def test_run_unitex_rejects_ref_text_blur_in_dry_run(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = subprocess.run([sys.executable, os.path.join(repo, "unitex", "run_unitex.py"), "--eval-dir", str(tmp_path),
                        "--dry-run", "--ref-text-blur"], capture_output=True, text=True)
    assert r.returncode == 2 and "--ref-text-blur needs the real pipeline" in r.stderr
