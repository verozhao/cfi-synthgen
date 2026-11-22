"""
Tests for unitex/delight_luma.py (the delight pass's shading as brightness only), CPU only.

Run: python -m pytest unitex/tests/test_delight_luma.py -v
"""

import json
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from unitex import common as C  # noqa: E402
from unitex import delight_luma as DL  # noqa: E402

R = 32


def strip(colors, seed=0, noise=20):
    """An R x 6R strip, one base colour per view tile plus the same fine pattern on every channel."""
    rng = np.random.default_rng(seed)
    a = np.zeros((R, 6 * R, 3), np.float32)
    for i, c in enumerate(colors):
        a[:, i * R:(i + 1) * R] = c
    a += rng.integers(-noise, noise + 1, (R, 6 * R, 1)).astype(np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


GOLD = [(200, 160, 40)] * 6


def test_merging_a_strip_with_itself_changes_nothing():
    hi = strip(GOLD)
    for luma in (True, False):
        assert np.array_equal(np.asarray(DL.merge(hi, hi, 2.0, luma=luma)), np.asarray(hi))


def test_luma_merge_keeps_the_texture_pass_colours_where_per_channel_copies_the_delight_colours():
    hi = strip(GOLD, noise=0)                                   # texture pass: gold
    low = strip([(170, 170, 30)] * 6, noise=0)                  # delight pass: darker and yellow
    per_channel = np.asarray(DL.merge(low, hi, 2.0, luma=False), np.float32)[R // 2, R // 2]
    luma = np.asarray(DL.merge(low, hi, 2.0, luma=True), np.float32)[R // 2, R // 2]
    assert np.abs(per_channel - (170, 170, 30)).max() <= 2      # the delight pass's colour, yellow
    gold = np.array(GOLD[0], np.float32)
    assert np.abs(luma / luma.sum() - gold / gold.sum()).max() < 0.01           # still gold
    lw = np.array(DL.LUMA, np.float32)
    assert abs(luma @ lw - np.array((170, 170, 30), np.float32) @ lw) < 2.5     # at the delight pass's brightness


def test_luma_merge_keeps_fine_detail_of_the_texture_pass():
    hi = strip(GOLD, seed=1)
    low = Image.fromarray(np.asarray(strip(GOLD, noise=0)) // 2)                # half as bright, no detail
    out = np.asarray(DL.merge(low, hi, 2.0), np.float32)
    assert out.std() > 3                                                        # the pattern is still there
    assert abs(out.mean() / np.asarray(hi, np.float32).mean() - 0.5) < 0.05     # at half the brightness


def test_the_blur_stays_inside_each_view_tile():
    hi = strip([(40, 40, 40), (220, 220, 220)] * 3, noise=0)
    low = strip([(80, 80, 80), (110, 110, 110)] * 3, noise=0)
    out = np.asarray(DL.merge(low, hi, 3.0), np.float32)
    assert np.array_equal(out[:, R - 1], out[:, R // 2]) and np.array_equal(out[:, R], out[:, R + R // 2])
    assert abs(out[0, R - 1, 0] - 40 * 84 / 44) <= 0.5 and abs(out[0, R, 0] - 220 * 114 / 224) <= 0.5       # eps = 4


def test_strip_to_grid_puts_every_view_where_split_grid_reads_it():
    views = [np.full((R, R, 3), 10 + 40 * k, np.uint8) for k in range(6)]       # raw order front right back left top bottom
    for k in range(6):
        views[k][0, 0] = (255, 0, k)                                            # a mark in each view's top left corner
    strip_img = Image.fromarray(np.concatenate([views[i] for i in C.FULL_INDEX], axis=1))
    grid = np.asarray(DL.strip_to_grid(strip_img))
    assert grid.shape == (2 * R, 3 * R, 3)
    tiles = [grid[r * R:(r + 1) * R, c * R:(c + 1) * R] for r in range(2) for c in range(3)]
    for tile, (raw, rolled) in zip(tiles, C.GRID_TILE_TO_RAW):
        want = views[raw][::-1, ::-1] if rolled else views[raw]
        assert np.array_equal(tile, want), raw


def make_run(d, shading=2.0):
    cache = d / "cache"
    (cache / "w_LTM").mkdir(parents=True)
    hi = strip(GOLD, seed=2)
    low = strip([(170, 170, 30)] * 6, noise=0).resize((3 * R, R // 2), Image.BICUBIC)
    hi.save(cache / "mv_rgb_w_light.png")
    low.save(cache / f"mv_rgb_delit_{R // 2}.png")
    DL.strip_to_grid(DL.merge(low, hi, shading * R / 512, luma=False)).save(cache / "mv_rgb.png")
    (cache / "w_LTM" / "textured_mesh.glb").write_bytes(b"bake")
    (cache / "textured_mesh.glb").write_bytes(b"bake")
    (cache / "mv_alpha.png").write_bytes(b"alpha")
    (d / "rmbg_mask_1024.png").write_bytes(b"mask")
    (d / "run_info.json").write_text(json.dumps({"sku": "1", "delight_shading": shading, "seed": 63}))
    return hi, low


def test_remerge_writes_a_new_run_without_bakes_and_leaves_the_source_alone(tmp_path):
    src, dst = tmp_path / "1" / "a", tmp_path / "1" / "a_luma"
    hi, low = make_run(src)
    before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    info = DL.remerge(src, dst, check=True)
    assert info["check_max_abs_diff"] == 0 and info["delight_luma"] and info["src_run"] == "a" and info["seed"] == 63
    assert {p: p.read_bytes() for p in src.rglob("*") if p.is_file()} == before
    want = np.asarray(DL.strip_to_grid(DL.merge(low, hi, 2.0 * R / 512, luma=True)))
    assert np.array_equal(np.asarray(Image.open(dst / "cache" / "mv_rgb.png")), want)
    assert np.array_equal(np.asarray(Image.open(dst / "mv_rgb.png")), want)
    assert not (dst / "cache" / "w_LTM").exists() and not (dst / "cache" / "textured_mesh.glb").exists()
    assert (dst / "cache" / "mv_alpha.png").read_bytes() == b"alpha" and (dst / "rmbg_mask_1024.png").exists()
    assert json.loads((dst / "run_info.json").read_text())["run_name"] == "a_luma"


def test_remerge_check_stops_when_the_stored_views_are_not_the_runs_own_merge(tmp_path):
    src, dst = tmp_path / "1" / "a", tmp_path / "1" / "b"
    make_run(src)
    g = np.asarray(Image.open(src / "cache" / "mv_rgb.png")).copy()
    g[0, 0, 0] ^= 8
    Image.fromarray(g).save(src / "cache" / "mv_rgb.png")
    with pytest.raises(ValueError, match="differs"):
        DL.remerge(src, dst, check=True)
    assert not dst.exists()


def test_remerge_needs_the_saved_passes(tmp_path):
    src = tmp_path / "1" / "a"
    (src / "cache").mkdir(parents=True)
    (src / "run_info.json").write_text(json.dumps({"delight_shading": 4.0}))
    with pytest.raises(FileNotFoundError, match="--delight-shading"):
        DL.remerge(src, tmp_path / "1" / "b")


def test_cli_runs_over_the_products_and_counts_errors(tmp_path, capsys):
    make_run(tmp_path / "1" / "a")
    (tmp_path / "2" / "a" / "cache").mkdir(parents=True)
    assert DL.main(["--eval-dir", str(tmp_path), "--src-run", "a", "--run-name", "a_luma", "--skus", "1,2", "--check"]) == 1
    out = capsys.readouterr().out
    assert "[1] (1/2) merged on luminance" in out and "[2] ERROR" in out and "Done: 1 ok, 1 errors." in out
    with pytest.raises(SystemExit):
        DL.main(["--eval-dir", str(tmp_path), "--src-run", "a", "--run-name", "a"])


def test_install_swaps_the_patched_pipelines_merge(monkeypatch):
    fake = types.ModuleType("pipeline")
    fake.shading_detail = lambda *a, **k: "per channel"
    monkeypatch.setitem(sys.modules, "pipeline", fake)
    DL.install()
    assert fake.shading_detail is DL.merge
    stock = types.ModuleType("pipeline")
    monkeypatch.setitem(sys.modules, "pipeline", stock)
    with pytest.raises(SystemExit):
        DL.install()
