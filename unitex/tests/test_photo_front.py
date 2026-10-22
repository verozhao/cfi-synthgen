import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")

from unitex import common as C  # noqa: E402
from unitex import photo_front as P  # noqa: E402


def test_put_view_inverts_split_grid():
    R = 16
    grid = np.random.default_rng(0).integers(0, 255, (2 * R, 3 * R, 3), dtype=np.uint8)
    views = C.split_grid(grid, R)
    out = np.zeros_like(grid)
    for raw in range(6):
        P.put_view(out, raw, views[raw], R)
    assert np.array_equal(out, grid)


def test_warp_into_view_maps_photo_pixels_through_the_affine():
    photo = np.zeros((200, 200), np.uint8)
    photo[50:70, 80:100] = 255                      # x 80..100, y 50..70 in photo pixels
    aff = (0.5, 0.5, 10.0, 4.0)                     # -> x 50..60, y 29..39 in view pixels
    v = P.warp_into_view(Image.fromarray(photo), aff, 128).astype(np.float64)
    ys, xs = np.nonzero(v > 127)
    assert abs(xs.mean() - 55) < 1 and abs(ys.mean() - 34) < 1
    assert (v > 127).sum() == pytest.approx(100, abs=25)


def test_warp_into_view_identity():
    img = np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    v = P.warp_into_view(Image.fromarray(img), (1.0, 1.0, 0.0, 0.0), 32)
    assert np.abs(v.astype(int) - img.astype(int)).max() <= 1


def test_masked_blur_uses_only_masked_pixels():
    img = np.zeros((64, 64, 3), np.float32)
    mask = np.zeros((64, 64), bool)
    mask[:, :32] = True
    img[mask] = 100.0
    b = P.masked_blur(img, mask, 6.0)
    assert np.allclose(b[:, :32], 100.0, atol=1e-3)


def test_blend_modes():
    rng = np.random.default_rng(2)
    gen = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    photo = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    core = np.ones((32, 32), bool)
    zero, one = np.zeros((32, 32), np.float32), np.ones((32, 32), np.float32)
    assert np.array_equal(P.blend(gen, photo, zero, core, "detail"), gen)
    assert np.array_equal(P.blend(gen, photo, one, core, "full"), photo)
    assert np.array_equal(P.blend(gen, photo, one, core, "none"), gen)
    # the photo's own detail on its own blur gives the photo back: detail(gen = photo) = photo
    assert np.abs(P.blend(photo, photo, one, core, "detail").astype(int) - photo.astype(int)).max() <= 1


def test_front_weight_follows_silhouette_and_facing():
    R = 64
    mask0 = np.zeros((R, R), bool)
    mask0[8:56, 8:56] = True
    normals = np.zeros((R, R, 3))
    normals[mask0] = P.TO_FRONT_CAMERA               # facing the front camera
    normals[8:56, 40:56] = (1.0, 0.0, 0.0)           # grazing: perpendicular to the view direction
    w, core = P.front_weight(mask0, normals, mask0.astype(float), R)
    assert w[~mask0].max() == 0.0
    assert w[32, 24] > 0.95                          # inside, facing
    assert w[32, 48] == 0.0                          # inside, grazing
    assert w[8, 24] < 0.1 and w[8, 24] < w[10, 24] < w[14, 24]    # feathered inward from the edge
    assert not core[9, 24] and core[10, 24]          # eroded by 2 * feather px (feather 1 at R 64)


def test_local_agreement_trusts_matching_layout_and_flat_views_only():
    R = 128
    rng = np.random.default_rng(3)
    base = np.full((R, R, 3), 200, np.uint8)
    for _ in range(12):                                   # blobs: layout the blur keeps
        y, x = rng.integers(8, R - 24, 2)
        base[y:y + 16, x:x + 16] = rng.integers(0, 120, 3)
    region = np.ones((R, R), bool)
    same = P.local_agreement(base, base, region, R)
    assert same.min() > 0.95
    shifted = np.roll(base, 24, axis=0)                   # misregistered photo
    assert P.local_agreement(base, shifted, region, R).mean() < 0.5
    flat = np.full((R, R, 3), 180, np.uint8)              # blank generated label
    assert P.local_agreement(flat, base, region, R).min() > 0.95


def test_auto_registration_tries_flow_only_when_the_homography_fits_poorly():
    import cv2
    rng = np.random.default_rng(0)
    R = 128
    tex = cv2.GaussianBlur(rng.random((R, R)).astype(np.float32), (0, 0), 1.5)
    tex = (255 * (tex - tex.min()) / (tex.max() - tex.min())).astype(np.uint8)
    gen0 = np.repeat(tex[..., None], 3, axis=2)
    mask0 = np.zeros((R, R), bool)
    mask0[8:120, 8:120] = True
    photo = Image.fromarray(gen0)
    _, _, fit = P.register_photo(gen0, mask0, photo, mask0, (1.0, 1.0, 0.0, 0.0), R, "auto")
    assert fit["method"] == "homography" and fit["ncc_homography"] > 0.95 and "ncc_flow" not in fit
    assert fit["layout"] == "generated"
    _, _, forced = P.register_photo(gen0, mask0, photo, mask0, (1.0, 1.0, 0.0, 0.0), R, "auto", flow_below=1.01)
    assert "ncc_flow" in forced                           # tried once the homography counts as poor
    assert forced["method"] == "flow" and forced["layout"] == "generated"      # and it reaches flow_min 0.9
    _, _, own = P.register_photo(gen0, mask0, photo, mask0, (1.0, 1.0, 0.0, 0.0), R, "auto", flow_below=1.01,
                                 flow_min=1.01)
    assert own["layout"] == "photo" and own["method"] == "affine"             # nothing fits: the photo's layout,
    assert own["ncc_matched"] > 0.95 and own["ncc_final"] == own["ncc_affine"]  # placed by the silhouettes


def test_blend_under_replaces_what_shows_where_the_photo_fades():
    gen = np.full((8, 8, 3), 200, np.uint8)
    photo = np.zeros((8, 8, 3), np.uint8)
    core = np.ones((8, 8), bool)
    w = np.full((8, 8), 0.25, np.float32)
    under = np.full((8, 8, 3), 100, np.uint8)
    out = P.blend(gen, photo, w, core, "full", under=under)
    assert np.all(out == 75)                              # 0.25 * 0 + 0.75 * 100, the generated 200 never shows


def test_text_weight_takes_whole_lines_from_the_photo(tmp_path):
    R = 128
    mask0 = np.zeros((R, R), bool)
    mask0[8:120, 8:120] = True
    normals = np.zeros((R, R, 3))
    normals[mask0] = P.TO_FRONT_CAMERA
    normals[8:120, 96:120] = (0.95, -0.31, 0.0)          # turning away: cos 0.31, below the 0.35 photo ramp start
    normals[8:120, 112:120] = (1.0, 0.0, 0.0)            # grazing: cos 0
    photo_mask = mask0.astype(float)
    # one line at 512 px in the file = 4x the view: x 40..116, y 40..48 in the view, its right end turns away
    q = [[160, 160], [464, 160], [464, 192], [160, 192]]
    (tmp_path / "text_reg.json").write_text(json.dumps({"res": 512, "items": [{"view0_quad": q}, {"text": "no quad"}]}))
    w, n = P.text_weight(tmp_path / "text_reg.json", mask0, normals, photo_mask, R)
    assert n == 1
    assert w[44, 60] > 0.95                               # on the line, facing
    assert w[44, 100] > 0.95                              # on the line where the facing ramp gives 0
    assert w[44, 116] == 0.0                              # on the line but grazing: kept generated
    assert w[80, 60] == 0.0                               # off the line
    assert w[38, 60] > 0.5                                # padded by 0.3 x the 8 px line height
    w0, n0 = P.text_weight(tmp_path / "missing.json", mask0, normals, photo_mask, R)
    assert n0 == 0 and w0.max() == 0.0


def test_own_layout_fades_into_the_photos_own_colours():
    R = 64
    gen = np.full((R, R, 3), (240, 80, 20), np.uint8)    # generated tub: orange
    photo = np.full((R, R, 3), (200, 40, 50), np.uint8)  # photo: red
    mask0 = np.zeros((R, R), bool)
    mask0[4:60, 4:60] = True
    pmask = np.zeros((R, R), np.float32)
    pmask[4:60, 4:40] = 1.0                               # the photo covers the left part of the silhouette
    u = P.own_layout_under(gen, photo, mask0, pmask, 4.0)
    assert np.abs(u[30, 15] - (200, 40, 50)).max() < 2    # inside the photo: its own colour
    assert np.abs(u[30, 55] - (240, 80, 20)).max() < 2    # beyond it: the generated colour
    assert np.array_equal(u[1, 1], gen[1, 1].astype(np.float32))      # outside the silhouette: untouched
