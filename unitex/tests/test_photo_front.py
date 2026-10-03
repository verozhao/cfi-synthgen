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
