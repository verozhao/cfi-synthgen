import numpy as np
from PIL import Image

from unitex import bakedview as BV


def test_render_lands_on_the_view0_silhouette():
    R = 128
    alpha0 = np.zeros((R, R), bool)
    alpha0[20:100, 40:80] = True                       # view-0 silhouette: 40 wide, 80 tall
    ren = np.zeros((256, 256, 4), np.uint8)            # render: same object, 1.5x larger, elsewhere
    ren[60:180, 100:160] = (200, 30, 30, 255)
    ren[60:66, 100:160, :3] = (20, 20, 220)            # a blue band along the object's top edge
    out, info = BV.render_onto_view0(Image.fromarray(ren, "RGBA"), alpha0, R)
    assert abs(info["sx"] - 2 / 3) < 1e-3 and abs(info["sy"] - 2 / 3) < 1e-3
    red = (out[..., 0] > 150) & (out[..., 2] < 100)
    ys, xs = np.nonzero(red | (out[..., 2] > 150))
    assert abs(xs.min() - 40) <= 1 and abs(xs.max() - 79) <= 1 and abs(ys.min() - 20) <= 1 and abs(ys.max() - 99) <= 1
    assert out[21, 60, 2] > 150                        # the top band stays at the top
    assert tuple(out[5, 5]) == (128, 128, 128)          # background grey


def test_aniso_matches_width_and_height_separately():
    R = 100
    alpha0 = np.zeros((R, R), bool)
    alpha0[10:90, 30:70] = True                        # 40 x 80
    ren = np.zeros((100, 100, 4), np.uint8)
    ren[10:90, 25:75] = 255                            # 50 x 80: a slightly wider mesh
    _, iso = BV.render_onto_view0(Image.fromarray(ren, "RGBA"), alpha0, R)
    _, ani = BV.render_onto_view0(Image.fromarray(ren, "RGBA"), alpha0, R, aniso=True)
    assert iso["sx"] == iso["sy"] and ani["sx"] == 0.8 and ani["sy"] == 1.0
