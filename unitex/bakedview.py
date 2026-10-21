"""
Baked models scored like the final view. A baked front render (mvgen.py --mode views --views 0) is an orthographic
front view of the same mesh as UniTEX's view 0. It is moved onto view 0 by the scale and shift that map its
silhouette bbox onto the view-0 silhouette bbox (mv_alpha tile 0), and written as the front tile of a copy of the
run, <sku>/<run>_bakedview/cache/mv_rgb.png. eval_text then scores it as stage delit with --align homography, so each
line is cropped where the registered photo puts it.

Why: the baked stage crops lines through a box fit of the photo onto the render. For a photo that is not exactly
frontal those crops land beside the line (on the 6 test products "General" read "neral" and "Cream Cheese" read the
"16 OZ" below it). With these crops a photo front run's baked models read what its final view reads.

  python -m unitex.bakedview --eval-dir EVAL --run RUN --renders 'renders_baked/{run}/{sku}/view_00.png' --skus FILE
  python -m unitex.bakedview ... --alpha-run RUN2 --name gemini   # renders of another pipeline on the same meshes
Then: python -m unitex.eval_text --run-name RUN_bakedview --stages delit --align homography --register-run REG ...
"""
import argparse
import json
import pathlib
import shutil

import numpy as np
from PIL import Image

from unitex import common as C
from unitex import photo_front as PF


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def render_onto_view0(render, alpha0, res, aniso=False):
    """(RGB view-0 image with the render placed by the silhouette bboxes on grey, info)."""
    bv, br = bbox(alpha0), bbox(np.asarray(render)[..., 3] > 127)
    sx, sy = (bv[2] - bv[0]) / (br[2] - br[0]), (bv[3] - bv[1]) / (br[3] - br[1])
    if not aniso:
        sx = sy = (sx + sy) / 2
    cr, cv = ((br[0] + br[2]) / 2, (br[1] + br[3]) / 2), ((bv[0] + bv[2]) / 2, (bv[1] + bv[3]) / 2)
    inv = (1 / sx, 0, cr[0] - cv[0] / sx, 0, 1 / sy, cr[1] - cv[1] / sy)      # view pixel -> render pixel
    warped = render.transform((res, res), Image.AFFINE, inv, resample=Image.BICUBIC)
    out = Image.new("RGB", (res, res), (128, 128, 128))
    out.paste(warped, (0, 0), warped)
    return np.asarray(out), {"sx": round(float(sx), 4), "sy": round(float(sy), 4),
                             "view_bbox": [int(v) for v in bv], "render_bbox": [int(v) for v in br]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--run", required=True, help="run whose baked renders are scored")
    p.add_argument("--renders", required=True, help="render path pattern with {run} and {sku}")
    p.add_argument("--skus", required=True, help="file with one SKU per line, or a comma list")
    p.add_argument("--alpha-run", default=None, help="run giving view 0 and its silhouette (default --run)")
    p.add_argument("--name", default=None, help="name of the scored copy, <name>_bakedview (default --run)")
    p.add_argument("--aniso", action="store_true", help="match width and height separately")
    a = p.parse_args(argv)
    ev = pathlib.Path(a.eval_dir)
    f = pathlib.Path(a.skus)
    skus = [l.split("#")[0].strip() for l in open(f)] if f.is_file() else a.skus.split(",")
    for sku in [s for s in skus if s]:
        src = ev / sku / (a.alpha_run or a.run)
        ren_p = pathlib.Path(a.renders.format(run=a.run, sku=sku))
        if not (src / "cache" / "mv_rgb.png").exists() or not ren_p.exists():
            print(f"[{sku}] missing view 0 or render, skipped")
            continue
        grid = np.array(Image.open(src / "cache" / "mv_rgb.png").convert("RGB"))
        res = grid.shape[0] // 2
        alpha = np.array(Image.open(src / "cache" / "mv_alpha.png").convert("L"))
        a0 = C.split_grid(np.repeat(alpha[..., None], 3, axis=2), res)[0][..., 0] > 127
        view0, info = render_onto_view0(Image.open(ren_p).convert("RGBA"), a0, res, a.aniso)
        dst = ev / sku / f"{a.name or a.run}_bakedview"
        (dst / "cache").mkdir(parents=True, exist_ok=True)
        Image.fromarray(PF.put_view(grid.copy(), 0, view0, res)).save(dst / "cache" / "mv_rgb.png")
        shutil.copy(src / "cache" / "mv_alpha.png", dst / "cache" / "mv_alpha.png")
        meta = json.loads((src / "run_info.json").read_text()) if (src / "run_info.json").exists() else {}
        meta["bakedview"] = dict(info, render=str(ren_p))
        (dst / "run_info.json").write_text(json.dumps(meta, indent=1))
        print(f"[{sku}] scale x {info['sx']} y {info['sy']}")


if __name__ == "__main__":
    main()
