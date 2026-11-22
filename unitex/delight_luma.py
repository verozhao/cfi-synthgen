"""
The delight pass's shading as brightness only.

--delight-shading S merges the two FLUX passes as lit x G(delit) / G(lit) with one ratio per colour channel. That ratio
carries the delight pass's colour changes onto the texture pass: a gold lid came out yellow. Here the ratio is one number
per pixel, taken on luminance. The texture pass keeps its colours and only its brightness follows the delight pass.

  run_unitex.py / run_unitex_fast.py ... --delight-shading 4 --delight-luma       at generation
  python unitex/delight_luma.py --eval-dir EVAL --src-run RUN --run-name RUN_luma [--skus a,b | file] [--check]
      the same merge for a finished --delight-shading run, from the two passes it saved (cache/mv_rgb_w_light.png and
      cache/mv_rgb_delit_<res>.png). No GPU and no FLUX. The source run is read only. The new run holds the views
      and no baked model: bake it with photo_front.py. --check first redoes the source run's own merge and stops if
      that does not give its mv_rgb.png back.
"""

import argparse
import datetime
import json
import os
import pathlib
import shutil
import sys

import numpy as np
from PIL import Image

LUMA = (0.299, 0.587, 0.114)


def merge(low, high, sigma, n_views=6, eps=4.0, luma=True):
    """high x G(up(low)) / G(high) on the strip of n_views tiles, G a Gaussian of sigma px inside each tile.

    low is the delight pass (PIL, any size), high the texture pass. luma=False is UniTEX.patch's shading_detail, one
    ratio per channel. luma=True takes one ratio per pixel from the luminance of the two blurs. The ratio is clipped
    to [1/5, 5]. With low = high it returns high.
    """
    import torch
    W, H = high.size
    w = W // n_views
    up = torch.from_numpy(np.asarray(low.convert("RGB").resize((W, H), Image.BICUBIC), dtype=np.float32))
    hi = torch.from_numpy(np.asarray(high.convert("RGB"), dtype=np.float32))
    r = max(1, int(round(4 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / sigma) ** 2)
    k = k / k.sum()

    def blur(x):      # (H, W, 3) -> the same, blurred per view tile with reflected borders
        t = x.permute(2, 0, 1).reshape(3, H, n_views, w).permute(2, 0, 1, 3).reshape(3 * n_views, 1, H, w)
        t = torch.nn.functional.pad(t, (r, r, r, r), mode="reflect")
        t = torch.nn.functional.conv2d(torch.nn.functional.conv2d(t, k.view(1, 1, -1, 1)), k.view(1, 1, 1, -1))
        return t.reshape(n_views, 3, H, w).permute(1, 2, 0, 3).reshape(3, H, W).permute(1, 2, 0)

    bu, bh = blur(up), blur(hi)
    if luma:
        lw = torch.tensor(LUMA, dtype=torch.float32)
        ratio = ((bu @ lw + eps) / (bh @ lw + eps)).clamp(0.2, 5.0)[..., None]
    else:
        ratio = ((bu + eps) / (bh + eps)).clamp(0.2, 5.0)
    return Image.fromarray(np.clip(np.rint((hi * ratio).numpy()), 0, 255).astype(np.uint8))


def install():
    """Make the patched UniTEX merge on luminance: infer_mv looks shading_detail up in its module at each call."""
    import pipeline as U
    if not hasattr(U, "shading_detail"):
        raise SystemExit("this UniTEX checkout has no shading_detail: --delight-luma needs UniTEX.patch with "
                         "--delight-shading")
    U.shading_detail = merge


def strip_to_grid(strip):
    """The merged strip (R x 6R, front left right back top bottom) as the 2 x 3 grid infer_mv writes to mv_rgb.png
    (front right top / back left bottom, the bottom tile turned 180 degrees)."""
    a = np.asarray(strip)
    R = a.shape[0]
    t = a.reshape(R, 6, R, -1).copy()
    t[:, 5] = t[::-1, 5, ::-1].copy()
    grid = t.transpose(1, 0, 2, 3)[[0, 2, 4, 3, 1, 5]].reshape(2, 3, R, R, -1).transpose(0, 2, 1, 3, 4)
    return Image.fromarray(np.ascontiguousarray(grid.reshape(2 * R, 3 * R, -1)))


def saved_passes(run_dir):
    """(lit strip, raw delight strip, view_res) of a finished --delight-shading run."""
    cache = pathlib.Path(run_dir) / "cache"
    lit = cache / "mv_rgb_w_light.png"
    delit = sorted(cache.glob("mv_rgb_delit_*.png"))
    if not lit.exists() or not delit:
        raise FileNotFoundError(f"{cache} has no saved passes (mv_rgb_w_light.png and mv_rgb_delit_<res>.png): "
                                "the run was not made with --delight-shading")
    high = Image.open(lit).convert("RGB")
    return high, Image.open(delit[-1]).convert("RGB"), high.size[1]


def remerge(src, dst, shading=None, check=False):
    """Write run dst: src's cache without its bakes, with mv_rgb.png merged on luminance. Returns the run info."""
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    info = json.loads((src / "run_info.json").read_text()) if (src / "run_info.json").exists() else {}
    if shading is None:
        shading = info.get("delight_shading")
    if shading is None:
        raise ValueError(f"{src}: no delight_shading in run_info.json, pass --delight-shading")
    high, low, res = saved_passes(src)
    sigma = float(shading) * res / 512
    out = {"delight_shading": float(shading), "view_res": res}
    if check:
        old = np.asarray(strip_to_grid(merge(low, high, sigma, luma=False)), np.int16)
        stored = np.asarray(Image.open(src / "cache" / "mv_rgb.png").convert("RGB"), np.int16)
        out["check_max_abs_diff"] = int(np.abs(old - stored).max()) if old.shape == stored.shape else -1
        if out["check_max_abs_diff"] != 0:
            raise ValueError(f"{src}: redoing its own merge differs from its mv_rgb.png "
                             f"(max abs difference {out['check_max_abs_diff']}), nothing written")
    grid = strip_to_grid(merge(low, high, sigma, luma=True))
    if (dst / "cache").exists():
        shutil.rmtree(dst / "cache")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "run_info.json").unlink(missing_ok=True)
    shutil.copytree(src / "cache", dst / "cache",
                    ignore=shutil.ignore_patterns("w_LTM", "wo_LTM", "textured_mesh.glb", "*.mp4", "NO_BAKE.txt"))
    grid.save(dst / "cache" / "mv_rgb.png")
    grid.save(dst / "mv_rgb.png")
    for name in ("rembg_image.png", "rmbg_mask_1024.png"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    info.update(out, run_name=dst.name, src_run=src.name, delight_luma=True, status="ok",
                time=datetime.datetime.now().isoformat(timespec="seconds"))
    (dst / "run_info.json").write_text(json.dumps(info, indent=1))
    return info


def sku_list(spec):
    if os.path.isfile(spec):
        with open(spec) as f:
            skus = [line.split("#")[0].strip() for line in f]
    else:
        skus = [s.strip() for s in spec.split(",")]
    return [s for s in skus if s]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--src-run", required=True, help="finished --delight-shading run per SKU, read only")
    p.add_argument("--run-name", required=True)
    p.add_argument("--skus", default=None, help="file or comma list (default: SKUs with the source run's passes)")
    p.add_argument("--delight-shading", type=float, default=None,
                   help="blur of the merge in px per view at 512 (default: the source run's own)")
    p.add_argument("--check", action="store_true",
                   help="first redo the source run's per-channel merge and stop if it differs from its mv_rgb.png")
    args = p.parse_args(argv)
    if args.run_name == args.src_run:
        p.error("--run-name must differ from --src-run (the source run is read only)")
    eval_dir = pathlib.Path(args.eval_dir).resolve()
    if args.skus:
        skus = sku_list(args.skus)
    else:
        skus = sorted(d.name for d in eval_dir.iterdir() if (d / args.src_run / "cache" / "mv_rgb_w_light.png").exists())
    n_err = 0
    for i, sku in enumerate(skus):
        try:
            info = remerge(eval_dir / sku / args.src_run, eval_dir / sku / args.run_name, args.delight_shading,
                           args.check)
            print(f"  [{sku}] ({i + 1}/{len(skus)}) merged on luminance at {info['view_res']} px"
                  + (", own merge reproduced" if args.check else ""), flush=True)
        except Exception as e:
            n_err += 1
            print(f"  [{sku}] ERROR {type(e).__name__}: {e}", flush=True)
    print(f"Done: {len(skus) - n_err} ok, {n_err} errors.")
    return n_err


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
