"""
FLUX VAE round trip of the UniTEX-framed reference: the representational text ceiling.

UniTEX encodes the 512 px reference with the FLUX VAE and decodes every generated view with it,
so text that does not survive encode -> decode at 512 px cannot appear in any output view.
The 1024 px round trip shows what a move to 1024 px per view would buy.

Reference source per SKU (--source auto picks the first available):
  run    UniTEX's own framing from <sku>/<run-name>/cache: processed_image.png (512) and
         rembg_image.png composited on grey (1024), i.e. exactly what FLUX saw
  local  the same framing replicated from <sku>/ref.png (prepare_eval.unitex_reference)
         with the border-colour mask standing in for RMBG-2.0

Writes <eval>/<sku>/vae/{ref512.png, vae512.png, ref1024.png, vae1024.png, vae_meta.json},
read by eval_text.py as stages ref1024 / vae512 / vae1024.

VAE: either a FLUX.1-dev diffusers dir or repo id (--flux-dir, subfolder "vae") or a single-file
ae.safetensors plus a diffusers config dir (--vae-file, --vae-config). CPU or CUDA.

Usage:
  python -m unitex.vae_ceiling --eval-dir /data/unitex_eval --flux-dir black-forest-labs/FLUX.1-dev --device cuda
  python -m unitex.vae_ceiling --eval-dir ./eval --vae-file ae.safetensors --vae-config vaecfg --device cpu --skus 012000046445
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitex.prepare_eval import read_sku_list, ref1024_box_to_photo, unitex_frame_boxes, unitex_reference

DTYPES = ("fp32", "bf16", "fp16")


# ────────────────────────────────────────────────────────────────────────────
# VAE
# ────────────────────────────────────────────────────────────────────────────

def load_vae(flux_dir=None, vae_file=None, vae_config=None, device="cpu", dtype="fp32"):
    import torch
    from diffusers import AutoencoderKL
    td = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    if vae_file:
        vae = AutoencoderKL.from_single_file(vae_file, config=vae_config, torch_dtype=td)
    elif flux_dir:
        vae = AutoencoderKL.from_pretrained(flux_dir, subfolder="vae", torch_dtype=td)
    else:
        raise ValueError("need --flux-dir or --vae-file")
    return vae.to(device).eval()


def round_trip(vae, img):
    """RGB PIL -> encode (latent mode) -> decode -> RGB PIL. Shift / scale cancel out."""
    import torch
    p = next(vae.parameters())
    x = torch.from_numpy(np.asarray(img.convert("RGB")).copy()).permute(2, 0, 1)[None]
    x = x.to(device=p.device, dtype=p.dtype) / 127.5 - 1
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
        y = vae.decode(z).sample
    y = ((y[0].float().permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
    return Image.fromarray(y)


def psnr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else round(float(10 * np.log10(255.0 ** 2 / mse)), 2)


# ────────────────────────────────────────────────────────────────────────────
# Framed references
# ────────────────────────────────────────────────────────────────────────────

def framed_references(sku_dir, run_name, source="auto"):
    """-> (refs {512: PIL, 1024: PIL}, info) where info carries the framing boxes when known."""
    sku_dir = pathlib.Path(sku_dir)
    cache = sku_dir / run_name / "cache"
    with open(sku_dir / "ref_meta.json") as f:
        meta = json.load(f)
    use_run = source == "run" or (source == "auto" and (cache / "processed_image.png").exists()
                                  and (cache / "rembg_image.png").exists())
    if use_run:
        if not (cache / "processed_image.png").exists():
            raise FileNotFoundError(f"{cache}/processed_image.png missing (--source run)")
        # rembg_image RGB is already composited on grey, processed_image = its RGB resized to 512
        ref1024 = Image.open(cache / "rembg_image.png").convert("RGB")
        refs = {512: Image.open(cache / "processed_image.png").convert("RGB"), 1024: ref1024}
        info = {"source": f"run:{run_name}"}
        mask_p = sku_dir / run_name / "rmbg_mask_1024.png"
        if mask_p.exists():
            src, dst = unitex_frame_boxes(Image.open(mask_p).convert("L"), 1024, 1024)
            info.update(src_box_1024=src, dst_box_1024=dst, mask="rmbg_mask_1024.png")
    else:
        rembg, processed, src, dst = unitex_reference(sku_dir / "ref.png")
        refs = {512: processed, 1024: rembg.convert("RGB")}
        info = {"source": "local", "src_box_1024": src, "dst_box_1024": dst, "mask": "border-colour threshold"}
    if "src_box_1024" in info:
        info["photo_box"] = ref1024_box_to_photo(info["src_box_1024"], meta)
        info["dst_box_512"] = [v / 2 for v in info["dst_box_1024"]]
    return refs, info


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)      # progress shows up in redirected logs
    p = argparse.ArgumentParser(description="FLUX VAE round trip of the UniTEX-framed reference.")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--skus", default=None, help="file or comma list (default <eval>/skus.txt)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run-name", default="unitex", help="UniTEX run to take the framing from")
    p.add_argument("--source", choices=("auto", "run", "local"), default="auto")
    p.add_argument("--flux-dir", default=None, help="FLUX.1-dev diffusers dir or repo id")
    p.add_argument("--vae-file", default=None, help="single-file ae.safetensors")
    p.add_argument("--vae-config", default=None, help="diffusers AutoencoderKL config dir for --vae-file")
    p.add_argument("--device", default=None, help="cpu / cuda (default: cuda when available)")
    p.add_argument("--dtype", choices=DTYPES, default="fp32")
    p.add_argument("--res", type=int, nargs="+", default=[512, 1024], choices=[512, 1024])
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_dir = pathlib.Path(args.eval_dir)
    skus = read_sku_list(args.skus or str(eval_dir / "skus.txt"))
    if args.limit:
        skus = skus[:args.limit]
    t0 = time.time()
    vae = load_vae(args.flux_dir, args.vae_file, args.vae_config, device, args.dtype)
    print(f"VAE loaded on {device} ({args.dtype}) in {time.time() - t0:.1f} s, {len(skus)} SKUs", flush=True)

    for sku in skus:
        d = eval_dir / sku
        out = d / "vae"
        meta_p = out / "vae_meta.json"
        old = {}
        if meta_p.exists():
            with open(meta_p) as f:
                old = json.load(f)
        # a local-framed ceiling made before the UniTEX run existed is redone from the run's framing
        cache = d / args.run_name / "cache"
        want_run = args.source == "run" or (args.source == "auto" and (cache / "processed_image.png").exists()
                                            and (cache / "rembg_image.png").exists())
        same_source = str(old.get("source", "")).startswith("run") == want_run
        if (old and not args.force and all((out / f"vae{r}.png").exists() for r in args.res)
                and old.get("run_name") == args.run_name and same_source):
            print(f"  [{sku}] done, skipping (--force to redo)")
            continue
        try:
            refs, info = framed_references(d, args.run_name, args.source)
        except Exception as e:
            print(f"  [{sku}] skipped: {type(e).__name__}: {e}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        if old and old.get("source") != info["source"]:
            for r in {512, 1024} - set(args.res):          # other framing, would not match the new boxes
                for stem in ("ref", "vae"):
                    (out / f"{stem}{r}.png").unlink(missing_ok=True)
        rec = {"sku": sku, "run_name": args.run_name, **info, "device": device, "dtype": args.dtype,
               "vae": args.vae_file or args.flux_dir, "res": {}}
        for r in args.res:
            ts = time.time()
            ref = refs[r]
            rec_img = round_trip(vae, ref)
            ref.save(out / f"ref{r}.png")
            rec_img.save(out / f"vae{r}.png")
            rec["res"][str(r)] = {"time_s": round(time.time() - ts, 1), "psnr": psnr(ref, rec_img)}
            print(f"  [{sku}] {r}px round trip {rec['res'][str(r)]['time_s']} s, "
                  f"PSNR {rec['res'][str(r)]['psnr']} dB (source {info['source']})", flush=True)
        rec["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        with open(meta_p, "w") as f:
            json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
