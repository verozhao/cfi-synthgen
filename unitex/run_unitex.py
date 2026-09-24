"""
Run stock UniTEX over a prepared eval dir (GPU server, inside the UniTEX env).

Self-contained on purpose: needs only the UniTEX repo and the eval dir written by
prepare_eval.py (<eval>/<sku>/{ref.png, mesh.glb}), so it can be copied to the server alone.
The working directory is switched to --unitex-root because UniTEX loads its LTM config by a
relative path (pipeline.py:131).

One CustomRGBTextureFullPipeline(super_resolutions=False, filt_gradient_points=False,
filt_large_angle_points=True) is built once. Before every SKU pipe.generator is reset to
torch.Generator().manual_seed(seed) (UniTEX shares one CPU generator across both FLUX passes
and all calls, pipeline.py:148), so results do not depend on processing order.

Outputs per SKU, save_dir = <eval>/<sku>/<run-name>/ (file names from UniTEX pipeline.py):
  rembg_image.png            copy of cache/rembg_image.png
  mv_rgb.png                 copy of cache/mv_rgb.png
  textured_mesh.glb          copy of cache/textured_mesh.glb (the w_LTM bake)
  cache/processed_mesh.obj   input mesh bbox-normalized to 0.95, UVAtlas unwrap (Open3D)
  cache/rembg_image.png      RGBA 1024: ref resized to 1024x1024, RMBG-2.0 alpha, bbox-fit 0.95 on grey
  cache/processed_image.png  RGB 512 of rembg_image, the FLUX reference          -> stage ref512
  cache/mv_alpha.png         L   1536x1024, 2x3 grid f r t / b l d (raw input mesh, no AA)
  cache/mv_ccm.png           RGB 1536x1024, same grid, p * 0.5 + 0.5 on grey
  cache/mv_normal.png        RGB 1536x1024, same grid, world normals on grey
  cache/camera_info.pth      {"c2ws", "intrinsics", "perspective"}
  cache/mv_rgb_w_light.png   RGB 3072x512 strip f l r b t d(rolled 180), texture LoRA -> stage lit (slot 0)
  cache/mv_rgb.png           RGB 1536x1024 grid f r t / b l d, delight LoRA         -> stage delit (tile 0)
  cache/mv_rgb_lr.png        only with super-resolution (not used here)
  cache/sharp_pcd.ply, coarse_pcd.ply, sharp_pcd_fps.ply, coarse_pcd_fps.ply   LTM sampling
  cache/wo_LTM/              textured_mesh.glb, visable_uv_mask.png, valid_uv_mask.png,
                             completed_uv.png, textured_mesh.mp4 (only with --videos)
  cache/w_LTM/               same files plus pcd_input.ply, pcd_input_coarse.ply,
                             pcd_input_coarse_fps.ply, pcd_output.ply
  cache/textured_mesh.glb    = w_LTM/textured_mesh.glb
Added by this script:
  rmbg_mask_1024.png         RMBG-2.0 alpha on the 1024 resized ref.png (lets eval_text.py map
                             photo pixels into UniTEX's framing exactly)
  run_info.json              completion marker: seed, wall time, peak CUDA memory, versions
  dry_run.json               only in --dry-run output (placeholder content, not UniTEX)
and one JSON line per SKU in <eval>/run_log.jsonl.

Usage (GPU server):
  cd /path/to/UniTEX && python /path/to/cfi-synthgen/unitex/run_unitex.py \\
      --unitex-root . --eval-dir /data/unitex_eval --run-name unitex_s63 --seed 63 --resume
Local loop test (no GPU, no models):
  python unitex/run_unitex.py --eval-dir /tmp/eval --dry-run
"""

import argparse
import datetime
import json
import os
import shutil
import socket
import sys
import time
import traceback

GREY = (128, 128, 128)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def read_sku_list(spec):
    if spec and os.path.exists(spec):
        out = []
        with open(spec) as f:
            for line in f:
                s = line.split("#", 1)[0].strip()
                if s:
                    out.append(s.split()[0])
        return out
    return [s.strip() for s in str(spec).split(",") if s.strip()]


def git_head(root):
    """Commit of a git checkout without running git."""
    try:
        gd = os.path.join(root, ".git")
        with open(os.path.join(gd, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1]
        p = os.path.join(gd, ref)
        if os.path.exists(p):
            with open(p) as f:
                return f.read().strip()
        with open(os.path.join(gd, "packed-refs")) as f:
            for line in f:
                if line.strip().endswith(ref):
                    return line.split()[0]
    except Exception:
        pass
    return None


class MaskCapture:
    """Wraps UniTEX's RMBG2 callable to also save the predicted alpha (1024 input frame).
    UniTEX keeps only the framed rembg_image.png, which cannot be mapped back to the photo."""

    def __init__(self, inner):
        self.inner = inner
        self.path = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, image):
        out = self.inner(image)
        if self.path:
            out.getchannel("A").save(self.path)
        return out


# ────────────────────────────────────────────────────────────────────────────
# Fake pipeline (--dry-run)
# ────────────────────────────────────────────────────────────────────────────

def _border_mask(arr, tol=12):
    """Stand-in for RMBG-2.0: differs from the border colour, enclosed holes filled so white
    print inside the product is kept."""
    import numpy as np
    a = arr[..., :3].astype(np.int16)
    w = 4
    frame = np.concatenate([a[:w].reshape(-1, 3), a[-w:].reshape(-1, 3), a[:, :w].reshape(-1, 3), a[:, -w:].reshape(-1, 3)])
    bg = np.median(frame, axis=0)
    mask = np.abs(a - bg).max(-1) > tol
    try:
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(mask)
    except ImportError:
        from PIL import Image, ImageDraw
        m = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8))
        W, H = m.size
        px = m.load()
        for xy in [(x, 0) for x in range(W)] + [(x, H - 1) for x in range(W)] + \
                  [(0, y) for y in range(H)] + [(W - 1, y) for y in range(H)]:
            if px[xy] == 255:
                ImageDraw.floodfill(m, xy, 128)
        return np.asarray(m) != 128


def _unitex_frame(rgb, alpha, H, W, scale=0.95):
    """TextureTools process_image.preprocess (RGB + L alpha given), replicated."""
    import numpy as np
    from PIL import Image
    a = np.asarray(alpha) > 0
    rows, cols = np.nonzero(a.sum(-1) > 0)[0], np.nonzero(a.sum(-2) > 0)[0]
    x1, y1, x2, y2 = int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())
    dy, dx = y2 - y1, x2 - x1
    s = min(H * scale / dy, W * scale / dx)
    Ht, Wt = int(dy * s), int(dx * s)
    ox, oy = int((W - Wt) / 2), int((H - Ht) / 2)
    rgbc = rgb.crop((x1, y1, x2, y2)).resize((Wt, Ht))
    alphac = alpha.crop((x1, y1, x2, y2)).resize((Wt, Ht))
    alphat = Image.new("L", (W, H))
    alphat.paste(alphac, (ox, oy, ox + Wt, oy + Ht))
    out = Image.new("RGBA", (W, H), GREY)
    out.paste(rgbc, (ox, oy, ox + Wt, oy + Ht), alphac)
    out.putalpha(alphat)
    return out


class FakePipeline:
    """Writes UniTEX's output file set with placeholder content, for testing the loop and the
    evaluation plumbing without a GPU. The front view of both the lit strip and the delit grid
    is the UniTEX-framed photo, mv_alpha tile 0 is its silhouette, so lit / delit scores should
    reproduce ref512 exactly. textured_mesh.glb is a copy of the UNTEXTURED input mesh."""

    def __init__(self, seed=0, **kwargs):
        self.seed = seed
        self.generator = None
        self.rembg_session = None
        self.kwargs = kwargs

    def __call__(self, save_dir, input_image_path, input_mesh_path, clear_cache=False):
        import numpy as np
        from PIL import Image
        cache = os.path.join(os.path.abspath(save_dir), "cache")
        for d in (cache, os.path.join(cache, "wo_LTM"), os.path.join(cache, "w_LTM")):
            os.makedirs(d, exist_ok=True)

        im = Image.open(input_image_path).convert("RGB").resize((1024, 1024))
        mask = Image.fromarray((_border_mask(np.asarray(im)) * 255).astype(np.uint8))
        if self.rembg_session is not None and getattr(self.rembg_session, "path", None):
            mask.save(self.rembg_session.path)
        rembg = _unitex_frame(im, mask, 1024, 1024)
        rembg.save(os.path.join(cache, "rembg_image.png"))
        processed = rembg.convert("RGB").resize((512, 512))
        processed.save(os.path.join(cache, "processed_image.png"))

        alpha512 = np.asarray(rembg.getchannel("A").resize((512, 512)))
        grid_a = np.zeros((1024, 1536), np.uint8)
        grid_a[:512, :512] = np.where(alpha512 > 127, 255, 0)
        Image.fromarray(grid_a, "L").save(os.path.join(cache, "mv_alpha.png"))
        grey_grid = Image.new("RGB", (1536, 1024), GREY)
        grey_grid.save(os.path.join(cache, "mv_ccm.png"))
        grey_grid.save(os.path.join(cache, "mv_normal.png"))
        try:
            import torch
            torch.save({"c2ws": None, "intrinsics": None, "perspective": False, "dry_run": True},
                       os.path.join(cache, "camera_info.pth"))
        except ImportError:
            open(os.path.join(cache, "camera_info.pth"), "wb").close()

        strip = Image.new("RGB", (3072, 512), GREY)
        strip.paste(processed, (0, 0))
        strip.save(os.path.join(cache, "mv_rgb_w_light.png"))
        grid = Image.new("RGB", (1536, 1024), GREY)
        grid.paste(processed, (0, 0))
        grid.save(os.path.join(cache, "mv_rgb.png"))

        with open(os.path.join(cache, "processed_mesh.obj"), "w") as f:
            f.write("# dry-run placeholder\n")
        for name in ("sharp_pcd.ply", "coarse_pcd.ply", "sharp_pcd_fps.ply", "coarse_pcd_fps.ply"):
            open(os.path.join(cache, name), "w").close()
        uv = Image.new("RGB", (64, 64), GREY)
        for sub in ("wo_LTM", "w_LTM"):
            d = os.path.join(cache, sub)
            shutil.copy(input_mesh_path, os.path.join(d, "textured_mesh.glb"))
            for name in ("visable_uv_mask.png", "valid_uv_mask.png", "completed_uv.png"):
                uv.save(os.path.join(d, name))
        for name in ("pcd_input.ply", "pcd_input_coarse.ply", "pcd_input_coarse_fps.ply", "pcd_output.ply"):
            open(os.path.join(cache, "w_LTM", name), "w").close()
        shutil.copy(os.path.join(cache, "w_LTM", "textured_mesh.glb"), os.path.join(cache, "textured_mesh.glb"))

        for name in ("rembg_image.png", "mv_rgb.png", "textured_mesh.glb"):
            shutil.copy(os.path.join(cache, name), os.path.join(save_dir, name))
        with open(os.path.join(save_dir, "dry_run.json"), "w") as f:
            json.dump({"dry_run": True, "note": "placeholder outputs from run_unitex.py --dry-run: "
                       "lit/delit front = framed photo, textured_mesh.glb = untextured input mesh"}, f, indent=1)
        return os.path.join(save_dir, "rembg_image.png"), os.path.join(save_dir, "textured_mesh.glb")


# ────────────────────────────────────────────────────────────────────────────
# Real pipeline
# ────────────────────────────────────────────────────────────────────────────

def build_pipeline(args):
    root = os.path.abspath(args.unitex_root)
    if not os.path.exists(os.path.join(root, "pipeline.py")):
        raise SystemExit(f"{root} is not a UniTEX checkout (no pipeline.py)")
    os.chdir(root)                 # LTM/configs/... is opened relative to the cwd
    sys.path.insert(0, root)
    from pipeline import CustomRGBTextureFullPipeline
    pipe = CustomRGBTextureFullPipeline(
        super_resolutions=False,
        filt_gradient_points=False,
        filt_large_angle_points=True,
        seed=args.seed,
        add_lora_path=args.add_lora_path,
        add_lora_weights=args.add_lora_weights,
    )
    if not args.videos:
        # step_2_ablition always renders two 120-frame orbit mp4s, minutes per SKU for nothing
        pipe.export_video = lambda *a, **k: None
    return pipe


def install_mask_capture(pipe):
    """Only UniTEX's RMBG2 callable is wrapped. A rembg BaseSession (enable_rembg=True) is
    dispatched by isinstance inside TextureTools and must stay unwrapped."""
    inner = getattr(pipe, "rembg_session", None)
    if inner is None or type(inner).__module__.startswith("rembg"):
        return None
    cap = MaskCapture(inner)
    pipe.rembg_session = cap
    return cap


def env_info(args):
    info = {"host": socket.gethostname(), "python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_mem_gb"] = round(p.total_memory / 1024 ** 3, 1)
    except ImportError:
        pass
    try:
        import diffusers
        info["diffusers"] = diffusers.__version__
    except ImportError:
        pass
    if not args.dry_run:
        info["unitex_commit"] = git_head(os.path.abspath(args.unitex_root))
    return info


# ────────────────────────────────────────────────────────────────────────────
# Loop
# ────────────────────────────────────────────────────────────────────────────

def run(args):
    eval_dir = os.path.abspath(args.eval_dir)          # before build_pipeline changes the cwd
    if args.add_lora_path:                             # local files too (anything else is a hub id)
        args.add_lora_path = [os.path.abspath(p) if os.path.exists(p) else p for p in args.add_lora_path]
    skus = read_sku_list(args.skus or os.path.join(eval_dir, "skus.txt"))
    if args.limit:
        skus = skus[:args.limit]
    log_path = os.path.abspath(args.log or os.path.join(eval_dir, "run_log.jsonl"))

    todo = []
    for sku in skus:
        done = os.path.join(eval_dir, sku, args.run_name, "run_info.json")
        if args.resume and os.path.exists(done):
            print(f"  [{sku}] done, skipping (--resume)")
            continue
        todo.append(sku)
    print(f"{len(todo)} of {len(skus)} SKUs to run, run name {args.run_name!r}, seed {args.seed}"
          f"{' (dry run)' if args.dry_run else ''}")
    if not todo:
        return

    try:
        import torch
    except ImportError:
        torch = None
    if args.dry_run:
        pipe = FakePipeline(seed=args.seed)
        cap = pipe.rembg_session = MaskCapture(None)
    else:
        pipe = build_pipeline(args)
        cap = install_mask_capture(pipe)
    info = env_info(args)
    use_cuda = torch is not None and torch.cuda.is_available() and not args.dry_run

    n_ok = n_err = 0
    for i, sku in enumerate(todo):
        sku_dir = os.path.join(eval_dir, sku)
        save_dir = os.path.join(sku_dir, args.run_name)
        ref = os.path.join(sku_dir, "ref.png")
        mesh = os.path.join(sku_dir, "mesh.glb")
        rec = {"sku": sku, "run_name": args.run_name, "seed": args.seed, "dry_run": args.dry_run,
               "add_lora_path": args.add_lora_path, "add_lora_weights": args.add_lora_weights,
               "time": datetime.datetime.now().isoformat(timespec="seconds"), **info}
        if not (os.path.exists(ref) and os.path.exists(mesh)):
            rec.update(status="missing_input", error=f"need {ref} and {mesh}")
            print(f"  [{sku}] missing ref.png or mesh.glb, skipped")
        else:
            os.makedirs(save_dir, exist_ok=True)
            stale = os.path.join(save_dir, "run_info.json")
            if os.path.exists(stale):
                os.remove(stale)
            # front renders of the previous textured_mesh.glb (eval_text / run_eval.sh reuse them)
            shutil.rmtree(os.path.join(save_dir, "renders"), ignore_errors=True)
            if torch is not None:
                pipe.generator = torch.Generator().manual_seed(args.seed)
            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
            if cap is not None:
                cap.path = os.path.join(save_dir, "rmbg_mask_1024.png")
            print(f"  [{sku}] ({i + 1}/{len(todo)}) running")
            t0 = time.time()
            try:
                pipe(save_dir, ref, mesh, clear_cache=False)
                rec["status"] = "ok"
            except KeyboardInterrupt:
                raise
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
                print(f"  [{sku}] ERROR {rec['error']}")
            rec["wall_s"] = round(time.time() - t0, 1)
            if use_cuda:
                rec["max_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                rec["max_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
                if rec["status"] != "ok":
                    torch.cuda.empty_cache()
            if rec["status"] == "ok":
                with open(os.path.join(save_dir, "run_info.json"), "w") as f:
                    json.dump(rec, f, indent=1)
                n_ok += 1
                print(f"  [{sku}] ok in {rec['wall_s']} s"
                      + (f", peak {rec['max_mem_alloc_gb']} GB" if "max_mem_alloc_gb" in rec else ""))
            else:
                n_err += 1
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"Done: {n_ok} ok, {n_err} errors. Log: {log_path}")


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)      # progress shows up in redirected logs
    p = argparse.ArgumentParser(description="Run stock UniTEX over a prepared eval dir.")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--unitex-root", default=".", help="UniTEX repo root (becomes the cwd)")
    p.add_argument("--skus", default=None, help="file or comma list (default <eval>/skus.txt)")
    p.add_argument("--run-name", default="unitex", help="per-SKU output subdir")
    p.add_argument("--seed", type=int, default=63, help="63 is UniTEX run.py's seed")
    p.add_argument("--resume", action="store_true", help="skip SKUs with run_info.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--videos", action="store_true", help="keep UniTEX's mp4 exports")
    p.add_argument("--add-lora-path", nargs="+", default=None, help="extra LoRAs stacked on both passes")
    p.add_argument("--add-lora-weights", nargs="+", type=float, default=None)
    p.add_argument("--log", default=None, help="default <eval>/run_log.jsonl")
    p.add_argument("--dry-run", action="store_true", help="fake pipeline, placeholder outputs")
    args = p.parse_args(argv)
    if (args.add_lora_path is None) != (args.add_lora_weights is None) or (
            args.add_lora_path and len(args.add_lora_path) != len(args.add_lora_weights)):
        p.error("--add-lora-path and --add-lora-weights need the same number of values")
    run(args)


if __name__ == "__main__":
    main()
