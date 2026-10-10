"""
UniTEX-FLUX training dataset over mvgen output (PNG dirs and / or LMDB envs), drop-in for launch.py

Yields exactly the keys UniTEX-FLUX tasks/texturing/trainer.py (and tasks/delight_mv/trainer.py)
index, with the tensor conventions of UniTEX-FLUX data/datasets.py (MVDataset + ReconstructDataset
+ PBRTextureGenerationDataset with launch.py's config), plus the glyph fields:
  rgbs, albedos      [3, R, 6R] float [0, 1], strip slot i = raw view FULL_INDEX[i],
                     composited on white with the *_rgb alpha
  alphas             [1, R, 6R] *_rgb alpha (alpha_source="rgb") or the *_nocs hard mask ("mask")
  ccms               [3, R, 6R] CCM rotated to glTF world by c2w_post, white where the mask is 0
  native_normals     [3, R, 6R] camera normal -> glTF world normal (c2w_post @ c2w), white outside
  rgbs_ip, albedos_ip [3, Rr, Rr] one render_random view, composited on white with its alpha
  alphas_ip          [1, Rr, Rr]
  prompts, uids      str (caption/<uid>/prompt.txt, "" if missing)
  text_json          str, content of render/<uid>/text.json ("" if missing)
  ref_index          int, the chosen render_random view
  ref_text_blurred   int, 1 when the reference's small text was blurred (ref_text_blur)
R = view_res (512 default, 1024 option), Rr = ref_res (default R). Images at another size are
resized like datasets.py: bilinear + antialias for rgb / albedo / alpha, nearest for nocs / normal.

Steps per sample:
  1. render/<uid>: 6 raw views in FULL_INDEX order, alpha / mask, white compositing, c2w_post
  2. render_random/<uid>: pick a view with datasets.py:785-792 weights (cosine of the camera z axis
     to raw view 0, ((cos(clamp(1.5 theta, 0, pi)) + 1) / 2)^4, python random.choices) and decode
     only that one (the original decodes all 20)
  3. caption and text.json as strings

Glyph patches are NOT built here. The trainer builds them (unitex.glyph_tokens.build_train_glyphs)
because the stage schedule needs the global step and gt crops must come from the target the
trainer picks (rgb or albedo, trainer.py:838-843). The dataset only ships text_json.

Differences from UniTEX-FLUX on purpose: a broken sample raises with its uid instead of being
swapped for a random other one (datasets.py:555-561), render/ and render_random/ always come from
the same uid, LMDB envs are opened read-only without a lock file.

CLI (loads every uid, extra consistency checks, exit code 1 on any failure):
  python -m unitex.flux_dataset check --root <data_root> [--root <data_root2>] [--view-res 1024]
      [--export <dir>] [--json report.json] [--strict]
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitex.common import FULL_INDEX, RAW_C2W

# UniTEX-FLUX datasets.py: pt_world(glTF) = C2W_POST @ pt_blender
C2W_POST = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)
WHITE = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
N_REF_VIEWS = 20          # ReconstructDatasetConfig.full_index = 0..19
IMAGE_EXTS = ("auto", ".png", ".mdb")
ALPHA_SOURCES = ("rgb", "mask")


# ────────────────────────────────────────────────────────────────────────────
# datasets.py primitives (same maths, re-implemented so no UniTEX-FLUX import is needed)
# ────────────────────────────────────────────────────────────────────────────

def _resize(x, res, mode):
    """[N, C, H, W] -> [N, C, res, res], like datasets.load_images."""
    if x.shape[-2] == res and x.shape[-1] == res:
        return x
    if mode == "nearest":
        return resize(x, (res, res), interpolation=InterpolationMode.NEAREST, antialias=False)
    return resize(x, (res, res), interpolation=InterpolationMode.BILINEAR, antialias=True)


def on_background(x, alpha, color=WHITE):
    """datasets.change_background_color: x * a + (1 - a) * color."""
    return x * alpha + (1 - alpha) * color.to(x).view(-1, 1, 1)


def convert_image_c2w(img, c2ws, alphas=None):
    """datasets.convert_image_c2w: [N, 3, H, W] in [0, 1] rotated by c2ws[..., :3, :3], 0 outside."""
    v = (img * 2.0 - 1.0).permute(0, 2, 3, 1)
    v = torch.matmul(v, c2ws[..., :3, :3].transpose(-1, -2).unsqueeze(-3)).permute(0, 3, 1, 2)
    if alphas is not None:
        v = torch.lerp(torch.zeros_like(v), v, alphas)
    return v * 0.5 + 0.5


def make_strip(x):
    """[6, C, H, W] (strip order) -> [C, H, 6W], datasets.PBRTextureGenerationDataset.make_grid(1, 6)."""
    n, c, h, w = x.shape
    return x.reshape(1, n, c, h, w).permute(2, 0, 3, 1, 4).reshape(c, h, n * w)


def ref_weights(z_front, z_refs):
    """datasets.py:785-791: weights of the reference candidates (torch float32, same op order)."""
    cos = torch.nn.functional.cosine_similarity(z_front, z_refs, dim=-1)
    return cos.clamp(0.0, 1.0).arccos().mul(1.5).clamp(0.0, torch.pi).cos().mul(0.5).add(0.5) \
        .clamp(0.0, 1.0).square().square()


# ────────────────────────────────────────────────────────────────────────────
# Readers
# ────────────────────────────────────────────────────────────────────────────

class ImageDir:
    """One render/ or render_random/ uid directory: LMDB env (data.mdb) or loose PNGs."""

    def __init__(self, path, image_ext="auto"):
        self.path = path
        self.env = None
        self.txn = None
        use_mdb = image_ext == ".mdb" or (image_ext == "auto" and os.path.exists(os.path.join(path, "data.mdb")))
        if use_mdb:
            import lmdb
            if not os.path.exists(os.path.join(path, "data.mdb")):
                raise FileNotFoundError(f"{path}: no data.mdb (image_ext .mdb)")
            self.env = lmdb.open(path, readonly=True, lock=False, readahead=False, subdir=True)
            self.txn = self.env.begin()

    def open(self, stem):
        if self.txn is not None:
            b = self.txn.get(stem.encode())
            if b is None:
                raise KeyError(f"{self.path}/data.mdb has no key {stem!r}")
            return Image.open(BytesIO(b))
        p = os.path.join(self.path, f"{stem}.png")
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        return Image.open(p)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = self.txn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def channel(im, name):
    """to_tensor of one channel. A missing alpha channel raises (datasets.py silently skipped it)."""
    if name == "A" and "A" not in im.getbands():
        raise ValueError(f"image mode {im.mode} has no alpha channel")
    return to_tensor(im.getchannel(name))


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

class CFIFluxDataset(Dataset):
    """mvgen roots -> UniTEX-FLUX texturing samples. See the module docstring for the keys.

    roots: one data root or a list (each holds training_uid.json, render/, render_random/,
    caption/). image_ext: "auto" (data.mdb when present, else PNG), ".mdb" or ".png".
    ref_views: number of render_random candidates (UniTEX-FLUX uses 0..19), None = all in its
    metadata.json. skip_broken: log loudly and use the next uid instead of raising.
    ref_text_dir / ref_text_blur: with probability ref_text_blur, text up to ref_text_max_height
    px (at 1024, default unitex.ref_text.MAX_HEIGHT) is blurred in the reference, using the polygons
    unitex.ref_text precompute wrote for every uid, so the model has to take it from the glyphs.
    """

    def __init__(self, roots, view_res=512, ref_res=None, image_ext="auto", alpha_source="rgb",
                 ref_views=N_REF_VIEWS, trigger_prompt=None, skip_broken=False, uids=None,
                 ref_text_dir=None, ref_text_blur=0.0, ref_text_max_height=None):
        roots = [roots] if isinstance(roots, (str, os.PathLike)) else list(roots)
        if image_ext not in IMAGE_EXTS:
            raise ValueError(f"image_ext must be one of {IMAGE_EXTS}")
        if alpha_source not in ALPHA_SOURCES:
            raise ValueError(f"alpha_source must be one of {ALPHA_SOURCES}")
        if view_res % 16:
            raise ValueError(f"view_res {view_res} is not a multiple of 16 (FLUX token size)")
        self.roots = [str(r) for r in roots]
        self.view_res = int(view_res)
        self.ref_res = int(ref_res or view_res)
        self.image_ext = image_ext
        self.alpha_source = alpha_source
        self.ref_views = ref_views
        self.trigger_prompt = trigger_prompt
        self.skip_broken = skip_broken
        self.n_broken = 0
        self.samples = []
        for root in self.roots:
            with open(os.path.join(root, "training_uid.json")) as f:
                for uid in json.load(f):
                    if uids is None or uid in uids:
                        self.samples.append((root, uid))
        if not 0.0 <= float(ref_text_blur) <= 1.0:
            raise ValueError(f"ref_text_blur {ref_text_blur} is not a probability")
        if ref_text_blur > 0 and not ref_text_dir:
            raise ValueError("ref_text_blur needs ref_text_dir (unitex.ref_text precompute)")
        self.ref_text_dir = ref_text_dir
        self.ref_text_blur = float(ref_text_blur)
        self.ref_text_max_height = ref_text_max_height
        if self.ref_text_blur > 0:
            from unitex import ref_text
            missing = [u for _, u in self.samples if not os.path.isfile(ref_text.ref_json_path(ref_text_dir, u))]
            if missing:
                msg = (f"{len(missing)} of {len(self.samples)} uids have no reference text polygons in "
                       f"{ref_text_dir} (first: {missing[0]})")
                if not skip_broken:
                    raise FileNotFoundError(msg)
                print(f"[flux_dataset] {msg}: they count as broken samples", file=sys.stderr, flush=True)

    def __len__(self):
        return len(self.samples)

    def collate_fn(self, batch):
        return torch.utils.data.default_collate(batch)

    # ── pieces ──────────────────────────────────────────────────────────────

    def load_views(self, root, uid):
        """render/<uid> -> dict of [6, C, R, R] tensors in strip order + c2w [6, 4, 4] (glTF world)."""
        d = os.path.join(root, "render", uid)
        meta = load_json(os.path.join(d, "metadata.json"))
        c2w = torch.as_tensor(meta["cam2world_matrixs"], dtype=torch.float32)[list(FULL_INDEX)]
        R = self.view_res
        with ImageDir(d, self.image_ext) as src:
            rgb_ims = [src.open(f"{i:04d}_rgb") for i in FULL_INDEX]
            alpha = _resize(torch.stack([channel(im, "A") for im in rgb_ims]), R, "bilinear")
            rgb = _resize(torch.stack([to_tensor(im.convert("RGB")) for im in rgb_ims]), R, "bilinear")
            albedo = _resize(torch.stack([to_tensor(src.open(f"{i:04d}_albedo").convert("RGB"))
                                          for i in FULL_INDEX]), R, "bilinear")
            nocs_ims = [src.open(f"{i:04d}_nocs") for i in FULL_INDEX]
            mask = _resize(torch.stack([channel(im, "A") for im in nocs_ims]), R, "nearest")
            ccm = _resize(torch.stack([to_tensor(im.convert("RGB")) for im in nocs_ims]), R, "nearest")
            normal = _resize(torch.stack([to_tensor(src.open(f"{i:04d}_normal").convert("RGB"))
                                          for i in FULL_INDEX]), R, "nearest")
        rgb = on_background(rgb, alpha)
        albedo = on_background(albedo, alpha)
        ccm = on_background(convert_image_c2w(ccm, C2W_POST, mask), mask)
        normal = on_background(convert_image_c2w(normal, C2W_POST @ c2w, mask), mask)
        return {"rgb": rgb, "albedo": albedo, "alpha": alpha, "mask": mask, "ccm": ccm,
                "normal": normal, "c2w": C2W_POST @ c2w, "meta": meta}

    def ref_candidates(self, root, uid):
        d = os.path.join(root, "render_random", uid)
        c2w = torch.as_tensor(load_json(os.path.join(d, "metadata.json"))["cam2world_matrixs"],
                              dtype=torch.float32)
        n = len(c2w) if self.ref_views is None else int(self.ref_views)
        if len(c2w) < n:
            raise ValueError(f"{d}: {len(c2w)} reference cameras, need {n}")
        return d, C2W_POST @ c2w[:n]

    def pick_reference(self, c2w_views, c2w_refs):
        """datasets.py:785-792. random.choices over a tensor of weights, like the original."""
        w = ref_weights(c2w_views[[0], :3, 2], c2w_refs[:, :3, 2])
        return random.choices(range(len(c2w_refs)), weights=w, k=1)[0], w

    def load_reference(self, ref_dir, k):
        Rr = self.ref_res
        with ImageDir(ref_dir, self.image_ext) as src:
            im = src.open(f"{k:04d}_rgb")
            alpha = _resize(channel(im, "A")[None], Rr, "bilinear")
            rgb = _resize(to_tensor(im.convert("RGB"))[None], Rr, "bilinear")
            albedo = _resize(to_tensor(src.open(f"{k:04d}_albedo").convert("RGB"))[None], Rr, "bilinear")
        return {"rgbs_ip": on_background(rgb, alpha)[0], "albedos_ip": on_background(albedo, alpha)[0],
                "alphas_ip": alpha[0]}

    def load_prompt(self, root, uid):
        p = os.path.join(root, "caption", uid, "prompt.txt")
        prompt = open(p, encoding="utf-8").read().strip() if os.path.isfile(p) else ""
        return (self.trigger_prompt or "") + prompt

    def load_text_json(self, root, uid):
        p = os.path.join(root, "render", uid, "text.json")
        return open(p, encoding="utf-8").read() if os.path.isfile(p) else ""

    # ── sample ──────────────────────────────────────────────────────────────

    def get(self, index):
        root, uid = self.samples[index]
        v = self.load_views(root, uid)
        ref_dir, c2w_refs = self.ref_candidates(root, uid)
        k, _ = self.pick_reference(v["c2w"], c2w_refs)
        alphas = v["alpha"] if self.alpha_source == "rgb" else v["mask"]
        out = {
            "uids": uid,
            "prompts": self.load_prompt(root, uid),
            "rgbs": make_strip(v["rgb"]),
            "albedos": make_strip(v["albedo"]),
            "alphas": make_strip(alphas),
            "ccms": make_strip(v["ccm"]),
            "native_normals": make_strip(v["normal"]),
            "text_json": self.load_text_json(root, uid),
            "ref_index": k,
            "ref_text_blurred": 0,
        }
        out.update(self.load_reference(ref_dir, k))
        if self.ref_text_blur > 0:
            from unitex import ref_text
            if not os.path.isfile(ref_text.ref_json_path(self.ref_text_dir, uid)):
                raise FileNotFoundError(f"no reference text polygons for {uid} in {self.ref_text_dir}")
        if self.ref_text_blur > 0 and random.random() < self.ref_text_blur:
            polys, res = ref_text.load_ref_polys(self.ref_text_dir, uid, k)
            kw = {"max_height": self.ref_text_max_height} if self.ref_text_max_height else {}
            s = self.ref_res / res[0]
            if polys and ref_text.n_blurred(polys, s, self.ref_res, **kw):
                out["rgbs_ip"] = ref_text.blur_tensor(out["rgbs_ip"], polys, s, **kw)
                out["albedos_ip"] = ref_text.blur_tensor(out["albedos_ip"], polys, s, **kw)
                out["ref_text_blurred"] = 1
        return out

    def __getitem__(self, index):
        tried = 0
        while True:
            try:
                return self.get(index)
            except Exception as e:
                root, uid = self.samples[index]
                msg = f"[flux_dataset] broken sample {root}::{uid}: {type(e).__name__}: {e}"
                if not self.skip_broken or tried >= len(self):
                    raise RuntimeError(msg) from e
                self.n_broken += 1
                print(f"{msg} (skip_broken: using the next uid, {self.n_broken} skipped so far)",
                      file=sys.stderr, flush=True)
                index = (index + 1) % len(self)
                tried += 1


def build_dataset(roots, view_res=512, **kw):
    """launch.py --dataset_impl cfi entry point."""
    return CFIFluxDataset(roots, view_res=view_res, **kw)


# ────────────────────────────────────────────────────────────────────────────
# check_dataset
# ────────────────────────────────────────────────────────────────────────────

def _save_strip(x, path):
    a = x.detach().float().clamp(0, 1).mul(255).add(0.5).floor().to(torch.uint8)
    a = a.permute(1, 2, 0).numpy()
    Image.fromarray(a[..., 0] if a.shape[-1] == 1 else a).save(path)


def check_sample(ds, index, glyph_cfg=None):
    """Load one sample and run consistency checks. Returns (errors, warnings, info)."""
    errors, warns, info = [], [], {}
    root, uid = ds.samples[index]
    R = ds.view_res
    try:
        s = ds.get(index)
    except Exception as e:
        return [f"load: {type(e).__name__}: {e}"], warns, {"traceback": traceback.format_exc(limit=3)}

    shapes = {"rgbs": (3, R, 6 * R), "albedos": (3, R, 6 * R), "alphas": (1, R, 6 * R),
              "ccms": (3, R, 6 * R), "native_normals": (3, R, 6 * R),
              "rgbs_ip": (3, ds.ref_res, ds.ref_res), "albedos_ip": (3, ds.ref_res, ds.ref_res),
              "alphas_ip": (1, ds.ref_res, ds.ref_res)}
    for k, shp in shapes.items():
        t = s[k]
        if tuple(t.shape) != shp:
            errors.append(f"{k} shape {tuple(t.shape)} != {shp}")
        elif not torch.isfinite(t).all() or t.min() < -1e-6 or t.max() > 1 + 1e-6:
            errors.append(f"{k} outside [0, 1] or not finite (min {t.min():.3f} max {t.max():.3f})")

    meta = load_json(os.path.join(root, "render", uid, "metadata.json"))
    c2w = np.asarray(meta["cam2world_matrixs"], np.float64)
    if c2w.shape[0] < 6:
        errors.append(f"metadata.json has {c2w.shape[0]} cameras, need 6")
    else:
        drot = np.abs(c2w[:6, :3, :3] - RAW_C2W[:, :3, :3]).max()
        if drot > 1e-3:
            errors.append(f"camera rotations differ from common.RAW_C2W by {drot:.3g} (view order / convention)")
    src_res = int(meta.get("res", R))
    info["source_res"] = src_res
    if src_res < R:
        warns.append(f"rendered at {src_res} px, upsampled to view_res {R} (render at {R} instead)")

    # per-view masks: strip slot order
    alpha = s["alphas"][0].reshape(R, 6, R).permute(1, 0, 2)
    v = ds.load_views(root, uid)
    m = v["mask"][:, 0] > 0.5
    a = v["alpha"][:, 0] > 0.5
    for slot in range(6):
        if m[slot].sum() == 0 or alpha[slot].sum() == 0:
            errors.append(f"slot {slot} (raw {FULL_INDEX[slot]}) has an empty mask or alpha")
            continue
        iou = float((m[slot] & a[slot]).sum()) / float((m[slot] | a[slot]).sum())
        if iou < 0.9:
            warns.append(f"slot {slot}: nocs mask vs rgb alpha IoU {iou:.3f} < 0.9")
    info["mask_alpha_iou_min"] = min(
        float((m[i] & a[i]).sum()) / max(float((m[i] | a[i]).sum()), 1.0) for i in range(6))

    # CCM range (normalized frame: longest half-extent 0.95) and unit normals
    p = v["ccm"].permute(0, 2, 3, 1)[m] * 2 - 1
    n = v["normal"].permute(0, 2, 3, 1)[m] * 2 - 1
    if len(p):
        ext = float(p.abs().max())
        info["ccm_max_abs"] = ext
        if not 0.85 <= ext <= 1.0:
            warns.append(f"CCM max |p| {ext:.3f}, expected about 0.95 (normalization)")
        dn = float((n.norm(dim=-1) - 1).abs().mean())
        info["normal_unit_err"] = dn
        if dn > 0.05:
            errors.append(f"world normals are not unit length (mean ||n| - 1| = {dn:.3f})")

    # every reference candidate decodes and has alpha
    try:
        ref_dir, c2w_refs = ds.ref_candidates(root, uid)
        info["ref_views"] = len(c2w_refs)
        with ImageDir(ref_dir, ds.image_ext) as src:
            for k in range(len(c2w_refs)):
                if channel(src.open(f"{k:04d}_rgb"), "A").max() <= 0:
                    errors.append(f"render_random view {k} has an empty alpha")
                src.open(f"{k:04d}_albedo").convert("RGB")
        w = ref_weights(v["c2w"][[0], :3, 2], c2w_refs[:, :3, 2])
        info["ref_weight_sum"] = float(w.sum())
        if float(w.sum()) <= 0:
            errors.append("all reference weights are 0 (no render_random view within 60 deg of the front)")
    except Exception as e:
        errors.append(f"render_random: {type(e).__name__}: {e}")

    if not s["prompts"]:
        warns.append("empty prompt (caption/<uid>/prompt.txt missing)")

    info["text_items"] = 0
    if s["text_json"]:
        try:
            from unitex import glyph_tokens as gt
            text = json.loads(s["text_json"])
            info["text_items"] = len(text.get("items", []))
            info["text_res"] = text.get("res")
            if text.get("res") not in (None, src_res):
                warns.append(f"text.json res {text.get('res')} != render res {src_res}")
            insts = gt.build_infer_glyphs(text, glyph_cfg, view_res=R)
            info["glyph_instances"] = len(insts)
            info["glyph_tokens"] = int(sum(i.n_keep for i in insts))
            gt.text_token_mask(text, R)
        except Exception as e:
            errors.append(f"text.json: {type(e).__name__}: {e}")
    return errors, warns, info


def check_dataset(ds, export_dir=None, export_first=4, glyph_cfg=None, verbose=True):
    """Check every sample of ds. Returns a report dict, report["n_failed"] > 0 means broken data."""
    report = {"roots": ds.roots, "view_res": ds.view_res, "ref_res": ds.ref_res, "n": len(ds),
              "failed": {}, "warnings": {}, "info": {}}
    t0 = time.time()
    for i in range(len(ds)):
        root, uid = ds.samples[i]
        errs, warns, info = check_sample(ds, i, glyph_cfg)
        key = f"{root}::{uid}"
        report["info"][key] = info
        if errs:
            report["failed"][key] = errs
        if warns:
            report["warnings"][key] = warns
        if verbose:
            state = "FAIL" if errs else ("warn" if warns else "ok")
            print(f"[{i + 1}/{len(ds)}] {state:4s} {uid}" + "".join(f"\n    error: {e}" for e in errs)
                  + "".join(f"\n    warning: {w}" for w in warns), flush=True)
        if export_dir and i < export_first and not errs:
            s = ds.get(i)
            d = os.path.join(export_dir, uid.replace("/", "_"))
            os.makedirs(d, exist_ok=True)
            for k in ("rgbs", "albedos", "alphas", "ccms", "native_normals", "rgbs_ip", "albedos_ip", "alphas_ip"):
                _save_strip(s[k], os.path.join(d, f"{k}.png"))
    report["n_failed"] = len(report["failed"])
    report["n_warned"] = len(report["warnings"])
    report["seconds"] = round(time.time() - t0, 2)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="load every uid and report failures")
    c.add_argument("--root", action="append", required=True, help="data root (repeatable)")
    c.add_argument("--view-res", type=int, default=512)
    c.add_argument("--ref-res", type=int, default=None)
    c.add_argument("--image-ext", choices=IMAGE_EXTS, default="auto")
    c.add_argument("--alpha-source", choices=ALPHA_SOURCES, default="rgb")
    c.add_argument("--ref-views", type=int, default=N_REF_VIEWS)
    c.add_argument("--glyph-config", default=None, help="GlyphConfig JSON (file or inline) for the text.json dry run")
    c.add_argument("--export", default=None, help="write strip PNGs of the first samples here")
    c.add_argument("--export-first", type=int, default=4)
    c.add_argument("--json", default=None, help="write the report here")
    c.add_argument("--strict", action="store_true", help="warnings also fail")
    args = p.parse_args(argv)

    ds = CFIFluxDataset(args.root, view_res=args.view_res, ref_res=args.ref_res,
                        image_ext=args.image_ext, alpha_source=args.alpha_source, ref_views=args.ref_views)
    cfg = None
    if args.glyph_config:
        from unitex import glyph_tokens as gt
        cfg = gt.glyph_config(args.glyph_config)
    rep = check_dataset(ds, args.export, args.export_first, cfg)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    bad = rep["n_failed"] + (rep["n_warned"] if args.strict else 0)
    print(f"check_dataset: {rep['n']} uids, {rep['n_failed']} failed, {rep['n_warned']} with warnings, "
          f"{rep['seconds']} s")
    if rep["n_failed"]:
        print("FAILED UIDS:\n  " + "\n  ".join(f"{k}: {'; '.join(v)}" for k, v in rep["failed"].items()),
              file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
