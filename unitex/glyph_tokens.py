"""
GlyphAnchor glyph tokens for UniTEX-FLUX: text.json + target views -> packed VAE tokens + fp32 ids

Shared by the patched UniTEX-FLUX trainer and the UniTEX inference pipeline. numpy / torch / PIL
only, no UniTEX or UniTEX-FLUX imports (the VAE is any diffusers AutoencoderKL passed in).

Steps:
  1. text.json at any res -> 512 px geometry, because glyph.py / glyph_ids.py work at common.VIEW_RES
  2. glyph.build_glyphs (training: step schedule, rng) or glyph.build_glyphs_infer on 512 px views
  3. per-view resolution R != 512 (R a multiple of 512): every instance is rebuilt at R. Patch
     pixels grow by f = R / 512 (gt crops re-cropped from the R px views, renders upsampled) and
     ids are recomputed at R with glyph_ids (res=R) plus the instance's jitter / nudge shift x f.
     Glyphs then cover the same part of the object at every R, and the token budget grows by f^2.
  4. VAE encode patches grouped by size: z = (vae.encode(x * 2 - 1).latent_dist.sample() - shift)
     * scale, exactly like trainer.py:846-849 and pipeline.py:_encode_vae_image
  5. pack 2x2 latent pixels into one token (UniTEX _pack_latents(pixel_shuffle=True), row-major),
     drop tokens whose keep is False (warp mode), concatenate in instance order
Result: glyph_latents [1, Ng, 4 * latent_channels] and glyph_ids [Ng, 3] float32 (axis0 = glyph
plane, rows / cols on the target strip token grid). Append them AFTER UniTEX's random condition
drop and slice the transformer output to the first n_noise tokens before the loss.

Also: text_token_mask (target tokens inside text boxes, for the keep-mask boost) and
glyph_config (GlyphConfig from JSON + key=value overrides).
"""

import json
import math
import os
import warnings
from dataclasses import fields

import numpy as np
import torch
from PIL import Image

try:
    from unitex import glyph as gl
    from unitex import glyph_ids as gid
    from unitex.common import FULL_INDEX, TOKEN_PX, VIEW_RES
except ImportError:   # imported with unitex/ itself on sys.path
    import glyph as gl
    import glyph_ids as gid
    from common import FULL_INDEX, TOKEN_PX, VIEW_RES

_R = getattr(Image, "Resampling", Image)
LANCZOS, BOX = _R.LANCZOS, _R.BOX


# ────────────────────────────────────────────────────────────────────────────
# text.json handling
# ────────────────────────────────────────────────────────────────────────────

def load_text(src):
    """text.json dict / JSON string / path / "" or None -> dict or None (no text)."""
    if src is None:
        return None
    if isinstance(src, dict):
        return src
    if isinstance(src, (list, tuple)):
        return {"version": 1, "res": VIEW_RES, "items": list(src)}
    s = str(src)
    if not s.strip():
        return None
    if s.lstrip().startswith("{"):
        return json.loads(s)
    with open(s) as f:
        doc = json.load(f)
    if isinstance(doc, dict):
        doc.setdefault("_dir", os.path.dirname(os.path.abspath(s)))    # resolves a relative front_image
    return doc


def _scale_pts(v, s):
    if v is None:
        return None
    return [[float(p[0]) * s, float(p[1]) * s] if p is not None else None for p in v]


def rescale_text(text, to_res):
    """Copy of a text.json dict with every view-pixel quantity scaled to `to_res` px per view.

    bbox, quad, grid xy and height_px scale by to_res / res, pixels by its square. src_quad is in
    texture or photo pixels and is left alone. coverage and cos are resolution free.
    """
    if text is None:
        return None
    res = int(text.get("res", VIEW_RES))
    if res == int(to_res):
        return text
    s = float(to_res) / res
    items = []
    for it in text.get("items", []):
        it = dict(it)
        views = {}
        for k, v in (it.get("views") or {}).items():
            v = dict(v)
            if v.get("bbox") is not None:
                v["bbox"] = [float(x) * s for x in v["bbox"]]
            if v.get("quad") is not None:
                v["quad"] = _scale_pts(v["quad"], s)
            if v.get("height_px") is not None:
                v["height_px"] = float(v["height_px"]) * s
            if v.get("pixels") is not None:
                v["pixels"] = float(v["pixels"]) * s * s
            if v.get("grid"):
                g = dict(v["grid"])
                g["xy"] = [_scale_pts(row, s) for row in g["xy"]]
                v["grid"] = g
            views[k] = v
        it["views"] = views
        items.append(it)
    return {**text, "res": int(to_res), "items": items}


def _items_by_id(text):
    return {int(it["id"]) if "id" in it else k: it for k, it in enumerate(text.get("items", []))}


# ────────────────────────────────────────────────────────────────────────────
# Views
# ────────────────────────────────────────────────────────────────────────────

def strip_to_views(strip, view_res=None):
    """Target strip -> 6 raw-ordered uint8 (R, R, 3) views.

    strip: torch [3, R, 6R] or [1, 3, R, 6R] float in [0, 1] (the trainer's training_image), or a
    numpy (R, 6R, 3|4) uint8 / float array. Strip slot i holds raw view FULL_INDEX[i].
    """
    if isinstance(strip, torch.Tensor):
        t = strip.detach()
        if t.ndim == 4:
            t = t[0]
        a = t.float().clamp(0, 1).mul(255).add(0.5).floor().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    else:
        a = np.asarray(strip)
        if a.dtype != np.uint8:
            a = np.clip(a.astype(np.float64) * (255.0 if a.max() <= 1.0 else 1.0) + 0.5, 0, 255).astype(np.uint8)
    a = a[..., :3]
    R = int(view_res or a.shape[0])
    if a.shape[1] != 6 * R:
        raise ValueError(f"strip is {a.shape[:2]}, expected ({R}, {6 * R})")
    views = [None] * 6
    for slot in range(6):
        views[FULL_INDEX[slot]] = np.ascontiguousarray(a[:, slot * R:(slot + 1) * R])
    return views


def resize_views(views, res):
    if views is None:
        return None
    out = []
    for v in views:
        if v is None or v.shape[0] == res:
            out.append(v)
            continue
        im = Image.fromarray(v)
        out.append(np.asarray(im.resize((res, res), BOX if v.shape[0] > res else LANCZOS)))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Rescale instances to the per-view resolution
# ────────────────────────────────────────────────────────────────────────────

def _scale_factor(view_res):
    f = view_res / VIEW_RES
    if view_res % VIEW_RES or f < 1:
        raise ValueError(f"view_res must be a multiple of {VIEW_RES}, got {view_res}")
    return int(f)


def _nominal_ids(inst, view, cfg, token_hw, f, res):
    """Ids glyph._anchor would give this instance at res = f * 512 before jitter / nudge."""
    x0, y0, x1, y1 = [float(v) * f for v in inst.bbox]
    if inst.mode == "center":
        return gid.center_ids((x0, y0, x1, y1), inst.raw_view, token_hw, cfg.frame,
                              cfg.quantize_ids, res=res), None
    if inst.mode == "stretch":
        return gid.stretch_ids((x0, y0, x1, y1), inst.raw_view, token_hw, cfg.frame, res=res), None
    G, _, _ = gl.warp_geometry(view, cfg)
    return gid.warp_ids(G * f, inst.raw_view, token_hw, cfg.frame, mask_null=cfg.mask_null, res=res)


def _shift_of(inst, nominal):
    """Integer (dr, dc) that jitter + nudge added to the instance ids (median over kept tokens)."""
    ok = inst.keep & np.isfinite(inst.ids[:, 1]) & np.isfinite(nominal[:, 1])
    if not ok.any():
        return 0, 0
    d = inst.ids[ok, 1:].astype(np.float64) - nominal[ok, 1:].astype(np.float64)
    return int(np.round(np.median(d[:, 0]))), int(np.round(np.median(d[:, 1])))


def _natural_gt_size(inst, view, cfg):
    if inst.mode == "warp":
        _, gh, gw = gl.warp_geometry(view, cfg)
        return gw * TOKEN_PX, gh * TOKEN_PX
    x0, y0, x1, y1 = inst.bbox
    return gl.round16(x1 - x0), gl.round16(y1 - y0)


def scale_instances(instances, text512, views, view_res, cfg):
    """512 px GlyphInstances -> the same instances at view_res (see module docstring, step 3).

    text512: the text.json dict at 512 px the instances were built from. views: raw-ordered
    uint8 views at view_res for gt re-crops (None or None entries: the 512 patch is upsampled).
    """
    f = _scale_factor(view_res)
    if f == 1:
        return instances
    by_id = _items_by_id(text512)
    out = []
    for inst in instances:
        view = by_id[inst.item_id]["views"][str(inst.raw_view)]
        W, H = inst.patch.size
        hw = (inst.token_hw[0] * f, inst.token_hw[1] * f)
        nominal512 = _nominal_ids(inst, view, cfg, inst.token_hw, 1, VIEW_RES)[0]
        dr, dc = _shift_of(inst, nominal512)
        ids, keep = _nominal_ids(inst, view, cfg, hw, f, view_res)
        if keep is None:
            keep = np.ones(len(ids), bool)
        (rl, rh), (cl, ch) = gid._shift_range(ids, keep, inst.raw_view, view_res)
        ids = gid.shift_ids(ids, min(max(f * dr, rl), rh), min(max(f * dc, cl), ch))
        img = views[inst.raw_view] if views is not None and inst.raw_view < len(views) else None
        recrop = (inst.kind == "gt" and img is not None and not cfg.augment_gt
                  and (W, H) == _natural_gt_size(inst, view, cfg))
        if recrop and inst.mode == "warp":
            G, _, _ = gl.warp_geometry(view, cfg)
            patch = gl.crop_gt_grid(img, G * f, (W * f, H * f))
        elif recrop:
            x0, y0, x1, y1 = [float(v) * f for v in inst.bbox]
            patch = gl.crop_gt(img, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], (W * f, H * f))
        else:
            patch = inst.patch.resize((W * f, H * f), LANCZOS)
        out.append(gl.GlyphInstance(
            item_id=inst.item_id, raw_view=inst.raw_view, patch=patch, token_hw=hw,
            ids=np.asarray(ids, np.float32), keep=np.asarray(keep, bool), kind=inst.kind,
            text=inst.text, mode=inst.mode, bbox=tuple(float(v) * f for v in inst.bbox),
            quad=tuple((float(p[0]) * f, float(p[1]) * f) for p in inst.quad), rank=inst.rank))
    ids, slices = gid.concat_ids(out)
    n = gid.count_collisions(ids, slices)
    if n and cfg.warn_collisions:
        warnings.warn(f"{n} glyph tokens share an id triple after rescaling to {view_res} px")
    return out


# ────────────────────────────────────────────────────────────────────────────
# Build (training / inference) at any per-view resolution
# ────────────────────────────────────────────────────────────────────────────

def glyph_rng(seed, *keys):
    """Deterministic per-sample generator, e.g. glyph_rng(seed, global_step, micro_step, rank)."""
    return np.random.default_rng([int(seed or 0) & 0xFFFFFFFF] + [int(k) & 0xFFFFFFFF for k in keys])


def build_train_glyphs(text, target, step, rng, cfg=None, view_res=VIEW_RES):
    """Training glyph instances for one sample at view_res.

    text: text.json (anything load_text accepts). target: the training target strip (torch
    [3, R, 6R] float in [0, 1], what the trainer VAE-encodes) or 6 raw-ordered uint8 views, used
    for gt crops. None makes gt fall back to box renders.
    """
    cfg = cfg or gl.GlyphConfig()
    text = load_text(text)
    if text is None or not text.get("items"):
        return []
    text512 = rescale_text(text, VIEW_RES)
    if target is None:
        views = None
    elif isinstance(target, (list, tuple)):
        views = list(target)
    else:
        views = strip_to_views(target, view_res)
    insts = gl.build_glyphs(text512, resize_views(views, VIEW_RES), step, rng, cfg)
    return scale_instances(insts, text512, views, view_res, cfg)


def front_view_image(path, base=None, res=VIEW_RES):
    """The photo registered into view 0 (anchors.py --front-image) -> uint8 (res, res, 3)."""
    p = path if (base is None or os.path.isabs(path)) else os.path.join(base, path)
    im = Image.open(p).convert("RGB")
    if im.size != (res, res):
        im = im.resize((res, res), BOX if im.width > res else LANCZOS)
    return np.asarray(im)


def build_infer_glyphs(text, cfg=None, view_res=VIEW_RES, views=None):
    """Inference glyph instances at view_res (views: raw-ordered images at view_res for kind gt).

    Without views, kind gt crops view 0 from the text.json's "front_image" (the photo registered
    into the front view, path relative to the text.json). Gt instances on other views become box.
    """
    cfg = cfg or gl.GlyphConfig()
    text = load_text(text)
    if text is None or not text.get("items"):
        return []
    if views is None and cfg.infer_kind == "gt" and text.get("front_image"):
        views = [front_view_image(text["front_image"], text.get("_dir"), view_res)] + [None] * 5
    text512 = rescale_text(text, VIEW_RES)
    insts = gl.build_glyphs_infer(text512, cfg, resize_views(views, VIEW_RES) if views else None)
    return scale_instances(insts, text512, views, view_res, cfg)


# ────────────────────────────────────────────────────────────────────────────
# VAE encode + pack
# ────────────────────────────────────────────────────────────────────────────

def pack_latents(latents):
    """[B, C, H, W] -> [B, (H/2)(W/2), 4C]: UniTEX-FLUX _pack_latents(pixel_shuffle=True)."""
    b, c, h, w = latents.shape
    x = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def patch_tensor(patch):
    """PIL RGB patch -> float32 [3, H, W] in [0, 1]."""
    a = np.asarray(patch.convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def encode_patches(patches, vae, sample_mode="sample", generator=None, device=None, max_batch=16):
    """PIL patches -> list of latents [1, C, H/8, W/8] (float, VAE dtype), same order as input.

    Patches of equal size share one VAE call (at most max_batch at a time).
    """
    device = device or next(vae.parameters()).device
    dtype = vae.dtype
    shift = vae.config.shift_factor or 0.0
    scale = vae.config.scaling_factor
    groups = {}
    for k, p in enumerate(patches):
        groups.setdefault(p.size, []).append(k)
    out = [None] * len(patches)
    for size in sorted(groups):
        idx = groups[size]
        for s in range(0, len(idx), max_batch):
            chunk = idx[s:s + max_batch]
            x = torch.stack([patch_tensor(patches[k]) for k in chunk]).to(device=device, dtype=dtype)
            with torch.no_grad():
                dist = vae.encode(x.mul(2.0).sub(1.0)).latent_dist
                z = dist.sample(generator) if sample_mode == "sample" else dist.mode()
                z = (z - shift) * scale
            for j, k in enumerate(chunk):
                out[k] = z[j:j + 1]
    return out


def encode_glyphs(instances, vae, sample_mode="sample", generator=None, device=None,
                  dtype=None, max_batch=16):
    """GlyphInstances -> (glyph_latents [1, Ng, 4C] in dtype, glyph_ids [Ng, 3] float32).

    Token k of instance i is packed patch token k (row-major), kept only where keep[k]. Ids come
    from glyph_ids.concat_ids in the same order. Ng == 0 gives [1, 0, 4C] and [0, 3].
    """
    device = device or next(vae.parameters()).device
    dtype = dtype or vae.dtype
    C = int(vae.config.latent_channels) * 4
    if not instances:
        return (torch.zeros(1, 0, C, device=device, dtype=dtype),
                torch.zeros(0, 3, device=device, dtype=torch.float32))
    lat = encode_patches([i.patch for i in instances], vae, sample_mode, generator, device, max_batch)
    toks = []
    for inst, z in zip(instances, lat):
        t = pack_latents(z)
        gh, gw = inst.token_hw
        if t.shape[1] != gh * gw:
            raise ValueError(f"patch {inst.patch.size} packs to {t.shape[1]} tokens, token_hw {inst.token_hw}")
        keep = torch.as_tensor(np.asarray(inst.keep, bool), device=t.device)
        toks.append(t[:, keep])
    ids, _ = gid.concat_ids(instances)
    if not np.isfinite(ids).all():
        raise ValueError("glyph ids contain NaN after dropping keep == False tokens")
    latents = torch.cat(toks, dim=1).to(dtype=dtype)
    return latents, torch.from_numpy(ids).to(device=device, dtype=torch.float32)


# ────────────────────────────────────────────────────────────────────────────
# Target tokens inside text boxes (keep-mask boost)
# ────────────────────────────────────────────────────────────────────────────

def text_token_mask(text, view_res=VIEW_RES, min_height_px=0.0, pad_tokens=0):
    """Bool [(R/16) * (6R/16)] over the target strip tokens (row-major like the packed target):
    True where the token's 16 x 16 px cell overlaps some item's view bbox, grown by pad_tokens
    tokens and kept inside the item's view slot."""
    R = int(view_res)
    nr, nc = R // TOKEN_PX, 6 * R // TOKEN_PX
    m = np.zeros((nr, nc), bool)
    text = load_text(text)
    if text is not None:
        text = rescale_text(text, R)
        for it in text.get("items", []):
            if not str(it.get("text") or "").strip():
                continue
            for k, v in (it.get("views") or {}).items():
                if not (v.get("bbox") or v.get("quad")):
                    continue
                if gl._view_height(v) < min_height_px:
                    continue
                x0, y0, x1, y1 = gl._view_bbox(v)
                off = FULL_INDEX.index(int(k)) * R
                p = pad_tokens * TOKEN_PX
                # token (r, c) covers strip pixels [16c, 16c + 16) x [16r, 16r + 16)
                c0 = max(int(math.floor((x0 + off - p) / TOKEN_PX)), off // TOKEN_PX)
                c1 = min(int(math.ceil((x1 + off + p) / TOKEN_PX)) - 1, (off + R) // TOKEN_PX - 1)
                r0 = max(int(math.floor((y0 - p) / TOKEN_PX)), 0)
                r1 = min(int(math.ceil((y1 + p) / TOKEN_PX)) - 1, nr - 1)
                if c1 >= c0 and r1 >= r0:
                    m[r0:r1 + 1, c0:c1 + 1] = True
    return torch.from_numpy(m.reshape(-1))


def boosted_keep_prob(keep_p, text_mask, boost):
    """Per-token keep probability: keep_p outside text, keep_p + boost * (1 - keep_p) inside."""
    p = torch.full(text_mask.shape, float(keep_p), dtype=torch.float32)
    p[text_mask] = float(keep_p) + float(boost) * (1.0 - float(keep_p))
    return p


# ────────────────────────────────────────────────────────────────────────────
# GlyphConfig from JSON / overrides
# ────────────────────────────────────────────────────────────────────────────

def _parse_value(s):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


def _inf(v):
    return math.inf if v is None or (isinstance(v, str) and v.lower() in ("inf", "infinity")) else float(v)


def glyph_config(spec=None, overrides=()):
    """GlyphConfig from a JSON file path or inline JSON object, then "key=value" overrides.

    Values are parsed as JSON when possible ("24", "true", "[0.8, 1.25]"). In stages, an
    until_step of null or "inf" means infinity.
    """
    d = {}
    if spec:
        if str(spec).lstrip().startswith("{"):
            d = json.loads(spec)
        else:
            with open(spec) as f:
                d = json.load(f)
    for o in overrides or ():
        if "=" not in o:
            raise ValueError(f"glyph override {o!r} is not key=value")
        k, v = o.split("=", 1)
        d[k.strip()] = _parse_value(v.strip())
    known = {f.name for f in fields(gl.GlyphConfig)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown GlyphConfig fields: {sorted(unknown)}")
    if "stages" in d:
        d["stages"] = [(_inf(s[0]), float(s[1]), float(s[2]), float(s[3])) for s in d["stages"]]
    for k in ("drop_flags", "aug_scale"):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return gl.GlyphConfig(**d)


def glyph_config_dict(cfg):
    """JSON-safe dict of a GlyphConfig (infinity written as "inf")."""
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if f.name == "stages":
            v = [["inf" if math.isinf(s[0]) else s[0], *s[1:]] for s in v]
        elif isinstance(v, tuple):
            v = list(v)
        out[f.name] = v
    return out
