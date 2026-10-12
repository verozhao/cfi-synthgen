"""
Glyph layout: every text line drawn at its exact place and size in each view of the target strip,
dark ink on white, VAE-encoded and given to the texture LoRA on extra input channels of the target
tokens (the way FLUX.1 Canny / Depth take their control image), instead of glyph tokens appended to
the sequence.

Why: appended glyph tokens reach the target only through attention, placed by their position ids.
In training those ids were jittered by up to 2 tokens at 1024 px per view, and fixed renders are 2 to
5 times the height of small text, so a glyph patch almost never lies on its letters at their size.
The texture LoRA learned to ignore them (positions, reference blur and text loss tests). Here target
token (r, c) gets the latent of exactly the layout pixels it covers, through a projection that starts
at zero, so the warm-started LoRA behaves as before until training gives the projection weight.

Ink per line (`kind`):
  font   the line's OCR text drawn into its view quad, stretched to fill it (any view)
  edges  Canny edges of the view's own pixels inside the quad, drawn dark on white. Training: the
         target view. Inference: the photo registered into the front view (text.json "front_image"),
         so edges exist on the front view only and other views fall back to font.

API (numpy / torch / PIL / cv2, no UniTEX imports, the VAE is any diffusers AutoencoderKL):
  layout_config(spec, overrides)                 LayoutConfig from JSON + key=value overrides
  layout_lines(text, view_res, cfg, rng)         filtered lines (item, raw view, quad at view_res, kind)
  layout_strip(text, view_res, cfg, rng, views)  uint8 (R, 6R, 3) strip + info (lines drawn, drawn text.json)
  encode_layout(strip, vae)                      [1, (R/16)(6R/16), 4C] = packed latent - packed blank latent
  GlyphLayoutEmbedder, wrap_x_embedder           x_embedder + zero-initialised projection of the extra channels
  save_layout_proj, load_layout_proj             LAYOUT_PROJ_FILE next to pytorch_lora_weights.safetensors
"""

import json
import math
import os
from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from unitex import glyph as gl
    from unitex import glyph_tokens as gt
    from unitex.common import FULL_INDEX, TOKEN_PX, VIEW_RES
except ImportError:   # imported with unitex/ itself on sys.path
    import glyph as gl
    import glyph_tokens as gt
    from common import FULL_INDEX, TOKEN_PX, VIEW_RES

LAYOUT_PROJ_FILE = "glyph_layout_proj.safetensors"
PROJ_KEY = "layout_proj.weight"
KINDS = ("font", "edges")
WHITE = 255
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


@dataclass
class LayoutConfig:
    """Which text.json lines are drawn and how. Heights are px at 512 per view (scaled with the
    resolution). The drop probabilities and p_edges apply in training only (an rng is given)."""
    use_generated: bool = True       # lines on generated (non-photo) text too: many more training lines
    min_height_px: float = 6.0       # px at 512, i.e. 12 px at 1024
    min_coverage: float = 0.6
    min_cos: float = 0.3
    min_conf: float = 0.0
    min_view_ocr_ned: float = 0.7    # the view OCR has to agree with the line's text (text_regions.py)
    drop_flags: tuple = ("template_leak", "low_conf")
    p_item_drop: float = 0.15        # a line missing from the layout: paint it from the reference
    p_all_drop: float = 0.1          # no layout at all: the model keeps working without one
    p_edges: float = 0.3             # training: share of lines drawn as edges instead of font
    infer_kind: str = "font"         # inference: font, or edges (front view, registered photo)
    pad_frac: float = 0.06           # ink margin inside the quad, as a fraction of its height
    font_path: str = None
    edge_low: int = 60               # Canny thresholds on the 8-bit gray view
    edge_high: int = 160


def layout_config(spec=None, overrides=()):
    """LayoutConfig from a JSON file path or inline JSON object, then "key=value" overrides."""
    d = {}
    if spec:
        s = str(spec)
        d = json.loads(s) if s.lstrip().startswith("{") else json.load(open(s))
    for o in overrides or ():
        if "=" not in o:
            raise ValueError(f"layout override {o!r} is not key=value")
        k, v = o.split("=", 1)
        d[k.strip()] = gt._parse_value(v.strip())
    known = {f.name for f in fields(LayoutConfig)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown LayoutConfig fields: {sorted(unknown)}")
    if "drop_flags" in d:
        d["drop_flags"] = tuple(d["drop_flags"])
    cfg = LayoutConfig(**d)
    if cfg.infer_kind not in KINDS:
        raise ValueError(f"infer_kind must be one of {KINDS}, got {cfg.infer_kind!r}")
    for k in ("p_item_drop", "p_all_drop", "p_edges"):
        if not 0.0 <= float(getattr(cfg, k)) <= 1.0:
            raise ValueError(f"{k} must be a probability, got {getattr(cfg, k)}")
    return cfg


def layout_config_dict(cfg):
    d = asdict(cfg)
    d["drop_flags"] = list(d["drop_flags"])
    return d


# ────────────────────────────────────────────────────────────────────────────
# Lines
# ────────────────────────────────────────────────────────────────────────────

def _num(d, key, default):
    v = d.get(key)
    return float(default) if v is None else float(v)


def layout_lines(text, view_res, cfg=None, rng=None):
    """text.json (anything glyph_tokens.load_text accepts) -> list of line dicts at view_res:
    {item_id, raw_view, text, quad (4 [x, y], reading order TL TR BR BL), height, kind}.

    rng (numpy Generator) turns on the training draws: p_all_drop, p_item_drop per item (all its
    views), and edges with probability p_edges per item. Without rng every line passing the filters
    is drawn with cfg.infer_kind.
    """
    cfg = cfg or LayoutConfig()
    text = gt.load_text(text)
    if text is None or not text.get("items"):
        return []
    R = int(view_res)
    tR = gt.rescale_text(text, R)
    if rng is not None and rng.random() < cfg.p_all_drop:
        return []
    min_h = float(cfg.min_height_px) * R / VIEW_RES
    out = []
    for k, it in enumerate(tR.get("items", [])):
        s = " ".join(str(it.get("text") or "").split())
        # one draw each per item, whatever the filters say, so a line's fate does not depend on others
        drop = rng is not None and rng.random() < cfg.p_item_drop
        edges = rng is not None and rng.random() < cfg.p_edges
        if not s or drop:
            continue
        if set(it.get("flags") or ()) & set(cfg.drop_flags):
            continue
        if it.get("provenance") == "generated" and not cfg.use_generated:
            continue
        if _num(it, "conf", 1.0) < cfg.min_conf:
            continue
        kind = ("edges" if edges else "font") if rng is not None else cfg.infer_kind
        for rv, v in sorted((it.get("views") or {}).items(), key=lambda kv: int(kv[0])):
            if not (v.get("bbox") or v.get("quad")):
                continue
            h = gl._view_height(v)
            if h < min_h or _num(v, "coverage", 1.0) < cfg.min_coverage or _num(v, "cos", 1.0) < cfg.min_cos:
                continue
            if _num(v, "view_ocr_ned", 1.0) < cfg.min_view_ocr_ned:
                continue
            out.append({"item_id": int(it.get("id", k)), "raw_view": int(rv), "text": s,
                        "quad": [[float(p[0]), float(p[1])] for p in gl._view_quad(v)],
                        "height": float(h), "kind": kind})
    return out


# ────────────────────────────────────────────────────────────────────────────
# Drawing
# ────────────────────────────────────────────────────────────────────────────

def _quad_size(q):
    tl, tr, br, bl = q
    return (float(np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2,
            float(np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)


def _bbox(q, shape, margin=2):
    H, W = shape
    x0 = max(int(math.floor(q[:, 0].min())) - margin, 0)
    y0 = max(int(math.floor(q[:, 1].min())) - margin, 0)
    x1 = min(int(math.ceil(q[:, 0].max())) + margin, W)
    y1 = min(int(math.ceil(q[:, 1].max())) + margin, H)
    return x0, y0, x1, y1


def line_patch(text, w, h, pad_frac=0.06, font_path=None):
    """Gray uint8 (h, w): the text's ink stretched to fill the patch minus a margin, black on white
    (like a rectified crop of the line, so letters land where the line's letters are)."""
    w, h = max(int(round(w)), 1), max(int(round(h)), 1)
    m = int(round(pad_frac * min(w, h)))
    aw, ah = max(w - 2 * m, 1), max(h - 2 * m, 1)
    img = Image.new("L", (w, h), WHITE)
    big = gl.load_font(int(min(max(8, 4 * ah), 160)), font_path)
    l, t, r, b = big.getbbox(text)
    if r <= l or b <= t:
        return np.asarray(img)
    ink = Image.new("L", (r - l + 2, b - t + 2), WHITE)
    ImageDraw.Draw(ink).text((1 - l, 1 - t), text, fill=0, font=big)
    img.paste(ink.resize((aw, ah), _LANCZOS), (m, m))
    return np.asarray(img)


def _src_to_quad(patch_hw, q, x0, y0):
    """Perspective matrix (cv2 pixel-centre coordinates) taking the patch rectangle onto quad q
    (continuous coordinates, pixel centres at +0.5) inside the canvas crop starting at (x0, y0)."""
    ph, pw = patch_hw
    src = np.float32([[-0.5, -0.5], [pw - 0.5, -0.5], [pw - 0.5, ph - 0.5], [-0.5, ph - 0.5]])
    dst = np.float32(q - np.array([x0 + 0.5, y0 + 0.5]))
    return cv2.getPerspectiveTransform(src, dst)


def draw_font(canvas, text, quad, cfg=None):
    """Draw text into quad on a gray uint8 canvas (darkest wins). False when the quad is degenerate."""
    cfg = cfg or LayoutConfig()
    q = np.asarray(quad, np.float64)
    L, H = _quad_size(q)
    if L < 2 or H < 2:
        return False
    patch = line_patch(text, L, H, cfg.pad_frac, cfg.font_path)
    x0, y0, x1, y1 = _bbox(q, canvas.shape)
    if x1 <= x0 or y1 <= y0:
        return False
    M = _src_to_quad(patch.shape, q, x0, y0)
    warped = cv2.warpPerspective(patch, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=WHITE)
    np.minimum(canvas[y0:y1, x0:x1], warped, out=canvas[y0:y1, x0:x1])
    return True


def quad_mask(quad, shape):
    """Bool mask of the pixels whose centres lie in the quad."""
    m = np.zeros(shape, np.uint8)
    pts = np.round((np.asarray(quad, np.float64) - 0.5) * 16).astype(np.int32)   # 4 fractional bits
    cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def draw_edges(canvas, view, quad, cfg=None):
    """Canny edges of view (uint8 RGB / gray at the canvas size) inside quad, drawn black."""
    cfg = cfg or LayoutConfig()
    q = np.asarray(quad, np.float64)
    x0, y0, x1, y1 = _bbox(q, canvas.shape, margin=4)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return False
    v = np.asarray(view)
    if v.ndim == 3:
        v = cv2.cvtColor(np.ascontiguousarray(v[..., :3]), cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(np.ascontiguousarray(v[y0:y1, x0:x1]), cfg.edge_low, cfg.edge_high, L2gradient=True) > 0
    m = quad_mask(q, canvas.shape)[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1][e & m] = 0
    return True


def layout_strip(text, view_res, cfg=None, rng=None, views=None):
    """(uint8 (R, 6R, 3) strip in FULL_INDEX slot order, info) for one asset.

    views: 6 raw-ordered images at view_res (None entries allowed) for kind edges, e.g.
    glyph_tokens.strip_to_views(training_image) in training or [registered photo] + [None] * 5 at
    inference. An edges line on a view without an image is drawn as font.
    info: {"lines": [...], "n_lines", "kinds": {kind: n}, "per_view": {raw view: n},
           "drawn_text": text.json dict at view_res with only the drawn line views (text loss weight)}.
    """
    cfg = cfg or LayoutConfig()
    R = int(view_res)
    canv = [np.full((R, R), WHITE, np.uint8) for _ in range(6)]
    lines = layout_lines(text, R, cfg, rng)
    drawn, kinds, per_view, items = [], {}, {}, {}
    for ln in lines:
        img = views[ln["raw_view"]] if views is not None and ln["raw_view"] < len(views) else None
        kind = ln["kind"] if (ln["kind"] == "font" or img is not None) else "font"
        if img is not None and np.asarray(img).shape[:2] != (R, R):
            raise ValueError(f"view {ln['raw_view']} is {np.asarray(img).shape[:2]}, expected ({R}, {R})")
        ok = (draw_edges(canv[ln["raw_view"]], img, ln["quad"], cfg) if kind == "edges"
              else draw_font(canv[ln["raw_view"]], ln["text"], ln["quad"], cfg))
        if not ok:
            continue
        drawn.append({**ln, "kind": kind})
        kinds[kind] = kinds.get(kind, 0) + 1
        per_view[str(ln["raw_view"])] = per_view.get(str(ln["raw_view"]), 0) + 1
        it = items.setdefault(ln["item_id"], {"id": ln["item_id"], "text": ln["text"], "views": {}})
        it["views"][str(ln["raw_view"])] = {"quad": ln["quad"], "height_px": ln["height"]}
    strip = np.concatenate([canv[FULL_INDEX[s]] for s in range(6)], axis=1)
    info = {"view_res": R, "lines": drawn, "n_lines": len(drawn), "kinds": kinds, "per_view": per_view,
            "drawn_text": {"version": 1, "res": R, "items": list(items.values())}}
    return np.repeat(strip[..., None], 3, axis=2), info


# ────────────────────────────────────────────────────────────────────────────
# VAE encode
# ────────────────────────────────────────────────────────────────────────────

_BLANK = {}


def _encode(x, vae):
    shift = vae.config.shift_factor or 0.0
    z = vae.encode(x.mul(2.0).sub(1.0)).latent_dist.mode()
    return (z - shift) * vae.config.scaling_factor


@torch.no_grad()
def encode_layout(strip, vae, device=None, dtype=None):
    """uint8 (R, 6R, 3) strip -> [1, (R/16)(6R/16), 4C] packed latent minus the packed latent of a
    white strip (posterior mode), so text-free tokens are about zero. An all-white strip gives exact
    zeros without running the VAE."""
    device = device or next(vae.parameters()).device
    dtype = dtype or vae.dtype
    a = np.asarray(strip)
    H, W = a.shape[:2]
    C = int(vae.config.latent_channels) * 4
    n = (H // TOKEN_PX) * (W // TOKEN_PX)
    if (a == WHITE).all():
        return torch.zeros(1, n, C, device=device, dtype=dtype)
    vd = next(vae.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(a[..., :3].transpose(2, 0, 1))).float().div(255.0)[None]
    x = x.to(device=vd, dtype=vae.dtype)
    key = (H, W, str(vd), str(vae.dtype), id(vae))
    if key not in _BLANK:
        _BLANK.clear()
        _BLANK[key] = gt.pack_latents(_encode(torch.ones_like(x), vae))
    z = gt.pack_latents(_encode(x, vae)) - _BLANK[key]
    if z.shape[1] != n:
        raise ValueError(f"layout strip {a.shape[:2]} packs to {z.shape[1]} tokens, expected {n}")
    return z.to(device=device, dtype=dtype)


def layout_tokens_full(layout, n_total, keep_index=None):
    """Layout channels for the whole transformer sequence: the target tokens' layout (optionally
    only the kept target tokens, keep_index) followed by zeros for every other token."""
    lay = layout if keep_index is None else layout[:, keep_index]
    pad = n_total - lay.shape[1]
    if pad < 0:
        raise ValueError(f"layout has {lay.shape[1]} tokens, the sequence only {n_total}")
    return torch.cat([lay, lay.new_zeros(lay.shape[0], pad, lay.shape[2])], dim=1)


# ────────────────────────────────────────────────────────────────────────────
# x_embedder with layout channels
# ────────────────────────────────────────────────────────────────────────────

class GlyphLayoutEmbedder(torch.nn.Module):
    """FLUX x_embedder plus a bias-free projection of `channels` extra input channels.

    Input [..., channels] goes through the original embedder only. Input [..., 2 * channels] is
    base(x[..., :channels]) + layout_proj(x[..., channels:]). layout_proj starts at zero, so a
    wrapped model gives exactly the old output until it is trained, and zero layout channels always
    do (no bias)."""

    def __init__(self, base, channels, device=None, dtype=torch.float32):
        super().__init__()
        self.base = base
        self.channels = int(channels)
        out = int(getattr(base, "out_features"))
        if device is None:
            device = next(base.parameters()).device
        self.layout_proj = torch.nn.Linear(self.channels, out, bias=False, device=device, dtype=dtype)
        torch.nn.init.zeros_(self.layout_proj.weight)

    def forward(self, x):
        c = self.channels
        if x.shape[-1] == c:
            return self.base(x)
        if x.shape[-1] != 2 * c:
            raise ValueError(f"x_embedder input has {x.shape[-1]} channels, expected {c} or {2 * c}")
        h = self.base(x[..., :c].contiguous())
        p = self.layout_proj(x[..., c:].contiguous().to(self.layout_proj.weight.dtype))
        return h + p.to(h.dtype)


def wrap_x_embedder(transformer, weight=None, device=None, dtype=torch.float32):
    """Put a GlyphLayoutEmbedder around transformer.x_embedder (once) and optionally load its
    projection weight. Returns the embedder."""
    xe = transformer.x_embedder
    if not isinstance(xe, GlyphLayoutEmbedder):
        xe = GlyphLayoutEmbedder(xe, int(transformer.config.in_channels), device=device, dtype=dtype)
        transformer.x_embedder = xe
    if weight is not None:
        w = torch.as_tensor(weight)
        if tuple(w.shape) != tuple(xe.layout_proj.weight.shape):
            raise ValueError(f"layout projection {tuple(w.shape)} does not fit {tuple(xe.layout_proj.weight.shape)}")
        with torch.no_grad():
            xe.layout_proj.weight.copy_(w.to(xe.layout_proj.weight))
    return xe


def layout_proj_path(path):
    """The projection file for a LoRA file or checkpoint directory."""
    p = str(path)
    d = p if os.path.isdir(p) or not p.endswith(".safetensors") else os.path.dirname(p)
    return os.path.join(d, LAYOUT_PROJ_FILE)


def save_layout_proj(embedder, save_dir):
    from safetensors.torch import save_file
    os.makedirs(save_dir, exist_ok=True)
    w = embedder.layout_proj.weight.detach().float().cpu().contiguous()
    path = os.path.join(save_dir, LAYOUT_PROJ_FILE)
    save_file({PROJ_KEY: w}, path)
    return path


def load_layout_proj(path):
    """The projection weight (float32 [out, channels]) for a LoRA file, checkpoint dir or the file itself."""
    from safetensors.torch import load_file
    p = str(path)
    f = p if os.path.basename(p) == LAYOUT_PROJ_FILE else layout_proj_path(p)
    if not os.path.isfile(f):
        raise FileNotFoundError(f"no {LAYOUT_PROJ_FILE} for {path} (a LoRA trained without --glyph_layout?)")
    return load_file(f)[PROJ_KEY]
