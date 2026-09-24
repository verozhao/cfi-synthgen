"""
GlyphAnchor glyph conditions for UniTEX-FLUX: per-view text items -> glyph patches + position ids

Steps (training, build_glyphs):
  1. expand text.json items x visible views into instances, filter by text size, visibility,
     flags and provenance
  2. per item: dropout, then one glyph source from the staged schedule (gt crop of the target
     view, box-size render, fixed-size render); whole-condition dropout per sample
  3. rank (front view first, then larger text, then more frontal) and keep greedily under a
     glyph-token budget
  4. build each patch (black on white, sides multiples of 16 px), augment rendered patches, anchor
     its tokens on the target strip (glyph_ids), jitter the ids by +-1 token
Inference (build_glyphs_infer) uses one kind (fixed-size renders by default), no dropout,
augmentation or jitter.

Each patch is VAE-encoded and packed like a UniTEX condition image (16 px per token, row-major),
so GlyphInstance.ids[k] belongs to packed token k of GlyphInstance.patch. Drop the tokens with
keep == False (warp mode only) from both before concatenating after the other conditions.

Patch orientation: center / stretch patches are in view orientation (a word reading downwards
in the view is rendered rotated, snapped to multiples of 90 degrees), because their tokens sit in
the view's axis-aligned box. warp patches stay in the item's reading frame, since the grid does
the rotation and bending.
"""

import functools
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from unitex.common import TOKEN_PX, VIEW_RES
    from unitex import glyph_ids as gid
except ImportError:   # imported with unitex/ itself on sys.path
    from common import TOKEN_PX, VIEW_RES
    import glyph_ids as gid

_R = getattr(Image, "Resampling", Image)
LANCZOS, BICUBIC = _R.LANCZOS, _R.BICUBIC
_T = getattr(Image, "Transpose", Image)
_Q = getattr(Image, "Transform", Image)

KINDS = ("gt", "box", "fixed")
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FONT_ENV = "CFI_GLYPH_FONT"
FONT_CANDIDATES = (
    # Linux (GPU server): Noto first for coverage, then DejaVu
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    # macOS
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def ceil16(v):
    return max(TOKEN_PX, int(math.ceil(v / TOKEN_PX)) * TOKEN_PX)


def round16(v):
    return max(TOKEN_PX, int(math.floor(v / TOKEN_PX + 0.5)) * TOKEN_PX)


# ────────────────────────────────────────────────────────────────────────────
# Fonts
# ────────────────────────────────────────────────────────────────────────────

_logged_fonts = set()


def resolve_font_path(path=None):
    """Explicit path, else $CFI_GLYPH_FONT, else the first existing candidate, else None."""
    for p in (path, os.environ.get(FONT_ENV)):
        if p:
            if os.path.exists(p):
                return p
            if ("missing", p) not in _logged_fonts:   # called per render: warn once, not per sample
                _logged_fonts.add(("missing", p))
                print(f"  [glyph] font {p} not found, falling back")
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


@functools.lru_cache(maxsize=512)
def _load_font_cached(size, path):
    if path is not None:
        return ImageFont.truetype(path, size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)   # PIL searches the system font dirs
    except OSError:
        pass
    try:
        return ImageFont.load_default(size=size)            # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def load_font(size, path=None):
    """TrueType font at `size` px. Order: path, $CFI_GLYPH_FONT, Noto / DejaVu (Linux),
    Arial Unicode / Arial / Helvetica (macOS), PIL default. Logs the choice once per file."""
    resolved = resolve_font_path(path)
    font = _load_font_cached(int(max(1, size)), resolved)
    fp = getattr(font, "path", None)
    name = resolved or (fp if isinstance(fp, str) else "PIL default")
    if name not in _logged_fonts:
        _logged_fonts.add(name)
        print(f"  [glyph] font: {name}")
    return font


def _metrics(font):
    try:
        asc, desc = font.getmetrics()
        return asc, desc
    except AttributeError:
        l, t, r, b = font.getbbox("Ag")
        return b, 0


def _length(font, text):
    try:
        return float(font.getlength(text))
    except AttributeError:
        l, t, r, b = font.getbbox(text)
        return float(r - l)


# ────────────────────────────────────────────────────────────────────────────
# Line wrapping
# ────────────────────────────────────────────────────────────────────────────

def _is_cjk(ch):
    o = ord(ch)
    return 0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF or 0x3040 <= o <= 0x30FF


def _wrap_tokens(text):
    """Break units: words, or characters for a single CJK run (no spaces to break at)."""
    words = text.split()
    if len(words) <= 1 and any(_is_cjk(c) for c in text):
        return [c for c in text if not c.isspace()], ""
    return words, " "


def _wrap_balanced(widths, sep_w, n):
    """Split tokens into n contiguous lines minimising the widest line (DP). Returns bounds."""
    m = len(widths)
    n = max(1, min(n, m))
    cum = np.concatenate([[0.0], np.cumsum(widths)])
    dp = np.full((n + 1, m + 1), np.inf)
    arg = np.zeros((n + 1, m + 1), int)
    dp[0, 0] = 0.0
    for k in range(1, n + 1):
        for j in range(k, m + 1):
            i = np.arange(k - 1, j)
            lw = cum[j] - cum[i] + (j - i - 1) * sep_w
            v = np.maximum(dp[k - 1, i], lw)
            a = int(np.argmin(v))
            dp[k, j], arg[k, j] = v[a], i[a]
    bounds, j = [], m
    for k in range(n, 0, -1):
        i = arg[k, j]
        bounds.append((i, j))
        j = i
    return bounds[::-1]


def wrap_lines(text, font, n):
    toks, sep = _wrap_tokens(text)
    if not toks:
        return [""]
    widths = [_length(font, t) for t in toks]
    return [sep.join(toks[i:j]) for i, j in _wrap_balanced(widths, _length(font, sep), n)]


def max_lines_for(text, max_lines=8):
    toks, _ = _wrap_tokens(text)
    return max(1, min(max_lines, len(toks)))


# ────────────────────────────────────────────────────────────────────────────
# Rendering: fixed size (inference type) and box size
# ────────────────────────────────────────────────────────────────────────────

def fixed_layout(text, box_wh_px, font_px=24, line_px=32, margin_px=8, lines=None,
                 max_lines=8, max_wh=None, font_path=None):
    """Line split and patch size of render_fixed without drawing. Returns (rows, (W, H)).

    lines=None picks the line count whose text block aspect best matches the box aspect,
    preferring counts whose patch fits inside max_wh; an int forces that many lines.
    """
    font = load_font(font_px, font_path)
    nmax = max_lines_for(text, max_lines)
    cands = [int(min(max(lines, 1), nmax))] if lines is not None else range(1, nmax + 1)
    bw, bh = max(float(box_wh_px[0]), 1.0), max(float(box_wh_px[1]), 1.0)
    target = math.log(bw / bh)
    best = None
    for n in cands:
        rows = wrap_lines(text, font, n)
        w = max(_length(font, r) for r in rows)
        size = (ceil16(w + 2 * margin_px), ceil16(len(rows) * line_px + 2 * margin_px))
        fits = max_wh is None or (size[0] <= max_wh[0] and size[1] <= max_wh[1])
        score = (0 if fits else 1, abs(math.log(max(w, 1.0) / (len(rows) * line_px)) - target))
        if best is None or score < best[0]:
            best = (score, rows, size)
    return best[1], best[2]


def render_fixed(text, box_wh_px, font_px=24, line_px=32, margin_px=8, lines=None,
                 max_lines=8, max_wh=None, font_path=None):
    """GlyphAnchor inference-type patch: black on white, fixed font size and line height.

    box_wh_px is the target box in the text's reading frame; it only chooses the line count.
    Size is padded up to multiples of 16 px. `lines` overrides the wrap (random line breaks).
    """
    rows, (W, H) = fixed_layout(text, box_wh_px, font_px, line_px, margin_px, lines,
                                max_lines, max_wh, font_path)
    font = load_font(font_px, font_path)
    asc, desc = _metrics(font)
    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)
    y0 = (H - len(rows) * line_px) / 2
    for k, row in enumerate(rows):
        x = (W - _length(font, row)) / 2
        y = y0 + k * line_px + (line_px - (asc + desc)) / 2
        draw.text((x, y), row, fill=BLACK, font=font)
    return img


def _ink_font(rows, avail_w, avail_h, font_path, line_gap=1.15):
    """Largest font whose ink (single line) or metric block (multi-line) fits avail_w x avail_h."""
    ref = 64
    f = load_font(ref, font_path)
    if len(rows) == 1:
        l, t, r, b = f.getbbox(rows[0] or " ")
        iw, ih = max(r - l, 1), max(b - t, 1)
    else:
        asc, desc = _metrics(f)
        iw = max(max(_length(f, r) for r in rows), 1)
        ih = max((asc + desc) * (1 + line_gap * (len(rows) - 1)), 1)
    size = max(4, int(ref * min(avail_w / iw, avail_h / ih)))
    for _ in range(8):
        f = load_font(size, font_path)
        if len(rows) == 1:
            l, t, r, b = f.getbbox(rows[0] or " ")
            ok = (r - l) <= avail_w + 0.5 and (b - t) <= avail_h + 0.5
        else:
            ok = True
        if ok or size <= 4:
            break
        size -= 1
    return f


def render_box(text, box_wh_px, single_line=True, fit="contain", pad_frac=0.06, font_path=None):
    """Patch sized like the target box (w, h rounded to multiples of 16, min 16), text scaled to fill.

    box_wh_px is in the text's reading frame. fit="contain" keeps the glyph aspect and centres
    the ink; fit="stretch" scales the ink to the full patch like a rectified gt crop (used in
    warp mode, where the patch spans the item's whole (s, t) frame).
    """
    W, H = round16(box_wh_px[0]), round16(box_wh_px[1])
    m = max(1, int(round(pad_frac * min(W, H))))
    aw, ah = W - 2 * m, H - 2 * m
    img = Image.new("RGB", (W, H), WHITE)
    if not text.strip():
        return img
    if single_line:
        rows = [" ".join(text.split())]
    else:
        # the line count that gives the largest font
        best = None
        for n in range(1, max_lines_for(text) + 1):
            rows_n = wrap_lines(text, load_font(64, font_path), n)
            size = _ink_font(rows_n, aw, ah, font_path).size
            if best is None or size > best[0]:
                best = (size, rows_n)
        rows = best[1]
    if fit == "stretch" and len(rows) == 1:
        big = load_font(max(8, 4 * ah), font_path)
        l, t, r, b = big.getbbox(rows[0])
        tmp = Image.new("RGB", (max(r - l, 1) + 2, max(b - t, 1) + 2), WHITE)
        ImageDraw.Draw(tmp).text((1 - l, 1 - t), rows[0], fill=BLACK, font=big)
        img.paste(tmp.resize((aw, ah), LANCZOS), (m, m))
        return img
    font = _ink_font(rows, aw, ah, font_path)
    draw = ImageDraw.Draw(img)
    if len(rows) == 1:
        l, t, r, b = font.getbbox(rows[0])
        draw.text(((W - (r - l)) / 2 - l, (H - (b - t)) / 2 - t), rows[0], fill=BLACK, font=font)
    else:
        asc, desc = _metrics(font)
        lh = (asc + desc) * 1.15
        y0 = (H - (asc + desc) - lh * (len(rows) - 1)) / 2
        for k, row in enumerate(rows):
            draw.text(((W - _length(font, row)) / 2, y0 + k * lh), row, fill=BLACK, font=font)
    return img


# ────────────────────────────────────────────────────────────────────────────
# Ground-truth crops
# ────────────────────────────────────────────────────────────────────────────

def _on_white(view_img, alpha=None):
    """HxWx3 / HxWx4 uint8 array (float in [0, 1] also accepted) or PIL image -> PIL RGB on white."""
    if isinstance(view_img, Image.Image):
        im = view_img
        a = np.asarray(im.convert("RGBA")) if im.mode in ("RGBA", "LA", "P") else np.asarray(im.convert("RGB"))
    else:
        a = np.asarray(view_img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if alpha is None and a.shape[-1] == 4:
        alpha = a[..., 3]
    if alpha is None and a.dtype == np.uint8:
        return Image.fromarray(np.ascontiguousarray(a[..., :3]), "RGB")
    rgb = a[..., :3].astype(np.float32)
    if a.dtype != np.uint8 and rgb.size and rgb.max() <= 1.0:
        rgb = rgb * 255.0     # trainer tensors are float [0, 1]; without this the crop comes out black
    if alpha is not None:
        al = np.asarray(alpha, np.float32)
        al = al / 255.0 if al.max() > 1.0 else al
        rgb = rgb * al[..., None] + 255.0 * (1 - al[..., None])
    return Image.fromarray(np.clip(rgb + 0.5, 0, 255).astype(np.uint8), "RGB")


def crop_gt(view_img, quad, out_wh, alpha=None):
    """Rectified crop of a quad (view pixels, reading order TL, TR, BR, BL) resized to out_wh.

    Uses PIL Image.transform(QUAD), whose corner order is TL, BL, BR, TR. Pixels where alpha is 0
    come out white. Samples at the quad's native size first, then LANCZOS-resizes, so shrinking a
    large crop does not alias.
    """
    img = _on_white(view_img, alpha)
    tl, tr, br, bl = [np.asarray(p, np.float64) for p in quad]
    ln = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    ht = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    out_wh = (int(out_wh[0]), int(out_wh[1]))
    nat = (max(out_wh[0], int(math.ceil(ln))), max(out_wh[1], int(math.ceil(ht))))
    data = tuple(float(v) for v in (*tl, *bl, *br, *tr))
    crop = img.transform(nat, _Q.QUAD, data, resample=BICUBIC, fillcolor=WHITE)
    return crop if nat == out_wh else crop.resize(out_wh, LANCZOS)


def crop_gt_grid(view_img, grid, out_wh, alpha=None, supersample=None):
    """gt crop sampled through the item's (s, t) grid (warp mode), null cells white.

    Pixel (u, v) of the patch shows reading-frame point ((u+0.5)/W, (v+0.5)/H), the same map the
    warp ids use, so gt pixels and token positions agree on curved or partially visible items
    (a QUAD crop of the visible part would not). supersample=None samples 2x only when the item
    covers more view pixels than the patch has (shrinking), else 1x.
    """
    img = np.asarray(_on_white(view_img, alpha))
    G = gid.parse_grid(grid)
    if supersample is None:
        length, height = gid.grid_extent(G)
        shrink = max(length / out_wh[0] if np.isfinite(length) else 1.0,
                     height / out_wh[1] if np.isfinite(height) else 1.0)
        supersample = 2 if shrink > 1.25 else 1
    W, H = int(out_wh[0]) * supersample, int(out_wh[1]) * supersample
    S, T = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H)
    xy, keep = gid.sample_grid(G, S.ravel(), T.ravel())
    x = np.clip(xy[:, 0] - 0.5, 0, img.shape[1] - 1)
    y = np.clip(xy[:, 1] - 0.5, 0, img.shape[0] - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, img.shape[1] - 1), np.minimum(y0 + 1, img.shape[0] - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    px = (img[y0, x0] * ((1 - fx) * (1 - fy)) + img[y0, x1] * (fx * (1 - fy))
          + img[y1, x0] * ((1 - fx) * fy) + img[y1, x1] * (fx * fy))
    px[~keep] = 255.0
    out = Image.fromarray(np.clip(px + 0.5, 0, 255).astype(np.uint8).reshape(H, W, 3), "RGB")
    return out if supersample == 1 else out.resize((int(out_wh[0]), int(out_wh[1])), LANCZOS)


# ────────────────────────────────────────────────────────────────────────────
# Augmentation (rendered patches)
# ────────────────────────────────────────────────────────────────────────────

def _pad_to16(img):
    W, H = ceil16(img.width), ceil16(img.height)
    if (W, H) == img.size:
        return img
    out = Image.new("RGB", (W, H), WHITE)
    out.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return out


def _ink_bbox(img, thresh=245):
    g = np.asarray(img.convert("L"))
    ys, xs = np.nonzero(g < thresh)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def augment(patch, rng, cfg):
    """Random resize (x cfg.aug_scale) and rotation (+-cfg.aug_rot_deg on white).

    The output keeps sides at multiples of 16 px. Rotation expands the canvas, trims it to the
    ink plus cfg.aug_margin_px and pads back to 16. Translation jitter is applied to the ids
    (glyph_ids.jitter_ids), not to pixels. Always draws the same number of random numbers.
    """
    u_s, u_r = rng.random(), rng.random()
    scale = math.exp(rng.uniform(math.log(cfg.aug_scale[0]), math.log(cfg.aug_scale[1])))
    angle = rng.uniform(-cfg.aug_rot_deg, cfg.aug_rot_deg)
    out = patch
    if u_s < cfg.p_aug_scale:
        out = out.resize((round16(out.width * scale), round16(out.height * scale)), LANCZOS)
    if u_r < cfg.p_aug_rot and abs(angle) > 0.5:
        rot = out.rotate(angle, resample=BICUBIC, expand=True, fillcolor=WHITE)
        bb = _ink_bbox(rot)
        if bb is not None:
            m = cfg.aug_margin_px
            rot = rot.crop((max(bb[0] - m, 0), max(bb[1] - m, 0),
                            min(bb[2] + m, rot.width), min(bb[3] + m, rot.height)))
        out = _pad_to16(rot)
    return out


def _downscale(patch, f):
    W, H = max(TOKEN_PX, int(patch.width * f) // TOKEN_PX * TOKEN_PX), \
        max(TOKEN_PX, int(patch.height * f) // TOKEN_PX * TOKEN_PX)
    return patch.resize((W, H), LANCZOS)


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class GlyphConfig:
    # staged SFT: (until_step, p_gt, p_box, p_fixed), first stage with step < until_step wins
    stages: list = field(default_factory=lambda: [
        (2000, 0.6, 0.2, 0.2), (6000, 0.3, 0.3, 0.4), (math.inf, 0.05, 0.25, 0.7)])
    p_item_drop: float = 0.15
    p_all_drop: float = 0.1            # whole-condition drop: the model keeps a no-glyph mode
    # instance filters (per item x view)
    min_height_px: float = 8.0
    min_coverage: float = 0.6
    min_cos: float = 0.3
    min_conf: float = 0.0
    drop_flags: tuple = ("template_leak", "low_conf")
    use_generated: bool = False
    token_budget: int = 1536           # 1.5 views' worth of tokens on top of the 7168 conditions
    max_views_per_item: int = 3
    # anchoring
    anchor_mode: str = "center"        # center | stretch | warp
    quantize_ids: bool = True          # center mode only; stretch / warp ids stay continuous
    frame: int = 1                     # axis0 of the glyph plane
    mask_null: bool = True             # warp: drop tokens on invisible grid cells
    nudge_collisions: bool = True
    max_nudge: int = 2
    warn_collisions: bool = True
    # rendering
    infer_kind: str = "fixed"
    font_path: Optional[str] = None
    fixed_font_px: int = 24
    fixed_line_px: int = 32
    fixed_margin_px: int = 8
    max_lines: int = 8
    max_warp_tokens_w: int = 64
    # augmentation (rendered kinds; gt too if augment_gt)
    augment_gt: bool = False
    p_aug_scale: float = 0.5
    aug_scale: tuple = (0.8, 1.25)
    p_aug_rot: float = 0.5
    aug_rot_deg: float = 10.0
    aug_margin_px: int = 4
    p_rewrap: float = 0.2              # random line count for fixed renders (center / stretch)
    jitter_tokens: int = 1

    def __post_init__(self):
        if self.anchor_mode not in gid.ANCHOR_MODES:
            raise ValueError(f"anchor_mode must be one of {gid.ANCHOR_MODES}, got {self.anchor_mode}")
        if self.infer_kind not in KINDS:
            raise ValueError(f"infer_kind must be one of {KINDS}, got {self.infer_kind}")
        # axis0 = 0 would give glyph tokens the target's own position ids
        if self.frame < 1:
            raise ValueError(f"frame must be >= 1 (0 is the target plane), got {self.frame}")


def stage_probs(cfg, step):
    """(p_gt, p_box, p_fixed) for a training step, normalised."""
    stages = sorted(cfg.stages, key=lambda s: s[0])
    row = next((s for s in stages if step < s[0]), stages[-1])
    p = np.asarray(row[1:4], np.float64)
    return tuple(p / p.sum())


# ────────────────────────────────────────────────────────────────────────────
# Instances: expand, filter, rank, budget
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    item: dict
    item_id: int
    raw_view: int
    view: dict
    kind: str = "fixed"
    lines: Optional[int] = None
    token_hw: tuple = (0, 0)
    n_tokens: int = 0

    @property
    def rank_key(self):
        return (0 if self.raw_view == 0 else 1, -_view_height(self.view),
                -_num(self.view, "cos", 1.0), self.item_id, self.raw_view)


@dataclass
class GlyphInstance:
    item_id: int
    raw_view: int
    patch: Image.Image                 # RGB, black text on white, sides multiples of 16
    token_hw: tuple                    # (gh, gw) = (H / 16, W / 16)
    ids: np.ndarray                    # float32 [gh*gw, 3] = (axis0, row, col), row-major
    keep: np.ndarray                   # bool [gh*gw], False = drop this token (warp, invisible)
    kind: str                          # gt | box | fixed
    text: str = ""
    mode: str = "center"
    bbox: tuple = ()
    quad: tuple = ()
    rank: int = 0

    @property
    def n_keep(self):
        return int(self.keep.sum())


def items_of(text_items):
    """text.json dict, its "items" list, or a path -> list of item dicts."""
    if isinstance(text_items, (str, os.PathLike)):
        import json
        with open(text_items) as f:
            text_items = json.load(f)
    if isinstance(text_items, dict) and text_items.get("res", VIEW_RES) != VIEW_RES:
        # view boxes are in pixels, so ids would be silently wrong at another resolution
        raise ValueError(f"text.json res {text_items['res']} != common.VIEW_RES {VIEW_RES}")
    items = text_items.get("items", []) if isinstance(text_items, dict) else (text_items or [])
    # ids key the per-item random draws, so every item needs one
    return [it if "id" in it else {**it, "id": k} for k, it in enumerate(items)]


def _num(d, key, default):
    """d[key] as float; a missing key or an explicit JSON null gives the default."""
    v = d.get(key)
    return float(default) if v is None else float(v)


def _view_quad(view):
    if view.get("quad"):
        return [list(map(float, p)) for p in view["quad"]]
    x0, y0, x1, y1 = [float(v) for v in view["bbox"]]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _view_bbox(view):
    if view.get("bbox"):
        return [float(v) for v in view["bbox"]]
    q = np.asarray(view["quad"], np.float64)
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def _view_height(view):
    if view.get("height_px") is not None:
        return float(view["height_px"])
    tl, tr, br, bl = [np.asarray(p, np.float64) for p in _view_quad(view)]
    return float((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)


def _view_orientation(view):
    """Reading direction in the view snapped to 0 / 90 / 180 / 270 degrees (clockwise, y down)."""
    if not view.get("quad"):
        return 0
    tl, tr, br, bl = [np.asarray(p, np.float64) for p in view["quad"]]
    d = (tr - tl) + (br - bl)
    ang = math.degrees(math.atan2(d[1], d[0]))
    return int(round(ang / 90.0)) % 4 * 90


def _reading_box_wh(view):
    """Axis-aligned box size in the text's reading frame (swapped for 90 / 270)."""
    x0, y0, x1, y1 = _view_bbox(view)
    w, h = x1 - x0, y1 - y0
    return (h, w) if _view_orientation(view) in (90, 270) else (w, h)


def _reading_quad_wh(view):
    """(length, height) of the view quad in the text's reading frame; picks the fixed wrap.

    For bent or tilted lines the axis-aligned bbox is much taller than the text, and its aspect
    would wrap a single OCR line into several.
    """
    if not view.get("quad"):
        return _reading_box_wh(view)
    tl, tr, br, bl = [np.asarray(p, np.float64) for p in view["quad"]]
    return (float(np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2,
            max(_view_height(view), 1.0))


def _orient(img, orient):
    """Rotate an upright reading-frame patch into view orientation."""
    if orient == 90:
        return img.transpose(_T.ROTATE_270)     # clockwise 90: reads downwards
    if orient == 180:
        return img.transpose(_T.ROTATE_180)
    if orient == 270:
        return img.transpose(_T.ROTATE_90)      # counter-clockwise 90: reads upwards
    return img


def warp_geometry(view, cfg):
    """(grid array, gh, gw) for warp mode. gh from height_px, gw from the projected length."""
    G = gid.parse_grid(view["grid"]) if view.get("grid") else gid.grid_from_quad(_view_quad(view))
    length, height = gid.grid_extent(G)
    if not np.isfinite(length):
        tl, tr, br, bl = [np.asarray(p, np.float64) for p in _view_quad(view)]
        length = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = _view_height(view) if view.get("height_px") is not None or not np.isfinite(height) else height
    gh = max(1, int(math.floor(h / TOKEN_PX + 0.5)))
    gw = min(cfg.max_warp_tokens_w, max(1, int(math.floor(length / TOKEN_PX + 0.5))))
    return G, gh, gw


def _fixed_kwargs(cfg):
    return dict(font_px=cfg.fixed_font_px, line_px=cfg.fixed_line_px, margin_px=cfg.fixed_margin_px,
                max_lines=cfg.max_lines, font_path=cfg.font_path)


def estimate_tokens(c, cfg):
    """(token_hw, n_tokens) the candidate's patch will have before augmentation."""
    mode = cfg.anchor_mode
    if mode == "warp":
        G, gh, gw = warp_geometry(c.view, cfg)
        if c.kind == "fixed":
            _, (W, H) = fixed_layout(c.item["text"], (gw * TOKEN_PX, gh * TOKEN_PX), lines=1,
                                     **_fixed_kwargs(cfg))
            gh, gw = H // TOKEN_PX, W // TOKEN_PX
        return (gh, gw), gid.warp_token_count(G, (gh, gw))
    if c.kind in ("gt", "box"):
        x0, y0, x1, y1 = _view_bbox(c.view)
        gh, gw = round16(y1 - y0) // TOKEN_PX, round16(x1 - x0) // TOKEN_PX
    else:
        _, (W, H) = fixed_layout(c.item["text"], _reading_quad_wh(c.view), lines=c.lines,
                                 max_wh=(VIEW_RES, VIEW_RES), **_fixed_kwargs(cfg))
        if _view_orientation(c.view) in (90, 270):
            W, H = H, W
        gh, gw = H // TOKEN_PX, W // TOKEN_PX
    if mode == "center":
        f = gid.downscale_factor((gh, gw))
        if f < 1:
            gh = max(1, int(gh * TOKEN_PX * f) // TOKEN_PX)
            gw = max(1, int(gw * TOKEN_PX * f) // TOKEN_PX)
    return (gh, gw), gh * gw


def expand_candidates(items, cfg):
    """items x visible views -> Candidates passing the item and per-view filters."""
    out = []
    for item in items_of(items):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if set(item.get("flags") or ()) & set(cfg.drop_flags):
            continue
        if item.get("provenance") == "generated" and not cfg.use_generated:
            continue
        if _num(item, "conf", 1.0) < cfg.min_conf:
            continue
        for k, view in (item.get("views") or {}).items():
            if not (view.get("bbox") or view.get("quad")):
                continue
            if _view_height(view) < cfg.min_height_px:
                continue
            if _num(view, "coverage", 1.0) < cfg.min_coverage:
                continue
            if _num(view, "cos", 1.0) < cfg.min_cos:
                continue
            out.append(Candidate(item=item, item_id=int(item["id"]),
                                 raw_view=int(k), view=view, kind=cfg.infer_kind))
    return out


def select_instances(items, cfg, kinds=None, lines=None):
    """Filter, rank and keep instances greedily under cfg.token_budget (deterministic).

    kinds / lines map item id -> glyph kind / forced line count (default cfg.infer_kind, auto).
    Items mapped to kind None are dropped. Rank: front view first, then larger height_px, then
    higher cos, ties by item id and view. At most cfg.max_views_per_item views per item; an
    instance that does not fit is skipped and smaller ones later in the order may still fit.
    """
    kinds, lines = kinds or {}, lines or {}
    cands = []
    for c in expand_candidates(items, cfg):
        c.kind = kinds.get(c.item_id, cfg.infer_kind)
        if c.kind is None:
            continue
        c.lines = lines.get(c.item_id)
        c.token_hw, c.n_tokens = estimate_tokens(c, cfg)
        cands.append(c)
    cands.sort(key=lambda c: c.rank_key)
    kept, used, per_item = [], 0, {}
    for c in cands:
        if per_item.get(c.item_id, 0) >= cfg.max_views_per_item or c.n_tokens <= 0:
            continue
        if used + c.n_tokens > cfg.token_budget:
            continue
        kept.append(c)
        used += c.n_tokens
        per_item[c.item_id] = per_item.get(c.item_id, 0) + 1
    return kept


# ────────────────────────────────────────────────────────────────────────────
# Build
# ────────────────────────────────────────────────────────────────────────────

def _make_patch(c, views, cfg):
    text = " ".join(c.item["text"].split())
    fk = _fixed_kwargs(cfg)
    if cfg.anchor_mode == "warp":
        G, gh, gw = warp_geometry(c.view, cfg)
        wh = (gw * TOKEN_PX, gh * TOKEN_PX)
        if c.kind == "gt":
            return crop_gt_grid(views[c.raw_view], G, wh)
        if c.kind == "box":
            return render_box(text, wh, single_line=True, fit="stretch", font_path=cfg.font_path)
        return render_fixed(text, wh, lines=1, **fk)
    x0, y0, x1, y1 = _view_bbox(c.view)
    if c.kind == "gt":
        box_quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        return crop_gt(views[c.raw_view], box_quad, (round16(x1 - x0), round16(y1 - y0)))
    if c.kind == "box":
        patch = render_box(text, _reading_box_wh(c.view), single_line=True, font_path=cfg.font_path)
    else:
        patch = render_fixed(text, _reading_quad_wh(c.view), lines=c.lines,
                             max_wh=(VIEW_RES, VIEW_RES), **fk)
    return _orient(patch, _view_orientation(c.view))


def _anchor(c, patch, cfg):
    """(patch, ids, keep) with ids for the patch's token grid; downscales oversize center patches."""
    th, tw = patch.height // TOKEN_PX, patch.width // TOKEN_PX
    if cfg.anchor_mode == "center":
        f = gid.downscale_factor((th, tw))
        if f < 1:
            patch = _downscale(patch, f)
            th, tw = patch.height // TOKEN_PX, patch.width // TOKEN_PX
        ids = gid.center_ids(_view_bbox(c.view), c.raw_view, (th, tw), cfg.frame, cfg.quantize_ids)
        keep = np.ones(th * tw, bool)
    elif cfg.anchor_mode == "stretch":
        ids = gid.stretch_ids(_view_bbox(c.view), c.raw_view, (th, tw), cfg.frame)
        keep = np.ones(th * tw, bool)
    else:
        G, _, _ = warp_geometry(c.view, cfg)
        ids, keep = gid.warp_ids(G, c.raw_view, (th, tw), cfg.frame, mask_null=cfg.mask_null)
    return patch, ids, keep


def _finish(instances, cfg):
    """Enforce the budget on the real (post-augmentation) sizes, then separate colliding ids."""
    out, used = [], 0
    for inst in instances:
        if inst.n_keep == 0 or used + inst.n_keep > cfg.token_budget:
            continue
        out.append(inst)
        used += inst.n_keep
    gid.verify_unique(out, nudge=cfg.nudge_collisions, max_shift=cfg.max_nudge,
                      warn=cfg.warn_collisions)
    return out


def _gt_or_box(c, views):
    """A gt instance whose view image is missing (e.g. only the front has a photo) becomes box.

    gt and box patches have the same token size in every mode, so the budget estimate holds.
    """
    if c.kind == "gt" and (views is None or c.raw_view >= len(views) or views[c.raw_view] is None):
        c.kind = "box"
    return c


def _as_rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def build_glyphs(text_items, target_views, step, rng, cfg=None) -> List[GlyphInstance]:
    """Training glyph conditions for one sample.

    text_items: text.json dict / items list / path. target_views: 6 raw-ordered HxWx3 (or x4)
    uint8 images of the training target, used for gt crops (None, or a None entry for a view:
    gt falls back to box there).
    step drives the stage schedule. rng: np.random.Generator or seed; the output is a pure
    function of (inputs, cfg, rng state).
    """
    cfg = cfg or GlyphConfig()
    rng = _as_rng(rng)
    items = sorted(items_of(text_items), key=lambda it: int(it["id"]))
    if rng.random() < cfg.p_all_drop:
        return []
    p_gt, p_box, _ = stage_probs(cfg, step)
    kinds, lines = {}, {}
    for it in items:
        # fixed number of draws per item keeps the stream aligned whatever the branch
        u_drop, u_kind, u_wrap = rng.random(3)
        n_rand = int(rng.integers(1, max_lines_for(str(it.get("text") or "x"), cfg.max_lines) + 1))
        iid = int(it["id"])
        if u_drop < cfg.p_item_drop:
            kinds[iid] = None
            continue
        kind = "gt" if u_kind < p_gt else "box" if u_kind < p_gt + p_box else "fixed"
        if kind == "gt" and target_views is None:
            kind = "box"
        kinds[iid] = kind
        if kind == "fixed" and cfg.anchor_mode != "warp" and u_wrap < cfg.p_rewrap:
            lines[iid] = n_rand
    out = []
    for rank, c in enumerate(select_instances(items, cfg, kinds, lines)):
        patch = _make_patch(_gt_or_box(c, target_views), target_views, cfg)
        if c.kind != "gt" or cfg.augment_gt:
            patch = augment(patch, rng, cfg)
        patch, ids, keep = _anchor(c, patch, cfg)
        if cfg.jitter_tokens > 0:
            ids, _ = gid.jitter_ids(ids, keep, c.raw_view, rng, cfg.jitter_tokens)
        out.append(_instance(c, patch, ids, keep, cfg, rank))
    return _finish(out, cfg)


def build_glyphs_infer(text_items, cfg=None, views=None) -> List[GlyphInstance]:
    """Inference glyph conditions: every selected instance as cfg.infer_kind, no randomness.

    views (optional, raw-ordered) are only needed for infer_kind "gt", e.g. the reference photo
    mapped into the front view with None for the other five; gt instances in views without an
    image fall back to box.
    """
    cfg = cfg or GlyphConfig()
    items = sorted(items_of(text_items), key=lambda it: int(it["id"]))
    kind = cfg.infer_kind if (cfg.infer_kind != "gt" or views is not None) else "box"
    kinds = {int(it["id"]): kind for it in items}
    out = []
    for rank, c in enumerate(select_instances(items, cfg, kinds)):
        patch, ids, keep = _anchor(c, _make_patch(_gt_or_box(c, views), views, cfg), cfg)
        out.append(_instance(c, patch, ids, keep, cfg, rank))
    return _finish(out, cfg)


def _instance(c, patch, ids, keep, cfg, rank):
    return GlyphInstance(item_id=c.item_id, raw_view=c.raw_view, patch=patch,
                         token_hw=(patch.height // TOKEN_PX, patch.width // TOKEN_PX),
                         ids=np.asarray(ids, np.float32), keep=np.asarray(keep, bool),
                         kind=c.kind, text=c.item["text"], mode=cfg.anchor_mode,
                         bbox=tuple(_view_bbox(c.view)), quad=tuple(map(tuple, _view_quad(c.view))),
                         rank=rank)


def total_tokens(instances):
    return int(sum(i.n_keep for i in instances))


def patch_arrays(instances):
    """Patches as float32 [3, H, W] in [0, 1] (UniTEX feeds images as x.mul(2).sub(1) to the VAE)."""
    return [np.asarray(i.patch, np.float32).transpose(2, 0, 1) / 255.0 for i in instances]
