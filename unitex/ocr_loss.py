"""
OCR loss for the texture LoRA: a pretrained text recognizer (TrOCR) reads the text lines of the predicted
clean image during training, and the loss is its cross entropy of the right reading.

A training step whose noise level sigma is at most max_sigma:
  1. ocr_lines: the lines drawn in the step's layout (or any text.json), between min_height and max_height
     px tall at 1024 per view and at most max_aspect times as long as tall, at most max_lines of them at
     random.
  2. ocr_keep_mask: the target tokens of each line's box (its quad's bounding box plus margin_tokens) are
     kept in the noise drop, so the step predicts every one of them. They stay out of the flow matching
     loss when the drop would have removed them.
  3. the predicted clean latent x0 = x_t - sigma * v of each box goes through the frozen VAE decoder with
     gradients, one box at a time (decode_box), and the line's quad, padded by pad_frac of its height, is
     resampled into a straight line image (rectify, bilinear).
  4. label: the recognizer's own reading (greedy, its token ids) of the same box of the clean target
     latent, decoded the same way. A line whose reading does not match its text (case-folded NED below
     min_label_ned) is left out: the recognizer cannot read it even from the right latent.
  5. loss: mean over the kept lines of the label's per-token cross entropy given the predicted line image
     (teacher forcing).

The label comes from the clean latent after the same VAE decode, so the loss asks for nothing the VAE
cannot show, and it uses the recognizer's own tokens, so no tokenizer convention or letter case can
disagree with what it reads.

API (torch / numpy, transformers for the recognizer, the VAE is any diffusers AutoencoderKL):
  OcrConfig, ocr_config(spec, overrides)          config from JSON + key=value overrides
  ocr_lines(text, view_res, cfg, rng)             line dicts (item, raw view, text, quad, height, length)
  line_token_box(line, view_res, margin)          (r0, r1, c0, c1): the line's token box in the strip
  ocr_keep_mask(lines, view_res, margin)          bool [n_tokens] over the packed target
  rectify(img, quad, pad_frac)                    the quad's pixels as a straight line image (differentiable)
  decode_box(z, box, vae)                         pixels of a token box of a latent strip
  Reader(model)                                   greedy reading and teacher-forced NLL
  ocr_loss(lines, x0, z_clean, vae, reader, ...)  (loss or None, info)
"""

import json
import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F

try:
    from unitex import glyph as gl
    from unitex import glyph_tokens as gt
    from unitex import text_metrics as tm
    from unitex.common import FULL_INDEX, TOKEN_PX
except ImportError:   # imported with unitex/ itself on sys.path
    import glyph as gl
    import glyph_tokens as gt
    import text_metrics as tm
    from common import FULL_INDEX, TOKEN_PX

LATENT_PX = 8         # VAE downsampling: a token is 2 x 2 latent pixels, 16 x 16 image pixels


@dataclass
class OcrConfig:
    """Which lines the OCR loss reads and how. Heights are px at 1024 per view (scaled with the
    resolution)."""
    model: str = "microsoft/trocr-small-printed"
    max_sigma: float = 0.5           # the loss only where the one-step prediction can show letters
    min_height: float = 12.0
    max_height: float = 64.0
    max_aspect: float = 24.0         # length / height: the recognizer squeezes every line to a square
    max_lines: int = 6
    margin_tokens: int = 2           # around the quad's bounding box: the VAE decode is off at box borders
    pad_frac: float = 0.15           # line image margin, a fraction of the line height on every side
    min_label_ned: float = 0.8
    max_label_tokens: int = 48


def ocr_config(spec=None, overrides=()):
    """OcrConfig from a JSON file path or inline JSON object, then "key=value" overrides."""
    d = {}
    if spec:
        s = str(spec)
        d = json.loads(s) if s.lstrip().startswith("{") else json.load(open(s))
    for o in overrides or ():
        if "=" not in o:
            raise ValueError(f"OCR loss override {o!r} is not key=value")
        k, v = o.split("=", 1)
        d[k.strip()] = gt._parse_value(v.strip())
    known = {f.name for f in fields(OcrConfig)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown OcrConfig fields: {sorted(unknown)}")
    cfg = OcrConfig(**d)
    if not 0.0 < float(cfg.max_sigma) <= 1.0:
        raise ValueError(f"max_sigma must be in (0, 1], got {cfg.max_sigma}")
    if not 0.0 <= float(cfg.min_height) < float(cfg.max_height):
        raise ValueError(f"need 0 <= min_height < max_height, got {cfg.min_height}, {cfg.max_height}")
    if int(cfg.max_lines) < 1 or int(cfg.margin_tokens) < 0 or float(cfg.pad_frac) < 0:
        raise ValueError("max_lines must be >= 1, margin_tokens and pad_frac >= 0")
    if float(cfg.max_aspect) <= 0 or not 0.0 <= float(cfg.min_label_ned) <= 1.0:
        raise ValueError("max_aspect must be > 0 and min_label_ned in [0, 1]")
    return cfg


def ocr_config_dict(cfg):
    return asdict(cfg)


# ────────────────────────────────────────────────────────────────────────────
# Lines and token boxes
# ────────────────────────────────────────────────────────────────────────────

def _quad_size(q):
    tl, tr, br, bl = q
    return (float(np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2,
            float(np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)


def ocr_lines(text, view_res, cfg=None, rng=None):
    """text.json (anything glyph_tokens.load_text accepts, e.g. a layout's drawn_text) -> list of
    {item_id, raw_view, text, quad (4 x 2 float array at view_res, reading order TL TR BR BL), height,
    length} for every line view within the height and aspect limits. At most cfg.max_lines: a random
    subset with an rng (numpy Generator), else the first ones."""
    cfg = cfg or OcrConfig()
    text = gt.load_text(text)
    if text is None or not text.get("items"):
        return []
    R = int(view_res)
    tR = gt.rescale_text(text, R)
    lo, hi = float(cfg.min_height) * R / 1024, float(cfg.max_height) * R / 1024
    out = []
    for k, it in enumerate(tR.get("items", [])):
        s = " ".join(str(it.get("text") or "").split())
        if not s:
            continue
        for rv, v in sorted((it.get("views") or {}).items(), key=lambda kv: int(kv[0])):
            if not (v.get("bbox") or v.get("quad")):
                continue
            q = np.asarray(gl._view_quad(v), np.float64)
            length, height = _quad_size(q)
            if not lo <= height <= hi or length > float(cfg.max_aspect) * height:
                continue
            out.append({"item_id": int(it.get("id", k)), "raw_view": int(rv), "text": s, "quad": q,
                        "height": height, "length": length})
    if len(out) > int(cfg.max_lines):
        pick = rng.permutation(len(out))[:int(cfg.max_lines)] if rng is not None else range(int(cfg.max_lines))
        out = [out[i] for i in sorted(pick)]
    return out


def strip_quad(line, view_res):
    """The line's quad in strip pixels (its view sits in slot FULL_INDEX.index(raw_view))."""
    q = np.asarray(line["quad"], np.float64).copy()
    q[:, 0] += FULL_INDEX.index(int(line["raw_view"])) * int(view_res)
    return q


def line_token_box(line, view_res, margin=2):
    """(r0, r1, c0, c1): token rows r0..r1-1 and strip token columns c0..c1-1 covering the line's quad
    plus margin tokens, clipped to its view."""
    R = int(view_res)
    n = R // TOKEN_PX
    q = np.asarray(line["quad"], np.float64)
    off = FULL_INDEX.index(int(line["raw_view"])) * n
    r0 = max(int(math.floor(q[:, 1].min() / TOKEN_PX)) - margin, 0)
    r1 = min(int(math.ceil(q[:, 1].max() / TOKEN_PX)) + margin, n)
    c0 = max(int(math.floor(q[:, 0].min() / TOKEN_PX)) - margin, 0)
    c1 = min(int(math.ceil(q[:, 0].max() / TOKEN_PX)) + margin, n)
    return r0, max(r1, r0 + 1), c0 + off, max(c1, c0 + 1) + off


def ocr_keep_mask(lines, view_res, margin=2):
    """Bool [(R/16) (6R/16)] over the packed target (row-major): the token boxes of the lines."""
    R = int(view_res)
    nr, nc = R // TOKEN_PX, 6 * R // TOKEN_PX
    m = torch.zeros(nr, nc, dtype=torch.bool)
    for ln in lines:
        r0, r1, c0, c1 = line_token_box(ln, R, margin)
        m[r0:r1, c0:c1] = True
    return m.flatten()


# ────────────────────────────────────────────────────────────────────────────
# Pixels
# ────────────────────────────────────────────────────────────────────────────

def rectify(img, quad, pad_frac=0.15, size=None):
    """The pixels of img [B, C, h, w] inside quad (4 x 2, continuous coordinates of img with pixel
    centres at +0.5, reading order TL TR BR BL), padded by pad_frac of the line height on every side,
    resampled to a straight line image [B, C, oh, ow] (bilinear over the quad's bilinear map,
    differentiable). size (oh, ow) defaults to the padded line's own size in pixels."""
    q = torch.as_tensor(np.asarray(quad, np.float64), dtype=torch.float32, device=img.device)
    tl, tr, br, bl = q
    length = float(((tr - tl).norm() + (br - bl).norm()) / 2)
    height = float(((bl - tl).norm() + (br - tr).norm()) / 2)
    b = float(pad_frac)
    a = b * height / max(length, 1e-6)
    if size is None:
        size = (max(int(round(height * (1 + 2 * b))), 2), max(int(round(length * (1 + 2 * a))), 2))
    oh, ow = int(size[0]), int(size[1])
    u = -a + (1 + 2 * a) * (torch.arange(ow, device=img.device, dtype=torch.float32) + 0.5) / ow
    v = -b + (1 + 2 * b) * (torch.arange(oh, device=img.device, dtype=torch.float32) + 0.5) / oh
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uu, vv = uu[..., None], vv[..., None]
    p = (1 - uu) * (1 - vv) * tl + uu * (1 - vv) * tr + uu * vv * br + (1 - uu) * vv * bl    # [oh, ow, 2]
    h, w = img.shape[-2:]
    grid = torch.stack([p[..., 0] / w * 2 - 1, p[..., 1] / h * 2 - 1], dim=-1)
    grid = grid[None].expand(img.shape[0], -1, -1, -1).to(img.dtype)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


def decode_box(z, box, vae):
    """Pixels [B, 3, 16 (r1 - r0), 16 (c1 - c0)] in [0, 1] of token box (r0, r1, c0, c1) of the latent
    strip z [B, C, H / 8, W / 8] (model space: (latent - shift) * scale), through the VAE decoder.
    Gradients reach z (the decoder itself is frozen by the caller)."""
    r0, r1, c0, c1 = box
    k = TOKEN_PX // LATENT_PX
    zc = z[:, :, k * r0:k * r1, k * c0:k * c1]
    zc = zc / vae.config.scaling_factor + (vae.config.shift_factor or 0.0)
    x = vae.decode(zc.to(vae.dtype), return_dict=False)[0]
    return (x.float() / 2 + 0.5).clamp(0, 1)


# ────────────────────────────────────────────────────────────────────────────
# Recognizer
# ────────────────────────────────────────────────────────────────────────────

class Reader:
    """A TrOCR model (transformers VisionEncoderDecoderModel) for line images in [0, 1]: greedy reading
    and the teacher-forced NLL of a token sequence. Frozen, eval mode, fp32."""

    def __init__(self, model, device=None, dtype=torch.float32, processor=None):
        if isinstance(model, str):
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            processor = TrOCRProcessor.from_pretrained(model)
            model = VisionEncoderDecoderModel.from_pretrained(model)
        self.model = model.to(device=device, dtype=dtype).eval()
        self.model.requires_grad_(False)
        self.processor = processor
        ip = processor.image_processor
        self.size = (int(ip.size["height"]), int(ip.size["width"]))
        self.mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(ip.image_std, dtype=torch.float32).view(1, -1, 1, 1)
        gc = self.model.generation_config
        self.start_id = self.model.config.decoder_start_token_id
        self.eos_id = gc.eos_token_id if gc.eos_token_id is not None else self.model.config.eos_token_id
        if isinstance(self.eos_id, (list, tuple)):
            self.eos_id = self.eos_id[0]

    @property
    def device(self):
        return next(self.model.parameters()).device

    def prepare(self, x):
        """[B, 3, h, w] in [0, 1], any size -> the model's normalized input (resized like its image
        processor, antialiased bilinear)."""
        x = F.interpolate(x.float().to(self.device), size=self.size, mode="bilinear", antialias=True,
                          align_corners=False)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    @torch.no_grad()
    def read(self, x, max_tokens=48, prepared=False):
        """-> (texts, ids [B, T]): greedy readings, ids starting with the decoder start token. prepared:
        x is already the output of prepare()."""
        px = x if prepared else self.prepare(x)
        ids = self.model.generate(px, max_new_tokens=int(max_tokens), num_beams=1, do_sample=False)
        texts = self.processor.batch_decode(ids, skip_special_tokens=True)
        return [" ".join(t.split()) for t in texts], ids

    def valid_targets(self, ids):
        """(decoder inputs [B, T-1], targets [B, T-1], bool mask of the targets up to and with the first
        end token) for token sequences from read()."""
        inp, tgt = ids[:, :-1], ids[:, 1:]
        is_eos = tgt == self.eos_id
        after = (is_eos.long().cumsum(1) - is_eos.long()) > 0
        return inp, tgt, ~after

    def nll(self, x, ids, prepared=False):
        """Per-line mean NLL [B] of the token sequences ids (from read()) given line images x, teacher
        forced. Differentiable in x."""
        px = x if prepared else self.prepare(x)
        inp, tgt, valid = self.valid_targets(ids.to(px.device))
        logits = self.model(pixel_values=px, decoder_input_ids=inp).logits
        ce = F.cross_entropy(logits.float().transpose(1, 2), tgt, reduction="none")
        v = valid.float()
        return (ce * v).sum(1) / v.sum(1).clamp(min=1.0)


def label_ok(reading, text, min_ned):
    """The recognizer's reading of a clean line agrees with the line's text: NED at least min_ned after both
    are normalized (case-folded, punctuation dropped) and their spaces removed, since text.json lines from
    texture OCR often run words together. min_ned 0 accepts every reading."""
    if float(min_ned) <= 0:
        return True
    r, t = (tm.normalize(x).replace(" ", "") for x in (reading, text))
    return bool(r) and bool(t) and tm.ned(t, r) >= float(min_ned)


# ────────────────────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────────────────────

def ocr_loss(lines, x0, z_clean, vae, reader, view_res, cfg=None, read_pred=False):
    """OCR loss of one sample: lines from ocr_lines, x0 and z_clean the predicted and the clean latent
    strip [1, C, R / 8, 6R / 8] in model space. -> (loss or None when no line is kept, info). The loss is
    the mean NLL over the kept lines, differentiable in x0. info: n_lines, n_kept, labels (readings of
    the clean lines), nll (per kept line) and, with read_pred, pred (readings of the predicted lines)."""
    cfg = cfg or OcrConfig()
    R = int(view_res)
    info = {"n_lines": len(lines), "n_kept": 0, "labels": [], "nll": []}
    if not lines:
        return None, info
    clean, pred = [], []
    for ln in lines:
        box = line_token_box(ln, R, int(cfg.margin_tokens))
        q = strip_quad(ln, R) - np.array([box[2] * TOKEN_PX, box[0] * TOKEN_PX], np.float64)
        with torch.no_grad():
            clean.append(reader.prepare(rectify(decode_box(z_clean, box, vae), q, cfg.pad_frac)))
        pred.append(reader.prepare(rectify(decode_box(x0, box, vae), q, cfg.pad_frac)))
    readings, ids = reader.read(torch.cat(clean), cfg.max_label_tokens, prepared=True)
    keep = [i for i, ln in enumerate(lines) if label_ok(readings[i], ln["text"], cfg.min_label_ned)]
    info["labels"] = readings
    info["n_kept"] = len(keep)
    if read_pred:
        info["pred"] = reader.read(torch.cat(pred).detach(), cfg.max_label_tokens, prepared=True)[0]
    if not keep:
        return None, info
    nll = reader.nll(torch.cat([pred[i] for i in keep]), ids[keep], prepared=True)
    info["nll"] = [float(v) for v in nll.detach()]
    return nll.mean(), info
