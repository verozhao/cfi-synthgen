"""
Tests for unitex/ocr_loss.py: line selection, token boxes, the differentiable line rectification, the VAE box
decode, the recognizer wrapper and the loss, with a tiny random VAE and a tiny random TrOCR model. One test
reads a rendered line with a real TrOCR model when it is in the local Hugging Face cache (OCR_TEST_MODEL).
CPU only.

Run: python -m pytest unitex/tests/test_ocr_loss.py -v
"""

import os
import sys
import types

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from unitex import ocr_loss as ol
from unitex.common import FULL_INDEX

TINY_VAE = dict(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 8, 8),
                layers_per_block=1, latent_channels=16, norm_num_groups=4, scaling_factor=0.3611,
                shift_factor=0.1159, use_quant_conv=False, use_post_quant_conv=False,
                mid_block_add_attention=False)
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 "
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]


def tiny_trocr(seed=0, size=32):
    """A random VisionEncoderDecoderModel (ViT encoder, TrOCR decoder) over a character vocabulary:
    pad 0, bos 1, eos 2 (also the decoder start, like TrOCR), unk 3, then CHARS."""
    from transformers import TrOCRConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
    enc = ViTConfig(image_size=size, patch_size=8, num_channels=3, hidden_size=32, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=64)
    dec = TrOCRConfig(vocab_size=len(SPECIALS) + len(CHARS), d_model=32, decoder_layers=1, decoder_attention_heads=2,
                      decoder_ffn_dim=64, max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                      decoder_start_token_id=2, cross_attention_hidden_size=32)
    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc, dec)
    cfg.decoder_start_token_id, cfg.pad_token_id, cfg.eos_token_id = 2, 0, 2
    torch.manual_seed(seed)
    model = VisionEncoderDecoderModel(config=cfg)
    model.generation_config.decoder_start_token_id = 2
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = 2
    model.generation_config.bos_token_id = 1
    return model


def tiny_processor(size=32):
    """Processor stand-in: image size, mean and std, and a character batch_decode."""
    ip = types.SimpleNamespace(size={"height": size, "width": size}, image_mean=[0.5] * 3, image_std=[0.5] * 3)

    def batch_decode(ids, skip_special_tokens=True):
        return ["".join(CHARS[i - len(SPECIALS)] for i in row.tolist() if i >= len(SPECIALS)) for row in ids]
    return types.SimpleNamespace(image_processor=ip, batch_decode=batch_decode)


def save_tiny_reader(path, seed=0, size=32):
    """The tiny model and a real TrOCRProcessor (ViT image processor, character tokenizer) saved to path, so
    Reader(str(path)) and the trainer load it like a hub model."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, TrOCRProcessor, ViTImageProcessor
    vocab = {t: i for i, t in enumerate(SPECIALS + list(CHARS))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                                   unk_token="<unk>")
    proc = TrOCRProcessor(image_processor=ViTImageProcessor(size={"height": size, "width": size},
                                                            image_mean=[0.5] * 3, image_std=[0.5] * 3),
                          tokenizer=fast)
    tiny_trocr(seed, size).save_pretrained(str(path))
    proc.save_pretrained(str(path))
    return str(path)


@pytest.fixture(scope="module")
def reader():
    return ol.Reader(tiny_trocr(), processor=tiny_processor())


@pytest.fixture(scope="module")
def vae():
    from diffusers import AutoencoderKL
    torch.manual_seed(0)
    return AutoencoderKL(**TINY_VAE).eval().requires_grad_(False)


def _line(i, text, view, x0, y0, length, height, res=1024):
    q = [[x0, y0], [x0 + length, y0], [x0 + length, y0 + height], [x0, y0 + height]]
    return {"id": i, "text": text, "views": {str(view): {"quad": q}}}


def _text(items, res=1024):
    return {"version": 1, "res": res, "items": items}


# ────────────────────────────────────────────────────────────────────────────
# Lines, boxes and the keep mask
# ────────────────────────────────────────────────────────────────────────────

def test_ocr_lines_filters_and_picks():
    items = [_line(0, "OK SMALL", 0, 100, 100, 200, 20), _line(1, "TOO SMALL", 0, 100, 200, 100, 8),
             _line(2, "TOO BIG", 0, 100, 300, 400, 80), _line(3, "TOO LONG", 0, 0, 500, 900, 20),
             _line(4, "SIDE", 3, 50, 50, 300, 30), _line(5, " ", 0, 100, 600, 200, 20)]
    got = ol.ocr_lines(_text(items), 1024)
    assert [(ln["item_id"], ln["raw_view"]) for ln in got] == [(0, 0), (4, 3)]
    assert got[0]["height"] == pytest.approx(20) and got[0]["length"] == pytest.approx(200)
    # a layout's drawn_text at 512 is rescaled: heights are px at 1024 per view whatever the resolution
    half = ol.ocr_lines(_text([_line(0, "OK", 0, 50, 50, 100, 10)], res=512), 512)
    assert len(half) == 1 and half[0]["height"] == pytest.approx(10)
    assert ol.ocr_lines(_text([_line(0, "OK", 0, 50, 50, 100, 5)], res=512), 512) == []
    # at most max_lines: a random subset with an rng, repeatable, else the first ones
    many = _text([_line(i, f"L{i}", 0, 100, 40 * i, 300, 20) for i in range(20)])
    cfg = ol.OcrConfig(max_lines=4)
    a = ol.ocr_lines(many, 1024, cfg, np.random.default_rng(3))
    b = ol.ocr_lines(many, 1024, cfg, np.random.default_rng(3))
    assert [ln["item_id"] for ln in a] == [ln["item_id"] for ln in b] and len(a) == 4
    assert [ln["item_id"] for ln in ol.ocr_lines(many, 1024, cfg)] == [0, 1, 2, 3]
    assert ol.ocr_lines(None, 1024) == [] and ol.ocr_lines("", 1024) == []


def test_token_box_and_keep_mask():
    ln = {"raw_view": 3, "quad": np.array([[100.0, 180.0], [400.0, 180.0], [400.0, 200.0], [100.0, 200.0]])}
    s = FULL_INDEX.index(3) * 64
    assert ol.line_token_box(ln, 1024, 0) == (11, 13, 6 + s, 25 + s)
    assert ol.line_token_box(ln, 1024, 2) == (9, 15, 4 + s, 27 + s)
    edge = {"raw_view": 0, "quad": np.array([[2.0, 2.0], [60.0, 2.0], [60.0, 20.0], [2.0, 20.0]])}
    assert ol.line_token_box(edge, 1024, 3) == (0, 5, 0, 7)          # clipped to the view
    m = ol.ocr_keep_mask([ln, edge], 1024, 2).view(64, 384)
    assert m.numel() == 64 * 384 and int(m.sum()) == 6 * 23 + 4 * 6
    assert m[9:15, 4 + s:27 + s].all() and m[0:4, 0:6].all() and not m[15:, :].any()
    q = ol.strip_quad(ln, 1024)
    assert q[0, 0] == 100 + FULL_INDEX.index(3) * 1024 and q[0, 1] == 180


def test_config_parsing():
    cfg = ol.ocr_config('{"max_lines": 3}', ["max_sigma=0.4", "model=\"some/model\""])
    assert cfg.max_lines == 3 and cfg.max_sigma == 0.4 and cfg.model == "some/model"
    for bad in (["nope=1"], ["max_sigma=0"], ["min_height=70"], ["max_lines=0"], ["min_label_ned=2"]):
        with pytest.raises(ValueError):
            ol.ocr_config(None, bad)
    assert ol.ocr_config_dict(cfg)["max_lines"] == 3


# ────────────────────────────────────────────────────────────────────────────
# Pixels
# ────────────────────────────────────────────────────────────────────────────

def test_rectify_axis_aligned_rotated_and_padded():
    img = torch.rand(1, 3, 64, 96)
    # an axis-aligned quad on pixel borders, no padding: exactly the crop
    out = ol.rectify(img, np.array([[8, 8], [56, 8], [56, 24], [8, 24]], float), 0.0)
    assert out.shape == (1, 3, 16, 48) and torch.allclose(out, img[:, :, 8:24, 8:56], atol=1e-5)
    # a line reading downwards (quad TL at the top right): the image turned a quarter clockwise
    rot = ol.rectify(img, np.array([[40, 8], [40, 56], [24, 56], [24, 8]], float), 0.0)
    assert rot.shape == (1, 3, 16, 48)
    assert torch.allclose(rot, img[:, :, 8:56, 24:40].rot90(1, dims=(2, 3)), atol=1e-5)
    # padding adds pad_frac of the height on every side
    pad = ol.rectify(img, np.array([[20, 20], [60, 20], [60, 40], [20, 40]], float), 0.2)
    assert pad.shape == (1, 3, 28, 48) and torch.allclose(pad, img[:, :, 16:44, 16:64], atol=1e-4)
    # differentiable, and only the pixels it samples get a gradient
    x = img.clone().requires_grad_(True)
    ol.rectify(x, np.array([[8, 8], [56, 8], [56, 24], [8, 24]], float), 0.0).sum().backward()
    g = x.grad[0, 0]
    assert (g[8:24, 8:56] > 0).all() and g.sum() == pytest.approx(16 * 48) and float(g[30:, :].abs().sum()) == 0
    assert ol.rectify(img, np.array([[8, 8], [56, 8], [56, 24], [8, 24]], float), 0.0, size=(32, 32)).shape == (1, 3, 32, 32)


def test_decode_box_is_the_vae_decode_of_its_latent_box(vae):
    z = torch.randn(1, 16, 8, 48)                   # a 64 x 384 px strip, 4 x 24 tokens
    box = (1, 3, 5, 9)
    got = ol.decode_box(z, box, vae)
    ref = vae.decode(z[:, :, 2:6, 10:18] / vae.config.scaling_factor + vae.config.shift_factor).sample
    assert got.shape == (1, 3, 32, 64) and torch.allclose(got, (ref / 2 + 0.5).clamp(0, 1), atol=1e-6)
    zz = z.clone().requires_grad_(True)
    ol.decode_box(zz, box, vae).sum().backward()
    inside = torch.zeros_like(z, dtype=torch.bool)
    inside[:, :, 2:6, 10:18] = True
    assert zz.grad[inside].abs().sum() > 0 and zz.grad[~inside].abs().sum() == 0


# ────────────────────────────────────────────────────────────────────────────
# Recognizer
# ────────────────────────────────────────────────────────────────────────────

def test_reader_read_targets_and_nll(reader):
    x = torch.rand(3, 3, 20, 70)
    texts, ids = reader.read(x, max_tokens=6)
    assert len(texts) == 3 and ids.shape[0] == 3 and (ids[:, 0] == 2).all() and ids.shape[1] <= 7
    inp, tgt, valid = reader.valid_targets(torch.tensor([[2, 5, 6, 2, 0, 0], [2, 7, 2, 2, 2, 2]]))
    assert inp.tolist() == [[2, 5, 6, 2, 0], [2, 7, 2, 2, 2]]
    assert valid.tolist() == [[True, True, True, False, False], [True, True, False, False, False]]
    xx = x.clone().requires_grad_(True)
    nll = reader.nll(xx, ids)
    assert nll.shape == (3,) and torch.isfinite(nll).all() and (nll >= 0).all()
    nll.sum().backward()
    assert xx.grad.abs().sum() > 0
    # prepared input gives the same numbers
    assert torch.allclose(reader.nll(reader.prepare(x), ids, prepared=True), nll.detach(), atol=1e-6)
    # the NLL of a token sequence is the sum of its own steps' cross entropies over its valid targets
    one = reader.nll(x[:1], ids[:1]).item()
    lp = torch.log_softmax(reader.model(pixel_values=reader.prepare(x[:1]), decoder_input_ids=ids[:1, :-1]).logits, -1)[0]
    _, t, v = reader.valid_targets(ids[:1])
    manual = -lp[torch.arange(t.shape[1]), t[0]][v[0]].mean().item()
    assert one == pytest.approx(manual, rel=1e-5)


def test_label_ok():
    assert ol.label_ok("Gluten Free", "GLUTEN FREE", 0.8)
    assert ol.label_ok("GLUTEN FRE", "GLUTEN FREE", 0.8) and not ol.label_ok("BLUTEN", "GLUTEN FREE", 0.8)
    assert not ol.label_ok("", "GLUTEN", 0.5) and not ol.label_ok("x", "", 0.5)
    assert ol.label_ok("", "anything", 0.0)


def test_saved_tiny_reader_loads_by_path(tmp_path):
    p = save_tiny_reader(tmp_path / "tiny_reader")
    r = ol.Reader(p)
    texts, ids = r.read(torch.rand(2, 3, 16, 64), max_tokens=5)
    assert len(texts) == 2 and (ids[:, 0] == 2).all() and r.size == (32, 32) and r.eos_id == 2


# ────────────────────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────────────────────

def _scene():
    """A 64 x 384 strip at view res 64 (tokens 4 x 24) with two lines, one in view 0 and one in view 3."""
    R = 64
    text = _text([_line(0, "abc", 0, 8, 20, 40, 12, res=R), _line(1, "xy", 3, 10, 24, 36, 10, res=R)], res=R)
    cfg = ol.OcrConfig(min_height=0.0, max_height=200.0, margin_tokens=1, min_label_ned=0.0)
    return R, ol.ocr_lines(text, R, cfg), cfg


def test_ocr_loss_gradient_stays_in_the_line_boxes_and_lowers_the_loss(vae, reader):
    R, lines, cfg = _scene()
    assert len(lines) == 2
    torch.manual_seed(1)
    z_clean = torch.randn(1, 16, R // 8, 6 * R // 8)
    x0 = (z_clean + 0.8 * torch.randn_like(z_clean)).requires_grad_(True)
    loss, info = ol.ocr_loss(lines, x0, z_clean, vae, reader, R, cfg, read_pred=True)
    assert info["n_lines"] == info["n_kept"] == 2 and len(info["labels"]) == len(info["pred"]) == 2
    assert torch.isfinite(loss) and loss.item() == pytest.approx(np.mean(info["nll"]), rel=1e-5)
    loss.backward()
    inside = torch.zeros_like(z_clean, dtype=torch.bool)
    for ln in lines:
        r0, r1, c0, c1 = ol.line_token_box(ln, R, cfg.margin_tokens)
        inside[:, :, 2 * r0:2 * r1, 2 * c0:2 * c1] = True
    assert x0.grad[inside].abs().sum() > 0 and x0.grad[~inside].abs().sum() == 0
    # descending the loss on x0 lowers it (the labels come from z_clean, so they stay fixed)
    opt = torch.optim.Adam([x0], lr=0.1)
    first = loss.item()
    for _ in range(20):
        opt.zero_grad()
        loss, _ = ol.ocr_loss(lines, x0, z_clean, vae, reader, R, cfg)
        loss.backward()
        opt.step()
    last = ol.ocr_loss(lines, x0, z_clean, vae, reader, R, cfg)[0].item()
    print(f"OCR loss on x0: {first:.4f} -> {last:.4f} after 20 Adam steps")
    assert last < first - 1e-3


def test_ocr_loss_skips_lines_the_recognizer_cannot_read(vae, reader):
    R, lines, cfg = _scene()
    z = torch.randn(1, 16, R // 8, 6 * R // 8)
    strict = ol.OcrConfig(**dict(ol.ocr_config_dict(cfg), min_label_ned=1.0))
    loss, info = ol.ocr_loss(lines, z.clone().requires_grad_(True), z, vae, reader, R, strict)
    assert loss is None and info["n_kept"] == 0 and info["n_lines"] == 2
    assert ol.ocr_loss([], z, z, vae, reader, R, cfg) == (None, {"n_lines": 0, "n_kept": 0, "labels": [], "nll": []})


# ────────────────────────────────────────────────────────────────────────────
# A real recognizer (skipped unless the model is in the local Hugging Face cache)
# ────────────────────────────────────────────────────────────────────────────

def _cached_model():
    name = os.environ.get("OCR_TEST_MODEL", "microsoft/trocr-small-printed")
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(name, "config.json")
    except Exception:
        hit = None
    return name if isinstance(hit, str) else None


def _render_line(text, h=40, pad=12):
    from PIL import Image, ImageDraw, ImageFont
    from unitex import glyph as gl
    font = gl.load_font(h)
    l, t, r, b = font.getbbox(text)
    img = Image.new("RGB", (r - l + 2 * pad, b - t + 2 * pad), (235, 230, 220))
    ImageDraw.Draw(img).text((pad - l, pad - t), text, fill=(20, 30, 90), font=font)
    return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None]


def test_real_trocr_reads_a_line_and_its_nll_follows_the_letters():
    name = _cached_model()
    if name is None:
        pytest.skip("no TrOCR model in the local Hugging Face cache (set OCR_TEST_MODEL)")
    rd = ol.Reader(name)
    x = _render_line("GLUTEN FREE")
    texts, ids = rd.read(x)
    assert ol.label_ok(texts[0], "GLUTEN FREE", 0.8), texts
    blurred = torch.nn.functional.avg_pool2d(x, 9, stride=1, padding=4)
    wrong = _render_line("BLUTEN FREE")
    clean_nll = rd.nll(x, ids).item()
    assert rd.nll(blurred, ids).item() > clean_nll and rd.nll(wrong, ids).item() > clean_nll
    # gradient descent on a noisy copy lowers the NLL of the clean reading
    noisy = (x + 0.35 * torch.randn_like(x)).clamp(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([noisy], lr=0.01)
    start = rd.nll(noisy, ids).item()
    for _ in range(10):
        opt.zero_grad()
        rd.nll(noisy, ids).sum().backward()
        opt.step()
    assert rd.nll(noisy, ids).item() < start
