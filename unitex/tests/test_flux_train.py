"""
Tests for the UniTEX-FLUX GlyphAnchor training integration on CPU with a tiny random FLUX:
unitex/glyph_tokens.py, unitex/flux_dataset.py and unitex/patches/UniTEX-FLUX.patch.

Setup (session fixtures):
  pristine   UniTEX-FLUX at 036a736 ($UNITEX_FLUX_ROOT), used read-only
  patched    a copy of it with our patch applied by `git apply` (so the patch file itself is tested)
  pretrained tiny FluxTransformer2DModel (1 + 1 blocks, 2 heads of 16, rope (4, 6, 6)) and tiny
             AutoencoderKL (16 latent channels, FLUX scaling / shift) in the subfolder layout
             trainer.py loads (scheduler/, vae/, transformer/), no tokenizers or text encoders
  synth      mvgen-layout dataset written here (6 views with text, 20 reference views, text.json),
             as PNG and as LMDB (pack_lmdb.py)
  real       mvgen renders with a text.json ($CFI_FLUX_TEST_DATA), tests skip without it

Trainer runs happen in subprocesses (this file is also the worker: `python test_flux_train.py
worker spec.json`), because trainer.py imports its sibling `pipeline` module by bare name and the
pristine and patched copies must not share sys.modules.

Run: python -m pytest unitex/tests/test_flux_train.py -v      (FLUX_TEST_SLOW=0 skips e2e / 1024)
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from unitex import flux_dataset as fd
from unitex import glyph as gl
from unitex import glyph_ids as gid
from unitex import glyph_tokens as gt
from unitex.common import FULL_INDEX, RAW_C2W

PRISTINE = os.environ.get("UNITEX_FLUX_ROOT", "/Users/test/.claude/jobs/5ee54a5d/tmp/UniTEX-FLUX")
PATCH = os.path.join(REPO, "unitex", "patches", "UniTEX-FLUX.patch")
REAL_DATA = os.environ.get("CFI_FLUX_TEST_DATA", "/Users/test/.claude/jobs/5ee54a5d/tmp/stage2_trainer/data512")
REAL_DATA_1024 = os.environ.get("CFI_FLUX_TEST_DATA_1024", "/Users/test/.claude/jobs/5ee54a5d/tmp/stage2_trainer/data1024")
SLOW = os.environ.get("FLUX_TEST_SLOW", "1") != "0"
THIS = os.path.abspath(__file__)

TINY_FLUX = dict(patch_size=1, in_channels=64, num_layers=1, num_single_layers=1, attention_head_dim=16,
                 num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=16,
                 guidance_embeds=True, axes_dims_rope=(4, 6, 6))
TINY_VAE = dict(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 8, 8),
                layers_per_block=1, latent_channels=16, norm_num_groups=4, scaling_factor=0.3611,
                shift_factor=0.1159, use_quant_conv=False, use_post_quant_conv=False,
                mid_block_add_attention=False)
FLUX_SCHEDULER = dict(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True, base_shift=0.5,
                      max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures: repos, tiny FLUX, synthetic data
# ────────────────────────────────────────────────────────────────────────────

def _need_pristine():
    if not os.path.isfile(os.path.join(PRISTINE, "tasks", "texturing", "trainer.py")):
        pytest.skip(f"UniTEX-FLUX checkout not found at {PRISTINE} (set UNITEX_FLUX_ROOT)")


@pytest.fixture(scope="session")
def patched(tmp_path_factory):
    _need_pristine()
    dst = str(tmp_path_factory.mktemp("uf") / "UniTEX-FLUX")
    shutil.copytree(PRISTINE, dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    r = subprocess.run(["git", "apply", "--verbose", PATCH], cwd=dst, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return dst


@pytest.fixture(scope="session")
def pretrained(tmp_path_factory):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    d = tmp_path_factory.mktemp("tiny_flux")
    torch.manual_seed(0)
    FluxTransformer2DModel(**TINY_FLUX).save_pretrained(str(d / "transformer"))
    AutoencoderKL(**TINY_VAE).save_pretrained(str(d / "vae"))
    FlowMatchEulerDiscreteScheduler(**FLUX_SCHEDULER).save_pretrained(str(d / "scheduler"))
    return str(d)


@pytest.fixture(scope="session")
def tiny_vae():
    from diffusers import AutoencoderKL
    torch.manual_seed(0)
    return AutoencoderKL(**TINY_VAE).eval()


def _rz(deg):
    a = math.radians(deg)
    m = np.eye(4)
    m[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return m


SYNTH_TEXT = {0: ("FRONT LABEL", (96, 200, 416, 248)), 1: ("SIDE", (176, 96, 336, 128)),
              2: ("BACK PANEL", (128, 300, 384, 332))}


def _view_images(raw, res):
    """rgb RGBA (soft alpha ring), albedo RGB, nocs RGBA (hard mask), normal RGB for one raw view."""
    x0, y0, x1, y1 = 128, 64, 384, 448
    mask = np.zeros((res, res), bool)
    mask[y0:y1, x0:x1] = True
    alpha = np.zeros((res, res), np.uint8)
    alpha[y0 - 1:y1 + 1, x0 - 1:x1 + 1] = 96
    alpha[mask] = 255
    col = np.array([200 - 20 * raw, 60 + 25 * raw, 90 + 10 * raw], np.uint8)
    rgb = np.zeros((res, res, 3), np.uint8)
    rgb[alpha > 0] = col
    img = Image.fromarray(rgb)
    if raw in SYNTH_TEXT:
        text, box = SYNTH_TEXT[raw]
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, fill=(245, 240, 225))
        font = gl._ink_font([text], box[2] - box[0] - 8, box[3] - box[1] - 8, None)
        l, t, r, b = font.getbbox(text)
        draw.text(((box[0] + box[2] - (r - l)) / 2 - l, (box[1] + box[3] - (b - t)) / 2 - t), text,
                  fill=(10, 10, 10), font=font)
    rgb = np.asarray(img)
    rgba = np.dstack([rgb, alpha])
    albedo = rgb.copy()
    albedo[alpha == 0] = 255
    yy, xx = np.mgrid[0:res, 0:res]
    p = np.stack([(xx / res * 2 - 1) * 0.95, (yy / res * 2 - 1) * 0.95, np.full(xx.shape, 0.3)], -1)
    ccm = ((p + 1) / 2 * 255).astype(np.uint8)
    ccm[~mask] = 0
    nocs = np.dstack([ccm, mask.astype(np.uint8) * 255])
    normal = np.full((res, res, 3), 255, np.uint8)
    normal[mask] = (127, 127, 255)
    return rgba, albedo, nocs, normal


def _synth_text_json(res):
    items = []
    for raw, (text, box) in SYNTH_TEXT.items():
        x0, y0, x1, y1 = [float(v) for v in box]
        q = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        G = gid.grid_from_quad(q)
        grid = {"ns": 16, "nt": 4, "xy": [[[float(v) for v in G[i, j]] for j in range(16)] for i in range(4)]}
        items.append({"id": len(items), "text": text, "conf": 1.0, "provenance": "front", "flags": [],
                      "views": {str(raw): {"bbox": [x0, y0, x1, y1], "quad": q, "pixels": 1000,
                                           "coverage": 1.0, "cos": 0.95, "height_px": y1 - y0, "grid": grid}}})
    return {"version": 1, "res": res, "source": "manual", "items": items}


def make_synth_root(root, uids=("synth/a", "synth/b"), res=512, n_ref=20):
    for u, uid in enumerate(uids):
        rd = os.path.join(root, "render", uid)
        qd = os.path.join(root, "render_random", uid)
        cd = os.path.join(root, "caption", uid)
        for d in (rd, qd, cd):
            os.makedirs(d, exist_ok=True)
        for raw in range(6):
            rgba, albedo, nocs, normal = _view_images((raw + u) % 6, res)
            Image.fromarray(rgba, "RGBA").save(os.path.join(rd, f"{raw:04d}_rgb.png"))
            Image.fromarray(albedo, "RGB").save(os.path.join(rd, f"{raw:04d}_albedo.png"))
            Image.fromarray(nocs, "RGBA").save(os.path.join(rd, f"{raw:04d}_nocs.png"))
            Image.fromarray(normal, "RGB").save(os.path.join(rd, f"{raw:04d}_normal.png"))
            Image.new("RGB", (1, 1), (128, 128, 255)).save(os.path.join(rd, f"{raw:04d}_bump.png"))
            Image.new("L", (1, 1), 0).save(os.path.join(rd, f"{raw:04d}_metallic.png"))
            Image.new("L", (1, 1), 153).save(os.path.join(rd, f"{raw:04d}_roughness.png"))
        with open(os.path.join(rd, "metadata.json"), "w") as f:
            json.dump({"cam2world_matrixs": RAW_C2W.tolist(), "res": res}, f)
        with open(os.path.join(rd, "text.json"), "w") as f:
            json.dump(_synth_text_json(res), f)
        cams = []
        for k in range(n_ref):
            yaw = -100 + 200 * k / max(n_ref - 1, 1)
            cams.append((_rz(yaw) @ RAW_C2W[0]).tolist())
            rgba, albedo, _, _ = _view_images((k + u) % 3, res)
            rgba = np.roll(rgba, 3 * k, axis=1)
            Image.fromarray(rgba, "RGBA").save(os.path.join(qd, f"{k:04d}_rgb.png"))
            Image.fromarray(np.roll(albedo, 3 * k, axis=1), "RGB").save(os.path.join(qd, f"{k:04d}_albedo.png"))
        with open(os.path.join(qd, "metadata.json"), "w") as f:
            json.dump({"cam2world_matrixs": cams}, f)
        with open(os.path.join(cd, "prompt.txt"), "w") as f:
            f.write("[MVFLUX]")
    with open(os.path.join(root, "training_uid.json"), "w") as f:
        json.dump(list(uids), f)
    return root


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    return make_synth_root(str(tmp_path_factory.mktemp("synth")))


@pytest.fixture(scope="session")
def synth_mdb(synth, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("synth_mdb"))
    r = subprocess.run([sys.executable, os.path.join(REPO, "pack_lmdb.py"), "--root", synth, "--out", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(out, "render", "synth", "a", "data.mdb"))
    return out


def _need_real():
    if not os.path.isfile(os.path.join(REAL_DATA, "training_uid.json")):
        pytest.skip(f"no mvgen test data at {REAL_DATA} (set CFI_FLUX_TEST_DATA)")


def _import_pristine_datasets():
    import importlib.util
    os.environ.setdefault("CAUGHT_ALL_EXCEPTIONS", "1")    # datasets.py:70 zipfile NameError otherwise
    # datasets.py:168 parses PIL.ImageColor.colormap as "#rrggbb" strings at import, but Pillow's
    # getrgb() caches parsed tuples into that dict, so any earlier named-colour draw in this process
    # (other test files) breaks the import. Put the strings back.
    from PIL import ImageColor
    for k, v in list(ImageColor.colormap.items()):
        if not isinstance(v, str):
            ImageColor.colormap[k] = "#%02x%02x%02x" % tuple(v[:3])
    spec = importlib.util.spec_from_file_location("_uf_datasets", os.path.join(PRISTINE, "data", "datasets.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_pristine_pipeline():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_uf_pipeline", os.path.join(PRISTINE, "tasks", "texturing", "pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ────────────────────────────────────────────────────────────────────────────
# glyph_tokens: packing, encoding, masks, config
# ────────────────────────────────────────────────────────────────────────────

def test_pack_latents_matches_unitex_flux():
    _need_pristine()
    P = _import_pristine_pipeline().PBRFluxPipeline
    lat = torch.randn(2, 16, 6, 10)
    assert torch.equal(gt.pack_latents(lat), P._pack_latents(lat, 2, 16, 6, 10, pixel_shuffle=True))


def _inst(w, h, keep=None, raw=0, r0=0, c0=0, text="X"):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((2, 2, w - 3, h - 3), outline=(0, 0, 0))
    gh, gw = h // 16, w // 16
    ids = gid._pack(r0 + np.arange(gh), c0 + np.arange(gw), 1)
    keep = np.ones(gh * gw, bool) if keep is None else np.asarray(keep, bool)
    ids[~keep, 1:] = np.nan
    return gl.GlyphInstance(item_id=0, raw_view=raw, patch=img, token_hw=(gh, gw), ids=ids, keep=keep,
                            kind="fixed", text=text, mode="center", bbox=(0, 0, w, h), quad=())


def test_encode_glyphs_matches_manual_encode_and_order(tiny_vae):
    vae = tiny_vae
    keep = np.ones(6, bool)
    keep[[1, 4]] = False
    insts = [_inst(32, 16, c0=10), _inst(48, 32, keep=keep, r0=5, c0=40), _inst(32, 16, c0=100)]
    lat, ids = gt.encode_glyphs(insts, vae, sample_mode="argmax")
    assert lat.shape == (1, 2 + 4 + 2, 64) and ids.shape == (8, 3) and ids.dtype == torch.float32
    assert torch.isfinite(ids).all() and (ids[:, 0] == 1).all()
    # manual reference: one VAE call per patch, (z - shift) * scale, UniTEX packing, keep mask
    ref = []
    for inst in insts:
        x = gt.patch_tensor(inst.patch)[None] * 2 - 1
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
        ref.append(gt.pack_latents(z)[:, torch.from_numpy(inst.keep)])
    ref = torch.cat(ref, 1)
    assert torch.allclose(lat, ref, atol=1e-5), (lat - ref).abs().max()
    exp_ids, _ = gid.concat_ids(insts)
    assert np.array_equal(ids.numpy(), exp_ids)
    # batching by size (two 32x16 patches share one call) gives the same tokens as max_batch=1
    lat1, _ = gt.encode_glyphs(insts, vae, sample_mode="argmax", max_batch=1)
    assert torch.allclose(lat, lat1, atol=1e-5)
    # empty
    lat0, ids0 = gt.encode_glyphs([], vae)
    assert lat0.shape == (1, 0, 64) and ids0.shape == (0, 3) and ids0.dtype == torch.float32


def test_text_token_mask_and_boost():
    text = {"res": 512, "items": [{"id": 0, "text": "A", "views": {"1": {"bbox": [20, 40, 50, 60]}}}]}
    m = gt.text_token_mask(text, 512).reshape(32, 192)
    slot = FULL_INDEX.index(1)
    rows, cols = np.nonzero(m.numpy())
    # pixels x 20..50 -> tokens 1..3, y 40..60 -> tokens 2..3, shifted by the slot
    assert set(rows) == {2, 3} and set(cols) == {slot * 32 + c for c in (1, 2, 3)}
    m2 = gt.text_token_mask(text, 1024).reshape(64, 384)
    rows2, cols2 = np.nonzero(m2.numpy())
    assert rows2.min() == 5 and rows2.max() == 7 and cols2.min() == slot * 64 + 2 and cols2.max() == slot * 64 + 6
    p = gt.boosted_keep_prob(0.25, m.reshape(-1), 0.5)
    assert float(p[m.reshape(-1)].unique()) == pytest.approx(0.625) and float(p[~m.reshape(-1)].unique()) == 0.25
    assert gt.text_token_mask("", 512).sum() == 0


def test_glyph_config_json_and_overrides(tmp_path):
    cfg = gt.glyph_config('{"anchor_mode": "warp", "stages": [[100, 1, 0, 0], ["inf", 0, 0, 1]]}',
                          ["token_budget=512", "aug_scale=[0.9, 1.1]", "drop_flags=[]"])
    assert cfg.anchor_mode == "warp" and cfg.token_budget == 512 and cfg.aug_scale == (0.9, 1.1)
    assert cfg.stages[1][0] == math.inf and cfg.drop_flags == ()
    p = tmp_path / "g.json"
    p.write_text(json.dumps(gt.glyph_config_dict(cfg)))
    assert gt.glyph_config(str(p)) == cfg
    with pytest.raises(ValueError):
        gt.glyph_config(None, ["no_such_field=1"])


# ────────────────────────────────────────────────────────────────────────────
# glyph_tokens: per-view resolution
# ────────────────────────────────────────────────────────────────────────────

def _fake_tokens(img):
    """8x8 average pool + UniTEX packing (a stand-in VAE with the real spatial footprint)."""
    a = np.asarray(img, np.float64)
    h, w = a.shape[0] // 8, a.shape[1] // 8
    lat = torch.from_numpy(a.reshape(h, 8, w, 8, 3).mean(axis=(1, 3)).transpose(2, 0, 1)[None].copy())
    lat = torch.cat([lat] * 5 + [lat[:, :1]], 1)
    return gt.pack_latents(lat)[0].numpy()


@pytest.mark.parametrize("mode", ["center", "stretch", "warp"])
def test_scale_to_1024_gt_tokens_sit_on_matching_target_tokens(mode):
    """At 1024 px per view, glyph token k of a gt crop holds the pixels of the target strip token at
    ids[k] (16-aligned box), so the 512 -> 1024 rebuild keeps patch pixels and ids consistent."""
    rng = np.random.default_rng(0)
    R = 1024
    views = [rng.integers(0, 256, (R, R, 3), dtype=np.uint8) for _ in range(6)]
    strip = np.concatenate([views[FULL_INDEX[s]] for s in range(6)], axis=1)
    tgt = _fake_tokens(strip)
    lut = {(int(r), int(c)): n for n, (r, c) in enumerate(np.ndindex(R // 16, 6 * R // 16))}
    cfg = gl.GlyphConfig(anchor_mode=mode, stages=[(math.inf, 1.0, 0.0, 0.0)], p_item_drop=0.0,
                         p_all_drop=0.0, jitter_tokens=0, nudge_collisions=False, warn_collisions=False)
    for raw in (0, 3, 5):
        box512 = [96, 64, 256, 112]     # 10 x 3 tokens at 512, 20 x 6 at 1024
        q = [[box512[0], box512[1]], [box512[2], box512[1]], [box512[2], box512[3]], [box512[0], box512[3]]]
        G = gid.grid_from_quad(q, ns=16, nt=4)
        text = {"res": 512, "items": [{"id": 0, "text": "X", "provenance": "front", "flags": [], "views": {
            str(raw): {"bbox": box512, "quad": q, "height_px": 48.0, "coverage": 1.0, "cos": 1.0,
                       "grid": {"ns": 16, "nt": 4, "xy": G.tolist()}}}}]}
        insts = gt.build_train_glyphs(text, views, 0, np.random.default_rng(1), cfg, view_res=R)
        assert len(insts) == 1
        inst = insts[0]
        assert inst.kind == "gt" and inst.token_hw == (6, 20) and inst.patch.size == (320, 96)
        toks = _fake_tokens(inst.patch)[inst.keep]
        ids = inst.ids[inst.keep]
        assert np.isfinite(ids).all() and (ids[:, 0] == 1).all()
        assert ids[:, 1].max() <= 63 and ids[:, 2].min() >= FULL_INDEX.index(raw) * 64
        assert ids[:, 2].max() <= FULL_INDEX.index(raw) * 64 + 63
        for k in range(len(ids)):
            n = lut[(int(round(ids[k, 1])), int(round(ids[k, 2])))]
            assert np.allclose(toks[k], tgt[n], atol=1e-6), (mode, raw, k)


def test_scale_to_1024_keeps_jitter_and_budget_semantics():
    text = _synth_text_json(512)
    cfg = gl.GlyphConfig(stages=[(math.inf, 0.0, 0.0, 1.0)], p_item_drop=0.0, p_all_drop=0.0, jitter_tokens=1)
    a = gt.build_train_glyphs(text, None, 0, np.random.default_rng(3), cfg, view_res=512)
    b = gt.build_train_glyphs(text, None, 0, np.random.default_rng(3), cfg, view_res=1024)
    assert [i.item_id for i in a] == [i.item_id for i in b]
    for i, j in zip(a, b):
        assert j.token_hw == (2 * i.token_hw[0], 2 * i.token_hw[1])
        assert j.patch.size == (2 * i.patch.size[0], 2 * i.patch.size[1])
        # the footprint centre follows the (jittered) 512 one: c1024 = 2 * c512 + 0.5, up to the
        # integer rounding of center mode at both resolutions (0.5 at 1024 + 2 x 0.5 at 512)
        c512 = i.ids[:, 1:].mean(0)
        c1024 = j.ids[:, 1:].mean(0)
        assert np.abs(c1024 - (2 * c512 + 0.5)).max() <= 1.5 + 1e-6
        assert np.array_equal(j.ids[:, 1:], np.round(j.ids[:, 1:]))
    assert gid.count_collisions(*gid.concat_ids(b)) == 0
    with pytest.raises(ValueError):
        gt.build_train_glyphs(text, None, 0, np.random.default_rng(3), cfg, view_res=768)


def test_text_json_written_at_1024_gives_the_same_glyphs():
    """text.json at res 1024 (renders at 1024) is brought to 512 geometry first: same instances."""
    t512 = _synth_text_json(512)
    t1024 = gt.rescale_text(t512, 1024)
    assert t1024["items"][0]["views"]["0"]["bbox"] == [2 * v for v in t512["items"][0]["views"]["0"]["bbox"]]
    cfg = gl.GlyphConfig(stages=[(math.inf, 0.0, 0.5, 0.5)], p_all_drop=0.0)
    for R in (512, 1024):
        a = gt.build_train_glyphs(t512, None, 0, np.random.default_rng(9), cfg, view_res=R)
        b = gt.build_train_glyphs(json.dumps(t1024), None, 0, np.random.default_rng(9), cfg, view_res=R)
        assert len(a) == len(b) > 0
        for i, j in zip(a, b):
            assert np.array_equal(i.ids, j.ids) and i.patch.size == j.patch.size and i.kind == j.kind


# ────────────────────────────────────────────────────────────────────────────
# flux_dataset: drop-in equivalence with UniTEX-FLUX datasets.py
# ────────────────────────────────────────────────────────────────────────────

def _pristine_sample(D, root, index, image_ext, seed, res=512):
    """launch.py's PBRTextureGenerationDataset config plus the four keys the trainer needs, built
    from the original loaders' own outputs (report section 2.6). res = target_image_resolution."""
    mv = D.MVDatasetConfig(image_ext=image_ext, full_index=(0, 3, 1, 2, 4, 5), load_bump=True,
                           load_metallic=True, load_roughness=True, load_prompt=True, n_rows=1, n_cols=6,
                           rgb_index=tuple(range(6)), albedo_index=tuple(range(6)), ccm_index=tuple(range(6)),
                           normal_index=tuple(range(6)), bump_index=tuple(range(6)),
                           metallic_index=tuple(range(6)), roughness_index=tuple(range(6)),
                           target_image_resolution=(res, res))
    ds = D.PBRTextureGenerationDataset(root, mv, extra_dataset=D.ReconstructDataset(
        root, D.ReconstructDatasetConfig(image_ext=image_ext, target_image_resolution=(res, res))))
    random.seed(seed)
    out = ds[index]
    ex = ds.dataset[index]
    extra = ds.extra_dataset[index]
    alb = D.ReconstructDataset(root, D.ReconstructDatasetConfig(image_ext=image_ext, rgb_postfix="_albedo",
                                                                target_image_resolution=(res, res)))
    alb.uid_list = ds.dataset.uid_list
    alb_ex = alb[index]
    # which reference did random.choices pick: find the matching candidate
    k = [i for i in range(len(extra["rgb"])) if torch.equal(extra["rgb"][i], out["rgbs_ip"])][0]
    out.update(alphas=ds.make_grid(ex["alpha"]), native_normals=out["normals"],
               alphas_ip=extra["alpha"][k], albedos_ip=alb_ex["rgb"][k], ref_index=k)
    return out


@pytest.mark.parametrize("fmt", ["png", "mdb"])
def test_dataset_matches_unitex_flux_loaders(fmt, synth, synth_mdb):
    _need_pristine()
    D = _import_pristine_datasets()
    root, ext = (synth, ".png") if fmt == "png" else (synth_mdb, ".mdb")
    ours = fd.CFIFluxDataset(root, image_ext=ext)
    for i in range(len(ours)):
        ref = _pristine_sample(D, root, i, ext, seed=11 + i)
        random.seed(11 + i)
        s = ours[i]
        assert s["ref_index"] == ref["ref_index"]
        for k in ("rgbs", "albedos", "alphas", "ccms", "native_normals", "rgbs_ip", "albedos_ip", "alphas_ip"):
            assert s[k].shape == ref[k].shape, k
            assert torch.equal(s[k], ref[k]), (k, (s[k] - ref[k]).abs().max())
        assert s["prompts"] == ref["prompts"] == "[MVFLUX]" and s["uids"] == ref["uids"]
        assert json.loads(s["text_json"])["items"]


def test_dataset_matches_unitex_flux_on_real_renders():
    _need_pristine()
    _need_real()
    D = _import_pristine_datasets()
    ours = fd.CFIFluxDataset(REAL_DATA, image_ext=".png")
    for i in range(len(ours)):
        ref = _pristine_sample(D, REAL_DATA, i, ".png", seed=5 + i)
        random.seed(5 + i)
        s = ours[i]
        assert s["ref_index"] == ref["ref_index"]
        for k in ("rgbs", "albedos", "alphas", "ccms", "native_normals", "rgbs_ip", "albedos_ip", "alphas_ip"):
            assert torch.equal(s[k], ref[k]), k


def test_reference_weights_follow_datasets_py(synth):
    ds = fd.CFIFluxDataset(synth)
    v = ds.load_views(*ds.samples[0])
    _, c2w = ds.ref_candidates(*ds.samples[0])
    w = fd.ref_weights(v["c2w"][[0], :3, 2], c2w[:, :3, 2]).numpy()
    yaw = np.linspace(-100, 100, 20)
    th = np.radians(np.abs(yaw))
    exp = ((np.cos(np.clip(1.5 * th, 0, np.pi)) + 1) / 2) ** 4
    exp[np.abs(yaw) >= 90] = ((np.cos(np.clip(1.5 * np.pi / 2, 0, np.pi)) + 1) / 2) ** 4
    assert np.allclose(w, exp, atol=1e-5)
    counts = np.bincount([ds.pick_reference(v["c2w"], c2w)[0] for _ in range(4000)], minlength=20)
    assert counts[9] + counts[10] > counts[0] + counts[19]      # frontal views dominate


def test_dataset_view_res_1024_and_collate(synth):
    ds = fd.CFIFluxDataset(synth, view_res=1024)
    s = ds[0]
    assert s["rgbs"].shape == (3, 1024, 6144) and s["alphas"].shape == (1, 1024, 6144)
    assert s["rgbs_ip"].shape == (3, 1024, 1024) and s["alphas_ip"].shape == (1, 1024, 1024)
    b = ds.collate_fn([ds[0], ds[1]])
    assert b["rgbs"].shape == (2, 3, 1024, 6144) and isinstance(b["text_json"], list) and len(b["uids"]) == 2
    rep = fd.check_dataset(ds, verbose=False)
    assert rep["n_failed"] == 0 and all("upsampled" in w[0] for w in rep["warnings"].values())


def test_check_dataset_passes_on_real_renders(capsys):
    _need_real()
    code = fd.main(["check", "--root", REAL_DATA])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "0 failed" in out


def test_check_dataset_real_1024_renders(capsys):
    """mvgen --res 1024 --ref-res 1024 output with a text.json at res 1024: no upsampling warning,
    glyphs build at 1024 from 1024 px geometry."""
    if not os.path.isfile(os.path.join(REAL_DATA_1024, "training_uid.json")):
        pytest.skip(f"no 1024 px mvgen data at {REAL_DATA_1024} (set CFI_FLUX_TEST_DATA_1024)")
    ds = fd.CFIFluxDataset(REAL_DATA_1024, view_res=1024)
    rep = fd.check_dataset(ds, verbose=False)
    assert rep["n_failed"] == 0 and rep["n_warned"] == 0, rep
    info = next(iter(rep["info"].values()))
    assert info["source_res"] == 1024 and info["text_res"] == 1024 and info["glyph_tokens"] > 0
    s = ds[0]
    assert s["rgbs"].shape == (3, 1024, 6144) and s["rgbs_ip"].shape == (3, 1024, 1024)


def test_check_dataset_reports_broken_samples_loudly(synth, tmp_path, capsys):
    root = str(tmp_path / "broken")
    shutil.copytree(synth, root)
    uids = ["synth/a", "synth/b", "synth/c", "synth/d"]
    for extra in ("c", "d"):
        shutil.copytree(os.path.join(root, "render", "synth", "a"), os.path.join(root, "render", "synth", extra))
        shutil.copytree(os.path.join(root, "render_random", "synth", "a"),
                        os.path.join(root, "render_random", "synth", extra))
    with open(os.path.join(root, "training_uid.json"), "w") as f:
        json.dump(uids, f)
    os.remove(os.path.join(root, "render", "synth", "b", "0003_albedo.png"))              # missing map
    p = os.path.join(root, "render", "synth", "c", "0002_rgb.png")
    Image.open(p).convert("RGB").save(p)                                                   # rgb without alpha
    meta = os.path.join(root, "render_random", "synth", "d", "metadata.json")
    with open(meta) as f:
        m = json.load(f)
    with open(meta, "w") as f:
        json.dump({"cam2world_matrixs": m["cam2world_matrixs"][:5]}, f)                    # 5 < 20 refs
    code = fd.main(["check", "--root", root])
    cap = capsys.readouterr()
    assert code == 1
    assert "3 failed" in cap.out
    for u, needle in (("synth/b", "0003_albedo"), ("synth/c", "no alpha channel"), ("synth/d", "need 20")):
        assert any(u in line and needle in line for line in cap.err.splitlines()), (u, cap.err)
    ds = fd.CFIFluxDataset(root)
    with pytest.raises(RuntimeError, match="synth/b"):
        ds[1]
    ds_skip = fd.CFIFluxDataset(root, skip_broken=True)
    assert ds_skip[1]["uids"] == "synth/a" and ds_skip.n_broken == 3     # b, c, d skipped, wraps to a


# ────────────────────────────────────────────────────────────────────────────
# Trainer runs (subprocess workers)
# ────────────────────────────────────────────────────────────────────────────

BASE_ARGV = ["--dataset_name", "unused", "--use_complex_dataset", "--six_views_or_four_views", "--dual_image",
             "--control_image", "--both_ccm_normal_condition", "--resolution", "512", "3072", "--n_rows", "1",
             "--n_cols", "6", "--lora_rank", "4", "--lora_alpha", "4", "--optimizer", "prodigy",
             "--learning_rate", "1.0", "--guidance_scale", "1.0", "--validation_steps", "10000000",
             "--checkpointing_steps", "1000000", "--train_batch_size", "1", "--gradient_accumulation_steps", "1",
             "--random_drop_noise", "--random_drop_noise_probability", "0.75", "--random_drop_condition",
             "--random_drop_condition_probability", "0.25", "--tasks", "texturing"]


class StopForward(Exception):
    pass


def worker(spec):
    """Build a Trainer from `spec` in this fresh process, run it, write a JSON result."""
    os.environ.setdefault("CAUGHT_ALL_EXCEPTIONS", "1")
    repo = spec["repo"]
    sys.path.insert(0, repo)
    import importlib
    import logging
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    launch = importlib.import_module("launch")
    tm = importlib.import_module("tasks.texturing.trainer")
    logging.basicConfig(level=logging.WARNING)
    logger = get_logger("worker")

    out_dir = spec["out_dir"]
    argv = BASE_ARGV + ["--pretrained_model_name_or_path", spec["pretrained"], "--output_dir", out_dir,
                        "--mixed_precision", spec.get("mixed_precision", "no"), "--seed", str(spec.get("seed", 7)),
                        "--max_train_steps", str(spec.get("steps", 1))] + spec.get("argv", [])
    if not spec.get("pristine"):
        argv += ["--dataset_impl", "cfi"]     # parse-time checks only, the batch comes from spec["sample"]
    drop = spec.get("no_drop", False)
    if drop:
        argv = [a for a in argv if a not in ("--random_drop_noise", "--random_drop_condition")]
    argv = [a for a in argv if a not in spec.get("drop_flags", [])]
    args = launch.parse_args(argv)
    if spec.get("mode") == "lora_keys":
        return lora_keys_worker(tm, args)
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=args.mixed_precision)
    sample = torch.load(spec["sample"])

    class One(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return sample

    dl = torch.utils.data.DataLoader(One(), batch_size=1, shuffle=False)
    res = {}
    if spec.get("checkpoint0"):
        os.makedirs(os.path.join(out_dir, "checkpoint-0"), exist_ok=True)
        shutil.copy(spec["checkpoint0"], os.path.join(out_dir, "checkpoint-0", "pytorch_lora_weights.safetensors"))
    if spec.get("from_args"):
        torch.manual_seed(0)
        trainer = tm.Trainer.from_args(args, accelerator, logger, dl)
    else:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(spec["pretrained"], subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(spec["pretrained"], subfolder="vae")
        transformer = FluxTransformer2DModel.from_pretrained(spec["pretrained"], subfolder="transformer")
        procs = tm.get_mha_processor(args=args, device=accelerator.device)
        transformer.set_attn_processor({k: procs[1] if "transformer_blocks" in k.split(".") else
                                        procs[2] if "single_transformer_blocks" in k.split(".") else procs[0]
                                        for k in transformer.attn_processors})
        te1 = te2 = None
        if spec.get("pristine"):
            # stock trainer: dummy encoders and the zero embeddings --zero_text_embeds produces
            te1, te2 = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)

            def zero_embeds(self, prompt, device, weight_dtype, num_images_per_prompt=1, text_input_ids_list=None):
                n = len([prompt] if isinstance(prompt, str) else prompt)
                return (torch.zeros(n, self.args.max_sequence_length, TINY_FLUX["joint_attention_dim"], dtype=weight_dtype),
                        torch.zeros(n, TINY_FLUX["pooled_projection_dim"], dtype=weight_dtype),
                        torch.zeros(self.args.max_sequence_length, 3, dtype=weight_dtype))
            tm.Trainer.compute_text_embeddings = zero_embeds
            # stock get_sigmas defaults to device="cuda:0" (trainer.py:410), the patch passes the device
            stock_sigmas = tm.Trainer.get_sigmas
            tm.Trainer.get_sigmas = lambda self, t, n_dim=4, device="cpu", dtype=torch.float32: \
                stock_sigmas(self, t, n_dim=n_dim, device="cpu", dtype=dtype)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        try:
            trainer = tm.Trainer(scheduler, vae, te1, None, te2, None, transformer, args, accelerator, logger, dl)
        except Exception as e:
            if spec.get("mode") != "params":
                raise
            return {"trainable": [n for n, p in transformer.named_parameters() if p.requires_grad],
                    "error": f"{type(e).__name__}: {e}"}

    res["trainable"] = [n for n, p in trainer.transformer.named_parameters() if p.requires_grad]
    xe = trainer.transformer.x_embedder
    w = xe.modules_to_save["default"].weight if hasattr(xe, "modules_to_save") else xe.weight
    base = FluxTransformer2DModel.from_pretrained(spec["pretrained"], subfolder="transformer").x_embedder.weight
    res["x_embedder_zero"] = bool((w == 0).all())
    res["x_embedder_is_base"] = bool(torch.equal(w.detach().float(), base.detach().float()))
    if spec.get("checkpoint0"):
        from safetensors.torch import load_file
        sd = load_file(spec["checkpoint0"])
        k = "transformer_blocks.0.attn.to_q.lora_B"
        res["lora_loaded"] = bool(torch.equal(dict(trainer.transformer.named_parameters())[f"{k}.default.weight"].float(),
                                              sd[f"transformer.{k}.weight"].float()))
    if spec.get("mode") == "params":
        return res

    rec = {"calls": []}

    def pre(mod, a, kw):
        ids = kw["img_ids"]
        rec["calls"].append({"n_tokens": int(kw["hidden_states"].shape[1]), "ids_dtype": str(ids.dtype),
                             "n_glyph": int(getattr(trainer, "last_n_glyph", 0))})
        rec["ids"] = ids.detach().float().cpu()
        if spec.get("mode") == "ids":
            raise StopForward
        ng = int(getattr(trainer, "last_n_glyph", 0))
        if spec.get("poison_glyph_inputs") and ng:
            hs = kw["hidden_states"].clone()
            hs[:, -ng:] = 30.0
            return a, dict(kw, hidden_states=hs)

    def post(mod, a, kw, output):
        ng = int(getattr(trainer, "last_n_glyph", 0))
        if spec.get("poison_glyph_outputs") and ng:
            o = output[0].clone()
            o[:, -ng:] = 1e4
            return (o,) + tuple(output[1:])
        return output

    trainer.transformer.register_forward_pre_hook(pre, with_kwargs=True)
    trainer.transformer.register_forward_hook(post, with_kwargs=True)
    losses = []
    orig_backward = accelerator.backward

    def backward(loss, **kw):
        losses.append(float(loss.detach()))
        return orig_backward(loss, **kw)
    accelerator.backward = backward
    lora = [(n, p) for n, p in trainer.transformer.named_parameters() if p.requires_grad and "lora_B" in n]
    before = lora[0][1].detach().clone()
    try:
        trainer.train(accelerator, logger)
    except StopForward:
        pass
    except AttributeError as e:
        # the stock trainer ends with args.validation_config, which launch.py never defines
        # (trainer.py:1158). The LoRA is saved and the step done before it, so record and go on
        if not (spec.get("pristine") and "validation_config" in str(e)):
            raise
        res["end_error"] = f"AttributeError: {e}"
    res.update(losses=losses, calls=rec["calls"], lora_changed=bool((lora[0][1].detach() - before).abs().sum() > 0))
    if "ids" in rec:
        torch.save(rec["ids"], os.path.join(out_dir, "img_ids.pt"))
    return res


def run_worker(tmp_path, repo, pretrained, sample, **spec):
    out_dir = tmp_path / f"run_{len(list(tmp_path.glob('run_*')))}"
    out_dir.mkdir()
    spec = dict(spec, repo=repo, pretrained=pretrained, sample=str(sample), out_dir=str(out_dir))
    sp = out_dir / "spec.json"
    sp.write_text(json.dumps(spec))
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""), USE_TF="0")
    r = subprocess.run([sys.executable, THIS, "worker", str(sp)], capture_output=True, text=True, env=env,
                       cwd=repo, timeout=1200)
    assert r.returncode == 0, f"worker failed\nSTDOUT:\n{r.stdout[-3000:]}\nSTDERR:\n{r.stderr[-6000:]}"
    res = json.loads((out_dir / "result.json").read_text())
    res["out_dir"] = str(out_dir)
    res["stderr"] = r.stderr
    return res


@pytest.fixture(scope="session")
def sample512(synth, tmp_path_factory):
    random.seed(3)
    s = fd.CFIFluxDataset(synth)[0]
    p = tmp_path_factory.mktemp("sample") / "s512.pt"
    torch.save(s, str(p))
    return p


@pytest.fixture(scope="session")
def sample_real(tmp_path_factory):
    _need_real()
    ds = fd.CFIFluxDataset(REAL_DATA)
    i = max(range(len(ds)), key=lambda k: len(json.loads(ds.load_text_json(*ds.samples[k]) or '{"items": []}')["items"]))
    random.seed(4)
    s = ds[i]
    p = tmp_path_factory.mktemp("sample") / "real512.pt"
    torch.save(s, str(p))
    return p


@pytest.mark.parametrize("mp", ["no", "bf16"])
def test_regression_glyph_off_reproduces_pristine_loss(mp, tmp_path, patched, pretrained, sample512):
    """(a) glyph off, fp32 ids: the patched step gives the pristine trainer's loss on the same batch."""
    a = run_worker(tmp_path, PRISTINE, pretrained, sample512, pristine=True, mixed_precision=mp)
    b = run_worker(tmp_path, patched, pretrained, sample512, mixed_precision=mp, argv=["--zero_text_embeds"])
    print(f"[{mp}] pristine loss {a['losses']}, patched loss {b['losses']}")
    assert len(a["losses"]) == len(b["losses"]) == 1
    assert math.isfinite(a["losses"][0])
    assert b["losses"][0] == pytest.approx(a["losses"][0], rel=1e-6, abs=1e-7)
    assert a["calls"][0]["n_tokens"] == b["calls"][0]["n_tokens"]
    assert a["calls"][0]["ids_dtype"] == ("torch.bfloat16" if mp == "bf16" else "torch.float32")
    assert b["calls"][0]["ids_dtype"] == "torch.float32"
    # peft 0.15.2: the stock modules_to_save x_embedder copy is never trainable, so both train the same LoRA
    assert a["trainable"] == b["trainable"] and not any("x_embedder" in n for n in a["trainable"])
    assert "validation_config" in a.get("end_error", "") and "end_error" not in b


def test_x_embedder_off_by_default_and_lora_layers_fixed(tmp_path, patched, pretrained, sample512):
    d = run_worker(tmp_path, patched, pretrained, sample512, mode="params", from_args=True,
                   argv=["--zero_text_embeds"])
    assert d["trainable"] and not any("x_embedder" in n for n in d["trainable"])
    e = run_worker(tmp_path, patched, pretrained, sample512, mode="params", from_args=True,
                   argv=["--zero_text_embeds", "--train_x_embedder"])
    assert sorted(n for n in e["trainable"] if "x_embedder" in n) == [
        "x_embedder.modules_to_save.default.bias", "x_embedder.modules_to_save.default.weight"]
    f = run_worker(tmp_path, patched, pretrained, sample512, mode="params", from_args=True,
                   argv=["--zero_text_embeds", "--lora_layers", "attn.to_q,attn.to_k"])
    assert f["trainable"] and all(("to_q" in n or "to_k" in n or "x_embedder" in n) for n in f["trainable"])
    g = run_worker(tmp_path, PRISTINE, pretrained, sample512, mode="params", pristine=True,
                   argv=["--lora_layers", "attn.to_q,attn.to_k"])
    print(f"pristine --lora_layers: {len(g['trainable'])} trainable tensors, error: {g.get('error')}")
    assert g["trainable"] == []                        # the original bug: no adapter at all


def test_warm_start_drops_released_x_embedder_unless_trained(tmp_path, patched, pretrained, sample512):
    """checkpoint-0 = a LoRA plus x_embedder keys (the released UniTEX files have them). Default: the
    LoRA loads and the base x_embedder is kept (peft 0.15.2 would otherwise overwrite the frozen
    base weight). --train_x_embedder: the keys load into the trainable copy."""
    from safetensors.torch import load_file, save_file
    first = run_worker(tmp_path, patched, pretrained, sample512, argv=["--zero_text_embeds"])
    sd = load_file(os.path.join(first["out_dir"], "pytorch_lora_weights.safetensors"))
    assert sd and not any("x_embedder" in k for k in sd)
    inner = TINY_FLUX["num_attention_heads"] * TINY_FLUX["attention_head_dim"]
    sd["transformer.x_embedder.weight"] = torch.zeros(inner, 64)
    sd["transformer.x_embedder.bias"] = torch.zeros(inner)
    ckpt = tmp_path / "released_like.safetensors"
    save_file(sd, str(ckpt))
    resume = ["--zero_text_embeds", "--resume_from_checkpoint", "latest"]
    d = run_worker(tmp_path, patched, pretrained, sample512, mode="params", from_args=True, checkpoint0=str(ckpt), argv=resume)
    assert d["lora_loaded"] and d["x_embedder_is_base"] and not d["x_embedder_zero"]
    assert "ignoring ['x_embedder.bias', 'x_embedder.weight']" in d["stderr"]
    e = run_worker(tmp_path, patched, pretrained, sample512, mode="params", from_args=True, checkpoint0=str(ckpt),
                   argv=resume + ["--train_x_embedder"])
    assert e["lora_loaded"] and e["x_embedder_zero"]


def _glyph_argv(stage="gt"):
    p = {"gt": "[[\"inf\", 1, 0, 0]]", "fixed": "[[\"inf\", 0, 0, 1]]"}[stage]
    return ["--zero_text_embeds", "--glyph", "--glyph_config", '{"stages": ' + p + ', "p_all_drop": 0.0, "p_item_drop": 0.0}']


def test_glyph_step_runs_and_glyph_tokens_skip_the_loss(tmp_path, patched, pretrained, sample_real):
    """(b) glyph on: full step, finite loss, sequence grows by Ng, fp32 ids with axis0 = 1 on the
    glyph tokens, and the glyph token outputs do not enter the loss."""
    off = run_worker(tmp_path, patched, pretrained, sample_real, argv=["--zero_text_embeds"])
    on = run_worker(tmp_path, patched, pretrained, sample_real, argv=_glyph_argv())
    poisoned = run_worker(tmp_path, patched, pretrained, sample_real, argv=_glyph_argv(), poison_glyph_outputs=True)
    c_off, c_on = off["calls"][0], on["calls"][0]
    ng = c_on["n_glyph"]
    print(f"glyph off: {c_off['n_tokens']} tokens, loss {off['losses']} | glyph on: +{ng} tokens, "
          f"loss {on['losses']} | poisoned glyph outputs: loss {poisoned['losses']}")
    assert ng > 0 and c_on["n_tokens"] == c_off["n_tokens"] + ng
    assert c_on["ids_dtype"] == "torch.float32"
    ids = torch.load(os.path.join(on["out_dir"], "img_ids.pt"))
    assert (ids[-ng:, 0] == 1).all() and (ids[:-ng, 0] == 0).all()
    assert ids[-ng:, 1].max() <= 31 and ids[-ng:, 2].max() <= 191
    assert math.isfinite(on["losses"][0]) and on["lora_changed"]
    assert poisoned["losses"][0] == pytest.approx(on["losses"][0], rel=1e-6)
    assert os.path.isfile(os.path.join(on["out_dir"], "glyph_config.json"))


def test_text_keep_boost(tmp_path, patched, pretrained, sample_real):
    z = run_worker(tmp_path, patched, pretrained, sample_real, argv=["--zero_text_embeds", "--text_keep_boost", "0"])
    one = run_worker(tmp_path, patched, pretrained, sample_real, argv=["--zero_text_embeds", "--text_keep_boost", "1"])
    ids = torch.load(os.path.join(one["out_dir"], "img_ids.pt"))
    s = torch.load(sample_real)
    m = gt.text_token_mask(s["text_json"], 512).reshape(32, 192)
    tgt = ids[(ids[:, 0] == 0) & (ids[:, 1] < 32)]
    kept = torch.zeros(32, 192, dtype=torch.bool)
    kept[tgt[:, 1].long(), tgt[:, 2].long()] = True
    assert m.sum() > 0 and bool(kept[m].all())         # every text token kept at boost 1
    assert one["calls"][0]["n_tokens"] > z["calls"][0]["n_tokens"]


@pytest.mark.skipif(not SLOW, reason="FLUX_TEST_SLOW=0")
def test_view_resolution_1024_strip_and_ids(tmp_path, patched, pretrained, synth):
    """(c) 1024 px per view: 1024 x 6144 strip, target cols 0..383, reference 384..447, all id
    triples unique in fp32 (bf16 would merge them), glyph ids inside their 64-token slots."""
    random.seed(3)
    s = fd.CFIFluxDataset(synth, view_res=1024)[0]
    assert s["rgbs"].shape == (3, 1024, 6144) and s["rgbs_ip"].shape == (3, 1024, 1024)
    p = tmp_path / "s1024.pt"
    torch.save(s, str(p))
    r = run_worker(tmp_path, patched, pretrained, p, mode="ids", no_drop=True,
                   argv=["--view_resolution", "1024", "--resolution", "1024", "6144"] + _glyph_argv("fixed"))
    ids = torch.load(os.path.join(r["out_dir"], "img_ids.pt"))
    ng = r["calls"][0]["n_glyph"]
    base = ids[:-ng] if ng else ids
    assert r["calls"][0]["ids_dtype"] == "torch.float32"
    assert len(base) == 64 * 384 * 2 + 64 * 64
    tgt, ctrl, ref = base[:64 * 384], base[64 * 384:2 * 64 * 384], base[2 * 64 * 384:]
    assert tgt[:, 1].max() == 63 and tgt[:, 2].max() == 383
    assert ctrl[:, 1].min() == 64 and ctrl[:, 1].max() == 127 and ctrl[:, 2].max() == 383
    assert ref[:, 1].min() == 64 and ref[:, 1].max() == 127 and ref[:, 2].min() == 384 and ref[:, 2].max() == 447
    assert len(torch.unique(base, dim=0)) == len(base)
    bf = base.to(torch.bfloat16).float()
    n_bf16 = len(base) - len(torch.unique(bf, dim=0))
    print(f"1024: {len(base)} image tokens + {ng} glyph tokens, max col {int(base[:, 2].max())}, "
          f"bf16 would merge {n_bf16} ids")
    assert n_bf16 > 0
    g = ids[-ng:]
    assert ng > 0 and (g[:, 0] == 1).all() and g[:, 1].max() <= 63 and g[:, 2].max() <= 383


@pytest.mark.skipif(not SLOW, reason="FLUX_TEST_SLOW=0")
def test_launch_py_end_to_end(tmp_path, patched, pretrained):
    """(e) accelerate launch launch.py, 2 optimizer steps on the real LMDB renders with glyphs, then a
    warm start from checkpoint-0 holding a LoRA file with x_embedder keys (like the released one)."""
    _need_real()
    mdb = REAL_DATA + "_mdb" if os.path.isdir(REAL_DATA + "_mdb") else REAL_DATA
    common = ["--pretrained_model_name_or_path", pretrained, "--dataset_impl", "cfi", "--cfi_root", REPO,
              "--dataset_name_list", mdb, "--use_complex_dataset", "--six_views_or_four_views", "--dual_image",
              "--control_image", "--both_ccm_normal_condition", "--mixed_precision", "no",
              "--resolution", "512", "3072", "--n_rows", "1", "--n_cols", "6", "--lora_rank", "4",
              "--lora_alpha", "4", "--optimizer", "prodigy", "--learning_rate", "1.0", "--guidance_scale", "1.0",
              "--max_train_steps", "2", "--validation_steps", "10000000", "--checkpointing_steps", "1",
              "--train_batch_size", "1", "--gradient_accumulation_steps", "1", "--gradient_checkpointing",
              "--random_drop_noise", "--random_drop_noise_probability", "0.75", "--random_drop_condition",
              "--random_drop_condition_probability", "0.25", "--report_to", "all", "--tasks", "texturing",
              "--seed", "666", "--zero_text_embeds", "--glyph", "--text_keep_boost", "0.5"]
    env = dict(os.environ, USE_TF="0")
    env.pop("CAUGHT_ALL_EXCEPTIONS", None)          # the patch imports zipfile, so this is not needed
    launcher = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", "1", "--num_machines", "1",
                "--mixed_precision", "no", "--dynamo_backend", "no", "--cpu", "launch.py"]
    out1 = tmp_path / "e2e1"
    r = subprocess.run(launcher + common + ["--output_dir", str(out1)], cwd=patched, env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-6000:]
    from safetensors.torch import load_file, save_file
    for d in ("checkpoint-1", "checkpoint-2", "."):
        f = out1 / d / "pytorch_lora_weights.safetensors"
        assert f.is_file(), f
    sd = load_file(str(out1 / "pytorch_lora_weights.safetensors"))
    assert sd and not any("x_embedder" in k for k in sd)
    assert (out1 / "glyph_config.json").is_file()
    # warm start: checkpoint-0 = trained LoRA + an x_embedder like the released UniTEX files
    out2 = tmp_path / "e2e2"
    (out2 / "checkpoint-0").mkdir(parents=True)
    inner = TINY_FLUX["num_attention_heads"] * TINY_FLUX["attention_head_dim"]
    sd["transformer.x_embedder.weight"] = torch.zeros(inner, 64)
    sd["transformer.x_embedder.bias"] = torch.zeros(inner)
    save_file(sd, str(out2 / "checkpoint-0" / "pytorch_lora_weights.safetensors"))
    r2 = subprocess.run(launcher + common + ["--output_dir", str(out2), "--resume_from_checkpoint", "latest",
                                             "--gradient_accumulation_steps", "2"],
                        cwd=patched, env=env, capture_output=True, text=True, timeout=1800)
    assert r2.returncode == 0, r2.stderr[-6000:]
    log2 = r2.stdout + r2.stderr
    assert "Resuming from checkpoint checkpoint-0" in log2
    assert "ignoring ['x_embedder.bias', 'x_embedder.weight']" in log2
    assert (out2 / "checkpoint-2" / "pytorch_lora_weights.safetensors").is_file()
    unexpected = [ln for ln in log2.splitlines() if "unexpected" in ln.lower()]
    print("e2e run 1 tail:\n" + "\n".join((r.stderr or r.stdout).splitlines()[-3:]))
    print("e2e run 2 (warm start, accumulation 2) tail:\n" + "\n".join(r2.stderr.splitlines()[-3:]))
    print("warm start unexpected-key lines:", [ln[:200] for ln in unexpected])

    # stock dataset path: builds now (cond_image_type was a TypeError), then stops at the first key
    # the shipped PBRTextureGenerationDataset does not provide (trainer.py:835), as documented
    stock = [a for a in common if a not in ("--glyph", "--dataset_impl", "cfi", "--text_keep_boost", "0.5")]
    r3 = subprocess.run(launcher + stock + ["--dataset_impl", "unitex", "--output_dir", str(tmp_path / "e2e3")],
                        cwd=patched, env=dict(env, CAUGHT_ALL_EXCEPTIONS="1"), capture_output=True, text=True,
                        timeout=1800)
    assert r3.returncode != 0 and "KeyError: 'alphas'" in r3.stderr and "cond_image_type" not in r3.stderr, r3.stderr[-3000:]


# ────────────────────────────────────────────────────────────────────────────
# Adversarial review tests (riskiest assumptions)
# ────────────────────────────────────────────────────────────────────────────

RELEASED_LORA_HEADER = os.environ.get(
    "UNITEX_LORA_HEADER", "/Users/test/.claude/jobs/5ee54a5d/tmp/scratch_unitex-inference/mv_lora_weights.hdr.json")
FLUX_DEV = dict(patch_size=1, in_channels=64, num_layers=19, num_single_layers=38, attention_head_dim=128,
                num_attention_heads=24, joint_attention_dim=4096, pooled_projection_dim=768,
                guidance_embeds=True, axes_dims_rope=(16, 56, 56))


def lora_keys_worker(tm, args):
    """Run the patched Trainer.add_LORA on a FLUX.1-dev shaped skeleton (meta weights) and return
    the peft state dict keys and shapes it would save / load."""
    import types
    from accelerate import init_empty_weights
    from diffusers import FluxTransformer2DModel
    from peft.utils import get_peft_model_state_dict
    with init_empty_weights():
        tr = FluxTransformer2DModel(**FLUX_DEV)
    tr.requires_grad_(False)
    fake = types.SimpleNamespace(args=args, transformer=tr, load_LoRA_from_checkpoint=lambda *a: None)
    tm.Trainer.add_LORA(fake, None, None)
    sd = get_peft_model_state_dict(tr)
    return {"keys": {k: list(v.shape) for k, v in sd.items()},
            "trainable": sorted(n for n, p in tr.named_parameters() if p.requires_grad and "lora" not in n)}


def test_adv_lora_keys_match_released_unitex_lora(tmp_path, patched, pretrained, sample512):
    """Warm start: the patched adapter on a FLUX.1-dev shaped model saves exactly the 684 LoRA
    tensors (names and shapes) of the released mv_lora_weights.safetensors, minus its two
    x_embedder tensors, which --train_x_embedder adds back with the released shapes."""
    if not os.path.isfile(RELEASED_LORA_HEADER):
        pytest.skip(f"no released LoRA safetensors header at {RELEASED_LORA_HEADER}")
    hdr = json.load(open(RELEASED_LORA_HEADER))
    rel = {k[len("transformer."):]: v["shape"] for k, v in hdr.items() if k != "__metadata__"}
    rank = ["--lora_rank", "16", "--lora_alpha", "16"]
    d = run_worker(tmp_path, patched, pretrained, sample512, mode="lora_keys", argv=["--zero_text_embeds"] + rank)
    lora_rel = {k: v for k, v in rel.items() if "x_embedder" not in k}
    print(f"released: {len(rel)} tensors ({len(lora_rel)} LoRA), patched adapter saves {len(d['keys'])}")
    assert len(lora_rel) == 684 and d["keys"] == lora_rel
    assert d["trainable"] == []
    e = run_worker(tmp_path, patched, pretrained, sample512, mode="lora_keys",
                   argv=["--zero_text_embeds", "--train_x_embedder"] + rank)
    assert e["keys"] == rel
    assert e["trainable"] == ["x_embedder.modules_to_save.default.bias", "x_embedder.modules_to_save.default.weight"]


def test_adv_regression_without_conditions(tmp_path, patched, pretrained, sample512):
    """No --dual_image / --control_image: the stock trainer does not slice model_pred at all
    (trainer.py:1030-1031), the patch slices to n_noise. Same loss and token count both ways."""
    no_cond = ["--dual_image", "--control_image", "--both_ccm_normal_condition"]
    a = run_worker(tmp_path, PRISTINE, pretrained, sample512, pristine=True, drop_flags=no_cond)
    b = run_worker(tmp_path, patched, pretrained, sample512, argv=["--zero_text_embeds"], drop_flags=no_cond)
    print(f"no conditions: pristine loss {a['losses']} ({a['calls'][0]['n_tokens']} tokens), "
          f"patched loss {b['losses']} ({b['calls'][0]['n_tokens']} tokens)")
    assert a["calls"][0]["n_tokens"] == b["calls"][0]["n_tokens"] < 32 * 192
    assert b["losses"][0] == pytest.approx(a["losses"][0], rel=1e-6, abs=1e-7)


def test_adv_glyph_inputs_reach_the_loss_through_attention(tmp_path, patched, pretrained, sample_real):
    """The converse of the output-poisoning check: glyph token INPUTS must change the target
    prediction (they are attended, not inert), while their outputs stay out of the loss."""
    on = run_worker(tmp_path, patched, pretrained, sample_real, argv=_glyph_argv())
    hit = run_worker(tmp_path, patched, pretrained, sample_real, argv=_glyph_argv(), poison_glyph_inputs=True)
    ng = on["calls"][0]["n_glyph"]
    print(f"glyph inputs: clean loss {on['losses']}, glyph inputs set to 30: loss {hit['losses']} ({ng} tokens)")
    assert ng > 0 and hit["calls"][0]["n_glyph"] == ng
    # the tiny random FLUX attends weakly (glyph on vs off moves the loss by about 5e-5), but runs
    # are bit-reproducible (the output-poisoning test matches to every digit), so any change counts
    assert abs(hit["losses"][0] - on["losses"][0]) > 1e-6 * abs(on["losses"][0])


@pytest.mark.skipif(not SLOW, reason="FLUX_TEST_SLOW=0")
def test_adv_dataset_resize_paths_match_unitex_flux(synth):
    """Renders at another size than view_res go through the resize branches (bilinear +
    antialias for rgb / albedo / alpha, nearest for nocs / normal; ours resizes the reference
    alone, datasets.py resizes all 20 at once). Upsample 512 -> 1024 on synthetic renders and
    downsample 1024 -> 512 on the real 1024 renders must still be bit-identical."""
    _need_pristine()
    D = _import_pristine_datasets()
    cases = [(synth, 1024)]
    if os.path.isfile(os.path.join(REAL_DATA_1024, "training_uid.json")):
        cases.append((REAL_DATA_1024, 512))
    for root, res in cases:
        ours = fd.CFIFluxDataset(root, view_res=res, image_ext=".png")
        ref = _pristine_sample(D, root, 0, ".png", seed=21, res=res)
        random.seed(21)
        s = ours[0]
        assert s["ref_index"] == ref["ref_index"]
        for k in ("rgbs", "albedos", "alphas", "ccms", "native_normals", "rgbs_ip", "albedos_ip", "alphas_ip"):
            assert s[k].shape == ref[k].shape, (root, res, k)
            assert torch.equal(s[k], ref[k]), (root, res, k, float((s[k] - ref[k]).abs().max()))
    print(f"resize paths bit-identical: {[(os.path.basename(r), res) for r, res in cases]}")


# ────────────────────────────────────────────────────────────────────────────
# Worker entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["worker"])
    ap.add_argument("spec")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    result = worker(spec)
    with open(os.path.join(spec["out_dir"], "result.json"), "w") as f:
        json.dump(result, f)
