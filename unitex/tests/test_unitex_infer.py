"""
Tests for the UniTEX inference GlyphAnchor integration on CPU: unitex/patches/UniTEX.patch,
unitex/run_unitex.py glyph / view-res options and the run_eval.sh anchors step.

Setup (session fixtures):
  repos     pipeline.py, flux_piplines/texturing/pipeline.py, LTM/rgb_field.py of UniTEX affa1e2 ($UNITEX_ROOT),
            copied twice: pristine, and patched with `git apply` (so the patch file itself is tested)
  tiny      tiny random FluxTransformer2DModel (1 + 1 blocks, 2 heads of 16, rope (4, 6, 6)) with
            UniTEX's text widths (T5 4096, pooled CLIP 768, the zeros UniTEX feeds), tiny
            AutoencoderKL (16 latent channels, FLUX scaling / shift), FLUX scheduler, all bf16 like
            UniTEX. Pristine and patched PBRFluxPipeline share these modules.
  eval2     2 SKUs of the 28-SKU eval set ($CFI_EVAL28) copied to tmp (never written in place),
            with a photo-lift text.json made by `python -m unitex.anchors --geometry unitex` from
            the stock run's cache grids
The full UniTEX class (CustomRGBTextureFullPipeline) needs nvdiffrast, LTM and RMBG. Its three
classes are exec'd from the patched / pristine pipeline.py source with those pieces stubbed, so
infer_mv, preprocess_reference_image, render_geometry_images and __call__ run for real.

Run: python -m pytest unitex/tests/test_unitex_infer.py -v      (INFER_TEST_SLOW=0 skips 1024)
"""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.join(REPO, "unitex"))
from unitex import glyph as gl
from unitex import glyph_ids as gid
from unitex import glyph_tokens as gt
import run_unitex  # noqa: E402  (unitex/run_unitex.py, a script)

UNITEX = os.environ.get("UNITEX_ROOT", "/Users/test/.claude/jobs/5ee54a5d/tmp/UniTEX")
EVAL28 = os.environ.get("CFI_EVAL28", "/Users/test/CyLab/unitex_eval_28")
EVAL_SKUS = ("016000263192", "031146270606")
PATCH = os.path.join(REPO, "unitex", "patches", "UniTEX.patch")
PATCHED_FILES = ("pipeline.py", "flux_piplines/texturing/pipeline.py", "LTM/rgb_field.py")
SLOW = os.environ.get("INFER_TEST_SLOW", "1") != "0"
PY = sys.executable

TINY_FLUX = dict(patch_size=1, in_channels=64, num_layers=1, num_single_layers=1, attention_head_dim=16,
                 num_attention_heads=2, joint_attention_dim=4096, pooled_projection_dim=768,
                 guidance_embeds=True, axes_dims_rope=(4, 6, 6))
TINY_VAE = dict(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 8, 8),
                layers_per_block=1, latent_channels=16, norm_num_groups=4, scaling_factor=0.3611,
                shift_factor=0.1159, use_quant_conv=False, use_post_quant_conv=False,
                mid_block_add_attention=False)
FLUX_SCHEDULER = dict(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True, base_shift=0.5,
                      max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures: pristine / patched UniTEX, tiny FLUX, eval copy with text.json
# ────────────────────────────────────────────────────────────────────────────

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def repos(tmp_path_factory):
    if not all(os.path.isfile(os.path.join(UNITEX, f)) for f in PATCHED_FILES):
        pytest.skip(f"UniTEX checkout not found at {UNITEX} (set UNITEX_ROOT)")
    out = {}
    for kind in ("pristine", "patched"):
        d = tmp_path_factory.mktemp(kind) / "UniTEX"
        for f in PATCHED_FILES:
            os.makedirs(d / os.path.dirname(f), exist_ok=True)
            shutil.copy(os.path.join(UNITEX, f), d / f)
        if kind == "patched":
            r = subprocess.run(["git", "apply", "--verbose", PATCH], cwd=d, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        out[kind] = str(d)
        out[kind + "_flux"] = _load(str(d / PATCHED_FILES[1]), f"unitex_flux_{kind}")
    return out


@pytest.fixture(scope="session")
def tiny():
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    torch.manual_seed(0)
    tr = FluxTransformer2DModel(**TINY_FLUX).to(torch.bfloat16).eval()
    vae = AutoencoderKL(**TINY_VAE).to(torch.bfloat16).eval()
    return tr, vae, FlowMatchEulerDiscreteScheduler(**FLUX_SCHEDULER)


@pytest.fixture(scope="session")
def pipes(repos, tiny):
    tr, vae, sch = tiny
    return {k: repos[k + "_flux"].PBRFluxPipeline(sch, vae, None, None, None, None, tr)
            for k in ("pristine", "patched")}


def _copy_eval(dst):
    os.makedirs(dst, exist_ok=True)
    for s in EVAL_SKUS:
        src = os.path.join(EVAL28, s)
        os.makedirs(os.path.join(dst, s, "unitex_s63", "cache"), exist_ok=True)
        for f in ("ref.png", "mesh.glb", "ref_meta.json", "gt_ocr.json", "gt_text.txt"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, s, f))
        shutil.copy(os.path.join(src, "unitex_s63", "rmbg_mask_1024.png"), os.path.join(dst, s, "unitex_s63"))
        for f in ("mv_alpha.png", "mv_ccm.png", "mv_normal.png", "mv_rgb.png"):
            shutil.copy(os.path.join(src, "unitex_s63", "cache", f), os.path.join(dst, s, "unitex_s63", "cache"))
    with open(os.path.join(dst, "skus.txt"), "w") as f:
        f.write("\n".join(EVAL_SKUS) + "\n")


@pytest.fixture(scope="session")
def eval2(tmp_path_factory):
    if not all(os.path.isfile(os.path.join(EVAL28, s, "unitex_s63", "cache", "mv_ccm.png")) for s in EVAL_SKUS):
        pytest.skip(f"eval set with UniTEX caches not found at {EVAL28} (set CFI_EVAL28)")
    d = str(tmp_path_factory.mktemp("eval2") / "eval")
    _copy_eval(d)
    for s in EVAL_SKUS:
        r = subprocess.run([PY, "-m", "unitex.anchors", "--eval-dir", d, "--sku", s, "--geometry", "unitex",
                            "--run-name", "unitex_s63", "--out", os.path.join(d, s, "text.json")],
                           cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    return d


def _text(eval2, sku=EVAL_SKUS[0]):
    return os.path.join(eval2, sku, "text.json")


def _inputs(R, seed=0):
    rng = np.random.default_rng(seed)
    ctrl = Image.fromarray(rng.integers(0, 255, (R, 6 * R, 3), dtype=np.uint8))
    dual = Image.fromarray(rng.integers(0, 255, (R, R, 3), dtype=np.uint8))
    return ctrl, dual


def _run(pipe, R, seed=0, steps=2, **kw):
    """One UniTEX texture-pass call (infer_mv arguments) at R px per view."""
    ctrl, dual = _inputs(R)
    out = pipe(prompt="[MVFLUX]", control_image=ctrl, dual_image=dual, prompt_embeds=None,
               pooled_prompt_embeds=None, height=R, width=6 * R, n_rows=1, n_cols=6,
               num_inference_steps=steps, guidance_scale=3.5, max_sequence_length=512,
               generator=torch.Generator().manual_seed(seed), output_type="np", **kw)
    return out.images


class Capture:
    """Records the transformer inputs of every step (optionally aborts after the first)."""

    class Stop(Exception):
        pass

    def __init__(self, transformer, stop=False):
        self.tr, self.stop, self.calls = transformer, stop, []

    def __enter__(self):
        def hook(module, args, kwargs):
            self.calls.append({k: kwargs[k].detach().clone() for k in ("hidden_states", "img_ids", "txt_ids")})
            if self.stop:
                raise Capture.Stop()
        self.h = self.tr.register_forward_pre_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.h.remove()
        return exc[0] is Capture.Stop


def _n_tokens(R):
    """(noise, control, dual) token counts of the UniTEX texture pass at R px per view."""
    t = R // 16
    return t * 6 * t, t * 6 * t, t * t


# ────────────────────────────────────────────────────────────────────────────
# Patch file
# ────────────────────────────────────────────────────────────────────────────

def test_patch_applies_reverses_and_touches_three_files(repos):
    files = [ln.split()[2][2:] for ln in open(PATCH) if ln.startswith("diff --git")]
    assert files == ["LTM/rgb_field.py", "flux_piplines/texturing/pipeline.py", "pipeline.py"]
    # setup_unitex.sh detects an applied patch with `git apply --reverse --check`
    r = subprocess.run(["git", "apply", "--reverse", "--check", PATCH], cwd=repos["patched"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in PATCHED_FILES:
        compile(open(os.path.join(repos["patched"], f)).read(), f, "exec")


# ────────────────────────────────────────────────────────────────────────────
# (a) glyphs off: identical to the unpatched pipeline
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("R", [64, 512])
def test_a_no_glyphs_bitwise_equal_to_pristine(pipes, tiny, R):
    tr = tiny[0]
    with Capture(tr) as cp:
        ref = _run(pipes["pristine"], R)
    with Capture(tr) as cn:
        out = _run(pipes["patched"], R)
    assert out.shape == ref.shape == (1, R, 6 * R, 3)
    assert np.array_equal(out, ref)
    # empty glyph list (a SKU whose text.json has no usable item) is the same as no glyphs
    assert np.array_equal(_run(pipes["patched"], R, glyph_instances=[]), ref)
    assert pipes["patched"]._last_n_glyph_tokens == 0
    # ids: same values, now fp32 (stock: bf16, exact for these values <= 256)
    for a, b in zip(cp.calls, cn.calls):
        assert a["img_ids"].dtype == torch.bfloat16 and b["img_ids"].dtype == torch.float32
        assert torch.equal(a["img_ids"].float(), b["img_ids"]) and torch.equal(a["hidden_states"], b["hidden_states"])
        assert b["txt_ids"].dtype == torch.float32


# ────────────────────────────────────────────────────────────────────────────
# (b) glyphs on: Ng more clean tokens on the glyph plane, stripped before decoding
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,kind", [("center", "fixed"), ("stretch", "box"), ("warp", "fixed")])
def test_b_glyph_tokens_appended_clean_every_step_and_stripped(pipes, tiny, eval2, mode, kind):
    tr, vae, _ = tiny
    R = 512
    cfg = gl.GlyphConfig(anchor_mode=mode, infer_kind=kind)
    insts = gt.build_infer_glyphs(_text(eval2), cfg, view_res=R)
    lat, ids = gt.encode_glyphs(insts, vae, sample_mode="argmax")
    Ng = lat.shape[1]
    assert Ng == gl.total_tokens(insts) > 0
    n_noise, n_ctrl, n_dual = _n_tokens(R)
    with Capture(tr) as c0:
        base = _run(pipes["patched"], R)
    with Capture(tr) as c1:
        out = _run(pipes["patched"], R, glyph_instances=insts)
    assert pipes["patched"]._last_n_glyph_tokens == Ng
    assert out.shape == base.shape == (1, R, 6 * R, 3) and np.isfinite(out).all()
    assert not np.array_equal(out, base)                    # the glyphs do reach the target
    for s, (a, b) in enumerate(zip(c0.calls, c1.calls)):
        assert a["hidden_states"].shape[1] == n_noise + n_ctrl + n_dual
        assert b["hidden_states"].shape[1] == a["hidden_states"].shape[1] + Ng
        # clean glyph latents re-inserted at every step, after control + reference
        assert torch.equal(b["hidden_states"][:, -Ng:], lat.to(b["hidden_states"].dtype))
        assert torch.equal(b["hidden_states"][:, n_noise:-Ng], a["hidden_states"][:, n_noise:])
        assert torch.equal(b["img_ids"][:-Ng], a["img_ids"]) and torch.equal(b["img_ids"][-Ng:], ids)
        assert b["img_ids"].dtype == torch.float32 and (b["img_ids"][-Ng:, 0] == 1).all()
        if s == 0:   # posterior mode: the pipeline generator draws the same noise as without glyphs
            assert torch.equal(b["hidden_states"][:, :n_noise], a["hidden_states"][:, :n_noise])
    # glyph ids sit on the target strip token grid (rows < R/16, cols < 6R/16)
    assert ids[:, 1].min() >= 0 and ids[:, 1].max() < R // 16
    assert ids[:, 2].min() >= 0 and ids[:, 2].max() < 6 * R // 16


def test_b_latents_and_ids_path_equals_instances_path(pipes, tiny, eval2):
    insts = gt.build_infer_glyphs(_text(eval2), gl.GlyphConfig(), view_res=512)
    lat, ids = gt.encode_glyphs(insts, tiny[1], sample_mode="argmax")
    a = _run(pipes["patched"], 512, glyph_instances=insts)
    b = _run(pipes["patched"], 512, glyph_latents=lat, glyph_ids=ids)
    assert np.array_equal(a, b)


def test_b_glyphs_only_without_other_conditions(pipes, tiny):
    """No control / reference image: the glyphs alone become the condition tokens."""
    R = 64
    lat = torch.randn(1, 6, 64, dtype=torch.bfloat16)
    ids = torch.tensor([[1, r, c] for r in range(2) for c in range(3)], dtype=torch.float32)
    with Capture(tiny[0]) as c:
        out = pipes["patched"](prompt="[MVFLUX]", height=R, width=6 * R, num_inference_steps=2,
                               generator=torch.Generator().manual_seed(0), output_type="np",
                               glyph_latents=lat, glyph_ids=ids).images
    assert out.shape == (1, R, 6 * R, 3) and np.isfinite(out).all()
    assert c.calls[0]["hidden_states"].shape[1] == _n_tokens(R)[0] + 6


def test_b_glyph_input_validation(pipes):
    p = pipes["patched"]
    lat = torch.zeros(1, 2, 64)
    bad = {
        "axis 0": dict(glyph_latents=lat, glyph_ids=torch.tensor([[0., 0, 0], [1, 0, 1]])),
        "count": dict(glyph_latents=lat, glyph_ids=torch.tensor([[1., 0, 0]])),
        "channels": dict(glyph_latents=torch.zeros(1, 1, 16), glyph_ids=torch.tensor([[1., 0, 0]])),
        "pair": dict(glyph_latents=lat),
        "both": dict(glyph_instances=[], glyph_latents=lat, glyph_ids=torch.ones(2, 3)),
        "nan": dict(glyph_latents=lat, glyph_ids=torch.tensor([[1., 0, float("nan")], [1, 0, 1]])),
    }
    for name, kw in bad.items():
        with pytest.raises(ValueError):
            p.prepare_glyph_tokens(1, torch.bfloat16, torch.device("cpu"), **kw)
    l2, i2 = p.prepare_glyph_tokens(3, torch.bfloat16, torch.device("cpu"), glyph_latents=lat[0],
                                    glyph_ids=torch.tensor([[1, 0, 0], [1, 0, 1]], dtype=torch.float64))
    assert l2.shape == (3, 2, 64) and l2.dtype == torch.bfloat16 and i2.dtype == torch.float32
    assert p.prepare_glyph_tokens(1, torch.bfloat16, torch.device("cpu")) == (None, None)


# ────────────────────────────────────────────────────────────────────────────
# (c) 1024 px per view: fp32 ids, no collisions
# ────────────────────────────────────────────────────────────────────────────

def _unique_rows(t):
    return torch.unique(t, dim=0).shape[0]


@pytest.mark.skipif(not SLOW, reason="INFER_TEST_SLOW=0")
def test_c_view_res_1024_ids_unique_in_fp32(pipes, tiny, eval2):
    tr, vae, _ = tiny
    R = 1024
    n_noise, n_ctrl, n_dual = _n_tokens(R)
    insts = gt.build_infer_glyphs(_text(eval2), gl.GlyphConfig(), view_res=R)
    gids, slices = gid.concat_ids(insts)
    Ng = len(gids)
    with Capture(tr) as c:
        out = _run(pipes["patched"], R, glyph_instances=insts)
    assert out.shape == (1, R, 6 * R, 3) and np.isfinite(out).all()
    ids = c.calls[0]["img_ids"]
    assert ids.dtype == torch.float32 and ids.shape[0] == n_noise + n_ctrl + n_dual + Ng
    base = ids[:-Ng]
    assert base.shape[0] == 53248 and _unique_rows(base) == 53248          # every image token distinct
    assert base[:, 2].max().item() == 447 and base[:, 1].max().item() == 127
    glyph = ids[-Ng:]
    assert (glyph[:, 0] == 1).all() and (base[:, 0] == 0).all()           # planes never overlap
    # glyph ids reach the transformer exactly as glyph_ids built them (no rounding on the way)
    assert torch.equal(glyph, torch.from_numpy(gids))
    assert glyph[:, 1].max() < R // 16 and glyph[:, 2].max() < 6 * R // 16
    # glyph / glyph overlaps between instances come from the layout (center-mode footprints of
    # nearby lines), not from the dtype: bf16 can only add to them
    n_g32 = gid.count_collisions(glyph.numpy(), slices)
    n_g16 = gid.count_collisions(glyph.to(torch.bfloat16).float().numpy(), slices)
    assert n_g16 >= n_g32
    # the same ids in bf16 (the stock pipeline's dtype) collide
    n_bf16 = _unique_rows(base.to(torch.bfloat16).float())
    assert n_bf16 < 53248
    # and the stock pipeline really builds them in bf16
    with Capture(tr, stop=True) as cp:
        _run(pipes["pristine"], R)
    assert cp.calls[0]["img_ids"].dtype == torch.bfloat16
    assert _unique_rows(cp.calls[0]["img_ids"].float()) == n_bf16
    n512 = gid.count_collisions(*gid.concat_ids(gt.build_infer_glyphs(_text(eval2), gl.GlyphConfig(), view_res=512)))
    print(f"\n1024: {Ng} glyph tokens, 53248 image ids unique in fp32, {n_bf16} unique in bf16; "
          f"glyph tokens sharing an id with another instance: {n_g32} fp32, {n_g16} bf16 (512 px: {n512})")


# ────────────────────────────────────────────────────────────────────────────
# UniTEX pipeline.py classes (exec'd with nvdiffrast / LTM / RMBG stubbed)
# ────────────────────────────────────────────────────────────────────────────

CLASSES = ("RGBTextureFullPipelineBase", "RGBTextureFullPipeline", "CustomRGBTextureFullPipeline")


class FakeExporter:
    def __init__(self):
        self.calls = []

    def export_condition(self, mesh_path, **kw):
        self.calls.append(kw)
        H, W = kw["H"], kw["W"]
        n_rows, n_cols = kw["n_rows"], kw["n_cols"]
        rng = np.random.default_rng(1)
        rgb = lambda: Image.fromarray(rng.integers(0, 255, (n_rows * H, n_cols * W, 3), dtype=np.uint8))
        return {"alpha": Image.new("L", (n_cols * W, n_rows * H), 255), "ccm": rgb(), "normal": rgb(),
                "c2ws": torch.eye(4).repeat(6, 1, 1), "intrinsics": torch.eye(3).repeat(6, 1, 1),
                "perspective": False}


class FakeFlux:
    """Stands in for PBRFluxPipeline: records kwargs, returns its control image."""

    def __init__(self):
        self.calls = []
        self._last_n_glyph_tokens = None

    def set_adapters(self, adapter_names, adapter_weights):
        self.calls.append(("set_adapters", list(adapter_weights)))

    def __call__(self, **kw):
        self.calls.append(("call", kw))
        insts = kw.get("glyph_instances")
        self._last_n_glyph_tokens = None if insts is None else gl.total_tokens(insts)
        return types.SimpleNamespace(images=[kw["control_image"].copy()])


class TinyProxy:
    """The real tiny PBRFluxPipeline behind UniTEX's set_adapters calls (no LoRAs loaded)."""

    def __init__(self, pipe, steps=2):
        self.pipe, self._num_inference_steps, self.calls = pipe, steps, []

    def set_adapters(self, adapter_names, adapter_weights):
        pass

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def __call__(self, **kw):
        self.calls.append(sorted(kw))
        return self.pipe(**kw)


def unitex_classes(src_dir, flux=None):
    """The three pipeline classes from <src_dir>/pipeline.py with every heavy dependency stubbed."""
    path = os.path.join(src_dir, "pipeline.py")
    tree = ast.parse(open(path).read(), path)
    body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in CLASSES]
    assert [n.name for n in body] == list(CLASSES)

    def fake_preprocess(image, alpha=None, H=1024, W=1024, **kw):
        # TextureTools preprocess stand-in: an RGBA of the requested size
        return image.convert("RGBA").resize((W, H))

    ns = {"os": os, "shutil": shutil, "json": json, "np": np, "torch": torch, "Image": Image,
          "Tuple": tuple, "CPUTimer": lambda name: (lambda f: f), "preprocess": fake_preprocess,
          "preprocess_blank_mesh": lambda i, o, **kw: open(o, "w").close(),
          "build_pipeline": lambda **kw: (flux if flux is not None else FakeFlux(), [1., 0.], [0., 1.],
                                          ["texture", "delight"]),
          "build_ltm": lambda **kw: (None, False, False), "RMBG2": lambda **kw: None,
          "build_rembg": lambda: None, "VideoExporter": FakeExporter,
          "NVDiffRendererInverse": lambda device=None: None, "TSDSRPipeline": None}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return types.SimpleNamespace(**{k: ns[k] for k in CLASSES})


def _grids(d, R, seed=0):
    rng = np.random.default_rng(seed)
    paths = []
    for name in ("mv_normal.png", "mv_ccm.png"):
        Image.fromarray(rng.integers(0, 255, (2 * R, 3 * R, 3), dtype=np.uint8)).save(os.path.join(d, name))
        paths.append(os.path.join(d, name))
    Image.fromarray(rng.integers(0, 255, (R, R, 3), dtype=np.uint8)).save(os.path.join(d, "processed_image.png"))
    return os.path.join(d, "processed_image.png"), *paths


def test_infer_mv_512_identical_to_pristine(repos, tmp_path):
    outs = {}
    for kind in ("pristine", "patched"):
        C = unitex_classes(repos[kind])
        pipe = C.CustomRGBTextureFullPipeline(seed=0)
        d = tmp_path / kind
        d.mkdir()
        pipe.infer_mv(str(d), *_grids(str(d), 512))
        calls = [(c[0], c[1] if c[0] == "set_adapters" else {k: v for k, v in c[1].items() if k != "generator"})
                 for c in pipe.pipeline.calls]
        outs[kind] = (calls, [np.asarray(Image.open(d / f)) for f in ("mv_rgb_w_light.png", "mv_rgb.png")])
    (ca, ia), (cb, ib) = outs["pristine"], outs["patched"]
    assert len(ca) == len(cb) == 4
    for a, b in zip(ca, cb):
        if a[0] == "set_adapters":
            assert a == b
        else:
            assert sorted(a[1]) == sorted(b[1])
            for k in a[1]:
                va, vb = a[1][k], b[1][k]
                if isinstance(va, Image.Image):
                    assert np.array_equal(np.asarray(va), np.asarray(vb)), k
                else:
                    assert va == vb, k
    assert all(np.array_equal(x, y) for x, y in zip(ia, ib))
    assert ia[0].shape == (512, 3072, 3) and ia[1].shape == (1024, 1536, 3)


@pytest.mark.parametrize("R", [64, 512, 1024])
def test_infer_mv_view_res_strip_order_round_trip(repos, tmp_path, R):
    """With a pass-through FLUX the texture strip is f l r b t d (bottom rolled back to raw view 5)
    and the delit grid equals the averaged input grids: the reorder is right at every R."""
    from unitex.common import FULL_INDEX, split_grid
    C = unitex_classes(repos["patched"])
    pipe = C.CustomRGBTextureFullPipeline(seed=0, view_res=R)
    ref, normal, ccm = _grids(str(tmp_path), R)
    pipe.infer_mv(str(tmp_path), ref, normal, ccm)
    avg = (0.5 * np.asarray(Image.open(normal)) + 0.5 * np.asarray(Image.open(ccm))).astype(np.uint8)
    strip = np.asarray(Image.open(tmp_path / "mv_rgb_w_light.png"))
    assert strip.shape == (R, 6 * R, 3)
    raw = split_grid(avg, res=R)
    for slot in range(6):
        assert np.array_equal(strip[:, slot * R:(slot + 1) * R], raw[FULL_INDEX[slot]]), slot
    assert np.array_equal(np.asarray(Image.open(tmp_path / "mv_rgb.png")), avg)
    call = pipe.pipeline.calls[1][1]
    assert (call["height"], call["width"]) == (R, 6 * R) and call["dual_image"].size == (R, R)


def test_infer_mv_rejects_grids_of_another_resolution(repos, tmp_path):
    C = unitex_classes(repos["patched"])
    pipe = C.CustomRGBTextureFullPipeline(seed=0, view_res=1024)
    with pytest.raises(ValueError, match="view_res 1024"):
        pipe.infer_mv(str(tmp_path), *_grids(str(tmp_path), 512))
    with pytest.raises(ValueError, match="multiple of 16"):
        C.CustomRGBTextureFullPipeline(seed=0, view_res=500)


@pytest.mark.parametrize("delight", [False, True])
def test_glyphs_go_to_the_texture_pass_and_are_logged(repos, eval2, tmp_path, delight):
    C = unitex_classes(repos["patched"])
    cfg = gl.GlyphConfig(anchor_mode="warp")
    pipe = C.CustomRGBTextureFullPipeline(seed=0, glyph_config=cfg, glyph_delight=delight)
    pipe.infer_mv(str(tmp_path), *_grids(str(tmp_path), 512), glyph_text=_text(eval2))
    calls = [c[1] for c in pipe.pipeline.calls if c[0] == "call"]
    tex, dl = calls
    assert "dual_image" in tex and "dual_image" not in dl
    insts = tex["glyph_instances"]
    assert len(insts) > 0 and tex["glyph_sample_mode"] == "argmax"
    assert ("glyph_instances" in dl) == delight
    if delight:
        assert dl["glyph_instances"] is insts
    info = json.load(open(tmp_path / "glyph_tokens.json"))
    assert info == json.loads(json.dumps(pipe.last_glyph_info))
    assert info["n_tokens"] == gl.total_tokens(insts) and info["anchor_mode"] == "warp"
    assert info["n_instances"] == len(insts) and info["delight"] == delight
    # run_unitex --dry-run logs the same summary without UniTEX
    _, dry = run_unitex.glyph_summary(_text(eval2), cfg, 512, delight, "argmax")
    assert json.loads(json.dumps(dry)) == info


def test_glyph_token_count_mismatch_is_an_error(repos, eval2, tmp_path):
    C = unitex_classes(repos["patched"])
    pipe = C.CustomRGBTextureFullPipeline(seed=0)
    real_call = FakeFlux.__call__

    def wrong(self, **kw):
        out = real_call(self, **kw)
        if self._last_n_glyph_tokens is not None:
            self._last_n_glyph_tokens -= 1
        return out
    pipe.pipeline.__class__ = type("WrongFlux", (FakeFlux,), {"__call__": wrong})
    with pytest.raises(RuntimeError, match="glyph tokens"):
        pipe.infer_mv(str(tmp_path), *_grids(str(tmp_path), 512), glyph_text=_text(eval2))


@pytest.mark.parametrize("R", [512, 1024])
def test_full_call_plumbs_view_res_and_glyph_text(repos, eval2, tmp_path, R):
    C = unitex_classes(repos["patched"])
    pipe = C.CustomRGBTextureFullPipeline(seed=0, view_res=R, glyph_config='{"anchor_mode": "stretch"}')

    def fake_bake(cache_dir, **kw):
        open(os.path.join(cache_dir, "textured_mesh.glb"), "w").close()
    pipe.step_seq = ["step_1_1", "fake_bake"]
    pipe.fake_bake = fake_bake
    sku = os.path.join(eval2, EVAL_SKUS[0])
    save = tmp_path / "run"
    pipe(str(save), os.path.join(sku, "ref.png"), os.path.join(sku, "mesh.glb"), glyph_text=_text(eval2))
    cache = save / "cache"
    assert pipe.video_exporter.calls[0]["H"] == pipe.video_exporter.calls[0]["W"] == R
    assert Image.open(cache / "processed_image.png").size == (R, R)
    assert Image.open(cache / "rembg_image.png").size == (max(1024, R),) * 2
    assert Image.open(cache / "mv_rgb_w_light.png").size == (6 * R, R)
    assert Image.open(save / "mv_rgb.png").size == (3 * R, 2 * R)
    info = json.load(open(cache / "glyph_tokens.json"))
    assert info["view_res"] == R and info["anchor_mode"] == "stretch" and info["n_tokens"] > 0
    # stock call (no glyph_text): no glyph kwargs at all, no glyph_tokens.json
    save2 = tmp_path / "run_stock"
    pipe.pipeline.calls.clear()
    pipe(str(save2), os.path.join(sku, "ref.png"), os.path.join(sku, "mesh.glb"))
    assert not any(k.startswith("glyph") for c in pipe.pipeline.calls if c[0] == "call" for k in c[1])
    assert not (save2 / "cache" / "glyph_tokens.json").exists() and pipe.last_glyph_info is None


def test_infer_mv_with_the_tiny_flux_pipeline(repos, pipes, eval2, tmp_path):
    """infer_mv end to end on the real (tiny) patched PBRFluxPipeline: both passes, glyph count check."""
    proxy = TinyProxy(pipes["patched"])
    C = unitex_classes(repos["patched"], flux=proxy)
    pipe = C.CustomRGBTextureFullPipeline(seed=0, glyph_delight=True)
    pipe.generator = torch.Generator().manual_seed(0)
    pipe.infer_mv(str(tmp_path), *_grids(str(tmp_path), 512), glyph_text=_text(eval2))
    info = pipe.last_glyph_info
    assert info["n_tokens"] > 0 and pipes["patched"]._last_n_glyph_tokens == info["n_tokens"]
    assert all("glyph_instances" in c for c in proxy.calls) and len(proxy.calls) == 2
    a = np.asarray(Image.open(tmp_path / "mv_rgb.png"))
    assert a.shape == (1024, 1536, 3)


# ────────────────────────────────────────────────────────────────────────────
# (d) run_unitex.py --dry-run and run_eval.sh
# ────────────────────────────────────────────────────────────────────────────

STOCK_KEYS = {"sku", "run_name", "seed", "dry_run", "add_lora_path", "add_lora_weights", "time", "host", "python",
              "torch", "diffusers", "status", "wall_s"}


def _run_unitex(*args):
    r = subprocess.run([PY, os.path.join(REPO, "unitex", "run_unitex.py"), *args], cwd=REPO,
                       capture_output=True, text=True)
    return r


def _log(eval_dir, run):
    return [json.loads(ln) for ln in open(os.path.join(eval_dir, "run_log.jsonl")) if json.loads(ln)["run_name"] == run]


def test_d_dry_run_without_glyph_flags_logs_as_before(eval2):
    r = _run_unitex("--eval-dir", eval2, "--dry-run", "--run-name", "stock_dry")
    assert r.returncode == 0, r.stderr
    recs = _log(eval2, "stock_dry")
    assert len(recs) == 2 and all(set(x) == STOCK_KEYS and x["status"] == "ok" for x in recs)
    for s in EVAL_SKUS:
        cache = os.path.join(eval2, s, "stock_dry", "cache")
        assert Image.open(os.path.join(cache, "mv_rgb_w_light.png")).size == (3072, 512)
        assert not os.path.exists(os.path.join(cache, "glyph_tokens.json"))


@pytest.mark.parametrize("R", [512, 1024])
def test_d_dry_run_logs_glyph_settings_and_token_counts(eval2, R):
    run = f"glyph_dry_{R}"
    r = _run_unitex("--eval-dir", eval2, "--dry-run", "--run-name", run, "--glyph-json-name", "text.json",
                    "--glyph-mode", "warp", "--glyph-kind", "box", "--glyph-set", "token_budget=2048",
                    "--glyph-delight", "--view-res", str(R))
    assert r.returncode == 0, r.stderr
    recs = _log(eval2, run)
    assert len(recs) == 2
    totals = {}
    for rec in recs:
        g = rec["glyph"]
        assert rec["status"] == "ok" and rec["view_res"] == R
        assert (g["mode"], g["kind"], g["delight"], g["sample_mode"]) == ("warp", "box", True, "argmax")
        assert g["config"]["token_budget"] == 2048 and g["json"] == _text(eval2, rec["sku"])
        cfg = gt.glyph_config(None, ["anchor_mode=warp", "infer_kind=box", "token_budget=2048"])
        insts = gt.build_infer_glyphs(g["json"], cfg, view_res=R)
        assert g["n_tokens"] == gl.total_tokens(insts) and g["n_instances"] == len(insts)
        assert sum(g["tokens_per_view"].values()) == g["n_tokens"]
        cache = os.path.join(eval2, rec["sku"], run, "cache")
        info = json.load(open(os.path.join(cache, "glyph_tokens.json")))
        assert info["n_tokens"] == g["n_tokens"] and info["view_res"] == R
        assert Image.open(os.path.join(cache, "mv_rgb_w_light.png")).size == (6 * R, R)
        assert Image.open(os.path.join(cache, "mv_alpha.png")).size == (3 * R, 2 * R)
        info_file = json.load(open(os.path.join(eval2, rec["sku"], run, "run_info.json")))
        assert info_file["glyph"]["n_tokens"] == g["n_tokens"]
        totals[rec["sku"]] = (g["n_instances"], g["n_tokens"], g["tokens_per_view"])
    print(f"\nview_res {R}: " + ", ".join(f"{k}: {v[0]} instances, {v[1]} tokens {v[2]}" for k, v in totals.items()))


def test_d_missing_text_json_and_flag_validation(eval2, tmp_path):
    r = _run_unitex("--eval-dir", eval2, "--dry-run", "--run-name", "nojson", "--glyph-json-name", "nope.json")
    assert r.returncode == 0
    recs = _log(eval2, "nojson")
    assert [x["status"] for x in recs] == ["missing_input"] * 2 and "nope.json" in recs[0]["error"]
    # {eval} / {sku} patterns and absolute paths
    r = _run_unitex("--eval-dir", eval2, "--dry-run", "--run-name", "pattern", "--skus", EVAL_SKUS[0],
                    "--glyph-json-name", "{eval}/{sku}/text.json")
    assert r.returncode == 0 and _log(eval2, "pattern")[0]["status"] == "ok"
    for bad in (["--glyph-mode", "warp"], ["--glyph-delight"], ["--glyph-set", "token_budget=1"],
                ["--glyph-json-name", "text.json", "--glyph-set", "no_such_field=1"],
                ["--glyph-json-name", "text.json", "--view-res", "768"], ["--view-res", "500"]):
        r = _run_unitex("--eval-dir", eval2, "--dry-run", "--run-name", "bad", *bad)
        assert r.returncode == 2, (bad, r.stderr)


def test_d_run_eval_sh_anchors_and_glyph_passthrough(tmp_path):
    if not all(os.path.isfile(os.path.join(EVAL28, s, "unitex_s63", "cache", "mv_ccm.png")) for s in EVAL_SKUS):
        pytest.skip("eval set not found")
    ev = str(tmp_path / "eval")
    _copy_eval(ev)
    env = dict(os.environ, EVAL_DIR=ev, DRY_RUN="1", STEPS="anchors unitex", PY=PY, UNITEX_PY=PY,
               RUN="glyph_sh", GLYPH="1", GLYPH_MODE="stretch", GLYPH_SET="token_budget=1024 max_lines=4",
               ANCHOR_JSON="text_lift.json")
    r = subprocess.run(["bash", os.path.join(REPO, "unitex", "run_eval.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for s in EVAL_SKUS:
        doc = json.load(open(os.path.join(ev, s, "text_lift.json")))
        assert doc["source"] == "photo-lift" and doc["items"]
        assert os.path.exists(os.path.join(ev, s, "text_lift_debug.png"))
    recs = _log(ev, "glyph_sh")
    assert len(recs) == 2 and all(x["status"] == "ok" for x in recs)
    for x in recs:
        assert x["glyph"]["json_name"] == "text_lift.json" and x["glyph"]["mode"] == "stretch"
        assert x["glyph"]["config"]["token_budget"] == 1024 and x["glyph"]["config"]["max_lines"] == 4
        assert x["glyph"]["n_tokens"] > 0


# ────────────────────────────────────────────────────────────────────────────
# Adversarial review: LTM at R != 512, schedule / noise invariance, flags-off regression vs HEAD
# ────────────────────────────────────────────────────────────────────────────

def _encode_geometry(src_dir):
    """LTM RGBFieldVAE.encode_geometry from <src_dir>/LTM/rgb_field.py (no craftsman imports)."""
    path = os.path.join(src_dir, "LTM", "rgb_field.py")
    tree = ast.parse(open(path).read(), path)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RGBFieldVAE")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "encode_geometry")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
    return ns["encode_geometry"]


class _ShapeModel:
    device = torch.device("cpu")

    def decode(self, latents, surface, sharp_surface=None):
        self.seen = latents.clone()
        return latents


def _ltm_inputs(R, f=1):
    """infer_field's alpha / ccm / albedo [1, 6, C, R*f, R*f] (nearest x f upsample of the R grid)."""
    g = torch.Generator().manual_seed(0)
    up = lambda t: t.repeat_interleave(f, -1).repeat_interleave(f, -2)
    alb = up(torch.rand(1, 6, 3, R, R, generator=g))
    ccm = up(torch.rand(1, 6, 3, R, R, generator=g))
    alpha = up((torch.rand(1, 6, 1, R, R, generator=g) > 0.5).float())
    surface = torch.cat([torch.rand(1, 50, 3, generator=g), torch.rand(1, 50, 3, generator=g)], -1)
    return dict(alpha_fbrltd=alpha, albedo_fbrltd=alb, ccm_fbrltd=ccm, surface=surface)


def test_ltm_encode_geometry_runs_at_view_res_1024(repos):
    """step_2_ablition -> infer_field -> LTM encode_geometry gets the 2R x 3R grids at R px per view.
    Upstream resizes to 512 by slice assignment, which raises at R = 1024 (so every 1024 SKU
    would end in status error). Patched: same as stock at 512, and a 1024 grid that is a nearest
    x2 upsample of a 512 grid reaches the LTM exactly as that 512 grid."""
    stock, ours = _encode_geometry(repos["pristine"]), _encode_geometry(repos["patched"])
    me = lambda: types.SimpleNamespace(shape_model=_ShapeModel(), shape_model_dtype=torch.float32)
    with pytest.raises(RuntimeError):
        stock(me(), **_ltm_inputs(512, 2))
    a, b, c = me(), me(), me()
    stock(a, **_ltm_inputs(512))
    ours(b, **_ltm_inputs(512))
    ours(c, **_ltm_inputs(512, 2))
    assert a.shape_model.seen.shape == (1, 7, 512, 6 * 512)
    assert torch.equal(a.shape_model.seen, b.shape_model.seen)
    assert torch.equal(b.shape_model.seen, c.shape_model.seen)


class _StepLog:
    """Transformer timestep / hidden_states of every step."""

    def __init__(self, tr):
        self.tr, self.t, self.h = tr, [], []

    def __enter__(self):
        def hook(module, args, kwargs):
            self.t.append(kwargs["timestep"].detach().clone())
            self.h.append(kwargs["hidden_states"].detach().clone())
        self.handle = self.tr.register_forward_pre_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def test_glyphs_leave_schedule_noise_stream_and_batch_alone(pipes, tiny, eval2):
    """Glyph tokens must not change the timestep shift (mu from the sequence length), must not
    draw from the pipeline generator even in posterior-sample mode, and with 2 images per prompt
    every image gets the same clean glyph tokens and the latents come back without them."""
    tr, vae, _ = tiny
    R, p = 512, pipes["patched"]
    insts = gt.build_infer_glyphs(_text(eval2), gl.GlyphConfig(anchor_mode="stretch"), view_res=R)
    Ng = gl.total_tokens(insts)
    n_noise = _n_tokens(R)[0]
    ctrl, dual = _inputs(R)
    states, logs, outs = [], [], []
    for kw in ({}, dict(glyph_instances=insts, glyph_sample_mode="sample")):
        g = torch.Generator().manual_seed(5)
        with _StepLog(tr) as lg:
            out = p(prompt="[MVFLUX]", control_image=ctrl, dual_image=dual, height=R, width=6 * R,
                    n_rows=1, n_cols=6, num_inference_steps=3, generator=g, num_images_per_prompt=2,
                    output_type="latent", **kw).images
        states.append(g.get_state())
        logs.append(lg)
        outs.append(out)
    assert torch.equal(states[0], states[1])                      # generator untouched by glyphs
    assert all(torch.equal(a, b) for a, b in zip(logs[0].t, logs[1].t)) and len(logs[1].t) == 3
    assert torch.equal(logs[0].h[0][:, :n_noise], logs[1].h[0][:, :n_noise])
    for h in logs[1].h:                                           # same clean glyphs in both images
        assert h.shape[0] == 2 and h.shape[1] == logs[0].h[0].shape[1] + Ng
        assert torch.equal(h[0, -Ng:], h[1, -Ng:]) and torch.equal(h[:, -Ng:], logs[1].h[0][:, -Ng:])
    assert outs[1].shape == outs[0].shape == (2, n_noise, 64)     # stripped before decode
    assert not torch.equal(outs[0], outs[1])


def _git_head_file(rel, dst):
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO, capture_output=True)
    if r.returncode:
        pytest.skip(f"git show HEAD:{rel} failed")
    with open(dst, "wb") as f:
        f.write(r.stdout)
    return str(dst)


def test_run_unitex_flags_off_matches_head_dry_run(tmp_path):
    """--dry-run without the new flags: every output file and log field (minus times) equals the
    committed run_unitex.py."""
    if not all(os.path.isfile(os.path.join(EVAL28, s, "ref.png")) for s in EVAL_SKUS):
        pytest.skip("eval set not found")
    old = _git_head_file("unitex/run_unitex.py", tmp_path / "run_unitex_head.py")
    runs = {}
    for kind, script in (("head", old), ("new", os.path.join(REPO, "unitex", "run_unitex.py"))):
        ev = str(tmp_path / kind)
        _copy_eval(ev)
        r = subprocess.run([PY, script, "--eval-dir", ev, "--dry-run", "--run-name", "r", "--seed", "7"],
                           cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        runs[kind] = ev
    drop = {"time", "wall_s"}
    la, lb = (_log(runs[k], "r") for k in ("head", "new"))
    assert [{k: v for k, v in x.items() if k not in drop} for x in la] == \
           [{k: v for k, v in x.items() if k not in drop} for x in lb]
    n = 0
    for s in EVAL_SKUS:
        da, db = os.path.join(runs["head"], s, "r"), os.path.join(runs["new"], s, "r")
        fa = sorted(os.path.relpath(os.path.join(w, f), da) for w, _, fs in os.walk(da) for f in fs)
        fb = sorted(os.path.relpath(os.path.join(w, f), db) for w, _, fs in os.walk(db) for f in fs)
        assert fa == fb
        for rel in fa:
            if rel.endswith(".pth"):
                ta, tb = torch.load(os.path.join(da, rel)), torch.load(os.path.join(db, rel))
                assert all(torch.equal(ta[k], tb[k]) if torch.is_tensor(ta[k]) else ta[k] == tb[k] for k in ta)
            elif rel == "run_info.json":
                ja, jb = (json.load(open(os.path.join(d, rel))) for d in (da, db))
                assert {k: v for k, v in ja.items() if k not in drop} == {k: v for k, v in jb.items() if k not in drop}
            else:
                assert open(os.path.join(da, rel), "rb").read() == open(os.path.join(db, rel), "rb").read(), rel
            n += 1
    assert n >= 10


def test_run_eval_sh_flags_off_passes_head_args(tmp_path):
    """run_eval.sh without GLYPH / VIEW_RES hands run_unitex.py exactly the committed arguments,
    and a relative GLYPH_CONFIG file reaches the real (cd UNITEX_ROOT) run as an absolute path."""
    stub = tmp_path / "py_stub.sh"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "${@:2}" > "$ARGV_OUT"\n')
    stub.chmod(0o755)
    ev = tmp_path / "eval"
    ev.mkdir()
    (tmp_path / "unitex_root").mkdir()
    head_dir = tmp_path / "head_unitex"
    head_dir.mkdir()
    head_sh = _git_head_file("unitex/run_eval.sh", head_dir / "run_eval.sh")
    base = dict(os.environ, REPO=REPO, EVAL_DIR=str(ev), STEPS="unitex", UNITEX_PY=str(stub),
                UNITEX_ROOT=str(tmp_path / "unitex_root"), RUN="x", SEED="5",
                EXTRA_LORA="/a.safetensors", EXTRA_LORA_WEIGHTS="0.5")
    argv = {}
    for kind, sh in (("head", head_sh), ("new", os.path.join(REPO, "unitex", "run_eval.sh"))):
        for dry in ("", "1"):
            out = tmp_path / f"argv_{kind}_{dry or 0}"
            r = subprocess.run(["bash", sh], env=dict(base, DRY_RUN=dry, ARGV_OUT=str(out)),
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            argv[kind, dry] = out.read_text().splitlines()
    assert argv["head", ""] == argv["new", ""] and argv["head", "1"] == argv["new", "1"]
    assert "--add-lora-path" in argv["new", ""] and not any("glyph" in a or "view-res" in a for a in argv["new", ""])
    # a relative GLYPH_CONFIG file (relative to REPO, where run_eval.sh runs) must survive the cd
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"anchor_mode": "warp"}')
    out = tmp_path / "argv_glyph"
    r = subprocess.run(["bash", os.path.join(REPO, "unitex", "run_eval.sh")],
                       env=dict(base, ARGV_OUT=str(out), GLYPH="1", GLYPH_CONFIG=os.path.relpath(cfg, REPO)),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    a = out.read_text().splitlines()
    assert os.path.realpath(a[a.index("--glyph-config") + 1]) == os.path.realpath(cfg)
    assert os.path.isabs(a[a.index("--glyph-config") + 1])
