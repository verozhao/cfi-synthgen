"""
Tests for unitex/run_unitex_fast.py and unitex/bake_steps.py.

CPU: the first block cache on a tiny FluxTransformer2DModel, the per-pass wrapper, the adapter filter, the bake steps.
CUDA (skipped without a GPU): block streaming gives the same numbers as the model held on the GPU, with and without
prefetch, with resident blocks, over several steps and together with the cache. Page-locked copies are exact, pinned,
of exactly the size asked for and in a private mapping that forked children do not get.

Run: python -m pytest unitex/tests/test_fast.py -v
"""

import copy
import mmap
import os
import subprocess
import sys
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.join(REPO, "unitex"))
import bake_steps  # noqa: E402
import run_unitex_fast as F  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def tiny(n_double=2, n_single=3, seed=0):
    from diffusers import FluxTransformer2DModel
    torch.manual_seed(seed)
    return FluxTransformer2DModel(patch_size=1, in_channels=16, num_layers=n_double, num_single_layers=n_single,
                                  attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
                                  pooled_projection_dim=8, guidance_embeds=True, axes_dims_rope=(4, 6, 6)).eval()


def inputs(n=12, t=0.5, seed=1, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(n, 3)
    ids[:, 1] = torch.arange(n) // 4
    ids[:, 2] = torch.arange(n) % 4
    k = dict(hidden_states=torch.randn(1, n, 16, generator=g), encoder_hidden_states=torch.randn(1, 5, 32, generator=g),
             pooled_projections=torch.randn(1, 8, generator=g), timestep=torch.tensor([t]),
             guidance=torch.tensor([3.5]), img_ids=ids, txt_ids=torch.zeros(5, 3))
    return {name: v.to(device) for name, v in k.items()}


def run(model, **k):
    with torch.no_grad():
        return model(**k, return_dict=False)[0]


def count_block_calls(model):
    """Counter per block of calls that reach its real forward (installed before any cache wraps it)."""
    counts = []
    for b in list(model.transformer_blocks) + list(model.single_transformer_blocks):
        counts.append(0)

        def fwd(*a, _orig=b.forward, _i=len(counts) - 1, **kw):
            counts[_i] += 1
            return _orig(*a, **kw)
        b.forward = fwd
    return counts


# ── first block cache ───────────────────────────────────────────────────────

def test_cache_off_changes_nothing():
    model = tiny()
    ref = [run(model, **inputs(seed=s, t=t)) for s, t in ((1, 0.9), (2, 0.5))]
    cache = F.FirstBlockCache(model)
    assert cache.threshold is None
    for (s, t), r in zip(((1, 0.9), (2, 0.5)), ref):
        assert torch.equal(run(model, **inputs(seed=s, t=t)), r)
    assert cache.count == [0, 0]


def test_cache_with_a_tiny_threshold_never_skips_and_matches_the_plain_model():
    model = tiny()
    steps = [(1, 0.9), (2, 0.7), (3, 0.5)]
    ref = [run(model, **inputs(seed=s, t=t)) for s, t in steps]
    cache = F.FirstBlockCache(model)
    cache.reset(1e-9)
    for (s, t), r in zip(steps, ref):
        assert torch.equal(run(model, **inputs(seed=s, t=t)), r)
    assert cache.count == [3, 0]


def test_cache_skips_all_but_the_first_block_and_reuses_the_last_computed_tail():
    model = tiny(n_double=2, n_single=3)
    counts = count_block_calls(model)
    cache = F.FirstBlockCache(model)
    cache.reset(1e9)                              # any step after a computed one is skipped
    a = run(model, **inputs(seed=1, t=0.9))
    assert counts == [1, 1, 1, 1, 1] and cache.count == [1, 0]
    # the same input again: the first block gives the same output, so adding the stored tail reproduces the step
    b = run(model, **inputs(seed=1, t=0.9))
    assert counts == [2, 1, 1, 1, 1] and cache.count == [1, 1]
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    # another input: skipped as well, and the result differs from the full computation of that input
    c = run(model, **inputs(seed=2, t=0.5))
    assert counts == [3, 1, 1, 1, 1] and cache.count == [1, 2]
    cache.reset(None)
    full = run(model, **inputs(seed=2, t=0.5))
    assert counts == [4, 2, 2, 2, 2]
    assert not torch.allclose(c, full, atol=1e-4)


def test_cache_decides_on_the_change_since_the_last_computed_step():
    model = tiny()
    cache = F.FirstBlockCache(model)
    x = inputs(seed=1, t=0.9)
    cache.reset(0.05)
    run(model, **x)
    near = dict(x, hidden_states=x["hidden_states"] + 1e-5)
    run(model, **near)
    assert cache.count == [1, 1]
    run(model, **inputs(seed=7, t=0.1))           # a different latent: the first block's residual moves a lot
    assert cache.count == [2, 1]


def test_cache_recomputes_when_the_token_count_changes_and_reset_clears_it():
    model = tiny()
    cache = F.FirstBlockCache(model)
    cache.reset(1e9)
    run(model, **inputs(n=12))
    run(model, **inputs(n=20))                    # another pass shape: no reuse, no shape error
    assert cache.count == [2, 0]
    cache.reset(1e9)
    assert cache.first is None and cache.tail is None and cache.count == [0, 0]


# ── per-pass wrapper, adapter filter ────────────────────────────────────────

class FakeVAE:
    def __init__(self):
        self.decoded = 0

    def decode(self, z):
        self.decoded += 1
        return z


def fake_pipe():
    calls = []

    class Flux:
        def __init__(self):
            self.vae = FakeVAE()
            self.adapters = []

        def __call__(self, **k):
            calls.append(dict(k, threshold=getattr(self, "cache", None) and self.cache.threshold))
            self.vae.decode(0)
            return "image"

        def set_adapters(self, adapter_names, adapter_weights=None):
            self.adapters.append((adapter_names, adapter_weights))
    return types.SimpleNamespace(pipeline=Flux()), calls


def test_wrap_passes_sets_steps_and_thresholds_per_pass(capsys):
    pipe, calls = fake_pipe()
    cache = types.SimpleNamespace(threshold=None, count=[0, 0])
    cache.reset = lambda t: setattr(cache, "threshold", t or None)
    pipe.pipeline.cache = cache
    F.wrap_passes(pipe, texture_steps=20, delight_steps=8, cache=cache, texture_cache=0.08, delight_cache=None,
                  trim=False)
    assert pipe.pipeline(dual_image="photo", num_inference_steps=28, width=6144, height=1024) == "image"
    assert pipe.pipeline(num_inference_steps=28, width=3072, height=512) == "image"
    assert [c["num_inference_steps"] for c in calls] == [20, 8]
    assert [c["threshold"] for c in calls] == [0.08, None]
    assert cache.threshold is None and pipe.pipeline.vae.decoded == 2
    out = capsys.readouterr().out
    assert ">>> texture_pass" in out and ">>> delight_pass" in out and "20 steps" in out and "8 steps" in out


def test_wrap_passes_leaves_the_step_count_alone_by_default():
    pipe, calls = fake_pipe()
    F.wrap_passes(pipe, trim=False)
    pipe.pipeline(dual_image="photo", num_inference_steps=28)
    pipe.pipeline(num_inference_steps=28)
    assert [c["num_inference_steps"] for c in calls] == [28, 28]


def test_active_loras_only_drops_zero_weight_adapters():
    pipe, _ = fake_pipe()
    F.active_loras_only(pipe)
    names = ["texture", "delight", "cfi_texture"]
    pipe.pipeline.set_adapters(adapter_names=names, adapter_weights=[0.0, 0.0, 1.0])
    pipe.pipeline.set_adapters(adapter_names=names, adapter_weights=[0.0, 1.0, 0.0])
    pipe.pipeline.set_adapters(adapter_names=names, adapter_weights=[0.0, 0.0, 0.0])      # nothing left: unchanged
    pipe.pipeline.set_adapters(adapter_names=names, adapter_weights=[{"down": 1.0}, 0.0, 1.0])   # not plain numbers
    pipe.pipeline.set_adapters("texture", 1.0)
    assert pipe.pipeline.adapters == [(["cfi_texture"], [1.0]), (["delight"], [1.0]), (names, [0.0, 0.0, 0.0]),
                                      (names, [{"down": 1.0}, 0.0, 1.0]), ("texture", 1.0)]
    assert names == ["texture", "delight", "cfi_texture"]


# ── bake steps ──────────────────────────────────────────────────────────────

def fake_baker(tmp_path):
    log = []
    pipe = types.SimpleNamespace(super_resolutions=False)
    pipe.sampling_on_mesh = lambda *a, **k: log.append(("sampling_on_mesh", a, k))
    pipe.infer_field = lambda *a, **k: log.append(("infer_field", a, k))

    def reproject(save_dir, *a, **k):
        log.append(("reproject_and_query_field", (save_dir,) + a, k))
        with open(os.path.join(save_dir, "textured_mesh.glb"), "w") as f:
            f.write("baked with LTM" if k.get("inpainting") else "baked without LTM")
    pipe.reproject_and_query_field = reproject
    pipe.export_video = lambda *a, **k: log.append(("export_video", a, k))
    cache = tmp_path / "cache"
    cache.mkdir()
    mesh = tmp_path / "mesh.glb"
    mesh.write_text("untextured mesh")
    return pipe, log, cache, mesh


def test_one_bake_runs_only_the_ltm_bake(tmp_path):
    pipe, log, cache, mesh = fake_baker(tmp_path)
    bake_steps.one_bake(pipe)
    pipe.step_2_ablition(cache_dir=str(cache), input_image_path="ref.png", input_mesh_path=str(mesh), clear_cache=False,
                         glyph_text=None)
    assert [name for name, _, _ in log] == ["sampling_on_mesh", "infer_field", "reproject_and_query_field",
                                            "export_video"]
    reproject = log[2]
    assert reproject[1][0] == str(cache / "w_LTM") and reproject[2] == {"inpainting": True}
    assert log[1][1][0] == str(cache / "w_LTM") and log[1][1][-1] == str(cache / "mv_rgb.png")
    assert (cache / "textured_mesh.glb").read_text() == "baked with LTM"
    assert not (cache / "wo_LTM").exists()


def test_one_bake_refuses_super_resolution(tmp_path):
    pipe, _, _, _ = fake_baker(tmp_path)
    pipe.super_resolutions = True
    with pytest.raises(SystemExit):
        bake_steps.one_bake(pipe)


def test_no_bake_leaves_the_input_mesh_and_a_note(tmp_path):
    pipe, log, cache, mesh = fake_baker(tmp_path)
    bake_steps.no_bake(pipe)
    pipe.step_2_ablition(cache_dir=str(cache), input_image_path="ref.png", input_mesh_path=str(mesh), clear_cache=False,
                         glyph_text=None)
    assert log == []
    assert (cache / "textured_mesh.glb").read_text() == "untextured mesh"
    assert "untextured input mesh" in (cache / "NO_BAKE.txt").read_text()


# ── CUDA: page-locked copies and block streaming ────────────────────────────

@cuda
def test_pinned_copies_are_exact_pinned_and_of_the_size_asked_for():
    torch.manual_seed(0)
    tensors = [torch.randn(300, 70).to(torch.bfloat16), torch.randn(1000), torch.randint(0, 9, (5, 3, 2))]
    n = len(F._REGISTERED)
    copies, note = F.pinned_copies(tensors)
    assert note is None, note
    for t, c in zip(tensors, copies):
        assert c.dtype == t.dtype and c.shape == t.shape and torch.equal(c, t) and c.is_pinned() and not t.is_pinned()
        assert torch.equal(c.to("cuda", non_blocking=True).cpu(), t)
    size = F._REGISTERED[n][1].numel()
    need = sum((t.numel() * t.element_size() + 63) // 64 * 64 for t in tensors)
    assert need <= size < need + mmap.PAGESIZE and size % mmap.PAGESIZE == 0
    # the buffer is a private mapping that forked children do not get ("dc" in its VmFlags)
    addr = F._REGISTERED[n][1].data_ptr()
    entry = None
    with open("/proc/self/smaps") as f:
        for line in f:
            head = line.split()
            if len(head) >= 2 and "-" in head[0] and len(head[1]) == 4 and not head[0].endswith(":"):
                lo, hi = (int(x, 16) for x in head[0].split("-"))
                entry = {"perms": head[1]} if lo <= addr < hi else None
            elif entry is not None and line.startswith("VmFlags:"):
                entry["flags"] = head[1:]
                break
    assert entry and entry["perms"] == "rw-p" and "dc" in entry["flags"], entry
    assert subprocess.run([sys.executable, "-c", "pass"]).returncode == 0     # fork and exec with the buffer live
    assert all(torch.equal(c, t) for t, c in zip(tensors, copies))


@cuda
def test_pinned_copies_fall_back_to_torch_pinning(monkeypatch):
    monkeypatch.setenv("UNITEX_FAST_TORCH_PIN", "1")
    t = torch.randn(64, 64)
    copies, note = F.pinned_copies([t])
    assert "UNITEX_FAST_TORCH_PIN" in note and copies[0].is_pinned() and torch.equal(copies[0], t)


@cuda
@pytest.mark.parametrize("prefetch", [True, False])
@pytest.mark.parametrize("resident", [False, True])
def test_block_streamer_gives_the_numbers_of_the_model_on_the_gpu(monkeypatch, prefetch, resident):
    monkeypatch.setattr(F.BlockStreamer, "SMALL", 512)        # the tiny model's weight matrices count as large
    base = tiny(n_double=2, n_single=3)
    first = sum(p.numel() * p.element_size() for p in base.transformer_blocks[0].parameters())
    resident_gb = first / 1024 ** 3 if resident else 0.0      # room for the first block and nothing more
    steps = [(1, 0.9), (2, 0.6), (3, 0.3)]
    on_gpu = copy.deepcopy(base).cuda()
    ref = [run(on_gpu, **inputs(seed=s, t=t, device="cuda")) for s, t in steps]
    model = copy.deepcopy(base)
    streamer = F.BlockStreamer(model, torch.device("cuda"), resident_gb=resident_gb, prefetch=prefetch)
    assert streamer.info["blocks"] == 5 and streamer.info["exact_pinning"]
    assert streamer.info["resident_blocks"] == (1 if resident else 0)
    assert streamer.info["streamed_gb_per_step"] >= 0 and len(streamer.streamed) == 5 - streamer.info["resident_blocks"]
    assert all(params for params, _ in streamer.streamed)
    for (s, t), r in zip(steps, ref):
        out = run(model, **inputs(seed=s, t=t, device="cuda"))
        assert torch.equal(out, r)
        assert not streamer.inflight
        for params, pinned in streamer.streamed:              # the large weights are back in page-locked memory
            assert all(p.device.type == "cpu" and p.data_ptr() == c.data_ptr() for p, c in zip(params, pinned))


@cuda
def test_cache_and_streamer_skip_the_copies_of_skipped_blocks(monkeypatch):
    monkeypatch.setattr(F.BlockStreamer, "SMALL", 512)
    base = tiny(n_double=2, n_single=3)
    on_gpu = copy.deepcopy(base).cuda()
    plain_cache = F.FirstBlockCache(on_gpu)
    plain_cache.reset(1e9)
    ref = [run(on_gpu, **inputs(seed=s, t=t, device="cuda")) for s, t in ((1, 0.9), (2, 0.5))]

    model = copy.deepcopy(base)
    streamer = F.BlockStreamer(model, torch.device("cuda"))
    sent = []
    send = streamer._send
    streamer._send = lambda i: (sent.append(i), send(i))[1]
    cache = F.FirstBlockCache(model, streamer)
    cache.reset(1e9)
    a = run(model, **inputs(seed=1, t=0.9, device="cuda"))
    assert sent == [0, 1, 2, 3, 4] and cache.count == [1, 0]
    b = run(model, **inputs(seed=2, t=0.5, device="cuda"))
    assert cache.count == [1, 1]
    assert sent == [0, 1, 2, 3, 4, 0, 1]          # the first block, and the one prefetched while it ran
    assert streamer.loaded == {0: True, 1: False, 2: False, 3: False, 4: False}
    assert torch.equal(a, ref[0]) and torch.equal(b, ref[1])
    # a full step afterwards uses the block that was already sent
    cache.reset(None)
    full = run(model, **inputs(seed=2, t=0.5, device="cuda"))
    assert sent == [0, 1, 2, 3, 4, 0, 1, 0, 2, 3, 4]
    plain_cache.reset(None)
    assert torch.equal(full, run(on_gpu, **inputs(seed=2, t=0.5, device="cuda")))


# ── stage_times ─────────────────────────────────────────────────────────────

def test_stage_times_reads_passes_stages_and_products():
    import stage_times
    text = (
        "Loading pipeline components...: 100%|##| 5/5 [00:00<00:00, 21.66it/s]\n"
        ">>> preprocess_blank_mesh 1.0 >>>\n"
        ">>> render_geometry_images 0.5 >>>\n"
        "  4%|#| 1/28 [00:36<16:12, 36.01s/it]\r100%|##| 28/28 [16:20<00:00, 35.0s/it]\r"
        "100%|##| 28/28 [16:20<00:00, 35.0s/it]\n"
        "100%|##| 28/28 [04:12<00:00,  9.0s/it]\n"
        ">>> texture_pass 985.0 >>> 28 steps\n"
        ">>> infer_mv 1242.0 >>>\n"
        ">>> reproject_and_query_field 10.0 >>>\n>>> sampling_on_mesh 1.5 >>>\n>>> infer_field 0.5 >>>\n"
        ">>> reproject_and_query_field 16.0 >>>\n"
        "  [012345678905] ok in 1273.5 s, peak 15.28 GB\n"
        ">>> preprocess_blank_mesh 1.0 >>>\n")          # a product that did not finish is not counted
    rows = stage_times.parse(text)
    assert len(rows) == 1 and rows[0]["sku"] == "012345678905" and rows[0]["peak"] == 15.28
    assert rows[0]["passes"] == [(28, 980), (28, 252)]
    out, wall = stage_times.summary(rows)
    d = dict(out)
    assert wall == 1273.5
    assert d["texture pass (FLUX), 28 steps, 35.0 s per step"] == 980
    assert d["delight pass (FLUX), 28 steps, 9.0 s per step"] == 252
    assert d["infer_mv, other than the two passes"] == 10.0
    assert d["reproject_and_query_field (x2)"] == 26.0
    assert not any(k.startswith("texture_pass") for k in d)
    assert abs(sum(d.values()) - wall) < 1e-6


def test_fps_height_raises_the_tree_height_for_large_point_sets_only(monkeypatch):
    calls = []
    fake = types.ModuleType("fpsample")
    fake.bucket_fps_kdline_sampling = \
        lambda pc, n_samples, h, start_idx=None: calls.append((len(pc), n_samples, h, start_idx)) or "idx"
    monkeypatch.setitem(sys.modules, "fpsample", fake)
    bake_steps.fps_height(9, min_points=1000)
    bake_steps.fps_height(9, min_points=1000)          # a second call does not stack wrappers
    assert fake.bucket_fps_kdline_sampling(range(999), n_samples=8, h=5) == "idx"
    assert fake.bucket_fps_kdline_sampling(range(1000), n_samples=8, h=5, start_idx=3) == "idx"
    assert fake.bucket_fps_kdline_sampling(range(5000), 8, 11) == "idx"
    assert calls == [(999, 8, 5, None), (1000, 8, 9, 3), (5000, 8, 11, None)]
