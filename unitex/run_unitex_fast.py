"""
run_unitex_lowmem.py, faster: the released bf16 FLUX weights and LoRAs unchanged on a 24 GB GPU, with a cheaper way of
getting the weights to the GPU. The model and its outputs stay the same unless one of the lossy options is given
(mv_rgb.png is pixel-identical to run_unitex_lowmem.py's). DESIGN.md, "Inference speed (step 9)", has the numbers.

Where the time goes (28 products, RTX 4090, run_unitex_lowmem.py): the two FLUX passes take 95 to 97 % of a product.
  1024 px views, delight at 512   21.2 min   texture pass 28 steps of 35 s (77 %), delight pass 28 steps of 9 s (20 %)
  512 px views                     8.7 min   texture pass 28 steps of 9 s (48 %), delight pass 28 steps of 9 s (47 %)
The two bakes take 18 to 26 s, everything else under 20 s. With this file: 10.8 min and 2.8 min.

run_unitex_lowmem.py streams the weights with accelerate.cpu_offload: before every layer runs, each of its weights is
copied from ordinary CPU memory with a blocking copy, and torch.cuda.empty_cache() is called after every such copy,
thousands of times per step, so the activations are allocated from the driver again each time. Here the large weights
sit in page-locked CPU memory (private mappings of exactly their size, see pinned_copies) and go to the GPU a whole
block at a time, the next block on a second CUDA stream while the current block computes (--stream pinned). Adapters
with weight 0 are switched off for the pass instead of being computed and multiplied by 0. The CUDA cache is emptied
before each VAE decode and after each pass.

Usage: the arguments of run_unitex.py, plus
  --stream accelerate | pinned | pinned-serial
        accelerate      as run_unitex_lowmem.py (for comparison)
        pinned          default. Blocks from page-locked memory, the next one copied during the current block's compute
        pinned-serial   the same without the second stream: a block is copied, then it runs
  --resident-gb G       GB of FLUX blocks kept on the GPU and never copied again (default 0). No gain measured with
                        --stream pinned (the copies already overlap the compute), and the GPU must still hold the
                        activations (--pass-memory prints them).
  --all-loras           keep the adapters with weight 0 active, as UniTEX does. PEFT computes every active adapter in
                        all 342 adapted layers and multiplies by the weight, so two of the three adapters are computed
                        for nothing. The numbers are the same either way.
  --one-bake            skip the bake without LTM (bake_steps.one_bake)
  --no-bake             stop after the views (mv_rgb.png): for a photo_front.py run next, which bakes itself
  --fps-height H        tree height of the bake's farthest point sampling on large point sets (bake_steps.fps_height,
                        UniTEX: 5, e.g. 9). The sample differs as it does between two runs.
  --roughness R         roughness of the baked model's material (bake_steps.material). UniTEX writes 1.0, fully
                        matte. The current Gemini pipeline's models have 0.6. Also drops the source path from the
                        model.
  --pass-memory         print each pass's time and peak GPU memory, the transformer's and the VAE decode's apart. The
                        peak in run_info.json then covers the delight pass and the bake only.
Lossy, off by default (the output changes and has to be looked at):
  --texture-steps N     denoising steps of the texture pass (UniTEX: 28)
  --delight-steps N     denoising steps of the delight pass (UniTEX: 28)
                        On the 28 eval products, 20 texture steps with 8 delight steps kept every front and 16
                        texture steps did not.
  --texture-cache T     first block cache with threshold T (e.g. 0.08) on the texture pass / the delight pass: a step
  --delight-cache T     whose first block changes less than T since the last computed step reuses that step's result
                        for the other 56 blocks

The settings go to run_info.json as "fast".
For diagnosis, environment variables: UNITEX_FAST_TORCH_PIN=1 takes the page-locked copies from torch's own allocator,
UNITEX_FAST_NO_TRIM=1 never empties the CUDA cache, UNITEX_FAST_STACKS=seconds:file dumps the Python stacks of all
threads every so many seconds.
"""

import argparse
import functools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bake_steps            # noqa: E402
import run_unitex            # noqa: E402

FAST = None
lowmem = None                # run_unitex_lowmem, imported in main(): its import swaps run_unitex.build_pipeline

_REGISTERED = []             # (mmap, tensor over it) of every buffer page-locked with cudaHostRegister, kept to the end


def _unregister_all():
    try:
        import torch
        for _, raw in _REGISTERED:
            torch.cuda.cudart().cudaHostUnregister(raw.data_ptr())
    except Exception:
        pass


def pinned_copies(tensors):
    """Page-locked CPU copies of the tensors, in one buffer of exactly their size. Returns (copies, note).

    torch's own pinned allocator (Tensor.pin_memory) rounds every allocation up to a power of two: FLUX's 22 GB of
    large weights locked 34 GB of RAM per process. Here an anonymous mmap is locked with cudaHostRegister, which
    locks what it is given. If that fails the copies come from pin_memory and the note says why.

    The mapping is private. Locked pages cannot be moved, and the kernel's memory compaction (Linux 5.15) passes
    over locked private pages at a glance, but takes locked shared pages off their lists and fails to move each
    one. With shared mappings (Python's default for mmap(-1, n)) a compaction pass over the buffers of other
    processes took up to a minute, and a process that had asked for a huge page waited for it (DESIGN.md, step 9).
    MADV_DONTFORK keeps the mapping out of forked children: a fork would copy locked private pages at once.
    """
    import mmap
    import torch
    align = 64
    offsets, total = [], 0
    for t in tensors:
        offsets.append(total)
        total += (t.numel() * t.element_size() + align - 1) // align * align
    size = max(mmap.PAGESIZE, (total + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE)
    try:
        if os.environ.get("UNITEX_FAST_TORCH_PIN"):
            raise RuntimeError("UNITEX_FAST_TORCH_PIN is set")
        mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        mm.madvise(mmap.MADV_DONTFORK)
        raw = torch.frombuffer(mm, dtype=torch.uint8)
        rc = int(torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), size, 0))
        if rc != 0:
            raise RuntimeError(f"cudaHostRegister returned {rc}")
        if not raw.is_pinned():
            torch.cuda.cudart().cudaHostUnregister(raw.data_ptr())
            raise RuntimeError("the registered buffer does not report as pinned")
    except Exception as e:
        return [t.pin_memory() for t in tensors], f"{type(e).__name__}: {e}"
    if not _REGISTERED:
        import atexit
        atexit.register(_unregister_all)
    _REGISTERED.append((mm, raw))
    copies = []
    for t, o in zip(tensors, offsets):
        v = raw[o:o + t.numel() * t.element_size()].view(t.dtype).view(t.shape)
        v.copy_(t)
        copies.append(v)
    return copies, None


class BlockStreamer:
    """FLUX's blocks run on the GPU while their large weights live in page-locked CPU memory.

    A streamed block gets its weights just before it runs and gives the GPU memory back right after. With prefetch,
    the next streamed block's weights are copied on a second CUDA stream while the current block computes. Resident
    blocks stay on the GPU. Small tensors of every block (LoRA factors, biases, norm weights: a few hundred MB in all)
    stay on the GPU as well, since they would be thousands of tiny copies per step.
    """

    SMALL = 2 << 20      # bytes

    def __init__(self, transformer, device, resident_gb=0.0, prefetch=True):
        import torch
        self.torch = torch
        self.device = device
        self.side = torch.cuda.Stream(device) if prefetch else None
        blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
        for name, child in transformer.named_children():     # embedders, output norm and projection: small, resident
            if name not in ("transformer_blocks", "single_transformer_blocks"):
                child.to(device)
        budget = resident_gb * 1024 ** 3
        self.streamed = []           # per streamed block: (parameters, their page-locked CPU tensors)
        self.inflight = {}           # streamed block index -> GPU tensors on their way or ready
        self.bypass = False          # FirstBlockCache: the blocks still to come in this transformer call do not run
        self.loaded = {}             # streamed block index -> its weights are on the GPU for the running forward
        resident = small = big = 0
        n_resident = 0
        notes = []
        for b in blocks:
            size = sum(p.numel() * p.element_size() for p in b.parameters())
            if resident + size <= budget:
                b.to(device)
                resident += size
                n_resident += 1
                continue
            for m in b.modules():
                for k, buf in list(m._buffers.items()):
                    if buf is not None:
                        m._buffers[k] = buf.to(device)
            params = []
            for p in b.parameters():
                nbytes = p.numel() * p.element_size()
                if nbytes < self.SMALL:
                    p.data = p.data.to(device)
                    small += nbytes
                else:
                    params.append(p)
                    big += nbytes
            pinned, note = pinned_copies([p.data for p in params])
            if note and note not in notes:
                notes.append(note)
            for p, c in zip(params, pinned):
                p.data = c
            i = len(self.streamed)
            self.streamed.append((params, pinned))
            b.register_forward_pre_hook(functools.partial(self._pre, i))
            b.register_forward_hook(functools.partial(self._post, i), always_call=True)
        gb = 1024 ** 3
        self.info = {"blocks": len(blocks), "resident_blocks": n_resident, "resident_gb": round(resident / gb, 2),
                     "streamed_gb_per_step": round(big / gb, 2), "small_on_gpu_gb": round(small / gb, 3),
                     "prefetch": self.side is not None, "exact_pinning": not notes}
        print(f"  [fast] FLUX blocks: {n_resident} of {len(blocks)} resident ({resident / gb:.1f} GB), "
              f"{big / gb:.1f} GB streamed per step from page-locked memory"
              f"{' with prefetch' if self.side is not None else ''}, {small / gb:.2f} GB of small tensors on the GPU")
        for note in notes:
            print(f"  [fast] exact-size page locking failed, torch's pinned allocator used instead ({note})")

    def _send(self, i):
        pinned = self.streamed[i][1]
        if self.side is None:
            self.inflight[i] = [c.to(self.device, non_blocking=True) for c in pinned]
        else:
            with self.torch.cuda.stream(self.side):
                self.inflight[i] = [c.to(self.device, non_blocking=True) for c in pinned]

    def _pre(self, i, module, args):
        if self.bypass:
            self.loaded[i] = False
            return
        self.loaded[i] = True
        if i not in self.inflight:
            self._send(i)
        gpu = self.inflight.pop(i)
        if self.side is not None:
            cur = self.torch.cuda.current_stream(self.device)
            cur.wait_stream(self.side)          # this block's kernels start after its copy
            for g in gpu:
                g.record_stream(cur)            # its memory is not handed out again before those kernels are done
        for p, g in zip(self.streamed[i][0], gpu):
            p.data = g
        if self.side is not None and i + 1 < len(self.streamed) and i + 1 not in self.inflight:
            self._send(i + 1)                   # copied while this block computes

    def _post(self, i, module, args, output):
        if not self.loaded.get(i):
            return
        for p, c in zip(*self.streamed[i]):
            p.data = c


class FirstBlockCache:
    """Skips 56 of FLUX's 57 blocks on steps that change little (first block cache, as in ParaAttention and diffusers).

    Every step runs the first block. When its residual (output minus input of the image stream) is within `threshold`
    of the residual at the last fully computed step (mean absolute difference over mean absolute value), the other
    blocks are not run and their combined residual from that step is added instead. Not exact: a larger threshold
    skips more steps and can soften fine detail, so the result has to be looked at. threshold None or 0: off.
    """

    def __init__(self, transformer, streamer=None):
        import torch
        self.torch = torch
        self.streamer = streamer
        self.reset(None)
        double = list(transformer.transformer_blocks)
        single = list(transformer.single_transformer_blocks)
        self._wrap(double[0], self._first)
        for b in double[1:]:
            self._wrap(b, self._double)
        for b in single[:-1]:
            self._wrap(b, self._single)
        self._wrap(single[-1], self._last)
        if streamer is not None:      # a new transformer call starts with every block due to run
            transformer.register_forward_pre_hook(lambda module, args: setattr(streamer, "bypass", False))

    def reset(self, threshold):
        """Start of a pass."""
        self.threshold = threshold or None
        self.first = None            # first block's residual at the last fully computed step
        self.tail = None             # (all blocks' output) - (first block's output) at that step, text and image tokens
        self.after_first = None
        self.skip = False
        self.count = [0, 0]          # steps computed in full, steps skipped
        if self.streamer is not None:
            self.streamer.bypass = False

    @staticmethod
    def _wrap(block, fn):
        orig = block.forward
        block.forward = lambda **k: fn(orig, k)

    def _first(self, orig, k):
        if not self.threshold:
            return orig(**k)
        hid_in = k["hidden_states"]
        enc, hid = orig(**k)
        res = hid - hid_in
        self.skip = False
        if self.first is not None and self.tail is not None and self.first.shape == res.shape:
            self.skip = ((res - self.first).abs().mean() / self.first.abs().mean()).item() < self.threshold
        if self.skip:
            self.count[1] += 1
        else:
            self.count[0] += 1
            self.first = res
            self.after_first = (enc, hid)
        if self.streamer is not None:
            self.streamer.bypass = self.skip
        return enc, hid

    def _double(self, orig, k):
        if self.threshold and self.skip:
            return k["encoder_hidden_states"], k["hidden_states"]
        return orig(**k)

    def _single(self, orig, k):
        if self.threshold and self.skip:
            return k["hidden_states"]
        return orig(**k)

    def _last(self, orig, k):
        if not self.threshold:
            return orig(**k)
        if self.skip:
            return k["hidden_states"] + self.tail
        out = orig(**k)
        self.tail = out - self.torch.cat(self.after_first, dim=1)
        self.after_first = None
        return out


def wrap_passes(pipe, texture_steps=None, delight_steps=None, pass_memory=False, cache=None, texture_cache=None,
                delight_cache=None, trim=True):
    """Per-pass step counts and cache thresholds, each pass's time, and with pass_memory its peak GPU memory.

    UniTEX's infer_mv calls the FLUX pipeline twice: the texture pass (with dual_image, the reference photo) and the
    delight pass (without).

    trim: torch.cuda.empty_cache() before each VAE decode and after each pass. accelerate.cpu_offload emptied the
    cache after every weight copy. With no emptying at all the cache keeps the loop's blocks, the decode's blocks and
    the second stream's blocks side by side and grows to the whole GPU (23.3 of 24.0 GB at 1024 px), which leaves
    nothing for the rasteriser and the bake, whose GPU memory does not come from torch.
    """
    import torch
    cls = type(pipe.pipeline)
    orig_call = cls.__call__
    state = {"decode_start": None}
    vae = pipe.pipeline.vae
    dec = vae.decode
    gb = 1024 ** 3

    def decode(*a, **k):
        if trim:
            torch.cuda.empty_cache()
        if pass_memory:
            torch.cuda.synchronize()
            state["decode_start"] = (time.perf_counter(), torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        return dec(*a, **k)
    vae.decode = decode

    def call(self, *a, **k):
        texture = k.get("dual_image") is not None
        n = texture_steps if texture else delight_steps
        if n:
            k["num_inference_steps"] = n
        steps = k.get("num_inference_steps")
        if cache is not None:
            cache.reset(texture_cache if texture else delight_cache)
        if pass_memory:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        state["decode_start"] = None
        t0 = time.perf_counter()
        try:
            return orig_call(self, *a, **k)
        finally:
            if pass_memory:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            note = ""
            if cache is not None and cache.threshold:
                note += (f", first block cache {cache.threshold}: {cache.count[0]} steps computed, "
                         f"{cache.count[1]} skipped")
            if pass_memory and state["decode_start"]:
                td, peak = state["decode_start"]
                note += (f", {(td - t0) / max(steps or 1, 1):.2f} s per step with the encodes, peak {peak / gb:.2f} GB "
                         f"before the VAE decode, VAE decode {t1 - td:.1f} s peak "
                         f"{torch.cuda.max_memory_allocated() / gb:.2f} GB")
            print(f">>> {'texture' if texture else 'delight'}_pass {t1 - t0} >>> {steps} steps, "
                  f"{k.get('width')}x{k.get('height')}, ended {time.strftime('%H:%M:%S')}{note}")
            if cache is not None:
                cache.reset(None)
            if trim:
                torch.cuda.empty_cache()
    cls.__call__ = call


def active_loras_only(pipe):
    """Adapters with weight 0 are switched off for the pass.

    UniTEX keeps all its adapters active and picks one per pass with the weights (texture pass: ours 1, the rest 0;
    delight pass: delight 1, the rest 0). PEFT evaluates every active adapter in each adapted layer and multiplies the
    result by its weight, so two of the three adapters are computed in all 342 layers for nothing. result + 0 * x is
    result, so leaving them out gives the same numbers.
    """
    orig = pipe.pipeline.set_adapters

    def set_adapters(adapter_names, adapter_weights=None):
        if (isinstance(adapter_names, (list, tuple)) and isinstance(adapter_weights, (list, tuple))
                and len(adapter_names) == len(adapter_weights)
                and all(isinstance(w, (int, float)) for w in adapter_weights)):
            keep = [(n, w) for n, w in zip(adapter_names, adapter_weights) if w != 0]
            if keep:
                adapter_names, adapter_weights = [n for n, _ in keep], [w for _, w in keep]
        return orig(adapter_names, adapter_weights)
    pipe.pipeline.set_adapters = set_adapters


def build_pipeline(args):
    import importlib
    import torch
    root = os.path.abspath(args.unitex_root)
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    unitex_pipeline = importlib.import_module("pipeline")
    unitex_pipeline.build_pipeline = lowmem.lowmem_build_pipeline
    pipe = lowmem._orig_build(args)
    if FAST.stream == "accelerate":
        if FAST.resident_gb:
            raise SystemExit("--resident-gb needs --stream pinned or pinned-serial")
        lowmem.stream_transformer(pipe)
    else:
        t0 = time.perf_counter()
        pipe.block_streamer = BlockStreamer(pipe.pipeline.transformer, torch.device("cuda"), FAST.resident_gb,
                                            prefetch=FAST.stream == "pinned")
        print(f"  [fast] weights placed in {time.perf_counter() - t0:.1f} s")
    if not FAST.all_loras:
        active_loras_only(pipe)
    cache = None
    if FAST.texture_cache or FAST.delight_cache:
        cache = FirstBlockCache(pipe.pipeline.transformer, getattr(pipe, "block_streamer", None))
    wrap_passes(pipe, FAST.texture_steps, FAST.delight_steps, FAST.pass_memory, cache, FAST.texture_cache,
                FAST.delight_cache, trim=FAST.stream != "accelerate" and not os.environ.get("UNITEX_FAST_NO_TRIM"))
    if FAST.no_bake:
        bake_steps.no_bake(pipe)
    else:
        if FAST.one_bake:
            bake_steps.one_bake(pipe)
        if FAST.roughness is not None:
            bake_steps.material(pipe, FAST.roughness)
    if FAST.fps_height:
        bake_steps.fps_height(FAST.fps_height)
    if os.environ.get("UNITEX_FAST_STACKS"):
        # Diagnosis: "seconds:file", the Python stacks of all threads every so often. Started here, after the imports:
        # a dump during them (an import hook was rewriting an AST) ended in a segfault once.
        import faulthandler
        every, _, path = os.environ["UNITEX_FAST_STACKS"].partition(":")
        faulthandler.dump_traceback_later(float(every), repeat=True, file=open(path or "fast_stacks.log", "w"))
    return pipe


def main(argv=None):
    global FAST, lowmem
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--stream", choices=("accelerate", "pinned", "pinned-serial"), default="pinned")
    ap.add_argument("--resident-gb", type=float, default=0.0)
    ap.add_argument("--texture-steps", type=int, default=None)
    ap.add_argument("--delight-steps", type=int, default=None)
    ap.add_argument("--texture-cache", type=float, default=None)
    ap.add_argument("--delight-cache", type=float, default=None)
    ap.add_argument("--all-loras", action="store_true")
    ap.add_argument("--one-bake", action="store_true")
    ap.add_argument("--no-bake", action="store_true")
    ap.add_argument("--fps-height", type=int, default=None)
    ap.add_argument("--roughness", type=float, default=None)
    ap.add_argument("--pass-memory", action="store_true")
    argv = sys.argv[1:] if argv is None else list(argv)
    if "-h" in argv or "--help" in argv:
        print(__doc__)                               # this file's options, then run_unitex.py's own help
    FAST, rest = ap.parse_known_args(argv)
    import run_unitex_lowmem
    lowmem = run_unitex_lowmem
    run_unitex.build_pipeline = build_pipeline       # after the import above, which puts its own builder there
    env_info = run_unitex.env_info

    def env_info_fast(args):
        info = env_info(args)
        info["fast"] = {k: v for k, v in vars(FAST).items() if v not in (None, False)}
        return info
    run_unitex.env_info = env_info_fast
    return run_unitex.main(rest)


if __name__ == "__main__":
    sys.exit(main())
