"""
Stage 2 integration checks that cross component boundaries (trainer output -> UniTEX inference).

  run_unitex.use_texture_lora: our trained texture LoRA replaces the released texture adapter in
  the texture pass and is off in the delight pass (UniTEX's own --add-lora-path stacks on both).

Run: python -m pytest unitex/tests/test_stage2_integration.py -v
"""

import os
import sys
import types

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.join(REPO, "unitex"))
import run_unitex  # noqa: E402
from unitex.tests import test_unitex_infer as T  # noqa: E402

repos = T.repos          # session fixture: pristine / patched UniTEX sources


class LoraFlux(T.FakeFlux):
    """FakeFlux that also records load_lora_weights."""

    def load_lora_weights(self, path, adapter_name=None):
        self.calls.append(("load", path, adapter_name))


def test_texture_lora_replaces_released_texture_adapter_in_texture_pass_only(repos, tmp_path):
    flux = LoraFlux()
    C = T.unitex_classes(repos["patched"], flux=flux)
    pipe = C.CustomRGBTextureFullPipeline(seed=0)
    assert pipe.adapter_names == ["texture", "delight"]
    run_unitex.use_texture_lora(pipe, "/ckpt/ours.safetensors")
    assert ("load", "/ckpt/ours.safetensors", "cfi_texture") in flux.calls
    assert pipe.adapter_names == ["texture", "delight", "cfi_texture"]
    assert pipe.weights_for_texture == [0.0, 0.0, 1.0]
    assert pipe.weights_for_delight == [0.0, 1.0, 0.0]
    pipe.generator = torch.Generator().manual_seed(0)
    pipe.infer_mv(str(tmp_path), *T._grids(str(tmp_path), 512))
    sets = [c[1] for c in flux.calls if c[0] == "set_adapters"]
    assert sets == [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]       # texture pass, then delight pass


def test_texture_lora_flag_is_logged_and_dry_run_ok(tmp_path):
    import json
    import shutil
    ev = tmp_path / "eval"
    sku = T.EVAL_SKUS[0]
    os.makedirs(ev / sku)
    for f in ("ref.png", "mesh.glb"):
        src = os.path.join(T.EVAL28, sku, f)
        if not os.path.isfile(src):
            pytest.skip(f"eval set not found at {T.EVAL28}")
        shutil.copy(src, ev / sku / f)
    (ev / "skus.txt").write_text(sku + "\n")
    run_unitex.main(["--eval-dir", str(ev), "--dry-run", "--run-name", "r", "--texture-lora", "/ckpt/ours.safetensors"])
    rec = json.loads((ev / "run_log.jsonl").read_text().splitlines()[-1])
    assert rec["status"] == "ok" and rec["texture_lora"] == "/ckpt/ours.safetensors"
    # without the flag the record keeps the stock key set
    run_unitex.main(["--eval-dir", str(ev), "--dry-run", "--run-name", "r2"])
    rec2 = json.loads((ev / "run_log.jsonl").read_text().splitlines()[-1])
    assert "texture_lora" not in rec2
