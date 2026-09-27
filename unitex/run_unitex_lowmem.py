"""
run_unitex.py for GPUs with less memory than bf16 FLUX needs (e.g. a shared 24 GB RTX 4090).

Stock UniTEX builds FLUX.1-dev in bf16 on one GPU when it has more than 30 GiB, and otherwise
falls back to an NF4-quantized transformer (a different model, and the branch crashes on an
unbound `cpu_offload`). This wrapper keeps the released bf16 weights and LoRAs unchanged and
instead streams the transformer's weights from CPU RAM to the GPU module by module
(accelerate.cpu_offload), so the GPU holds only activations, the VAE and one block at a time.
Outputs are the same model; only speed changes.

Usage (same arguments as run_unitex.py):
  CUDA_VISIBLE_DEVICES=2 python unitex/run_unitex_lowmem.py --unitex-root /path/UniTEX --eval-dir EVAL ...
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_unitex  # noqa: E402


def lowmem_build_pipeline(pretrain_models=None, pipeline_name='texture_plus', model='rgb', super_resolutions=False,
                          add_lora_path=None, add_lora_weights=None, speedup_mode=False):
    """Mirror of UniTEX pipeline.build_pipeline (commit affa1e2) with bf16 weights streamed from CPU."""
    import torch
    from accelerate import cpu_offload
    from diffusers import FluxTransformer2DModel
    from huggingface_hub import hf_hub_download
    from flux_piplines.texturing.pipeline import PBRFluxPipeline as FluxPipeline

    transformer = FluxTransformer2DModel.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder='transformer', torch_dtype=torch.bfloat16)
    pipeline = FluxPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev', transformer=transformer,
        text_encoder=None, text_encoder_2=None, torch_dtype=torch.bfloat16)
    lora_id = hf_hub_download(repo_id='lyxun/UniTEX', filename='mv_lora_weights.safetensors', repo_type='model')
    lora_id_delight = hf_hub_download(repo_id='lyxun/UniTEX', filename='delight_lora_weights.safetensors',
                                      repo_type='model')
    pipeline.load_lora_weights(lora_id, adapter_name='texture')
    pipeline.load_lora_weights(lora_id_delight, adapter_name='delight')
    weights_for_texture = [1., 0.]
    weights_for_delight = [0., 1.]
    adapter_names = ['texture', 'delight']
    if add_lora_path is not None:
        for i in range(len(add_lora_path)):
            pipeline.load_lora_weights(add_lora_path[i], adapter_name=f'add_lora_{i}')
            adapter_names.append(f'add_lora_{i}')
            weights_for_texture.append(add_lora_weights[i])
            weights_for_delight.append(add_lora_weights[i])
    pipeline.transformer.__class__.__name__ = 'FluxTransformer2DModel'
    pipeline._num_inference_steps = 28
    # the VAE stays resident: PBRFluxPipeline moves latents to self.vae.device before decoding
    pipeline.vae.to('cuda')
    cpu_offload(pipeline.transformer, execution_device=torch.device('cuda'))
    print('  [lowmem] FLUX transformer streamed from CPU (bf16, LoRAs loaded), VAE on cuda')
    return pipeline, weights_for_texture, weights_for_delight, adapter_names


_orig_build = run_unitex.build_pipeline


def build_pipeline(args):
    # run_unitex chdirs into the UniTEX root before importing its pipeline module
    import importlib
    root = os.path.abspath(args.unitex_root)
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    unitex_pipeline = importlib.import_module('pipeline')
    unitex_pipeline.build_pipeline = lowmem_build_pipeline
    return _orig_build(args)


run_unitex.build_pipeline = build_pipeline

if __name__ == '__main__':
    sys.exit(run_unitex.main())
