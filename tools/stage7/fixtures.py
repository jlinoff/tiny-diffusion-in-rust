#!/usr/bin/env python3
"""
fixtures.py: build tiny DiT and decoder models in diffusers, with random
weights made here, and save what the Rust code must reproduce.

Stage 7 (PLAN.md) runs DiT-XL/2 and its decoder through this project's
code. Those weights can't be in the repository or in continuous
integration, so these fixtures stand in for them: the same layer types,
names, settings and layout as the real configs (attention biases,
adaLN-Zero, GELU's tanh form, 2 × 2 patches, a decoder of residual
blocks, one attention layer and upsampling), at a tiny size, with
random weights. They carry no licence but this project's.

Each file holds the model's weights under diffusers' own names, plus:
  input.*   the inputs it was run on
  inter.*   intermediate results (DiT: after the patch embedding and
            after each block)
  output    the result
and the model's configuration as JSON in the metadata.

Usage (needs PyTorch, diffusers and safetensors):
  python3 tools/stage7/fixtures.py diffusion/tests/data
"""

import json
import sys

import torch
from diffusers import AutoencoderKL, Transformer2DModel
from safetensors.torch import save_file

DIT_CONFIG = {
    # DiT-XL/2's configuration, at a tiny size: width 16 (2 heads of 8),
    # 2 blocks, 10 classes, an 8 × 8 latent.
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "in_channels": 4,
    "out_channels": 8,
    "num_layers": 2,
    "sample_size": 8,
    "patch_size": 2,
    "norm_type": "ada_norm_zero",
    "num_embeds_ada_norm": 10,
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "norm_elementwise_affine": False,
}

VAE_CONFIG = {
    # sd-vae-ft-ema's configuration, at a tiny size: widths 8 and 16
    # instead of 128 to 512, two blocks instead of four, groups of 4.
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ["DownEncoderBlock2D"] * 2,
    "up_block_types": ["UpDecoderBlock2D"] * 2,
    "block_out_channels": [8, 16],
    "layers_per_block": 2,
    "latent_channels": 4,
    "norm_num_groups": 4,
    "sample_size": 16,
}


def perturb(model, scale):
    """Add noise to every weight, so no layer starts as the identity or
    zero (adaLN-Zero's would otherwise hide mistakes behind closed gates)."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn_like(p))


def dit_fixture():
    """The tiny DiT, run on two latents, two times and two labels (one of
    them the "no label" class, index 10)."""
    model = Transformer2DModel(**DIT_CONFIG).eval()
    perturb(model, 0.1)
    latent = torch.randn(2, 4, 8, 8)
    times = torch.tensor([10, 500])
    labels = torch.tensor([3, 10])
    inter = {}

    def keep(name):
        def hook(_module, _inputs, output):
            inter[name] = output.detach().clone()
        return hook

    model.pos_embed.register_forward_hook(keep("inter.patches"))
    for i, block in enumerate(model.transformer_blocks):
        block.register_forward_hook(keep(f"inter.block{i}"))
    with torch.no_grad():
        out = model(latent, timestep=times, class_labels=labels).sample
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    tensors.update(inter)
    tensors["input.latent"] = latent
    tensors["input.times"] = times.float()
    tensors["input.labels"] = labels.float()
    tensors["output"] = out
    return tensors, DIT_CONFIG


def vae_fixture():
    """The tiny decoder (and post_quant_conv before it), on one latent."""
    model = AutoencoderKL(**VAE_CONFIG).eval()
    perturb(model, 0.05)
    latent = torch.randn(1, 4, 4, 4)
    with torch.no_grad():
        out = model.decode(latent).sample
    tensors = {
        k: v.contiguous()
        for k, v in model.state_dict().items()
        if k.startswith(("decoder.", "post_quant_conv."))
    }
    tensors["input.latent"] = latent
    tensors["output"] = out
    return tensors, VAE_CONFIG


def save(tensors, config, path):
    """f32 safetensors, the configuration in the metadata."""
    tensors = {k: v.to(torch.float32).contiguous() for k, v in tensors.items()}
    save_file(tensors, path, metadata={"config": json.dumps(config)})
    weights = [v for k, v in tensors.items() if "." in k and not k.startswith(("input.", "inter."))]
    count = sum(v.numel() for v in weights)
    print(f"wrote {path}: {count} weights, {len(tensors)} tensors")


def main():
    """Write dit-tiny.safetensors and vae-tiny.safetensors into the folder
    given."""
    out = sys.argv[1] if len(sys.argv) > 1 else "diffusion/tests/data"
    torch.manual_seed(7)
    save(*dit_fixture(), f"{out}/dit-tiny.safetensors")
    save(*vae_fixture(), f"{out}/vae-tiny.safetensors")


if __name__ == "__main__":
    main()
