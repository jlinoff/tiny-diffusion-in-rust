#!/usr/bin/env python3
"""
reference.py: the official code's outputs for stage 7's check (PLAN.md,
stage 7, prediction 2), in the same format as the test fixtures, so
`diffusion dit-check` compares them with this project's number for number.

  models/reference-dit.safetensors: two latents at two times with two
      classes (one of them "no label"); after the patch embedding, after
      the first and last blocks, and the output (ε and the variance)
  models/reference-vae.safetensors: one latent and its decoded image

Run by diffusers in f32 on the CPU: the DiT built from its configuration
with the weights convert.py wrote (so the PyTorch pickle is never loaded
here), the autoencoder from its own safetensors file.

Usage:
  python3 tools/stage7/reference.py models
"""

import os
import sys

import torch
from diffusers import AutoencoderKL, Transformer2DModel
from safetensors.torch import load_file, save_file


def dit(out):
    """DiT-XL/2 on fixed inputs, keeping the intermediate results."""
    # Built from its configuration, with the weights from the converted
    # file: the PyTorch pickle is opened only by convert.py's safe loader.
    config = Transformer2DModel.load_config(os.path.join(out, "DiT-XL-2-256", "transformer"))
    model = Transformer2DModel.from_config(config)
    model.load_state_dict(load_file(os.path.join(out, "dit-xl-2-256.safetensors")))
    model = model.float().eval()
    cfg = model.config
    size, classes = cfg.sample_size, cfg.num_embeds_ada_norm
    latent = torch.randn(2, cfg.in_channels, size, size)
    times = torch.tensor([50, 700])
    labels = torch.tensor([min(207, classes - 1), classes])
    inter = {}

    def keep(name):
        def hook(_module, _inputs, output):
            inter[name] = output.detach().clone()
        return hook

    last = len(model.transformer_blocks) - 1
    model.pos_embed.register_forward_hook(keep("inter.patches"))
    model.transformer_blocks[0].register_forward_hook(keep("inter.block0"))
    model.transformer_blocks[last].register_forward_hook(keep(f"inter.block{last}"))
    with torch.no_grad():
        result = model(latent, timestep=times, class_labels=labels).sample
    tensors = dict(inter)
    tensors.update({"input.latent": latent, "input.times": times.float(),
                    "input.labels": labels.float(), "output": result})
    path = os.path.join(out, "reference-dit.safetensors")
    save_file({k: v.float().contiguous() for k, v in tensors.items()}, path)
    print(f"wrote {path}")


def vae(out):
    """The decoder on one latent."""
    model = AutoencoderKL.from_pretrained(os.path.join(out, "sd-vae-ft-ema")).float().eval()
    latent = torch.randn(1, model.config.latent_channels, 32, 32)
    with torch.no_grad():
        image = model.decode(latent).sample
    path = os.path.join(out, "reference-vae.safetensors")
    save_file({"input.latent": latent.contiguous(), "output": image.contiguous()}, path)
    print(f"wrote {path}")


def main():
    """Both references, from a fixed seed."""
    out = sys.argv[1] if len(sys.argv) > 1 else "models"
    torch.manual_seed(1)
    dit(out)
    vae(out)


if __name__ == "__main__":
    main()
