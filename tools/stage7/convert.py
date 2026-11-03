#!/usr/bin/env python3
"""
convert.py: turn stage 7's downloads into the f32 safetensors files the
Rust code reads (PLAN.md, stage 7).

  models/DiT-XL-2-256/transformer/diffusion_pytorch_model.bin
      read with PyTorch's safe loader (weights_only=True: tensors only,
      the file can't run code), counted, the 28 blocks' copies of the time
      and class embedding checked identical (prediction 1)
      -> models/dit-xl-2-256.safetensors
  models/sd-vae-ft-ema/
      read by diffusers (which renames its older attention weights),
      the decoder's tensors kept
      -> models/sd-vae-ft-ema-decoder.safetensors
  models/DiT-XL-2-256/model_index.json
      the 1,000 ImageNet class names, one per line, line n = class n
      -> models/imagenet_classes.txt

Usage:
  python3 tools/stage7/convert.py models
"""

import json
import os
import sys

import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file

EMBED = "norm1.emb."


def check_copies(state):
    """Count the values, and check that the 28 blocks' copies of the time
    and class embedding are identical (prediction 1)."""
    total = sum(v.numel() for v in state.values())
    copies = {k: v for k, v in state.items() if EMBED in k}
    per_block = sum(v.numel() for k, v in copies.items() if k.startswith("transformer_blocks.0."))
    blocks = len({k.split(".")[1] for k in copies})
    print(f"DiT-XL/2: {total:,} values in {len(state)} tensors")
    print(f"  the time and class embedding: {per_block:,} values, copied into {blocks} blocks")
    print(f"  without the extra copies: {total - (blocks - 1) * per_block:,}"
          " (the paper: about 675 million)")
    same = True
    for k, v in copies.items():
        if not torch.equal(v, state["transformer_blocks.0." + k.split(".", 2)[2]]):
            same = False
            print(f"  differs from block 0's copy: {k}")
    print(f"  the copies are {'identical' if same else 'NOT identical'}")


def convert_dit(out):
    """The DiT: read safely, count, check the copies, write."""
    folder = os.path.join(out, "DiT-XL-2-256", "transformer")
    state = torch.load(
        os.path.join(folder, "diffusion_pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    check_copies(state)
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        config = f.read()
    tensors = {k: v.to(torch.float32).contiguous() for k, v in state.items()}
    path = os.path.join(out, "dit-xl-2-256.safetensors")
    save_file(tensors, path, metadata={"config": config, "source": "facebook/DiT-XL-2-256",
                                       "licence": "CC BY-NC 4.0"})
    print(f"wrote {path}")


def convert_vae(out):
    """The decoder, under diffusers' current names."""
    folder = os.path.join(out, "sd-vae-ft-ema")
    vae = AutoencoderKL.from_pretrained(folder)
    tensors = {
        k: v.to(torch.float32).contiguous()
        for k, v in vae.state_dict().items()
        if k.startswith(("decoder.", "post_quant_conv."))
    }
    path = os.path.join(out, "sd-vae-ft-ema-decoder.safetensors")
    save_file(tensors, path, metadata={"config": json.dumps(dict(vae.config)),
                                       "source": "stabilityai/sd-vae-ft-ema", "licence": "MIT"})
    count = sum(v.numel() for v in tensors.values())
    print(f"autoencoder's decoder: {count:,} values; wrote {path}")


def class_names(out):
    """ImageNet's class names from the pipeline's index, in class order."""
    with open(os.path.join(out, "DiT-XL-2-256", "model_index.json"), encoding="utf-8") as f:
        index = json.load(f)
    labels = index["id2label"]
    path = os.path.join(out, "imagenet_classes.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(labels)):
            f.write(labels[str(i)] + "\n")
    print(f"wrote {path}: {len(labels)} class names")


def main():
    """Convert both models and write the class names."""
    out = sys.argv[1] if len(sys.argv) > 1 else "models"
    convert_dit(out)
    convert_vae(out)
    class_names(out)


if __name__ == "__main__":
    main()
