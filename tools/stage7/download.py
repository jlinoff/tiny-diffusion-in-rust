#!/usr/bin/env python3
"""
download.py: fetch stage 7's two models from their own sources into
models/, after showing their licences (PLAN.md, stage 7).

  DiT-XL/2 (facebook/DiT-XL-2-256): CC BY-NC 4.0, non-commercial use only
  its autoencoder (stabilityai/sd-vae-ft-ema): MIT

Nothing is downloaded unless DIT_LICENSE=accept is set: running this is
agreeing to use DiT-XL/2 under its licence.

Usage:
  DIT_LICENSE=accept python3 tools/stage7/download.py models
"""

import os
import sys

from huggingface_hub import hf_hub_download

TERMS = """\
Stage 7 downloads two models, each under its own licence:

  DiT-XL/2, from facebook/DiT-XL-2-256 (Peebles and Xie 2023)
    CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
    Non-commercial use only (learning, teaching, research), with credit.
    Treat the images it generates as non-commercial too.

  The autoencoder, from stabilityai/sd-vae-ft-ema
    MIT: any use, keeping its copyright and licence notice.

Neither is part of this repository. PLAN.md, stage 7, sets out the terms.
This is a reading of the published licences, not legal advice.
"""

FILES = [
    ("facebook/DiT-XL-2-256", "transformer/config.json"),
    ("facebook/DiT-XL-2-256", "transformer/diffusion_pytorch_model.bin"),
    ("facebook/DiT-XL-2-256", "model_index.json"),
    ("stabilityai/sd-vae-ft-ema", "config.json"),
    ("stabilityai/sd-vae-ft-ema", "diffusion_pytorch_model.safetensors"),
]


def main():
    """Show the terms; download only if they were accepted."""
    out = sys.argv[1] if len(sys.argv) > 1 else "models"
    print(TERMS)
    if os.environ.get("DIT_LICENSE") != "accept":
        print("Not downloading. To accept these terms and download:")
        print("  make dit-download DIT_LICENSE=accept")
        sys.exit(1)
    for repo, name in FILES:
        folder = os.path.join(out, repo.split("/")[1])
        path = hf_hub_download(repo, name, local_dir=folder)
        print(f"{repo}: {name} -> {path}")


if __name__ == "__main__":
    main()
