#!/bin/bash
# Every command, end to end, on a tiny model in seconds: a check that each
# step still runs and hands its files to the next (used by
# .github/workflows/main.yaml and make smoke; runs locally too, from the top
# of the repository). Nothing learned is worth keeping; everything goes in a
# temporary folder.
set -euo pipefail
D=diffusion/target/release/diffusion
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
step() { printf '\n== %s ==\n' "$1"; }

step info
$D info
step exact
$D exact --steps 1,10 --n 2000
for obj in flow ddpm; do
    step "train2d ($obj)"
    $D train2d --data moons --objective $obj --steps 200 --width 32 --log-every 100 -o "$T/m-$obj.safetensors"
    step "sample2d ($obj)"
    $D sample2d --model "$T/m-$obj.safetensors" --n 300 --steps 8 --svg "$T/m-$obj.svg"
    step "steps2d ($obj)"
    $D steps2d --model "$T/m-$obj.safetensors" --n 300 --steps 1,4 --svg "$T/m-$obj-steps.svg"
done
$D sample2d --model "$T/m-ddpm.safetensors" --n 300 --sampler ancestral > /dev/null
step "train-img (shapes)"
$D train-img --data shapes --size 8 --channels 3 --patch 4 --width 16 --blocks 1 --heads 2 \
    --steps 20 --batch 8 --warmup 5 --log-every 10 --save-every 10 -o "$T/img.safetensors"
step "sample-img"
$D sample-img --model "$T/img.safetensors" --n 4 --steps 4 --png "$T/img.png" \
    --progress "$T/img-progress.png" --real "$T/img-real.png"
python3 -c 'import sys; sys.exit(open(sys.argv[1], "rb").read(8) != b"\x89PNG\r\n\x1a\n")' "$T/img.png"
grep -q '<circle' "$T/m-flow.svg"
printf '\nEvery command ran.\n'
