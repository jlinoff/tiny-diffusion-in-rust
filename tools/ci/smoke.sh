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
step "classifier (shapes)"
$D classifier --data shapes --size 8 --channels 3 --hidden 16 --steps 20 --log-every 10 -o "$T/clf.safetensors"
step "measure"
$D measure --model "$T/img.safetensors" --classifier "$T/clf.safetensors" --n 20 --steps 4
step "train-img with labels (shapes)"
$D train-img --data shapes --size 8 --channels 3 --patch 4 --width 16 --blocks 1 --heads 2 \
    --steps 20 --batch 8 --warmup 5 --log-every 10 --save-every 0 --conditional -o "$T/cond.safetensors"
step "sample-img with labels and guidance"
$D sample-img --model "$T/cond.safetensors" --classes --n 2 --steps 4 --guidance 2 --png "$T/cond.png"
$D sample-img --model "$T/cond.safetensors" --label 1 --n 2 --steps 4 --png "$T/cond1.png"
step "measure with guidance"
$D measure --model "$T/cond.safetensors" --classifier "$T/clf.safetensors" --n 12 --steps 4 --guidance 0,1,3
step "style"
$D style --model "$T/cond.safetensors" --n 4 --steps 4
step "inside"
$D inside --model "$T/img.safetensors" --classifier "$T/clf.safetensors" --n 6 --steps 4 \
    --preview "$T/prev.png" --attention "$T/att.png"
$D inside --model "$T/cond.safetensors" --classifier "$T/clf.safetensors" --n 6 --steps 4 > /dev/null
step "train-text (tiny-llm-in-rust's fixture)"
$D train-text --ids diffusion/tests/data/tiny.ids --width 16 --blocks 1 --heads 2 --context 32 \
    --ffn 32 --steps 20 --log-every 10 --draws 2 -o "$T/text.safetensors"
step "sample-text"
$D sample-text --model "$T/text.safetensors" --tokenizer diffusion/tests/data/tiny.bpe \
    --ids diffusion/tests/data/tiny.ids --steps 2,8 --n 2 --out-prefix "$T/text"
python3 tools/text_stats.py "$T/text-real.txt" --tiny-llm "$T/text-s2.txt" "$T/text-s8.txt"
python3 -c 'import sys; sys.exit(open(sys.argv[1], "rb").read(8) != b"\x89PNG\r\n\x1a\n")' "$T/img.png"
grep -q '<circle' "$T/m-flow.svg"
printf '\nEvery command ran.\n'
