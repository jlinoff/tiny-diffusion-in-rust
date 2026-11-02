#!/usr/bin/env python3
"""
text_stats.py — measure generated or real text the same way (PLAN.md,
stage 6), so tiny-llm-in-rust's stories, the masked diffusion model's and
real ones can be compared.

For each file:
  words        the number of words (runs of letters, digits or
               apostrophes, lowercased; "<|endoftext|>" markers removed)
  repetition   the share of words equal to the word just before them
               ("the the"), within a story
  distinct-2   the share of different two-word pairs among all of them:
               1 if no pair ever repeats, lower for repetitive text

Stories are separated by blank lines or "<|endoftext|>"; pairs and
repetitions don't cross from one story to the next.

A file written by tiny-llm-in-rust's `transformer generate` (its header,
the story, then "Stopped at …" and the speed lines, once per run) is read
with --tiny-llm: only the stories are measured, and the decoding speed it
printed is averaged.

Usage:
  python3 tools/text_stats.py out/text-real.txt out/text-s16.txt ...
  python3 tools/text_stats.py --tiny-llm out/llm-stories.txt out/text-s64.txt
"""
import argparse
import re
import sys

WORD = re.compile(r"[a-z0-9']+")
SPLIT = re.compile(r"<\|endoftext\|>|\n\s*\n")


def stories(text):
    """The text's stories, each as its list of words."""
    out = []
    for part in SPLIT.split(text):
        words = WORD.findall(part.lower())
        if words:
            out.append(words)
    return out


SPEED = re.compile(r"^\s*decode:.*\(([0-9.]+) tokens/s\)", re.M)


def tiny_llm_stories(text):
    """The stories in tiny-llm-in-rust's generate output, joined by blank
    lines, and the mean of the decoding speeds it printed (None if none)."""
    out, keep, story = [], False, []
    for line in text.splitlines():
        if line.startswith("Prompt:"):
            keep, story = True, []
        elif line.startswith("Stopped at"):
            if keep:
                out.append("\n".join(story).strip())
            keep = False
        elif keep:
            story.append(line)
    speeds = [float(s) for s in SPEED.findall(text)]
    return "\n\n".join(out), (sum(speeds) / len(speeds) if speeds else None)


def measure(text):
    """(words, repetition, distinct-2) for a text."""
    words, repeats, pairs = 0, 0, []
    for s in stories(text):
        words += len(s)
        repeats += sum(1 for a, b in zip(s, s[1:]) if a == b)
        pairs.extend(zip(s, s[1:]))
    repetition = repeats / words if words else 0.0
    distinct = len(set(pairs)) / len(pairs) if pairs else 0.0
    return words, repetition, distinct


def main(argv=None):
    """Measure each file named on the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    p.add_argument("files", nargs="*")
    p.add_argument("--tiny-llm", action="append", default=[], metavar="FILE",
                   help="a file of tiny-llm-in-rust's generate output")
    a = p.parse_args(argv)
    print(f"{'file':<40} {'words':>8} {'repetition':>11} {'distinct-2':>11}")
    for f in a.tiny_llm + a.files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as e:
            print(f"text_stats: {e}", file=sys.stderr)
            return 1
        note = ""
        if f in a.tiny_llm:
            text, speed = tiny_llm_stories(text)
            if speed is not None:
                note = f"   (tiny-llm-in-rust, decoding at {speed:.0f} tokens/s)"
        w, r, d = measure(text)
        print(f"{f:<40} {w:>8} {100 * r:>10.2f}% {d:>11.3f}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
