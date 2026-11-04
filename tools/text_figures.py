#!/usr/bin/env python3
"""
text_figures.py: stage 6's text figures, as SVG (PLAN.md, stage 6).

  masking    one sentence masked at t = 1, 0.75, 0.5, 0.25 and 0: text's
             noise, as the noise-to-image figures show an image's
  settling   an infill step by step, from the trace sample-text writes
             (<prefix>-infill-trace.tsv): the gap at chosen moments, the
             story around it, the tokens settled so far, the newest
             highlighted

Usage:
  python3 tools/text_figures.py masking -o docs/stage6/masking.svg
  python3 tools/text_figures.py settling out/text-infill-trace.tsv -o docs/stage6/settling.svg
"""

import argparse
import random
import sys

FONT = 13
CHAR = 7.8          # the width of one character of the monospace font
SENTENCE = "Once there was a little boy named Josh who was three years old ."
MASK = "MASK"
TEXT_X = 178      # where the settling figure's text starts, after its labels


def esc(text):
    """Text made safe inside SVG: markup characters escaped, and control
    characters (which XML forbids; a token can be a lone byte) shown as
    the replacement character."""
    text = "".join(c if c >= " " or c in "\t\n" else "\ufffd" for c in text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def masked_rows(words, times, seed=3):
    """For each t, which words are masked: each word draws one number u,
    and is masked when u < 1 - t, so a word masked at one t stays masked at
    every smaller t, as in training's forward process."""
    rng = random.Random(seed)
    u = [rng.random() for _ in words]
    return [[ui < 1 - t for ui in u] for t in times]


def word_row(y, words, widths, masks, left):
    """One row of the masking figure: a box per word, MASK where masked."""
    out, x = [], left
    for w, wd, m in zip(words, widths, masks):
        fill, colour, shown = ("#d9d9d9", "#666", MASK) if m else ("#eaf2fb", "#123", w)
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{wd:.1f}" height="28" rx="4" '
                   f'fill="{fill}" stroke="#9ab"/>')
        out.append(f'<text x="{x + wd / 2:.1f}" y="{y + 19}" text-anchor="middle" '
                   f'fill="{colour}">{esc(shown)}</text>')
        x += wd + 6
    return out


def svg_open(width, height, title):
    """An SVG's opening: its size, a white background, its title."""
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="monospace" font-size="{FONT}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            '<text x="16" y="24" font-family="sans-serif" font-size="14" font-weight="bold">'
            f"{esc(title)}</text>"]


def masking_svg(sentence=SENTENCE, times=(1.0, 0.75, 0.5, 0.25, 0.0)):
    """The masking figure: a row of word boxes per t."""
    words = sentence.split()
    widths = [max(len(w), len(MASK)) * CHAR + 12 for w in words]
    left, top, row_h = 150, 46, 40
    width = int(left + sum(widths) + 6 * len(words) + 20)
    height = top + row_h * len(times) + 34
    out = svg_open(width, height, "Text's noise: each token masked with probability 1 − t")
    for r, (t, masks) in enumerate(zip(times, masked_rows(words, times))):
        y = top + r * row_h
        label = {1.0: "t = 1, the text", 0.0: "t = 0, all MASK"}.get(t, f"t = {t:g}")
        out.append(f'<text x="16" y="{y + 20}" font-family="sans-serif">{label}</text>')
        out += word_row(y, words, widths, masks, left)
    out.append(f'<text x="16" y="{height - 12}" font-family="sans-serif" fill="#555">'
               "Words stand in for tokens here; the model's tokens are often pieces of "
               "words. A token masked at one t stays masked at every smaller t.</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def read_trace(path):
    """(steps, order, [(position, settled step or -1, token text)])."""
    steps, order, rows = 0, "", []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# steps"):
                parts = line.split("\t")
                steps, order = int(parts[1]), parts[3]
            elif line and not line.startswith("#"):
                pos, at, tok = line.split("\t", 2)
                tok = tok.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
                rows.append((int(pos), int(at), tok))
    return steps, order, rows


def snapshot(rows, done):
    """The text after `done` steps, as (text, kind) pieces: kind is
    'given', 'settled', 'new' (settled in the last stretch) or 'masked'."""
    pieces = []
    for _, at, tok in rows:
        if at < 0:
            pieces.append((tok, "given"))
        elif at < done:
            pieces.append((tok, "settled"))
        else:
            pieces.append(("▒", "masked"))
    return pieces


def wrap(pieces, chars):
    """Pieces in lines of at most `chars` characters, newlines kept as
    spaces (the figure shows a stretch of text, not its layout)."""
    lines, line, n = [], [], 0
    for text, kind in pieces:
        text = text.replace("\n", " ")
        if n + len(text) > chars and line:
            lines.append(line)
            line, n = [], 0
            text = text.lstrip()
        line.append((text, kind))
        n += len(text)
    if line:
        lines.append(line)
    return lines


COLOURS = {"given": "#222", "settled": "#1f6fb4", "new": "#e07b00", "masked": "#bbb"}


def moments(shown, steps, fractions, chars):
    """For each fraction of the steps: (fraction, steps done, wrapped lines),
    tokens settled since the moment before marked new."""
    blocks, prev = [], 0
    for f in fractions:
        done = round(f * steps)
        pieces = [(t, "new" if k == "settled" and prev <= shown[i][1] < done else k)
                  for i, (t, k) in enumerate(snapshot(shown, done))]
        blocks.append((f, done, wrap(pieces, chars)))
        prev = done
    return blocks


def draw_moments(blocks, y, line_h):
    """Each moment's label and its lines of coloured text, from y down."""
    out = []
    for f, done, lines in blocks:
        word = "step" if done == 1 else "steps"
        out.append(f'<text x="16" y="{y + 12}" font-family="sans-serif">'
                   f'after {done} {word} ({f:.0%})</text>')
        for line in lines:
            spans = "".join(f'<tspan fill="{COLOURS[k]}"'
                            + (' font-weight="bold"' if k == "new" else "")
                            + f'>{esc(t)}</tspan>' for t, k in line)
            out.append(f'<text x="{TEXT_X}" y="{y + 12}" xml:space="preserve">{spans}</text>')
            y += line_h
        y += 16
    return out


def settling_svg(trace, fractions, context, chars=96):
    """The settling figure from a trace (steps, order, rows): the gap and
    `context` tokens either side, at each fraction of the steps, the newest
    tokens in orange."""
    steps, order, rows = trace
    gap = [i for i, (_, at, _) in enumerate(rows) if at >= 0]
    shown = rows[max(0, gap[0] - context):min(len(rows), gap[-1] + 1 + context)]
    blocks = moments(shown, steps, fractions, chars)
    line_h = 18
    width = int(TEXT_X + chars * CHAR + 20)
    height = 52 + sum(len(b[2]) * line_h + 16 for b in blocks) + 30
    out = svg_open(width, height, f"Infilling a story's middle, {order} order, {steps} steps")
    out += draw_moments(blocks, 52, line_h)
    out.append(f'<text x="16" y="{height - 12}" font-family="sans-serif" fill="#555">'
               "black: the story around the gap; ▒: still masked; blue: settled; "
               "orange: settled since the moment above.</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv=None):
    """The two figures, by subcommand."""
    p = argparse.ArgumentParser(description="Stage 6's text figures, as SVG.")
    sub = p.add_subparsers(dest="figure", required=True)
    m = sub.add_parser("masking", help="one sentence masked at several t")
    m.add_argument("-o", "--output", required=True)
    s = sub.add_parser("settling", help="an infill step by step, from its trace")
    s.add_argument("trace")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--moments", default="0,0.1,0.25,0.5,0.75,1",
                   help="fractions of the steps to show, comma-separated")
    s.add_argument("--context", type=int, default=12, help="tokens of story shown each side")
    a = p.parse_args(argv)
    if a.figure == "masking":
        text = masking_svg()
    else:
        fractions = [float(x) for x in a.moments.split(",")]
        text = settling_svg(read_trace(a.trace), fractions, a.context)
    with open(a.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
