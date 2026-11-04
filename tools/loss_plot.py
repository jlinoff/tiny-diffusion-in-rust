#!/usr/bin/env python3
"""
loss_plot.py: draw training-loss curves from a CSV file as an SVG figure.

The CSV's first column is the step; each further column is one run's
loss. Lines starting with '#' are comments (where the numbers came from).
A run's name sets its line: "control" in the name draws it dashed, "rope"
in orange, "learned" in blue (stage 6's four runs), other runs in turn
from a palette. Reference levels are drawn as thin grey lines.

Usage:
  python3 tools/loss_plot.py docs/stage6/loss-2000.csv -o docs/stage6/loss-2000.svg \\
      --title "..." --ref "6.03:knowing only how common each token is"
"""

import argparse
import csv
import sys

WIDTH, HEIGHT = 780, 420
LEFT, RIGHT, TOP, BOTTOM = 64, 250, 40, 50
PALETTE = ["#1f6fb4", "#e07b00", "#2a9d4b", "#b03a8c", "#6b6b6b"]
LABELS = {
    "masked-learned": "masked, learned positions",
    "control-learned": "next-token control, learned positions",
    "masked-rope": "masked, RoPE",
    "control-rope": "next-token control, RoPE",
}


def read_runs(path):
    """The steps and, for each run, its losses, from the CSV at path."""
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.reader(line for line in f if not line.startswith("#")) if r]
    header, body = rows[0], rows[1:]
    steps = [int(r[0]) for r in body]
    runs = {name: [float(r[i + 1]) for r in body] for i, name in enumerate(header[1:])}
    return steps, runs


def style(name, index):
    """Colour and dash pattern for a run, from its name."""
    if "rope" in name:
        colour = "#e07b00"
    elif "learned" in name:
        colour = "#1f6fb4"
    else:
        colour = PALETTE[index % len(PALETTE)]
    dash = ' stroke-dasharray="7 4"' if "control" in name else ""
    return colour, dash


def nice_ticks(low, high, count=6):
    """Round tick values covering low to high."""
    span = high - low
    step = 10 ** len(str(int(span))) / 10 if span >= 1 else 0.1
    for mult in (0.1, 0.2, 0.25, 0.5, 1, 2, 5):
        if span / (step * mult) <= count:
            step *= mult
            break
    first = int(low / step) * step
    ticks = []
    t = first
    while t <= high + 1e-9:
        if t >= low - 1e-9:
            ticks.append(round(t, 6))
        t += step
    return ticks


class Frame:
    """The plot area, and how steps and losses map onto it."""

    def __init__(self, steps, low, high):
        self.x0, self.x1 = min(steps), max(steps)
        self.low, self.high = low, high
        self.w, self.h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(self, step):
        """The x position of a step."""
        return LEFT + (step - self.x0) / (self.x1 - self.x0) * self.w

    def y(self, loss):
        """The y position of a loss."""
        return TOP + (self.high - loss) / (self.high - self.low) * self.h


def axes(fr, title):
    """The background, title, grid, ticks and axis labels."""
    out = [
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        f'<text x="{LEFT}" y="22" font-size="14" font-weight="bold">{title}</text>',
    ]
    right, bottom = LEFT + fr.w, TOP + fr.h
    for t in nice_ticks(fr.low, fr.high):
        y = fr.y(t)
        out.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e4e4e4"/>')
        out.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in nice_ticks(fr.x0, fr.x1):
        x = fr.x(t)
        out.append(f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 5}" '
                   'stroke="#444"/>')
        out.append(f'<text x="{x:.1f}" y="{bottom + 19}" text-anchor="middle">{t:,.0f}</text>')
    out.append(f'<rect x="{LEFT}" y="{TOP}" width="{fr.w}" height="{fr.h}" '
               'fill="none" stroke="#444"/>')
    out.append(f'<text x="{LEFT + fr.w / 2}" y="{HEIGHT - 10}" text-anchor="middle">'
               'training step</text>')
    mid = TOP + fr.h / 2
    out.append(f'<text x="16" y="{mid}" text-anchor="middle" '
               f'transform="rotate(-90 16 {mid})">loss, nats per token</text>')
    return out


def references(fr, refs):
    """Thin grey lines at the reference levels, labelled."""
    out = []
    for level, label in refs:
        y = fr.y(level)
        out.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + fr.w}" y2="{y:.1f}" '
                   'stroke="#888" stroke-width="1" stroke-dasharray="2 3"/>')
        out.append(f'<text x="{LEFT + fr.w - 4}" y="{y - 4:.1f}" text-anchor="end" '
                   f'fill="#555">{label}</text>')
    return out


def curves(fr, steps, runs):
    """Each run's line, and its entry in the legend."""
    out = []
    for i, (name, losses) in enumerate(runs.items()):
        colour, dash = style(name, i)
        points = " ".join(f"{fr.x(s):.1f},{fr.y(v):.1f}" for s, v in zip(steps, losses))
        out.append(f'<polyline points="{points}" fill="none" stroke="{colour}" '
                   f'stroke-width="2"{dash}/>')
        ly, lx = TOP + 14 + i * 22, LEFT + fr.w + 14
        out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="{colour}" '
                   f'stroke-width="2"{dash}/>')
        out.append(f'<text x="{lx + 32}" y="{ly + 4}">{LABELS.get(name, name)}</text>')
    return out


def svg(steps, runs, title, refs):
    """The figure, as SVG text."""
    values = [v for losses in runs.values() for v in losses] + [r[0] for r in refs]
    pad = (max(values) - min(values)) * 0.05
    fr = Frame(steps, min(values) - pad, max(values) + pad)
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="sans-serif" font-size="12">')
    body = axes(fr, title) + references(fr, refs) + curves(fr, steps, runs)
    return "\n".join([head] + body + ["</svg>"]) + "\n"


def main(argv=None):
    """Read the CSV, write the SVG."""
    p = argparse.ArgumentParser(description="Training-loss curves from a CSV, as SVG.")
    p.add_argument("csv")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--ref", action="append", default=[], metavar="LEVEL:LABEL",
                   help="a reference level, drawn as a thin line")
    a = p.parse_args(argv)
    refs = []
    for r in a.ref:
        level, _, label = r.partition(":")
        refs.append((float(level), label))
    steps, runs = read_runs(a.csv)
    with open(a.output, "w", encoding="utf-8") as f:
        f.write(svg(steps, runs, a.title, refs))
    print(f"wrote {a.output}: {len(runs)} runs, {len(steps)} points each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
