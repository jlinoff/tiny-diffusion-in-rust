#!/usr/bin/env python3
"""
euler_figure.py: the Euler step, drawn (PLAN.md, Terms: Euler step).

The velocity field is a rotation, v(x, y) = (π/2)·(−y, x), chosen because
its exact paths are known: circles about the origin, a quarter turn from
t = 0 to t = 1. Euler's straight steps leave the circle, outward, by an
amount that shrinks as the steps do.

  left    one step: the velocity at the start, the straight step along it,
          and the true path curving away
  right   the whole journey from t = 0 to 1 in 2, 4 and 16 steps

Usage:
  python3 tools/euler_figure.py -o docs/euler.svg
"""

import argparse
import math
import sys

K = math.pi / 2                     # a quarter turn over t = 0 to 1
START = (1.0, 0.0)
PANEL, GAP, TOP, FOOT = 400, 30, 64, 64
WHOLE = (-0.45, 1.75, -0.35, 1.85)   # the right panel's world: x0, x1, y0, y1
ZOOM = (0.25, 1.45, -0.22, 0.98)     # the left panel's, closer in on one step
COLOURS = {2: "#e07b00", 4: "#1f6fb4", 16: "#2a9d4b"}


def velocity(p):
    """The field at a point: a quarter turn per unit of t."""
    return (-K * p[1], K * p[0])


def euler(steps):
    """The points Euler visits from START, t = 0 to 1, in `steps` steps."""
    dt, p, out = 1 / steps, START, [START]
    for _ in range(steps):
        v = velocity(p)
        p = (p[0] + v[0] * dt, p[1] + v[1] * dt)
        out.append(p)
    return out


class Panel:
    """World coordinates onto one square panel, its left edge at `left`,
    showing the world rectangle `box` (x0, x1, y0, y1)."""

    def __init__(self, left, box, name):
        self.left, self.box, self.name = left, box, name

    def at(self, p):
        """A world point as SVG x, y."""
        x0, x1, y0, y1 = self.box
        sx = self.left + (p[0] - x0) / (x1 - x0) * PANEL
        sy = TOP + (y1 - p[1]) / (y1 - y0) * PANEL
        return sx, sy

    def line(self, a, b, colour, width=2, extra=""):
        """A straight line between two world points."""
        (x1, y1), (x2, y2) = self.at(a), self.at(b)
        return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{colour}" stroke-width="{width}"{extra}/>')

    def dot(self, p, colour, r=3.5):
        """A filled dot at a world point."""
        x, y = self.at(p)
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{colour}"/>'

    def text(self, p, words, colour="#222", **place):
        """A label at a world point, nudged by place's dx and dy pixels,
        anchored by its anchor (start, middle or end)."""
        x, y = self.at(p)
        x, y = x + place.get("dx", 0), y + place.get("dy", 0)
        return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{colour}" '
                f'text-anchor="{place.get("anchor", "start")}">{words}</text>')

    def arc(self, radius=1.0, upto=math.pi / 2, colour="#222", dash=' stroke-dasharray="5 4"'):
        """The true path: part of a circle about the origin."""
        pts = [self.at((radius * math.cos(a), radius * math.sin(a)))
               for a in (upto * i / 60 for i in range(61))]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        return (f'<polyline points="{d}" fill="none" stroke="{colour}" stroke-width="1.6"'
                f'{dash} clip-path="url(#clip-{self.name})"/>')

    def field(self):
        """The velocity field as faint arrows on a grid, each pointing along
        the velocity, its length growing with the speed."""
        x0, x1, y0, _ = self.box
        cell = (x1 - x0) / 9
        out = [f'<g clip-path="url(#clip-{self.name})">']
        for i in range(10):
            for j in range(10):
                p = (x0 + (i + 0.3) * cell, y0 + (j + 0.3) * cell)
                v = velocity(p)
                speed = math.hypot(*v)
                if speed < 1e-9:
                    continue
                k = 0.75 * cell * min(1.0, speed / 2.5) / speed
                q = (p[0] + k * v[0], p[1] + k * v[1])
                out.append(self.line(p, q, "#c3ccd5", 1.1, ' marker-end="url(#faint)"'))
        return out + ["</g>"]

    def frame(self, title):
        """The panel's border and title."""
        return [f'<clipPath id="clip-{self.name}"><rect x="{self.left}" y="{TOP}" '
                f'width="{PANEL}" height="{PANEL}"/></clipPath>',
                f'<rect x="{self.left}" y="{TOP}" width="{PANEL}" height="{PANEL}" '
                'fill="white" stroke="#888"/>',
                f'<text x="{self.left}" y="{TOP - 8}" font-weight="bold">{title}</text>']


def one_step(pn):
    """The left panel: one Euler step of Δt = 1/2 from START."""
    out = pn.frame("One Euler step, Δt = 1/2 (zoomed in)") + pn.field() + [pn.arc()]
    dt = 0.5
    v = velocity(START)
    end = (START[0] + v[0] * dt, START[1] + v[1] * dt)
    true_end = (math.cos(K * dt), math.sin(K * dt))
    out.append(pn.line(START, end, COLOURS[2], 2.5, ' marker-end="url(#step)"'))
    out.append(pn.line(end, true_end, "#b03a8c", 1.5, ' stroke-dasharray="3 3"'))
    out += [pn.dot(START, "#222"), pn.dot(true_end, "#222"), pn.dot(end, COLOURS[2])]
    out.append(pn.text(START, "x, at time t", dy=20, anchor="middle"))
    out.append(pn.text(START, "v(x, t)", COLOURS[2], dx=10, dy=-40))
    out.append(pn.text(end, "x + v(x, t)·Δt", COLOURS[2], dx=12, dy=4))
    out.append(pn.text(end, "at time t + Δt", COLOURS[2], dx=12, dy=20))
    out.append(pn.text(true_end, "where the true path is", dx=-10, dy=4, anchor="end"))
    out.append(pn.text(true_end, "at time t + Δt", dx=-10, dy=20, anchor="end"))
    out.append(pn.text(((end[0] + true_end[0]) / 2, (end[1] + true_end[1]) / 2),
                       "the error", "#b03a8c", dy=-8, anchor="middle"))
    out.append(pn.text((math.cos(0.45), math.sin(0.45)), "the true path", dx=-10,
                       anchor="end"))
    return out


def journeys(pn):
    """The right panel: t = 0 to 1 in 2, 4 and 16 steps."""
    out = pn.frame("t = 0 to 1 in 2, 4 and 16 steps") + pn.field() + [pn.arc()]
    for steps, colour in COLOURS.items():
        pts = euler(steps)
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(pn.at, pts))
        out.append(f'<polyline points="{d}" fill="none" stroke="{colour}" stroke-width="2"/>')
        out += [pn.dot(p, colour, 3 if steps < 16 else 1.8) for p in pts]
        r = math.hypot(*pts[-1])
        out.append(pn.text(pts[-1], f"{steps} steps: {r - 1:.0%} too far out",
                           colour, dx=-8, dy=-8 if steps == 2 else 14, anchor="end"
                           if steps > 2 else "start"))
    out += [pn.dot(START, "#222"), pn.dot((0, 1), "#222")]
    out.append(pn.text(START, "start (t = 0)", dy=20, anchor="middle"))
    out.append(pn.text((0, 1), "true end", dx=-8, dy=18, anchor="end"))
    return out


def svg():
    """Both panels, with arrowheads and a caption."""
    width = 2 * PANEL + GAP + 32
    height = TOP + PANEL + FOOT
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
            '<defs>'
            '<marker id="faint" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
            'markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#b9c4cf"/></marker>'
            '<marker id="step" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{COLOURS[2]}"/>'
            '</marker></defs>',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            '<text x="16" y="26" font-size="14" font-weight="bold">'
            "The Euler step: follow the velocity in a straight line for Δt</text>"]
    head[-1] = head[-1].replace('y="26"', 'y="24"')
    body = (one_step(Panel(16, ZOOM, "zoom"))
            + journeys(Panel(16 + PANEL + GAP, WHOLE, "whole")))
    y = TOP + PANEL + 22
    foot = [f'<text x="16" y="{y}" fill="#555">Grey arrows: the velocity field v (here a '
            "rotation, the same at every t, so its true paths are circles: dashed).</text>",
            f'<text x="16" y="{y + 18}" fill="#555">Each step goes straight along the velocity '
            "where it starts; the path curves, so the step lands off it. Smaller steps, "
            "smaller errors.</text>"]
    return "\n".join(head + body + foot + ["</svg>"]) + "\n"


def main(argv=None):
    """Write the figure."""
    p = argparse.ArgumentParser(description="The Euler step, drawn, as SVG.")
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args(argv)
    with open(a.output, "w", encoding="utf-8") as f:
        f.write(svg())
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
