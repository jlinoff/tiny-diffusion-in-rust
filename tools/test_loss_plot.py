"""Tests for loss_plot.py."""

import os
import tempfile
import unittest

import loss_plot


class LossPlotTest(unittest.TestCase):
    """The CSV read, the styles, and the figure written."""

    def setUp(self):
        """A small CSV of two runs."""
        self.dir = tempfile.mkdtemp()
        self.csv = os.path.join(self.dir, "l.csv")
        with open(self.csv, "w", encoding="utf-8") as f:
            f.write("# a comment\nstep,masked-rope,control-learned\n100,6.0,5.0\n200,5.0,3.0\n")

    def test_reads_runs_and_skips_comments(self):
        """Steps and runs from the CSV, comments left out."""
        steps, runs = loss_plot.read_runs(self.csv)
        self.assertEqual(steps, [100, 200])
        self.assertEqual(runs, {"masked-rope": [6.0, 5.0], "control-learned": [5.0, 3.0]})

    def test_styles_follow_names(self):
        """RoPE orange, learned blue, the control dashed."""
        self.assertEqual(loss_plot.style("masked-rope", 0), ("#e07b00", ""))
        colour, dash = loss_plot.style("control-learned", 1)
        self.assertEqual(colour, "#1f6fb4")
        self.assertIn("dasharray", dash)

    def test_writes_one_line_per_run_and_reference(self):
        """A line per run, the reference labelled, a whole SVG."""
        out = os.path.join(self.dir, "l.svg")
        loss_plot.main([self.csv, "-o", out, "--title", "t", "--ref", "4.5:half way"])
        with open(out, encoding="utf-8") as f:
            text = f.read()
        self.assertEqual(text.count("<polyline"), 2)
        self.assertIn("half way", text)
        self.assertTrue(text.startswith("<svg") and text.rstrip().endswith("</svg>"))

    def test_ticks_cover_the_range(self):
        """Ticks inside the range, enough of them."""
        ticks = loss_plot.nice_ticks(2.8, 6.7)
        self.assertTrue(ticks[0] >= 2.8 and ticks[-1] <= 6.7 and len(ticks) >= 3)


if __name__ == "__main__":
    unittest.main()
