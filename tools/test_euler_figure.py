"""Tests for euler_figure.py."""

import math
import os
import tempfile
import unittest

import euler_figure as ef


class EulerFigureTest(unittest.TestCase):
    """The steps against the closed form, and the figure."""

    def test_each_step_scales_the_radius_exactly(self):
        """For a rotation, each Euler step of Δt multiplies the distance
        from the centre by √(1 + (πΔt/2)²): 1.617, 1.332 and 1.080 after
        2, 4 and 16 steps."""
        for steps, want in ((2, 1.617), (4, 1.332), (16, 1.080)):
            r = math.hypot(*ef.euler(steps)[-1])
            exact = math.sqrt(1 + (math.pi / 2 / steps) ** 2) ** steps
            self.assertAlmostEqual(r, exact, places=12)
            self.assertAlmostEqual(r, want, places=3)

    def test_many_steps_approach_the_true_end(self):
        """The true path ends at (0, 1), a quarter turn; Euler gets there
        as the steps shrink."""
        x, y = ef.euler(10_000)[-1]
        self.assertAlmostEqual(x, 0.0, places=3)
        self.assertAlmostEqual(y, 1.0, places=3)

    def test_writes_an_svg(self):
        """The figure is written, with both panels and the three journeys."""
        path = os.path.join(tempfile.mkdtemp(), "e.svg")
        self.assertEqual(ef.main(["-o", path]), 0)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertTrue(text.startswith("<svg") and text.rstrip().endswith("</svg>"))
        for words in ("One Euler step", "2 steps: 62%", "4 steps: 33%", "16 steps: 8%"):
            self.assertIn(words, text)


if __name__ == "__main__":
    unittest.main()
