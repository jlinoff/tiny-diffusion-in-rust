"""Tests for text_figures.py."""

import os
import tempfile
import unittest

import text_figures as tf


class TextFiguresTest(unittest.TestCase):
    """Masking, the trace, the snapshots and the figures."""

    def test_masks_nest_and_follow_t(self):
        """All shown at t = 1, all masked at t = 0, and a word masked at
        one t stays masked at every smaller t."""
        rows = tf.masked_rows(list("abcdefghij"), (1.0, 0.6, 0.3, 0.0))
        self.assertEqual(rows[0], [False] * 10)
        self.assertEqual(rows[-1], [True] * 10)
        for upper, lower in zip(rows, rows[1:]):
            self.assertTrue(all(lo or not up for up, lo in zip(upper, lower)))

    def test_reads_a_trace_and_takes_snapshots(self):
        """Given tokens always shown; a settled one shown only from the
        step after it settled; escapes undone."""
        path = os.path.join(tempfile.mkdtemp(), "t.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# steps\t4\torder\trandom\n# position\tsettled at\ttoken\n"
                    "0\t-1\tOnce\n1\t2\t up\\n\n2\t0\t a\n")
        steps, order, rows = tf.read_trace(path)
        self.assertEqual((steps, order), (4, "random"))
        self.assertEqual(rows[1], (1, 2, " up\n"))
        kinds = [k for _, k in tf.snapshot(rows, 1)]
        self.assertEqual(kinds, ["given", "masked", "settled"])
        kinds = [k for _, k in tf.snapshot(rows, 3)]
        self.assertEqual(kinds, ["given", "settled", "settled"])

    def test_escapes_markup_and_control_characters(self):
        """Markup escaped; a control byte becomes the replacement character."""
        self.assertEqual(tf.esc("a<b&\x12"), "a&lt;b&amp;�")

    def test_figures_are_whole_svgs(self):
        """Both figures start and end as SVG; the settling one marks the
        newest tokens."""
        m = tf.masking_svg()
        self.assertTrue(m.startswith("<svg") and m.rstrip().endswith("</svg>"))
        rows = [(0, -1, "Once"), (1, 0, " there"), (2, 1, " was"), (3, -1, " a")]
        s = tf.settling_svg((2, "confidence", rows), [0, 0.5, 1], 1)
        self.assertIn("after 1 step (50%)", s)
        self.assertIn('font-weight="bold"', s)


if __name__ == "__main__":
    unittest.main()
