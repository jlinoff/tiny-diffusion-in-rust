"""Tests for text_stats.py, on texts worked by hand."""
import unittest

from text_stats import measure, stories, tiny_llm_stories


class TextStatsTest(unittest.TestCase):
    """measure() and stories() against hand-worked texts."""

    def test_stories_split_on_markers_and_blank_lines(self):
        """Stories end at a marker or a blank line."""
        t = "Once upon a time.<|endoftext|>The end.\n\nAnother one"
        self.assertEqual(
            stories(t),
            [["once", "upon", "a", "time"], ["the", "end"], ["another", "one"]],
        )

    def test_repetition(self):
        """6 words, one repeat ("the the"), and none across the break."""
        words, rep, _ = measure("the the cat sat\n\nsat down")
        self.assertEqual(words, 6)
        self.assertAlmostEqual(rep, 1 / 6)

    def test_distinct_2(self):
        """Pairs (a b) (b a) (a b): 2 different of 3."""
        _, _, d = measure("a b a b")
        self.assertAlmostEqual(d, 2 / 3)
        self.assertEqual(measure("one two three")[2], 1.0)

    def test_tiny_llm_output(self):
        """Only the stories are kept from generate's output; speeds averaged."""
        out = (
            "Model: out/model.safetensors\nSampling: temperature 0.8\n"
            "Prompt: 4 token(s), the text\n\nOnce upon a time a cat.\n\n"
            "Stopped at <|endoftext|> after 6 tokens.\n"
            "  prefill: 4 tokens in 1.0 ms (4000 tokens/s)\n"
            "  decode:  6 tokens in 0.01 s (2000 tokens/s)\n"
            "Prompt: 4 token(s), the text\n\nOnce upon a dog.\n"
            "Stopped at the limit of 4 tokens.\n"
            "  decode:  4 tokens in 0.01 s (3000 tokens/s)\n"
        )
        text, speed = tiny_llm_stories(out)
        self.assertEqual(text, "Once upon a time a cat.\n\nOnce upon a dog.")
        self.assertEqual(speed, 2500.0)
        self.assertEqual(tiny_llm_stories("nothing here"), ("", None))

    def test_empty(self):
        """An empty text measures nothing."""
        self.assertEqual(measure(""), (0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
