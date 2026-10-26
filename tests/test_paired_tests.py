"""Tests for the paired significance tests.

The counterexample below is the one that motivated this module: with
``y = [1, 0]``, model A correct on both proteins and model B wrong on both,
the released script reports ``p = 1.0`` because it counts *label*
disagreements.  The accuracy-based test must not.
"""
import unittest

import pandas as pd

from tpclass.analysis.paired_tests import (
    compare_config_groups,
    compare_models,
    mcnemar_exact,
    mcnemar_exact_on_predictions,
)


class TestMcNemarCorrectness(unittest.TestCase):
    def test_perfect_model_vs_wrong_model_is_not_p_equal_one(self):
        y = [1, 0]
        a = [1, 0]  # A is right on both
        b = [0, 1]  # B is wrong on both
        n01, n10, p = mcnemar_exact(y, a, b)
        self.assertEqual(n01, 2)
        self.assertEqual(n10, 0)
        # exact binomial on 2 discordant pairs, both in A's favour
        self.assertAlmostEqual(p, 0.5, places=6)
        self.assertLess(p, 1.0)

    def test_identical_models_give_p_one(self):
        y = [1, 0, 1, 0]
        a = [1, 0, 1, 0]
        n01, n10, p = mcnemar_exact(y, a, a)
        self.assertEqual((n01, n10), (0, 0))
        self.assertEqual(p, 1.0)

    def test_label_variant_reproduces_the_released_behaviour(self):
        """The released script counts label disagreements, not correctness."""
        y = [1, 0]
        a = [1, 0]
        b = [0, 1]
        n01, n10, p = mcnemar_exact_on_predictions(a, b)
        self.assertEqual((n01, n10), (1, 1))
        self.assertEqual(p, 1.0)
        # ... which is why the released p-value cannot answer the question.
        self.assertGreater(p, 0.05)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mcnemar_exact([1, 0, 1], [1, 0], [1, 0])


class TestCompareModels(unittest.TestCase):
    def test_reports_both_mcnemar_variants(self):
        y = [1, 0, 1, 0]
        a = [1, 0, 1, 0]
        b = [0, 0, 1, 1]
        row = compare_models(y, a, b, n_boot=200, seed=0)
        self.assertEqual(row["n"], 4)
        self.assertIn("mcnemar_p", row)
        self.assertIn("mcnemar_label_variant_p", row)
        self.assertIn("d_accuracy", row)
        self.assertIn("pos_jaccard", row)

    def test_jaccard_of_positive_sets(self):
        row = compare_models([1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], n_boot=50)
        self.assertEqual(row["pos_n_a"], 2)
        self.assertEqual(row["pos_n_b"], 2)
        self.assertAlmostEqual(row["pos_jaccard"], 0.5, places=6)


class TestCompareConfigGroups(unittest.TestCase):
    def test_long_format_table(self):
        df = pd.DataFrame(
            {
                "protein_id": ["p1", "p2", "p1", "p2"],
                "config": ["A", "A", "B", "B"],
                "y_true": [1, 0, 1, 0],
                "y_pred": [1, 0, 0, 1],
            }
        )
        rows = compare_config_groups(df, [("A", "B", "A vs B")], n_boot=50)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["comparison"], "A vs B")
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["mcnemar_n01"], 2)
        self.assertEqual(row["mcnemar_n10"], 0)

    def test_missing_configuration_raises(self):
        df = pd.DataFrame(
            {
                "protein_id": ["p1"],
                "config": ["A"],
                "y_true": [1],
                "y_pred": [1],
            }
        )
        with self.assertRaises(ValueError):
            compare_config_groups(df, [("A", "B", "A vs B")])


if __name__ == "__main__":
    unittest.main()
