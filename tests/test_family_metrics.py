"""Tests for the family-level metrics.

The two released evaluation scripts compute "fuzzy recall" under different
rules; the minimal example below is the one that shows the difference:

* two ground-truth transporters, both predicted YES;
* one of them received a 4-digit TC family, the other did not.

``eval_family_precision.py`` counts only the first  -> fuzzy recall 0.50
``eval_hybrid_v3.py`` counts both                  -> fuzzy recall 1.00
"""
import unittest

import pandas as pd

from tpclass.analysis.family_metrics import (
    FUZZY_ALL_YES,
    FUZZY_WITH_FAMILY,
    evaluate_family,
    load_ground_truth,
    to_4digit,
)


def _predictions():
    return pd.DataFrame(
        {
            "query": ["p1", "p2", "p3"],
            "is_transport": ["YES", "YES", "NO"],
            "tc_number": ["1.A.1.1", "", ""],
            "tc_family": ["1.A.1", "", ""],
        }
    )


def _ground_truth():
    # p1 and p2 are transporters with a TC label; p2's prediction got no family
    return {"p1": "1.A.1.1", "p2": "2.A.1.1"}


class TestTo4digit(unittest.TestCase):
    def test_truncation(self):
        self.assertEqual(to_4digit("2.A.1.2.98"), "2.A.1.2")
        self.assertEqual(to_4digit("1.A.9.1"), "1.A.9.1")

    def test_empty_and_malformed(self):
        self.assertEqual(to_4digit(""), "")
        self.assertEqual(to_4digit("nan"), "")
        self.assertEqual(to_4digit(None), "")


class TestFuzzyRecallRules(unittest.TestCase):
    def test_the_two_rules_differ(self):
        pred = _predictions()
        gt = _ground_truth()

        with_family = evaluate_family(pred, gt, fuzzy_rule=FUZZY_WITH_FAMILY)
        all_yes = evaluate_family(pred, gt, fuzzy_rule=FUZZY_ALL_YES)

        self.assertEqual(with_family.n_gt, 2)
        self.assertEqual(with_family.n_yes_in_gt, 2)
        self.assertEqual(with_family.n_matched, 1)
        self.assertEqual(with_family.n_correct, 1)

        self.assertAlmostEqual(with_family.recall_fuzzy, 0.5, places=6)
        self.assertAlmostEqual(all_yes.recall_fuzzy, 1.0, places=6)

    def test_both_rules_are_always_reported(self):
        row = evaluate_family(_predictions(), _ground_truth()).as_row()
        self.assertAlmostEqual(row["recall_fuzzy_matched"], 0.5, places=6)
        self.assertAlmostEqual(row["recall_fuzzy_all_yes"], 1.0, places=6)

    def test_strict_equals_fuzzy_times_precision(self):
        for rule in (FUZZY_WITH_FAMILY, FUZZY_ALL_YES):
            m = evaluate_family(_predictions(), _ground_truth(), fuzzy_rule=rule)
            if m.recall_fuzzy > 0:
                self.assertAlmostEqual(
                    m.recall_strict,
                    m.recall_fuzzy * m.precision_4digit,
                    places=6,
                )

    def test_the_two_precision_denominators_differ(self):
        m = evaluate_family(_predictions(), _ground_truth())
        # eval_family_precision.py: correct / (YES and GT and has family)
        self.assertAlmostEqual(m.precision_with_family, 1.0, places=6)
        # eval_hybrid_v3.py: correct / (YES and GT)
        self.assertAlmostEqual(m.precision_all_yes, 0.5, places=6)
        # the selected rule picks the matching precision
        with_family = evaluate_family(
            _predictions(), _ground_truth(), fuzzy_rule=FUZZY_WITH_FAMILY
        )
        all_yes = evaluate_family(
            _predictions(), _ground_truth(), fuzzy_rule=FUZZY_ALL_YES
        )
        self.assertAlmostEqual(with_family.precision_4digit, 1.0, places=6)
        self.assertAlmostEqual(all_yes.precision_4digit, 0.5, places=6)

    def test_precision_uses_the_top1_family(self):
        m = evaluate_family(_predictions(), _ground_truth())
        self.assertAlmostEqual(m.precision_4digit, 1.0, places=6)
        self.assertAlmostEqual(m.coverage_4digit, 0.5, places=6)

    def test_unknown_fuzzy_rule_raises(self):
        with self.assertRaises(ValueError):
            evaluate_family(_predictions(), _ground_truth(), fuzzy_rule="nope")


class TestLoadGroundTruth(unittest.TestCase):
    def test_rows_without_a_tc_label_are_dropped(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gt.tsv"
            path.write_text(
                "accession\ttc_family_4digit\n"
                "p1\t1.A.1.1\n"
                "p2\t\n"
                "p3\t2.A.1.2.9\n",
                encoding="utf-8",
            )
            gt = load_ground_truth(path)
            self.assertEqual(gt, {"p1": "1.A.1.1", "p3": "2.A.1.2"})


if __name__ == "__main__":
    unittest.main()
