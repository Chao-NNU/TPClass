"""Tests for the row-by-row report comparison.

The comparison tool backs the claim that the ``tpclass`` package reproduces the
released pipeline, so it has to distinguish three different situations:

* the tables are identical;
* a numeric value differs slightly (below or above the tolerance);
* a query is present in one table but missing from the other.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tpclass.analysis.compare_reports import compare_reports, format_report


def _write(directory: Path, name: str, rows) -> str:
    path = directory / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


class TestCompareReports(unittest.TestCase):
    def test_identical_tables(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rows = [
                {"query": "p1", "transport_prob": 0.9, "tc_number": "1.A.1.1"},
                {"query": "p2", "transport_prob": 0.1, "tc_number": ""},
            ]
            diff = compare_reports(
                _write(tmp, "a.csv", rows),
                _write(tmp, "b.csv", rows),
            )
            self.assertTrue(diff.identical)
            self.assertEqual(diff.total_mismatch, 0)
            self.assertEqual(diff.n_matched, 2)
            self.assertIn("IDENTICAL", format_report(diff))

    def test_numeric_difference_below_tolerance_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            diff = compare_reports(
                _write(tmp, "a.csv", [{"query": "p1", "transport_prob": 0.9000}]),
                _write(tmp, "b.csv", [{"query": "p1", "transport_prob": 0.90005}]),
                tol=1e-4,
            )
            self.assertTrue(diff.identical)

    def test_numeric_difference_above_tolerance_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            diff = compare_reports(
                _write(tmp, "a.csv", [{"query": "p1", "transport_prob": 0.9000}]),
                _write(tmp, "b.csv", [{"query": "p1", "transport_prob": 0.7500}]),
                tol=1e-4,
            )
            self.assertFalse(diff.identical)
            self.assertEqual(diff.total_mismatch, 1)
            self.assertAlmostEqual(diff.columns[0].max_abs_diff, 0.15, places=4)
            self.assertEqual(diff.columns[0].examples[0]["query"], "p1")

    def test_missing_query_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            diff = compare_reports(
                _write(tmp, "a.csv", [{"query": "p1"}, {"query": "p2"}]),
                _write(tmp, "b.csv", [{"query": "p1"}]),
            )
            self.assertFalse(diff.identical)
            self.assertEqual(diff.n_only_a, 1)
            self.assertEqual(diff.n_only_b, 0)

    def test_empty_cells_are_not_treated_as_zero(self):
        """A blank ``tc_number`` must not compare equal to the string ``0``."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            diff = compare_reports(
                _write(tmp, "a.csv", [{"query": "p1", "tc_number": ""}]),
                _write(tmp, "b.csv", [{"query": "p1", "tc_number": "0"}]),
            )
            self.assertFalse(diff.identical)
            self.assertEqual(diff.total_mismatch, 1)

    def test_missing_key_raises(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            a = _write(tmp, "a.csv", [{"name": "p1"}])
            b = _write(tmp, "b.csv", [{"name": "p1"}])
            with self.assertRaises(ValueError):
                compare_reports(a, b, key="query")


if __name__ == "__main__":
    unittest.main()
