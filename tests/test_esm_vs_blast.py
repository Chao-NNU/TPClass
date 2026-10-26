"""Tests for the ESM-2 vs BLAST comparison.

The two failure modes these tests guard against are:

* computing the query coverage against the alignment span instead of the full
  query length, which makes a short local hit look like a full-length hit;
* accepting ``coverage_low`` as an argument but never applying it.
"""
import unittest

import pandas as pd

from tpclass.analysis.esm_vs_blast import classify_hits, query_coverage


class TestQueryCoverage(unittest.TestCase):
    def test_coverage_uses_the_full_query_length(self):
        # A 100-residue alignment inside a 1000-residue protein covers 10 %.
        self.assertAlmostEqual(query_coverage(1, 100, 1000), 0.1)
        # A full-length alignment covers 100 %.
        self.assertAlmostEqual(query_coverage(1, 1000, 1000), 1.0)
        # An internal alignment is measured against the whole query too.
        self.assertAlmostEqual(query_coverage(101, 200, 1000), 0.1)

    def test_alignment_span_would_have_given_a_different_answer(self):
        """The old denominator (alignment span) would have returned 1.0."""
        span = 100 - 1 + 1
        self.assertAlmostEqual(span / 100, 1.0)
        self.assertAlmostEqual(query_coverage(1, 100, 1000), 0.1)

    def test_unknown_query_length_gives_nan(self):
        self.assertTrue(pd.isna(query_coverage(1, 100, 0)))


class TestClassifyHits(unittest.TestCase):
    def _series(self, identity, coverage, evalue):
        return pd.Series(identity), pd.Series(coverage), pd.Series(evalue)

    def test_coverage_threshold_is_applied(self):
        ident, cov, ev = self._series([0.9], [0.20], [1e-40])
        found = classify_hits(
            ident, cov, ev, id_low=0.30, coverage_low=0.50, evalue_max=1e-3
        )
        self.assertFalse(bool(found.iloc[0]))

    def test_hit_passes_when_all_thresholds_are_met(self):
        ident, cov, ev = self._series([0.9], [0.80], [1e-40])
        found = classify_hits(
            ident, cov, ev, id_low=0.30, coverage_low=0.50, evalue_max=1e-3
        )
        self.assertTrue(bool(found.iloc[0]))

    def test_identity_threshold_is_applied(self):
        ident, cov, ev = self._series([0.10], [0.80], [1e-40])
        found = classify_hits(
            ident, cov, ev, id_low=0.30, coverage_low=0.50, evalue_max=1e-3
        )
        self.assertFalse(bool(found.iloc[0]))

    def test_evalue_threshold_is_applied(self):
        ident, cov, ev = self._series([0.9], [0.80], [1.0])
        found = classify_hits(
            ident, cov, ev, id_low=0.30, coverage_low=0.50, evalue_max=1e-3
        )
        self.assertFalse(bool(found.iloc[0]))

    def test_missing_hit_is_not_a_hit(self):
        ident, cov, ev = self._series([None], [None], [None])
        found = classify_hits(
            ident, cov, ev, id_low=0.30, coverage_low=0.50, evalue_max=1e-3
        )
        self.assertFalse(bool(found.iloc[0]))


if __name__ == "__main__":
    unittest.main()
