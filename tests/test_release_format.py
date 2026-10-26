"""Tests for the release-output format and the TCDB annotation table.

A row-by-row comparison against the published tables is only meaningful when
the package writes the *same cell values* as the released script:

* the Phase 1 decision uses a strict ``>``, so a protein whose probability is
  exactly the threshold is not a transporter;
* a protein without a TC family gets ``0.0`` / ``0`` in the evidence columns,
  not empty cells;
* three columns are rounded to four decimals before they are written;
* the TC description comes from the TCDB annotation table, keyed by TC number.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tpclass.core.phase2 import load_tcdb_annotations
from tpclass.core.predictor import _is_transport, _match_release_output_format


class TestThresholdRule(unittest.TestCase):
    def test_comparison_is_strict(self):
        # The released script uses ``prob > threshold``.
        self.assertFalse(_is_transport(0.5, 0.5))
        self.assertTrue(_is_transport(0.5000001, 0.5))
        self.assertFalse(_is_transport(0.4999999, 0.5))

    def test_custom_threshold(self):
        self.assertFalse(_is_transport(0.3, 0.3))
        self.assertTrue(_is_transport(0.31, 0.3))


class TestReleaseOutputFormat(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            {
                "query": ["p1", "p2"],
                "transport_prob": [0.87654321, 0.12345678],
                "tc_number": ["1.A.1.1", None],
                "tc_family": ["1.A.1", None],
                "top1_similarity": [0.912345678, None],
                "top1_sim_gap": [0.112345678, None],
                "top_family_votes": [7, None],
                "top_family_purity": [0.712345678, None],
            }
        )

    def test_missing_family_gets_zeros_not_blanks(self):
        out = _match_release_output_format(self._frame())
        row = out[out["query"] == "p2"].iloc[0]
        self.assertEqual(row["top1_similarity"], 0.0)
        self.assertEqual(row["top1_sim_gap"], 0.0)
        self.assertEqual(row["top_family_purity"], 0.0)
        self.assertEqual(row["top_family_votes"], 0)

    def test_votes_stay_integral(self):
        out = _match_release_output_format(self._frame())
        self.assertEqual(out["top_family_votes"].dtype.kind, "i")

    def test_released_columns_are_rounded_to_four_decimals(self):
        out = _match_release_output_format(self._frame())
        self.assertAlmostEqual(out.loc[0, "transport_prob"], 0.8765, places=4)
        self.assertAlmostEqual(out.loc[0, "top1_sim_gap"], 0.1123, places=4)
        self.assertAlmostEqual(out.loc[0, "top_family_purity"], 0.7123, places=4)
        # top1_similarity is not rounded by the released script.
        self.assertAlmostEqual(out.loc[0, "top1_similarity"], 0.912345678, places=8)

    def test_text_columns_are_left_alone(self):
        out = _match_release_output_format(self._frame())
        self.assertEqual(out.loc[0, "tc_number"], "1.A.1.1")
        self.assertTrue(pd.isna(out.loc[1, "tc_number"]))


class TestTcdbAnnotations(unittest.TestCase):
    def test_parses_tsv_and_skips_a_header(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tcdb_annot.clean.tsv"
            path.write_text(
                "tc_number\tdescription\n"
                "1.A.1.1\tABC transporter\n"
                "2.A.1.2\tXYZ permease\n",
                encoding="utf-8",
            )
            annot = load_tcdb_annotations(path)
            self.assertEqual(len(annot), 2)
            self.assertEqual(annot["1.A.1.1"], "ABC transporter")
            self.assertEqual(annot["2.A.1.2"], "XYZ permease")

    def test_missing_file_yields_an_empty_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_tcdb_annotations(Path(d) / "nope.tsv"), {})
            self.assertEqual(load_tcdb_annotations(None), {})

    def test_comma_separated_file_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "annot.csv"
            path.write_text("1.A.1.1,ABC transporter\n", encoding="utf-8")
            annot = load_tcdb_annotations(path)
            self.assertEqual(annot["1.A.1.1"], "ABC transporter")


if __name__ == "__main__":
    unittest.main()
