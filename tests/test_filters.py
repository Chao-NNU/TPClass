"""Smoke tests for TPClass filters."""
import unittest
from pathlib import Path

import pandas as pd

from tpclass.filters.substrate import SubstrateFilter


class TestSubstrateFilter(unittest.TestCase):
    def setUp(self):
        # Mock v3 CSV with test data
        self.test_csv = Path(__file__).parent / "fixtures" / "test_v3.csv"
        self.test_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "query": ["A", "B", "C", "D", "E"],
            "tc_number": ["2.A.1.7.2", "2.A.7.9.17", "3.A.1.101.1", "1.A.30.1.1", "9.B.198.2.1"],
            "tc_description_short": [
                "Glucose/galactose transporter - E. coli",
                "UDP-sugar translocator - Arabidopsis",
                "Polysialic acid transport protein - E. coli",
                "ATP synthase - E. coli",
                "Membrane-anchored lipid-binding protein YSP2",
            ],
            "transport_prob": [0.95, 0.85, 0.75, 0.65, 0.55],
            "max_ref_sim": [0.99, 0.85, 0.75, 0.50, 0.60],
        })
        df.to_csv(self.test_csv, index=False)

    def test_tc_class_filter(self):
        """Test that TC class whitelist works."""
        flt = SubstrateFilter(
            tc_classes=["2.A.1", "3.A.1", "9.B"],
            keywords=[],
        )
        result = flt.filter(self.test_csv)
        # Should keep A (2.A.1.7.2), C (3.A.1.101.1), E (9.B.198.2.1)
        # Should drop B (2.A.7.9.17 — 2.A.7 not whitelisted), D (1.A.30.1.1)
        self.assertEqual(len(result), 3)
        self.assertIn("A", result["query"].values)
        self.assertIn("C", result["query"].values)
        self.assertIn("E", result["query"].values)
        self.assertNotIn("B", result["query"].values)
        self.assertNotIn("D", result["query"].values)

    def test_keyword_filter(self):
        """Test that keyword filter works."""
        flt = SubstrateFilter(
            tc_classes=[],
            keywords=["lipid"],
        )
        result = flt.filter(self.test_csv)
        # Should keep E (lipid-binding)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["query"], "E")

    def test_combined_filter(self):
        """Test that TC class + keyword filter combined works."""
        flt = SubstrateFilter(
            tc_classes=["9.B"],
            keywords=["lipid"],
        )
        result = flt.filter(self.test_csv)
        # Should keep E (9.B.198.2.1 + lipid)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["query"], "E")

    def test_max_ref_sim_filter(self):
        """Test that max_ref_sim threshold works."""
        flt = SubstrateFilter(
            tc_classes=[],
            keywords=[],
            min_max_ref_sim=0.7,
        )
        result = flt.filter(self.test_csv)
        # Should keep A (0.99), B (0.85), C (0.75)
        self.assertEqual(len(result), 3)
        self.assertNotIn("D", result["query"].values)
        self.assertNotIn("E", result["query"].values)


if __name__ == "__main__":
    unittest.main()
