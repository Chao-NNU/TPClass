"""Smoke tests for ESM-2 and s4pred utilities."""
import tempfile
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile


from tpclass.core.embeddings import (
    extract_s4pred_features,
    parse_s4pred_horiz,
    read_fasta,
    scan_horiz_cache,
)


class TestReadFasta(unittest.TestCase):
    def test_uniprot_format(self):
        """Test reading UniProt-style FASTA headers."""
        with NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(
                ">sp|P12345|TEST1 description\n"
                "MKILV\n"
                "VYDKF\n"
                ">sp|Q67890|TEST2 description\n"
                "MATGC\n"
            )
            fasta_path = Path(f.name)
        try:
            seqs = read_fasta(fasta_path)
            self.assertEqual(len(seqs), 2)
            self.assertIn("P12345", seqs)
            self.assertIn("Q67890", seqs)
            self.assertEqual(seqs["P12345"], "MKILVYDKF")
            self.assertEqual(seqs["Q67890"], "MATGC")
        finally:
            fasta_path.unlink()

    def test_bare_accession(self):
        """Test reading FASTA with bare accessions."""
        with NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(
                ">P12345 description\n"
                "MKILV\n"
            )
            fasta_path = Path(f.name)
        try:
            seqs = read_fasta(fasta_path)
            self.assertEqual(len(seqs), 1)
            self.assertIn("P12345", seqs)
        finally:
            fasta_path.unlink()

    def test_multi_line_sequences(self):
        """Test that multi-line sequences are concatenated."""
        with NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(
                ">sp|P12345|TEST\n"
                "MKILV\n"
                "VYDKF\n"
                "GYDER\n"
            )
            fasta_path = Path(f.name)
        try:
            seqs = read_fasta(fasta_path)
            self.assertEqual(seqs["P12345"], "MKILVYDKFGYDER")
        finally:
            fasta_path.unlink()


class TestS4predHorizCache(unittest.TestCase):
    """The released pipeline stores s4pred output as one .horiz file per query."""

    def _write_horiz(self, directory, name, pred):
        path = Path(directory) / f"{name}.horiz"
        path.write_text(f"Pred: {pred}\n", encoding="utf-8")
        return path

    def test_parse_horiz_fractions(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write_horiz(d, "p1", "HHHECC")
            feat = parse_s4pred_horiz(path)
            self.assertAlmostEqual(feat[0], 3 / 6, places=6)
            self.assertAlmostEqual(feat[1], 2 / 6, places=6)
            self.assertAlmostEqual(feat[2], 1 / 6, places=6)

    def test_parse_horiz_without_prediction_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.horiz"
            path.write_text("no prediction here\n", encoding="utf-8")
            self.assertIsNone(parse_s4pred_horiz(path))

    def test_scan_cache_strips_the_balanced_split_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_horiz(d, "P12345", "HHH")
            found = scan_horiz_cache(d, ["pos__P12345"])
            self.assertIn("pos__P12345", found)
            self.assertAlmostEqual(float(found["pos__P12345"][0]), 1.0, places=6)

    def test_scan_cache_missing_directory_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(scan_horiz_cache(Path(d) / "nope", ["p1"]), {})
            self.assertEqual(scan_horiz_cache(None, ["p1"]), {})

    def test_cached_features_are_used_instead_of_the_predictor(self):
        """A cached sequence must not be re-predicted, and its values must win."""
        with tempfile.TemporaryDirectory() as d:
            self._write_horiz(d, "p1", "HHH")
            feats = extract_s4pred_features(
                {"p1": "MKILV"}, names=["p1"], cache_dir=d
            )
            self.assertAlmostEqual(float(feats[0][0]), 1.0, places=6)
            self.assertAlmostEqual(float(feats[0][1]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
