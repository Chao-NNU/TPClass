"""Tests for the Phase 1 feature construction.

The bugs these tests guard against are ordering bugs: the ESM-2 block and the
s4pred block of the feature matrix have to be built from the *same* ordered
accession list, otherwise the two halves of a row describe different
proteins.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from tpclass.core.phase1 import Phase1Classifier


class FakeBackbone:
    """A backbone that returns deterministic, distinguishable embeddings."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.pad_id = 1
        self.eos_id = 2
        self.device = "cpu"

    def encode_all(self, seqs, batch_size=None):
        n = len(seqs)
        # Row i carries the value i+1 in every position, so a mean-pooled
        # vector identifies the sequence it came from.
        token_emb = torch.zeros((n, 3, self.dim))
        for i in range(n):
            token_emb[i] = float(i + 1)
        input_ids = torch.zeros((n, 3), dtype=torch.long)
        return list(seqs), token_emb, input_ids

    def phase1_mean_pool(self, token_emb, input_ids, pad_id, eos_id):
        return token_emb.mean(dim=1)


def write_fasta(tmpdir, records):
    """records: list of (accession, sequence)."""
    path = Path(tmpdir) / "input.fasta"
    with open(path, "w") as handle:
        for acc, seq in records:
            handle.write(f">{acc}\n{seq}\n")
    return path


class TestPhase1Features(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.records = [
            ("P00001", "A" * 35),
            ("P00002", "C" * 47),
            ("P00003", "D" * 61),
        ]
        self.fasta = write_fasta(self.tmpdir, self.records)

        self.classifier = Phase1Classifier(
            esm2_path="/mock/esm2",
            checkpoint="/mock/ckpt.pth",
            backbone="650M",
            device="cpu",
        )
        # Pretend the assets are already loaded so that no ESM-2 or checkpoint
        # file is needed; the backbone is replaced by the fake below.
        self.classifier._mlp = object()
        self.classifier._backbone = FakeBackbone(dim=4)

    def tearDown(self):
        self._tmp.cleanup()

    def _s4pred_stub(self, sequences, names=None, strict=False):
        """Return a distinctive s4pred block per accession."""
        order = list(names) if names is not None else list(sequences.keys())
        block = np.zeros((len(order), 3), dtype=np.float32)
        for i, acc in enumerate(order):
            block[i] = (i + 1, i + 1, i + 1)
        return block

    def test_features_follow_fasta_order(self):
        with patch(
            "tpclass.core.phase1.extract_s4pred_features", side_effect=self._s4pred_stub
        ):
            names, features, token_emb, input_ids = next(
                self.classifier.iter_batches(self.fasta)
            )

        self.assertEqual(names, [r[0] for r in self.records])
        self.assertEqual(features.shape, (3, 4 + 3))
        # The ESM-2 block identifies the source row (mean-pooled value i+1).
        for i in range(3):
            self.assertAlmostEqual(float(features[i, 0]), float(i + 1))
        # The s4pred block must be aligned with the same row.
        for i in range(3):
            self.assertAlmostEqual(float(features[i, 4]), float(i + 1))
            self.assertAlmostEqual(float(features[i, 6]), float(i + 1))

    def test_seq_length_is_the_protein_length(self):
        """seq_length must be the residue count, not the embedding dimension."""
        with patch(
            "tpclass.core.phase1.extract_s4pred_features", side_effect=self._s4pred_stub
        ):
            with patch.object(Phase1Classifier, "score", lambda self, f: np.array([0.9] * 3)):
                df = self.classifier.predict(self.fasta)

        self.assertEqual(df["seq_length"].tolist(), [35, 47, 61])
        self.assertEqual(df["query"].tolist(), [r[0] for r in self.records])
        self.assertIn("transport_prob", df.columns)

    def test_empty_fasta_raises(self):
        empty = Path(self.tmpdir) / "empty.fasta"
        empty.write_text("")
        with self.assertRaises(ValueError):
            list(self.classifier.iter_batches(empty))


if __name__ == "__main__":
    unittest.main()
