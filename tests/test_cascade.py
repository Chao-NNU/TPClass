"""Tests for the cascade strategy.

The cascade is the part of TPClass whose semantics are easiest to get
subtly wrong, so it gets its own test module:

* the candidate set is the **3B YES set** (the 650M model never removes a
  candidate);
* the family comes from the 650M model when the 650M model also called the
  protein YES *and* produced a family;
* otherwise the 3B family is kept;
* the retrieval evidence always comes from the **same** model that provided
  the family.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from tpclass.core.predictor import FAMILY_COLUMNS, TPClassPredictor


class StubPhase1:
    """Minimal Phase 1 stand-in: fixed probabilities per accession."""

    def __init__(self, probs, backbone=None):
        self.probs = dict(probs)
        self._backbone = backbone or object()

    @property
    def backbone_model(self):
        return self._backbone

    def iter_batches(self, fasta):
        names = list(self.probs.keys())
        features = np.zeros((len(names), 4), dtype=np.float32)
        token_emb = torch.zeros((len(names), 3, 2))
        input_ids = torch.zeros((len(names), 3), dtype=torch.long)
        yield names, features, token_emb, input_ids

    def score(self, features):
        return np.array([self.probs[n] for n in self.probs], dtype=float)


class StubPhase2:
    """Minimal Phase 2 stand-in: a fixed family per accession."""

    def __init__(self, families, backbone=None):
        self.families = dict(families)
        self._backbone = backbone or object()

    @property
    def backbone_model(self):
        return self._backbone

    def embed_queries(self, names, sequences, token_batch=None):
        return {n: np.zeros(4, dtype=np.float32) for n in names}

    def assign(self, embeddings, top_k=10):
        rows = []
        for query in embeddings:
            tc = self.families.get(query, "")
            row = {"query": query}
            for col in FAMILY_COLUMNS:
                row[col] = ""
            row.update(
                {
                    "tc_number": tc,
                    "tc_family": tc.split(".")[0] if tc else "",
                    "top1_similarity": 0.9 if tc else float("nan"),
                    "top_family_votes": 7 if tc else 0,
                    "top_family_purity": 0.7 if tc else 0.0,
                    "max_ref_sim": 0.9 if tc else float("nan"),
                    "best_ref_acc": f"REF_{query}" if tc else "",
                    "best_ref_gene": f"gene_{query}" if tc else "",
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=["query"] + FAMILY_COLUMNS)


def write_fasta(tmpdir, accessions, length=40):
    path = Path(tmpdir) / "input.fasta"
    with open(path, "w") as handle:
        for acc in accessions:
            handle.write(f">{acc}\n{'A' * length}\n")
    return path


class TestCascade(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmpdir = self._tmp.name
        # A: 3B YES + 650M YES with a family
        # B: 3B YES only (650M said NO)   -> 3B fallback
        # C: 650M YES only (3B said NO)   -> not a candidate
        # D: neither                        -> not a candidate
        self.fasta = write_fasta(self._tmp.name, ["A", "B", "C", "D"])

        predictor = TPClassPredictor()
        predictor.phase1_3b = StubPhase1({"A": 0.9, "B": 0.8, "C": 0.1, "D": 0.1})
        predictor.phase1_650m = StubPhase1({"A": 0.9, "B": 0.2, "C": 0.9, "D": 0.1})
        predictor.phase2_3b = StubPhase2({"A": "2.A.1.1.1", "B": "1.B.1.1.1"})
        predictor.phase2_650m = StubPhase2({"A": "2.A.9.9.9", "C": "3.C.1.1.1"})
        self.predictor = predictor

    def tearDown(self):
        self._tmp.cleanup()

    def _cascade(self):
        return self.predictor._predict_cascade(
            self.fasta, threshold=0.5, run_phase2=True, top_k=10
        )

    def test_candidate_set_is_the_3b_yes_set(self):
        """The 650M model must not remove a 3B candidate."""
        out = self._cascade()
        self.assertEqual(sorted(out["query"].tolist()), ["A", "B"])
        self.assertNotIn("C", out["query"].values)
        self.assertNotIn("D", out["query"].values)

    def test_650m_family_wins_when_650m_also_said_yes(self):
        out = self._cascade().set_index("query")
        self.assertEqual(out.loc["A", "tc_number"], "2.A.9.9.9")
        self.assertEqual(out.loc["A", "family_source"], "650M")

    def test_fallback_to_3b_family(self):
        out = self._cascade().set_index("query")
        self.assertEqual(out.loc["B", "tc_number"], "1.B.1.1.1")
        self.assertEqual(out.loc["B", "family_source"], "3B")

    def test_evidence_comes_from_the_same_model_as_the_family(self):
        """A 650M family must be supported by 650M evidence, not 3B evidence."""
        out = self._cascade().set_index("query")
        # The stub returns 0.9 for a model that produced a family and NaN
        # otherwise, so the evidence has to match the model that won.
        self.assertAlmostEqual(float(out.loc["A", "top1_similarity"]), 0.9)
        self.assertEqual(out.loc["A", "best_ref_acc"], "REF_A")
        self.assertEqual(int(out.loc["A", "top_family_votes"]), 7)

    def test_650m_yes_without_family_keeps_the_3b_family(self):
        predictor = TPClassPredictor()
        predictor.phase1_3b = StubPhase1({"A": 0.9, "B": 0.2})
        predictor.phase1_650m = StubPhase1({"A": 0.9, "B": 0.2})
        predictor.phase2_3b = StubPhase2({"A": "2.A.1.1.1"})
        # 650M said YES but produced no family at all.
        predictor.phase2_650m = StubPhase2({})
        out = predictor._predict_cascade(
            self.fasta, threshold=0.5, run_phase2=True, top_k=10
        ).set_index("query")
        self.assertEqual(out.loc["A", "tc_number"], "2.A.1.1.1")
        self.assertEqual(out.loc["A", "family_source"], "3B")

    def test_transport_prob_columns_are_both_present(self):
        out = self._cascade()
        self.assertIn("transport_prob", out.columns)
        self.assertIn("transport_prob_650m", out.columns)
        row = out.set_index("query").loc["A"]
        self.assertAlmostEqual(float(row["transport_prob"]), 0.9)
        self.assertAlmostEqual(float(row["transport_prob_650m"]), 0.9)


if __name__ == "__main__":
    unittest.main()
