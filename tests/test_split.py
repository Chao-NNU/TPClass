"""Tests for the dataset splitting.

The cluster split is the leakage-free alternative to the random stratified
split: homologous proteins have to end up in the same subset, otherwise a
held-out score can be optimistic.
"""
import random
import unittest

from tpclass.train.split import cluster_sequences, cluster_split, stratified_split

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def make_sequence(seed: int, length: int = 60) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(ALPHABET) for _ in range(length))


def make_homologue(seed: int, length: int = 60) -> str:
    """A sequence that differs from `make_sequence(seed)` by one residue."""
    seq = list(make_sequence(seed, length))
    seq[length // 2] = "W" if seq[length // 2] != "W" else "A"
    return "".join(seq)


class TestClusterSequences(unittest.TestCase):
    def test_homologues_share_a_cluster(self):
        sequences = {
            "A": make_sequence(1),
            "A2": make_homologue(1),
            "B": make_sequence(2),
        }
        assignment = cluster_sequences(sequences, threshold=0.5)
        self.assertEqual(assignment["A"], assignment["A2"])
        self.assertNotEqual(assignment["A"], assignment["B"])

    def test_distant_sequences_are_separate(self):
        sequences = {f"s{i}": make_sequence(100 + i) for i in range(5)}
        assignment = cluster_sequences(sequences, threshold=0.5)
        self.assertEqual(len(set(assignment.values())), 5)


class TestClusterSplit(unittest.TestCase):
    def _records(self, n_clusters=10):
        records = []
        for i in range(n_clusters):
            records.append(
                {
                    "accession": f"P{i}a",
                    "sequence": make_sequence(i),
                    "tc_family": f"1.A.{i}",
                    "tc_number": f"1.A.{i}.1.1",
                }
            )
            records.append(
                {
                    "accession": f"P{i}b",
                    "sequence": make_homologue(i),
                    "tc_family": f"1.A.{i}",
                    "tc_number": f"1.A.{i}.1.1",
                }
            )
        return records

    def test_homologues_stay_together(self):
        records = self._records()
        splits = cluster_split(records, train_frac=0.8, val_frac=0.1, seed=42)
        placement = {}
        for name, items in splits.items():
            for rec in items:
                placement[rec["accession"]] = name
        for i in range(10):
            self.assertEqual(
                placement[f"P{i}a"],
                placement[f"P{i}b"],
                f"cluster {i} was split across subsets",
            )

    def test_every_record_is_placed_exactly_once(self):
        records = self._records()
        splits = cluster_split(records, train_frac=0.8, val_frac=0.1, seed=42)
        placed = [rec["accession"] for items in splits.values() for rec in items]
        self.assertEqual(len(placed), len(records))
        self.assertEqual(len(set(placed)), len(records))

    def test_split_sizes_are_close_to_the_targets(self):
        records = self._records()
        splits = cluster_split(records, train_frac=0.8, val_frac=0.1, seed=42)
        total = len(records)
        self.assertGreaterEqual(len(splits["train"]) / total, 0.7)
        self.assertLessEqual(len(splits["train"]) / total, 0.9)


class TestStratifiedSplit(unittest.TestCase):
    def test_families_are_spread_over_all_splits(self):
        records = []
        for fam in range(4):
            for i in range(10):
                records.append(
                    {
                        "accession": f"F{fam}_{i}",
                        "sequence": make_sequence(fam * 100 + i),
                        "tc_family": f"1.A.{fam}",
                    }
                )
        splits = stratified_split(records, train_frac=0.8, val_frac=0.1, seed=42)
        for name in ("train", "val", "test"):
            families = {rec["tc_family"] for rec in splits[name]}
            self.assertEqual(len(families), 4, f"family missing from {name}")

    def test_every_record_is_placed_exactly_once(self):
        records = [
            {"accession": f"X{i}", "sequence": make_sequence(i), "tc_family": "1.A.1"}
            for i in range(25)
        ]
        splits = stratified_split(records, train_frac=0.8, val_frac=0.1, seed=42)
        placed = [rec["accession"] for items in splits.values() for rec in items]
        self.assertEqual(len(placed), 25)
        self.assertEqual(len(set(placed)), 25)


if __name__ == "__main__":
    unittest.main()
