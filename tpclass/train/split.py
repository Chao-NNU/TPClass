"""Dataset splitting for TPClass training.

Two strategies are provided:

``stratified_split``
    The split used by the released pipeline: a random split stratified by TC
    family.  It is simple and reproducible, but it puts homologous proteins
    into different subsets, so a model can be evaluated on sequences that are
    close to its training data (homology leakage).

``cluster_split``
    A homology-aware split.  Sequences are clustered by k-mer similarity and
    whole clusters are assigned to a subset, so no test protein has a close
    homologue in train or validation.  This is the split to use when the
    reported performance has to be free of homology leakage.

Both return the same structure so they are drop-in replacements.
"""
from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)

#: k-mer size used for the similarity clustering.
KMER_SIZE = 5
#: Minimum Jaccard similarity for two sequences to end up in the same cluster.
CLUSTER_THRESHOLD = 0.5


# ══════════════════════════════════════════════════════════════════════════
#  Stratified split (released-pipeline behaviour)
# ══════════════════════════════════════════════════════════════════════════
def stratified_split(
    records: Sequence[dict],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """Split records into train/val/test, stratified by TC family.

    Parameters
    ----------
    records : sequence of dict
        Each record must carry a ``tc_family`` key (and ``accession`` /
        ``sequence`` for the FASTA writer).
    train_frac, val_frac : float
        Target fractions; the remainder goes to the test split.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``.
    """
    test_frac = max(0.0, 1.0 - train_frac - val_frac)
    rng = random.Random(seed)

    by_family: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        by_family[str(rec.get("tc_family", "?"))].append(rec)

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for family in sorted(by_family):
        items = list(by_family[family])
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for name in splits:
        rng.shuffle(splits[name])
    logger.info(
        "stratified split: "
        + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        + f" (train_frac={train_frac}, val_frac={val_frac})"
    )
    return splits


# ══════════════════════════════════════════════════════════════════════════
#  Homology-aware cluster split
# ══════════════════════════════════════════════════════════════════════════
def kmer_set(sequence: str, k: int = KMER_SIZE) -> frozenset:
    """The set of k-mers of a sequence."""
    if len(sequence) < k:
        return frozenset([sequence])
    return frozenset(sequence[i : i + k] for i in range(len(sequence) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def cluster_sequences(
    sequences: Dict[str, str],
    threshold: float = CLUSTER_THRESHOLD,
    k: int = KMER_SIZE,
) -> Dict[str, int]:
    """Cluster sequences by k-mer similarity.

    Pairs are found through an inverted k-mer index, so only sequences that
    share at least one k-mer are compared; with a positive similarity
    threshold this does not miss any qualifying pair.

    Returns
    -------
    dict
        ``{accession: cluster_id}``.
    """
    kmers = {acc: kmer_set(seq, k) for acc, seq in sequences.items()}

    inverted: Dict[str, List[str]] = defaultdict(list)
    for acc, ks in kmers.items():
        for kmer in ks:
            inverted[kmer].append(acc)

    parent: Dict[str, str] = {acc: acc for acc in sequences}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    checked: set = set()
    for acc, ks in kmers.items():
        candidates: Counter = Counter()
        for kmer in ks:
            for other in inverted[kmer]:
                if other != acc:
                    candidates[other] += 1
        for other in candidates:
            key = (acc, other) if acc < other else (other, acc)
            if key in checked:
                continue
            checked.add(key)
            if jaccard(ks, kmers[other]) >= threshold:
                union(acc, other)

    roots = sorted({find(a) for a in sequences})
    root_to_id = {r: i for i, r in enumerate(roots)}
    return {acc: root_to_id[find(acc)] for acc in sequences}


def cluster_split(
    records: Sequence[dict],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    threshold: float = CLUSTER_THRESHOLD,
    k: int = KMER_SIZE,
) -> Dict[str, List[dict]]:
    """Split records so that homologous sequences stay in the same subset.

    Parameters
    ----------
    records : sequence of dict
        Each record must carry ``accession``, ``sequence`` and ideally
        ``tc_family``.
    train_frac, val_frac : float
        Target fractions; the remainder goes to the test split.
    seed : int
        Random seed (used to break ties between equally sized clusters).
    threshold : float
        Minimum Jaccard similarity for two sequences to be in one cluster.
    k : int
        k-mer size.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``.
    """
    rng = random.Random(seed)
    sequences = {str(r["accession"]): str(r.get("sequence", "")) for r in records}
    assignment = cluster_sequences(sequences, threshold=threshold, k=k)

    clusters: Dict[int, List[dict]] = defaultdict(list)
    for rec in records:
        clusters[assignment[str(rec["accession"])]].append(rec)

    # Deterministic ordering: biggest cluster first, then by dominant family.
    def cluster_key(item):
        cid, members = item
        fam = Counter(str(m.get("tc_family", "?")) for m in members).most_common(1)
        return (-len(members), fam[0][0] if fam else "", cid)

    ordered = sorted(clusters.items(), key=cluster_key)

    total = len(records)
    targets = {
        "train": int(round(total * train_frac)),
        "val": int(round(total * val_frac)),
        "test": total - int(round(total * train_frac)) - int(round(total * val_frac)),
    }
    assigned = {"train": 0, "val": 0, "test": 0}
    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    order = ("train", "val", "test")

    # Greedy deficit filling: every cluster goes to the split that is
    # furthest below its target (ties resolved towards train, then val).
    for _cid, members in ordered:
        target_split = max(order, key=lambda s: (targets[s] - assigned[s], -order.index(s)))
        splits[target_split].extend(members)
        assigned[target_split] += len(members)

    for name in splits:
        rng.shuffle(splits[name])
    logger.info(
        "cluster split (Jaccard >= %.2f, k=%d): "
        % (threshold, k)
        + ", ".join(f"{k2}={len(v)}" for k2, v in splits.items())
        + f" from {len(clusters)} clusters"
    )
    return splits


# ══════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ══════════════════════════════════════════════════════════════════════════
def write_fasta(records: Sequence[dict], path: str | Path) -> Path:
    """Write records to a FASTA file (``accession`` + ``sequence``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for rec in records:
            handle.write(f">{rec['accession']}\n{rec.get('sequence', '')}\n")
    logger.info(f"[Save] {path} ({len(records)} sequences)")
    return path


def write_split_csv(splits: Dict[str, List[dict]], path: str | Path) -> Path:
    """Write a CSV with one row per record: ``accession, split, tc_family``."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accession", "split", "tc_family", "tc_number"])
        for split, records in splits.items():
            for rec in records:
                writer.writerow(
                    [
                        rec.get("accession", ""),
                        split,
                        rec.get("tc_family", ""),
                        rec.get("tc_number", ""),
                    ]
                )
    logger.info(f"[Save] {path}")
    return path
