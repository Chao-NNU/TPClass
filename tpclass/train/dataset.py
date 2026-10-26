"""Dataset construction for TPClass training.

Turns a TCDB FASTA (positives, with TC numbers) plus a set of negative
sequences into the feature matrices and labels the Phase 1 trainer expects:

``X_train.npy`` / ``Y_train.npy`` / ``X_val.npy`` / ``Y_val.npy``

The features are produced by the same code path used at inference time, so a
model trained here is guaranteed to see the representation it will see later.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: ``>sp|P12345|...`` or a bare accession.
_ACC_RE = re.compile(r"^>(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|")
#: TC number in a TCDB header, e.g. ``2.A.1.2``.
_TC_RE = re.compile(r"\b(\d+\.[A-Z]\.\d+(?:\.\d+)*)\b")


def tc_family(tc_number: str) -> str:
    """The three-component family of a TC number (``2.A.1.2`` -> ``2.A.1``)."""
    parts = str(tc_number).split(".")
    return ".".join(parts[:3]) if len(parts) >= 3 else str(tc_number)


def load_tcdb_records(
    fasta: str | Path,
    negatives: Optional[str | Path] = None,
) -> List[dict]:
    """Read a TCDB FASTA into records, optionally adding negative sequences.

    Parameters
    ----------
    fasta : str or Path
        TCDB protein FASTA.  The TC number is taken from the header when
        present.
    negatives : str or Path, optional
        FASTA of non-transporter sequences.  These become label 0.

    Returns
    -------
    list of dict
        Each record has ``accession``, ``sequence``, ``tc_number``,
        ``tc_family`` and ``label`` (1 for TCDB entries, 0 for negatives).
    """
    from tpclass.core.embeddings import read_fasta_ordered

    records: List[dict] = []
    names, sequences = read_fasta_ordered(fasta)
    headers = _read_headers(fasta)
    for acc in names:
        m = _TC_RE.search(headers.get(acc, ""))
        tc_number = m.group(1) if m else ""
        records.append(
            {
                "accession": acc,
                "sequence": sequences[acc],
                "tc_number": tc_number,
                "tc_family": tc_family(tc_number) if tc_number else "",
                "label": 1,
            }
        )
    logger.info(f"Loaded {len(records)} TCDB records from {fasta}")

    if negatives:
        neg_names, neg_sequences = read_fasta_ordered(negatives)
        for acc in neg_names:
            records.append(
                {
                    "accession": acc,
                    "sequence": neg_sequences[acc],
                    "tc_number": "",
                    "tc_family": "",
                    "label": 0,
                }
            )
        logger.info(f"Loaded {len(neg_names)} negative sequences from {negatives}")

    return records


def _read_headers(fasta: str | Path) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    acc: Optional[str] = None
    with open(fasta) as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            m = _ACC_RE.match(line)
            if m:
                acc = m.group(1)
            else:
                parts = line[1:].split()
                acc = parts[0] if parts else None
            if acc:
                headers[acc] = line[1:].strip()
    return headers


def build_training_dataset(
    positives: str | Path,
    negatives: Optional[str | Path],
    esm2_path: str | Path,
    out_dir: str | Path,
    splits: Dict[str, List[dict]],
    device: str = "cuda",
    max_len: int = 800,
    batch_size: int = 8,
) -> Dict[str, Path]:
    """Extract features for every split and write the ``.npy`` training arrays.

    Parameters
    ----------
    positives : str or Path
        TCDB FASTA.
    negatives : str or Path, optional
        Non-transporter FASTA.
    esm2_path : str or Path
        ESM-2 checkpoint used to embed the sequences.
    out_dir : str or Path
        Directory that receives ``X_<split>.npy`` / ``Y_<split>.npy``.
    splits : dict
        ``{"train": [...], "val": [...], "test": [...]}`` of records, as
        produced by :func:`tpclass.train.split.cluster_split`.

    Returns
    -------
    dict
        ``{split: path-of-X-array}``.
    """
    from tpclass.train.phase1 import extract_features

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    for split, records in splits.items():
        if not records:
            logger.warning(f"Split '{split}' is empty; skipping")
            continue
        split_fasta = out_dir / f"{split}.fasta"
        with open(split_fasta, "w") as handle:
            for rec in records:
                handle.write(f">{rec['accession']}\n{rec['sequence']}\n")
        logger.info(f"[{split}] {len(records)} sequences -> {split_fasta}")

        names, features = extract_features(
            split_fasta, esm2_path, device=device, max_len=max_len, batch_size=batch_size
        )
        name_to_label = {str(r["accession"]): int(r.get("label", 1)) for r in records}
        labels = np.array([name_to_label.get(n, 1) for n in names], dtype=np.int64)

        x_path = out_dir / f"X_{split}.npy"
        y_path = out_dir / f"Y_{split}.npy"
        np.save(x_path, features)
        np.save(y_path, labels)
        logger.info(
            f"[{split}] features {features.shape}, positives {int(labels.sum())}/{len(labels)}"
        )
        written[split] = x_path

    return written
