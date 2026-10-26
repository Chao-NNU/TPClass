"""Reference embedding computation.

Two kinds of "reference" are used by TPClass and they are **not** the same
object:

* **Phase 2 reference index** -- ``tcdb_embeddings.npy`` plus
  ``tcdb_index_metadata.csv``.  These are *contrastive* embeddings produced by
  the Phase 2 head and must be built with the Phase 2 checkpoint; they drive
  TC family assignment.
* **Substrate reference set** -- an ``.npz`` of *mean-pooled ESM-2* embeddings
  for a handful of known substrate transporters.  These drive the third layer
  of the three-layer filter (cosine similarity to a known reference protein).

This module implements the second kind; :mod:`tpclass.core.phase2` consumes
the first.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tpclass.core.esm2 import ESM2Backbone
from tpclass.core.embeddings import is_valid_sequence, read_fasta_ordered

logger = logging.getLogger(__name__)

#: Accepted key names when reading an ``.npz`` of embeddings.
ACCESSION_KEYS = ("accessions", "acc", "ids", "id", "proteins")
EMBEDDING_KEYS = ("embeddings", "embedding", "embs", "emb")
GENE_KEYS = ("genes", "gene", "gene_names")

_GENE_RE = re.compile(r"\b(?:gene=|GN=)\s*([A-Za-z0-9_.\-]+)")


def load_embedding_npz(path: str | Path) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    """Load an embedding ``.npz``.

    Accepts both the key spelling used by the released pipeline
    (``accessions`` / ``embeddings``) and the shorter ``acc`` / ``embedding``
    spelling.

    Returns
    -------
    (accessions, matrix, genes)
    """
    data = np.load(path, allow_pickle=True)
    acc_key = next((k for k in ACCESSION_KEYS if k in data.files), None)
    emb_key = next((k for k in EMBEDDING_KEYS if k in data.files), None)
    if acc_key is None or emb_key is None:
        raise ValueError(
            f"{path} does not contain a usable embedding matrix; found keys {data.files}. "
            f"Expected one of {ACCESSION_KEYS} and one of {EMBEDDING_KEYS}."
        )
    accessions = [str(a) for a in data[acc_key]]
    matrix = np.asarray(data[emb_key], dtype=np.float32)
    if matrix.shape[0] != len(accessions):
        raise ValueError(
            f"{path}: {len(accessions)} accessions but {matrix.shape[0]} embedding rows"
        )
    gene_key = next((k for k in GENE_KEYS if k in data.files), None)
    genes: Dict[str, str] = {}
    if gene_key:
        genes = dict(zip(accessions, [str(g) for g in data[gene_key]]))
    return accessions, matrix, genes


def save_embedding_npz(
    path: str | Path,
    accessions: Sequence[str],
    embeddings: np.ndarray,
    genes: Optional[Sequence[str]] = None,
) -> Path:
    """Save an embedding matrix as ``.npz`` using the pipeline's key names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "accessions": np.array(list(accessions)),
        "embeddings": np.asarray(embeddings, dtype=np.float32),
    }
    if genes is not None:
        payload["genes"] = np.array(list(genes))
    np.savez(path, **payload)
    logger.info(f"[Save] {path} ({embeddings.shape[0]} x {embeddings.shape[1]})")
    return path


def compute_reference_embeddings(
    fasta: str | Path,
    esm2_path: str | Path,
    out_npz: Optional[str | Path] = None,
    device: str = "cuda",
    max_len: int = 800,
    batch_size: int = 8,
) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    """Compute mean-pooled ESM-2 embeddings for a reference FASTA.

    The embeddings are the same Phase 1 representation (mean over the last
    layer, pads and ``eos`` excluded), so they are directly comparable with
    the ``embeddings.npz`` written by a prediction run.

    Parameters
    ----------
    fasta : str or Path
        Reference FASTA.
    esm2_path : str or Path
        ESM-2 checkpoint (``.pt`` or HuggingFace directory).
    out_npz : str or Path, optional
        When given, the embeddings are also written to this ``.npz``.
    device : str
        Torch device.
    max_len, batch_size : int
        Encoding controls.

    Returns
    -------
    (accessions, embeddings, genes)
    """
    names, sequences = read_fasta_ordered(fasta)
    if not names:
        raise ValueError(f"No sequences found in {fasta}")
    keep = [n for n in names if is_valid_sequence(sequences[n])]
    if len(keep) != len(names):
        logger.warning(
            f"Dropped {len(names) - len(keep)}/{len(names)} sequences failing the validity filter"
        )
    if not keep:
        raise ValueError(f"All sequences in {fasta} failed the validity filter")

    backbone = ESM2Backbone(
        path=esm2_path, device=device, max_len=max_len, batch_size=batch_size
    ).load()

    vectors: List[np.ndarray] = []
    for start in range(0, len(keep), batch_size):
        batch = keep[start : start + batch_size]
        _, token_emb, input_ids = backbone.encode_all(
            [sequences[n] for n in batch], batch_size=len(batch)
        )
        token_emb = token_emb.to(backbone.device)
        input_ids = input_ids.to(backbone.device)
        pooled = backbone.phase1_mean_pool(
            token_emb, input_ids, backbone.pad_id, backbone.eos_id
        )
        vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

    matrix = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), np.float32)
    headers = read_fasta_headers(fasta)
    genes: Dict[str, str] = {}
    for name in keep:
        m = _GENE_RE.search(headers.get(name, ""))
        if m:
            genes[name] = m.group(1)
    if out_npz:
        save_embedding_npz(
            out_npz,
            keep,
            matrix,
            [genes.get(n, "") for n in keep] if genes else None,
        )
    return keep, matrix, genes


def read_fasta_headers(fasta: str | Path) -> Dict[str, str]:
    """Return ``{accession: raw header}`` for a FASTA file."""
    headers: Dict[str, str] = {}
    acc: Optional[str] = None
    with open(fasta) as handle:
        for line in handle:
            if line.startswith(">"):
                m = re.match(r"^>(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|", line)
                if m:
                    acc = m.group(1)
                else:
                    parts = line[1:].split()
                    acc = parts[0] if parts else None
                if acc:
                    headers[acc] = line[1:].strip()
    return headers
