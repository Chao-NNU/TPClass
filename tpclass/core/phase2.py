"""Phase 2: TC family assignment via contrastive retrieval.

A contrastively trained ESM-2 head maps a query protein into the same
embedding space as a TCDB reference index; the top-k nearest neighbours are
retrieved by cosine similarity and the TC family is assigned by majority vote
over the ``tc_family`` field, with the top-1 hit providing the four-digit
``tc_number``.

Reference index layout (as produced by the released pipeline and by
``tpclass compute-refs``):

* ``tcdb_embeddings.npy`` -- ``(n_ref, embed_dim)`` float matrix, L2-normalised;
* ``tcdb_index_metadata.csv`` -- one row per reference, with at least
  ``tc_number`` and ``tc_family`` columns.

Both files are expected next to the Phase 2 checkpoint: the reference index and
the head must come from the same training run, otherwise the embedding spaces
do not match.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tpclass.core.esm2 import ESM2Backbone
from tpclass.core.embeddings import extract_query_embeddings, load_contrastive_head

logger = logging.getLogger(__name__)

#: Reference metadata column candidates, in priority order.
ID_COLUMNS = ("acc", "accession", "id", "seq_id", "ref_id", "query", "header", "name")
DESCRIPTION_COLUMNS = (
    "description",
    "tc_description",
    "family_description",
    "tc_family_description",
    "annotation",
)
GENE_COLUMNS = ("gene", "gene_name", "locus_tag", "gene_name_or_locus")


def load_tcdb_annotations(annot_file: str | Path | None) -> Dict[str, str]:
    """Load a TCDB annotation table (``tc_number`` -> description).

    The file is the same one the released pipeline reads with ``--tcdb_annot``:
    one record per line, tab-separated (comma is accepted as a fallback), with
    the TC number in the first column and the family description in the second.
    A leading header line is detected and skipped.

    Parameters
    ----------
    annot_file : str or Path, optional
        Path to the annotation table.  A missing file yields an empty mapping
        rather than an error, which is what the released script does.

    Returns
    -------
    dict
        Mapping from TC number to family description.
    """
    if not annot_file:
        return {}
    path = Path(annot_file)
    if not path.exists():
        return {}
    annot: Dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        first = True
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sep = "\t" if "\t" in line else ","
            parts = [p.strip().strip('"') for p in line.split(sep)]
            if first:
                first = False
                if not parts or not re.match(r"^\d\.[A-Z]\.\d", parts[0]):
                    continue
            if len(parts) >= 2:
                annot[parts[0]] = parts[1]
    return annot


class Phase2Retriever:
    """Phase 2 contrastive retriever for TC family assignment.

    Parameters
    ----------
    checkpoint : str or Path
        Phase 2 contrastive checkpoint (``.pth``).
    ref_embeddings : str or Path, optional
        Reference embedding matrix (``.npy``).  Defaults to
        ``<checkpoint dir>/tcdb_embeddings.npy``.
    ref_metadata : str or Path, optional
        Reference metadata CSV.  Defaults to
        ``<checkpoint dir>/tcdb_index_metadata.csv``.
    tcdb_annot : str or Path, optional
        TCDB annotation table (tc_number -> description).  When given, the TC
        description in the output is looked up by TC number, exactly as the
        released pipeline does with --tcdb_annot.  Without it, the description
        of the matched reference protein is used.
    esm2_path : str or Path, optional
        Phase 2 ESM-2 backbone.  Required when the retriever has to embed the
        queries itself; can be skipped if the caller supplies embeddings.
    backbone : str
        ``"3B"`` or ``"650M"``.  The published Phase 2 head was trained on the
        650M backbone.
    device : str
        Torch device.
    embed_dim : int
        Dimensionality of the contrastive embedding.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        ref_embeddings: Optional[str | Path] = None,
        ref_metadata: Optional[str | Path] = None,
        tcdb_annot: Optional[str | Path] = None,
        esm2_path: Optional[str | Path] = None,
        backbone: str = "650M",
        device: str = "cuda",
        embed_dim: int = 512,
    ):
        self.checkpoint = Path(checkpoint)
        self.backbone = backbone
        self.device = device
        self.embed_dim = embed_dim

        ckpt_dir = self.checkpoint.parent
        self._ref_embeddings_override = (
            Path(ref_embeddings) if ref_embeddings else None
        )
        self._ref_metadata_override = Path(ref_metadata) if ref_metadata else None
        # Resolved lazily in _load(): the index may sit next to the checkpoint
        # (released layout) or one directory above it (release bundle layout).
        self.ref_embeddings = self._ref_embeddings_override or (
            ckpt_dir / "tcdb_embeddings.npy"
        )
        self.ref_metadata = self._ref_metadata_override or (
            ckpt_dir / "tcdb_index_metadata.csv"
        )
        self.esm2_path = Path(esm2_path) if esm2_path else None

        self._head = None
        self._backbone: Optional[ESM2Backbone] = None
        self._ref_embs: Optional[np.ndarray] = None
        self._ref_meta: Optional[pd.DataFrame] = None
        self._ref_ids: Optional[np.ndarray] = None
        self._ref_tc: Optional[np.ndarray] = None
        self._ref_family: Optional[np.ndarray] = None
        self._ref_desc: Dict[str, str] = {}
        self._ref_gene: Dict[str, str] = {}
        self._tcdb_annot: Optional[Path] = Path(tcdb_annot) if tcdb_annot else None
        self._tc_annot: Dict[str, str] = {}

    # ------------------------------------------------------------------ load
    @property
    def backbone_model(self) -> ESM2Backbone:
        """Phase 2 ESM-2 backbone (loaded on first access)."""
        if self._backbone is None:
            if self.esm2_path is None:
                raise RuntimeError(
                    "No Phase 2 ESM-2 backbone configured. Pass esm2_path=... so the "
                    "retriever can embed queries, or supply pre-computed embeddings."
                )
            self._backbone = ESM2Backbone(
                path=self.esm2_path, device=self.device
            ).load()
        return self._backbone

    def _resolve_ref_paths(self) -> None:
        """Locate the reference index and its metadata.

        Search order for each file: the explicitly supplied path, the
        checkpoint directory (the layout of the released pipeline), then the
        checkpoint's parent directory (the layout produced by
        ``models/download.sh``, where the bundle root holds the index).
        """
        ckpt_dir = self.checkpoint.parent

        def resolve(override: Optional[Path], filename: str) -> Path:
            if override is not None:
                return override
            candidates = [ckpt_dir / filename, ckpt_dir.parent / filename]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]

        self.ref_embeddings = resolve(
            self._ref_embeddings_override, "tcdb_embeddings.npy"
        )
        self.ref_metadata = resolve(
            self._ref_metadata_override, "tcdb_index_metadata.csv"
        )

    def _load(self) -> None:
        if self._ref_embs is not None:
            return

        self._resolve_ref_paths()

        if not self.ref_embeddings.exists():
            raise FileNotFoundError(
                f"Reference embeddings not found: {self.ref_embeddings}. "
                "Run `tpclass compute-refs` or point --ref_emb_* at the index that "
                "was built with the same Phase 2 checkpoint."
            )
        emb = np.load(self.ref_embeddings)
        if emb.ndim != 2:
            raise ValueError(f"Expected a 2-D reference matrix, got shape {emb.shape}")
        self._ref_embs = emb.astype(np.float32)
        logger.info(
            f"Loaded {self._ref_embs.shape[0]} reference embeddings "
            f"({self._ref_embs.shape[1]}-d) from {self.ref_embeddings}"
        )

        if not self.ref_metadata.exists():
            raise FileNotFoundError(f"Reference metadata not found: {self.ref_metadata}")
        meta = pd.read_csv(self.ref_metadata)
        if len(meta) != self._ref_embs.shape[0]:
            raise ValueError(
                f"Reference metadata has {len(meta)} rows but the embedding matrix has "
                f"{self._ref_embs.shape[0]} rows; they must correspond one-to-one."
            )
        if "tc_number" not in meta.columns:
            raise ValueError(
                f"Reference metadata must contain a 'tc_number' column; found "
                f"{list(meta.columns)}"
            )
        if "tc_family" not in meta.columns:
            # Fall back to the first three dot-separated components.
            meta["tc_family"] = meta["tc_number"].astype(str).str.split(".").str[:3].str.join(".")
            logger.warning("Reference metadata has no 'tc_family' column; derived it from tc_number")
        self._ref_meta = meta
        self._ref_tc = meta["tc_number"].astype(str).to_numpy()
        self._ref_family = meta["tc_family"].astype(str).to_numpy()

        id_col = next((c for c in ID_COLUMNS if c in meta.columns), None)
        if id_col is None:
            id_col = meta.columns[0]
            logger.warning(f"Could not detect a reference id column; using '{id_col}'")
        self._ref_ids = meta[id_col].astype(str).to_numpy()

        desc_col = next((c for c in DESCRIPTION_COLUMNS if c in meta.columns), None)
        if desc_col:
            self._ref_desc = dict(zip(meta[id_col].astype(str), meta[desc_col].astype(str)))
        gene_col = next((c for c in GENE_COLUMNS if c in meta.columns), None)
        if gene_col:
            self._ref_gene = dict(zip(meta[id_col].astype(str), meta[gene_col].astype(str)))

        # The released pipeline takes the TC description from the TCDB
        # annotation table, keyed by TC number.  When that table is supplied
        # it wins over the per-reference description, so that the output
        # table matches the released one row for row.
        if self._tcdb_annot is not None:
            self._tc_annot = load_tcdb_annotations(self._tcdb_annot)
            logger.info(
                f"Loaded {len(self._tc_annot)} TCDB annotations from {self._tcdb_annot}"
            )

        # The reference index and the head must come from the same training run.
        if self.ref_embeddings.parent.resolve() != self.checkpoint.parent.resolve():
            logger.warning(
                f"Reference embeddings live in {self.ref_embeddings.parent} while the Phase 2 "
                f"checkpoint lives in {self.checkpoint.parent}. The pairing of head and index "
                "cannot be verified from the paths alone; make sure both come from the same "
                "training run, otherwise the query and the reference sit in different "
                "embedding spaces."
            )
        else:
            logger.info(
                "Reference index sits next to the Phase 2 checkpoint (same training run assumed)."
            )

    # -------------------------------------------------------------- embedding
    def embed_queries(
        self,
        names: Sequence[str],
        sequences: Dict[str, str],
        token_batch: Optional[Tuple["object", "object"]] = None,
    ) -> Dict[str, np.ndarray]:
        """Embed query proteins with the Phase 2 backbone and head."""
        self._load()
        backbone = self.backbone_model
        if self._head is None:
            self._head = load_contrastive_head(
                self.checkpoint, backbone, device=self.device, embed_dim=self.embed_dim
            )
        return extract_query_embeddings(
            sequences,
            backbone=backbone,
            projection=self._head,
            names=list(names),
            token_batch=token_batch,
            batch_size=backbone.batch_size,
        )

    # -------------------------------------------------------------- retrieval
    def _description_for(self, tc_number: str, ref_id: str) -> str:
        """TC description, preferring the TCDB annotation table.

        The released pipeline looks the description up by TC number in the
        annotation table; the reference metadata only carries the description
        of the matched reference protein.  When the table is available it
        wins, and a TC number that is absent from it yields an empty
        description, which is what the released script writes.
        """
        if self._tc_annot:
            return self._tc_annot.get(tc_number, "")
        return self._ref_desc.get(ref_id, "")


    def assign(
        self,
        embeddings: Dict[str, np.ndarray],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Assign TC families from pre-computed query embeddings.

        Parameters
        ----------
        embeddings : dict
            ``{accession: embedding}``.
        top_k : int
            Number of neighbours retrieved per query.

        Returns
        -------
        pd.DataFrame
            Columns: ``query``, ``tc_number``, ``tc_family``, ``tc_description``,
            ``top1_similarity``, ``top1_sim_gap``, ``top_family_votes``,
            ``top_family_purity``, ``max_ref_sim``, ``best_ref_acc``,
            ``best_ref_gene``.
        """
        self._load()
        results = []
        for query, emb in embeddings.items():
            q = np.asarray(emb, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                continue
            sims = (self._ref_embs @ q) / (
                np.linalg.norm(self._ref_embs, axis=1) * q_norm + 1e-9
            )
            k = min(top_k, sims.shape[0])
            top_idx = np.argsort(-sims)[:k]
            top_sims = sims[top_idx]
            top_families = self._ref_family[top_idx]
            top_ids = self._ref_ids[top_idx]

            votes = Counter(top_families.tolist()).most_common(1)
            family = votes[0][0] if votes else "?"
            n_votes = int(votes[0][1]) if votes else 0

            top1_idx = int(top_idx[0])
            tc_number = str(self._ref_tc[top1_idx])
            top1_sim = float(top_sims[0])
            top2_sim = float(top_sims[1]) if k > 1 else 0.0
            best_id = str(top_ids[0])

            results.append(
                {
                    "query": query,
                    "tc_number": tc_number,
                    "tc_family": family,
                    "tc_description": self._description_for(tc_number, best_id),
                    "top1_similarity": top1_sim,
                    "top1_sim_gap": top1_sim - top2_sim,
                    "top_family_votes": n_votes,
                    "top_family_purity": n_votes / top_k if top_k else 0.0,
                    "max_ref_sim": top1_sim,
                    "best_ref_acc": best_id,
                    "best_ref_gene": self._ref_gene.get(best_id, ""),
                }
            )
        return pd.DataFrame(
            results,
            columns=[
                "query",
                "tc_number",
                "tc_family",
                "tc_description",
                "top1_similarity",
                "top1_sim_gap",
                "top_family_votes",
                "top_family_purity",
                "max_ref_sim",
                "best_ref_acc",
                "best_ref_gene",
            ],
        )

    def predict(
        self,
        queries: Sequence[str],
        sequences: Optional[Dict[str, str]] = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Embed and assign TC families for a list of accessions.

        Parameters
        ----------
        queries : sequence of str
            Accessions to annotate.
        sequences : dict, optional
            ``{accession: sequence}``.  Required when the retriever has to do
            the embedding itself.
        top_k : int
            Number of neighbours retrieved per query.
        """
        if not sequences:
            raise ValueError(
                "`sequences` is required: the retriever needs the query protein "
                "sequences to compute the Phase 2 embeddings."
            )
        embeddings = self.embed_queries(list(queries), sequences)
        return self.assign(embeddings, top_k=top_k)
