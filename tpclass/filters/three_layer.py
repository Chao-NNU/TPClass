"""Three-layer evidence filter for substrate-specific transporter mining.

The filter used for the substrate case studies combines three kinds of
evidence:

1. TCDB TC class whitelist (e.g. ``2.A.1``, ``3.A.1``, ``9.B``);
2. TCDB description keyword match (e.g. "fucose", "sialic", "HMO");
3. ESM-2 cosine similarity to a curated reference set of known substrate
   transporters.

The reference set can be supplied either as a pre-computed ``.npz`` of
mean-pooled ESM-2 embeddings or directly as a FASTA, in which case the
embeddings are computed on the fly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tpclass.core.refs import compute_reference_embeddings, load_embedding_npz

logger = logging.getLogger(__name__)


class ThreeLayerFilter:
    """Three-layer evidence filter (TC class + TCDB description + ESM-2 cosine sim).

    Parameters
    ----------
    ref_embeddings : str or Path, optional
        Pre-computed reference embeddings (``.npz``).
    ref_set : str or Path, optional
        Reference FASTA.  The embeddings are computed with ``esm2_path`` when
        this is given and ``ref_embeddings`` is not.
    esm2_path : str or Path, optional
        ESM-2 checkpoint used to embed ``ref_set``.
    tc_classes : list of str
        Allowed TC class prefixes.
    keywords : list of str
        TCDB description keywords to match.
    min_ref_sim : float
        Minimum maximum ESM-2 cosine similarity to the reference set.
    device : str
        Torch device, used only when ``ref_set`` has to be embedded.
    """

    def __init__(
        self,
        tc_classes: List[str],
        keywords: List[str],
        ref_embeddings: Optional[str | Path] = None,
        ref_set: Optional[str | Path] = None,
        esm2_path: Optional[str | Path] = None,
        min_ref_sim: float = 0.5,
        device: str = "cuda",
    ):
        if ref_embeddings is None and ref_set is None:
            raise ValueError("Provide either `ref_embeddings` or `ref_set`")
        self.ref_embeddings_path = Path(ref_embeddings) if ref_embeddings else None
        self.ref_set = Path(ref_set) if ref_set else None
        self.esm2_path = Path(esm2_path) if esm2_path else None
        self.tc_classes = list(tc_classes or [])
        self.keywords = [k.lower() for k in (keywords or [])]
        self.min_ref_sim = min_ref_sim
        self.device = device

        self._ref_embs: Optional[np.ndarray] = None
        self._ref_accs: List[str] = []
        self._ref_genes: Dict[str, str] = {}

    # ------------------------------------------------------------------ load
    def _load_refs(self) -> None:
        if self._ref_embs is not None:
            return
        if self.ref_embeddings_path is not None:
            logger.info(f"Loading reference embeddings from {self.ref_embeddings_path}")
            accs, matrix, genes = load_embedding_npz(self.ref_embeddings_path)
        else:
            if self.esm2_path is None:
                raise ValueError(
                    "`esm2_path` is required to embed a reference FASTA on the fly"
                )
            logger.info(f"Computing reference embeddings from {self.ref_set}")
            accs, matrix, genes = compute_reference_embeddings(
                self.ref_set, self.esm2_path, device=self.device
            )
        if matrix.size == 0:
            raise ValueError("The reference set is empty")
        self._ref_accs = accs
        self._ref_embs = matrix.astype(np.float32)
        self._ref_genes = genes
        norms = np.linalg.norm(self._ref_embs, axis=1, keepdims=True)
        self._ref_embs = self._ref_embs / np.clip(norms, 1e-9, None)
        logger.info(f"  {self._ref_embs.shape[0]} reference embeddings ready")

    # ---------------------------------------------------------------- filter
    def filter(
        self,
        v3_csv: str | Path,
        query_embeddings: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """Apply the three-layer filter.

        Parameters
        ----------
        v3_csv : str or Path
            Prediction table (e.g. the ``final_report.csv`` of the pipeline).
        query_embeddings : str or Path, optional
            ``.npz`` with the query ESM-2 embeddings.  When omitted, the
            ``max_ref_sim`` column already present in ``v3_csv`` is used.

        Returns
        -------
        pd.DataFrame
            Filtered candidates, sorted by ``max_ref_sim`` descending.
        """
        self._load_refs()
        logger.info(f"Loading candidates from {v3_csv}")
        df = pd.read_csv(v3_csv)
        logger.info(f"Before filter: {len(df)} candidates")

        # Layer 1: TC class whitelist
        if self.tc_classes:
            if "tc_number" not in df.columns:
                raise ValueError("The candidate table has no 'tc_number' column")
            df = df[df["tc_number"].apply(self._tc_matches)]
            logger.info(f"After TC class filter: {len(df)}")

        # Layer 2: TCDB description keyword
        if self.keywords:
            desc_col = self._description_column(df)
            kw_mask = df[desc_col].astype(str).str.lower().apply(
                lambda d: any(k in d for k in self.keywords)
            )
            df = df[kw_mask]
            logger.info(f"After keyword filter: {len(df)}")

        # Layer 3: ESM-2 cosine similarity to the reference set
        if query_embeddings is not None:
            q_accs, q_embs, _ = load_embedding_npz(query_embeddings)
            max_sims = self._max_sims(q_accs, q_embs)
            df["max_ref_sim"] = df["query"].map(max_sims).fillna(0.0)
        elif "max_ref_sim" not in df.columns:
            raise ValueError(
                "The candidate table has no 'max_ref_sim' column and no query "
                "embeddings were supplied; the reference-similarity layer cannot be applied."
            )

        df = df[df["max_ref_sim"] >= self.min_ref_sim]
        logger.info(f"After max_ref_sim >= {self.min_ref_sim}: {len(df)}")

        sort_cols = [c for c in ("max_ref_sim", "transport_prob", "top1_similarity") if c in df.columns]
        df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
        return df

    # -------------------------------------------------------------- internals
    def _max_sims(self, accessions: List[str], matrix: np.ndarray) -> Dict[str, float]:
        """Maximum cosine similarity of each query to any reference protein."""
        if matrix.size == 0:
            return {}
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        ref = matrix / np.clip(norms, 1e-9, None)
        sims: Dict[str, float] = {}
        for acc, emb in zip(accessions, matrix):
            q = np.asarray(emb, dtype=np.float32)
            qn = float(np.linalg.norm(q))
            if qn < 1e-9:
                continue
            sims[acc] = float(np.max((ref @ q) / qn))
        return sims

    @staticmethod
    def _description_column(df: pd.DataFrame) -> str:
        for col in ("tc_description_short", "tc_description", "description"):
            if col in df.columns:
                return col
        raise ValueError(
            f"The candidate table has no description column; found {list(df.columns)}"
        )

    def _tc_matches(self, tc: str) -> bool:
        """Check if a TC number starts with any of the allowed prefixes."""
        if not isinstance(tc, str) or not tc:
            return False
        for prefix in self.tc_classes:
            if tc == prefix or tc.startswith(prefix + "."):
                return True
        return False
