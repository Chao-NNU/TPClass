"""TPClass main predictor class.

Combines Phase 1 (binary transport classifier) and Phase 2 (TC family
assignment) into a single end-to-end pipeline, for a single backbone or for
the cascade strategy.

Cascade semantics (identical to the released pipeline and to the paper):

* the **candidate set is the set of proteins the 3B model calls YES** -- the
  650M model never removes a candidate;
* the TC family is taken from the **650M** model when the 650M model also
  called that protein YES *and* produced a family, and falls back to the **3B**
  family otherwise;
* the retrieval evidence (similarities, votes, best reference) is always taken
  from the **same model that provided the family**, so a label and its
  supporting evidence can never come from different models.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tpclass.core.esm2 import ESM2Backbone
from tpclass.core.phase1 import Phase1Classifier
from tpclass.core.phase2 import Phase2Retriever

logger = logging.getLogger(__name__)

#: Columns that carry the family assignment and its supporting evidence.
FAMILY_COLUMNS = [
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
]


#: Evidence columns the released pipeline fills with zeros when no TC family
#: was assigned to a protein.
_ZERO_FILL_COLUMNS = ("top1_similarity", "top1_sim_gap", "top_family_purity")
#: Columns the released pipeline rounds to four decimals before writing them.
_ROUNDED_COLUMNS = ("transport_prob", "top1_sim_gap", "top_family_purity")


def _is_transport(prob: float, threshold: float) -> bool:
    """Phase 1 decision rule of the released pipeline.

    The released script compares the ensemble probability with a strict ``>``:
    a protein whose probability is exactly the threshold is *not* called a
    transporter.  The same rule is used to pick the Phase 2 candidate set.
    """
    return prob > threshold


def _match_release_output_format(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the cell values of the released ``final_report.csv``.

    The released script writes ``0.0`` / ``0`` -- not empty cells -- for a
    protein that got no TC family, rounds three columns to four decimals, and
    leaves the remaining evidence columns empty.  Matching this exactly is
    what makes a row-by-row comparison against the published tables
    meaningful: without it every non-transporter row would look like a
    difference.
    """
    for col in _ZERO_FILL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)
    if "top_family_votes" in df.columns:
        df["top_family_votes"] = df["top_family_votes"].fillna(0).astype(int)
    for col in _ROUNDED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float).round(4)
    return df


class TPClassPredictor:
    """TPClass two-phase predictor.

    Parameters
    ----------
    phase1_3b_checkpoint : str or Path, optional
        Path to the 3B Phase 1 MLP weights.
    phase1_650m_checkpoint : str or Path, optional
        Path to the 650M Phase 1 MLP weights.
    phase1_xgb_3b : str or Path, optional
        Path to the 3B Phase 1 XGBoost model.
    phase1_xgb_650m : str or Path, optional
        Path to the 650M Phase 1 XGBoost model.
    phase2_3b_checkpoint : str or Path, optional
        Path to the 3B Phase 2 contrastive checkpoint.
    phase2_650m_checkpoint : str or Path, optional
        Path to the 650M Phase 2 contrastive checkpoint.
    esm2_3b_path : str or Path, optional
        ESM-2 3B checkpoint (``.pt`` or HuggingFace directory).
    esm2_650m_path : str or Path, optional
        ESM-2 650M checkpoint.
    esm2_phase2_3b : str or Path, optional
        ESM-2 backbone used by the 3B Phase 2 head.  Defaults to
        ``esm2_3b_path``.
    esm2_phase2_650m : str or Path, optional
        ESM-2 backbone used by the 650M Phase 2 head.  Defaults to
        ``esm2_650m_path``.
    ref_embeddings_3b : str or Path, optional
        3B Phase 2 reference index (``.npy``).  Defaults to
        ``tcdb_embeddings.npy`` next to the Phase 2 checkpoint.
    ref_embeddings_650m : str or Path, optional
        650M Phase 2 reference index (``.npy``).
    ref_metadata : str or Path, optional
        Reference metadata CSV.  Defaults to
        ``tcdb_index_metadata.csv`` next to the Phase 2 checkpoint.
    tcdb_annot : str or Path, optional
        TCDB annotation table (``tc_number`` -> description).  When given, the
        TC description is looked up by TC number, exactly as the released
        pipeline does with ``--tcdb_annot``.
    phase1_weights : tuple of float, optional
        ``(mlp_weight, xgb_weight)`` applied to **both** Phase 1 models.
        Defaults to the per-backbone values in
        :data:`tpclass.core.phase1.ENSEMBLE_WEIGHTS`.  The released script
        exposes the same knob (``--phase1_weights``) with the default
        ``"0.8,0.2"`` for every backbone, so setting it explicitly is what
        makes a run comparable with a specific published configuration.
    device : str, optional
        Torch device.  Defaults to CUDA when available.
    """

    def __init__(
        self,
        phase1_3b_checkpoint: Optional[str | Path] = None,
        phase1_650m_checkpoint: Optional[str | Path] = None,
        phase1_xgb_3b: Optional[str | Path] = None,
        phase1_xgb_650m: Optional[str | Path] = None,
        phase2_3b_checkpoint: Optional[str | Path] = None,
        phase2_650m_checkpoint: Optional[str | Path] = None,
        esm2_3b_path: Optional[str | Path] = None,
        esm2_650m_path: Optional[str | Path] = None,
        esm2_phase2_3b: Optional[str | Path] = None,
        esm2_phase2_650m: Optional[str | Path] = None,
        ref_embeddings_3b: Optional[str | Path] = None,
        ref_embeddings_650m: Optional[str | Path] = None,
        ref_metadata: Optional[str | Path] = None,
        tcdb_annot: Optional[str | Path] = None,
        s4pred_cache_dir: Optional[str | Path] = None,
        phase1_weights: Optional[Tuple[float, float]] = None,
        device: Optional[str] = None,
    ):
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Backbones are shared by path so that a Phase 1 and a Phase 2 stage
        # using the same ESM-2 checkpoint encode the input only once.
        self._backbones: Dict[str, ESM2Backbone] = {}

        def backbone_for(path: Optional[str | Path]) -> Optional[ESM2Backbone]:
            if path is None:
                return None
            key = str(Path(path).resolve())
            if key not in self._backbones:
                self._backbones[key] = ESM2Backbone(path=path, device=device)
            return self._backbones[key]

        # Phase 1 classifiers
        self.phase1_3b = None
        self.phase1_650m = None
        if phase1_3b_checkpoint and esm2_3b_path:
            self.phase1_3b = Phase1Classifier(
                esm2_path=esm2_3b_path,
                checkpoint=phase1_3b_checkpoint,
                xgb_model=phase1_xgb_3b,
                backbone="3B",
                device=device,
                ensemble_weights=phase1_weights,
                s4pred_cache_dir=s4pred_cache_dir,
            )
            self.phase1_3b._backbone = backbone_for(esm2_3b_path)
        if phase1_650m_checkpoint and esm2_650m_path:
            self.phase1_650m = Phase1Classifier(
                esm2_path=esm2_650m_path,
                checkpoint=phase1_650m_checkpoint,
                xgb_model=phase1_xgb_650m,
                backbone="650M",
                device=device,
                ensemble_weights=phase1_weights,
                s4pred_cache_dir=s4pred_cache_dir,
            )
            self.phase1_650m._backbone = backbone_for(esm2_650m_path)

        # Phase 2 retrievers.
        #
        # Phase 2 deliberately gets its **own** ESM-2 instance instead of sharing
        # the Phase 1 backbone: the Phase 2 checkpoint carries a partially
        # fine-tuned ESM-2, and restoring those weights into a shared backbone
        # would silently change the Phase 1 features as well.
        self.phase2_3b = None
        self.phase2_650m = None
        if phase2_3b_checkpoint:
            p2_esm = esm2_phase2_3b or esm2_3b_path
            self.phase2_3b = Phase2Retriever(
                checkpoint=phase2_3b_checkpoint,
                ref_embeddings=ref_embeddings_3b,
                ref_metadata=ref_metadata,
                tcdb_annot=tcdb_annot,
                esm2_path=p2_esm,
                backbone="3B",
                device=device,
            )
        if phase2_650m_checkpoint:
            p2_esm = esm2_phase2_650m or esm2_650m_path
            self.phase2_650m = Phase2Retriever(
                checkpoint=phase2_650m_checkpoint,
                ref_embeddings=ref_embeddings_650m,
                ref_metadata=ref_metadata,
                tcdb_annot=tcdb_annot,
                esm2_path=p2_esm,
                backbone="650M",
                device=device,
            )

    # ------------------------------------------------------------------ main
    def predict(
        self,
        fasta: str | Path,
        backbone: str = "3B",
        threshold: float = 0.50,
        run_phase2: bool = True,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Run TPClass on a FASTA file.

        Parameters
        ----------
        fasta : str or Path
            Path to the input FASTA file.
        backbone : str
            ``"3B"`` (high recall), ``"650M"`` (high family precision) or
            ``"cascade"`` (3B Phase 1 + 650M family assignment).
        threshold : float
            Phase 1 ``transport_prob`` threshold (default 0.50).
        run_phase2 : bool
            Whether to run Phase 2 (TC family assignment) on the YES proteins.
        top_k : int
            Top-k nearest neighbours used for the Phase 2 majority vote.

        Returns
        -------
        pd.DataFrame
            Columns: ``query``, ``transport_prob``, ``seq_length``,
            ``is_transport`` and, when Phase 2 ran, the family columns plus
            ``family_source`` (which model provided the family).
        """
        logger.info(f"Predicting on {fasta} with backbone={backbone}, threshold={threshold}")

        if backbone == "3B":
            return self._predict_single(
                self.phase1_3b, self.phase2_3b, fasta, threshold, run_phase2, top_k, "3B"
            )
        if backbone == "650M":
            return self._predict_single(
                self.phase1_650m, self.phase2_650m, fasta, threshold, run_phase2, top_k, "650M"
            )
        if backbone == "cascade":
            return self._predict_cascade(fasta, threshold, run_phase2, top_k)
        raise ValueError(f"Unknown backbone: {backbone}. Choose 3B, 650M, or cascade.")

    # --------------------------------------------------------------- internals
    def _predict_single(
        self, phase1, phase2, fasta, threshold, run_phase2, top_k, source_label=""
    ) -> pd.DataFrame:
        """Run a single-backbone prediction."""
        if phase1 is None:
            raise ValueError("No Phase 1 model was configured for this backbone")

        from tpclass.core.embeddings import read_fasta_ordered

        _all_names, sequences = read_fasta_ordered(fasta)
        rows: List[dict] = []
        family_rows: List[pd.DataFrame] = []

        for batch_names, features, token_emb, input_ids in phase1.iter_batches(fasta):
            probs = phase1.score(features)
            for name, p in zip(batch_names, probs):
                rows.append(
                    {
                        "query": name,
                        "transport_prob": float(p),
                        "seq_length": len(sequences[name]),
                    }
                )

            if run_phase2 and phase2 is not None:
                yes_idx = [i for i, p in enumerate(probs) if _is_transport(p, threshold)]
                if yes_idx:
                    yes_names = [batch_names[i] for i in yes_idx]
                    embeddings = self._embed_for_phase2(
                        phase1, phase2, yes_names, sequences, token_emb, input_ids, yes_idx
                    )
                    if embeddings:
                        family_rows.append(phase2.assign(embeddings, top_k=top_k))

        df = pd.DataFrame(rows, columns=["query", "transport_prob", "seq_length"])
        df["is_transport"] = np.where(
            df["transport_prob"].map(lambda p: _is_transport(p, threshold)), "YES", "NO"
        )
        df["family_source"] = ""
        if family_rows:
            families = pd.concat(family_rows, ignore_index=True)
            df = df.merge(families, on="query", how="left")
            has_family = df["tc_number"].notna() & (df["tc_number"].astype(str) != "")
            df["family_source"] = np.where(has_family, source_label, "")
        return _match_release_output_format(df)

    def _embed_for_phase2(self, phase1, phase2, names, sequences, token_emb, input_ids, yes_idx):
        """Compute Phase 2 embeddings, reusing the Phase 1 token batch if possible."""
        same_backbone = (
            phase1 is not None
            and phase1.backbone_model is not None
            and phase2.backbone_model is not None
            and phase1.backbone_model is phase2.backbone_model
        )
        if same_backbone:
            sub_emb = token_emb[yes_idx]
            sub_ids = input_ids[yes_idx]
            return phase2.embed_queries(
                names, sequences, token_batch=(sub_emb, sub_ids)
            )
        return phase2.embed_queries(names, sequences)

    def _predict_cascade(self, fasta, threshold, run_phase2, top_k) -> pd.DataFrame:
        """Run the cascade: 3B Phase 1 filter, 650M family with a 3B fallback."""
        if self.phase1_3b is None:
            raise ValueError("The cascade requires the 3B Phase 1 model")
        if self.phase1_650m is None:
            raise ValueError("The cascade requires the 650M Phase 1 model")
        if run_phase2 and self.phase2_650m is None:
            raise ValueError("The cascade requires the 650M Phase 2 model")
        if run_phase2 and self.phase2_3b is None:
            logger.warning(
                "No 3B Phase 2 model configured: candidates that the 650M model did not "
                "call YES will have no TC family (no 3B fallback)."
            )

        df_3b = self._predict_single(
            self.phase1_3b, self.phase2_3b, fasta, threshold, run_phase2, top_k, "3B"
        )
        df_650m = self._predict_single(
            self.phase1_650m, self.phase2_650m, fasta, threshold, run_phase2, top_k, "650M"
        )

        # The candidate set is the 3B YES set: the 650M model never removes a candidate.
        cand = df_3b[df_3b["is_transport"] == "YES"].copy().reset_index(drop=True)
        if not run_phase2:
            cand["family_source"] = ""
            return cand

        m650 = df_650m.set_index("query")
        fam_cols = [c for c in FAMILY_COLUMNS if c in df_650m.columns]

        for i in range(len(cand)):
            q = cand.at[i, "query"]
            if q not in m650.index:
                cand.at[i, "family_source"] = "3B"
                continue
            other = m650.loc[q]
            if other["is_transport"] != "YES":
                cand.at[i, "family_source"] = "3B"
                continue
            other_tc = other.get("tc_number", "")
            if not isinstance(other_tc, str) or not other_tc.strip():
                # 650M said YES but produced no family: keep the 3B family.
                cand.at[i, "family_source"] = "3B"
                continue
            # Use the 650M family *and* the 650M evidence that supports it.
            for col in fam_cols:
                cand.at[i, col] = other[col]
            cand.at[i, "family_source"] = "650M"

        # Keep the 650M Phase 1 probability for transparency.
        cand["transport_prob_650m"] = cand["query"].map(m650["transport_prob"])
        return cand
