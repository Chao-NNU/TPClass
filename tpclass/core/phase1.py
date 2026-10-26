"""Phase 1: binary transporter classifier.

Loads the ESM-2 backbone and the Phase 1 assets (MLP + XGBoost), builds the
feature matrix and outputs a calibrated ``transport_prob`` in ``[0, 1]``.

The feature matrix is ``[ESM-2 mean-pooled embedding | s4pred H/E/C]`` with one
row per input sequence **in FASTA order**; the ESM-2 block and the s4pred
block are assembled from the same ordered accession list, so they can never
drift out of alignment.

:meth:`Phase1Classifier.iter_batches` exposes the encoder one batch at a time so
that a Phase 2 stage sharing the same backbone can reuse the token embeddings
instead of paying for a second forward pass.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tpclass.core.esm2 import ESM2Backbone
from tpclass.core.embeddings import (
    extract_s4pred_features,
    is_valid_sequence,
    read_fasta_ordered,
)

logger = logging.getLogger(__name__)

#: Ensemble weights ``(mlp, xgb)`` per backbone, as used in the paper.
ENSEMBLE_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "3B": (0.8, 0.2),
    "650M": (0.4, 0.6),
}


class ResidualBlock(nn.Module):
    """Residual block of the Phase 1 MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.relu(self.bn(self.fc(x))))


class TransportClassifier(nn.Module):
    """Phase 1 MLP head.

    Mirrors ``TransportClassifier`` of the released pipeline:
    ``input_dim -> 512 -> 2 x residual 512 -> 128 -> 1`` (logit).
    """

    def __init__(self, input_dim: int = 1283):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.res_block1 = ResidualBlock(512)
        self.res_block2 = ResidualBlock(512)
        self.output_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.res_block2(self.res_block1(self.input_layer(x))))


class Phase1Classifier:
    """Phase 1 binary transport classifier.

    Parameters
    ----------
    esm2_path : str or Path
        ESM-2 checkpoint: a fair-esm ``.pt`` file or a HuggingFace directory.
    checkpoint : str or Path
        Phase 1 **MLP** weights (a ``torch`` state dict, ``.pth``).
    xgb_model : str or Path, optional
        Phase 1 **XGBoost** model (``.json``).  The published ensemble needs it.
    backbone : str
        ``"3B"`` (t36 3B) or ``"650M"`` (t33 650M); selects the ensemble weights.
    device : str
        Torch device.
    calibrator : str or Path, optional
        Joblib-serialised isotonic calibrator applied to the raw ensemble
        probability.  The released v3 script does not apply one at inference
        time, so this is opt-in.
    ensemble_weights : tuple of float, optional
        ``(mlp_weight, xgb_weight)`` overriding the per-backbone defaults.
        The released script exposes the same knob as ``--phase1_weights``
        (default ``0.8,0.2`` for every backbone); setting it explicitly is
        what makes a run comparable with a specific published configuration.
    max_len, batch_size : int
        Encoding controls.
    """

    def __init__(
        self,
        esm2_path: str | Path,
        checkpoint: str | Path,
        xgb_model: Optional[str | Path] = None,
        backbone: str = "3B",
        device: str = "cuda",
        calibrator: Optional[str | Path] = None,
        ensemble_weights: Optional[Tuple[float, float]] = None,
        max_len: int = 800,
        s4pred_cache_dir: Optional[str | Path] = None,
        batch_size: int = 8,
    ):
        if backbone not in ENSEMBLE_WEIGHTS:
            raise ValueError(
                f"Unknown backbone {backbone!r}; expected one of {sorted(ENSEMBLE_WEIGHTS)}"
            )
        self.esm2_path = Path(esm2_path)
        self.checkpoint = Path(checkpoint)
        self.xgb_model = Path(xgb_model) if xgb_model else None
        self.backbone = backbone
        self.device = device
        self.calibrator_path = Path(calibrator) if calibrator else None
        self.max_len = max_len
        self.s4pred_cache_dir = Path(s4pred_cache_dir) if s4pred_cache_dir else None
        self.batch_size = batch_size

        if ensemble_weights is None:
            self.w_mlp, self.w_xgb = ENSEMBLE_WEIGHTS[backbone]
        else:
            self.w_mlp, self.w_xgb = float(ensemble_weights[0]), float(ensemble_weights[1])
            logger.info(
                f"Phase 1 ensemble weights overridden to mlp={self.w_mlp}, xgb={self.w_xgb}"
            )

        self._backbone: Optional[ESM2Backbone] = None
        self._mlp: Optional[TransportClassifier] = None
        self._xgb = None
        self._calibrator = None

    # ------------------------------------------------------------------ load
    @property
    def backbone_model(self) -> ESM2Backbone:
        """The ESM-2 backbone (loaded on first access)."""
        if self._backbone is None:
            self._backbone = ESM2Backbone(
                path=self.esm2_path,
                device=self.device,
                max_len=self.max_len,
                batch_size=self.batch_size,
            ).load()
        return self._backbone

    def _load_assets(self) -> None:
        if self._mlp is not None:
            return
        import xgboost as xgb

        backbone = self.backbone_model
        input_dim = backbone.embed_dim + 3
        logger.info(f"Loading Phase 1 MLP from {self.checkpoint} (input_dim={input_dim})")
        mlp = TransportClassifier(input_dim=input_dim).to(self.device)
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        mlp.load_state_dict(state)
        mlp.eval()
        self._mlp = mlp

        if self.xgb_model is not None:
            logger.info(f"Loading Phase 1 XGBoost from {self.xgb_model}")
            booster = xgb.Booster()
            booster.load_model(str(self.xgb_model))
            self._xgb = booster
        else:
            logger.warning(
                "No XGBoost model supplied; Phase 1 will use the MLP probability alone. "
                "This is NOT the published ensemble."
            )

        if self.calibrator_path is not None:
            import joblib

            logger.info(f"Loading isotonic calibrator from {self.calibrator_path}")
            self._calibrator = joblib.load(self.calibrator_path)

    # --------------------------------------------------------------- scoring
    def score(self, features: np.ndarray) -> np.ndarray:
        """Score a feature matrix and return the ensemble probabilities."""
        self._load_assets()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(features, dtype=np.float32)).float().to(self.device)
            mlp_proba = torch.sigmoid(self._mlp(x)).cpu().numpy().flatten()
        if self._xgb is not None:
            import xgboost as xgb

            xgb_proba = self._xgb.predict(xgb.DMatrix(features))
        else:
            xgb_proba = np.zeros_like(mlp_proba)
        raw = self.w_mlp * mlp_proba + self.w_xgb * xgb_proba
        if self._calibrator is not None:
            return np.asarray(self._calibrator.predict(raw), dtype=float)
        return np.asarray(raw, dtype=float)

    # -------------------------------------------------------------- encoding
    def iter_batches(
        self, fasta: str | Path
    ) -> Iterator[Tuple[List[str], np.ndarray, torch.Tensor, torch.Tensor]]:
        """Encode the FASTA in batches.

        Yields
        ------
        (names, features, token_emb, input_ids)
            ``names`` are the accessions of the batch, ``features`` is the
            Phase 1 feature matrix of the batch and ``token_emb`` /
            ``input_ids`` are the token-level tensors of the same rows, so a
            Phase 2 stage on the same backbone can reuse them.
        """
        backbone = self.backbone_model
        all_names, sequences = read_fasta_ordered(fasta)
        if not all_names:
            raise ValueError(f"No sequences found in {fasta}")
        keep = [n for n in all_names if is_valid_sequence(sequences[n])]
        dropped = len(all_names) - len(keep)
        if dropped:
            logger.warning(
                f"Dropped {dropped}/{len(all_names)} sequences failing the length / alphabet filter"
            )
        if not keep:
            raise ValueError(f"All sequences in {fasta} failed the validity filter")

        s4_all = extract_s4pred_features(
            sequences, names=keep, cache_dir=self.s4pred_cache_dir
        )
        s4_map = {name: s4_all[i] for i, name in enumerate(keep)}

        for start in range(0, len(keep), self.batch_size):
            batch_names = keep[start : start + self.batch_size]
            _, token_emb, input_ids = backbone.encode_all(
                [sequences[n] for n in batch_names], batch_size=len(batch_names)
            )
            token_emb = token_emb.to(backbone.device)
            input_ids = input_ids.to(backbone.device)
            pooled = backbone.phase1_mean_pool(
                token_emb, input_ids, backbone.pad_id, backbone.eos_id
            )
            esm = pooled.detach().cpu().numpy().astype(np.float32)
            s4 = np.stack([s4_map[n] for n in batch_names]).astype(np.float32)
            features = np.concatenate([esm, s4], axis=1)
            yield batch_names, features, token_emb, input_ids

    # --------------------------------------------------------------- predict
    def predict(self, fasta: str | Path) -> pd.DataFrame:
        """Run Phase 1 on a FASTA file.

        Parameters
        ----------
        fasta : str or Path
            Input FASTA.

        Returns
        -------
        pd.DataFrame
            Columns: ``query`` (accession), ``transport_prob`` (float in [0, 1]),
            ``seq_length`` (residue count of the input sequence).
        """
        self._load_assets()
        all_names, sequences = read_fasta_ordered(fasta)
        rows = []
        for batch_names, features, _, _ in self.iter_batches(fasta):
            probs = self.score(features)
            for name, p in zip(batch_names, probs):
                rows.append(
                    {
                        "query": name,
                        "transport_prob": float(p),
                        "seq_length": len(sequences[name]),
                    }
                )
        return pd.DataFrame(rows, columns=["query", "transport_prob", "seq_length"])
