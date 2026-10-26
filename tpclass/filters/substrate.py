"""Generic substrate-specific filter for TPClass candidates.

Filters the v3 final_report.csv by:
1. TCDB TC class whitelist (prefix match)
2. TCDB description keyword match
3. v3 confidence threshold (transport_prob and/or top1_similarity)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class SubstrateFilter:
    """Filter v3 candidates for a target substrate.

    Parameters
    ----------
    tc_classes : list of str
        Allowed TC class prefixes (e.g., ["2.A.1", "3.A.1", "9.B"]).
    keywords : list of str
        TCDB description keywords to match (e.g., ["fucose", "sialic", "HMO"]).
    min_transport_prob : float, optional
        Minimum v3 transport_prob (default 0.0).
    min_top1_similarity : float, optional
        Minimum v3 top1 ESM-2 similarity (default 0.0).
    min_max_ref_sim : float, optional
        Minimum max_ref_sim to reference set (default 0.0).
    """

    def __init__(
        self,
        tc_classes: list[str],
        keywords: list[str],
        min_transport_prob: float = 0.0,
        min_top1_similarity: float = 0.0,
        min_max_ref_sim: float = 0.0,
    ):
        self.tc_classes = tc_classes
        self.keywords = [k.lower() for k in keywords]
        self.min_transport_prob = min_transport_prob
        self.min_top1_similarity = min_top1_similarity
        self.min_max_ref_sim = min_max_ref_sim

    def filter(self, v3_csv: str | Path) -> pd.DataFrame:
        """Apply the filter to a v3 final_report.csv.

        Parameters
        ----------
        v3_csv : str or Path
            Path to v3 final_report.csv (output of predict_transport_family_v3.py).

        Returns
        -------
        pd.DataFrame
            Filtered candidates.
        """
        logger.info(f"Loading v3 results from {v3_csv}")
        df = pd.read_csv(v3_csv)

        logger.info(f"Before filter: {len(df)} candidates")

        # Layer 1: TC class whitelist
        if self.tc_classes:
            tc_mask = df["tc_number"].apply(self._tc_matches)
            df = df[tc_mask]
            logger.info(f"After TC class filter ({len(self.tc_classes)} prefixes): {len(df)}")

        # Layer 2: TCDB description keyword
        if self.keywords:
            desc_col = "tc_description_short" if "tc_description_short" in df.columns else "tc_description"
            kw_mask = df[desc_col].astype(str).str.lower().apply(
                lambda d: any(k in d for k in self.keywords)
            )
            df = df[kw_mask]
            logger.info(f"After keyword filter ({len(self.keywords)} keywords): {len(df)}")

        # Layer 3: confidence thresholds
        if self.min_transport_prob > 0 and "transport_prob" in df.columns:
            df = df[df["transport_prob"] >= self.min_transport_prob]
        if self.min_top1_similarity > 0 and "top1_similarity" in df.columns:
            df = df[df["top1_similarity"] >= self.min_top1_similarity]
        if self.min_max_ref_sim > 0 and "max_ref_sim" in df.columns:
            df = df[df["max_ref_sim"] >= self.min_max_ref_sim]
            logger.info(f"After max_ref_sim filter (≥{self.min_max_ref_sim}): {len(df)}")

        logger.info(f"After all filters: {len(df)} candidates")
        return df

    def _tc_matches(self, tc: str) -> bool:
        """Check if a TC number starts with any of the allowed prefixes."""
        if not isinstance(tc, str) or not tc:
            return False
        for prefix in self.tc_classes:
            if tc == prefix or tc.startswith(prefix + "."):
                return True
        return False
