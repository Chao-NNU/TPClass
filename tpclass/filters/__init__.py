"""Substrate-specific filtering for TPClass candidates.

The three-layer evidence filter:
1. TCDB TC class whitelist (e.g., 2.A.1, 3.A.1, 4.A.1, 9.B for HMO)
2. TCDB description keyword match (e.g., "fucose", "sialic", "HMO")
3. ESM-2 cosine similarity ≥ threshold to substrate-specific reference set
"""
from tpclass.filters.substrate import SubstrateFilter
from tpclass.filters.three_layer import ThreeLayerFilter

__all__ = ["SubstrateFilter", "ThreeLayerFilter"]
