# Legacy code

TPClass is the third generation of this method and the only one maintained
here.

| Version | Method | Status |
|---|---|---|
| v1 | Homology-only (TCDB BLAST) | Deprecated, **not shipped** |
| v2 | Single ESM-2 650M, Phase 1 only | Deprecated, **not shipped** |
| v3 | Two-phase (binary + contrastive TC family) | Current -- this repository |

The v1 and v2 sources are not part of this repository, so the paths
`legacy/v1/` and `legacy/v2/` referenced in the changelog are historical
pointers rather than working directories.

If you need the older behaviour, it can be reproduced from the published
scripts: v1 is plain DIAMOND BLASTp against TCDB, v2 is the Phase 1 part of
`scripts/predict_transport_family_v3.py` run with a single backbone.
