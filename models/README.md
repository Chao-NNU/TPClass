# Models

TPClass needs pre-trained checkpoints and a Phase 2 reference index.  They are
too large for the git repository and are downloaded with
`bash models/download.sh`.

## Required assets

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| **ESM-2 t36 3B UR50D** | ~11 GB | Phase 1 backbone (high recall) | [facebook/esm2_t36_3B_UR50D](https://huggingface.co/facebook/esm2_t36_3B_UR50D) |
| **ESM-2 t33 650M UR50D** | ~2.5 GB | Phase 1 and Phase 2 backbone (high family precision) | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) |
| **TPClass 3B bundle** | ~1.2 GB | 3B Phase 1 (MLP + XGBoost), 3B Phase 2 head, 3B reference index | [Chao-NNU/TPClass](https://huggingface.co/Chao-NNU/TPClass) |
| **TPClass 650M bundle** | ~0.6 GB | 650M Phase 1 (MLP + XGBoost), 650M Phase 2 head, 650M reference index | [Chao-NNU/TPClass](https://huggingface.co/Chao-NNU/TPClass) |
| **TCDB annotation table** | ~5 MB | TC numbers and descriptions | [Chao-NNU/TPClass](https://huggingface.co/Chao-NNU/TPClass) |

**Total download size**: ~15 GB

## Download

### Option 1: run the script (recommended)

```bash
bash models/download.sh
```

It downloads the two ESM-2 backbones from the official Facebook repositories
and the TPClass bundles from HuggingFace `Chao-NNU/TPClass`.

### Option 2: manual download

```bash
huggingface-cli download facebook/esm2_t36_3B_UR50D --local-dir models/esm2_3b
huggingface-cli download facebook/esm2_t33_650M_UR50D --local-dir models/esm2_650m
huggingface-cli download Chao-NNU/TPClass --include "models/*" --local-dir .
huggingface-cli download Chao-NNU/TPClass --include "data/*"   --local-dir .
```

### Option 3: Python API

```python
from huggingface_hub import snapshot_download

snapshot_download("facebook/esm2_t36_3B_UR50D", local_dir="models/esm2_3b")
snapshot_download("facebook/esm2_t33_650M_UR50D", local_dir="models/esm2_650m")
snapshot_download("Chao-NNU/TPClass", local_dir=".", allow_patterns=["models/*", "data/*"])
```

## ESM-2 layout: `.pt` or HuggingFace directory

Both layouts are supported by the package:

- **fair-esm `.pt`** (`pip install -e ".[esm]"`).  This is the layout the
  published checkpoints were trained with, and it is required for **Phase 2
  training** (`tpclass train-phase2`).
- **HuggingFace directory** (the default of `download.sh`).  Sufficient for
  inference with the published weights.

## Directory layout after download

```
models/
  esm2_3b/                       # ESM-2 3B (HuggingFace directory)
    config.json
    tokenizer.json
    model.safetensors            (~11 GB)
  esm2_650m/                     # ESM-2 650M (HuggingFace directory)
    config.json
    tokenizer.json
    model.safetensors            (~2.5 GB)
  v3_retrained_3b/               # 3B bundle
    phase1/
      transport_classifier_best.pth    # Phase 1 MLP (torch state dict)
      transport_xgb_best.json          # Phase 1 XGBoost model
    phase2/
      best_phase2.pth                 # Phase 2 contrastive checkpoint
    tcdb_embeddings.npy               # Phase 2 reference index (n_ref x 512)
    tcdb_index_metadata.csv           # Reference metadata
  v3_retrained/                  # 650M bundle, same layout
    phase1/
      transport_classifier_best.pth
      transport_xgb_best.json
    phase2/
      best_phase2.pth
    tcdb_embeddings.npy
    tcdb_index_metadata.csv
data/
  tcdb_annot.clean.tsv
```

## The Phase 2 pairing rule

The contrastive head and the reference index must come from the **same
training run**, otherwise the query and the reference live in different
embedding spaces and the retrieval silently degrades.  The package therefore
derives the reference paths from the checkpoint directory:

```
models/v3_retrained_3b/phase2/best_phase2.pth   <- head
models/v3_retrained_3b/tcdb_embeddings.npy      <- index (same directory)
models/v3_retrained_3b/tcdb_index_metadata.csv  <- metadata (same directory)
```

Pass `--ref_emb_3b` / `--ref_emb_650m` / `--ref_metadata` only when you know
the index matches the checkpoint; a mismatch is reported as a warning.

## Reference index format

- `tcdb_embeddings.npy` -- float32 matrix of shape `(n_ref, 512)`, one
  L2-normalised contrastive embedding per reference protein.
- `tcdb_index_metadata.csv` -- one row per reference, in the same order as the
  embedding matrix.  Required column: `tc_number`.  Optional columns:
  `tc_family` (derived from `tc_number` when absent), an identifier column
  (`acc`, `accession`, `id`, ...) and a description column (`description`,
  `tc_description`, ...).

## Citing the models

If you use the TPClass pre-trained models, please cite:

- ESM-2: Lin et al. 2022, *bioRxiv* (DOI: 10.1101/2022.07.20.500902)
- TPClass: see [CITATION.cff](../CITATION.cff)
- TCDB: Saier et al. 2016, *Nucleic Acids Research*

## Issues?

If the download fails or the model files do not load, open an issue with:

- your OS and Python version;
- the exact error message;
- the output of `huggingface-cli --version`.
