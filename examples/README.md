# Examples

Reproducible case studies for TPClass on three substrate-specific transporter mining tasks.

## Structure

```
examples/
├── hmo/                  # HMO (human milk oligosaccharides) transporter mining
│   ├── run_hmo.sh       # One-shot pipeline
│   ├── hmo_refs.fasta   # 25 reference proteins (10 known + 15 user-curated)
│   └── hmo_top_candidates.csv  # Top 50 candidates (example output)
│
├── squalene/             # Squalene / terpenoid transporter discovery in K. marxianus
│   ├── run_squalene.sh
│   ├── k_marxianus.fasta
│   └── squalene_candidates.csv
│
└── isobutanol/           # Isobutanol exporter mining (in progress)
    ├── run_isobutanol.sh
    └── isobutanol_refs.fasta
```

## Quick start

Each example has a `run_*.sh` script that runs the full pipeline end-to-end:

```bash
# HMO
cd examples/hmo
bash run_hmo.sh

# Squalene
cd examples/squalene
bash run_squalene.sh

# Isobutanol
cd examples/isobutanol
bash run_isobutanol.sh
```

## Prerequisites

- Models downloaded (see [models/README.md](../models/README.md))
- Reference data in place (see [data/README.md](../data/README.md))
- DIAMOND installed (for ESM-2 vs BLAST comparison)
- ~12 GB GPU memory for ESM-2 3B inference

## What to expect

- **HMO**: ~1,066 candidates from 575k UniProt proteins; 79.1% are BLAST-missed novel findings.
- **Squalene**: ~66 candidates; conserved yeast LAM family (9.B.198.2.x) identified in K. marxianus.
- **Isobutanol**: in progress; expected several hundred candidates from yeast DHA1 family.

See each example's README for details.
