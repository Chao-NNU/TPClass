# TPClass: A Two-Phase Predictor for Membrane Transporter Annotation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**TPClass** is an ESM-2-based two-phase predictor for membrane transporter identification and TC (Transporter Classification) family assignment. It scales to proteome-wide prediction on UniProt (~575k proteins) and supports substrate-specific transporter mining (HMO, squalene, isobutanol, ...) through curated reference sets.

| | |
|---|---|
| **Publication** | TPClass: A Two-Phase Predictor for Membrane Transporter Annotation |
| **Targets** | HMO, terpenoid (squalene), isobutanol, and arbitrary substrate-specific transporters |
| **Backbone** | ESM-2 t33 650M + t36 3B, s4pred secondary structure |
| **Head** | MLP + XGBoost (Phase 1), contrastive ESM-2 (Phase 2) |
| **License** | MIT |

---

## Why TPClass?

Existing transporter predictors fall into one of two camps:

- **Homology-based** (e.g. DIAMOND BLASTp against TCDB): accurate for known families, blind to remote homologs.
- **Generic ML** (e.g. TransFun, DeepMito): cross-species, but small class-imbalanced training sets and no substrate-specific customization.

TPClass combines the strengths of both:

- **ESM-2 protein language model** for cross-species generalisation (575k UniProt proteins, 8 benchmark species)
- **TCDB hierarchy** as the target taxonomy (~550 curated families)
- **Two-phase cascade** for precision-recall balance (3B Phase 1 recall + 650M Phase 2 family precision)
- **Three-layer evidence filter** (TC class + TCDB description + ESM-2 cosine similarity) for substrate-specific mining

Headline result on the 4,785-sequence TCDB held-out test set (3B / fresh mode): **0.8804 accuracy, 0.8836 F1, 0.7615 MCC**. On the 8-species cross-species benchmark: **cascade P = 0.910, R_f = 0.941, R_s = 0.860**.

---

## Installation

```bash
git clone https://github.com/Chao-NNU/TPClass.git
cd TPClass
pip install -e .            # or: pip install -e ".[esm]" for the fair-esm .pt checkpoints
```

Requirements:

- Python 3.10+
- PyTorch 2.0+ with CUDA 11.8+ (CPU works but is slow)
- ~12 GB GPU memory for ESM-2 3B inference
- ~15 GB disk for the downloaded models

Then fetch the model assets and the reference data:

```bash
bash models/download.sh
```

The ESM-2 backends accept two layouts and both are supported:

- a **fair-esm** `.pt` file (the layout the published checkpoints were trained with) -- install with `pip install -e ".[esm]"`;
- a **HuggingFace** directory (the default of `models/download.sh`) -- works for inference.

See [`docs/installation.md`](docs/installation.md) for details and troubleshooting.

---

## Quick start

### 1. Predict transporters in a proteome

```bash
tpclass predict \
    --fasta proteome.fasta \
    --out results/candidates.csv \
    --backbone 3B \
    --phase1_3b models/v3_retrained_3b/phase1/transport_classifier_best.pth \
    --phase1_xgb_3b models/v3_retrained_3b/phase1/transport_xgb_best.json \
    --phase2_3b models/v3_retrained_3b/phase2/best_phase2.pth \
    --esm2_3b models/esm2_3b
```

The Phase 2 reference index and its metadata default to `tcdb_embeddings.npy`
and `tcdb_index_metadata.csv` **next to the Phase 2 checkpoint** (one directory
above is also accepted) -- the head and the index have to come from the same
training run.

The Phase 1 ensemble weights default to the per-backbone values (3B
`0.8/0.2`, 650M `0.4/0.6`). Use `--phase1_weights "mlp,xgb"` to override them;
the released script exposes the same knob with the default `"0.8,0.2"` for
every backbone, so set it explicitly when matching a published configuration.

### 2. Apply the three-layer evidence filter (substrate-specific mining)

```bash
tpclass filter \
    --v3_csv results/candidates.csv \
    --ref_set refs/hmo_refs.fasta \
    --esm2 models/esm2_3b \
    --tc_classes 2.A.1,2.A.2,2.A.7,2.A.21,3.A.1,4.A.1,9.B \
    --keywords fucose,sialic,lactose,galactose,GlcNAc,oligosaccharide,HMO,sugar \
    --sim_min 0.7 \
    --out results/hmo_filtered.csv
```

### 3. Compare ESM-2 vs BLAST (BLAST-missed discovery)

```bash
tpclass compare-esm-blast \
    --candidates results/hmo_filtered.csv \
    --fasta proteome.fasta \
    --ref_fasta refs/tcdb.fasta \
    --out results/hmo_esm_vs_blast.tsv
```

For more, see the case studies:

- [`examples/hmo`](examples/hmo) — HMO transporter mining
- [`examples/squalene`](examples/squalene) — Squalene / terpenoid transporter discovery
- [`examples/isobutanol`](examples/isobutanol) — Isobutanol exporter mining

---

## Architecture

### Phase 1 — Binary transport classifier

- Input: protein sequence
- Features: ESM-2 mean-pooled embedding (1280-d for 650M, 2560-d for 3B) + 3 s4pred secondary-structure features (H/E/C fractions)
- Head: MLP (`512 → 2× residual 512 → 128 → 1`) + XGBoost ensemble
- Output: `transport_prob ∈ [0, 1]` (default threshold 0.50)
- Backbone choice: 3B for high recall, 650M for high family precision

### Phase 2 — TC family assignment (contrastive retrieval)

- Input: protein sequence (only those passing the Phase 1 threshold)
- Features: ESM-2 token embeddings + a contrastively trained projection head
- Retrieval: top-k = 10 nearest neighbours by cosine similarity to the TCDB reference index
- Output: TC family by majority vote, four-digit `tc_number` from the top-1 hit

### Cascade

- 3B Phase 1 (high recall) provides the **candidate set**; the 650M model never removes a candidate
- The TC family comes from the **650M** model when the 650M model also called that protein YES and produced a family
- Fallback: the 3B family when the 650M model did not call the protein YES, or called it YES without producing a family
- The retrieval evidence always comes from the **same** model that provided the family; the `family_source` column records which one

### Three-layer evidence filter (substrate-specific)

1. TCDB TC class whitelist (e.g. 2.A.1, 3.A.1, 4.A.1, 9.B for HMO)
2. TCDB description keyword match (e.g. "fucose", "sialic", "lactose", "HMO")
3. ESM-2 cosine similarity ≥ threshold to a substrate-specific reference set

The filter narrows the candidate set from tens of thousands to tens of top-ranked proteins for experimental validation.

See [`docs/architecture.md`](docs/architecture.md) for full technical details.

---

## Application case studies

| Substrate | Reference set | TC class whitelist | Result |
|---|---|---|---|
| **HMO** (human milk oligosaccharides) | 10 known HMO transporters + 15 user-curated (lacY, fucP, mglC, kpsM, nanX, SetA/B, YdeA, ...) | 2.A.1, 2.A.2, 2.A.7, 2.A.21, 3.A.1, 3.D.1, 4.A.1, 9.B | 1,066 candidates from 575k UniProt proteins; 79.1% are BLAST-missed |
| **Squalene / terpenoid** | 6 squalene-related proteins (HpnN, POPT, CoxX) | 2.A.6, 3.D.1, 3.D.10, 3.A.2, 1.C.59, 3.E.4, 9.B.198 | 66 candidates; conserved yeast LAM family identified in *K. marxianus* |
| **Isobutanol** | 18 yeast DHA1 / MFS / SMR proteins (AQR1, QDR1-3, FLR1, TPO1-4, SGE1, MdfA, EmrB, ...) | 2.A.1, 2.A.51, 3.A.1, 2.A.6 | In progress |

Reproducing these studies:

```bash
bash examples/hmo/run_hmo.sh
bash examples/squalene/run_squalene.sh
bash examples/isobutanol/run_isobutanol.sh
```

---

## Reproducibility

What is needed to reproduce the published numbers, and what this repository
provides:

| Item | Status |
|---|---|
| Phase 1 / Phase 2 architectures, pooling masks, ensemble weights | Implemented in `tpclass/core/` and identical to the released pipeline |
| Phase 1 / Phase 2 training code | `tpclass/train/` (MLP, XGBoost, isotonic calibration, SupCon contrastive head) |
| Dataset splitting | `tpclass split` -- `--strategy stratified` reproduces the released random stratified split; `--strategy cluster` keeps sequences with a 5-mer Jaccard similarity of at least 0.5 in the same subset.  This reduces k-mer-level leakage; it is not a guarantee of homology-independent evaluation, because k-mer similarity is not sequence identity (see `scripts/README.md`) |
| Pretrained weights, TCDB reference index, TCDB annotation table | Downloaded by `bash models/download.sh` (HuggingFace `Chao-NNU/TPClass`) |
| Raw training/evaluation scripts | **Shipped verbatim** in `scripts/` (35 Python scripts) plus the shell drivers in `scripts/runs/` that were actually used to execute the published runs. See `scripts/README.md` for the script-to-paper-element mapping |
| Result tables, per-species outputs, trained checkpoints | **Not shipped**; the numbers quoted above come from the original working copy, so the *inputs* must be re-generated before the scripts can be re-run |
| Checking that the package still matches the released pipeline | `tpclass compare-reports --a <release>/final_report.csv --b <package>.csv` -- row-by-row comparison with a numeric tolerance; exits non-zero on any difference |

Three protocol details are easy to get wrong and are implemented explicitly here:

- **Phase 2 pairing.** The contrastive head and the reference index must come
  from the same training run. The reference paths are therefore derived from
  the checkpoint directory, and a mismatch is reported loudly. The Phase 2
  checkpoint also carries a *partially fine-tuned* ESM-2 (the last six
  transformer layers plus the final layer norm were unfrozen during contrastive
  training); those weights are restored into the backbone, because the
  reference index was built with them. Phase 2 therefore uses its own ESM-2
  instance, so the restored weights cannot leak into the Phase 1 features.
- **Metric denominators.** In the cross-species evaluation, family precision
  is computed over the predictions that fall inside the ground-truth set; the
  recall denominators are the ground-truth set itself. The strict and fuzzy
  recalls are not independent: `strict = fuzzy × precision`.
- **Two BLAST acceptance criteria are in use, and they are not
  interchangeable.** The cross-species family benchmark
  (`scripts/eval_blast_family_precision.py`) uses e-value ≤ 1e-30,
  identity ≥ 40 % and query coverage ≥ 80 %. The HMO case study
  (`scripts/compare_esm_vs_blast.py`) accepts a hit on **identity alone
  (≥ 30 %)** — its `--coverage_low` argument is accepted but never applied.
  The published BLAST-missed percentage follows the identity-only criterion.
  `tpclass compare-esm-blast` implements the stricter identity + coverage +
  e-value rule, so re-running it will report a *higher* missed fraction than
  the published number; state the criterion whenever such a number is quoted.

A BLAST-missed call is a *lead*, not proof of a new substrate-transport
function: it only means the sequence had no reference hit passing the stated
identity, coverage and e-value criteria.

---

## Repository layout

```
TPClass/
├── README.md                    # This file
├── LICENSE                       # MIT
├── CITATION.cff                  # GitHub citation
├── CHANGELOG.md                  # Version history
├── CONTRIBUTING.md               # How to contribute
├── setup.py / pyproject.toml     # Packaging
├── requirements.txt              # Runtime dependencies
├── Makefile                      # Common tasks
│
├── tpclass/                      # Main Python package
│   ├── core/                     # ESM-2 backbone, Phase 1, Phase 2, reference handling
│   ├── filters/                  # TC class + description + similarity filters
│   ├── analysis/                 # ESM-2 vs BLAST comparison
│   ├── train/                    # Dataset splitting and model training
│   └── cli.py                    # Command-line interface
│
├── models/                       # Model assets (download required, see models/README.md)
│   ├── esm2_3b/                  # ESM-2 3B weights
│   ├── esm2_650m/                # ESM-2 650M weights
│   ├── v3_retrained_3b/          # 3B Phase 1 + Phase 2 + reference index
│   └── v3_retrained/             # 650M Phase 1 + Phase 2 + reference index
│
├── data/                         # Reference data (downloaded, see data/README.md)
│   ├── tcdb_annot.clean.tsv      # TCDB annotation table
│   └── refs/                     # Reference FASTA files per substrate
│
├── examples/                     # Case-study scripts and notes
│   ├── hmo/
│   ├── squalene/
│   └── isobutanol/
│
├── scripts/                      # The original release scripts (verbatim)
│   └── runs/                     # The shell drivers that were actually executed
│
├── tests/                        # Smoke and unit tests
│
└── docs/
    ├── architecture.md
    ├── installation.md
    └── usage.md
```

---

## Documentation

- [`docs/installation.md`](docs/installation.md) — Detailed install + GPU setup
- [`docs/architecture.md`](docs/architecture.md) — Phase 1, Phase 2, cascade, three-layer filter
- [`docs/usage.md`](docs/usage.md) — CLI + Python API

---

## Citation

If you use TPClass in your research, please cite:

```bibtex
@article{tpclass2026,
  title={TPClass: A Two-Phase Predictor for Membrane Transporter Annotation},
  author={[Authors]},
  journal={[Journal]},
  year={2026},
  doi={[DOI]}
}
```

A `CITATION.cff` file is also provided for GitHub's automatic citation feature.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Contributing

We welcome bug reports, feature requests, and pull requests. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines. Major changes should be discussed via an issue first.

For substrate-specific transporter mining on a new target (e.g. a new biofuel, a new metabolite), open an issue with:

- Substrate name + literature references for known transporters
- TC classes likely involved (TCDB hierarchy link)
- Description keywords (TCDB-curated preferred)
- Initial reference set (5-25 known accessions, or 0 if no literature exists)

---

## Acknowledgements

- ESM-2: Lin et al. 2022, *bioRxiv* ([ESM GitHub](https://github.com/facebookresearch/esm))
- TCDB: Saier et al. 2016, *Nucleic Acids Research* ([TCDB](https://tcdb.org/))
- DIAMOND: Buchfink et al. 2015, *Nature Methods*
- XGBoost: Chen & Guestrin 2016, *KDD*
- s4pred: Magna et al. 2014, *BMC Bioinformatics*

---

## Contact

For questions, open an issue or contact the corresponding author.
