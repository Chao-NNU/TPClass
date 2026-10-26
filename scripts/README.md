# Scripts

Everything in this directory is a **verbatim copy of the scripts that produced
the published results**.  The working copy they were taken from is the
`scripts/` directory of the original project tree; every file here is
byte-for-byte identical to its counterpart there, so the numbers in the paper
can be re-derived exactly.

For new work prefer the `tpclass` package: it is the tested, importable
re-implementation of the same pipeline (`tpclass predict`, `tpclass filter`,
`tpclass compare-esm-blast`, `tpclass compute-refs`, `tpclass split`,
`tpclass train-phase1`, `tpclass train-phase2`).

To check that the package still reproduces the released pipeline, run both on
the same input and compare the two tables:

```bash
tpclass compare-reports --a <release_script>/final_report.csv --b <package>.csv
```

## Which script produces which published number

### Inference (the release pipeline)

| Script | Description |
|--------|-------------|
| `predict_transport_family_v3.py` | The release pipeline: Phase 1 + Phase 2 + plotting.  This is the script that produced the published tables, figures and case-study candidate lists. |
| `split_fasta.py` | Split a large FASTA into fixed-size chunks (used for the Swiss-Prot run). |
| `merge_final_reports.py` | Merge the per-chunk `final_report.csv` / `phase1_report.csv` / `phase2_report.csv` produced by parallel chunks. |
| `compute_ref_embeddings.py` | Compute ESM-2 reference embeddings for a custom FASTA (HMO / squalene / custom). |
| `extract_features_for_retrain.py` | Extract the Phase 1 feature matrices used for training. |

### Filtering and substrate mining

| Script | Description |
|--------|-------------|
| `filter_tcdb_esm_v2.py` | Three-layer evidence filter (TC class + TCDB description + ESM-2 similarity). |
| `filter_for_substrate.py` | Generic substrate-specific filter. |
| `filter_hmo_candidates.py` | HMO candidate list from the Swiss-Prot run (1,066 candidates in the published case study). |
| `filter_squalene.py` | Squalene / terpene / hopene candidate list (66 candidates in the published case study). |
| `parse_blast_to_tcdb.py` | Parse DIAMOND output and emit a v3-compatible `v3_vs_blast.csv` comparison table. |

### Evaluation

| Script | Description | Published element |
|--------|-------------|-------------------|
| `eval_ablation_direct.py` | Backbone x extraction-mode ablation on the held-out test set. | Table 1 |
| `eval_family_precision.py` | Family-level precision / fuzzy recall / strict recall on the cross-species benchmark. | Tables 2 and 4 |
| `eval_hybrid_v3.py` | Cascade / union / intersection strategies. | Table 3 |
| `eval_blast_family_precision.py` | DIAMOND BLASTp vs v3 on the same 8-species narrow GT. | BLAST family baseline |
| `evaluate_heldout.py` | Score the five Phase-1 ablation variants against the balanced held-out set. | Ablation table |
| `ablation_runner.py` | Per-variant metrics (protein / 4-digit / 3-digit level) for one species. | Ablation |
| `ablation_cross_species.py` | Aggregate the per-species ablation results. | Cross-species ablation |

### Statistics

| Script | Description | Published element |
|--------|-------------|-------------------|
| `dump_preds.py` | Dump per-protein Phase-1 predictions for the balanced held-out set. | Input to the paired tests |
| `paired_test_heldout.py` | McNemar exact test, paired bootstrap (10,000 resamples) and positive-set Jaccard for the four Table-1 configurations. | Paired significance tests |
| `aggregate_12species_precision.py` | Concatenate the per-species `precision.csv` files into one table. | Cross-species precision table |
| `summarize_3b_12species.py` | Quick cross-species summary from all 12 species' `final_report.csv`. | 3B cross-species summary |
| `cross_species_summary.py` | Per-species summary, head-to-head improvement, calibration overlay. | Cross-species figure |

### Case studies

| Script | Description |
|--------|-------------|
| `case_study_extract.py` | Extract the case-study numbers (top-10 YES hits, known-protein recalls) from a v3 run. |

### Figures

| Script | Description |
|--------|-------------|
| `plot_table1_ablation.py` | Three-panel Accuracy / F1 / MCC bar chart for Table 1. |
| `make_figure6_esm_vs_blast.py` | ESM-2 cosine similarity vs DIAMOND identity scatter for the HMO candidates. |
| `plot_distributions.py` | TC class / sub-class / family distributions and similarity histograms. |
| `plot_from_final_report.py` | Generic plotting from a `final_report.csv`. |

### Training

| Script | Description |
|--------|-------------|
| `train_mlp.py` | Train the Phase 1 MLP. |
| `train_xgb.py` | Train the Phase 1 XGBoost model. |
| `train_phase2_contrastive.py` | Train the Phase 2 contrastive head (SupCon). |
| `split_tcdb.py` | Split the TCDB FASTA into train/val/test. |

### Orchestration

`runs/` holds the shell drivers that were actually used to execute the
published runs.  They encode the real order of operations, the GPU
dispatch and the environment variables each stage needs.

| Script | Description |
|--------|-------------|
| `runs/run_ablation_train_val.sh` | Full retrain chain: split -> feature extraction -> MLP -> XGBoost -> 5 ablation variants -> held-out evaluation. |
| `runs/run_ablation.sh` | 4 species x 5 ablation variants, 2 GPUs in parallel. |
| `runs/run_5_variants_parallel.sh` | The 5 Phase-1 ablation variants on the balanced held-out set. |
| `runs/run_3b_12species.sh` | 3B inference on 12 species proteomes, 8 GPUs in parallel, reusing the cached s4pred results. |
| `runs/run_tclass_hmo_pipeline.sh` | Swiss-Prot: split -> dual-GPU inference -> merge -> HMO filter. |
| `runs/run_tclass_hmo_one_shot.sh` | Same pipeline, restarting the two chunks and the filter in one command. |
| `runs/select_idle_gpus.sh` | Pick the N least-loaded GPUs and export `CUDA_VISIBLE_DEVICES`. |

## Usage

Every script has its own `--help`:

```bash
python scripts/predict_transport_family_v3.py --help
```

## Paths and environment variables

The scripts were written against the original working copy and default to
absolute paths under `/data2/chao/transport`.  On that machine the defaults
are correct and the scripts run unchanged.

The release pipeline is fully environment-driven:

| Variable | Meaning |
|---|---|
| `TRANSPORT_ROOT` | Project root (default `/data2/chao/transport`) |
| `ESM2_PATH` | Phase 1 ESM-2 checkpoint (`.pt`) |
| `PHASE2_ESM2_PATH` | Phase 2 ESM-2 checkpoint (`.pt`, usually 650M) |
| `MLP_ASSET` / `XGB_ASSET` | Phase 1 MLP / XGBoost assets |
| `PHASE2_CKPT` | Phase 2 contrastive checkpoint |
| `REF_EMB` / `REF_META` | Phase 2 reference index and metadata (auto-paired with `PHASE2_CKPT` when unset) |

The substrate filters are environment-driven as well:

| Variable | Used by |
|---|---|
| `HMO_FASTA_PATHS`, `HMO_INPUT_CSV`, `HMO_OUTPUT_DIR` | `filter_hmo_candidates.py` |
| `SQ_FASTA_PATHS`, `SQ_INPUT_CSV`, `SQ_OUTPUT_DIR` | `filter_squalene.py` |

The remaining scripts take their paths as command-line arguments; run them
with `--help` to see the defaults.

## Notes

- The Phase 2 reference index and the Phase 2 checkpoint **must** come from the
  same training run; that is why the script derives `REF_EMB` / `REF_META`
  from the checkpoint directory and warns when they do not match.

- `split_tcdb.py` performs a random split stratified by TC family, which
  places homologues in different subsets.  `tpclass split --strategy cluster`
  reduces that leakage by keeping sequences with a 5-mer Jaccard similarity of
  at least 0.5 in the same subset.  Note that k-mer similarity is *not* the
  same as sequence identity: two proteins that are 80 % identical over their
  full length can still have a 5-mer Jaccard far below 0.5 when the
  substitutions are scattered, so this is a k-mer-level leakage reduction and
  not a guarantee of homology-independent evaluation.  A strict check should
  cluster by alignment identity and coverage and verify the nearest
  cross-split neighbour.

- **Two different BLAST acceptance criteria are in use, and they are not
  interchangeable.**  `eval_blast_family_precision.py` uses the cross-species
  family benchmark protocol (e-value <= 1e-30, identity >= 40 %,
  query coverage >= 80 %).  `compare_esm_vs_blast.py` - used for the HMO case
  study - accepts a hit on **identity alone (>= 30 %)**, and its
  `--coverage_low` argument is accepted but never applied.  Quote the criterion
  explicitly whenever a BLAST-missed percentage is reported.

- `predict_transport_family_v3.py` hard-codes `repr_layers=[33]` for the Phase 2
  head, so the Phase 2 backbone must be the 33-layer 650M model.  The
  `tpclass` package uses the last layer instead, which is the behaviour of
  `train_phase2_contrastive.py`; the two agree for 650M and differ for 3B.

- When comparing the release script against the `tpclass` package, set
  `PHASE2_ESM2_PATH` to a *different spelling of the same file* (for example
  `./models/esm2_t33_650M_UR50D.pt` next to an absolute
  `ESM2_PATH`).  The script compares the two strings and reuses one ESM-2
  object when they are identical, which would let the Phase 2 fine-tuned
  weights overwrite the Phase 1 backbone.

## Implementation details worth knowing before re-running anything

These are properties of the released code, verified against the sources in
this directory.  They are documented here because they change what the
published numbers mean; the scripts themselves are shipped unchanged.

### Phase 2 reads a fixed layer, but Phase 2 was trained on the last layer

`train_phase2_contrastive.py` encodes with `repr_layers=[esm.num_layers]`
(last layer: 33 for 650M, 36 for 3B), and the TCDB reference index was built
with that encoder.  `predict_transport_family_v3.py` encodes queries with
`repr_layers=[33]` regardless of the backbone.  For 650M the two agree; for
the 36-layer 3B model the query is read from layer 33 while the reference
index was built from layer 36, so the projection head sees an input
distribution it was not trained on.  The `tpclass` package uses the last
layer and therefore does not have this mismatch.

### A single-backbone run can reuse one ESM-2 object for both stages

`predict_transport_family_v3.py` loads a second ESM-2 only when the strings
`PHASE2_ESM2_PATH` and `ESM2_PATH` differ; otherwise Phase 2 falls back to the
Phase 1 backbone.  Loading the Phase 2 checkpoint then writes its partially
fine-tuned `esm.*` weights into that shared backbone, so the Phase 1 features
extracted afterwards come from a backbone that is no longer the pretrained one
Phase 1 was trained with.  In the cascade configuration the two paths differ by
construction, so the shared-object case only arises in single-backbone runs -
which is where the cached-versus-fresh difference of the held-out benchmark
should be checked first.

### "Fuzzy recall" is computed under two different rules

`eval_family_precision.py` (the family-precision tables) counts a protein
towards fuzzy recall only when it was predicted YES **and** received a 4- or
5-digit TC assignment **and** is in the ground truth; its precision uses the
same denominator.  `eval_hybrid_v3.py` (the hybrid-strategy table) counts
every predicted-YES protein inside the ground truth, whether or not it
received a family, and divides the precision by the same count.  Both are
internally consistent (`strict = fuzzy x precision`), but the two "fuzzy
recall" values are not the same quantity.  `tpclass family-metrics` computes
both rules from one implementation and reports them side by side.

### The hybrid table covers six species

`eval_hybrid_v3.py` lists six species in `SPECIES`.  The extended
eight-species comparison needs the two remaining species added to that list
(or the per-species results taken from the eight-species runs).

### The paired test in `paired_test_heldout.py` is not the accuracy test

Its `mcnemar_exact` counts the pairs where the two models predict *different
labels* and applies an exact binomial test to them.  That measures whether the
two models predict a different number of transporters; it does not measure
whether one of them is more accurate.  A model that is right wherever the other
is wrong produces the same label disagreement as the reverse, so the test can
return `p = 1` for a model that is perfect and another that is entirely wrong.
The accuracy version of McNemar's test counts the proteins where one model is
correct and the other is not; it is implemented in
`tpclass.analysis.paired_tests` and exposed as `tpclass paired-test`.

### The family evaluation scores the top-1 retrieval, not the majority vote

Phase 2 aggregates the TC **family** of the retrieved references by majority
vote, while the 4-digit sub-subfamily comes from the top-1 hit
(`tc_number`).  Both evaluation scripts read `tc_number` and truncate it to
four components, so the reported 4-digit precision and recall are top-1
retrieval metrics.  The voted family is exposed separately as `tc_family`.

