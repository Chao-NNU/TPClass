# Changelog

All notable changes to TPClass will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **McNemar's test.** `scripts/paired_test_heldout.py` counts the pairs where
  the two models predict *different labels* and applies an exact binomial test
  to them.  That measures whether the two models predict a different number
  of transporters, not whether one of them is more accurate: a model that is
  right wherever the other is wrong produces the same label disagreement as the
  reverse, so the test returns `p = 1` for a perfect model against a model
  that is entirely wrong.  The accuracy-based version of McNemar's test --
  which counts the proteins where one model is correct and the other is not --
  is implemented in `tpclass.analysis.paired_tests` and exposed as
  `tpclass paired-test`.  The released script is shipped unchanged, so the
  paired statistics reported in the paper have to be regenerated with the
  corrected test.
- **Two different "fuzzy recall" definitions.** `eval_family_precision.py`
  counts a protein towards fuzzy recall only when it was predicted YES *and*
  received a TC family *and* is in the ground truth; `eval_hybrid_v3.py`
  counts every predicted-YES protein inside the ground truth, and uses the
  same count as the precision denominator.  Both are internally consistent
  (`strict = fuzzy x precision`), but the two fuzzy-recall values are not the
  same quantity, so the family-precision tables and the hybrid table are not
  directly comparable.  `tpclass family-metrics` computes both rules from a
  single implementation and reports them side by side.

### Added

- The scripts that produced the published numbers are now shipped verbatim in
  `scripts/` (35 Python scripts), together with the shell drivers in
  `scripts/runs/` that were actually executed (`run_ablation_train_val.sh`,
  `run_ablation.sh`, `run_5_variants_parallel.sh`, `run_3b_12species.sh`,
  `run_tclass_hmo_pipeline.sh`, `run_tclass_hmo_one_shot.sh`,
  `select_idle_gpus.sh`). Every file is byte-for-byte identical to the
  original working copy.
- `scripts/README.md` now maps each script to the paper element it produces
  (Tables 1-4, the paired significance tests, the cross-species ablation, the
  HMO and squalene case studies, the figures), lists the environment variables
  each script honours, and records the two protocol details that are easy to
  get wrong: the Phase 2 transformer layer and the two different BLAST
  acceptance criteria.
- **Output table format.** The package wrote empty cells where the released
  script writes `0.0` / `0` for a protein without a TC family, left three
  columns unrounded and rounded none of them, and it did not have the
  TCDB annotation table at all -- so the TC description came from the
  reference metadata (keyed by accession) instead of the annotation table
  (keyed by TC number).  A row-by-row comparison against the published tables
  would have reported a difference on every non-transporter row and on every
  TC description.  The package now writes the same cell values as the released
  script and accepts `--tcdb_annot`.
- **Phase 1 threshold rule.** The released script compares the ensemble
  probability with a strict `>`, so a protein whose probability is exactly the
  threshold is not a transporter; the package used `>=` both for the
  `is_transport` column and for the Phase 2 candidate set.
- **s4pred source.** The released script runs s4pred as an external program
  and caches the result as one `.horiz` file per sequence; the package
  recomputed the secondary structure with the `s4pred` package, which can
  differ in the last decimals and shift `transport_prob`.  The package now
  reads the same `.horiz` cache (`--s4pred_cache`), so both pipelines consume
  identical secondary-structure features.
- `tpclass compare-reports`, a row-by-row comparison of two prediction tables.
  It aligns the rows on the query column, compares numeric columns with a
  tolerance and text columns exactly, and exits with status 1 when the tables
  disagree.  This is the tool used to verify that the package reproduces
  `scripts/predict_transport_family_v3.py` and the published output tables.

### Fixed

Core prediction path -- these changed the numbers the pipeline produces:

- **Phase 1 MLP head structure.** `TransportClassifier.output_layer` had an
  extra `Dropout` and was missing the `ReLU`, so the published Phase 1
  checkpoints could not be loaded (`load_state_dict` key mismatch). The layer
  is now `Linear(512, 128) -> BatchNorm1d -> ReLU -> Linear(128, 1)`, exactly
  as in the released pipeline.
- **Phase 2 fine-tuned ESM-2 weights.** Only the `proj.*` tensors of the Phase 2
  checkpoint were being restored; the partially fine-tuned `esm.*` tensors were
  dropped. The TCDB reference index was built with that fine-tuned encoder, so
  queries and references were being compared in two different embedding
  spaces. The `esm.*` weights are now restored into the backbone.
- **Phase 2 transformer layer.** The contrastive encoder now reads the *last*
  transformer layer, which is what `train_phase2_contrastive.py` used during
  training. The released inference script hard-codes `repr_layers=[33]`, which
  coincides with the last layer for the 33-layer 650M but selects layer 33 of
  the 36-layer 3B.
- **Phase 2 does not share the Phase 1 backbone.** Restoring the fine-tuned
  ESM-2 weights into a shared backbone also changed the Phase 1 features,
  which were trained on the pretrained ESM-2. Phase 2 now uses its own ESM-2
  instance.
- **Phase 1 feature matrix.** The embedding dictionary was concatenated
  directly; the matrix is now built in FASTA order, so the ESM-2 block, the
  s4pred block and the output table stay aligned.
- **`extract_query_embeddings` was unimplemented** (`NotImplementedError`), so
  Phase 2 could not assign a family. It now computes the contrastive
  embedding, and the Phase 2 query path is wired end to end.
- **Cascade candidate set.** The cascade kept only the proteins both models
  called YES. The candidate set is now the 3B YES set, the 650M family wins
  when the 650M model also said YES and produced a family, and the 3B family
  is the fallback.
- **Cascade evidence.** The family label and its supporting evidence (similarity,
  votes, best reference) can no longer come from different models; the
  evidence is taken from the model that supplied the family, and `family_source`
  records which one.
- **Phase 2 reference index format.** The index is read from
  `tcdb_embeddings.npy` + `tcdb_index_metadata.csv` (the released layout), the
  family is voted on the 3-digit `tc_family`, the 4-digit `tc_number` comes
  from the top-1 hit, and `top1_sim_gap` / `top_family_purity` are reported.
- **Phase 2 reference index discovery.** The index is looked up next to the
  checkpoint and one directory above it, so both the released layout and the
  `models/download.sh` bundle layout work without extra flags.
- **BLAST query coverage.** The coverage denominator is the length of the full
  query instead of the span of the alignment, `coverage_low` is actually
  applied, bare accessions are parsed correctly, and NaN values are handled.
- **Residue alphabet.** `X` counts as a standard residue, as in the released
  pipeline, so proteins that the release script keeps are not dropped.
- **Phase 1 ensemble weights are configurable.** `tpclass predict
  --phase1_weights "mlp,xgb"` overrides the per-backbone defaults, mirroring
  the released script's `--phase1_weights` (default `"0.8,0.2"` for every
  backbone). Needed to match a specific published configuration.
- **`compute-refs` called a script that did not exist**; it now computes the
  reference embeddings itself and accepts `--ref_set` / `--esm2` / `--sim_min`
  on `filter`, as the documentation promised.

### Added

- `tpclass/core/esm2.py`: one ESM-2 wrapper for the fair-esm `.pt` and the
  HuggingFace layouts, with the Phase 1 and Phase 2 pooling masks.
- `tpclass/train/`: Phase 1 MLP / XGBoost / isotonic training, Phase 2 SupCon
  contrastive training, TCDB dataset construction, and a homology-aware
  cluster split (`--strategy cluster`) next to the released stratified split.
- `tpclass/core/refs.py`: reference embedding computation and `.npz` handling
  for both the `accessions`/`embeddings` and the `acc`/`embedding` key spellings.
- CLI sub-commands `split`, `train-phase1` and `train-phase2`.
- The original release scripts are shipped under `scripts/`, and the README
  states explicitly that the raw result tables are **not** part of this
  repository, so the published numbers cannot be re-derived bit-for-bit here.
- Tests for the cascade semantics, the feature order, the BLAST coverage
  computation and the dataset splitting.

### Changed

- Project rename: TClass (v3) -> TPClass. Code, models, and outputs are unchanged; only the public-facing method name is updated.
- Paper: TClass -> TPClass (manuscript and supplementary).
- `requirements.txt` no longer lists `python>=3.10` as a pip dependency,
  `pyproject.toml` no longer declares the non-existent `tpclass.utils`
  sub-package, and the model download URLs point at `Chao-NNU/TPClass` instead
  of the `your-lab` placeholder.
- README, `docs/` and `models/README.md` were rewritten so that every path,
  CLI flag and architecture description matches the code.

## [1.0.0] - 2026-10-19

### Added
- Initial public release of TPClass (formerly TClass v3).
- Two-phase architecture: ESM-2 + s4pred → Phase 1 binary classifier → Phase 2 contrastive TC family assignment.
- 3B (high recall) and 650M (high family precision) backbones.
- Cascade strategy (3B Phase 1 + 650M Phase 2) for genome-wide prediction.
- Three-layer evidence filter for substrate-specific transporter mining (TC class + TCDB description keyword + ESM-2 cosine similarity to custom reference set).
- Three case studies:
  - HMO transporter mining (1,066 candidates from 575k UniProt; 79.1% BLAST-missed)
  - Squalene / terpenoid transporter discovery in K. marxianus (66 candidates; conserved yeast LAM family)
  - Isobutanol exporter mining (18-reference set, in progress)
- 5+5 wet-lab validation design: 5 literature positive controls + 5 BLAST-missed novel candidates.
- Pre-trained model checkpoints (3B Phase 1, 3B Phase 2, 650M Phase 1, 650M Phase 2).
- Pre-computed reference embeddings for 12 reference proteomes.
- ESM-2 vs BLAST comparison pipeline (`compare_esm_vs_blast.py`).
- Eight benchmark species cross-species evaluation (E. coli K-12/BL21, B. subtilis, L. lactis, C. glutamicum, A. niger, S. cerevisiae, H. sapiens).
- TCDB held-out validation set (n=4,785; balanced 1:1).

### Tested
- Phase 1 binary classifier: Acc 0.8804, F1 0.8836, MCC 0.7615 (3B / fresh mode).
- Cascade cross-species: P=0.910, R_f=0.941, R_s=0.860 (8-species mean).

### Notes
- Code is the same as the internal TClass v3 release; only the public name has changed.
- Pre-trained model checkpoints are large (~10 GB ESM-2 3B) and are not committed to git. Use `models/download.sh` or HuggingFace mirror.

## [0.x.x] - internal TClass v3 development (pre-release)

- TClass v1 (homology-based, deprecated): see `legacy/v1/`
- TClass v2 (single ESM-2 650M, deprecated): see `legacy/v2/`
- TClass v3 (current, renamed to TPClass): internal development
