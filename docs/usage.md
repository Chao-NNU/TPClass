# Usage

TPClass provides a command-line interface (`tpclass`) and a Python API for
transporter prediction.

## Quick start

```bash
# 1. Predict transporters in a proteome
tpclass predict \
    --fasta proteome.fasta \
    --out results/candidates.csv \
    --backbone 3B \
    --phase1_3b models/v3_retrained_3b/phase1/transport_classifier_best.pth \
    --phase1_xgb_3b models/v3_retrained_3b/phase1/transport_xgb_best.json \
    --phase2_3b models/v3_retrained_3b/phase2/best_phase2.pth \
    --esm2_3b models/esm2_3b

# 2. Apply the substrate-specific three-layer filter
tpclass filter \
    --v3_csv results/candidates.csv \
    --ref_set refs/hmo_refs.fasta \
    --esm2 models/esm2_3b \
    --tc_classes 2.A.1,3.A.1,4.A.1,9.B \
    --keywords fucose,sialic,lactose,HMO \
    --sim_min 0.7 \
    --out results/filtered.csv

# 3. Compare with BLAST
tpclass compare-esm-blast \
    --candidates results/filtered.csv \
    --fasta proteome.fasta \
    --ref_fasta refs/tcdb.fasta \
    --out results/esm_vs_blast.tsv
```

## Command-line interface

### `tpclass predict`

Run Phase 1 + Phase 2 on a FASTA file.

```bash
tpclass predict \
    --fasta <input.fasta> \
    --out <output.csv> \
    [--backbone {3B|650M|cascade}] \
    [--threshold <float, default 0.50>] \
    [--top_k <int, default 10>] \
    [--no_phase2] \
    [--phase1_3b <ckpt.pth>] [--phase1_650m <ckpt.pth>] \
    [--phase1_xgb_3b <model.json>] [--phase1_xgb_650m <model.json>] \
    [--phase2_3b <ckpt.pth>] [--phase2_650m <ckpt.pth>] \
    [--esm2_3b <path>] [--esm2_650m <path>] \
    [--esm2_phase2_3b <path>] [--esm2_phase2_650m <path>] \
    [--ref_emb_3b <index.npy>] [--ref_emb_650m <index.npy>] \
    [--ref_metadata <metadata.csv>] \
    [--tcdb_annot <tcdb_annot.tsv>] \
    [--s4pred_cache <ss_results_dir>] \
    [--phase1_weights "mlp,xgb"] \
    [--device {cuda|cpu}]
```

`--tcdb_annot` is the TCDB annotation table (`tc_number` -> description).  When
it is given, the `tc_description` column is looked up by TC number, exactly as
the released script does; without it the description of the matched reference
protein is used instead.  Supply it whenever the output is compared against
the published tables.

`--s4pred_cache` points at a directory of s4pred `.horiz` files (the
`ss_results/` directory the released script writes).  The cached secondary
structure is reused, so the three secondary-structure features are identical to
the released run -- without it, the package recomputes them with the `s4pred`
package, which can differ in the last decimals and shift `transport_prob`
slightly.

`--phase1_weights` overrides the Phase 1 ensemble weights for every Phase 1
model in the run.  Without it the per-backbone defaults apply (3B `0.8/0.2`,
650M `0.4/0.6`).  The released script exposes the same knob
(`--phase1_weights`, default `"0.8,0.2"` for every backbone), so set it
explicitly when a run has to match a specific published configuration.

The Phase 1 MLP (`.pth`) and the Phase 1 XGBoost model (`.json`) are separate
assets; supply both to reproduce the published ensemble.  The Phase 2
reference index defaults to `tcdb_embeddings.npy` and the metadata to
`tcdb_index_metadata.csv` **next to the Phase 2 checkpoint**, which is where
the release script expects them: the head and the index have to come from the
same training run.

`--esm2_phase2_*` selects the ESM-2 backbone of a Phase 2 head; it defaults to
the corresponding `--esm2_*` path.  The published Phase 2 head was trained on
the 650M backbone.

### `tpclass filter`

Substrate-specific filter: TC class whitelist + description keywords +
similarity to a reference set of known substrate transporters.

```bash
tpclass filter \
    --v3_csv <candidates.csv> \
    --out <filtered.csv> \
    --ref_set <reference.fasta> \
    --esm2 <esm2_path> \
    [--tc_classes <comma-separated>] \
    [--keywords <comma-separated>] \
    [--sim_min <float, default 0.5>] \
    [--query_embeddings <embeddings.npz>] \
    [--device {cuda|cpu}]
```

`--ref_set` is a FASTA of known substrate transporters; its ESM-2 embeddings
are computed on the fly with `--esm2`.  Use `tpclass compute-refs` once and
`tpclass three-layer` instead when the same reference set is reused.

### `tpclass three-layer`

Same filter, but with pre-computed reference embeddings.

```bash
tpclass three-layer \
    --v3_csv <candidates.csv> \
    --out <filtered.csv> \
    --ref_embeddings <ref_embeddings.npz> \
    [--query_embeddings <query_embeddings.npz>] \
    [--tc_classes <comma-separated>] \
    [--keywords <comma-separated>] \
    [--sim_min <float, default 0.5>]
```

The `.npz` must contain an `accessions` array and an `embeddings` matrix
(`acc` / `embedding` are also accepted).

### `tpclass compare-esm-blast`

Compare ESM-2 (TPClass) and DIAMOND BLASTp on a set of candidate proteins.

```bash
tpclass compare-esm-blast \
    --candidates <candidates.csv> \
    --fasta <candidates.fasta> \
    --ref_fasta <references.fasta> \
    --out <output.tsv> \
    [--esm_high <float, default 0.95>] \
    [--id_low <float, default 0.30>] \
    [--coverage_low <float, default 0.50>] \
    [--evalue <float, default 1e-3>] \
    [--max_target_seqs <int, default 1>] \
    [--diamond <path>] \
    [--threads <int, default 4>]
```

`--coverage_low` is the minimum fraction of the **full query** that the
alignment has to cover; the coverage denominator is the query length, not the
alignment span.  `--max_target_seqs 1` reproduces the published protocol
(only the single best DIAMOND hit is considered); use a larger value when
screening for missed calls, because with a single hit a query whose best
alignment fails the identity/coverage thresholds is counted as BLAST-missed
even when a weaker hit would have passed.

### `tpclass compare-reports`

Compare two prediction tables row by row.  This is the verification tool for
the claim that the package reproduces the released pipeline: run
`scripts/predict_transport_family_v3.py` and `tpclass predict` on the same
input FASTA, then compare the two tables.

```bash
tpclass compare-reports \
    --a <first.csv> \
    --b <second.csv> \
    [--key <column, default query>] \
    [--tol <float, default 1e-4>] \
    [--show <int, default 5>]
```

Rows are aligned on the join column; numeric columns
(`transport_prob`, `top1_similarity`, `top1_sim_gap`, `top_family_purity`,
...) are compared with a tolerance, every other column is compared as a
string.  Empty cells stay empty and are never treated as zero.  The command
exits with status 1 when the tables differ, so it can be used in a shell
check.

### `tpclass paired-test`

Paired significance tests between two configurations of the held-out
benchmark: an accuracy-based McNemar test, a paired bootstrap confidence
interval, and the Jaccard overlap of the predicted-positive sets.

```bash
tpclass paired-test \
    --preds <all_preds.csv> \
    [--out <paired_test.csv>] \
    [--pairs "3B_cached,650M_cached" "650M_fresh,650M_cached"] \
    [--n_boot <int, default 10000>] \
    [--seed <int, default 0>]
```

The input is the long-format table written by `scripts/dump_preds.py`
(`protein_id`, `config`, `y_true`, `y_pred`).  Without `--pairs` the four
Table-1 comparisons are used.

The McNemar test here counts the proteins where one model is **correct** and
the other is **not**.  The released `scripts/paired_test_heldout.py` instead
counts the pairs where the two models predict *different labels*, which
measures the difference in the number of predicted transporters and can
return `p = 1` for a perfect model against a model that is entirely wrong.
Both variants are reported in the output columns
(`mcnemar_p` and `mcnemar_label_variant_p`).

### `tpclass family-metrics`

Family-level (4-digit TC) precision, fuzzy recall and strict recall, computed
from a single implementation.

```bash
tpclass family-metrics \
    --pred_csv <final_report.csv> \
    --gt_tsv <ground_truth.tsv> \
    [--out <summary.csv>] \
    [--fuzzy_rule {with_family|all_yes}] \
    [--query_col query] [--family_col tc_number] [--is_yes_col is_transport]
```

The released pipeline computes "fuzzy recall" under two different rules:
`scripts/eval_family_precision.py` counts only predictions that received a TC
family, while `scripts/eval_hybrid_v3.py` counts every predicted positive
inside the ground truth.  Both rules are always reported
(`recall_fuzzy_matched` and `recall_fuzzy_all_yes`); `--fuzzy_rule` selects
which one fills `recall_fuzzy`.

Note that the 4-digit metric scores the **top-1** retrieval
(`tc_number`, truncated to four components).  The majority vote in Phase 2
aggregates the 3-digit family and is exposed separately as `tc_family`.

### `tpclass compute-refs`

Compute mean-pooled ESM-2 embeddings for a reference FASTA.

```bash
tpclass compute-refs \
    --fasta <reference.fasta> \
    --out <ref_embeddings.npz> \
    --esm2 <esm2_path> \
    [--device {cuda|cpu}] \
    [--max_len <int, default 800>] \
    [--batch_size <int, default 8>]
```

### `tpclass split`

Split a TCDB FASTA into train/val/test.

```bash
tpclass split \
    --input <tcdb.fasta> \
    [--negatives <non_transporters.fasta>] \
    --out_dir <out_dir> \
    [--strategy {cluster|stratified}] \
    [--train_frac <float, default 0.8>] \
    [--val_frac <float, default 0.1>] \
    [--seed <int, default 42>] \
    [--cluster_threshold <float, default 0.5>]
```

`cluster` keeps homologous proteins in the same subset (k-mer Jaccard
clustering); `stratified` reproduces the random split stratified by TC family
used by the released pipeline.

### `tpclass train-phase1` / `tpclass train-phase2`

```bash
tpclass train-phase1 \
    --data_dir <dir with X_train.npy / Y_train.npy / X_val.npy / Y_val.npy> \
    --out_dir <out_dir> \
    [--epochs 100] [--batch_size 1024] [--lr 1e-3] \
    [--backbone {3B|650M}] [--skip_xgb] [--calibrate]

tpclass train-phase2 \
    --data_csv <csv with 'sequence' and 'tc_number'> \
    --esm2 <esm2_path> \
    --out_dir <out_dir> \
    [--epochs 25] [--batch_size 12] [--seed 42]
```

## Python API

### Basic usage

```python
from tpclass import TPClassPredictor

predictor = TPClassPredictor(
    phase1_3b_checkpoint="models/v3_retrained_3b/phase1/transport_classifier_best.pth",
    phase1_xgb_3b="models/v3_retrained_3b/phase1/transport_xgb_best.json",
    phase1_650m_checkpoint="models/v3_retrained/phase1/transport_classifier_best.pth",
    phase1_xgb_650m="models/v3_retrained/phase1/transport_xgb_best.json",
    phase2_3b_checkpoint="models/v3_retrained_3b/phase2/best_phase2.pth",
    phase2_650m_checkpoint="models/v3_retrained/phase2/best_phase2.pth",
    esm2_3b_path="models/esm2_3b",
    esm2_650m_path="models/esm2_650m",
)

df = predictor.predict(fasta="proteome.fasta", backbone="3B", threshold=0.50)
print(f"Predicted {(df['is_transport'] == 'YES').sum()}/{len(df)} as transporters")
print(df.head())
```

The reference index and its metadata default to the files next to the Phase 2
checkpoint; pass `ref_embeddings_3b` / `ref_embeddings_650m` /
`ref_metadata` explicitly to override.

### Cascade

```python
df = predictor.predict(fasta="proteome.fasta", backbone="cascade")

# Which model supplied the family for each candidate?
print(df["family_source"].value_counts())
```

The candidate set is the 3B YES set.  The family comes from the 650M model
when the 650M model also called the protein YES and produced a family, and
from the 3B model otherwise; the retrieval evidence always comes from the
same model as the family.

### Substrate filter

```python
from tpclass.filters.three_layer import ThreeLayerFilter

flt = ThreeLayerFilter(
    ref_set="refs/hmo_refs.fasta",
    esm2_path="models/esm2_3b",
    tc_classes=["2.A.1", "3.A.1", "4.A.1", "9.B"],
    keywords=["fucose", "sialic", "lactose", "HMO"],
    min_ref_sim=0.7,
)
filtered = flt.filter("results/candidates.csv")
print(f"After filter: {len(filtered)} candidates")
```

## Output format

| Column | Type | Description |
|--------|------|-------------|
| `query` | str | UniProt accession |
| `transport_prob` | float | Phase 1 ensemble probability (0-1) |
| `seq_length` | int | Protein length in residues |
| `is_transport` | str | `YES` or `NO` (based on the threshold) |
| `tc_number` | str | Four-digit TC number of the top-1 reference hit |
| `tc_family` | str | Three-component TC family of the majority vote |
| `tc_description` | str | TCDB description of the top-1 reference |
| `top1_similarity` | float | Cosine similarity to the top-1 reference |
| `top1_sim_gap` | float | Top-1 minus top-2 similarity |
| `top_family_votes` | int | Number of neighbours voting for the winning family |
| `top_family_purity` | float | Votes / k |
| `max_ref_sim` | float | Same as `top1_similarity` for Phase 2 hits |
| `best_ref_acc` | str | Accession of the top-1 reference |
| `best_ref_gene` | str | Gene name of the top-1 reference (when the metadata has one) |
| `family_source` | str | Which model supplied the family (`3B` / `650M`); empty when Phase 2 did not run |

The cascade output additionally carries `transport_prob_650m`.

## Performance

Approximate runtimes on a single A100 (80 GB):

| Step | UniProt 575k | K. marxianus 5186 |
|------|--------------|-------------------|
| ESM-2 3B embedding extraction | ~12 h | ~7 min |
| ESM-2 650M embedding extraction | ~4 h | ~3 min |
| s4pred secondary structure | ~2 h | ~1 min |
| Phase 1 prediction | ~10 min | <1 min |
| Phase 2 retrieval | ~30 min | <1 min |
| Three-layer filter | ~5 min | <1 min |
| ESM-2 vs BLAST comparison | ~2 h | ~3 min |
| **Total** | **~20 h** | **~15 min** |

ESM-2 3B dominates the runtime. For faster iteration on small proteomes, use 650M.

## See also

- [Architecture](architecture.md) — How TPClass works
- [Installation](installation.md) — Setup and troubleshooting
- [Case studies](../examples) — HMO, squalene and isobutanol tutorials
