# TPClass Architecture

This document describes the technical architecture of TPClass, including the two-phase pipeline, the cascade strategy, and the three-layer evidence filter.

## Overview

TPClass is a **two-phase** predictor:

1. **Phase 1**: Binary classification (transporter vs. non-transporter) using ESM-2 protein language model embeddings + s4pred secondary structure features
2. **Phase 2**: TC family assignment via contrastive retrieval against a TCDB reference index

A **cascade strategy** combines the two backbones (3B and 650M) for optimal precision-recall balance.

A **three-layer evidence filter** enables substrate-specific mining by combining TCDB taxonomy, description keywords, and ESM-2 cosine similarity.

## Phase 1: Binary transport classifier

**Input**: protein sequence (FASTA)
**Output**: `transport_prob` in `[0, 1]`

### Features

For each protein:

- **ESM-2 mean-pooled embedding**: 1280-d (650M) or 2560-d (3B)
  - Computed as `mean(last_hidden_state * mask)`, where the mask excludes the
    padding and `eos` tokens and keeps the `cls` token
- **s4pred secondary structure**: 3-d vector of H/E/C fractions
- **Total feature dimension**: 1283 (650M) or 2563 (3B)

The s4pred features are concatenated, not appended as an alternative
representation: the published Phase 1 checkpoints were trained on the
concatenated vector. When s4pred cannot be run, the released pipeline
substitutes the constant `(0.5, 0.2, 0.3)` per protein; `tpclass` does the
same and logs a warning, because silently changing the feature space would
change the predictions.

### Architecture

```
ESM-2 mean-pool (1280/2560-d)  --\
                                   +--> MLP: Linear(->512) BN ReLU Drop
s4pred H/E/C fractions (3-d)   --/     + 2 x residual 512 + Linear(->128) BN ReLU
                                       + Linear(->1)  -> transport_prob_mlp

ESM-2 mean-pool + s4pred       ------> XGBoost          -> transport_prob_xgb

ensemble: 0.8 * mlp + 0.2 * xgb  (3B)
           0.4 * mlp + 0.6 * xgb  (650M)
           optional isotonic regression (opt-in)
```

The mean pooling excludes the padding and `eos` tokens and keeps the `cls`
token; the Phase 2 pooling excludes padding and `cls` and keeps `eos`. The two
are therefore different vectors and are not interchangeable.

### Training

- **Positive set**: 19,337 TCDB-derived transporters
- **Negative set**: 19,242 UniProt random non-transporters (1:1 balanced)
- **Split**: 80% train, 20% validation, separate held-out test (4,785 sequences)
- **Loss**: Binary cross-entropy
- **Calibration**: Isotonic regression on the validation set (opt-in at
  inference; the released v3 script scores with the raw ensemble probability)

> **Homology leakage.** A random split stratified by TC family puts close
> homologues into different subsets, so a held-out score can be optimistic.
> `tpclass split --strategy cluster` clusters the sequences by k-mer
> similarity (Jaccard >= 0.5 on 5-mers) and assigns whole clusters to one
> subset, which keeps homologues together. Use it whenever the reported
> performance has to be free of homology leakage.

### Performance (TCDB held-out test, 3B / fresh mode)

- Accuracy: 0.8804
- F1: 0.8836
- MCC: 0.7615

## Phase 2: TC family assignment (contrastive retrieval)

**Input**: protein sequence (only proteins passing Phase 1 threshold)
**Output**: 3-digit TC family + 4-digit `tc_number`

### Reference index

- 19,216 TCDB entries with 4-digit TC labels
- Contrastive embeddings produced by the Phase 2 head
- Stored as `tcdb_embeddings.npy` (matrix) plus `tcdb_index_metadata.csv`
  (one row per reference, with `tc_number` and `tc_family`)
- Auto-paired with the Phase 2 head to prevent cross-metric-space failure

### Retrieval

For each query protein:

1. Encode the sequence with the Phase 2 backbone and the contrastive head
   (the head sees the concatenation of the `cls` vector and the mean-pooled
   token vector and returns an L2-normalised embedding)
2. Compute the cosine similarity to all reference entries
3. Take the top-k = 10 nearest neighbours (default)
4. **Majority vote on the TC family** (three components) -> assigned family
5. The **top-1 hit** provides the four-digit `tc_number` (sub-subfamily)

`top1_sim_gap` (top-1 minus top-2 similarity) and `top_family_purity`
(votes / k) quantify how decisive the assignment was.

### Fidelity notes

Three details of the released pipeline are easy to get wrong and are handled
explicitly in `tpclass/core`:

1. **Transformer layer.** The contrastive encoder reads the **last**
   transformer layer, which is what `train_phase2_contrastive.py` used during
   training. The released inference script hard-codes `repr_layers=[33]`; for
   the 33-layer 650M that is the last layer and the two agree, but for the
   36-layer 3B it selects layer 33 instead of 36.
2. **Fine-tuned ESM-2 weights.** The Phase 2 checkpoint stores the whole
   contrastive encoder under `model_state`. Contrastive training unfroze the
   last six transformer layers plus the final layer norm, so the checkpoint
   contains a *partially fine-tuned* ESM-2. `tpclass` restores those `esm.*`
   tensors into the backbone before embedding the queries. Loading only the
   projection head would place the queries and the TCDB reference index in two
   different embedding spaces and silently degrade the family calls.
3. **Separate ESM-2 instances.** Phase 2 therefore gets its own ESM-2
   instance instead of sharing the Phase 1 backbone: restoring the fine-tuned
   weights into a shared backbone would also change the Phase 1 features,
   which were trained on the pretrained ESM-2.

### Cross-species performance (8 species, cascade)

- Precision: 0.910
- Fuzzy recall: 0.941
- Strict recall: 0.860

## Cascade strategy

The cascade combines the strengths of both backbones:

```
3B Phase 1 (high recall)
  |
  +--> 3B YES proteins  ----------------> candidate set
  |                                        (650M never removes a candidate)
  |
  +--> for a candidate that 650M also called YES
        |
        +--> 650M Phase 2 (high family precision)
               |
               +--> final TC family assignment
```

- **3B Phase 1** is the binary filter (broad recall)
- **650M Phase 2** is the family assigner (high precision)
- The **candidate set is the 3B YES set**; the 650M model only decides which
  family label (and which retrieval evidence) is used
- Fallback to the 3B TC family when the 650M model did not call the protein
  YES, or called it YES without producing a family
- The retrieval evidence (`top1_similarity`, `top_family_votes`,
  `best_ref_acc`, ...) is always taken from the **same** model that provided
  the family, so a label and its supporting evidence never come from
  different models

The `family_source` column of the output records which model supplied the
family for each candidate (`3B` or `650M`).

## Three-layer evidence filter

For substrate-specific transporter mining (e.g., HMO, squalene), the v3 output is post-filtered by three layers:

### Layer 1: TCDB TC class whitelist

Restrict to specific TC class prefixes. Examples:

- HMO: `2.A.1, 2.A.2, 2.A.7, 2.A.21, 2.A.3, 2.A.29, 3.A.1, 3.D.1, 4.A.1, 9.B`
- Squalene: `2.A.6, 9.B.198, 3.D.1, 3.D.10, 3.A.2.1, 1.C.59, 3.E.4` (tightened to avoid V-ATPase c-subunit flood)
- Isobutanol: `2.A.1, 2.A.51, 3.A.1, 2.A.6, 2.A.7`

A prefix matches when the TC number equals it or continues with `.`, so `9.B`
matches `9.B.198.2.1` but not `9.C.1.1.1`.

The whitelist encodes **biological prior knowledge** of which TC families are likely to contain the target transporter.

### Layer 2: TCDB description keyword

Match TCDB-curated family descriptions against a substrate-specific keyword list. Examples:

- HMO: `fucose, sialic, lactose, galactose, glucose, GlcNAc, oligosaccharide, HMO, sugar`
- Squalene: `squalene, hopene, hopanoid, polyprenyl, farnesyl, isoprenoid, terpenoid, sterol, lipid`

**TCDB descriptions are maintained by the Saier laboratory and provide a curated, cross-species-consistent signal independent of UniProt annotation noise.**

### Layer 3: ESM-2 cosine similarity to substrate-specific reference set

Compute max cosine similarity between each candidate's ESM-2 embedding and a curated reference set of 5-25 proteins known (or suspected) to be in the target substrate's transport pathway.

The embeddings on both sides are the Phase 1 representation (mean over the
last layer, `pad` and `eos` excluded), so a reference set computed with
`tpclass compute-refs` is directly comparable with the `max_ref_sim` column
of a prediction table.

Threshold typically 0.5-0.7 (lower for weak reference sets).

### Output

Candidates are ranked by **max_ref_sim** (descending), then by `transport_prob`, then by `top1_similarity`.

### Example: squalene case study

| Filter | Candidates remaining |
|--------|---------------------|
| v3 5,186 proteins -> 4,994 YES | 4,994 |
| TC class whitelist (2.A.6, 9.B.198, ...) | 921 |
| TCDB description keyword (squalene, hopene, ...) | 66 |
| ESM-2 cosine sim >= 0.5 to 6 squalene refs | 66 (all pass) |

-> 11 top-ranked candidates, of which 6 are LAM-family (9.B.198.2.x) including YSP2.

## Cross-species evaluation

The 8-species benchmark (4 bacteria, 1 actinobacterium, 1 fungus, 1 animal):

- E. coli K-12 MG1655, E. coli BL21
- B. subtilis 168, L. lactis IL1403
- C. glutamicum, A. niger CBS 513.88
- S. cerevisiae S288C, H. sapiens

Ground truth: UniProt-reviewed transport annotations -> TCDB cross-references (E. coli BL21 ground truth transferred from K-12 by DIAMOND BLASTp).

The cascade strategy achieves the best precision-recall balance:

- **Precision**: 0.910
- **Fuzzy recall**: 0.941
- **Strict recall**: 0.860

The three cross-species metrics are not independent: the strict recall is the
fuzzy recall times the family precision, by construction.

## Why two phases?

The two-phase design separates **detection** (Phase 1) from **classification** (Phase 2):

- Phase 1 optimizes **precision-recall** for the binary decision
- Phase 2 optimizes **family-level precision** for the multi-class problem

This decoupling allows:

- Different backbones for each phase (3B for recall, 650M for precision)
- Different feature extractors (s4pred for Phase 1, contrastive head for Phase 2)
- Different training objectives (binary X-entropy, SupCon)

## See also

- [Installation](installation.md) - Setup
- [Usage](usage.md) - CLI + Python API
- [Case studies](../examples) - Case studies
