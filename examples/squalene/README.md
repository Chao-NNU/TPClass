# Squalene / terpenoid transporter discovery with TPClass

## Background

Squalene is a triterpenoid precursor of sterols, ubiquinone, and hopanoids. In yeast, squalene moves between the ER, lipid droplets, and plasma membrane by a protein-mediated process, but the responsible transporter had not been identified in any yeast species before this work.

The squalene case study demonstrated TPClass's ability to mine for substrate-specific transporters with a **weak reference set** (only 6 directly validated squalene-related proteins, none from yeast), using a **tightened TC whitelist** (9.B.198 specifically) to avoid V-ATPase c-subunit false positives.

## Reference set (6 proteins)

| Accession | Gene | Species | TC family | Role |
|-----------|------|---------|-----------|------|
| B3QHB6 | HpnN | R. palustris | 2.A.6.7.5 | bacterial squalene transporter |
| A0A0H3KP92 | HpnN | B. multivorans | 2.A.6.7.5 | bacterial squalene transporter |
| Q2T2R2 | HpnN | B. thailandensis | 2.A.6.7.5 | bacterial squalene transporter |
| Q2FZY8 | POPT | S. aureus | 2.A.6.7.6 | polyprenyl-phosphate transporter |
| Q9KNC5 | POPT | V. cholerae | 2.A.6.7.6 | polyprenyl-phosphate transporter |
| B0R3W4 | CoxX | H. salinarum | 2.A.6.7.7 | carotenoid-associated transporter |

> Note: 3 metabolic enzymes (PimE, ERG1_YEAST, ERG1_SCHPO) were removed from the original reference set because they caused V-ATPase c-subunit (3.A.2.2) false-positive floods via ESM-2 structural similarity. The current set is purely transport proteins.

## Pipeline

```bash
# 1. Compute reference embeddings
python scripts/compute_ref_embeddings.py \
    --fasta data/refs/squalene_refs.fasta \
    --out models/ref_embeddings_squalene.npz

# 2. Run TPClass v3 on K. marxianus (5186 proteins)
python scripts/predict_transport_family_v3.py \
    --fasta inputs/max.fasta \
    --model_dir models/v3_retrained_3b \
    --ref_embeddings models/ref_embeddings_squalene.npz \
    --ref_metadata data/tcdb_annot.clean.tsv \
    --out outputs/max/final_report.csv

# 3. Apply substrate-specific filter with tightened TC whitelist
python scripts/filter_tcdb_esm_v2.py \
    --v3_csv outputs/max/final_report.csv \
    --emb outputs/max/embeddings.npz \
    --ref_npz models/ref_embeddings_squalene.npz \
    --annot data/tcdb_annot.clean.tsv \
    --tc_classes 2.A.6,9.B.198,3.D.1,3.D.10,3.A.2.1,1.C.59,3.E.4 \
    --keywords squalene,hopene,hopanoid,polyprenyl,farnesyl,isoprenoid,terpenoid,sterol,lipid \
    --sim_min 0.5 \
    --out_dir outputs/max/squalene_v2
```

## Result summary

- **66 candidates** from 5186 K. marxianus proteins
- **65 LAM-family** (9.B.198.2.x, lipid-anchored membrane proteins) including:
  - YSP2 (A0ABX6F7C0) — 9.B.198.2.1
  - 3× LAM5 paralogs (A0ABX6EPC2, F2B9, EZF9) — 9.B.198.2.3
  - 2× LAM6 paralogs (A0ABX6ERE3, EN35) — 9.B.198.2.4
- **1 HpnN** (B3QHB6) — bacterial squalene transporter homolog
- The V-ATPase c-subunit (3.A.2.2) was specifically excluded by the TC class narrowing (3.A.2 → 3.A.2.1)

## Wet-lab validation (in progress)

4 candidate transporters selected for uptake assay and K. marxianus engineering:
1. **B3QHB6** (HpnN, R. palustris) — bacterial positive control
2. **Q2U0C3** (LAM5, K. marxianus) — fungal squalene-binding candidate
3. **A7KAN4** (LAM6, K. marxianus) — fungal squalene-binding candidate
4. **P0CO40** (LAM5, S. cerevisiae) — yeast squalene-binding candidate

See [docs/tutorials/squalene_tutorial.md](../../docs/usage.md) for full details.
