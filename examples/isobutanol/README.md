# Isobutanol exporter mining with TPClass (in progress)

## Background

Isobutanol (2-methyl-1-propanol, 74 Da) is a potential biofuel produced via the keto-acid pathway from valine biosynthesis. For microbial production, isobutanol needs to be exported out of the cell, otherwise it accumulates to toxic levels.

Unlike HMO and squalene, the literature on isobutanol-specific transporters is **scarce** — the only well-validated exporter is **AQR1** in *Saccharomyces cerevisiae* (Yoshikawa 2014). The reference set for this case study is therefore expanded to the **DHA1/DHA2 family** of yeast drug:H+ antiporters, which have been experimentally shown to transport a broad range of small organic compounds including alcohols.

## Reference set (18 proteins, mixed yeast + bacterial)

### Yeast DHA1 family (MFS drug:H+ antiporter, 2.A.1.2) — 9 proteins
- AQR1 (P53943) — confirmed isobutanol exporter
- QDR1, QDR2, QDR3 (P40475, P40474, P38227) — quinidine resistance
- FLR1 (P38124) — fluconazole resistance
- TPO1, TPO2, TPO4 (Q07824, P53283, Q12256) — polyamine transport
- SGE1 (P33335) — multidrug

### Yeast additional (DHA1, 2.A.1.2) — 3 proteins
- TPO3 (Q06451) — polyamine paralog
- HOL1 (P53389) — cation/alcohol transport
- YHK8 (P38776) — drug/proton antiporter

### Yeast DHA2 (2.A.1.3) — 1 protein
- AZR1 (P50080) — azole resistance

### Bacterial MFS / SMR drug efflux — 4 proteins
- MdfA (P0AEY8) — E. coli MFS drug
- EmrD (P31459) — E. coli MFS drug
- EmrB (P0ADN0) — E. coli MFS drug
- SugE (P0AG31) — E. coli SMR (small multidrug resistance)

## Pipeline (planned)

```bash
# 1. Compute reference embeddings
python scripts/compute_ref_embeddings.py \
    --fasta data/refs/isobutanol_refs.fasta \
    --out models/ref_embeddings_isobutanol.npz

# 2. Run TPClass v3 on UniProt (575k proteins)
python scripts/predict_transport_family_v3.py \
    --fasta inputs/uniprot_sprot.fasta \
    --model_dir models/v3_retrained_3b \
    --ref_embeddings models/ref_embeddings_isobutanol.npz \
    --ref_metadata data/tcdb_annot.clean.tsv \
    --out outputs/uniprot_sprot/final_report_isobutanol.csv

# 3. Apply substrate-specific filter
python scripts/filter_tcdb_esm_v2.py \
    --v3_csv outputs/uniprot_sprot/final_report_isobutanol.csv \
    --emb outputs/uniprot_sprot/embeddings.npz \
    --ref_npz models/ref_embeddings_isobutanol.npz \
    --annot data/tcdb_annot.clean.tsv \
    --tc_classes 2.A.1,2.A.51,3.A.1,2.A.6,2.A.7 \
    --keywords isobutanol,ethanol,alcohol,organic solvent,drug efflux,multidrug,DHA1,MATE,MFS,ABC,biofuel \
    --sim_min 0.7 \
    --out_dir outputs/uniprot_sprot/isobutanol_v1
```

## Sanity check

After running, verify that **AQR1 (P53943) appears in the top of the candidate list** — this confirms the pipeline correctly identifies the only known isobutanol exporter.

## Expected output

- Several hundred candidates from UniProt (575k proteins)
- Top candidates should include known yeast DHA1 family members (FLR1, QDR1-3) and bacterial MFS drug efflux proteins (MdfA, EmrD)
- Novel candidates will be sequences with high ESM-2 similarity to the reference set but low BLAST identity
