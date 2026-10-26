# HMO transporter mining with TPClass

## Background

Human milk oligosaccharides (HMOs) are complex sugars that serve as prebiotics for infant gut microbiota. The proteins that import and process HMOs in bacteria (particularly *Bifidobacterium longum* subsp. *infantis* and *Escherichia coli*) are of interest for infant nutrition and microbiome engineering.

The HMO case study in TPClass was the first validation of the method. It identified 1,066 transporter candidates from 575k UniProt proteins, of which 79.1% are BLAST-missed novel findings (sequence identity < 30% to any reference).

## Reference set (25 proteins)

| Tier | Accession | Gene | Species | Substrate | Source |
|------|-----------|------|---------|-----------|--------|
| Literature (10) | P02920 | lacY | E. coli | lactose | Bradley 1987 |
| | P11551 | fucP | E. coli | L-fucose | Lu 2003 |
| | P39352 | nanX | E. coli | sialic acid | Vimr 2004 |
| | P23200 | mglC | E. coli | galactose | Hogg 1991 |
| | P23889 | kpsM | E. coli | polysialic acid | Cress 1988 |
| | P44776 | fucP | H. influenzae | fucose | Hood 1996 |
| | P18817 | lacY | K. oxytoca | lactose | Buemann 1993 |
| | P59832 | lacY | K. pneumoniae | lactose | Shukla 2005 |
| | Q9ZK41 | gluP | H. pylori | glucose | Psakis 2009 |
| | Q24JQ0 | SLC35D4 | H. sapiens | UDP-GlcNAc | Muraoka 2001 |
| User-curated (15) | SetA, SetB, YejA, OmpF, YhhS, CmSET, YbSET, ChiP | E. coli | sugar efflux | 2024 lab curation |
| | TauB, LivM, NikD, BtuD | E. coli | ABC SBPs | |
| | YdeA, MdtM, MdfA | E. coli | multidrug efflux | |

The reference FASTA belongs in `data/refs/hmo_refs.fasta` (not tracked in git; see `data/README.md`).

## Pipeline

```bash
# 1. Compute reference embeddings
python scripts/compute_ref_embeddings.py \
    --fasta data/refs/hmo_refs.fasta \
    --out models/ref_embeddings_hmo_v2.npz

# 2. Run TPClass v3 on UniProt (575k proteins)
python scripts/predict_transport_family_v3.py \
    --fasta inputs/uniprot_sprot.fasta \
    --model_dir models/v3_retrained_3b \
    --ref_embeddings models/ref_embeddings_hmo_v2.npz \
    --ref_metadata data/tcdb_annot.clean.tsv \
    --out outputs/uniprot_sprot/final_report.csv

# 3. Apply substrate-specific three-layer filter
python scripts/filter_tcdb_esm_v2.py \
    --v3_csv outputs/uniprot_sprot/final_report.csv \
    --emb outputs/uniprot_sprot/embeddings.npz \
    --ref_npz models/ref_embeddings_hmo_v2.npz \
    --annot data/tcdb_annot.clean.tsv \
    --tc_classes 2.A.1,2.A.2,2.A.7,2.A.21,2.A.3,2.A.29,3.A.1,3.D.1,4.A.1,9.B \
    --keywords fucose,sialic,lactose,galactose,glucose,GlcNAc,oligosaccharide,HMO,sugar \
    --sim_min 0.7 \
    --out_dir outputs/uniprot_sprot/hmo_v2

# 4. Compare ESM-2 vs BLAST
python scripts/compare_esm_vs_blast.py \
    --top_csv outputs/uniprot_sprot/hmo_v2/hmo_v2_top.csv \
    --fasta inputs/uniprot_sprot.fasta \
    --ref_fasta data/refs/hmo_refs_25.fasta \
    --out_dir outputs/uniprot_sprot/hmo_v2
```

## Result summary

- **1,066 candidates** from 575k UniProt proteins
- **79.1% BLAST-missed** (843/1066)
- **88.6% TClass-found** (944/1066, ESM-2 sim ≥ 0.95)
- **9.5% BLAST-found** (101/1066, identity ≥ 30%)

Top TC families: 2.A.1 (MFS, 416), 9.B (auxiliary factors, 323), 3.A.1 (ABC, 279), 4.A.1 (PTS, 48).

## Wet-lab validation

5+5 design: 5 literature positive controls + 5 BLAST-missed novel candidates.
See [docs/tutorials/hmo_tutorial.md](../../docs/usage.md) for full details.
