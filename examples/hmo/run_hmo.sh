#!/bin/bash
# HMO transporter mining with TPClass
# Run from project root: bash examples/hmo/run_hmo.sh
#
# Prerequisites (see models/README.md):
#   bash models/download.sh
# The input proteome and the reference FASTAs are not shipped with the
# repository; point the variables below at your own copies.

set -e
cd "$(dirname "$0")/../.."

echo "=== HMO transporter mining with TPClass ==="
echo "Working dir: $(pwd)"

# Configuration
HMO_REFS="${HMO_REFS:-data/refs/hmo_refs_25.fasta}"
UNIPROT="${UNIPROT:-inputs/uniprot_sprot.fasta}"
ESM2_3B="${ESM2_3B:-models/esm2_3b}"
MODEL_3B="${MODEL_3B:-models/v3_retrained_3b}"
FINAL_CSV="outputs/uniprot_sprot/final_report.csv"
FILTERED_CSV="outputs/uniprot_sprot/hmo_v2/hmo_filtered.csv"
HMO_OUT="outputs/uniprot_sprot/hmo_v2"

mkdir -p "$HMO_OUT"

# Step 1: Predict transporters (3B backbone, Phase 1 + Phase 2)
if [ ! -f "$FINAL_CSV" ]; then
    echo "[1/3] Running TPClass on the input proteome (this takes hours for UniProt)..."
    tpclass predict \
        --fasta "$UNIPROT" \
        --out "$FINAL_CSV" \
        --backbone 3B \
        --phase1_3b "$MODEL_3B/phase1/transport_classifier_best.pth" \
        --phase1_xgb_3b "$MODEL_3B/phase1/transport_xgb_best.json" \
        --phase2_3b "$MODEL_3B/phase2/best_phase2.pth" \
        --esm2_3b "$ESM2_3B"
else
    echo "[1/3] $FINAL_CSV already exists, skipping"
fi

# Step 2: Apply the three-layer substrate filter
echo "[2/3] Applying the three-layer filter..."
tpclass filter \
    --v3_csv "$FINAL_CSV" \
    --ref_set "$HMO_REFS" \
    --esm2 "$ESM2_3B" \
    --tc_classes 2.A.1,2.A.2,2.A.7,2.A.21,2.A.3,2.A.29,3.A.1,3.D.1,4.A.1,9.B \
    --keywords fucose,sialic,lactose,galactose,glucose,GlcNAc,oligosaccharide,HMO,sugar \
    --sim_min 0.7 \
    --out "$FILTERED_CSV"

# Step 3: ESM-2 vs BLAST comparison
echo "[3/3] ESM-2 vs BLAST comparison..."
tpclass compare-esm-blast \
    --candidates "$FILTERED_CSV" \
    --fasta "$UNIPROT" \
    --ref_fasta "$HMO_REFS" \
    --out "$HMO_OUT/hmo_esm_vs_blast.tsv"

echo "=== Done. Results in: $HMO_OUT ==="
ls -la "$HMO_OUT"
