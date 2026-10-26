#!/bin/bash
# Squalene / terpenoid transporter discovery with TPClass
# Run from project root: bash examples/squalene/run_squalene.sh
#
# Prerequisites (see models/README.md):
#   bash models/download.sh
# The input proteome and the reference FASTAs are not shipped with the
# repository; point the variables below at your own copies.

set -e
cd "$(dirname "$0")/../.."

echo "=== Squalene / terpenoid transporter discovery with TPClass ==="
echo "Working dir: $(pwd)"

# Configuration
SQU_REFS="${SQU_REFS:-data/refs/squalene_refs.fasta}"
K_MARXIANUS="${K_MARXIANUS:-inputs/max.fasta}"
ESM2_3B="${ESM2_3B:-models/esm2_3b}"
MODEL_3B="${MODEL_3B:-models/v3_retrained_3b}"
FINAL_CSV="outputs/max/final_report.csv"
FILTERED_CSV="outputs/max/squalene_v2/squalene_filtered.csv"
SQU_OUT="outputs/max/squalene_v2"

mkdir -p "$SQU_OUT"

# Step 1: Predict transporters
if [ ! -f "$FINAL_CSV" ]; then
    echo "[1/3] Running TPClass on the K. marxianus proteome..."
    tpclass predict \
        --fasta "$K_MARXIANUS" \
        --out "$FINAL_CSV" \
        --backbone 3B \
        --phase1_3b "$MODEL_3B/phase1/transport_classifier_best.pth" \
        --phase1_xgb_3b "$MODEL_3B/phase1/transport_xgb_best.json" \
        --phase2_3b "$MODEL_3B/phase2/best_phase2.pth" \
        --esm2_3b "$ESM2_3B"
else
    echo "[1/3] $FINAL_CSV already exists, skipping"
fi

# Step 2: Apply the three-layer filter with the tightened TC whitelist
echo "[2/3] Applying the three-layer filter (tightened TC whitelist)..."
tpclass filter \
    --v3_csv "$FINAL_CSV" \
    --ref_set "$SQU_REFS" \
    --esm2 "$ESM2_3B" \
    --tc_classes 2.A.6,9.B.198,3.D.1,3.D.10,3.A.2.1,1.C.59,3.E.4 \
    --keywords squalene,hopene,hopanoid,polyprenyl,farnesyl,isoprenoid,terpenoid,sterol,lipid \
    --sim_min 0.5 \
    --out "$FILTERED_CSV"

# Step 3: ESM-2 vs BLAST comparison
echo "[3/3] ESM-2 vs BLAST comparison..."
tpclass compare-esm-blast \
    --candidates "$FILTERED_CSV" \
    --fasta "$K_MARXIANUS" \
    --ref_fasta "$SQU_REFS" \
    --out "$SQU_OUT/squalene_esm_vs_blast.tsv"

echo "=== Done. Results in: $SQU_OUT ==="
ls -la "$SQU_OUT"
