#!/usr/bin/env bash
# ============================================================
# run_5_variants_parallel.sh
# Run the 5 ablation variants in parallel across 2 GPUs.
# Assumes the balanced val fasta is already at $BAL and the
# retrained Phase 1 assets exist at $MODELS/phase1/.
#
# Usage:
#   bash scripts/run_5_variants_parallel.sh
# ============================================================

SCRIPTS=/data2/chao/transport/scripts
ROOT=/data2/chao/transport
DATA=$ROOT/data
MODELS=$ROOT/models/v3_retrained
OUT=$ROOT/outputs/ablation
BAL=$DATA/tcdb_val_balanced.fasta

mkdir -p $OUT

if [ ! -f $BAL ]; then
    echo "[ERROR] Balanced val fasta not found at $BAL"
    echo "        Run run_ablation_train_val.sh first (it builds the balanced fasta)"
    exit 1
fi
if [ ! -f $MODELS/phase1/transport_classifier_best.pth ]; then
    echo "[ERROR] MLP asset not found at $MODELS/phase1/transport_classifier_best.pth"
    exit 1
fi
if [ ! -f $MODELS/phase1/transport_xgb_best.json ]; then
    echo "[ERROR] XGB asset not found at $MODELS/phase1/transport_xgb_best.json"
    exit 1
fi

# Detect available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "[GPU] Detected $NUM_GPUS GPU(s)"
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "[WARN] No GPU detected, will fall back to single-process"
fi

# Variant definitions: name|flag|logfile
VARIANTS=(
    "full_v3||val_full_v3"
    "no_phase1|--no_phase1|val_no_phase1"
    "no_s4pred|--no_s4pred|val_no_s4pred"
    "pure_mlp|--phase1_weights 1.0,0.0|val_pure_mlp"
    "pure_xgb|--phase1_weights 0.0,1.0|val_pure_xgb"
)

# Launch each variant on a round-robin GPU
i=0
for spec in "${VARIANTS[@]}"; do
    IFS='|' read -r name flag logdir <<< "$spec"
    VOUT=$OUT/$logdir
    mkdir -p $VOUT
    FINAL=$VOUT/final_report.csv
    if [ -f "$FINAL" ]; then
        echo "[$(date '+%H:%M:%S')] [Skip] $name (already done)"
        continue
    fi

    # Assign GPU: round-robin
    if [ "$NUM_GPUS" -gt 0 ]; then
        GPU_ID=$((i % NUM_GPUS))
    else
        GPU_ID=0
    fi
    i=$((i + 1))

    echo "[$(date '+%H:%M:%S')] [Launch] $name on GPU $GPU_ID  (flag: ${flag:-<default>})"
    MLP_ASSET=$MODELS/phase1/transport_classifier_best.pth \
    XGB_ASSET=$MODELS/phase1/transport_xgb_best.json \
    nohup python $SCRIPTS/predict_transport_family_v3.py \
        -in $BAL \
        -out $VOUT \
        --tcdb_annot $DATA/tcdb_annot.clean.tsv \
        --gpu $GPU_ID \
        --no_plot $flag \
        > $OUT/${logdir}.log 2>&1 &
done

echo ""
echo "[$(date '+%H:%M:%S')] All variants launched. PIDs:"
jobs -p
echo ""
echo "Monitor with:"
echo "  tail -f $OUT/val_full_v3.log"
echo "  ls $OUT/val_*/final_report.csv  (each appears when variant completes)"
