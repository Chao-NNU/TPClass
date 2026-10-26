#!/usr/bin/env bash
# ============================================================
# run_ablation.sh (v3 — parallel GPU dispatch)
# Run v3 ablation variants on 4 species across 2 least-loaded GPUs in parallel.
#
# v3 changes from v2:
#   - Auto-select 2 least-loaded GPUs via nvidia-smi
#   - Run 2 jobs in parallel (one per GPU)
#   - Wait for both to finish before launching next pair
#   - v3 --gpu flag controls per-job GPU pinning
#
# Usage (on gpu01):
#   bash /data2/chao/transport/scripts/run_ablation.sh
# ============================================================

SCRIPTS=/data2/chao/transport/scripts
ROOT=/data2/chao/transport
OUTROOT=$ROOT/outputs/ablation
TCDB_ANNOT=$ROOT/data/tcdb_annot.clean.tsv
PARALLEL=2  # number of parallel jobs (= number of GPUs to use)
mkdir -p "$OUTROOT"

# 4 species
SPECIES=("max" "s288c" "bsub" "MG1655")

# 5 ablation variants
VARIANTS=(
    "full_v3|"
    "no_phase1|--no_phase1"
    "no_s4pred|--no_s4pred"
    "pure_mlp|--phase1_weights 1.0,0.0"
    "pure_xgb|--phase1_weights 0.0,1.0"
)

# Find input fasta
find_fasta() {
    local sp=$1
    for dir in "$ROOT/inputs" "$ROOT/blastout"; do
        for ext in fasta fa faa; do
            local f="$dir/${sp}.${ext}"
            [ -f "$f" ] && { echo "$f"; return 0; }
        done
    done
    return 1
}

# Find existing s4pred cache
find_s4pred_cache() {
    local sp=$1
    local cache="$ROOT/outputs/${sp}/ss_results"
    if [ -d "$cache" ] && [ "$(ls -A "$cache" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "$cache"
    fi
}

find_gt() {
    local sp=$1
    local gt="$ROOT/data/uniprot_${sp}_transporters.tsv"
    if [ -f "$gt" ]; then echo "$gt"; else echo "$ROOT/data/uniprot_max_transporters.tsv"; fi
}

# Get N least-loaded GPU indices (most free memory first)
get_least_loaded_gpus() {
    local n=$1
    nvidia-smi --query-gpu=index,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' -v n="$n" '
    {
        idx = $1; used = $2; total = $3; free = total - used;
        printf "%d %d\n", idx, -free  # negative so default sort is descending free
    }' | sort -k2 -n | head -n "$n" | awk '{print $1}' | tr '\n' ' '
}

ts() { date '+%H:%M:%S'; }

# Build all jobs upfront
echo "[$(ts)] Building job list ..."
JOBS=()
for sp in "${SPECIES[@]}"; do
    IN=$(find_fasta "$sp")
    if [ -z "$IN" ]; then
        echo "  [SKIP] $sp (no fasta)"
        continue
    fi
    S4PRED_OUT=$(find_s4pred_cache "$sp")
    GT=$(find_gt "$sp")
    SP_OUT="$OUTROOT/$sp"
    mkdir -p "$SP_OUT"

    for v in "${VARIANTS[@]}"; do
        vname=$(echo "$v" | cut -d'|' -f1)
        vflag=$(echo "$v" | cut -d'|' -f2-)
        VOUT="$SP_OUT/$vname"
        mkdir -p "$VOUT"

        # Build job descriptor: species|vname|input|outdir|vflag|s4pred|gt
        JOBS+=("$sp|$vname|$IN|$VOUT|$vflag|$S4PRED_OUT|$GT")
    done
done

TOTAL=${#JOBS[@]}
echo "[$(ts)] Total jobs: $TOTAL (parallelism: $PARALLEL)"

# Run jobs in parallel
run_idx=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r sp vname IN VOUT vflag S4PRED_OUT GT <<< "$job"

    # Find an available GPU (refresh nvidia-smi each time)
    GPU_ID=$(get_least_loaded_gpus 1)
    if [ -z "$GPU_ID" ]; then
        echo "  [$(ts)] [WARN] No GPU detected, job $((run_idx+1)) will use CPU (slow)"
        GPU_ID=""
    else
        echo "  [$(ts)] [Job $((run_idx+1))/$TOTAL] $sp/$vname -> GPU $GPU_ID"
    fi

    # Launch in background
    (
        CMD="python $SCRIPTS/predict_transport_family_v3.py \
            -in $IN -out $VOUT --no_plot \
            --tcdb_annot $TCDB_ANNOT"
        if [ -n "$vflag" ]; then CMD="$CMD $vflag"; fi
        if [ -n "$S4PRED_OUT" ]; then CMD="$CMD --s4pred_outdir $S4PRED_OUT"; fi
        if [ -n "$GPU_ID" ]; then CMD="$CMD --gpu $GPU_ID"; fi
        echo "    $CMD" > "$VOUT/v3_cmd.log"
        eval "$CMD" > "$VOUT/v3_run.log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "  [$(ts)] [OK] $sp/$vname"
        else
            echo "  [$(ts)] [FAIL] $sp/$vname (exit $rc, see $VOUT/v3_run.log)"
        fi
    ) &

    run_idx=$((run_idx + 1))

    # Wait every PARALLEL jobs
    if [ $((run_idx % PARALLEL)) -eq 0 ]; then
        wait
    fi
done

# Wait for any remaining jobs
wait
echo "[$(ts)] All $TOTAL jobs finished"

# Per-species aggregate
for sp in "${SPECIES[@]}"; do
    SP_OUT="$OUTROOT/$sp"
    [ -d "$SP_OUT" ] || continue
    GT=$(find_gt "$sp")
    echo ""
    echo "[$(ts)] Aggregate $sp"
    VARIANT_ARGS=""
    for v in "${VARIANTS[@]}"; do
        vname=$(echo "$v" | cut -d'|' -f1)
        FR=$(find "$SP_OUT/$vname" -name "final_report.csv" -type f | head -1)
        [ -n "$FR" ] && VARIANT_ARGS="$VARIANT_ARGS \"$vname|$FR\""
    done
    if [ -n "$VARIANT_ARGS" ]; then
        eval "python $SCRIPTS/ablation_runner.py \
            --gt $GT --variants $VARIANT_ARGS --out_dir $SP_OUT" 2>&1 | tail -20
    fi
done

# Cross-species aggregate
echo ""
echo "[$(ts)] Cross-species aggregate"
python "$SCRIPTS/ablation_cross_species.py" \
    --ablation_root "$OUTROOT" \
    --out_dir "$OUTROOT/cross_species"

echo ""
echo "[$(ts)] [Done] All in $OUTROOT"
