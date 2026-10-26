#!/bin/bash
# run_3b_12species.sh
# ============================================================
# 3B v3 inference on 12 species proteomes, 8-GPU parallel.
# Auto-pairs REF_EMB / REF_META with PHASE2_CKPT (3B ref).
# Outputs: outputs/3b_12species/<name>/{final_report.csv, phase1_report.tsv, phase2_report.csv}
# Logs   : outputs/3b_12species/<name>.log
# ============================================================
set -e
cd /data2/chao/transport

INPUT_DIR=/data2/chao/transport/inputs
OUT_ROOT=/data2/chao/transport/outputs/3b_12species
mkdir -p "$OUT_ROOT"

# 12 species (10 cross-species + E. coli BL21 case study + Streptomyces actuosus)
declare -A species=(
    ["A_niger"]="aniger.fasta"
    ["F_fujikuroi"]="fuji.fasta"
    ["B_cinerea"]="bcin.fasta"
    ["Y_lipolytica"]="yli.fasta"
    ["K_marxianus"]="max.fasta"
    ["S_cerevisiae_S288C"]="s288c.fasta"
    ["C_glutamicum"]="cgl.fasta"
    ["L_lactis"]="lacto.fasta"
    ["B_subtilis_168"]="bsub.fasta"
    ["E_coli_K12_MG1655"]="MG1655.fasta"
    ["E_coli_BL21"]="BL21.fasta"
    ["S_actuosus"]="sact.fasta"
)

# Old 650M v3 runs left per-species s4pred caches at outputs/<old_name>/ss_results/.
# Reuse them directly (cache key = fasta header, same fasta = same key = cache hit).
# Y_lipolytica + K_marxianus had no 650M run; fall back to fresh per-species dir.
declare -A old_s4pred_cache=(
    ["A_niger"]="/data2/chao/transport/outputs/aniger/ss_results"
    ["F_fujikuroi"]="/data2/chao/transport/outputs/fuji/ss_results"
    ["B_cinerea"]="/data2/chao/transport/outputs/bcin/ss_results"
    ["C_glutamicum"]="/data2/chao/transport/outputs/cgl/ss_results"
    ["L_lactis"]="/data2/chao/transport/outputs/lacto/ss_results"
    ["B_subtilis_168"]="/data2/chao/transport/outputs/bsub/ss_results"
    ["S_cerevisiae_S288C"]="/data2/chao/transport/outputs/s288c/ss_results"
    ["E_coli_K12_MG1655"]="/data2/chao/transport/outputs/MG1655/ss_results"
    ["E_coli_BL21"]="/data2/chao/transport/outputs/BL21/ss_results"
    ["S_actuosus"]="/data2/chao/transport/outputs/sact/ss_results"
)

# 3B ESM-2 + retrained assets
export ESM2_PATH=/data2/chao/transport/models/esm2_t36_3B_UR50D.pt
export PHASE1_FEAT_DIM=2563
export MLP_ASSET=/data2/chao/transport/models/v3_retrained_3b/phase1/transport_classifier_best.pth
export XGB_ASSET=/data2/chao/transport/models/v3_retrained_3b/phase1/transport_xgb_best.json
export PHASE2_CKPT=/data2/chao/transport/phase2_contrastive_3b/best_phase2.pth
# REF_EMB / REF_META auto-pair from PHASE2_CKPT parent (the 3B-trained reference)

N_GPU=8
idx=0
n_launched=0
for name in "${!species[@]}"; do
    fasta="$INPUT_DIR/${species[$name]}"
    if [ ! -f "$fasta" ]; then
        echo "[SKIP] $fasta not found"
        continue
    fi
    gpu=$((idx % N_GPU))
    out="$OUT_ROOT/$name"
    mkdir -p "$out"

    # Always use a per-species NEW cache dir under 3b_12species/.
    # If old 650M cache exists, copy its .horiz files in as a one-time seed
    # so v3 hits the cache and skips re-prediction.
    s4pred_dir="$out/ss_results"
    mkdir -p "$s4pred_dir"
    if [ -n "${old_s4pred_cache[$name]:-}" ] && [ -d "${old_s4pred_cache[$name]}" ]; then
        n_seeded=$(cp -n "${old_s4pred_cache[$name]}"/*.horiz "$s4pred_dir/" 2>/dev/null | wc -l)
        s4pred_tag="SEEDED from ${old_s4pred_cache[$name]} ($n_seeded files)"
    else
        s4pred_tag="FRESH (no old cache to seed)"
    fi

    echo "[GPU $gpu] $name  <- $fasta  -> $out  s4pred=$s4pred_tag  -> $s4pred_dir"
    CUDA_VISIBLE_DEVICES=$gpu nohup python /data2/chao/transport/scripts/predict_transport_family_v3.py \
        -in "$fasta" \
        -out "$out" \
        --tcdb_annot /data2/chao/transport/data/tcdb_annot.clean.tsv \
        --s4pred_outdir "$s4pred_dir" \
        --no_plot \
        > "$OUT_ROOT/$name.log" 2>&1 &
    n_launched=$((n_launched+1))
    idx=$((idx+1))
done

echo ""
echo "Launched $n_launched jobs across up to $N_GPU GPUs."
echo "Waiting for all to complete..."
echo "Monitor: ls $OUT_ROOT/*.log | xargs -I {} sh -c 'echo {}; tail -1 {}'"
wait
echo ""
echo "All done at $(date)"

# Summary
echo ""
echo "=== Per-species final_report row counts ==="
for d in "$OUT_ROOT"/*/; do
    name=$(basename "$d")
    csv="$d/final_report.csv"
    if [ -f "$csv" ]; then
        n=$(wc -l < "$csv")
        n_yes=$(awk -F',' 'NR>1 && $4=="YES" {n++} END {print n+0}' "$csv")
        echo "  $name : $((n-1)) total, $n_yes Phase1 YES"
    else
        echo "  $name : FAILED (no final_report.csv)"
    fi
done
