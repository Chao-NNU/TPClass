#!/usr/bin/env bash
# select_idle_gpus.sh — find the N least-loaded GPUs and export CUDA_VISIBLE_DEVICES.
#
# Usage:
#   source scripts/select_idle_gpus.sh 2         # auto-select 2 GPUs
#   bash scripts/select_idle_gpus.sh 2          # same, in subshell
#   source scripts/select_idle_gpus.sh 2 4      # override nproc=2, num_gpu=4
#
# Then launch training:
#   torchrun --nproc_per_node="$NUM_GPU" ...

set -e

# Defaults
NPROC=${NPROC:-1}
NUM_GPU=${1:-2}
NUM_GPU=${2:-$NUM_GPU}   # second arg overrides; first arg also accepted

if [ "$#" -ge 1 ]; then
    NUM_GPU=$1
fi
if [ "$#" -ge 2 ]; then
    NPROC=$2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[select_idle_gpus] nvidia-smi not found, skipping auto-select"
    export NUM_GPU=0
    return 0 2>/dev/null || exit 0
fi

# Query all GPUs sorted by memory.used ascending
SELECTED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2 -n \
    | head -n "$NUM_GPU" \
    | cut -d',' -f1 \
    | paste -sd ',')

if [ -z "$SELECTED" ]; then
    echo "[select_idle_gpus] no GPUs found, falling back to default"
    return 0 2>/dev/null || exit 0
fi

# Print selection summary
echo "[select_idle_gpus] selected $NUM_GPU least-loaded GPU(s): $SELECTED"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F',' -v sel="$SELECTED" '
        BEGIN { n=split(sel, ids, ",") }
        { for (i=1; i<=n; i++) if ($1 == ids[i]) printf "  GPU %s: %s MiB / %s MiB, util %s%%\n", $1, $2, $3, $4 }
    '

export CUDA_VISIBLE_DEVICES="$SELECTED"
export NUM_GPU
export NPROC
