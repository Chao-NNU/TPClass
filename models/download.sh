#!/bin/bash
# Download the TPClass model assets and reference data.
# Run from the project root: bash models/download.sh
#
# Everything is fetched from the HuggingFace repository Chao-NNU/TPClass.
# The ESM-2 backbones come from the official Facebook repositories.

set -e
cd "$(dirname "$0")/.."

HF_REPO="Chao-NNU/TPClass"

echo "=== Downloading TPClass assets ==="

if ! command -v huggingface-cli > /dev/null 2>&1; then
    echo "ERROR: huggingface-cli not found. Install it with:"
    echo "       pip install 'huggingface_hub[cli]'"
    exit 1
fi

# ---------------------------------------------------------------- ESM-2 3B
if [ ! -f "models/esm2_3b/model.safetensors" ] && [ ! -f "models/esm2_3b/pytorch_model.bin" ]; then
    echo "[1/5] ESM-2 t36 3B UR50D (~11 GB)..."
    mkdir -p models/esm2_3b
    huggingface-cli download facebook/esm2_t36_3B_UR50D --local-dir models/esm2_3b
else
    echo "[1/5] ESM-2 3B already present, skipping"
fi

# -------------------------------------------------------------- ESM-2 650M
if [ ! -f "models/esm2_650m/model.safetensors" ] && [ ! -f "models/esm2_650m/pytorch_model.bin" ]; then
    echo "[2/5] ESM-2 t33 650M UR50D (~2.5 GB)..."
    mkdir -p models/esm2_650m
    huggingface-cli download facebook/esm2_t33_650M_UR50D --local-dir models/esm2_650m
else
    echo "[2/5] ESM-2 650M already present, skipping"
fi

# ------------------------------------------------- TPClass 3B model bundle
if [ ! -f "models/v3_retrained_3b/phase1/transport_classifier_best.pth" ]; then
    echo "[3/5] TPClass 3B models and reference index..."
    mkdir -p models/v3_retrained_3b
    huggingface-cli download "$HF_REPO" \
        --include "models/v3_retrained_3b/*" \
        --local-dir . --local-dir-use-symlinks False
else
    echo "[3/5] TPClass 3B models already present, skipping"
fi

# ------------------------------------------------ TPClass 650M model bundle
if [ ! -f "models/v3_retrained/phase1/transport_classifier_best.pth" ]; then
    echo "[4/5] TPClass 650M models and reference index..."
    mkdir -p models/v3_retrained
    huggingface-cli download "$HF_REPO" \
        --include "models/v3_retrained/*" \
        --local-dir . --local-dir-use-symlinks False
else
    echo "[4/5] TPClass 650M models already present, skipping"
fi

# ------------------------------------------------------------ reference data
if [ ! -f "data/tcdb_annot.clean.tsv" ]; then
    echo "[5/5] TCDB annotation table and reference data..."
    mkdir -p data
    huggingface-cli download "$HF_REPO" \
        --include "data/*" \
        --local-dir . --local-dir-use-symlinks False
else
    echo "[5/5] TCDB annotation already present, skipping"
fi

echo ""
echo "=== All assets downloaded ==="
du -sh models/esm2_3b models/esm2_650m models/v3_retrained_3b models/v3_retrained data 2>/dev/null || true

echo ""
echo "Verify the installation:"
check() {
    if [ -f "$1" ]; then
        echo "  OK   $1"
    else
        echo "  MISS $1"
    fi
}
check models/esm2_3b/model.safetensors
check models/esm2_650m/model.safetensors
check models/v3_retrained_3b/phase1/transport_classifier_best.pth
check models/v3_retrained_3b/phase1/transport_xgb_best.json
check models/v3_retrained_3b/phase2/best_phase2.pth
check models/v3_retrained_3b/tcdb_embeddings.npy
check models/v3_retrained_3b/tcdb_index_metadata.csv
check models/v3_retrained/phase1/transport_classifier_best.pth
check data/tcdb_annot.clean.tsv

cat <<'EOF'

Note on the Phase 2 reference index
-----------------------------------
The Phase 2 head and its reference index MUST come from the same training
run.  The released script derives the reference paths from the checkpoint
directory for exactly this reason.

This script lays the bundle out as:

    models/v3_retrained_3b/phase2/best_phase2.pth
    models/v3_retrained_3b/tcdb_embeddings.npy
    models/v3_retrained_3b/tcdb_index_metadata.csv

`tpclass` looks for the index next to the checkpoint first and one directory
above it second, so both this layout and the released one
(`<checkpoint dir>/tcdb_embeddings.npy`) work without extra flags.  If you
move the assets around, pass --ref_emb_3b / --ref_emb_650m / --ref_metadata
explicitly.
EOF
