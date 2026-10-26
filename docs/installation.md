# Installation

This guide covers the complete setup of TPClass, including Python environment, GPU support, model downloads, and optional tools.

## Prerequisites

### Hardware

- **CPU**: Any modern x86_64 or ARM64 CPU
- **GPU**: NVIDIA GPU with ≥ 12 GB VRAM (recommended for ESM-2 3B inference; CPU works but ~50x slower)
- **RAM**: ≥ 16 GB (32 GB recommended for 575k-protein prediction)
- **Disk**: ≥ 50 GB free (10 GB ESM-2 3B + 2 GB ESM-2 650M + ~20 GB for embedding cache + outputs)

### Software

- Python 3.10+
- pip 23+
- (Optional) CUDA 11.8+ and cuDNN for GPU acceleration
- (Optional) DIAMOND (for ESM-2 vs BLAST comparison)
- (Optional) MMseqs2 (for fast sequence search)

## Quick install

```bash
# Clone the repository
git clone https://github.com/Chao-NNU/TPClass.git
cd TPClass

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[gpu]"  # includes faiss-gpu for fast retrieval
# or just: pip install -e .  # for CPU-only

# Verify installation
tpclass --help
```

## Detailed setup

### 1. Python environment

```bash
# Option A: venv
python -m venv venv
source venv/bin/activate

# Option B: conda
conda create -n tpclass python=3.10
conda activate tpclass

# Option C: mamba (faster)
mamba create -n tpclass python=3.10
mamba activate tpclass
```

### 2. PyTorch with CUDA

For GPU acceleration, install PyTorch matching your CUDA version:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only (slower)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### 3. TPClass and dependencies

```bash
# From source
git clone https://github.com/Chao-NNU/TPClass.git
cd TPClass
pip install -e ".[gpu]"  # GPU + faiss
# or: pip install -e .  # CPU only
# or: pip install -e ".[dev,gpu,tools]"  # development install
```

### 4. Model downloads

Models are NOT included in the git repo (too large). Download them:

```bash
# ESM-2 3B (~10 GB) and 650M (~2.5 GB) from HuggingFace
python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D', cache_dir='./models/esm2_3b')
AutoModel.from_pretrained('facebook/esm2_t36_3B_UR50D', cache_dir='./models/esm2_3b')
AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D', cache_dir='./models/esm2_650m')
AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D', cache_dir='./models/esm2_650m')
"

# TPClass pre-trained models (Phase 1 + Phase 2 checkpoints, ~5 GB)
# Download from the HuggingFace repository Chao-NNU/TPClass
bash models/download.sh
```

Or see [models/README.md](../models/README.md) for the full download guide.

### 5. Reference data (small files in repo)

The TCDB annotation table and example reference FASTAs are in the repo:
- `data/tcdb_annot.clean.tsv`
- `data/refs/hmo_refs_25.fasta`
- `data/refs/squalene_refs.fasta`
- `data/refs/isobutanol_refs.fasta`

If you cloned the repo, these are already there.

### 6. Optional: DIAMOND for ESM-2 vs BLAST comparison

```bash
# Linux (conda)
conda install -c bioconda diamond

# Linux (apt)
sudo apt install diamond-aligner

# macOS (Homebrew)
brew install diamond

# Windows
# Download from https://github.com/bbuchfink/diamond/releases
```

### 7. Optional: s4pred for secondary structure

```bash
pip install s4pred
```

If s4pred is unavailable, TPClass falls back to zero-features (slight accuracy reduction, but pipeline still works).

## Verify the install

```bash
# Check that all commands are available
which tpclass python diamond
tpclass --help

# Run a quick prediction on a tiny test set
tpclass predict \
    --fasta tests/fixtures/example.fasta \
    --out /tmp/test_predictions.csv

# Expected output: 2-3 transporters in 10 sequences
head /tmp/test_predictions.csv
```

## Troubleshooting

### CUDA out of memory

ESM-2 3B inference needs ~10 GB GPU memory. If you have less:
- Use the 650M backbone instead: `--backbone 650M`
- Or run on CPU: `--device cpu` (slower but works)
- Or reduce batch size in `predict_transport_family_v3.py`

### DIAMOND not found

```bash
# If you see "diamond: command not found"
which diamond
# If not in PATH, set --diamond /path/to/diamond
```

### s4pred import error

TPClass will warn and use zero-features. Install s4pred if you want full accuracy:
```bash
pip install s4pred
```

### Model loading slow first time

ESM-2 3B model is ~10 GB. First-time download takes 5-10 minutes depending on bandwidth. Subsequent runs use the cache.

## Next steps

- [Usage guide](usage.md) — How to run predictions
- [Case studies](../examples) — Case study walkthroughs
- [Usage](usage.md) — Python API
