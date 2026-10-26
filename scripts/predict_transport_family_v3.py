#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined Phase 1 + Phase 2 prediction pipeline (v3 — predict + plot in one script).

= v2 + auto-plot, single script =

New in v3 vs v2:
  - Auto-generates 9 distribution plots + 1 summary into <output_dir>/plots/
  - --no_plot          : disable auto-plot
  - --plot_dir <path>  : custom plot output dir (default: <output_dir>/plots)
  - --with_sw          : enable Smith-Waterman alignment (default OFF, requires Biopython)

s4pred caching:
  - Cache is auto-managed via ss_results/<name>.horiz files (one file per sequence).
  - Deleting a .horiz file forces re-prediction of that sequence on next run.
  - No more growing .tsv cache file.

v2 confidence metrics (Phase 1/2 quality assessment):
  - top1_sim_gap       : gap between top-1 and top-2 similarity
  - top_family_votes   : votes for the winning family in top-k voting
  - top_family_purity  : votes / k
  - top1_seq_identity  : local sequence identity of top-1 hit (Smith-Waterman)
  - top1_qcov          : query coverage of top-1 hit
  - top1_aln_score     : raw SW alignment score

Usage:
    python predict_transport_family_v3.py -in input.fasta -out results/
    python predict_transport_family_v3.py -in input.fasta -out results/ --no_plot
    python predict_transport_family_v3.py -in input.fasta -out results/ --with_sw
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import re
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
import xgboost as xgb
import esm
import matplotlib
matplotlib.use("Agg")   # headless backend, no DISPLAY needed (must be before pyplot import)
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ── Optional: Biopython ─────────────────────────────────────────────────────
try:
    from Bio.Align import PairwiseAligner
    HAVE_BIOPYTHON = True
except ImportError:
    HAVE_BIOPYTHON = False

# ── CJK font fallback (so Chinese chars don't render as tofu boxes) ────────
for _cand in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei",
              "Microsoft YaHei", "PingFang SC", "DejaVu Sans"]:
    try:
        rcParams["font.sans-serif"] = [_cand] + rcParams["font.sans-serif"]
        break
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False

# ── Paths ────────────────────────────────────────────────────────────────────
# All paths support env-var override; defaults match the standard project layout.
import os as _os
_TRANSPORT_ROOT = _os.environ.get("TRANSPORT_ROOT", "/data2/chao/transport")

ESM2_PATH   = _os.environ.get("ESM2_PATH",
    "/data2/chao/transport/models/esm2_t36_3B_UR50D.pt")  # default ESM-2 3B (single .pt); override with 650M if needed
# Phase 2 (TC family) is a separate ESM-2 in the Cascade config: 3B Phase 1 + 650M Phase 2.
# Phase 2's ContrastiveEncoder hardcodes `repr_layers=[33]` so it must be 650M (33 layers), not 3B (36 layers).
# Default: same as ESM2_PATH (legacy single-model). For Cascade, set this to the 650M .pt.
PHASE2_ESM2_PATH = _os.environ.get("PHASE2_ESM2_PATH", ESM2_PATH)
MLP_ASSET   = _os.environ.get("MLP_ASSET",
    _os.path.join(_TRANSPORT_ROOT, "models", "v3_retrained", "phase1", "transport_classifier_best.pth"))
XGB_ASSET   = _os.environ.get("XGB_ASSET",
    _os.path.join(_TRANSPORT_ROOT, "models", "v3_retrained", "phase1", "transport_xgb_best.json"))
PHASE2_CKPT = Path(_os.environ.get("PHASE2_CKPT",
    _os.path.join(_TRANSPORT_ROOT, "phase2_contrastive", "best_phase2.pth")))

# REF_EMB / REF_META auto-pair with PHASE2_CKPT (same training run -> same metric space).
# Override via env var only if you really know what you're doing (e.g. legacy 650M ref).
if "REF_EMB" in _os.environ:
    REF_EMB = Path(_os.environ["REF_EMB"])
else:
    REF_EMB = PHASE2_CKPT.parent / "tcdb_embeddings.npy"
if "REF_META" in _os.environ:
    REF_META = Path(_os.environ["REF_META"])
else:
    REF_META = PHASE2_CKPT.parent / "tcdb_index_metadata.csv"

# Print actual paths at startup, so it's obvious if something is wrong
print(f"[Paths] TRANSPORT_ROOT = {_TRANSPORT_ROOT}")
print(f"[Paths] ESM2_PATH       = {ESM2_PATH}")
print(f"[Paths] PHASE2_ESM2_PATH = {PHASE2_ESM2_PATH}")
print(f"[Paths] MLP_ASSET   = {MLP_ASSET}")
print(f"[Paths] XGB_ASSET   = {XGB_ASSET}")
print(f"[Paths] PHASE2_CKPT = {PHASE2_CKPT}")
print(f"[Paths] REF_EMB     = {REF_EMB}")
print(f"[Paths] REF_META    = {REF_META}")

# Sanity: ref embeddings must come from the SAME training run as the Phase 2 ckpt,
# otherwise the query (projected by PHASE2_CKPT) and the ref (projected by a different
# projector) live in different metric spaces -> cosine sim meaningless (around 0.15).
if _os.path.normpath(str(REF_EMB.parent)) != _os.path.normpath(str(PHASE2_CKPT.parent)):
    print(f"[WARN] REF_EMB parent ({REF_EMB.parent}) != PHASE2_CKPT parent ({PHASE2_CKPT.parent})")
    print(f"[WARN] ref embeddings likely live in a different metric space than the projector (e.g. 3B vs 650M).")
    print(f"[WARN] Unset REF_EMB / REF_META env vars to auto-pair with PHASE2_CKPT.")

MAX_LEN         = 800
PHASE1_FEAT_DIM = int(_os.environ.get("PHASE1_FEAT_DIM", "1283"))  # 1283 for ESM-2 650M, 2563 for ESM-2 3B
EMBED_DIM       = 512
HIDDEN          = 1280
UNFREEZE_LAYERS = 6
ENCODE_BATCH    = 8
VALID_AA        = set("ACDEFGHIKLMNPQRSTVWXY")

TC_CLASS_NAME = {
    "1": "Channels/Pores",
    "2": "Electrochemical Potential-driven",
    "3": "Primary Active Transporters",
    "4": "Group Translocators",
    "5": "Transmembrane Electron Carriers",
    "8": "Accessory Factors",
    "9": "Incompletely Characterized",
}


# ════════════════════════════════════════════════════════════════════════════
#  Model definitions (Phase 1 MLP + Phase 2 Contrastive)
# ════════════════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        return x + self.dropout(self.relu(self.bn(self.fc(x))))


class TransportClassifier(nn.Module):
    def __init__(self, input_dim=PHASE1_FEAT_DIM):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3))
        self.res_block1 = ResidualBlock(512)
        self.res_block2 = ResidualBlock(512)
        self.output_layer = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        return self.output_layer(self.res_block2(self.res_block1(self.input_layer(x))))


class ContrastiveEncoder(nn.Module):
    def __init__(self, esm_model, embed_dim=EMBED_DIM, hidden=None, unfreeze_n=UNFREEZE_LAYERS):
        # Auto-detect `hidden` from ESM-2's actual embed dim (1280 for 650M, 2560 for 3B)
        if hidden is None:
            hidden = esm_model.embed_dim
        super().__init__()
        self.esm = esm_model
        for p in self.esm.parameters():
            p.requires_grad = False
        for layer in self.esm.layers[-unfreeze_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.esm.emb_layer_norm_after.parameters():
            p.requires_grad = True
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim), nn.LayerNorm(embed_dim),
        )

    def _pool(self, tok_emb, input_ids):
        pad_idx = 1
        mask = (input_ids != pad_idx)
        mask[:, 0] = False
        mf = mask.unsqueeze(-1).float()
        mean = (tok_emb * mf).sum(1) / mf.sum(1).clamp(min=1)
        cls = tok_emb[:, 0, :]
        return torch.cat([cls, mean], dim=-1)

    def forward(self, input_ids):
        out = self.esm(input_ids, repr_layers=[33], return_contacts=False)
        pooled = self._pool(out["representations"][33], input_ids)
        z = self.proj(pooled)
        return F.normalize(z, dim=-1)


# ════════════════════════════════════════════════════════════════════════════
#  Basic IO
# ════════════════════════════════════════════════════════════════════════════
# UniProt header pattern: >sp|P71447|PGMB_LACLA ... or >tr|A0A123|...
# Also handles isoform: >sp|P71447-2|...
_UNIPROT_HEADER_RE = re.compile(r"^>(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|")


def _extract_header_name(header: str, idx: int) -> str:
    """Extract a clean accession from a FASTA header.

    Priority:
      1. UniProt format: >sp|P71447|... -> 'P71447'
      2. Fallback: first whitespace-delimited token
      3. Last resort: seq_<idx>
    """
    m = _UNIPROT_HEADER_RE.match(header)
    if m:
        return m.group(1)
    # Fallback: first whitespace-delimited token (handles custom >XXX ... headers)
    return header[1:].split()[0] if len(header) > 1 else f"seq_{idx}"


def read_fasta(path):
    seqs, name, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line: continue
            if line.startswith(">"):
                if name is not None:
                    seqs.append((name, "".join(buf).upper()))
                name = _extract_header_name(line, len(seqs))
                buf = []
            else:
                buf.append(line)
    if name is not None:
        seqs.append((name, "".join(buf).upper()))
    return seqs


def tokenize(seq, alphabet, max_len=MAX_LEN):
    seq = seq[: max_len - 2]
    tokens = [alphabet.cls_idx]
    for aa in seq:
        t = alphabet.get_idx(aa)
        tokens.append(t if t != alphabet.unk_idx else alphabet.get_idx("X"))
    tokens.append(alphabet.eos_idx)
    pad = max_len - len(tokens)
    tokens = tokens + [alphabet.padding_idx] * pad
    return tokens[:max_len]


def is_valid(seq, min_len=30, max_len=5000):
    if not (min_len <= len(seq) <= max_len): return False
    bad = sum(c not in VALID_AA for c in seq) / max(1, len(seq))
    return bad <= 0.05


def smith_waterman_stats(query: str, target: str) -> tuple[float, float, float]:
    """Return (identity, qcov, aln_score)."""
    if not HAVE_BIOPYTHON or not query or not target:
        return (np.nan, np.nan, np.nan)
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = None
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    try:
        alignments = aligner.align(query, target)
        best = alignments[0]
        q_blocks = best.aligned[0]
        t_blocks = best.aligned[1]
        matched = 0
        q_covered = 0
        for (qs, qe), (ts, te) in zip(q_blocks, t_blocks):
            span = min(qe - qs, te - ts)
            q_covered += span
            for k in range(span):
                if query[qs + k] == target[ts + k]:
                    matched += 1
        qcov = q_covered / max(1, len(query))
        identity = matched / max(1, q_covered) if q_covered else 0.0
        score = float(best.score)
        return (identity, qcov, score)
    except Exception:
        return (np.nan, np.nan, np.nan)


# ════════════════════════════════════════════════════════════════════════════
#  s4pred secondary structure prediction (incremental cache + persistent)
# ════════════════════════════════════════════════════════════════════════════
def parse_s4pred_horiz(filepath):
    pred_chars = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Pred:"):
                pred_chars.extend(line[5:].strip().upper())
    if not pred_chars: return None
    n = len(pred_chars)
    h = sum(1 for c in pred_chars if c == 'H') / n
    e = sum(1 for c in pred_chars if c == 'E') / n
    c = sum(1 for c in pred_chars if c == 'C') / n
    return np.array([h, e, c], dtype=np.float32)


def _scan_horiz_cache(outdir, query_names):
    """Scan ss_results/ for existing .horiz files. Returns {name: [h,e,c]}.

    This replaces the old .tsv cache: each .horiz file is self-contained,
    parsed on demand. No single growing file, no sync issues, easy to delete
    per-sequence to force re-prediction.

    Cache lookup normalizes pos__/neg__ prefixes (added by extract_features_for_retrain
    balanced fasta prep) so that the same .horiz files can be reused across runs
    that use different header conventions.
    """
    if not outdir or not query_names:
        return {}
    outdir = Path(outdir)
    if not outdir.exists():
        return {}
    cache = {}
    for name in query_names:
        # Strip pos__/neg__ prefix for cache lookup (re-extract by raw header)
        cache_name = name
        if cache_name.startswith(("pos__", "neg__")) and "__" in cache_name:
            cache_name = cache_name.split("__", 1)[1]
        horiz_path = outdir / f"{cache_name}.horiz"
        if horiz_path.exists():
            try:
                feat = parse_s4pred_horiz(str(horiz_path))
                if feat is not None:
                    cache[name] = feat
            except Exception:
                continue
    return cache


def _load_legacy_s4pred_cache(path):
    """Backwards-compat: read old-style .tsv cache (one-time migration)."""
    if not path or not os.path.exists(path): return {}
    cache = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) >= 4:
                try:
                    cache[parts[0]] = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=np.float32)
                except ValueError:
                    continue
    return cache


def _run_s4pred_subset(seqs, s4pred_dir, outdir=None):
    """Run s4pred on a batch of (name, seq). Returns {name: [h,e,c]} or None."""
    if not seqs: return {}
    if outdir:
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)
        work_dir = outdir
        input_fa = os.path.abspath(os.path.join(outdir, "input.faa"))
        persistent = True
    else:
        tmp_fh = tempfile.NamedTemporaryFile(mode="w", suffix=".faa", delete=False)
        input_fa = tmp_fh.name
        tmp_fh.close()
        work_dir = tempfile.mkdtemp(prefix="s4pred_")
        persistent = False

    with open(input_fa, "w") as f:
        for name, seq in seqs:
            f.write(f">{name}\n{seq}\n")

    run_script = os.path.join(s4pred_dir, "run_model.py")
    cmd = (f"cd {s4pred_dir} && "
           f"python {run_script} --device gpu --outfmt horiz "
           f"--save-files --outdir {work_dir} {input_fa}")
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=7200)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[s4pred] FAILED: {e}")
        if not persistent:
            os.unlink(input_fa)
            import shutil; shutil.rmtree(work_dir)
        return None

    feats = {}
    for fname in os.listdir(work_dir):
        fpath = os.path.join(work_dir, fname)
        if not os.path.isfile(fpath): continue
        if os.path.abspath(fpath) == os.path.abspath(input_fa): continue
        if not fname.endswith(".horiz"): continue
        qname = os.path.splitext(fname)[0]
        feat = parse_s4pred_horiz(fpath)
        if feat is not None:
            feats[qname] = feat

    if not persistent:
        os.unlink(input_fa)
        import shutil; shutil.rmtree(work_dir)
    return feats


def run_s4pred(seqs, s4pred_dir="/data2/chao/s4pred", outdir=None,
               legacy_cache=None, force=False):
    """Incremental s4pred using .horiz files in outdir as the cache.

    Cache lookup order:
      1. ss_results/<name>.horiz files (preferred, self-managing)
      2. legacy_cache (.tsv) if explicitly provided (one-time migration path)

    A query is "cached" iff parse_s4pred_horiz() returns a valid result from
    its corresponding .horiz file. Deleting a .horiz file forces re-prediction
    of that sequence on next run.
    """
    seq_dict = {n: s for n, s in seqs}
    query_names = list(seq_dict.keys())

    # Force: ignore all caches, predict everything
    if force:
        print(f"[s4pred] --force: ignore cache, re-predict all {len(query_names)} sequences")
        cache = {}
        missing = query_names
    else:
        # Primary: scan .horiz files
        cache = _scan_horiz_cache(outdir, query_names)
        if cache:
            print(f"[s4pred] Found {len(cache)} cached predictions in {outdir}/")

        # Fallback: legacy .tsv cache (for migration)
        if legacy_cache and os.path.exists(legacy_cache):
            legacy = _load_legacy_s4pred_cache(legacy_cache)
            added = 0
            for name, feat in legacy.items():
                if name not in cache and name in seq_dict:
                    cache[name] = feat
                    added += 1
            if added:
                print(f"[s4pred] Loaded {added} additional entries from legacy cache "
                      f"{legacy_cache}")

        missing = [n for n in query_names if n not in cache]
        print(f"[s4pred] Cache hit: {len(cache)}/{len(query_names)}, "
              f"need to predict: {len(missing)}")

    if not missing:
        print(f"[s4pred] All {len(query_names)} queries cached, skipping s4pred run")
        return cache

    if s4pred_dir is None or not os.path.isdir(s4pred_dir):
        print(f"[s4pred] s4pred dir not available, mocking {len(missing)} with [0.5, 0.2, 0.3]")
        new_preds = {n: np.array([0.5, 0.2, 0.3], dtype=np.float32) for n in missing}
    else:
        print(f"[s4pred] Running s4pred on {len(missing)} sequences ...")
        result = _run_s4pred_subset(
            [(n, seq_dict[n]) for n in missing],
            s4pred_dir, outdir=outdir)
        if result is None:
            print(f"[s4pred] s4pred failed, mocking with [0.5, 0.2, 0.3]")
            new_preds = {n: np.array([0.5, 0.2, 0.3], dtype=np.float32) for n in missing}
        else:
            new_preds = result
            print(f"[s4pred] s4pred returned {len(new_preds)} new predictions")
            for n in missing:
                if n not in new_preds:
                    new_preds[n] = np.array([0.5, 0.2, 0.3], dtype=np.float32)

    cache.update(new_preds)
    # No .tsv cache to update; .horiz files are written directly by s4pred itself
    return cache


# ════════════════════════════════════════════════════════════════════════════
#  TCDB annotation + SW lookup
# ════════════════════════════════════════════════════════════════════════════
def load_tcdb_annotations(annot_file):
    if not annot_file or not os.path.exists(annot_file): return {}
    annot = {}
    with open(annot_file) as f:
        first = True
        for line in f:
            line = line.strip()
            if not line: continue
            sep = "\t" if "\t" in line else ","
            parts = [p.strip().strip('"') for p in line.split(sep)]
            if first:
                first = False
                if not parts or not re.match(r"^\d\.[A-Z]\.\d", parts[0]): continue
            if len(parts) >= 2:
                tc, desc = parts[0], parts[1]
                if re.match(r"^\d\.[A-Z]\.\d", tc): annot[tc] = desc
    return annot


def fetch_tcdb_online(tc_number, timeout=15):
    try:
        import requests
        from bs4 import BeautifulSoup
        url = f"https://www.tcdb.org/search/result?query={tc_number}"
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if not resp.ok: return None
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("Description"):
                return line.split(":", 1)[1].strip()[:200]
        return None
    except Exception:
        return None


def parse_tcdb_fasta(fasta_path, output_annot_file=None):
    annots = {}
    skipped = 0
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"): continue
            header = line[1:].strip()
            parts = header.split("|")
            if len(parts) < 4:
                skipped += 1; continue
            tc_part = parts[3].strip()
            m = re.match(r"^(\d+\.[A-Z]+\.\d+\.\d+(?:\.\d+)?)", tc_part)
            if not m:
                skipped += 1; continue
            tc = m.group(1)
            desc = tc_part[len(tc):].strip() or "(no description)"
            if tc not in annots: annots[tc] = desc
    if output_annot_file:
        with open(output_annot_file, "w") as f:
            f.write("# tc_number\tdescription\n")
            for tc, desc in sorted(annots.items()):
                f.write(f"{tc}\t{desc}\n")
    return annots


def load_tcdb_protein_seqs(fasta_path):
    if not fasta_path or not os.path.exists(fasta_path):
        return {}, {}
    print(f"[TCDB-SEQ] Loading reference protein sequences: {fasta_path}")
    seq_by_header, seq_by_tc = {}, {}
    header, buf = None, []
    tc_re = re.compile(r"(\d+\.[A-Z]+\.\d+\.\d+(?:\.\d+)?)")

    def _flush():
        nonlocal header
        if header is None: return
        seq = "".join(buf).upper()
        seq_by_header[header] = seq
        m = tc_re.search(header)
        if m:
            tc = m.group(1)
            if tc not in seq_by_tc:
                seq_by_tc[tc] = seq
        header = None
        buf.clear()

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if not line: continue
            if line.startswith(">"):
                _flush()
                header = line[1:].strip()
            else:
                buf.append(line)
        _flush()

    print(f"[TCDB-SEQ] Loaded {len(seq_by_header)} reference sequences (by header), "
          f"{len(seq_by_tc)} unique TC numbers")
    return seq_by_header, seq_by_tc


def _auto_detect_id_column(ref_meta, seq_by_header):
    if not seq_by_header: return None
    header_set = set(seq_by_header.keys())
    candidates = ["ref_id", "accession", "uniprot_id", "uniprot",
                  "tcdb_id", "header", "seq_id", "name", "id",
                  "pdb_id", "protein_id", "entry", "entry_name"]
    cols = list(candidates) + [c for c in ref_meta.columns if c not in candidates]
    for col in cols:
        if col not in ref_meta.columns: continue
        vals = ref_meta[col].astype(str).tolist()
        exact_hit = sum(1 for v in vals if v in header_set)
        if len(vals) and exact_hit / len(vals) > 0.5:
            print(f"[TCDB-SEQ]   column '{col}' exact-match: {exact_hit}/{len(vals)}")
            return col
        sub_hit = sum(1 for v in vals if any(v in h for h in header_set))
        if len(vals) and sub_hit / len(vals) > 0.5:
            print(f"[TCDB-SEQ]   column '{col}' substring-match: {sub_hit}/{len(vals)}")
            return col
    return None


def build_ref_seq_lookup(ref_meta, args=None):
    """Build ref sequence lookup. Prefer ref_meta['sequence'] column, fallback to FASTA."""
    stats = {"source": None, "hit": 0, "miss": 0, "id_col": None}

    seq_candidates = ["sequence", "seq", "protein_seq", "ref_seq"]
    seq_col = next((c for c in seq_candidates if c in ref_meta.columns), None)

    if seq_col is not None:
        non_empty = ref_meta[seq_col].notna() & ref_meta[seq_col].astype(str).str.strip().astype(bool)
        n_non_empty = int(non_empty.sum())
        print(f"[TCDB-SEQ] OK ref_meta contains '{seq_col}' column "
              f"({n_non_empty}/{len(ref_meta)} non-empty), using directly")
        lookup = {}
        for row_idx, row in ref_meta.iterrows():
            seq = row[seq_col]
            if isinstance(seq, str) and seq.strip():
                lookup[int(row_idx)] = seq.strip().upper()
        stats.update({"source": f"ref_meta['{seq_col}']",
                      "hit": len(lookup), "miss": len(ref_meta) - len(lookup)})
        print(f"[TCDB-SEQ] Lookup built: hit={len(lookup)}  miss={stats['miss']}")
        return lookup, stats

    if args is None or not getattr(args, "tcdb_fasta", None):
        print(f"[TCDB-SEQ] WARNING: ref_meta has no 'sequence' column and --tcdb_fasta "
              f"not provided, skipping L2 SW alignment")
        return {}, stats

    print(f"[TCDB-SEQ] ref_meta has no 'sequence' column, falling back to FASTA path")
    seq_by_header, seq_by_tc = load_tcdb_protein_seqs(args.tcdb_fasta)
    if not seq_by_header: return {}, stats

    id_col = getattr(args, "ref_id_col", None)
    if id_col is None:
        id_col = _auto_detect_id_column(ref_meta, seq_by_header)
    if id_col is None:
        print(f"[TCDB-SEQ] WARNING: cannot auto-detect ID column, falling back to TC number "
              f"indexing (may be inaccurate)")
        stats["id_col"] = "(TC fallback)"
    else:
        print(f"[TCDB-SEQ] Using ref_meta column '{id_col}' as ref_id")
        stats["id_col"] = id_col

    header_set = set(seq_by_header.keys())
    lookup = {}
    exact = substring = tc_fb = miss = 0
    for row_idx, row in ref_meta.iterrows():
        row_idx = int(row_idx)
        candidate = None
        if id_col and id_col in ref_meta.columns:
            ref_id = str(row[id_col])
            if ref_id in header_set:
                candidate = seq_by_header[ref_id]; exact += 1
            else:
                hit = next((h for h in header_set if ref_id in h), None)
                if hit: candidate = seq_by_header[hit]; substring += 1
        else:
            tc = str(row.get("tc_number", ""))
            if tc in seq_by_tc: candidate = seq_by_tc[tc]; tc_fb += 1
        if candidate: lookup[row_idx] = candidate
        else: miss += 1

    stats.update({"source": "FASTA", "hit": len(lookup), "miss": miss,
                  "id_col": id_col or "(TC fallback)",
                  "exact_hit": exact, "substring_hit": substring, "tc_fallback": tc_fb})
    print(f"[TCDB-SEQ] Lookup built (FASTA): exact={exact} sub={substring} "
          f"tc_fb={tc_fb} miss={miss} (total {len(ref_meta)})")
    return lookup, stats


# ════════════════════════════════════════════════════════════════════════════
#  Auto-plot (from plot_distributions.py)
# ════════════════════════════════════════════════════════════════════════════
def _tc_class(tc):
    if not isinstance(tc, str) or not tc: return ""
    return tc.split(".")[0].strip()


def _tc_subclass(tc):
    if not isinstance(tc, str) or not tc: return ""
    parts = tc.split(".")
    return ".".join(parts[:2]).strip() if len(parts) >= 2 else parts[0].strip()


def _save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tc_class(df, out):
    counts = df["tc_family"].dropna().map(_tc_class).value_counts()
    # Filter out empty/placeholder classes ("?", "", None)
    counts = counts[counts.index.isin(TC_CLASS_NAME.keys())].sort_values()
    if counts.empty: return
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.7 * len(counts) + 1.5)))
    labels = [f"Class {c}" for c in counts.index]
    bars = ax.barh(labels, counts.values, color="#4C72B0", edgecolor="black", linewidth=0.6)
    for bar, c, v in zip(bars, counts.index, counts.values):
        pct = 100 * v / counts.sum()
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {v}  ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"TC Class Distribution (n={counts.sum()})")
    legend_elements = [Patch(facecolor="white", edgecolor="white",
                             label=f"Class {c}: {TC_CLASS_NAME[c]}")
                       for c in counts.index]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=8, framealpha=0.95, title="TC Class name")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, counts.max() * 1.18)
    _save_fig(fig, out / "01_tc_class_distribution.png")


def _plot_tc_subclass(df, out, top=15):
    subs = df["tc_family"].dropna().map(_tc_subclass).value_counts()
    if subs.empty: return
    main = subs.head(top)
    other = subs.iloc[top:].sum()
    if other > 0: main["Other"] = other
    main = main.sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(main) + 1.5)))
    bars = ax.barh(main.index, main.values, color="#55A868", edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, main.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {v}", va="center", fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"TC Subclass Distribution (Top {top}, n={subs.sum()})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, main.max() * 1.12)
    _save_fig(fig, out / "02_tc_subclass_distribution.png")


def _plot_top_families(df, out, top=20):
    fam = df["tc_family"].dropna().value_counts().head(top).sort_values()
    if fam.empty: return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(fam) + 1.5)))
    bars = ax.barh(fam.index, fam.values, color="#C44E52", edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, fam.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {v}", va="center", fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"Top {top} TC Families")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, fam.max() * 1.12)
    _save_fig(fig, out / "03_top20_tc_families.png")


def _plot_similarity_hist(df, out, cutoffs=(0.5, 0.7, 0.8, 0.9)):
    sim = df.loc[df["is_transport"] == "YES", "top1_similarity"].dropna()
    if sim.empty: return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(sim, bins=30, color="#8172B2", edgecolor="black", linewidth=0.5)
    for c in cutoffs:
        n = (sim >= c).sum()
        ax.axvline(c, linestyle="--", linewidth=1)
        ax.text(c, ax.get_ylim()[1] * 0.95, f"≥{c}\n{n}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.set_xlabel("Top-1 similarity"); ax.set_ylabel("Count")
    ax.set_title(f"Top-1 Similarity (n={len(sim)}, mean={sim.mean():.3f})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save_fig(fig, out / "04_top1_similarity_hist.png")


def _plot_similarity_cdf(df, out, cutoffs=(0.5, 0.7, 0.8, 0.9)):
    sim = np.sort(df.loc[df["is_transport"] == "YES", "top1_similarity"].dropna().values)
    if len(sim) == 0: return
    cdf = np.arange(1, len(sim) + 1) / len(sim)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sim, cdf, color="#4C72B0", linewidth=2)
    for c in cutoffs:
        if sim.min() <= c <= sim.max():
            n = (sim >= c).sum()
            ax.axvline(c, linestyle="--", color="grey", linewidth=1)
            ax.axhline(n / len(sim), linestyle="--", color="grey", linewidth=1)
            ax.scatter([c], [n / len(sim)], s=60, zorder=5, color="red")
            ax.annotate(f"≥{c}: keep {n} ({100*n/len(sim):.1f}%)",
                        xy=(c, n / len(sim)),
                        xytext=(c + 0.05, max(0, n / len(sim) - 0.12)),
                        fontsize=9, arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_xlabel("Top-1 similarity threshold")
    ax.set_ylabel("Fraction kept (CDF)")
    ax.set_title("Similarity CDF — Pick Your Cutoff")
    ax.grid(linestyle="--", alpha=0.4); ax.set_ylim(0, 1.05)
    _save_fig(fig, out / "05_similarity_cdf.png")


def _plot_transport_prob(df, out):
    yes_p = df.loc[df["is_transport"] == "YES", "transport_prob"].dropna()
    no_p  = df.loc[df["is_transport"] == "NO",  "transport_prob"].dropna()
    if yes_p.empty and no_p.empty: return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(no_p,  bins=bins, alpha=0.55, label=f"NO  (n={len(no_p)})",
            color="#999999", edgecolor="black", linewidth=0.3)
    ax.hist(yes_p, bins=bins, alpha=0.65, label=f"YES (n={len(yes_p)})",
            color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set_xlabel("transport_prob (Phase 1 ensemble)")
    ax.set_ylabel("Count")
    ax.set_title("Phase 1 Probability — Actual Class")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save_fig(fig, out / "06_transport_prob_phase1.png")


def _plot_seq_length(df, out):
    yes_l = df.loc[df["is_transport"] == "YES", "seq_length"].dropna()
    no_l  = df.loc[df["is_transport"] == "NO",  "seq_length"].dropna()
    if yes_l.empty and no_l.empty: return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(yes_l.max(), no_l.max())
                       if len(yes_l) + len(no_l) else 1, 50)
    ax.hist(no_l,  bins=bins, alpha=0.55,
            label=f"NO  (n={len(no_l)},  median={no_l.median():.0f})",
            color="#999999", edgecolor="black", linewidth=0.3)
    ax.hist(yes_l, bins=bins, alpha=0.65,
            label=f"YES (n={len(yes_l)}, median={yes_l.median():.0f})",
            color="#55A868", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Sequence length (aa)"); ax.set_ylabel("Count")
    ax.set_title("Sequence Length — Actual Class")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save_fig(fig, out / "07_seq_length.png")


def _plot_identity(df, out):
    if "top1_seq_identity" not in df.columns: return
    ident = df.loc[(df["is_transport"] == "YES") & df["top1_seq_identity"].notna(),
                   "top1_seq_identity"]
    if ident.empty: return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(ident, bins=30, color="#937860", edgecolor="black", linewidth=0.5)
    for c in (0.2, 0.3, 0.5):
        n = (ident >= c).sum()
        ax.axvline(c, linestyle="--", linewidth=1)
        ax.text(c, ax.get_ylim()[1] * 0.95, f"≥{c}\n{n}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.set_xlabel("Top-1 sequence identity (Smith-Waterman)")
    ax.set_ylabel("Count")
    ax.set_title(f"Top-1 Sequence Identity (n={len(ident)}, median={ident.median():.2f})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save_fig(fig, out / "08_top1_identity.png")


def _plot_qcov(df, out):
    if "top1_qcov" not in df.columns: return
    qcov = df.loc[(df["is_transport"] == "YES") & df["top1_qcov"].notna(),
                  "top1_qcov"]
    if qcov.empty: return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(qcov, bins=30, color="#DA8BC3", edgecolor="black", linewidth=0.5)
    for c in (0.3, 0.5, 0.8):
        n = (qcov >= c).sum()
        ax.axvline(c, linestyle="--", linewidth=1)
        ax.text(c, ax.get_ylim()[1] * 0.95, f"≥{c}\n{n}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.set_xlabel("Top-1 query coverage (Smith-Waterman)")
    ax.set_ylabel("Count")
    ax.set_title(f"Top-1 Query Coverage (n={len(qcov)}, median={qcov.median():.2f})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save_fig(fig, out / "09_top1_qcov.png")


def _plot_dashboard(df, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    counts = df["tc_family"].dropna().map(_tc_class).value_counts()
    counts = counts[counts.index.isin(TC_CLASS_NAME.keys())].sort_values()
    if not counts.empty:
        labels = [f"Class {c}" for c in counts.index]
        axes[0, 0].barh(labels, counts.values, color="#4C72B0", edgecolor="black", linewidth=0.4)
        axes[0, 0].set_title("TC Class")
        axes[0, 0].grid(axis="x", linestyle="--", alpha=0.4)
        for i, v in enumerate(counts.values):
            axes[0, 0].text(v, i, f" {v}", va="center", fontsize=8)

    sim = df.loc[df["is_transport"] == "YES", "top1_similarity"].dropna()
    if not sim.empty:
        axes[0, 1].hist(sim, bins=25, color="#8172B2", edgecolor="black", linewidth=0.4)
        axes[0, 1].axvline(0.7, color="red", linestyle="--", label="cutoff=0.7")
        axes[0, 1].set_title("Top-1 Similarity")
        axes[0, 1].legend(); axes[0, 1].grid(axis="y", linestyle="--", alpha=0.4)

    for label, color in [("YES", "#4C72B0"), ("NO", "#999999")]:
        p = df.loc[df["is_transport"] == label, "transport_prob"].dropna()
        if not p.empty:
            axes[1, 0].hist(p, bins=30, alpha=0.55, label=label,
                            color=color, edgecolor="black", linewidth=0.3)
    axes[1, 0].axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
    axes[1, 0].set_title("Phase 1 Probability")
    axes[1, 0].legend(); axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)

    fam = df["tc_family"].dropna().value_counts().head(10).sort_values()
    if not fam.empty:
        axes[1, 1].barh(fam.index, fam.values, color="#C44E52", edgecolor="black", linewidth=0.4)
        axes[1, 1].set_title("Top 10 TC Families")
        axes[1, 1].grid(axis="x", linestyle="--", alpha=0.4)
        for i, v in enumerate(fam.values):
            axes[1, 1].text(v, i, f" {v}", va="center", fontsize=8)

    fig.suptitle(f"Transport Classification Dashboard — n={len(df)}",
                 fontsize=14, fontweight="bold")
    _save_fig(fig, out / "99_dashboard.png")


def _write_summary(df, out):
    yes = df[df["is_transport"] == "YES"]
    sim = yes["top1_similarity"]
    lines = [
        "=" * 60, "  Distribution Summary", "=" * 60,
        f"  Total rows                : {len(df)}",
        f"  Predicted transporters    : {len(yes)}  ({100*len(yes)/len(df):.1f}%)",
        "", "  -- TC class distribution --",
    ]
    cls = yes["tc_family"].dropna().map(_tc_class).value_counts()
    cls = cls[cls.index.isin(TC_CLASS_NAME.keys())].sort_index()
    for k, v in cls.items():
        pct = 100 * v / cls.sum() if cls.sum() else 0
        lines.append(f"    Class {k:<3} ({TC_CLASS_NAME[k]:<32})  "
                     f"{v:>5}  ({pct:.1f}%)")
    lines += ["", "  -- Top-1 similarity --"]
    if not sim.empty:
        lines.append(f"    mean={sim.mean():.4f}   median={sim.median():.4f}   "
                     f"std={sim.std():.4f}")
        for c in (0.5, 0.7, 0.8, 0.9, 0.95):
            lines.append(f"    ≥{c:.2f}  ->  {(sim >= c).sum():>5}  "
                         f"({(sim >= c).mean()*100:5.1f}%)")
    if "top1_seq_identity" in df.columns:
        ident = yes["top1_seq_identity"].dropna()
        if not ident.empty:
            lines += ["", "  -- Top-1 sequence identity (SW) --",
                      f"    mean={ident.mean():.4f}   median={ident.median():.4f}"]
            for c in (0.2, 0.3, 0.5, 0.7):
                lines.append(f"    ≥{c:.2f}  ->  {(ident >= c).sum():>5}  "
                             f"({(ident >= c).mean()*100:5.1f}%)")
    if "top1_qcov" in df.columns:
        qcov = yes["top1_qcov"].dropna()
        if not qcov.empty:
            lines += ["", "  -- Top-1 query coverage (SW) --",
                      f"    mean={qcov.mean():.4f}   median={qcov.median():.4f}"]
            for c in (0.3, 0.5, 0.8):
                lines.append(f"    ≥{c:.2f}  ->  {(qcov >= c).sum():>5}  "
                             f"({(qcov >= c).mean()*100:5.1f}%)")
    lines += ["", "  -- Phase 1 transport_prob (ensemble) --",
              f"    YES: mean={yes['transport_prob'].mean():.3f}  median={yes['transport_prob'].median():.3f}",
              f"    NO : mean={df.loc[df['is_transport']=='NO','transport_prob'].mean():.3f}  median={df.loc[df['is_transport']=='NO','transport_prob'].median():.3f}",
              "", "=" * 60]
    txt = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"  [save] {out / 'summary.txt'}")


def plot_all(df, plot_dir):
    """Run all plots in one call. Failures are caught and don't interrupt the rest."""
    print(f"\n[Plot] Output directory: {plot_dir}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_funcs = [
        ("01 TC class",         _plot_tc_class),
        ("02 TC subclass",      _plot_tc_subclass),
        ("03 top families",     _plot_top_families),
        ("04 similarity hist",  _plot_similarity_hist),
        ("05 similarity CDF",   _plot_similarity_cdf),
        ("06 phase1 prob",      _plot_transport_prob),
        ("07 seq length",       _plot_seq_length),
        ("08 top1 identity",    _plot_identity),
        ("09 top1 qcov",        _plot_qcov),
        ("99 dashboard",        _plot_dashboard),
    ]
    for name, fn in plot_funcs:
        try:
            print(f"[Plot] {name} ...")
            fn(df, plot_dir)
        except Exception as e:
            print(f"[Plot] {name} FAILED: {e}")

    try:
        print("[Sum ] summary ...")
        _write_summary(df, plot_dir)
    except Exception as e:
        print(f"[Sum ] FAILED: {e}")

    n_png = len(list(plot_dir.glob("*.png")))
    print(f"\n[Plot] Done: {n_png} PNG files + summary.txt")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1+2 one-stop prediction + auto-plot")
    parser.add_argument("-in", "--input_fasta", required=True)
    parser.add_argument("-out", "--output_dir", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--s4pred_dir", default="/data2/chao/s4pred")
    parser.add_argument("--s4pred_cache",
                        help="Legacy .tsv cache path (deprecated, for one-time migration from v1/v2). "
                             "Default: cache is auto-managed via .horiz files in ss_results/")
    parser.add_argument("--s4pred_outdir",
                        help="s4pred raw .horiz file persistence directory "
                             "(default: <output_dir>/ss_results). '' disables")
    parser.add_argument("--force_s4pred", action="store_true",
                        help="Ignore s4pred cache, re-predict all sequences")
    parser.add_argument("--tcdb_annot")
    parser.add_argument("--tcdb_fasta",
                        help="TCDB protein FASTA (only needed if ref_meta has no 'sequence' column)")
    parser.add_argument("--ref_id_col",
                        help="ref_meta column name to use as ref ID (auto-detected in FASTA path, "
                             "can be specified manually)")
    parser.add_argument("--fetch_tcdb", action="store_true")
    parser.add_argument("--sw_topk", type=int, default=1,
                        help="Run SW alignment for the first N of top-k neighbors (default 1)")
    # ── v3 new: SW alignment (default OFF, enable with --with_sw) ─────────────
    parser.add_argument("--with_sw", action="store_true",
                        help="Enable Smith-Waterman alignment (outputs ident/qcov/aln_score "
                             "columns, requires Biopython). Default OFF")
    # ── v3 new: auto-plot control ──────────────────────────────────────────────
    parser.add_argument("--no_plot", action="store_true",
                        help="Disable auto-plot")
    parser.add_argument("--plot_dir",
                        help="Plot output directory (default: <output_dir>/plots)")
    # ── v3 ablation flags (added 2026-07-20) ─────────────────────────────────
    parser.add_argument("--no_phase1", action="store_true",
                        help="Ablation: skip Phase 1 binary filter, run Phase 2 on ALL proteins. "
                             "Forces is_transport=YES for every sequence regardless of threshold.")
    parser.add_argument("--no_s4pred", action="store_true",
                        help="Ablation: zero out s4pred secondary-structure features "
                             "(Phase 1 input becomes 1280-dim ESM-2 only, was 1283-dim). "
                             "Tests contribution of secondary-structure features.")
    parser.add_argument("--phase1_weights", default="0.8,0.2",
                        help='Phase 1 ensemble weights "mlp_w,xgb_w" (default "0.8,0.2" — '
                             'grid-optimal on TCDB balanced held-out, +0.46 pp Acc over v3 "0.4,0.6"). '
                             'Use "1.0,0.0" for pure MLP, "0.0,1.0" for pure XGB, '
                             '"0.5,0.5" for equal weighting.')
    # ── GPU selection (added 2026-07-20) ─────────────────────────────────────
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU index to use. -1 = auto-select least-loaded GPU "
                             "(most free memory). Default: -1 (auto).")
    args = parser.parse_args()

    # Auto-select GPU: query nvidia-smi, pick GPU with most free memory
    if args.gpu == -1 and torch.cuda.is_available():
        try:
            import subprocess as _sp
            r = _sp.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            best_idx, best_free = 0, -1
            for line in r.stdout.strip().split("\n"):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3:
                    idx, used, total = int(parts[0]), int(parts[1]), int(parts[2])
                    free = total - used
                    if free > best_free:
                        best_free, best_idx = free, idx
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
            print(f"[GPU] Auto-selected GPU {best_idx} (free: {best_free} MiB)")
        except Exception as e:
            print(f"[GPU] Auto-select failed ({e}), falling back to GPU 0")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[GPU] User-selected GPU {args.gpu}")
    else:
        print(f"[GPU] No CUDA available, using CPU")

    # Print ablation config (visible at startup)
    if args.no_phase1 or args.no_s4pred or args.phase1_weights != "0.8,0.2":
        print(f"\n[Ablation] ACTIVE flags:")
        if args.no_phase1:     print(f"  --no_phase1      : ON  (skip Phase 1, all proteins -> Phase 2)")
        if args.no_s4pred:     print(f"  --no_s4pred      : ON  (s4pred features zeroed)")
        if args.phase1_weights != "0.8,0.2":
            print(f"  --phase1_weights : {args.phase1_weights}  (custom ensemble weights, default 0.8:0.2)")
        print()

    if not Path(args.input_fasta).exists():
        print(f"[ERROR] Not found: {args.input_fasta}"); return

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")
    print(f"[v3 pipeline] VERSION: 2026-09-10-r2 (saves ESM-2 mean-pool to embeddings.npz)")

    print("[Load] ESM2 ...")
    # Support two ESM-2 formats:
    # 1. ESM2_PATH = .pt file     -> fair-esm (load_model_and_alphabet_local)
    # 2. ESM2_PATH = directory   -> HuggingFace transformers (AutoModel)
    esm_is_hf = Path(ESM2_PATH).is_dir()
    esm_model, alphabet, hf_model, hf_tokenizer = None, None, None, None
    if esm_is_hf:
        from transformers import AutoModel, AutoTokenizer
        print(f"  using HF transformers from {ESM2_PATH}")
        hf_tokenizer = AutoTokenizer.from_pretrained(ESM2_PATH)
        hf_model = AutoModel.from_pretrained(ESM2_PATH, torch_dtype=torch.float32).to(device)
        # 3B = 36 layers (0-35), 650M = 33 layers (0-32)
        hf_num_layers = hf_model.config.num_hidden_layers
        hf_repr_layer = hf_num_layers  # last layer
        print(f"  HF model: hidden_size={hf_model.config.hidden_size}, num_layers={hf_num_layers} (on {device})")
    else:
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(ESM2_PATH)
        print(f"  fair-esm: embed_dim={esm_model.embed_dim}, num_layers={esm_model.num_layers}")

    # Sanity: PHASE1_FEAT_DIM must match the actual ESM-2 embed dim + 3 (s4pred).
    # We pass input_dim explicitly to TransportClassifier() below, so re-binding here works.
    actual_embed = hf_model.config.hidden_size if esm_is_hf else esm_model.embed_dim
    expected_feat = actual_embed + 3
    global PHASE1_FEAT_DIM  # needed because we may reassign below
    if PHASE1_FEAT_DIM != expected_feat:
        print(f"[WARN] PHASE1_FEAT_DIM={PHASE1_FEAT_DIM} but ESM-2 embed_dim={actual_embed} "
              f"-> expected PHASE1_FEAT_DIM={expected_feat}")
        print(f"[WARN] Auto-correcting PHASE1_FEAT_DIM={expected_feat} for this run "
              f"(set env var to silence this).")
        PHASE1_FEAT_DIM = expected_feat

    # Phase 2 (TC family) needs a SEPARATE ESM-2 in the Cascade config (650M, fair-esm).
    # The user sets PHASE2_ESM2_PATH explicitly. If not, default to ESM2_PATH (legacy).
    # Phase 2 always uses fair-esm API (ContrastiveEncoder.forward hardcodes `repr_layers=[33]`).
    phase2_esm_model, phase2_alphabet = None, None
    if PHASE2_ESM2_PATH != ESM2_PATH and Path(PHASE2_ESM2_PATH).is_file():
        print(f"[Load] Phase 2 ESM-2 (Cascade, separate from Phase 1): {PHASE2_ESM2_PATH}")
        phase2_esm_model, phase2_alphabet = esm.pretrained.load_model_and_alphabet_local(PHASE2_ESM2_PATH)
        print(f"  Phase 2 ESM-2: embed_dim={phase2_esm_model.embed_dim}, num_layers={phase2_esm_model.num_layers}")
    print("[Load] Phase 1 MLP ..."); phase1_mlp = TransportClassifier(input_dim=PHASE1_FEAT_DIM).to(device)
    phase1_mlp.load_state_dict(torch.load(MLP_ASSET, map_location=device, weights_only=True))
    phase1_mlp.eval()
    print("[Load] Phase 1 XGB ..."); xgb_model = xgb.Booster(); xgb_model.load_model(XGB_ASSET)
    # Phase 2 encoder (graceful fallback if checkpoint was trained with a different ESM-2 dim,
    # e.g. 3B sanity run with 650M-trained Phase 2 ckpt)
    phase2_model = None
    try:
        # Use separate Phase 2 ESM-2 (650M) if loaded; else fall back to Phase 1's ESM-2.
        phase2_backbone = phase2_esm_model if phase2_esm_model is not None else esm_model
        if phase2_backbone is None:
            # Phase 1 is HF transformers; Phase 2 needs fair-esm and none was provided.
            raise RuntimeError(
                "Phase 2 needs a fair-esm model, but Phase 1 is HF transformers and "
                "PHASE2_ESM2_PATH was not set to a .pt file. Set PHASE2_ESM2_PATH to the 650M .pt."
            )
        print("[Load] Phase 2 encoder ..."); phase2_model = ContrastiveEncoder(phase2_backbone).to(device)
        ckpt = torch.load(PHASE2_CKPT, map_location=device, weights_only=False)
        # load_state_dict returns _IncompatibleKeys(missing, unexpected) in PyTorch ≥ 1.9
        load_result = phase2_model.load_state_dict(ckpt["model_state"], strict=False)
        if load_result.missing_keys:
            print(f"[Load] WARN missing {len(load_result.missing_keys)} keys "
                  f"(e.g. {load_result.missing_keys[:3]})")
        if load_result.unexpected_keys:
            print(f"[Load] WARN unexpected {len(load_result.unexpected_keys)} keys "
                  f"(e.g. {load_result.unexpected_keys[:3]})")
        if not load_result.missing_keys and not load_result.unexpected_keys:
            print(f"[Load] All {sum(1 for _ in phase2_model.state_dict().keys())} keys loaded successfully")
        phase2_model.eval()
    except (RuntimeError, KeyError) as e:
        print(f"[WARN] Phase 2 encoder load failed ({type(e).__name__}: {e})")
        print(f"[WARN] Phase 2 will be SKIPPED. is_transport decision still works (Phase 1 only).")
        print(f"[WARN] TC family annotation will be empty. To fix, retrain Phase 2 with current ESM-2.")
        phase2_model = None
    print("[Load] TCDB reference index ...")
    ref_embs = np.load(REF_EMB); ref_meta = pd.read_csv(REF_META)

    # Build SW lookup (default OFF; add --with_sw to load ref sequences + run Biopython)
    if not args.with_sw:
        ref_seq_lookup, lookup_stats = {}, {"source": "(disabled, use --with_sw to enable)", "hit": 0, "miss": 0}
        print("[SW] disabled by default, skipping ref sequence load and Biopython "
              "(add --with_sw to enable)")
    else:
        ref_seq_lookup, lookup_stats = build_ref_seq_lookup(ref_meta, args)

    print(f"\n[Input] {args.input_fasta}")
    queries = read_fasta(args.input_fasta)
    print(f"[Input] {len(queries)} sequences")
    valid = [(n, s) for n, s in queries if is_valid(s)]
    print(f"[Valid] {len(valid)}/{len(queries)} pass filter")
    if not valid: print("[Error] No valid sequences"); return

    names = [n for n, _ in valid]
    seqs = [s[:MAX_LEN] for _, s in valid]

    # s4pred persistent output directory (also serves as the cache, .horiz files)
    if args.s4pred_outdir is None:
        s4pred_outdir = str(out_dir / "ss_results")
    elif args.s4pred_outdir == "":
        s4pred_outdir = None
    else:
        s4pred_outdir = args.s4pred_outdir
    if s4pred_outdir:
        print(f"[s4pred] Cache = raw .horiz files in: {s4pred_outdir}")
    else:
        print("[s4pred] WARNING: no outdir, cache disabled (re-runs every time)")

    # Legacy .tsv cache (optional, for one-time migration from old pipeline)
    legacy_cache_path = args.s4pred_cache if args.s4pred_cache else None

    s4pred_feats = run_s4pred(list(zip(names, seqs)),
                              s4pred_dir=args.s4pred_dir,
                              outdir=s4pred_outdir,
                              legacy_cache=legacy_cache_path,
                              force=args.force_s4pred)
    if s4pred_feats is None:
        s4pred_feats = {n: np.array([0.5, 0.2, 0.3], dtype=np.float32) for n in names}

    print(f"\n[ESM2] Encoding {len(seqs)} sequences (batch={ENCODE_BATCH}) ...")
    amp_ctx = autocast(dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else autocast()
    phase1_features, phase2_embs = [], []
    mean_pool_list = []  # collect for embeddings.npz (Phase 1 ESM-2 mean-pool, 2560-d for 3B / 1280-d for 650M)
    with torch.no_grad():
        for start in tqdm(range(0, len(seqs), ENCODE_BATCH), desc="ESM2"):
            batch_seqs = seqs[start: start + ENCODE_BATCH]
            batch_names = names[start: start + ENCODE_BATCH]
            if esm_is_hf:
                # HF transformers path: tokenizer + model from `esm2_3b/` dir
                enc = hf_tokenizer(batch_seqs, return_tensors='pt', padding=True,
                                   truncation=True, max_length=MAX_LEN + 2)
                input_ids = enc['input_ids'].to(device)
                attn_mask = enc['attention_mask'].to(device)
                with amp_ctx:
                    outputs = hf_model(input_ids=input_ids, attention_mask=attn_mask,
                                       output_hidden_states=True)
                # hidden_states is a tuple of (num_hidden_layers + 1) tensors; [-1] = final layer
                tok_emb = outputs.hidden_states[-1]
                pad_id = hf_tokenizer.pad_token_id
                eos_id = hf_tokenizer.eos_token_id
            else:
                # fair-esm path (legacy, single .pt)
                token_lists = [tokenize(s, alphabet) for s in batch_seqs]
                input_ids = torch.tensor(token_lists, dtype=torch.long).to(device)
                with amp_ctx:
                    out = esm_model(input_ids, repr_layers=[esm_model.num_layers],
                                    return_contacts=False)
                tok_emb = out["representations"][esm_model.num_layers]
                pad_id = alphabet.padding_idx
                eos_id = alphabet.eos_idx

            mask = ((input_ids != pad_id) & (input_ids != eos_id)).unsqueeze(-1).float()
            mean_pool = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
            mean_pool_list.append(mean_pool.cpu().numpy())  # save for embeddings.npz
            for i, name in enumerate(batch_names):
                s4p = s4pred_feats.get(name, np.array([0.5, 0.2, 0.3], dtype=np.float32))
                if args.no_s4pred:
                    # Ablation: zero out s4pred secondary-structure features
                    s4p = np.zeros_like(s4p)
                phase1_features.append(np.concatenate([mean_pool[i].cpu().numpy(), s4p]))
            if phase2_model is not None:
                # Phase 2 always uses fair-esm (Cascade: 3B Phase 1 + 650M Phase 2).
                # The tokenized input_ids from either path share the same vocabulary
                # (ESM-2 BPE), so we can pass them to the fair-esm-backed ContrastiveEncoder.
                # But: HF tokenizer's pad_token_id may be 1 (same as fair-esm padding_idx).
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=False):  # force fp32
                        z = phase2_model(input_ids)
                phase2_embs.append(z.cpu().float().numpy())
            else:
                # Phase 2 disabled: emit zero embeddings (same shape as ContrastiveEncoder output EMBED_DIM=512)
                phase2_embs.append(np.zeros((len(batch_names), 512), dtype=np.float32))

    # === Save ESM-2 mean-pool embeddings to embeddings.npz ===
    # Used by downstream filters (e.g. filter_tcdb_esm.py) to do cosine-similarity
    # to known reference proteins without re-encoding. Row i = final_report.csv row i.
    if mean_pool_list:
        emb_matrix = np.concatenate(mean_pool_list, axis=0).astype(np.float32)
        emb_path = out_dir / "embeddings.npz"
        np.savez(emb_path, accessions=np.array(names), embeddings=emb_matrix)
        print(f"[Save] {emb_path}  ({emb_matrix.shape[0]} proteins × {emb_matrix.shape[1]}-dim)")

    phase1_features = np.array(phase1_features, dtype=np.float32)
    phase2_embs = np.concatenate(phase2_embs, axis=0)

    print(f"\n[Phase 1] MLP + XGBoost ensemble ...")
    X_tensor = torch.from_numpy(phase1_features).float().to(device)
    with torch.no_grad():
        mlp_probs = torch.sigmoid(phase1_mlp(X_tensor)).cpu().numpy().flatten()
    dmatrix = xgb.DMatrix(phase1_features)
    xgb_probs = xgb_model.predict(dmatrix)

    # Parse ensemble weights (Ablation: --phase1_weights)
    try:
        mlp_w, xgb_w = [float(x.strip()) for x in args.phase1_weights.split(",")]
    except Exception as e:
        print(f"[ERROR] --phase1_weights must be 'mlp_w,xgb_w' (got '{args.phase1_weights}')")
        return
    assert abs(mlp_w + xgb_w - 1.0) < 1e-6, \
        f"weights must sum to 1.0, got {mlp_w}+{xgb_w}={mlp_w+xgb_w}"
    ensemble_probs = mlp_w * mlp_probs + xgb_w * xgb_probs

    if args.no_phase1:
        # Ablation: force ALL proteins to be classified as transporters
        print(f"[Phase 1] --no_phase1 ACTIVE: forcing all {len(seqs)} proteins to Phase 2 "
              f"(ensemble_probs still computed for analysis)")
        is_transport = np.ones(len(seqs), dtype=int)
    else:
        is_transport = (ensemble_probs > args.threshold).astype(int)
    print(f"[Phase 1] Transporters: {is_transport.sum()}/{len(is_transport)}")
    n_transport = int(is_transport.sum())

    print(f"\n[Phase 2] Retrieving TC family for {n_transport} "
          f"transporters (k={args.k}, sw_topk={args.sw_topk}) ...")

    sw_cache: dict = {}
    phase2_summary = {}
    for i in range(len(names)):
        if not is_transport[i]:
            phase2_summary[i] = None; continue
        if phase2_model is None:
            # Phase 2 disabled (e.g. 3B sanity with 650M-trained Phase 2 ckpt) —
            # skip TC family annotation rather than emit garbage from zero embeddings
            phase2_summary[i] = None; continue

        sim = phase2_embs[i] @ ref_embs.T
        topk_idx = np.argsort(-sim)[:args.k]
        top_fams = [ref_meta.iloc[idx]["tc_family"] for idx in topk_idx]
        fam_votes = Counter(top_fams).most_common(1)

        top_family = fam_votes[0][0] if fam_votes else "?"
        top_family_votes = fam_votes[0][1] if fam_votes else 0
        top_family_purity = top_family_votes / args.k

        top1_idx = topk_idx[0]
        top_tc = ref_meta.iloc[top1_idx]["tc_number"]
        top_sim = float(sim[top1_idx])
        top2_sim = float(sim[topk_idx[1]]) if len(topk_idx) > 1 else 0.0
        top1_sim_gap = top_sim - top2_sim

        topk_with_sw = []
        for r, idx in enumerate(topk_idx):
            idx = int(idx)
            tc_number = ref_meta.iloc[idx]["tc_number"]
            tc_family = ref_meta.iloc[idx]["tc_family"]
            entry = {"rank": r + 1, "tc_number": tc_number, "tc_family": tc_family,
                     "similarity": float(sim[idx])}
            if r < args.sw_topk and args.with_sw and ref_seq_lookup and HAVE_BIOPYTHON:
                ref_seq = ref_seq_lookup.get(idx)
                if ref_seq:
                    if idx not in sw_cache:
                        sw_cache[idx] = smith_waterman_stats(seqs[i], ref_seq)
                    ident, qcov, ascr = sw_cache[idx]
                    entry["seq_identity"] = round(ident, 4) if not np.isnan(ident) else None
                    entry["qcov"] = round(qcov, 4) if not np.isnan(qcov) else None
                    entry["aln_score"] = round(ascr, 1) if not np.isnan(ascr) else None
                else:
                    entry["seq_identity"] = entry["qcov"] = entry["aln_score"] = None
            else:
                entry["seq_identity"] = entry["qcov"] = entry["aln_score"] = None
            topk_with_sw.append(entry)

        s1 = topk_with_sw[0]
        phase2_summary[i] = {
            "top_family": top_family, "top_tc_number": top_tc,
            "top1_similarity": top_sim, "top1_sim_gap": round(top1_sim_gap, 4),
            "top_family_votes": top_family_votes,
            "top_family_purity": round(top_family_purity, 4),
            "top1_seq_identity": s1["seq_identity"],
            "top1_qcov": s1["qcov"],
            "top1_aln_score": s1["aln_score"],
            "topk": topk_with_sw,
        }

    # ── TCDB annotation ─────────────────────────────────────────────────────────
    tcdb_annots = {}
    if args.tcdb_annot and os.path.exists(args.tcdb_annot):
        tcdb_annots = load_tcdb_annotations(args.tcdb_annot)
        print(f"[TCDB-Annot] Loaded {len(tcdb_annots)} entries from {args.tcdb_annot}")
    elif args.tcdb_fasta and os.path.exists(args.tcdb_fasta):
        cache_path = Path(args.tcdb_fasta).with_suffix(".annot.tsv")
        tcdb_annots = (load_tcdb_annotations(str(cache_path)) if cache_path.exists()
                       else parse_tcdb_fasta(args.tcdb_fasta, str(cache_path)))

    if args.fetch_tcdb and tcdb_annots is not None:
        online_cache_path = out_dir / "tcdb_online_cache.tsv"
        online_cache = {}
        if online_cache_path.exists():
            with open(online_cache_path) as f:
                for line in f:
                    p = line.strip().split("\t")
                    if len(p) >= 2: online_cache[p[0]] = p[1]
        unique_tcs = {s["top_tc_number"] for s in phase2_summary.values()
                      if s and s["top_tc_number"] and s["top_tc_number"] not in tcdb_annots}
        to_fetch = unique_tcs - set(online_cache.keys())
        for tc in to_fetch:
            print(f"  Fetching {tc} ...", end=" ", flush=True)
            d = fetch_tcdb_online(tc)
            if d: online_cache[tc] = d; print("OK")
            else: print("FAIL")
            time.sleep(1)
        with open(online_cache_path, "w") as f:
            for tc, d in online_cache.items(): f.write(f"{tc}\t{d}\n")
        tcdb_annots.update(online_cache)

    # ── Write reports ───────────────────────────────────────────────────────────
    phase1_path = out_dir / "phase1_report.tsv"
    with open(phase1_path, "w") as f:
        f.write("Protein_Header\tSequence_Length\tMLP_Prob\tXGB_Prob\t"
                "Ensemble_Prob\tIs_Transport_Protein\n")
        for i, name in enumerate(names):
            status = "YES" if is_transport[i] else "NO"
            f.write(f"{name}\t{len(seqs[i])}\t{mlp_probs[i]:.4f}\t"
                    f"{xgb_probs[i]:.4f}\t{ensemble_probs[i]:.4f}\t{status}\n")
    print(f"[Save] {phase1_path}")

    phase2_path = out_dir / "phase2_report.csv"
    phase2_rows = []
    for i, name in enumerate(names):
        s = phase2_summary[i]
        if s is None:
            phase2_rows.append({"query": name, "is_transport": "NO"})
        else:
            row = {"query": name, "is_transport": "YES",
                   "top_family": s["top_family"], "top_tc_number": s["top_tc_number"],
                   "top1_similarity": s["top1_similarity"],
                   "top1_sim_gap": s["top1_sim_gap"],
                   "top_family_votes": s["top_family_votes"],
                   "top_family_purity": s["top_family_purity"],
                   "top1_seq_identity": s["top1_seq_identity"],
                   "top1_qcov": s["top1_qcov"],
                   "top1_aln_score": s["top1_aln_score"]}
            for d in s["topk"]:
                row[f"top{d['rank']}_tc"] = d["tc_number"]
                row[f"top{d['rank']}_family"] = d["tc_family"]
                row[f"top{d['rank']}_sim"] = d["similarity"]
                if d.get("seq_identity") is not None:
                    row[f"top{d['rank']}_ident"] = d["seq_identity"]
                    row[f"top{d['rank']}_qcov"] = d["qcov"]
                    row[f"top{d['rank']}_score"] = d["aln_score"]
            phase2_rows.append(row)
    pd.DataFrame(phase2_rows).to_csv(phase2_path, index=False)
    print(f"[Save] {phase2_path}")

    final_path = out_dir / "final_report.csv"
    final_rows = []
    for i, name in enumerate(names):
        s = phase2_summary[i]
        tc_num = s["top_tc_number"] if s else ""
        tc_desc = tcdb_annots.get(tc_num, "") if tc_num else ""
        # is_transport is decided by Phase 1 ensemble (line 1166), NOT by Phase 2
        # Phase 2 only annotates TC family — when s is None (Phase 2 disabled/failed),
        # is_transport stays as Phase 1 said, TC fields stay empty.
        is_yes = is_transport[i] == 1
        if s is None:
            final_rows.append({
                "query": name, "seq_length": len(seqs[i]),
                "transport_prob": round(ensemble_probs[i], 4),
                "is_transport": "YES" if is_yes else "NO", "tc_family": "", "tc_number": "",
                "tc_description": "", "top1_similarity": 0.0,
                "top1_sim_gap": 0.0, "top_family_votes": 0,
                "top_family_purity": 0.0,
                "top1_seq_identity": None, "top1_qcov": None, "top1_aln_score": None,
            })
        else:
            final_rows.append({
                "query": name, "seq_length": len(seqs[i]),
                "transport_prob": round(ensemble_probs[i], 4),
                "is_transport": "YES" if is_yes else "NO", "tc_family": s["top_family"],
                "tc_number": tc_num, "tc_description": tc_desc,
                "top1_similarity": s["top1_similarity"],
                "top1_sim_gap": s["top1_sim_gap"],
                "top_family_votes": s["top_family_votes"],
                "top_family_purity": s["top_family_purity"],
                "top1_seq_identity": s["top1_seq_identity"],
                "top1_qcov": s["top1_qcov"],
                "top1_aln_score": s["top1_aln_score"],
            })
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(final_path, index=False)
    print(f"[Save] {final_path}")

    # ── Print summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  Pipeline Summary\n{'='*60}")
    print(f"  Total sequences       : {len(names)}")
    print(f"  Phase 1 (transporters): {is_transport.sum()}")
    print(f"  Phase 2 (TC family)   : {sum(1 for s in phase2_summary.values() if s)}")
    print(f"  Output directory      : {out_dir}")
    if legacy_cache_path:
        print(f"  s4pred legacy cache   : {legacy_cache_path}")
    if s4pred_outdir:
        print(f"  s4pred raw outputs    : {s4pred_outdir}")
    print(f"{'='*60}\n")

    if n_transport > 0:
        sims = np.array([s["top1_similarity"] for s in phase2_summary.values() if s])
        gaps = np.array([s["top1_sim_gap"] for s in phase2_summary.values() if s])
        print(f"  Top-1 sim     : mean={sims.mean():.3f}  median={np.median(sims):.3f}  "
              f"<0.7: {(sims<0.7).sum()}")
        print(f"  Top1-top2 gap : mean={gaps.mean():.3f}  median={np.median(gaps):.3f}  "
              f"<0.05: {(gaps<0.05).sum()}")
        if HAVE_BIOPYTHON and ref_seq_lookup:
            print(f"  SW lookup     : source={lookup_stats['source']}  "
                  f"hit={lookup_stats['hit']}/{len(ref_meta)}  "
                  f"miss={lookup_stats['miss']}")
            qcovs = np.array([s["top1_qcov"] for s in phase2_summary.values()
                              if s and s["top1_qcov"] is not None])
            idents = np.array([s["top1_seq_identity"] for s in phase2_summary.values()
                               if s and s["top1_seq_identity"] is not None])
            if len(qcovs):
                print(f"  qcov (SW)     : mean={qcovs.mean():.3f}  median={np.median(qcovs):.3f}  "
                      f"<0.5: {(qcovs<0.5).sum()}")
            if len(idents):
                print(f"  identity (SW) : mean={idents.mean():.3f}  median={np.median(idents):.3f}  "
                      f"<0.3: {(idents<0.3).sum()}")

    transport_indices = np.where(is_transport == 1)[0][:10]
    if len(transport_indices) > 0:
        print("\n  Top transport protein predictions:")
        for i in transport_indices:
            s = phase2_summary[i]
            print(f"    {names[i]}")
            if s is None:
                # Phase 2 disabled / failed (e.g. 3B sanity with 650M-trained Phase 2 ckpt)
                print(f"      -> (TC family not annotated; Phase 2 unavailable)")
                print(f"      transport_prob={ensemble_probs[i]:.4f}")
                print()
                continue
            print(f"      -> {s['top_family']}  ({s['top_tc_number']})  "
                  f"sim={s['top1_similarity']:.4f}  gap={s['top1_sim_gap']:.4f}  "
                  f"votes={s['top_family_votes']}/{args.k}", end="")
            if s['top1_qcov'] is not None:
                print(f"  qcov={s['top1_qcov']:.2f}  ident={s['top1_seq_identity']:.2f}", end="")
            print()
        if n_transport > 10:
            print(f"    ... ({n_transport - 10} more, see CSV)")
        print()

    # ── v3 new: auto-plot at the end ───────────────────────────────────────────
    if not args.no_plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else out_dir / "plots"
        plot_all(final_df, plot_dir)


if __name__ == "__main__":
    main()