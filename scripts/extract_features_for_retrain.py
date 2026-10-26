#!/usr/bin/env python
"""
extract_features_for_retrain.py
================================

Extract 1283-dim Phase 1 features (1280 ESM-2 pooled + 3 s4pred H/E/C)
from TCDB (positives) + UniProt random (negatives) fastas, with a
stratified positive/negative split that produces a *balanced* train
and *balanced* val set so binary classification ablation is meaningful.

Output (under --output_dir):
  X_train.npy   shape (N_pos_train + N_neg_train, 1283) float32
  Y_train.npy   shape (N_pos_train + N_neg_train,) float32  (1=TCDB, 0=UniProt)
  X_val.npy     shape (N_pos_val + N_neg_val, 1283) float32
  Y_val.npy     shape (N_pos_val + N_neg_val,) float32
  names_train.txt  (one per line, parallel to X_train rows; prefix "pos__"/"neg__")
  names_val.txt

Splitting rules:
  - TCDB pos fasta is already 80/10/10 split by split_tcdb.py → use as-is
  - Negative pool is deterministically shuffled (seed=42) and split so that
    the val negatives are exactly N_pos_val in count (balanced val).
  - Remaining negatives after carving out the val set are used for training
    (typically slightly less than 1:1, but close).

Usage:
  python extract_features_for_retrain.py \
      --pos_train_fasta /data2/chao/transport/data/tcdb_train.fasta \
      --pos_val_fasta   /data2/chao/transport/data/tcdb_val.fasta \
      --neg_pool_fasta  /data2/chao/transport/uniprot_negative_c90.fasta \
      --output_dir      /data2/chao/transport/data/features_v3_retrain \
      --gpu 0
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

# Reuse v3 internals
SCRIPTS_DIR = "/data2/chao/transport/scripts"
sys.path.insert(0, SCRIPTS_DIR)
import predict_transport_family_v3 as v3


def read_fasta_simple(path):
    """Returns list of (name, seq) tuples. v3.read_fasta already provides this,
    but we wrap so we can attach a 'pos__'/'neg__' prefix for traceability."""
    return v3.read_fasta(path)


def extract_one_fasta(fasta_path: str, device, esm_model, alphabet, s4pred_cache_dir: str = None,
                       label_tag: str = ""):
    """
    Returns (X, names) for a single fasta.
      X:     (N, 1283) float32
      names: list of N strings, each prefixed with label_tag (e.g. "pos__" or "neg__")
    """
    queries = read_fasta_simple(fasta_path)
    valid = [(n, s) for n, s in queries if v3.is_valid(s)]
    print(f"  {fasta_path}: {len(valid)}/{len(queries)} pass filter")

    names = [f"{label_tag}{n}" for n, _ in valid]
    seqs = [s[: v3.MAX_LEN] for _, s in valid]

    # s4pred (per-sequence .horiz cache optional)
    cache_dir = s4pred_cache_dir if s4pred_cache_dir else str(Path(fasta_path).parent / "ss_results")
    s4pred_feats = v3.run_s4pred(
        list(zip([n for n in names], seqs)),
        s4pred_dir="/data2/chao/s4pred",
        outdir=cache_dir,
    )
    if s4pred_feats is None:
        s4pred_feats = {n: np.array([0.5, 0.2, 0.3], dtype=np.float32) for n in names}

    # ESM-2 encoding in batches
    amp_ctx = autocast(dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else autocast()
    X = []
    with torch.no_grad():
        for start in tqdm(range(0, len(seqs), v3.ENCODE_BATCH), desc=f"ESM2 {Path(fasta_path).name}"):
            batch_seqs = seqs[start:start + v3.ENCODE_BATCH]
            batch_names = names[start:start + v3.ENCODE_BATCH]
            token_lists = [v3.tokenize(s, alphabet) for s in batch_seqs]
            input_ids = torch.tensor(token_lists, dtype=torch.long).to(device)
            with amp_ctx:
                out = esm_model(input_ids, repr_layers=[33], return_contacts=False)
            tok_emb = out["representations"][33]
            mask = ((input_ids != alphabet.padding_idx) & (input_ids != alphabet.eos_idx)).unsqueeze(-1).float()
            mean_pool = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
            for i, name in enumerate(batch_names):
                s4p = s4pred_feats.get(name, np.array([0.5, 0.2, 0.3], dtype=np.float32))
                feat = np.concatenate([mean_pool[i].cpu().numpy(), s4p]).astype(np.float32)
                X.append(feat)

    return np.array(X, dtype=np.float32), names


def split_neg_pool_for_val(neg_path: str, n_pos_val: int, neg_train_target: int = None, seed: int = 42):
    """
    Deterministically shuffle the negative pool, then split:
      - first n_pos_val entries → val negatives (balanced with TCDB val)
      - next neg_train_target entries → train negatives (default = len(pos_train), i.e. 1:1)
      - remainder is discarded (uniprot_negative pool is huge: ~338k)

    Writes the two split fastas next to the original negative fasta
    (suffix _val.fasta / _train.fasta) so downstream s4pred can cache per split.
    Returns the two paths.
    """
    queries = read_fasta_simple(neg_path)
    print(f"[NegPool] {neg_path}: {len(queries)} sequences before shuffle")
    rnd = random.Random(seed)
    rnd.shuffle(queries)
    val_block = queries[:n_pos_val]
    after_val = queries[n_pos_val:]
    if neg_train_target is None or neg_train_target > len(after_val):
        train_block = after_val
    else:
        train_block = after_val[:neg_train_target]
    print(f"[NegPool] After split → train_neg={len(train_block)}, val_neg={len(val_block)} (discarded {len(after_val) - len(train_block)})")

    base = Path(neg_path)
    train_out = str(base).replace(".fasta", ".train.fasta")
    val_out = str(base).replace(".fasta", ".val.fasta")

    def write_fasta(path, items):
        with open(path, "w") as f:
            for n, s in items:
                f.write(f">{n}\n{s}\n")

    write_fasta(train_out, train_block)
    write_fasta(val_out, val_block)
    print(f"[NegPool] Wrote {train_out} and {val_out}")
    return train_out, val_out


def main():
    ap = argparse.ArgumentParser(description="Extract 1283-dim Phase 1 features for v3 retraining (pos + neg)")
    ap.add_argument("--pos_train_fasta", required=True, help="TCDB-derived 80% positives (train)")
    ap.add_argument("--pos_val_fasta", required=True, help="TCDB-derived 10% positives (val)")
    ap.add_argument("--neg_pool_fasta", required=True, help="UniProt random non-transport pool (CD-HIT 90%)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_s4pred_cache", default=None)
    ap.add_argument("--val_s4pred_cache", default=None)
    ap.add_argument("--gpu", type=int, default=0, help="GPU id (default 0; -1 = CPU)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg_train_target", type=int, default=None,
                    help="If set, subsample train negatives to this many (default: 1:1 with pos train, ~19337). "
                         "Use a smaller value to speed up ESM-2 extraction.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    esm_model, alphabet = v3.esm.pretrained.load_model_and_alphabet_local(v3.ESM2_PATH)
    esm_model = esm_model.to(device)
    esm_model.eval()
    print(f"[ESM2] {v3.ESM2_PATH}")

    # === 1. Inspect pos_val size so we can carve a balanced val_neg set ===
    pos_val_count_raw = sum(1 for _ in read_fasta_simple(args.pos_val_fasta))
    print(f"[Pos] val (raw, pre-filter) = {pos_val_count_raw}")

    # If user did not override neg_train_target, default to 1:1 with pos_train
    pos_train_count_raw = sum(1 for _ in read_fasta_simple(args.pos_train_fasta))
    print(f"[Pos] train (raw, pre-filter) = {pos_train_count_raw}")
    if args.neg_train_target is None:
        args.neg_train_target = pos_train_count_raw
    print(f"[Plan] neg_train_target = {args.neg_train_target}  (1:1 with pos_train)")

    # === 2. Split negative pool ===
    neg_train_fasta, neg_val_fasta = split_neg_pool_for_val(
        args.neg_pool_fasta, n_pos_val=pos_val_count_raw,
        neg_train_target=args.neg_train_target, seed=args.seed
    )

    # === 3. Extract features ===
    print("\n[Extract] TCDB pos train ...")
    X_pos_tr, names_pos_tr = extract_one_fasta(
        args.pos_train_fasta, device, esm_model, alphabet,
        s4pred_cache_dir=args.train_s4pred_cache, label_tag="pos__"
    )
    print(f"  X_pos_tr: {X_pos_tr.shape}")

    print("\n[Extract] TCDB pos val ...")
    X_pos_va, names_pos_va = extract_one_fasta(
        args.pos_val_fasta, device, esm_model, alphabet,
        s4pred_cache_dir=args.val_s4pred_cache, label_tag="pos__"
    )
    print(f"  X_pos_va: {X_pos_va.shape}")

    print("\n[Extract] UniProt neg train ...")
    X_neg_tr, names_neg_tr = extract_one_fasta(
        neg_train_fasta, device, esm_model, alphabet,
        s4pred_cache_dir=args.train_s4pred_cache, label_tag="neg__"
    )
    print(f"  X_neg_tr: {X_neg_tr.shape}")

    print("\n[Extract] UniProt neg val ...")
    X_neg_va, names_neg_va = extract_one_fasta(
        neg_val_fasta, device, esm_model, alphabet,
        s4pred_cache_dir=args.val_s4pred_cache, label_tag="neg__"
    )
    print(f"  X_neg_va: {X_neg_va.shape}")

    # === 4. Build mixed train and val with proper labels ===
    # Strategy: keep pos first then neg (for interpretability), but always shuffle rows
    # to prevent any positional bias in the train loop.
    X_train = np.vstack([X_pos_tr, X_neg_tr])
    Y_train = np.concatenate([
        np.ones(len(X_pos_tr), dtype=np.float32),
        np.zeros(len(X_neg_tr), dtype=np.float32),
    ])
    names_train = names_pos_tr + names_neg_tr

    X_val = np.vstack([X_pos_va, X_neg_va])
    Y_val = np.concatenate([
        np.ones(len(X_pos_va), dtype=np.float32),
        np.zeros(len(X_neg_va), dtype=np.float32),
    ])
    names_val = names_pos_va + names_neg_va

    # Shuffle train (and val) so pos/neg are interleaved
    rng = np.random.default_rng(args.seed)
    train_perm = rng.permutation(len(X_train))
    val_perm = rng.permutation(len(X_val))
    X_train, Y_train = X_train[train_perm], Y_train[train_perm]
    X_val, Y_val = X_val[val_perm], Y_val[val_perm]
    names_train = [names_train[i] for i in train_perm]
    names_val = [names_val[i] for i in val_perm]

    print(f"\n[Mix] train: pos={int(Y_train.sum())} neg={int((1 - Y_train).sum())} total={len(Y_train)}")
    print(f"[Mix] val:   pos={int(Y_val.sum())}   neg={int((1 - Y_val).sum())}   total={len(Y_val)}")

    # === 5. Persist ===
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "Y_train.npy", Y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "Y_val.npy", Y_val)
    with open(out_dir / "names_train.txt", "w") as f:
        for n in names_train: f.write(n + "\n")
    with open(out_dir / "names_val.txt", "w") as f:
        for n in names_val: f.write(n + "\n")

    print(f"\n[Save] {out_dir / 'X_train.npy'}  shape={X_train.shape}  pos={int(Y_train.sum())}  neg={int((1-Y_train).sum())}")
    print(f"[Save] {out_dir / 'X_val.npy'}    shape={X_val.shape}    pos={int(Y_val.sum())}    neg={int((1-Y_val).sum())}")


if __name__ == "__main__":
    main()
