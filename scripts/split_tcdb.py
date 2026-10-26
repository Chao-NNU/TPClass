#!/usr/bin/env python
"""
split_tcdb.py
=============

Stratified train/val/test split of TCDB reference proteins.

By default splits 80% / 10% / 10% at the TC CLASS level (3-digit, e.g. 2.A.1)
to ensure all splits have representative class coverage.

Output:
  data/tcdb_train.fasta
  data/tcdb_val.fasta
  data/tcdb_test.fasta

Usage:
  python split_tcdb.py \
      --input /data2/chao/transport/blastout/tcdb.fasta \
      --out_dir /data2/chao/transport/data \
      --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 \
      --seed 42
"""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path


def parse_tcdb_fasta(path):
    """Parse TCDB fasta → list of (header, sequence, tc_full, tc_class) tuples."""
    records = []
    header = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(buf)
                    tc_full = extract_tc(header)
                    tc_cls = tc_full.split(".")[0] + "." + tc_full.split(".")[1] if tc_full else None
                    if seq and tc_full:
                        records.append((header, seq, tc_full, tc_cls))
                header = line
                buf = []
            else:
                buf.append(line)
        if header is not None:
            seq = "".join(buf)
            tc_full = extract_tc(header)
            tc_cls = tc_full.split(".")[0] + "." + tc_full.split(".")[1] if tc_full else None
            if seq and tc_full:
                records.append((header, seq, tc_full, tc_cls))
    return records


def extract_tc(header):
    """>gnl|TC-DB|P12345|2.A.1.12.1 description → '2.A.1.12.1'"""
    parts = header.lstrip(">").split("|")
    if len(parts) >= 4:
        return parts[3].strip().split()[0]
    return None


def stratified_split(records, train_frac, val_frac, seed):
    """Stratify by TC class, then split into train/val/test."""
    by_class = defaultdict(list)
    for r in records:
        by_class[r[3]].append(r)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for cls, recs in by_class.items():
        rng.shuffle(recs)
        n = len(recs)
        n_train = max(1, int(n * train_frac))
        n_val   = max(0, int(n * val_frac))
        # remainder → test
        n_test  = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        train.extend(recs[:n_train])
        val.extend(recs[n_train:n_train + n_val])
        test.extend(recs[n_train + n_val:])
    return train, val, test


def write_fasta(records, path):
    with open(path, "w") as f:
        for header, seq, _, _ in records:
            f.write(f"{header}\n{seq}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TCDB fasta path")
    ap.add_argument("--out_dir", required=True, help="output dir for train/val/test fastas")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--test_frac",  type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert abs(args.train_frac + args.val_frac + args.test_frac - 1.0) < 1e-6, \
        f"Fractions must sum to 1.0, got {args.train_frac}+{args.val_frac}+{args.test_frac}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] Reading {args.input} ...")
    records = parse_tcdb_fasta(args.input)
    print(f"  Parsed {len(records)} records with valid TC")

    print(f"[Split] {args.train_frac}/{args.val_frac}/{args.test_frac} stratified by TC class, seed={args.seed}")
    train, val, test = stratified_split(records, args.train_frac, args.val_frac, args.seed)
    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    write_fasta(train, out_dir / "tcdb_train.fasta")
    write_fasta(val,   out_dir / "tcdb_val.fasta")
    write_fasta(test,  out_dir / "tcdb_test.fasta")

    # write a summary
    summary_path = out_dir / "tcdb_split_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"TCDB stratified split (seed={args.seed})\n")
        f.write(f"  Train: {len(train)} ({args.train_frac*100:.0f}%)\n")
        f.write(f"  Val:   {len(val)} ({args.val_frac*100:.0f}%)\n")
        f.write(f"  Test:  {len(test)} ({args.test_frac*100:.0f}%)\n")
        f.write(f"  Total: {len(records)}\n\n")
        f.write("Class distribution:\n")
        for split_name, recs in [("train", train), ("val", val), ("test", test)]:
            f.write(f"\n  {split_name}:\n")
            by_cls = defaultdict(int)
            for r in recs:
                by_cls[r[3]] += 1
            for cls, n in sorted(by_cls.items(), key=lambda x: -x[1])[:30]:
                f.write(f"    {cls}: {n}\n")
    print(f"[Save] {out_dir / 'tcdb_train.fasta'}")
    print(f"[Save] {out_dir / 'tcdb_val.fasta'}")
    print(f"[Save] {out_dir / 'tcdb_test.fasta'}")
    print(f"[Save] {summary_path}")


if __name__ == "__main__":
    main()
