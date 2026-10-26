#!/usr/bin/env python
"""
evaluate_heldout.py
===================

Evaluate v3 ablation variants on a balanced held-out set (TCDB positives
+ UniProt random negatives, both pre-split 90/10 with extract_features_for_retrain.py).

Each sequence is labeled via the header prefix added by extract_features_for_retrain.py:
  - `pos__...` → GT label 1 (transporter)
  - `neg__...` → GT label 0 (non-transporter)

Predicted label comes from the v3 final_report.csv's `is_transport` field
(YES → 1, anything else → 0). TC-number match is reported as a secondary
metric for the transporter-positive subset (memorized from training, so
saturated for `pos__`, useful for sanity-checking recall).

Outputs:
  <out>.csv     — long-form per-variant metrics
  <out>.md      — pretty markdown table (Bioinformatics style)

Usage:
    python evaluate_heldout.py \
        --heldout_fasta /data2/chao/transport/data/tcdb_val_balanced.fasta \
        --variants "full_v3|outputs/ablation/val_full_v3/final_report.csv" \
                  "no_phase1|outputs/ablation/val_no_phase1/final_report.csv" \
                  "no_s4pred|outputs/ablation/val_no_s4pred/final_report.csv" \
                  "pure_mlp|outputs/ablation/val_pure_mlp/final_report.csv" \
                  "pure_xgb|outputs/ablation/val_pure_xgb/final_report.csv" \
        --out outputs/ablation/val_table.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def parse_balanced_header(header: str):
    """
    Parse balanced val header written by extract_features_for_retrain.py.
    Format: `>pos__gnl|TC-DB|...` or `>neg__sp|P12345|...`
    Returns (gt_label, raw_accession) where gt_label ∈ {0, 1}.
    """
    h = header.lstrip(">").strip()
    if h.startswith("pos__"):
        return 1, h[len("pos__"):]
    elif h.startswith("neg__"):
        return 0, h[len("neg__"):]
    return None, h


def get_v3_acc(query: str):
    """
    Extract a stable accession key from a v3 final_report `query` field.
    The query field preserves the prefix added by extract_features_for_retrain.py
    (e.g. `pos__gnl|TC-DB|P12345|...`), so we strip that first.
    """
    if not isinstance(query, str) or not query:
        return None
    q = query.strip()
    if q.startswith("pos__") or q.startswith("neg__"):
        q = q.split("__", 1)[1] if "__" in q else q
    # sp|ACC|... or tr|ACC|...
    m = re.match(r"^(?:sp|tr)\|([A-Z0-9]+)", q)
    if m:
        return m.group(1)
    # gnl|TC-DB|ACC|TC ...
    if q.startswith("gnl|TC-DB|"):
        parts = q.split("|")
        if len(parts) >= 3:
            return parts[2]
    # WP_ / YP_ / NP_ with version
    m = re.match(r"^([WYN]P_[0-9]+(?:\.[0-9]+)?)", q)
    if m:
        return m.group(1)
    return q.split()[0]


def main():
    ap = argparse.ArgumentParser(description="Evaluate v3 ablation on balanced held-out (pos + neg)")
    ap.add_argument("--heldout_fasta", required=True,
                    help="Balanced val fasta with 'pos__'/'neg__' prefix on each header")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="space-separated 'name|/path/to/final_report.csv' pairs")
    ap.add_argument("--out", required=True,
                    help="output CSV path")
    args = ap.parse_args()

    # ====== Load GT (binary) from held-out fasta ======
    gt = {}        # acc -> label (0/1)
    n_pos = n_neg = 0
    with open(args.heldout_fasta) as f:
        for line in f:
            line = line.rstrip()
            if not line.startswith(">"):
                continue
            label, raw = parse_balanced_header(line)
            if label is None:
                continue
            # We key on the raw accession (the prefix was added by our pipeline)
            acc = get_v3_acc(raw)
            if not acc:
                continue
            # If duplicates, keep first; in practice each accession appears once.
            gt.setdefault(acc, label)
            if label == 1: n_pos += 1
            else:          n_neg += 1

    print(f"[GT] Parsed {len(gt)} entries from {args.heldout_fasta}  (pos={n_pos}, neg={n_neg})")
    if len(gt) == 0:
        print("[ERROR] No GT entries parsed! Check heldout fasta header format:")
        print("        expected: >pos__<orig_header>  or  >neg__<orig_header>")
        with open(args.heldout_fasta) as f:
            for i, line in enumerate(f):
                if i >= 3: break
                print(f"        sample: {line.rstrip()}")
        sys.exit(1)

    # ====== Evaluate each variant ======
    rows = []
    for spec in args.variants:
        if "|" not in spec:
            print(f"[WARN] skipping bad spec (no '|'): {spec}")
            continue
        name, csv_path = spec.split("|", 1)
        if not Path(csv_path).exists():
            print(f"[WARN] missing csv: {csv_path}")
            continue

        print(f"\n[Variant] {name}  ←  {csv_path}")
        df = pd.read_csv(csv_path)
        df["acc"]      = df["query"].apply(get_v3_acc)
        df["gt_label"] = df["acc"].map(gt)
        df["is_yes"]   = (df["is_transport"].astype(str).str.upper() == "YES")

        # Restrict to rows whose accession was in our balanced GT
        in_gt = df[df["gt_label"].notna()].copy()
        in_gt["gt_label"] = in_gt["gt_label"].astype(int)
        n_in_gt = len(in_gt)
        n_pos_in_gt  = int((in_gt["gt_label"] == 1).sum())
        n_neg_in_gt  = int((in_gt["gt_label"] == 0).sum())

        # Binary classification confusion matrix
        # GT=1 & pred=YES → TP   (correctly identifies transporter)
        # GT=0 & pred=NO  → TN   (correctly rejects non-transporter)
        # GT=0 & pred=YES → FP   (false alarm)
        # GT=1 & pred=NO  → FN   (missed transporter)
        tp = int(((in_gt["gt_label"] == 1) & (in_gt["is_yes"])).sum())
        tn = int(((in_gt["gt_label"] == 0) & (~in_gt["is_yes"])).sum())
        fp = int(((in_gt["gt_label"] == 0) & (in_gt["is_yes"])).sum())
        fn = int(((in_gt["gt_label"] == 1) & (~in_gt["is_yes"])).sum())

        total      = tp + tn + fp + fn
        accuracy   = (tp + tn) / total if total > 0 else 0.0
        precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1         = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc_num = (tp * tn - fp * fn)
        mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0

        # Per-class accuracy (balanced accuracy)
        sensitivity = recall
        bal_acc = 0.5 * (sensitivity + specificity)

        rows.append({
            "variant": name,
            "n_in_gt": n_in_gt,
            "n_pos_in_gt": n_pos_in_gt,
            "n_neg_in_gt": n_neg_in_gt,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": accuracy,
            "balanced_accuracy": bal_acc,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "mcc": mcc,
        })
        print(f"  in-GT rows: {n_in_gt}  (pos={n_pos_in_gt}, neg={n_neg_in_gt})")
        print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"  Acc={accuracy:.4f}  BalAcc={bal_acc:.4f}  P={precision:.4f}  R={recall:.4f}  "
              f"F1={f1:.4f}  MCC={mcc:.4f}")

    if not rows:
        print("[ERROR] No variants evaluated")
        sys.exit(1)

    # ====== Save CSV ======
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n[Save] {args.out}")

    # ====== Save Markdown ======
    md_path = Path(args.out).with_suffix(".md")
    md = ["# Balanced Held-out Ablation Results", "",
          "Held-out set: TCDB (10% stratified split) + UniProt random non-transport "
          "(CD-HIT 90% against TCDB), 1:1 mixed. Binary classification (transporter vs. non-transporter).",
          "",
          "| Variant | n | TP | TN | FP | FN | Acc | BalAcc | Precision | Recall | F1 | MCC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['n_in_gt']} | {r['TP']} | {r['TN']} | {r['FP']} | {r['FN']} | "
            f"{r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['mcc']:.4f} |"
        )
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[Save] {md_path}")


if __name__ == "__main__":
    main()
