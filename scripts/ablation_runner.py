#!/usr/bin/env python
"""
ablation_runner.py
==================

Evaluate v3 ablation variants against UniProt ground truth.

For each variant (full / -Phase1 / -s4pred / -MLP / -XGB / top-k=5 / +SW), the runner:
  1. Loads the final_report.csv
  2. Loads UniProt GT for the species
  3. Computes per-variant metrics:
     - Precision / Recall / F1 at protein level (transporter YES vs NO)
     - Precision / Recall / F1 at TC sub-family level (4-digit)
     - Precision / Recall / F1 at TC class level (3-digit)
  4. Outputs:
     - ablation_results.csv  (one row per variant)
     - ablation_results.md    (paper-style table)
     - ablation_bar.png       (precision comparison)
     - ablation_bar.png       (recall comparison)

Usage:
    python ablation_runner.py \\
        --gt /data2/chao/transport/data/uniprot_max_transporters.tsv \\
        --variants \\
            "Full v3|/path/to/full/final_report.csv" \\
            "-Phase1|/path/to/no_phase1/final_report.csv" \\
            "-s4pred|/path/to/no_s4pred/final_report.csv" \\
            "-MLP|/path/to/pure_mlp/final_report.csv" \\
            "-XGB|/path/to/pure_xgb/final_report.csv" \\
        --out_dir /path/to/output/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── TC level helpers ─────────────────────────────────────────────────────
def tc_class(tc):
    """3-digit TC class (e.g., 2.A.1 -> '2')."""
    if not isinstance(tc, str) or not tc or tc == "nan":
        return None
    return tc.split(".")[0].strip()


def tc_subclass(tc):
    """4-digit TC subclass (e.g., 2.A.1.12 -> '2.A.1')."""
    if not isinstance(tc, str) or not tc or tc == "nan":
        return None
    parts = tc.split(".")
    return ".".join(parts[:2]).strip() if len(parts) >= 2 else parts[0].strip()


def tc_full(tc):
    """5-digit TC (full, e.g., 2.A.1.12.4)."""
    if not isinstance(tc, str) or not tc or tc == "nan":
        return None
    return tc.strip()


# ── Metrics ──────────────────────────────────────────────────────────────
def compute_metrics(v3_df, gt_df, level="subclass", threshold=0.5):
    """
    Compute precision/recall/F1 against UniProt GT.

    Parameters
    ----------
    v3_df : pd.DataFrame
        v3 final_report.csv with columns: query, tc_number, transport_prob, is_transport
    gt_df : pd.DataFrame
        UniProt GT with columns: accession, tc_number (or 'accession' + 'tc')
    level : str
        'class' (3-digit), 'subclass' (4-digit), or 'full' (5-digit)
    threshold : float
        transport_prob threshold for binary classification (only used if v3 uses prob)
    """
    if level == "class":
        v3_col = lambda t: tc_class(t)
        gt_col = lambda t: tc_class(t)
    elif level == "subclass":
        v3_col = lambda t: tc_subclass(t)
        gt_col = lambda t: tc_subclass(t)
    else:
        v3_col = lambda t: tc_full(t)
        gt_col = lambda t: tc_full(t)

    # Normalize v3 predictions
    v3_df = v3_df.copy()
    # robust v3 TC column detection
    v3_tc_col = None
    for cand in ["tc_number", "top_tc_number", "TC", "tc", "top_family_tc"]:
        if cand in v3_df.columns:
            v3_tc_col = cand
            break
    if v3_tc_col is None:
        raise KeyError(
            f"v3 final_report.csv has no TC column. Tried: tc_number / top_tc_number / TC / tc / top_family_tc. "
            f"Actual columns: {list(v3_df.columns)}"
        )
    v3_df["tc_norm"] = v3_df[v3_tc_col].apply(v3_col)
    v3_df["is_yes"] = (v3_df["is_transport"] == "YES")
    v3_df["tc_norm"] = v3_df["tc_norm"].replace(["", "nan", None], np.nan)
    # robust v3 query column detection
    if "query" not in v3_df.columns:
        for cand in ["qseqid", "name", "protein_id", "accession"]:
            if cand in v3_df.columns:
                v3_df = v3_df.rename(columns={cand: "query"})
                break

    # Normalize GT
    gt_df = gt_df.copy()
    # robust TC column detection (try common names)
    gt_tc_col = None
    for cand in ["tc_number", "TC_number", "TC", "tc", "TC_id", "TCDB", "transporter_class"]:
        if cand in gt_df.columns:
            gt_tc_col = cand
            break
    if gt_tc_col is None:
        raise KeyError(
            f"GT file has no TC column. Tried: tc_number / TC_number / TC / tc / TC_id / TCDB / transporter_class. "
            f"Actual columns: {list(gt_df.columns)}"
        )
    # robust accession column detection
    if "accession" not in gt_df.columns:
        for cand in ["Entry", "entry", "uniprot_id", "protein_id", "acc"]:
            if cand in gt_df.columns:
                gt_df = gt_df.rename(columns={cand: "accession"})
                break
    if "accession" not in gt_df.columns:
        raise KeyError(
            f"GT file has no accession column. Tried: accession / Entry / entry / uniprot_id / protein_id / acc. "
            f"Actual columns: {list(gt_df.columns)}"
        )
    gt_df["tc_norm"] = gt_df[gt_tc_col].apply(gt_col)
    gt_df["tc_norm"] = gt_df["tc_norm"].replace(["", "nan", None], np.nan)

    # v3 YES predictions with assigned TC
    v3_yes = v3_df[v3_df["is_yes"] & v3_df["tc_norm"].notna()]
    pred_set = set(zip(v3_yes["query"], v3_yes["tc_norm"]))

    # GT set
    gt_set = set(zip(gt_df["accession"], gt_df["tc_norm"]))

    if not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": len(pred_set), "n_gt": 0}

    # TP / FP / FN
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": len(pred_set),
        "n_gt": len(gt_set),
        "n_tp": tp,
        "n_fp": fp,
        "n_fn": fn,
    }


# ── Plot ────────────────────────────────────────────────────────────────
def plot_comparison(results_df, out_path, metric="f1"):
    """Bar chart: x = variant, y = metric (per level)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    levels = ["class", "subclass", "full"]
    x = np.arange(len(results_df))
    width = 0.25
    colors = {"class": "#4C72B0", "subclass": "#55A868", "full": "#C44E52"}

    for i, level in enumerate(levels):
        col = f"{level}_{metric}"
        if col in results_df.columns:
            ax.bar(x + (i - 1) * width, results_df[col], width, label=level, color=colors[level])

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["variant"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Ablation Study — {metric.upper()} comparison")
    ax.set_ylim(0, 1.0)
    ax.legend(title="TC level", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out_path}")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate v3 ablation variants against UniProt GT")
    parser.add_argument("--gt", required=True, help="UniProt ground truth TSV (columns: accession, tc_number)")
    parser.add_argument("--variants", nargs="+", required=True,
                        help='List of "name|/path/to/final_report.csv" pairs')
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="transport_prob threshold (default 0.5)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load GT
    print(f"[Load] GT: {args.gt}")
    gt_df = pd.read_csv(args.gt, sep="\t")
    print(f"  GT entries: {len(gt_df)}")

    # Evaluate each variant
    rows = []
    for spec in args.variants:
        if "|" not in spec:
            print(f"[WARN] Bad spec: {spec} (expected 'name|path')")
            continue
        name, csv_path = spec.split("|", 1)
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing CSV: {csv_path}")
            continue
        print(f"\n[Variant] {name} <- {csv_path}")
        v3_df = pd.read_csv(csv_path)

        for level in ["class", "subclass", "full"]:
            m = compute_metrics(v3_df, gt_df, level=level, threshold=args.threshold)
            row = {"variant": name, "level": level, **m}
            rows.append(row)
            print(f"  [{level:8s}] P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                  f"(TP={m.get('n_tp',0)} FP={m.get('n_fp',0)} FN={m.get('n_fn',0)})")

    if not rows:
        print("[ERROR] No valid variants processed")
        return

    # Build wide-format results table
    results_df = pd.DataFrame(rows)
    wide = results_df.pivot(index="variant", columns="level", values=["precision", "recall", "f1"])
    wide.columns = [f"{lvl}_{m}" for m, lvl in wide.columns]
    wide = wide.reset_index()

    # Save CSV
    csv_out = out_dir / "ablation_results.csv"
    wide.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\n[Save] {csv_out}")

    # Save Markdown
    md_lines = ["# v3 Ablation Study Results", "",
                "| Variant | Class P | Class R | Class F1 | Subclass P | Subclass R | Subclass F1 | Full P | Full R | Full F1 |",
                "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in wide.iterrows():
        md_lines.append(
            f"| {r['variant']} "
            f"| {r.get('class_precision', 0):.3f} | {r.get('class_recall', 0):.3f} | {r.get('class_f1', 0):.3f} "
            f"| {r.get('subclass_precision', 0):.3f} | {r.get('subclass_recall', 0):.3f} | {r.get('subclass_f1', 0):.3f} "
            f"| {r.get('full_precision', 0):.3f} | {r.get('full_recall', 0):.3f} | {r.get('full_f1', 0):.3f} |"
        )
    md_out = out_dir / "ablation_results.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[Save] {md_out}")

    # Plot
    plot_comparison(wide, out_dir / "ablation_f1.png", metric="f1")
    plot_comparison(wide, out_dir / "ablation_precision.png", metric="precision")
    plot_comparison(wide, out_dir / "ablation_recall.png", metric="recall")
    print(f"\n[Done] {out_dir}")


if __name__ == "__main__":
    main()
