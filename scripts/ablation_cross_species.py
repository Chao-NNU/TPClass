#!/usr/bin/env python
"""
ablation_cross_species.py
=========================

Aggregate ablation_runner.py results across multiple species into one table.

Usage:
    python ablation_cross_species.py \\
        --ablation_root /data2/chao/transport/outputs/ablation/ \\
        --out_dir /data2/chao/transport/outputs/ablation/cross_species/
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Aggregate ablation across species")
    parser.add_argument("--ablation_root", required=True,
                        help="Root dir containing per-species ablation_results.csv files")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.ablation_root)

    # Find all per-species ablation_results.csv
    species_files = sorted(root.glob("*/ablation_results.csv"))
    if not species_files:
        print(f"[ERROR] No ablation_results.csv found under {root}")
        return

    print(f"[Found] {len(species_files)} species")
    for sf in species_files:
        print(f"  - {sf.parent.name}")

    # Combine
    dfs = []
    for sf in species_files:
        d = pd.read_csv(sf)
        d.insert(0, "species", sf.parent.name)
        dfs.append(d)
    combined = pd.concat(dfs, ignore_index=True)

    # Save combined CSV
    combined.to_csv(out_dir / "ablation_cross_species.csv", index=False, encoding="utf-8-sig")
    print(f"\n[Save] {out_dir / 'ablation_cross_species.csv'}")

    # Mean F1 per variant (across species)
    metric_cols = [c for c in combined.columns if c.endswith("_f1")]
    pivot = combined.groupby("variant")[metric_cols].mean().reset_index()
    pivot.to_csv(out_dir / "ablation_mean_f1.csv", index=False, encoding="utf-8-sig")

    # Markdown summary
    md = ["# Cross-Species Ablation Summary", "",
          "## Mean F1 across species", "",
          "| Variant | " + " | ".join(metric_cols) + " |",
          "|---" * (len(metric_cols) + 1) + "|"]
    for _, r in pivot.iterrows():
        md.append("| " + r["variant"] + " | " +
                  " | ".join(f"{r[c]:.3f}" for c in metric_cols) + " |")
    (out_dir / "ablation_mean_f1.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[Save] {out_dir / 'ablation_mean_f1.md'}")

    # Plot: bar chart of mean F1 per variant, per level
    levels = ["class", "subclass", "full"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot))
    width = 0.25
    colors = {"class": "#4C72B0", "subclass": "#55A868", "full": "#C44E52"}
    for i, lvl in enumerate(levels):
        col = f"{lvl}_f1"
        if col in pivot.columns:
            ax.bar(x + (i - 1) * width, pivot[col], width,
                   label=f"{lvl} F1", color=colors[lvl])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["variant"], rotation=15, ha="right")
    ax.set_ylabel("Mean F1 (across species)")
    ax.set_title("Cross-Species Ablation — Mean F1")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_cross_species_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Save] {out_dir / 'ablation_cross_species_f1.png'}")


if __name__ == "__main__":
    main()
