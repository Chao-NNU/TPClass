#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_table1_ablation.py
=======================

Render a 3-panel bar chart visualizing the 4-row TCDB held-out Table 1:
  - Panel 1: Accuracy
  - Panel 2: F1
  - Panel 3: MCC

The four conditions (cached/fresh x 650M/3B) are grouped along the x-axis;
each panel shows three colored bars per condition (Acc / F1 / MCC grouped
per ESM-2 x Mode combo).

Usage:
    python plot_table1_ablation.py --out /path/to/table1_bar.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 4 rows from Table 1 (TCDB held-out)
CONDITIONS = [
    # (esm, mode, Acc,    F1,     MCC)
    ("650M", "cached", 0.8704, 0.8724, 0.7410),
    ("650M", "fresh",  0.8434, 0.8375, 0.6893),
    ("3B",   "cached", 0.8775, 0.8815, 0.7564),
    ("3B",   "fresh",  0.8804, 0.8836, 0.7615),
]
METRICS = ["Accuracy", "F1", "MCC"]
COLORS = ["#3b6fb6", "#e07a3c", "#5fa55a"]  # Acc / F1 / MCC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output PNG path")
    args = ap.parse_args()

    n_cond = len(CONDITIONS)
    n_metric = len(METRICS)
    bar_w = 0.25
    x = np.arange(n_cond)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, metric in enumerate(METRICS):
        vals = [c[2 + i] for c in CONDITIONS]
        offset = (i - (n_metric - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=metric, color=COLORS[i],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7, rotation=0)

    labels = [f"{c[0]}\n{c[1]}" for c in CONDITIONS]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0.65, 0.92)
    ax.set_title("TCDB held-out v3 release: ESM-2 x Eval mode (n=4785)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[Save] {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
