#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_from_final_report.py
=========================

Regenerate the 9 distribution plots + dashboard summary from an existing
v3 final_report.csv, WITHOUT re-running the (slow) ESM-2 inference.

This is a thin wrapper around plot_all() defined inside predict_transport_family_v3.py:
we import the plotting functions from the v3 script, load the final_report.csv
as a DataFrame, and call plot_all(df, plot_dir).

Usage:
    python plot_from_final_report.py \
        --final_report /path/to/final_report.csv \
        --plot_dir      /path/to/output/plots
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_v3_plot_module(v3_script: str):
    """Import predict_transport_family_v3.py as a module to reuse plot_all()."""
    spec = importlib.util.spec_from_file_location("v3_predict", v3_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3_script", default="/data2/chao/transport/scripts/predict_transport_family_v3.py",
                    help="Path to predict_transport_family_v3.py (default: %(default)s)")
    ap.add_argument("--final_report", required=True,
                    help="Path to v3 final_report.csv (output of -out <dir>/final_report.csv)")
    ap.add_argument("--plot_dir", required=True,
                    help="Directory to write 9 PNG plots + dashboard summary into")
    args = ap.parse_args()

    final_csv = Path(args.final_report)
    if not final_csv.exists():
        sys.exit(f"[ERROR] final_report not found: {final_csv}")
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] {final_csv}")
    df = pd.read_csv(final_csv)
    print(f"[Load] {len(df)} rows  (cols: {list(df.columns)[:8]} ...)")

    v3 = _load_v3_plot_module(args.v3_script)
    v3.plot_all(df, plot_dir)

    print(f"\n[Done] Plots written to: {plot_dir}")
    for png in sorted(plot_dir.glob("*.png")):
        print(f"  - {png.name}  ({png.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
