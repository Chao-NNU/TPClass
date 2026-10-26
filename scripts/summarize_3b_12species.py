#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summarize_3b_12species.py
==========================

Dump a quick cross-species summary table from all 12 species' final_report.csv.
Pure CSV analysis, no GPU/UniProt needed.

Usage:
    python summarize_3b_12species.py \
        --in_dir  /data2/chao/transport/outputs/3b_12species \
        --out_csv /tmp/3b_12species_summary.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


SPECIES_ORDER = [
    ("A. niger CBS 513.88",     "A_niger",            "Filamentous fungus"),
    ("F. fujikuroi",            "F_fujikuroi",        "Filamentous fungus"),
    ("B. cinerea",              "B_cinerea",          "Filamentous fungus"),
    ("Y. lipolytica",           "Y_lipolytica",       "Yeast (industrial)"),
    ("K. marxianus",            "K_marxianus",        "Yeast (industrial)"),
    ("S. cerevisiae S288C",     "S_cerevisiae_S288C", "Yeast (model)"),
    ("C. glutamicum",           "C_glutamicum",       "Gram+ bacterium (industrial)"),
    ("L. lactis IL1403",        "L_lactis",           "Gram+ bacterium (industrial)"),
    ("B. subtilis 168",         "B_subtilis_168",     "Gram+ bacterium (model)"),
    ("E. coli K-12 MG1655",     "E_coli_K12_MG1655",  "Gram- bacterium (model)"),
    ("E. coli BL21 (case)",     "E_coli_BL21",        "Gram- bacterium (case study)"),
    ("S. actuosus (extra)",     "S_actuosus",         "Actinobacterium"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    rows = []
    for label, dirname, sp_type in SPECIES_ORDER:
        csv = in_dir / dirname / "final_report.csv"
        row = {
            "species": label, "type": sp_type, "status": "MISSING",
            "n_total": 0, "n_yes": 0, "yes_pct": 0.0,
            "sim_mean": None, "sim_median": None, "n_sim_low": 0,
            "top1_class": None, "top1_class_count": 0,
        }
        if not csv.exists():
            rows.append(row)
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            row["status"] = f"ERROR: {e}"
            rows.append(row)
            continue

        n = len(df)
        is_yes = df["is_transport"].astype(str).str.upper() == "YES"
        n_yes = int(is_yes.sum())
        row.update({
            "status": "OK",
            "n_total": n,
            "n_yes": n_yes,
            "yes_pct": round(100 * n_yes / n, 1) if n else 0.0,
        })

        # sim stats on YES
        if "top1_similarity" in df.columns:
            sim = df.loc[is_yes, "top1_similarity"].astype(float)
            if len(sim) > 0:
                row["sim_mean"]  = round(float(sim.mean()), 4)
                row["sim_median"] = round(float(sim.median()), 4)
                row["n_sim_low"] = int((sim < 0.7).sum())

        # top1 TC class distribution (first char of tc_family like "1.A.9.1" -> "1")
        if "tc_family" in df.columns:
            topfam = df.loc[is_yes, "tc_family"].astype(str)
            cls = topfam.str.extract(r"^(\d+)\.", expand=False)
            if len(cls) > 0 and not cls.dropna().empty:
                vc = cls.dropna().value_counts()
                if len(vc) > 0:
                    row["top1_class"] = str(vc.index[0])
                    row["top1_class_count"] = int(vc.iloc[0])

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Save] {args.out_csv}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
