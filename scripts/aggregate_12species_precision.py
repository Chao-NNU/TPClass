#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_12species_precision.py
=================================

Concatenate the per-species eval_family_precision.py outputs into a single
Table 2-style CSV, optionally with side-by-side 3B-vs-650M delta if a
reference CSV with the old 650M numbers is provided.

Usage:
    python aggregate_12species_precision.py \
        --in_dir  outputs/3b_12species \
        --suffix  precision.csv \
        --out_csv outputs/3b_12species/Table2_summary.csv \
        --old_csv outputs/old_650M_12species/Table2_summary.csv  # optional
"""

import argparse
from pathlib import Path

import pandas as pd


SPECIES_ORDER = [
    ("A. niger CBS 513.88",     "A_niger",            "Filamentous fungus (industrial)"),
    ("F. fujikuroi",            "F_fujikuroi",        "Filamentous fungus (industrial)"),
    ("C. glutamicum",           "C_glutamicum",       "Gram-positive bacterium (industrial)"),
    ("B. cinerea",              "B_cinerea",          "Filamentous fungus (plant pathogen)"),
    ("L. lactis IL1403",        "L_lactis",           "Gram-positive bacterium (industrial)"),
    ("Y. lipolytica",           "Y_lipolytica",       "Yeast (industrial)"),
    ("B. subtilis 168",         "B_subtilis_168",     "Gram-positive bacterium (model)"),
    ("K. marxianus",            "K_marxianus",        "Crabtree-negative yeast (industrial)"),
    ("S. cerevisiae S288C",     "S_cerevisiae_S288C", "Budding yeast (model)"),
    ("E. coli K-12 MG1655",     "E_coli_K12_MG1655",  "Gram-negative bacterium (model)"),
    ("E. coli BL21 (case)",     "E_coli_BL21",        "Gram-negative bacterium (case study)"),
    ("S. actuosus (extra)",     "S_actuosus",         "Actinobacterium (industrial)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",   required=True, help="directory with per-species precision.csv")
    ap.add_argument("--suffix",   default="precision.csv", help="file suffix (default: precision.csv)")
    ap.add_argument("--out_csv",  required=True, help="output aggregated Table 2 CSV")
    ap.add_argument("--old_csv",  default="",    help="optional 650M reference Table 2 CSV for delta column")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    rows = []
    for label, dirname, sp_type in SPECIES_ORDER:
        csv = in_dir / dirname / args.suffix
        if not csv.exists():
            rows.append({
                "species": label, "type": sp_type,
                "n_pred_total": 0, "n_pred_yes": 0, "n_pred_yes_with_4digit": 0,
                "n_matched_to_gt": 0, "n_correct": 0,
                "precision_4digit": None, "coverage_4digit": None,
                "status": "MISSING",
            })
            continue
        d = pd.read_csv(csv).iloc[0].to_dict()
        rows.append({
            "species": label, "type": sp_type,
            "n_pred_total": int(d.get("n_pred_total", 0)),
            "n_pred_yes": int(d.get("n_pred_yes", 0)),
            "n_pred_yes_with_4digit": int(d.get("n_pred_yes_with_4digit", 0)),
            "n_matched_to_gt": int(d.get("n_matched_to_gt", 0)),
            "n_correct": int(d.get("n_correct", 0)),
            "precision_4digit": float(d.get("precision_4digit", 0)) if d.get("precision_4digit") is not None else None,
            "coverage_4digit":  float(d.get("coverage_4digit", 0))  if d.get("coverage_4digit")  is not None else None,
            "status": "OK",
        })

    out = pd.DataFrame(rows)
    if args.old_csv and Path(args.old_csv).exists():
        old = pd.read_csv(args.old_csv)
        out = out.merge(old[["species", "precision_4digit"]].rename(columns={"precision_4digit": "precision_4digit_650M"}),
                        on="species", how="left")
        out["delta_pp"] = (out["precision_4digit"] - out["precision_4digit_650M"]).round(4) * 100
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Save] {args.out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
