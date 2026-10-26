#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_hybrid_v3.py
=================

Compute hybrid v3 (3B + 650M) strategies against the narrow GT:
  - 3B alone:    3B YES -> 3B family
  - 650M alone:  650M YES -> 650M family
  - Cascade:     3B YES -> 650M family (if 650M said YES), else 3B family
  - Union:       3B OR 650M YES -> 650M family (if available), else 3B family
  - Intersection: 3B AND 650M YES -> 650M family

Inputs:
  /data2/chao/transport/outputs/3b_12species/<sp>/final_report.csv   (3B)
  /data2/chao/transport/outputs/<sp>/final_report.csv                 (650M)
  /data2/chao/transport/data/uniprot_with_tc/uniprot_<sp>_transporters_with_tc.tsv  (GT)

Outputs:
  /data2/chao/transport/outputs/hybrid_v3/Table4_hybrid.csv
"""

import argparse
from pathlib import Path

import pandas as pd


# (3b_dir_name,    gt_fname,     650m_dir_name,  species_alias)
SPECIES = [
    ("E_coli_K12_MG1655",  "MG1655",     "MG1655",        "E_coli_K12_MG1655"),
    ("S_cerevisiae_S288C", "s288c",      "s288c",         "S_cerevisiae_S288C"),
    ("B_subtilis_168",     "bsub",       "bsub",          "B_subtilis_168"),
    ("L_lactis",           "IL1403",     "lacto",         "L_lactis"),
    ("C_glutamicum",       "cgl",        "cgl",           "C_glutamicum"),
    ("A_niger",            "aniger",     "aniger",        "A_niger"),
]


def to_4digit(tc):
    if not isinstance(tc, str) or not tc:
        return ""
    s = tc.strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    parts = s.split(".")
    if len(parts) >= 4:
        return ".".join(parts[:4])
    return s


def load_pred(path):
    """Load a v3 final_report.csv and return {query: tc_4digit} for YES rows."""
    df = pd.read_csv(path)
    yes = df[df["is_transport"].astype(str).str.upper() == "YES"].copy()
    pred = {}
    for _, r in yes.iterrows():
        pred[str(r["query"])] = to_4digit(str(r.get("tc_number", "")))
    return pred


def load_gt(path):
    """Load GT TSV and return {entry: tc_4digit} for non-empty entries."""
    df = pd.read_csv(path, sep="\t")
    gt = {}
    for _, r in df.iterrows():
        acc = str(r["Entry"])
        v = to_4digit(str(r["tc_family_4digit"]))
        if acc and v:
            gt[acc] = v
    return gt


def eval_strategy(yes_proteins, family_lookup, gt, strategy_name):
    """
    yes_proteins: set of query IDs (YES predictions)
    family_lookup: dict {query: 4-digit family} (from one or both models)
    gt: dict {entry: 4-digit family}
    """
    n_pred = len(yes_proteins)
    n_in_gt = 0
    n_correct = 0
    for q in yes_proteins:
        if q in gt:
            n_in_gt += 1
            fam = family_lookup.get(q, "")
            if fam and fam == gt[q]:
                n_correct += 1
    n_gt = len(gt)
    precision = n_correct / n_in_gt if n_in_gt > 0 else 0.0
    recall_fuzzy = n_in_gt / n_gt if n_gt > 0 else 0.0
    recall_strict = n_correct / n_gt if n_gt > 0 else 0.0
    return {
        "strategy": strategy_name,
        "n_pred": n_pred,
        "n_in_gt": n_in_gt,
        "n_correct": n_correct,
        "n_gt": n_gt,
        "precision": precision,
        "recall_fuzzy": recall_fuzzy,
        "recall_strict": recall_strict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/data2/chao/transport")
    args = ap.parse_args()
    base = Path(args.base)
    out_dir = base / "outputs" / "hybrid_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for sp_3b, gt_name, sp_650m_dir, sp_alias in SPECIES:
        pred_3b_path = base / "outputs" / "3b_12species" / sp_3b / "final_report.csv"
        pred_650m_path = base / "outputs" / sp_650m_dir / "final_report.csv"
        gt_path = base / "data" / "uniprot_with_tc" / f"uniprot_{gt_name}_transporters_with_tc.tsv"

        if not pred_3b_path.exists() or not pred_650m_path.exists() or not gt_path.exists():
            print(f"[SKIP] {sp_alias}: missing input")
            continue

        print(f"\n[{sp_alias}] loading predictions...")
        pred_3b = load_pred(pred_3b_path)
        pred_650m = load_pred(pred_650m_path)
        gt = load_gt(gt_path)
        print(f"  3B YES: {len(pred_3b)}, 650M YES: {len(pred_650m)}, GT (4-digit): {len(gt)}")

        # Strategy 1: 3B alone
        r1 = eval_strategy(set(pred_3b.keys()), pred_3b, gt, "3B alone")
        r1["species"] = sp_alias
        all_rows.append(r1)

        # Strategy 2: 650M alone
        r2 = eval_strategy(set(pred_650m.keys()), pred_650m, gt, "650M alone")
        r2["species"] = sp_alias
        all_rows.append(r2)

        # Strategy 3: Cascade (3B filter -> 650M family if both say YES, else 3B family)
        yes_cascade = set(pred_3b.keys())
        family_cascade = {}
        for q in yes_cascade:
            if q in pred_650m and pred_650m[q]:
                family_cascade[q] = pred_650m[q]  # 650M wins
            else:
                family_cascade[q] = pred_3b.get(q, "")  # fall back to 3B
        r3 = eval_strategy(yes_cascade, family_cascade, gt, "Cascade (3B->650M)")
        r3["species"] = sp_alias
        all_rows.append(r3)

        # Strategy 4: Union (3B or 650M YES)
        yes_union = set(pred_3b.keys()) | set(pred_650m.keys())
        family_union = {}
        for q in yes_union:
            if q in pred_650m and pred_650m[q]:
                family_union[q] = pred_650m[q]
            else:
                family_union[q] = pred_3b.get(q, "")
        r4 = eval_strategy(yes_union, family_union, gt, "Union (3B OR 650M)")
        r4["species"] = sp_alias
        all_rows.append(r4)

        # Strategy 5: Intersection (3B and 650M both YES)
        yes_inter = set(pred_3b.keys()) & set(pred_650m.keys())
        # 650M family for intersection
        r5 = eval_strategy(yes_inter, pred_650m, gt, "Intersection (3B AND 650M)")
        r5["species"] = sp_alias
        all_rows.append(r5)

        # Print summary
        print(f"  {'Strategy':35s} {'n_pred':>7s} {'n_in_gt':>8s} {'n_cor':>6s} {'prec':>7s} {'rec_f':>7s} {'rec_s':>7s}")
        for r in [r1, r2, r3, r4, r5]:
            print(f"  {r['strategy']:35s} {r['n_pred']:7d} {r['n_in_gt']:8d} {r['n_correct']:6d} "
                  f"{r['precision']:7.4f} {r['recall_fuzzy']:7.4f} {r['recall_strict']:7.4f}")

    # Save aggregated table
    df = pd.DataFrame(all_rows)
    out_path = out_dir / "Table4_hybrid.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Save] {out_path}")

    # Pivot for paper Table 4: rows = species, columns = (strategy, metric)
    pivot = df.pivot_table(index="species", columns="strategy",
                           values=["precision", "recall_fuzzy", "recall_strict"])
    pivot.columns = [f"{m}_{s}" for m, s in pivot.columns]
    pivot = pivot.reset_index()
    # Reorder columns
    strategies = ["3B alone", "650M alone", "Cascade (3B->650M)", "Union (3B OR 650M)", "Intersection (3B AND 650M)"]
    ordered_cols = ["species"]
    for s in strategies:
        for m in ["precision", "recall_fuzzy", "recall_strict"]:
            col = f"{m}_{s}"
            if col in pivot.columns:
                ordered_cols.append(col)
    pivot = pivot[ordered_cols]
    pivot_path = out_dir / "Table4_hybrid_pivot.csv"
    pivot.to_csv(pivot_path, index=False, encoding="utf-8-sig")
    print(f"[Save] {pivot_path}")
    print()
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
