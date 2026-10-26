#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_family_precision.py
========================

Compute family-level (4-digit TC) precision of v3 predictions against a
UniProt ground-truth TSV.

Inputs:
  --pred_csv  <path>     v3 final_report.csv with at least these columns:
                          query            (protein accession from fasta header)
                          top_family       (4-digit TC, e.g. 1.A.9.1)
                          is_transport     (YES / NO)
  --gt_tsv    <path>     UniProt ground truth TSV with at least these columns:
                          accession        (matches --pred_csv query)
                          tc_family_4digit (4-digit TC, e.g. 1.A.9.1)
  --out_csv   <path>     output summary CSV (1 row: n_pred, n_match, precision, ...)

Metric:
  precision_4digit = (v3 predictions whose top_family matches GT tc_family_4digit)
                     / (v3 predictions that are Phase 1 YES AND have a 4-digit TC prediction)

  Coverage = (v3 YES predictions with 4-digit TC) / (v3 YES predictions)

Usage:
    python eval_family_precision.py \
        --pred_csv outputs/3b_12species/K_marxianus/final_report.csv \
        --gt_tsv   data/uniprot_gt/K_marxianus.tsv \
        --out_csv  outputs/3b_12species/K_marxianus/precision.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def to_4digit(tc):
    """Truncate a 4- or 5-segment TCDB ID to its 4-digit family form.
    e.g. '2.A.1.2.98' -> '2.A.1.2',  '1.A.9.1' -> '1.A.9.1'.
    Empty / malformed / 'nan' input returns "".
    """
    if not isinstance(tc, str) or not tc:
        return ""
    s = tc.strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    parts = s.split(".")
    if len(parts) >= 4:
        return ".".join(parts[:4])
    return s  # short string, return as-is (will be a non-match)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_csv", required=True, help="v3 final_report.csv")
    ap.add_argument("--gt_tsv",   required=True, help="UniProt ground truth TSV (accession + tc_family_4digit)")
    ap.add_argument("--out_csv",  required=True, help="output precision summary CSV")
    ap.add_argument("--query_col",        default="query",            help="column in pred_csv with accession (default: query)")
    ap.add_argument("--family_col",       default="tc_number",        help="column in pred_csv with v3 TC (default: tc_number, e.g. '2.A.1.2.98'; will be truncated to 4-digit family)")
    ap.add_argument("--is_yes_col",       default="is_transport",     help="column in pred_csv with YES/NO flag (default: is_transport)")
    ap.add_argument("--gt_acc_col",       default="accession",        help="column in gt_tsv with accession (default: accession)")
    ap.add_argument("--gt_family_col",    default="tc_family_4digit", help="column in gt_tsv with TCDB ID (default: tc_family_4digit; will be truncated to 4-digit family)")
    ap.add_argument("--name", default="", help="optional species name to put in summary")
    ap.add_argument("--min_transport_prob", type=float, default=0.0,
                    help="min transport_prob threshold to keep a YES prediction (default 0.0 = no filter)")
    ap.add_argument("--transport_prob_col", default="transport_prob",
                    help="column with transport_prob (default: transport_prob)")
    ap.add_argument("--min_top1_sim", type=float, default=0.0,
                    help="min Phase 2 top1_similarity (default 0.0 = no filter)")
    ap.add_argument("--min_votes", type=int, default=0,
                    help="min Phase 2 top_family_votes (default 0 = no filter)")
    ap.add_argument("--min_top1_gap", type=float, default=0.0,
                    help="min Phase 2 top1_sim_gap (default 0.0 = no filter)")
    ap.add_argument("--top1_sim_col", default="top1_similarity",
                    help="column for top1 similarity (default: top1_similarity)")
    ap.add_argument("--votes_col", default="top_family_votes",
                    help="column for top family votes (default: top_family_votes)")
    ap.add_argument("--gap_col", default="top1_sim_gap",
                    help="column for top1_sim_gap (default: top1_sim_gap)")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)
    gt   = pd.read_csv(args.gt_tsv, sep="\t")

    print(f"[Load] pred {len(pred)} rows  ({Path(args.pred_csv).name})")
    print(f"[Load] gt   {len(gt)} rows  ({Path(args.gt_tsv).name})")
    if args.min_transport_prob > 0:
        print(f"[Filter] min_transport_prob >= {args.min_transport_prob}")
    if args.min_top1_sim > 0:
        print(f"[Filter] min_top1_sim >= {args.min_top1_sim}")
    if args.min_votes > 0:
        print(f"[Filter] min_votes >= {args.min_votes}")
    if args.min_top1_gap > 0:
        print(f"[Filter] min_top1_gap >= {args.min_top1_gap}")

    # restrict to Phase 1 YES predictions
    is_yes = pred[args.is_yes_col].astype(str).str.upper() == "YES"
    yes = pred[is_yes].copy()
    print(f"[Pred] Phase 1 YES: {len(yes)} / {len(pred)}")

    # apply transport_prob threshold (if set)
    if args.min_transport_prob > 0 and args.transport_prob_col in yes.columns:
        before = len(yes)
        yes = yes[yes[args.transport_prob_col].astype(float) >= args.min_transport_prob].copy()
        print(f"[Filter] transport_prob >= {args.min_transport_prob}: {len(yes)} / {before} kept")

    # apply Phase 2 confidence filters
    for thr, col, label in [
        (args.min_top1_sim, args.top1_sim_col, "top1_sim"),
        (args.min_votes,    args.votes_col,    "votes"),
        (args.min_top1_gap, args.gap_col,      "top1_gap"),
    ]:
        if thr > 0 and col in yes.columns:
            before = len(yes)
            yes = yes[pd.to_numeric(yes[col], errors="coerce") >= thr].copy()
            print(f"[Filter] {label} >= {thr}: {len(yes)} / {before} kept")

    # restrict to predictions with a 4- or 5-digit TC (e.g. "2.A.1.2" or "2.A.1.2.98")
    has_family = yes[args.family_col].astype(str).str.contains(r"\d+\.[A-Z]\.\d+\.\d+(\.\d+)?", na=False, regex=True)
    yes_with_family = yes[has_family].copy()
    print(f"[Pred] YES with TC (4- or 5-digit): {len(yes_with_family)}")

    # build accession -> 4-digit family lookup
    gt_lookup = {}
    for _, grow in gt.iterrows():
        acc = str(grow[args.gt_acc_col])
        gt4 = to_4digit(str(grow[args.gt_family_col]))
        if acc and gt4:
            gt_lookup[acc] = gt4
    print(f"[GT]   {len(gt_lookup)} unique accessions with 4-digit TC")

    # match at 4-digit family level
    correct = 0
    n_eval = 0
    for _, row in yes_with_family.iterrows():
        acc = str(row[args.query_col])
        pred4 = to_4digit(str(row[args.family_col]))
        if acc in gt_lookup and pred4:
            n_eval += 1
            if pred4 == gt_lookup[acc]:
                correct += 1

    precision_4digit = correct / n_eval if n_eval > 0 else 0.0
    coverage = len(yes_with_family) / len(yes) if len(yes) > 0 else 0.0
    n_gt = len(gt_lookup)
    recall_fuzzy = n_eval / n_gt if n_gt > 0 else 0.0
    recall_strict = correct / n_gt if n_gt > 0 else 0.0

    print(f"\n[Result] {args.name or Path(args.pred_csv).parent.name}")
    print(f"  Phase 1 YES with TC:          {len(yes_with_family)}")
    print(f"  Matched against GT (4-digit): {n_eval}")
    print(f"  Correct (4-digit match):     {correct}")
    print(f"  GT size (4-digit):            {n_gt}")
    print(f"  Precision (4-digit TC):      {precision_4digit:.4f}")
    print(f"  Coverage (TC assign):         {coverage:.4f}")
    print(f"  Recall (fuzzy, n_matched/GT): {recall_fuzzy:.4f}")
    print(f"  Recall (strict, correct/GT):  {recall_strict:.4f}")

    # write summary
    out = pd.DataFrame([{
        "name": args.name or Path(args.pred_csv).parent.name,
        "n_pred_total": len(pred),
        "n_pred_yes": len(yes),
        "n_pred_yes_with_4digit": len(yes_with_family),
        "n_matched_to_gt": n_eval,
        "n_correct": correct,
        "n_gt": n_gt,
        "precision_4digit": precision_4digit,
        "coverage_4digit": coverage,
        "recall_fuzzy": recall_fuzzy,
        "recall_strict": recall_strict,
    }])
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[Save] {args.out_csv}")


if __name__ == "__main__":
    main()
