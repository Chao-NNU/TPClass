#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
case_study_extract.py
======================

Extract case study numbers from 3B v3 final_report.csv:
  1. K. marxianus: top 10 YES + any 2.A.1.12.x (Jen1 family) hits
  2. E. coli BL21: v3 predictions for the 5 known HMO-related proteins

Output is plain markdown-ish text the user can paste into the paper.

Usage:
    python case_study_extract.py
"""

from pathlib import Path

import pandas as pd


BASE = Path("/data2/chao/transport")
HMO_PROTEINS = [
    ("WP_000108459.1", "NanT", "2.A.1.12.1"),
    ("WP_000124289.1", "KpsM", "3.A.1.101.1"),
    ("WP_000291549.1", "LacY", "2.A.1.5.1"),
    ("WP_000590251.1", "KpsM_paralog", "3.A.1.101.1"),
    ("WP_001023126.1", "PTS-GlcNAc", "4.A.1.1.2"),
]


def main():
    print("=" * 78)
    print("K. marxianus 3B v3 — top 10 YES predictions by transport_prob")
    print("=" * 78)
    km = pd.read_csv(BASE / "outputs" / "3b_12species" / "K_marxianus" / "final_report.csv")
    yes = km[km["is_transport"].astype(str).str.upper() == "YES"].copy()
    yes["transport_prob"] = yes["transport_prob"].astype(float)
    top = yes.nlargest(10, "transport_prob")
    cols = ["query", "seq_length", "transport_prob", "tc_family", "tc_number",
            "top1_similarity", "top1_sim_gap", "top_family_votes"]
    print(top[cols].to_string(index=False))

    print()
    print("=" * 78)
    print("K. marxianus 3B v3 — 2.A.1.12.x (Jen1 family) hits")
    print("=" * 78)
    jen1 = yes[yes["tc_number"].astype(str).str.startswith("2.A.1.12")]
    if len(jen1) > 0:
        print(jen1[cols].to_string(index=False))
    else:
        print("  (no 2.A.1.12.x hits — checking 2.A.1.x broadly)")
        a2a1 = yes[yes["tc_number"].astype(str).str.startswith("2.A.1.")]
        if len(a2a1) > 0:
            print(f"  2.A.1.x hits: {len(a2a1)}")
            print(a2a1[cols].head(20).to_string(index=False))
        else:
            print("  (no 2.A.1.x hits either)")

    print()
    print("=" * 78)
    print("E. coli BL21 3B v3 — HMO-component-relevant proteins (5 known)")
    print("=" * 78)
    bl = pd.read_csv(BASE / "outputs" / "3b_12species" / "E_coli_BL21" / "final_report.csv")
    for acc, name, expected_fam in HMO_PROTEINS:
        row = bl[bl["query"].astype(str) == acc]
        if len(row) == 0:
            print(f"  {acc} ({name}, expect {expected_fam}): NOT FOUND in v3 output")
            continue
        r = row.iloc[0]
        print(f"  {acc} ({name}, expect {expected_fam}):")
        print(f"    is_transport: {r['is_transport']}  transport_prob: {r['transport_prob']}")
        print(f"    v3 tc_family:  {r['tc_family']}  v3 tc_number: {r['tc_number']}")
        print(f"    top1_sim:      {r['top1_similarity']:.4f}  votes: {r['top_family_votes']}/10")
        match = "MATCH" if str(r["tc_family"]) == expected_fam[:5] else "MISMATCH"
        print(f"    [{match}] vs expected {expected_fam}")
        print()


if __name__ == "__main__":
    main()
