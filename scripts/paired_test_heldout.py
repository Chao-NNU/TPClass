#!/usr/bin/env python
"""
paired_test_heldout.py
======================
Paired significance testing for the TCDB held-out benchmark (paper Table 1).

Input: long-format predictions CSV with columns
    protein_id, config, y_true, y_pred
where config in {650M_cached, 650M_fresh, 3B_cached, 3B_fresh}
(optionally also a transport_prob column, unused here).

Comparisons, all paired on protein_id:
    3B_cached   vs 650M_cached    <- backbone scale, cached
    3B_fresh    vs 650M_fresh     <- backbone scale, fresh
    650M_fresh  vs 650M_cached    <- evaluation mode, 650M
    3B_fresh    vs 3B_cached      <- evaluation mode, 3B

For each comparison:
  * McNemar's exact test on the paired binary decisions
  * paired bootstrap over proteins (10,000 resamples) -> 95% CI for
    delta accuracy / F1 / MCC
  * Jaccard overlap of the Phase 1 predicted-positive sets

Usage:
  python paired_test_heldout.py --preds preds/all_preds.csv --out paired_test.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

PAIRS = [
    ("3B_cached", "650M_cached", "3B vs 650M (cached)"),
    ("3B_fresh", "650M_fresh", "3B vs 650M (fresh)"),
    ("650M_fresh", "650M_cached", "650M cached -> fresh"),
    ("3B_fresh", "3B_cached", "3B cached -> fresh"),
]

METRICS = {
    "accuracy": accuracy_score,
    "F1": f1_score,
    "MCC": matthews_corrcoef,
}


def mcnemar_exact(y, a, b):
    """Exact McNemar on paired binary decisions.

    n01 = pairs where a=1, b=0 ; n10 = pairs where a=0, b=1.
    Returns (n01, n10, two-sided p).
    """
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(n01, n, 0.5).pvalue)


def paired_bootstrap(y, a, b, fn, n_boot=10000, seed=0):
    """95% percentile CI for fn(a) - fn(b) under resampling of proteins."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    idx = np.arange(n)
    for i in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        diffs[i] = fn(y[s], a[s]) - fn(y[s], b[s])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="paired_test.csv")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    missing = {"protein_id", "config", "y_true", "y_pred"} - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")

    wide = {c: g.set_index("protein_id").sort_index() for c, g in df.groupby("config")}
    for ca, cb, _ in PAIRS:
        for c in (ca, cb):
            if c not in wide:
                raise SystemExit(f"config not present in CSV: {c}")

    common = None
    for g in wide.values():
        common = g.index if common is None else common.intersection(g.index)
    common = common.sort_values()
    print(f"[Align] {len(common)} proteins common to all {len(wide)} configs")
    if len(common) < 4000:
        print("[WARN] fewer than 4,000 aligned proteins - check the FASTA/feature order")

    rows = []
    for ca, cb, label in PAIRS:
        A, B = wide[ca].loc[common], wide[cb].loc[common]
        y = A["y_true"].to_numpy().astype(int)
        if not (y == B["y_true"].to_numpy().astype(int)).all():
            raise SystemExit(f"y_true mismatch between {ca} and {cb}")

        pa = A["y_pred"].to_numpy().astype(int)
        pb = B["y_pred"].to_numpy().astype(int)
        n01, n10, p = mcnemar_exact(y, pa, pb)

        r = {
            "comparison": label, "config_a": ca, "config_b": cb, "n": len(y),
            "mcnemar_n01": n01, "mcnemar_n10": n10, "mcnemar_p": p,
        }
        for mname, fn in METRICS.items():
            va, vb = float(fn(y, pa)), float(fn(y, pb))
            lo, hi = paired_bootstrap(y, pa, pb, fn, args.n_boot, args.seed)
            r[f"{mname}_a"] = va
            r[f"{mname}_b"] = vb
            r[f"d_{mname}"] = va - vb
            r[f"{mname}_ci_lo"] = lo
            r[f"{mname}_ci_hi"] = hi

        sa = set(A.index[pa == 1])
        sb = set(B.index[pb == 1])
        inter = len(sa & sb)
        union = len(sa | sb)
        r["pos_n_a"] = len(sa)
        r["pos_n_b"] = len(sb)
        r["pos_intersect"] = inter
        r["pos_jaccard"] = inter / union if union else float("nan")
        r["pos_identical"] = int(sa == sb)
        rows.append(r)

        print(f"\n=== {label} ===")
        print(f"  McNemar  n01={n01} n10={n10}  p={p:.3e}")
        for mname in METRICS:
            print(f"  {mname:9s} d={r['d_' + mname]:+.4f}  "
                  f"95% CI [{r[mname + '_ci_lo']:+.4f}, {r[mname + '_ci_hi']:+.4f}]")
        print(f"  positive set  |A|={r['pos_n_a']} |B|={r['pos_n_b']} "
              f"inter={inter}  Jaccard={r['pos_jaccard']:.4f}  identical={r['pos_identical']}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n[Save] {args.out}")


if __name__ == "__main__":
    main()
