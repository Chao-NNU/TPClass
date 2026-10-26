"""Row-by-row comparison of two TPClass prediction tables.

This is the verification tool for the claim that the ``tpclass`` package
reproduces the released pipeline.  It compares two CSV tables -- typically the
``final_report.csv`` written by ``scripts/predict_transport_family_v3.py``
against the ``candidates.csv`` written by ``tpclass predict`` -- and reports,
for every shared column, how many rows disagree.

Both files are read as strings first so that empty cells stay empty strings
instead of becoming NaN; numeric columns are converted only for the
comparison, which keeps "missing" distinguishable from "zero".

Two typical uses:

* package vs. released script, on the same input FASTA;
* either of those vs. a published output table from the original working copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

#: Columns compared with a numeric tolerance rather than as strings.
NUMERIC_COLUMNS = {
    "transport_prob",
    "top1_similarity",
    "top1_sim_gap",
    "top_family_purity",
    "top1_seq_identity",
    "top1_qcov",
    "top1_aln_score",
    "top_family_votes",
}


@dataclass
class ColumnDiff:
    """Comparison result for one column."""

    name: str
    n_mismatch: int
    n_rows: int
    numeric: bool
    max_abs_diff: Optional[float] = None
    examples: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ReportDiff:
    """Comparison result for a whole pair of tables."""

    a_path: str
    b_path: str
    key: str
    n_a: int
    n_b: int
    n_matched: int
    n_only_a: int
    n_only_b: int
    columns: List[ColumnDiff] = field(default_factory=list)

    @property
    def total_mismatch(self) -> int:
        """Total number of disagreeing cells across all compared columns."""
        return sum(c.n_mismatch for c in self.columns)

    @property
    def identical(self) -> bool:
        """True when both tables cover the same queries and agree everywhere."""
        return (
            self.total_mismatch == 0
            and self.n_only_a == 0
            and self.n_only_b == 0
        )


def _load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def compare_reports(
    a: str | Path,
    b: str | Path,
    key: str = "query",
    tol: float = 1e-4,
    show: int = 5,
) -> ReportDiff:
    """Compare two prediction tables row by row.

    Parameters
    ----------
    a, b
        Paths to the two CSV tables.
    key
        Column used to align the rows (``query`` by default).
    tol
        Absolute tolerance for numeric columns.
    show
        How many mismatching rows to keep as examples per column.

    Returns
    -------
    ReportDiff
        Per-column mismatch counts plus the overall verdict.
    """
    df_a, df_b = _load(a), _load(b)
    if key not in df_a.columns or key not in df_b.columns:
        raise ValueError(
            f"join key {key!r} is not present in both tables; "
            f"A has {list(df_a.columns)}, B has {list(df_b.columns)}"
        )

    merged = df_a.merge(
        df_b, on=key, how="outer", suffixes=("_a", "_b"), indicator=True
    )
    both = merged[merged["_merge"] == "both"]

    diff = ReportDiff(
        a_path=str(a),
        b_path=str(b),
        key=key,
        n_a=len(df_a),
        n_b=len(df_b),
        n_matched=len(both),
        n_only_a=int((merged["_merge"] == "left_only").sum()),
        n_only_b=int((merged["_merge"] == "right_only").sum()),
    )

    shared = [c for c in df_a.columns if c in df_b.columns and c != key]
    for col in shared:
        ca, cb = both[f"{col}_a"], both[f"{col}_b"]
        numeric = col in NUMERIC_COLUMNS
        if numeric:
            na = pd.to_numeric(ca, errors="coerce")
            nb = pd.to_numeric(cb, errors="coerce")
            delta = (na - nb).abs()
            bad = (delta > tol) & ~(na.isna() & nb.isna())
            finite = delta[~delta.isna()]
            max_abs = float(finite.max()) if len(finite) else 0.0
        else:
            sa = ca.astype(str).str.strip()
            sb = cb.astype(str).str.strip()
            bad = (sa != sb) & ~(ca.isna() & cb.isna())
            max_abs = None

        examples = []
        if int(bad.sum()) and show > 0:
            idx = both.index[bad][:show]
            for i in idx:
                examples.append(
                    {
                        key: str(both.at[i, key]),
                        "a": str(ca.at[i]),
                        "b": str(cb.at[i]),
                    }
                )

        diff.columns.append(
            ColumnDiff(
                name=col,
                n_mismatch=int(bad.sum()),
                n_rows=len(both),
                numeric=numeric,
                max_abs_diff=max_abs,
                examples=examples,
            )
        )
    return diff


def format_report(diff: ReportDiff) -> str:
    """Render a comparison result as a human-readable report."""
    lines = [
        f"A = {diff.a_path}  ({diff.n_a} rows)",
        f"B = {diff.b_path}  ({diff.n_b} rows)",
        f"join key = {diff.key!r}   matched = {diff.n_matched}   "
        f"only_in_A = {diff.n_only_a}   only_in_B = {diff.n_only_b}",
    ]
    if diff.n_only_a or diff.n_only_b:
        lines.append("  [WARN] the two tables do not cover the same set of queries")
    lines.append("")
    lines.append("shared columns:")
    for col in diff.columns:
        flag = "OK " if col.n_mismatch == 0 else "DIFF"
        extra = ""
        if col.numeric and col.max_abs_diff is not None:
            extra = f"  max_abs_diff={col.max_abs_diff:g}"
        lines.append(
            f"  [{flag}] {col.name:<24s} mismatch={col.n_mismatch:>5d}/{col.n_rows}{extra}"
        )
        for ex in col.examples:
            lines.append(f"          {ex[diff.key]}: A={ex['a']!r}  B={ex['b']!r}")
    lines.append("")
    lines.append(f"TOTAL mismatching cells: {diff.total_mismatch}")
    lines.append("RESULT: IDENTICAL" if diff.identical else "RESULT: DIFFERENCES FOUND")
    return "\n".join(lines)
