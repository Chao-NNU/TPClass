"""Family-level (4-digit TC) precision, fuzzy recall and strict recall.

The two evaluation scripts of the released pipeline compute a metric called
"fuzzy recall" under **different rules**, so their numbers are not directly
comparable:

``scripts/eval_family_precision.py`` (Tables 2 and 4)
    A protein counts towards fuzzy recall only when it was predicted YES
    **and** received a 4-/5-digit TC assignment **and** is in the ground
    truth.  A predicted transporter that got no family is not counted.

``scripts/eval_hybrid_v3.py`` (Table 3)
    Every predicted-YES protein that is in the ground truth counts, whether or
    not it received a family.

On a minimal example -- two predicted transporters, one of which received a
family -- the first rule gives 0.50 and the second 1.00, while both scripts
print the same metric name.  Both are internally consistent
(``strict = fuzzy x precision`` holds in each), but they answer different
questions.

This module computes every metric from a single code path and makes the rule
explicit, so that the tables can be regenerated under one definition and the
two rules can be compared side by side.  See :func:`evaluate_family`.

Note on the family level
------------------------
The majority vote in Phase 2 aggregates the **3-digit** TC family of the
retrieved references, while the 4-digit sub-subfamily comes from the top-1
hit (``tc_number``).  The evaluation therefore scores the top-1 assignment
truncated to four components -- a top-1 retrieval metric -- and not the
majority vote.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

#: Fuzzy recall counts only predictions that received a TC family.
#: This is the rule of ``scripts/eval_family_precision.py``.
FUZZY_WITH_FAMILY = "with_family"

#: Fuzzy recall counts every predicted-positive protein that is in the ground
#: truth.  This is the rule of ``scripts/eval_hybrid_v3.py``.
FUZZY_ALL_YES = "all_yes"

FUZZY_RULES = (FUZZY_WITH_FAMILY, FUZZY_ALL_YES)

#: A TC identifier with at least four components.
_TC_PATTERN = r"\d+\.[A-Z]\.\d+\.\d+(\.\d+)?"


def to_4digit(tc) -> str:
    """Truncate a 4- or 5-segment TC identifier to its 4-digit family form.

    ``'2.A.1.2.98'`` -> ``'2.A.1.2'``; empty or malformed input -> ``''``.
    """
    if not isinstance(tc, str) or not tc:
        return ""
    s = tc.strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    parts = s.split(".")
    if len(parts) >= 4:
        return ".".join(parts[:4])
    return s


@dataclass
class FamilyMetrics:
    """Result of one family-level evaluation."""

    name: str
    fuzzy_rule: str
    n_pred_total: int
    n_pred_yes: int
    n_pred_yes_with_family: int
    n_gt: int
    n_matched: int
    n_correct: int
    n_yes_in_gt: int
    precision_4digit: float
    precision_with_family: float
    precision_all_yes: float
    coverage_4digit: float
    recall_fuzzy: float
    recall_strict: float

    def as_row(self) -> Dict:
        """Flatten the result into a dictionary for CSV output."""
        row = asdict(self)
        row["recall_fuzzy_matched"] = (
            self.n_matched / self.n_gt if self.n_gt else 0.0
        )
        row["recall_fuzzy_all_yes"] = (
            self.n_yes_in_gt / self.n_gt if self.n_gt else 0.0
        )
        return row


def load_ground_truth(
    gt_path: str | Path,
    acc_col: str = "accession",
    family_col: str = "tc_family_4digit",
) -> Dict[str, str]:
    """Load the ground truth as ``{accession: 4-digit family}``.

    Rows whose family is empty or malformed are dropped, so the denominator
    of every recall is the number of ground-truth proteins that actually
    carry a TC label.
    """
    gt = pd.read_csv(gt_path, sep="\t")
    lookup: Dict[str, str] = {}
    for _, row in gt.iterrows():
        acc = str(row[acc_col])
        fam = to_4digit(str(row[family_col]))
        if acc and fam:
            lookup[acc] = fam
    return lookup


def evaluate_family(
    pred_df: pd.DataFrame,
    gt_lookup: Dict[str, str],
    fuzzy_rule: str = FUZZY_WITH_FAMILY,
    query_col: str = "query",
    family_col: str = "tc_number",
    is_yes_col: str = "is_transport",
    name: str = "",
) -> FamilyMetrics:
    """Evaluate one prediction table against the ground truth.

    Parameters
    ----------
    pred_df : pandas.DataFrame
        The prediction table (``final_report.csv`` or the package output).
    gt_lookup : dict
        ``{accession: 4-digit family}`` from :func:`load_ground_truth`.
    fuzzy_rule : str
        ``'with_family'`` (the rule of ``eval_family_precision.py``) or
        ``'all_yes'`` (the rule of ``eval_hybrid_v3.py``).  Both rules are
        always reported; this argument only selects which one is returned in
        ``recall_fuzzy``.
    query_col, family_col, is_yes_col : str
        Column names in ``pred_df``.
    name : str
        Label for the result.

    Returns
    -------
    FamilyMetrics
    """
    if fuzzy_rule not in FUZZY_RULES:
        raise ValueError(
            f"fuzzy_rule must be one of {FUZZY_RULES}, got {fuzzy_rule!r}"
        )

    is_yes = pred_df[is_yes_col].astype(str).str.upper() == "YES"
    yes = pred_df[is_yes].copy()

    has_family = yes[family_col].astype(str).str.contains(
        _TC_PATTERN, na=False, regex=True
    )
    yes_with_family = yes[has_family].copy()

    n_correct = 0
    n_matched = 0
    n_yes_in_gt = 0
    for _, row in yes.iterrows():
        acc = str(row[query_col])
        pred4 = to_4digit(str(row[family_col]))
        if acc not in gt_lookup:
            continue
        n_yes_in_gt += 1
        if not pred4:
            continue
        n_matched += 1
        if pred4 == gt_lookup[acc]:
            n_correct += 1

    n_gt = len(gt_lookup)
    # The two released scripts also use different precision denominators:
    # eval_family_precision.py divides by the matched-with-family count,
    # eval_hybrid_v3.py divides by the predicted-positive-inside-GT count.
    precision_with_family = n_correct / n_matched if n_matched else 0.0
    precision_all_yes = n_correct / n_yes_in_gt if n_yes_in_gt else 0.0
    coverage = len(yes_with_family) / len(yes) if len(yes) else 0.0
    fuzzy_with_family = n_matched / n_gt if n_gt else 0.0
    fuzzy_all_yes = n_yes_in_gt / n_gt if n_gt else 0.0
    strict = n_correct / n_gt if n_gt else 0.0

    if fuzzy_rule == FUZZY_WITH_FAMILY:
        recall_fuzzy = fuzzy_with_family
        precision = precision_with_family
    else:
        recall_fuzzy = fuzzy_all_yes
        precision = precision_all_yes

    metrics = FamilyMetrics(
        name=name,
        fuzzy_rule=fuzzy_rule,
        n_pred_total=len(pred_df),
        n_pred_yes=len(yes),
        n_pred_yes_with_family=len(yes_with_family),
        n_gt=n_gt,
        n_matched=n_matched,
        n_correct=n_correct,
        n_yes_in_gt=n_yes_in_gt,
        precision_4digit=precision,
        precision_with_family=precision_with_family,
        precision_all_yes=precision_all_yes,
        coverage_4digit=coverage,
        recall_fuzzy=recall_fuzzy,
        recall_strict=strict,
    )
    return metrics


def evaluate_from_files(
    pred_csv: str | Path,
    gt_tsv: str | Path,
    out_csv: Optional[str | Path] = None,
    fuzzy_rule: str = FUZZY_WITH_FAMILY,
    query_col: str = "query",
    family_col: str = "tc_number",
    is_yes_col: str = "is_transport",
    gt_acc_col: str = "accession",
    gt_family_col: str = "tc_family_4digit",
    name: str = "",
) -> FamilyMetrics:
    """Evaluate a prediction CSV against a ground-truth TSV and optionally save it."""
    pred = pd.read_csv(pred_csv)
    gt = load_ground_truth(gt_tsv, acc_col=gt_acc_col, family_col=gt_family_col)
    metrics = evaluate_family(
        pred,
        gt,
        fuzzy_rule=fuzzy_rule,
        query_col=query_col,
        family_col=family_col,
        is_yes_col=is_yes_col,
        name=name or Path(pred_csv).parent.name,
    )
    if out_csv:
        out = Path(out_csv)
        if str(out.parent) not in ("", "."):
            out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics.as_row()]).to_csv(out, index=False, encoding="utf-8-sig")
    return metrics
