"""Paired significance tests for two TPClass prediction sets.

Why this module exists
----------------------
The released ``scripts/paired_test_heldout.py`` implements McNemar's test on
the *predicted labels* of the two models:

    n01 = pairs where A predicts 1 and B predicts 0
    n10 = pairs where A predicts 0 and B predicts 1

and applies an exact binomial test to them.  That is a test of "do the two
models call a different number of proteins transporters".  It is **not** a
test of "is one model more accurate than the other": a model that is right
wherever the other is wrong produces the same number of label disagreements as
a model that is wrong wherever the other is right, and the test returns
p = 1.  (Concretely: with ``y = [1, 0]``, ``A = [1, 0]`` (all correct) and
``B = [0, 1]`` (all wrong), the released function returns p = 1.0 even though
A is perfect and B is perfectly wrong.)

The accuracy question is answered by the *correctness* version of McNemar's
test.  Among the proteins where the two models disagree about being right,

    b = number where A is correct and B is wrong
    c = number where A is wrong and B is correct

and under the null hypothesis "both models are equally accurate", b follows a
Binomial(b + c, 0.5) distribution.

Both variants are implemented here.  :func:`mcnemar_exact` is the
correctness-based test and should be used for the comparisons in the paper;
:func:`mcnemar_exact_on_predictions` is the released variant, kept so the two
numbers can be compared and the difference explained.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

#: Metrics reported for every comparison.
DEFAULT_METRICS: Dict[str, Callable] = {
    "accuracy": accuracy_score,
    "F1": f1_score,
    "MCC": matthews_corrcoef,
}


def _as_int_array(values: Sequence, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr.astype(int)


def mcnemar_exact(
    y: Sequence[int], a: Sequence[int], b: Sequence[int]
) -> Tuple[int, int, float]:
    """Exact McNemar test on *correctness*.

    Parameters
    ----------
    y : sequence of int
        Ground-truth labels.
    a, b : sequence of int
        Predicted labels of the two models being compared.

    Returns
    -------
    tuple
        ``(n01, n10, p_value)`` where ``n01`` counts the proteins where A is
        correct and B is wrong, ``n10`` counts the proteins where A is wrong
        and B is correct, and ``p_value`` is the two-sided exact binomial
        p-value under the null hypothesis of equal accuracy.
    """
    yv = _as_int_array(y, "y")
    av = _as_int_array(a, "a")
    bv = _as_int_array(b, "b")
    if not (len(yv) == len(av) == len(bv)):
        raise ValueError(
            f"length mismatch: y={len(yv)}, a={len(av)}, b={len(bv)}"
        )

    a_correct = av == yv
    b_correct = bv == yv
    n01 = int((a_correct & ~b_correct).sum())  # A right, B wrong
    n10 = int((~a_correct & b_correct).sum())  # A wrong, B right
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(n01, n, 0.5).pvalue)


def mcnemar_exact_on_predictions(
    a: Sequence[int], b: Sequence[int]
) -> Tuple[int, int, float]:
    """Exact McNemar test on the *predicted labels* (the released variant).

    Kept for comparison only: it answers "do the two models predict positive
    at different rates", not "which model is more accurate".  See the module
    docstring.
    """
    av = _as_int_array(a, "a")
    bv = _as_int_array(b, "b")
    if len(av) != len(bv):
        raise ValueError(f"length mismatch: a={len(av)}, b={len(bv)}")
    n01 = int(((av == 1) & (bv == 0)).sum())
    n10 = int(((av == 0) & (bv == 1)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(n01, n, 0.5).pvalue)


def paired_bootstrap(
    y: Sequence[int],
    a: Sequence[int],
    b: Sequence[int],
    fn: Callable,
    n_boot: int = 10000,
    seed: int = 0,
) -> Tuple[float, float]:
    """95 % percentile bootstrap CI for ``fn(a) - fn(b)``.

    Proteins are resampled with replacement, which keeps the pairing intact
    and therefore the interval comparable to the paired test.
    """
    yv = _as_int_array(y, "y")
    av = _as_int_array(a, "a")
    bv = _as_int_array(b, "b")
    n = len(yv)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        diffs[i] = fn(yv[s], av[s]) - fn(yv[s], bv[s])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def jaccard_of_positives(a: Sequence[int], b: Sequence[int]) -> float:
    """Jaccard overlap of the two predicted-positive sets."""
    sa = {i for i, v in enumerate(a) if int(v) == 1}
    sb = {i for i, v in enumerate(b) if int(v) == 1}
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else float("nan")


def compare_models(
    y: Sequence[int],
    a: Sequence[int],
    b: Sequence[int],
    metrics: Optional[Dict[str, Callable]] = None,
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict:
    """Run the full paired comparison of two prediction sets.

    Returns a flat dictionary with the McNemar counts and p-value, the
    per-metric values of both models, their difference with a bootstrap
    confidence interval, and the overlap of the predicted-positive sets.
    """
    metrics = metrics or DEFAULT_METRICS
    yv = _as_int_array(y, "y")
    av = _as_int_array(a, "a")
    bv = _as_int_array(b, "b")

    n01, n10, p = mcnemar_exact(yv, av, bv)
    pred_n01, pred_n10, pred_p = mcnemar_exact_on_predictions(av, bv)

    row: Dict = {
        "n": len(yv),
        "mcnemar_n01": n01,
        "mcnemar_n10": n10,
        "mcnemar_p": p,
        "mcnemar_label_variant_n01": pred_n01,
        "mcnemar_label_variant_n10": pred_n10,
        "mcnemar_label_variant_p": pred_p,
    }
    for name, fn in metrics.items():
        va = float(fn(yv, av))
        vb = float(fn(yv, bv))
        lo, hi = paired_bootstrap(yv, av, bv, fn, n_boot=n_boot, seed=seed)
        row[f"{name}_a"] = va
        row[f"{name}_b"] = vb
        row[f"d_{name}"] = va - vb
        row[f"{name}_ci_lo"] = lo
        row[f"{name}_ci_hi"] = hi

    sa = {i for i, v in enumerate(av) if int(v) == 1}
    sb = {i for i, v in enumerate(bv) if int(v) == 1}
    inter = len(sa & sb)
    union = len(sa | sb)
    row["pos_n_a"] = len(sa)
    row["pos_n_b"] = len(sb)
    row["pos_intersect"] = inter
    row["pos_jaccard"] = inter / union if union else float("nan")
    row["pos_identical"] = int(sa == sb)
    return row


def compare_config_groups(
    df,
    pairs: Sequence[Tuple[str, str]],
    label_col: str = "config",
    id_col: str = "protein_id",
    y_col: str = "y_true",
    pred_col: str = "y_pred",
    n_boot: int = 10000,
    seed: int = 0,
) -> List[Dict]:
    """Compare several pairs of configurations in a long-format prediction table.

    Parameters
    ----------
    df : pandas.DataFrame
        Long format with one row per (protein, configuration); the columns
        ``id_col``, ``label_col``, ``y_col`` and ``pred_col`` must exist.
    pairs : sequence of (str, str)
        Configuration names to compare, in order.

    Returns
    -------
    list of dict
        One result dictionary per pair, with an added ``comparison`` label.
    """
    required = {id_col, label_col, y_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in the prediction table: {sorted(missing)}")

    wide = {
        name: grp.set_index(id_col).sort_index() for name, grp in df.groupby(label_col)
    }
    for ca, cb, _ in pairs:
        for c in (ca, cb):
            if c not in wide:
                raise ValueError(f"configuration not present in the table: {c}")

    common = None
    for grp in wide.values():
        common = grp.index if common is None else common.intersection(grp.index)
    common = common.sort_values()

    rows: List[Dict] = []
    for ca, cb, label in pairs:
        A, B = wide[ca].loc[common], wide[cb].loc[common]
        y = A[y_col].to_numpy().astype(int)
        if not (y == B[y_col].to_numpy().astype(int)).all():
            raise ValueError(f"y_true mismatch between {ca} and {cb}")

        row = compare_models(
            y,
            A[pred_col].to_numpy().astype(int),
            B[pred_col].to_numpy().astype(int),
            n_boot=n_boot,
            seed=seed,
        )
        row["comparison"] = label
        row["config_a"] = ca
        row["config_b"] = cb
        rows.append(row)
    return rows
