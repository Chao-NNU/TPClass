#!/usr/bin/env python3
"""
validate_v3_only.py - Validate credibility of v3_only (BLAST-missed) predictions.

Reads v3_vs_blast.csv + final_report.csv under the threshold dir, outputs:
  - TC number validity (v3_only's TC in TCDB_annot; 100% means Phase 2 is sane)
  - ESM2 similarity distribution (v3_only vs TP vs neither)
  - Per-bucket v3_only counts by similarity (for estimated_precision)
  - Family / class distribution
  - Phase 1 confidence (transport_prob) distribution
  - 2-subplot figure: similarity histogram + cumulative count
  - v3_only detail CSV (one row per query)

Usage:
  python validate_v3_only.py \\
      --thresh_dir /data2/chao/transport/blastout/sweep/per_threshold/thresh_0.30 \\
      --annot     /data2/chao/transport/data/tcdb_annot.clean.tsv

Output goes to {thresh_dir}/validation/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIM_BUCKETS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def load_annot_set(path: Path) -> set[str]:
    """Load tcdb_annot.clean.tsv -> set of valid TC numbers."""
    valid = set()
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            tc = line.split("\t", 1)[0].strip()
            if tc:
                valid.add(tc)
    return valid


def normalize_query(q: str) -> str:
    """sp|XXX|... -> XXX (in case UniProt returns full headers)."""
    if not isinstance(q, str) or not q:
        return q
    import re as _re
    m = _re.match(r"^(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|", q)
    if m:
        return m.group(1)
    return q.split()[0] if " " in q else q


def load_uniprot_accessions(path: Path) -> set[str]:
    """Load UniProt TSV (column 1 = accession) -> set of accessions."""
    accs = set()
    with open(path) as f:
        header = f.readline()  # skip
        for line in f:
            parts = line.split("\t")
            if parts and parts[0].strip():
                accs.add(parts[0].strip())
    return accs


def main():
    ap = argparse.ArgumentParser(
        description="Validate v3_only (BLAST-missed) predictions")
    ap.add_argument("--thresh_dir", required=True,
                    help="per_threshold/thresh_X.XX/ dir containing final_report.csv + v3_vs_blast.csv")
    ap.add_argument("--annot", required=True,
                    help="tcdb_annot.clean.tsv path (for TC validity check)")
    ap.add_argument("--uniprot_tsv", default=None,
                    help="(optional) UniProt K. marxianus transporter TSV (download via curl, see below). "
                         "When provided, v3_only precision is computed as "
                         "|v3_only ∩ UniProt_transporter| / |v3_only|.")
    ap.add_argument("--sim_cutoff", type=float, default=1.01,
                    help="apply top1_similarity >= cutoff to v3_only before computing "
                         "precision and head-to-head (default 1.01 = no filter). "
                         "Use 0.95 for high-confidence novel discovery.")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <thresh_dir>/validation/)")
    args = ap.parse_args()

    thresh_dir = Path(args.thresh_dir)
    out_dir = Path(args.out) if args.out else thresh_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    v3_csv = thresh_dir / "final_report.csv"
    cmp_csv = thresh_dir / "v3_vs_blast.csv"

    print(f"[Load] {v3_csv}")
    v3 = pd.read_csv(v3_csv)
    print(f"  {len(v3)} v3 rows")

    print(f"[Load] {cmp_csv}")
    cmp = pd.read_csv(cmp_csv)
    print(f"  {len(cmp)} compare rows")

    # Merge to get match category + similarity in one DataFrame
    df = cmp.merge(
        v3[["query", "top1_similarity", "top1_sim_gap",
            "top_family_votes", "top_family_purity",
            "transport_prob", "tc_description"]],
        on="query", how="left",
    )

    print(f"[Load] {args.annot}")
    valid_tcs = load_annot_set(Path(args.annot))
    print(f"  {len(valid_tcs)} valid TC numbers")

    # Optional: UniProt K. marxianus transporter accessions (real ground truth)
    uniprot_accs: set[str] = set()
    if args.uniprot_tsv:
        print(f"[Load] {args.uniprot_tsv}")
        uniprot_accs = load_uniprot_accessions(Path(args.uniprot_tsv))
        print(f"  {len(uniprot_accs)} UniProt transporter accessions")
    print()

    # ── 1. category counts ─────────────────────────────────────────────
    cats = df["match"].value_counts().to_dict()
    print("=" * 60)
    print(" Category counts")
    print("=" * 60)
    for k in ("yes", "v3_only", "disagree", "blast_only", "neither"):
        print(f"  {k:<12}: {cats.get(k, 0):>5}")
    print()

    # ── 2. v3_only subset ──────────────────────────────────────────────
    v3_only = df[df["match"] == "v3_only"].copy()
    n_v3_only = len(v3_only)
    print("=" * 60)
    print(f" v3_only validation (n={n_v3_only})")
    print("=" * 60)

    if n_v3_only == 0:
        print("  (empty - nothing to validate)")
        return 0

    # ── 2.5. Real ground truth: UniProt K. marxianus transporter intersection ──
    real_tp_count: int | None = None
    real_tp_by_sim: list[tuple[float, float, int, int, float]] = []  # (lo, hi, n_overlap, n_bin, p)
    n_tp_real: int = 0
    n_blast_only_real: int = 0
    if uniprot_accs:
        v3_only["acc"] = v3_only["query"].astype(str).map(normalize_query)
        v3_only["in_uniprot"] = v3_only["acc"].isin(uniprot_accs)
        real_tp_count = int(v3_only["in_uniprot"].sum())
        real_precision = real_tp_count / n_v3_only
        print(f"  UniProt overlap (real TP): {real_tp_count}/{n_v3_only} "
              f"= {100*real_precision:.1f}%  <- HARD GROUND TRUTH")
        # Per-bucket real precision
        non_overlap_bins = [
            (0.95, 1.01), (0.90, 0.95), (0.80, 0.90),
            (0.70, 0.80), (0.60, 0.70), (0.50, 0.60), (0.00, 0.50),
        ]
        print(f"  Per-bucket real precision (UniProt-validated):")
        print(f"    {'Sim range':<14} {'# v3_only':>10} {'# in UniProt':>14} {'real P':>8}")
        for lo, hi in non_overlap_bins:
            # explicit comparison (NaN -> False)
            in_bin = ((v3_only["top1_similarity"] >= lo)
                      & (v3_only["top1_similarity"] < hi)).fillna(False)
            n_bin = int(in_bin.sum())
            n_overlap = int((in_bin & v3_only["in_uniprot"]).sum())
            p = n_overlap / n_bin if n_bin > 0 else 0.0
            real_tp_by_sim.append((lo, hi, n_overlap, n_bin, p))
            print(f"    [{lo:.2f}, {hi:.2f})  {n_bin:>10} {n_overlap:>14} {p:>8.3f}")
        print()

        # Sanity: also validate TP (BLAST-confirmed) and blast_only against UniProt
        tp = df[df["match"] == "yes"].copy()
        tp["acc"] = tp["query"].astype(str).map(normalize_query)
        n_tp_real = int(tp["acc"].isin(uniprot_accs).sum())
        bl = df[df["match"] == "blast_only"].copy()
        bl["acc"] = bl["query"].astype(str).map(normalize_query)
        n_blast_only_real = int(bl["acc"].isin(uniprot_accs).sum())
        print(f"  Sanity check:")
        print(f"    TP  (BLAST-confirmed) in UniProt: {n_tp_real}/{len(tp)} "
              f"({100*n_tp_real/max(1,len(tp)):.1f}%)  <- should be ~100%")
        print(f"    blast_only                in UniProt: {n_blast_only_real}/{len(bl)} "
              f"({100*n_blast_only_real/max(1,len(bl)):.1f}%)")
        print()

    # TC validity
    v3_only["tc_valid"] = v3_only["v3_tc"].fillna("").astype(str).isin(valid_tcs)
    n_tc_valid = int(v3_only["tc_valid"].sum())
    print(f"  TC valid in TCDB: {n_tc_valid}/{n_v3_only} "
          f"({100*n_tc_valid/n_v3_only:.1f}%)  <- Phase 2 sanity check")

    # ESM2 similarity distribution
    sims = v3_only["top1_similarity"].dropna()
    print(f"\n  ESM2 similarity (v3_only):")
    print(f"    mean={sims.mean():.3f}  median={sims.median():.3f}  "
          f"std={sims.std():.3f}  min={sims.min():.3f}  max={sims.max():.3f}")
    print(f"    bucket counts:")
    for b in SIM_BUCKETS:
        n = int((sims >= b).sum())
        print(f"      >= {b:.2f}: {n:>5}  ({100*n/n_v3_only:.1f}%)")

    # sim_gap (top1 - top2, larger = more confident)
    gaps = v3_only["top1_sim_gap"].dropna()
    if len(gaps) > 0:
        print(f"\n  sim_gap (top1 - top2, larger = more confident):")
        print(f"    mean={gaps.mean():.3f}  median={gaps.median():.3f}")
        print(f"    >= 0.05: {int((gaps >= 0.05).sum())}/{n_v3_only}")

    # transport_prob (Phase 1 confidence)
    probs = v3_only["transport_prob"].dropna()
    print(f"\n  Phase 1 transport_prob:")
    print(f"    mean={probs.mean():.3f}  median={probs.median():.3f}")
    print(f"    >= 0.5: {int((probs >= 0.5).sum())}/{n_v3_only}  "
          f">= 0.7: {int((probs >= 0.7).sum())}/{n_v3_only}")

    # ── 3. Comparison: TP vs v3_only vs neither (similarity distributions) ──
    print(f"\n{'=' * 60}")
    print(" Similarity distribution comparison")
    print("=" * 60)
    tp = df[df["match"] == "yes"]
    print(f"  TP  (yes,        n={len(tp):>4}): "
          f"mean={(tp['top1_similarity'].mean() if len(tp) else 0):.3f}  "
          f"median={(tp['top1_similarity'].median() if len(tp) else 0):.3f}")
    print(f"  v3_only          (n={n_v3_only:>4}): "
          f"mean={sims.mean():.3f}  median={sims.median():.3f}")
    neither = df[df["match"] == "neither"]
    n_neither = len(neither)
    n_neither_v3_no = int((neither["v3_is_transport"] == "NO").sum())
    print(f"  Neither (both NO, n={n_neither:>4}, v3_NO={n_neither_v3_no}): "
          f"Phase 1 prob "
          f"mean={(neither['transport_prob'].mean() if len(neither) else 0):.3f}")

    # ── 4. Family / class distribution ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print(" v3_only: TC class distribution")
    print("=" * 60)
    v3_only["v3_class"] = (v3_only["v3_tc"].fillna("").astype(str)
                           .str.split(".").str[0])
    cls_dist = v3_only["v3_class"].value_counts().sort_index()
    for k, v in cls_dist.items():
        pct = 100 * v / n_v3_only
        print(f"  Class {k}: {v:>5}  ({pct:.1f}%)")

    # Top 10 TC families
    print(f"\n  Top 10 TC families (4-digit) in v3_only:")
    v3_only["v3_family"] = (v3_only["v3_tc"].fillna("").astype(str)
                            .apply(lambda x: ".".join(x.split(".")[:4]) if x else ""))
    fam_dist = v3_only["v3_family"].value_counts().head(10)
    for tc, v in fam_dist.items():
        desc_raw = v3_only.loc[v3_only["v3_family"] == tc, "tc_description"].iloc[0] \
            if (v3_only["v3_family"] == tc).any() else ""
        desc = "" if pd.isna(desc_raw) else str(desc_raw)
        desc_short = (desc[:55] + "...") if len(desc) > 55 else desc
        print(f"    {tc:<12}: {v:>4}  {desc_short}")

    # ── 5. Plot (2x2) ────────────────────────────────────────────────────
    # ── 4.5 Compute head-to-head numbers BEFORE plotting (so plot can use them) ──
    n_filt: int | None = None
    n_filt_real: int = 0
    p_filt: float = 0.0
    n_blast_real: int = 0
    n_v3_blast_real: int = 0
    improvement: float = 0.0
    if uniprot_accs:
        sim_mask = (v3_only["top1_similarity"] >= args.sim_cutoff).fillna(False)
        v3_only_filt = v3_only[sim_mask]
        n_filt = int(sim_mask.sum())
        n_filt_real = int(v3_only_filt["in_uniprot"].sum())
        p_filt = n_filt_real / max(1, n_filt)
        n_blast_real = n_tp_real + n_blast_only_real
        n_v3_blast_real = n_blast_real + n_filt_real
        improvement = n_filt_real / max(1, n_blast_real)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: similarity histogram v3_only vs TP
    ax = axes[0, 0]
    bins = np.arange(0.3, 1.01, 0.05)
    if len(tp) > 0 and tp["top1_similarity"].notna().any():
        ax.hist(tp["top1_similarity"].dropna(), bins=bins, alpha=0.65,
                label=f"TP (n={len(tp)})", color="#1b9e77", edgecolor="white")
    ax.hist(sims, bins=bins, alpha=0.65,
            label=f"v3_only (n={n_v3_only})", color="#d95f02", edgecolor="white")
    ax.set_xlabel("ESM2 top-1 similarity")
    ax.set_ylabel("Count")
    ax.set_title("Similarity: TP (BLAST-confirmed) vs v3_only (novel)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Top-right: cumulative v3_only count above similarity threshold
    ax2 = axes[0, 1]
    thresholds = np.arange(0.5, 1.01, 0.05)
    counts = [int((sims >= t).sum()) for t in thresholds]
    ax2.plot(thresholds, counts, "o-", color="#d95f02", lw=2, ms=8)
    for t, c in zip(thresholds, counts):
        ax2.annotate(f"{c}", (t, c), textcoords="offset points",
                     xytext=(0, 8), fontsize=9, ha="center")
    ax2.set_xlabel("Similarity cutoff")
    ax2.set_ylabel(f"# v3_only predictions above cutoff")
    ax2.set_title("v3_only cumulative count by similarity cutoff\n"
                  f"(total = {n_v3_only}; stricter cutoff -> higher TP probability)")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0.45, 1.0)
    ax2.set_ylim(0, max(counts) * 1.15)

    # Bottom-left: CALIBRATION CURVE (real precision per similarity bin)
    ax3 = axes[1, 0]
    if uniprot_accs and real_tp_by_sim:
        bin_data = [(lo, hi, n_bin, p)
                    for (lo, hi, _ovl, n_bin, p) in real_tp_by_sim if n_bin > 0]
        if bin_data:
            centers = [(lo + hi) / 2 for lo, hi, _, _ in bin_data]
            precs = [p for _, _, _, p in bin_data]
            ns = [n for _, _, n, _ in bin_data]
            sizes = [max(20, n * 1.5) for n in ns]
            ax3.scatter(centers, precs, s=sizes, c=precs, cmap="RdYlGn",
                        vmin=0, vmax=1, edgecolor="black", linewidth=0.5, alpha=0.85)
            for c, p, n in zip(centers, precs, ns):
                ax3.annotate(f"{p:.0%}\n(n={n})", (c, p), textcoords="offset points",
                             xytext=(0, 10), fontsize=8, ha="center")
            # Reference: no-filter real precision
            ax3.axhline(100 * real_tp_count / n_v3_only, ls="--", color="gray",
                        label=f"no-filter real P = {100*real_tp_count/n_v3_only:.1f}%")
            ax3.set_xlabel("ESM2 similarity (bin mid-point)")
            ax3.set_ylabel("Real precision (UniProt-validated)")
            ax3.set_title("Calibration curve: similarity -> real precision\n"
                          "(bubble size = # v3_only in bin)")
            ax3.set_xlim(0.5, 1.0)
            ax3.set_ylim(0, 1.0)
            ax3.grid(alpha=0.3)
            ax3.legend(loc="lower left", fontsize=9)
    else:
        ax3.text(0.5, 0.5,
                 "Calibration curve requires --uniprot_tsv\n(UniProt ground truth)",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=11)
        ax3.set_axis_off()

    # Bottom-right: v3 (high-conf) + BLAST vs BLAST alone (bar chart)
    ax4 = axes[1, 1]
    if uniprot_accs and n_filt is not None:
        methods = ["BLAST alone", f"v3 (sim>={args.sim_cutoff})\n+ BLAST"]
        real_tps = [n_blast_real, n_v3_blast_real]
        colors = ["#7570b3", "#d95f02"]
        bars = ax4.bar(methods, real_tps, color=colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, real_tps):
            ax4.annotate(f"{v}", (bar.get_x() + bar.get_width() / 2, v),
                         textcoords="offset points", xytext=(0, 5),
                         fontsize=12, ha="center", fontweight="bold")
        ax4.set_ylabel("Real TP (UniProt-validated)")
        ax4.set_title(f"Head-to-head: v3 high-conf novel + BLAST\n"
                      f"(+{n_filt_real} = +{100*improvement:.1f}% over BLAST alone)")
        ax4.grid(alpha=0.3, axis="y")
        ax4.set_ylim(0, max(real_tps) * 1.15)
    else:
        ax4.text(0.5, 0.5,
                 "Head-to-head bar requires --uniprot_tsv",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=11)
        ax4.set_axis_off()

    fig.suptitle(f"v3_only validation @ {thresh_dir.name}", fontsize=14)
    fig.tight_layout()
    png = out_dir / "v3_only_validation.png"
    fig.savefig(png, dpi=150)
    print(f"\n[Save] {png}")

    # ── 6. Per-row CSV (for manual inspection / case studies) ──────────
    detail_cols = ["query", "v3_tc", "v3_tc_desc", "top1_similarity",
                   "top1_sim_gap", "transport_prob", "tc_valid",
                   "v3_family", "v3_class"]
    if uniprot_accs:
        detail_cols.insert(1, "in_uniprot")  # for sorting real-TP first
    detail_csv = out_dir / "v3_only_detail.csv"
    v3_only[detail_cols].sort_values(
        ["in_uniprot", "top1_similarity"] if uniprot_accs else "top1_similarity",
        ascending=[False, False] if uniprot_accs else False,
    ).to_csv(detail_csv, index=False)
    print(f"[Save] {detail_csv}")

    # ── 7. Precision summary ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(" Precision summary")
    print("=" * 60)
    if uniprot_accs and real_tp_count is not None:
        # Real numbers from UniProt
        print(f"  REAL PRECISION (UniProt-validated): {real_tp_count}/{n_v3_only}  "
              f"= {100*real_tp_count/n_v3_only:.1f}%")
        print(f"  Source: GO:0022857 (transmembrane transporter activity) OR KW-0813 (Transporter)")
        print(f"  (Use this number in the paper; below are details)")
        print()

        # sim_cutoff filtered stats (n_filt etc. computed in section 4.5 for plot)
        print(f"  With --sim_cutoff {args.sim_cutoff}:")
        print(f"    v3_only filtered:   {n_filt} predictions")
        print(f"    UniProt-confirmed:  {n_filt_real} = {100*p_filt:.1f}% precision")
        print()

        # Head-to-head: v3 (high-conf) + BLAST vs BLAST alone
        print(f"  Head-to-head: v3 (sim >= {args.sim_cutoff}) + BLAST vs BLAST alone")
        print(f"    BLAST alone real TP (UniProt):  {n_blast_real}  "
              f"({n_tp_real} TP + {n_blast_only_real} blast_only)")
        print(f"    v3 (high-conf) real novel TP:   {n_filt_real}")
        print(f"    v3 + BLAST union real TP:        {n_v3_blast_real}")
        print(f"    Improvement over BLAST alone:    +{n_filt_real} = +{100*improvement:.1f}%")
        print()
    else:
        # Heuristic placeholders
        print(f"  No UniProt ground truth provided -> showing heuristic estimate only.")
        print(f"  (Use --uniprot_tsv to get real precision)")
        print()
        non_overlap_bins = [
            (0.95, 1.01, 0.95), (0.90, 0.95, 0.90), (0.80, 0.90, 0.80),
            (0.70, 0.80, 0.65), (0.60, 0.70, 0.50), (0.50, 0.60, 0.35),
            (0.00, 0.50, 0.15),
        ]
        print(f"  {'Sim range':<14} {'# v3_only':>10} {'est.P':>8} {'est.TP':>8}")
        est_total_tp = 0.0
        for lo, hi, p in non_overlap_bins:
            n = int(((sims >= lo) & (sims < hi)).sum())
            est_tp = n * p
            est_total_tp += est_tp
            print(f"  [{lo:.2f}, {hi:.2f})  {n:>10} {p:>8.2f} {est_tp:>8.1f}")
        print(f"  {'-' * 50}")
        print(f"  estimated_total_TP ~ {est_total_tp:.0f} / {n_v3_only}  "
              f"({100*est_total_tp/n_v3_only:.1f}%)  <- HEURISTIC ONLY")

    return 0


if __name__ == "__main__":
    sys.exit(main())
