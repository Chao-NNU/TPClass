#!/usr/bin/env python3
"""
cross_species_summary.py - Aggregate all per-species results and produce paper Figure 5 candidates.

Reads outputs/<species>/v3_vs_blast.csv for each species and computes counts;
for species with UniProt GT also computes calibration curve and head-to-head
improvement. Outputs:

  - cross_species_summary.csv   (per-species summary, paper Table 1 draft)
  - cross_species_summary.md    (paper-ready Markdown table)
  - species_bar.png             (v3_only vs BLAST, sorted by v3_only)
  - improvement_bar.png         (v3+BLAST vs BLAST alone, %)
  - calibration_compare.png     (multi-species calibration overlay)
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


# ── Species -> (taxid, full name, domain) mapping ────────────────────────
SPECIES_INFO = {
    "bsub":   (224308, "Bacillus subtilis 168",        "prokaryote"),
    "MG1655": (511145, "E. coli K-12 MG1655",          "prokaryote"),
    "cgl":    (196627, "C. glutamicum ATCC 13032",     "prokaryote"),
    "lacto":  (272623, "Lactococcus lactis IL1403",    "prokaryote"),
    "s288c":  (559292, "Saccharomyces cerevisiae S288C", "eukaryote"),
    "yli":    (495328, "Yarrowia lipolytica",          "eukaryote"),
    "aniger": (425011, "Aspergillus niger CBS 513.88", "eukaryote"),
    "max":    (4911,   "Kluyveromyces marxianus",      "eukaryote"),
    "fuji":   (5127,   "Fusarium fujikuroi",           "eukaryote"),
    "bcin":   (332648, "Botrytis cinerea B05.10",     "eukaryote"),
}


def normalize_query(q: str) -> str:
    if not isinstance(q, str) or not q:
        return q
    import re as _re
    m = _re.match(r"^(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|", q)
    if m:
        return m.group(1)
    return q.split()[0] if " " in q else q


def load_uniprot(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size < 50:
        return set()
    accs = set()
    with open(path) as f:
        f.readline()  # header
        for line in f:
            p = line.split("\t")
            if p and p[0].strip():
                accs.add(p[0].strip())
    return accs


def compute_species_stats(out_dir: Path, uni_accs: set[str]):
    """Return dict of stats for one species. None if files missing."""
    v3_csv = out_dir / "v3_vs_blast.csv"
    final = out_dir / "final_report.csv"
    if not v3_csv.exists() or not final.exists():
        return None
    v3 = pd.read_csv(final)
    cmp = pd.read_csv(v3_csv)

    total = len(v3)
    cats = cmp["match"].value_counts().to_dict()
    n_yes = cats.get("yes", 0)
    n_v3_only = cats.get("v3_only", 0)
    n_blast_only = cats.get("blast_only", 0)
    n_disagree = cats.get("disagree", 0)
    n_neither = cats.get("neither", 0)

    # UniProt validation (if available)
    real_tp = 0
    real_precision = None
    real_precision_high = None
    real_tp_high = 0
    n_v3_yes_high = 0

    if uni_accs:
        cmp["acc"] = cmp["query"].astype(str).map(normalize_query)
        cmp["in_uniprot"] = cmp["acc"].isin(uni_accs)

        n_v3_yes = cmp[(cmp["match"] == "yes")]["in_uniprot"].sum()
        n_blast_only_real = cmp[(cmp["match"] == "blast_only")]["in_uniprot"].sum()
        n_blast_real = int(n_v3_yes + n_blast_only_real)

        # v3_only real TP (all)
        real_tp = int(cmp[cmp["match"] == "v3_only"]["in_uniprot"].sum())
        real_precision = real_tp / n_v3_only if n_v3_only else 0.0

        # High-confidence v3_only (sim >= 0.90), use detail CSV if available
        det = out_dir / "validation" / "v3_only_detail.csv"
        if det.exists():
            det_df = pd.read_csv(det)
            if "top1_similarity" in det_df.columns and "in_uniprot" in det_df.columns:
                hi = det_df[(det_df["top1_similarity"] >= 0.90)
                            & (det_df["in_uniprot"] == True)]
                real_tp_high = int(len(hi))
                n_v3_yes_high = int((det_df["top1_similarity"] >= 0.90).sum())
                real_precision_high = (real_tp_high / n_v3_yes_high
                                       if n_v3_yes_high > 0 else 0.0)

    # BLAST alone real TP (from UniProt)
    blast_real = 0
    if uni_accs:
        blast_real = int(n_v3_yes + n_blast_only_real)

    improvement = (real_tp_high / blast_real) if (blast_real > 0 and real_tp_high > 0) else None

    return {
        "total_proteins": total,
        "blast_hits": n_yes + n_blast_only,
        "tp_yes": n_yes,
        "v3_only": n_v3_only,
        "blast_only": n_blast_only,
        "disagree": n_disagree,
        "neither": n_neither,
        "v3_yes_total": n_yes + n_v3_only + n_disagree,
        "real_tp_v3_only": real_tp,
        "real_precision_v3_only": real_precision,
        "real_tp_v3_only_high_conf": real_tp_high,
        "n_v3_only_high_conf": n_v3_yes_high,
        "blast_real_tp": blast_real,
        "improvement_pct": (improvement * 100.0 if improvement is not None else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/data2/chao/transport/outputs",
                    help="root with per-species subdirs (default /data2/chao/transport/outputs)")
    ap.add_argument("--data_root", default="/data2/chao/transport/data",
                    help="root with uniprot_<species>_transporters.tsv")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    data_root = Path(args.data_root)

    rows = []
    cal_data = []   # (species, real precision per sim bin)
    for species, (taxid, full_name, domain) in SPECIES_INFO.items():
        sp_dir = out_root / species
        uni_path = data_root / f"uniprot_{species}_transporters.tsv"
        uni_accs = load_uniprot(uni_path)
        stats = compute_species_stats(sp_dir, uni_accs)
        if stats is None:
            print(f"[Skip] {species}: missing v3_vs_blast.csv or final_report.csv")
            continue

        stats["species"] = species
        stats["full_name"] = full_name
        stats["domain"] = domain
        stats["taxid"] = taxid
        stats["uniprot_gt_size"] = len(uni_accs)
        rows.append(stats)

        # Calibration data (if validated)
        det = sp_dir / "validation" / "v3_only_detail.csv"
        if det.exists() and uni_accs:
            df = pd.read_csv(det)
            if "top1_similarity" in df.columns and "in_uniprot" in df.columns:
                bins = [(0.95, 1.01), (0.90, 0.95), (0.80, 0.90),
                        (0.70, 0.80), (0.60, 0.70), (0.50, 0.60)]
                for lo, hi in bins:
                    m = (df["top1_similarity"] >= lo) & (df["top1_similarity"] < hi)
                    n_bin = int(m.sum())
                    n_overlap = int((m & df["in_uniprot"]).sum())
                    p = n_overlap / n_bin if n_bin > 0 else None
                    cal_data.append({
                        "species": species, "lo": lo, "hi": hi,
                        "n_bin": n_bin, "n_overlap": n_overlap, "real_p": p,
                    })

        print(f"[OK] {species}: total={stats['total_proteins']}, "
              f"BLAST={stats['blast_hits']}, v3_only={stats['v3_only']}, "
              f"UniProt_GT={len(uni_accs)}, "
              f"improvement={stats['improvement_pct'] if stats['improvement_pct'] is not None else 'N/A'}")

    if not rows:
        print("No species data found")
        return 1

    df_all = pd.DataFrame(rows)

    # ── Save summary CSV ──
    csv_path = out_root / "cross_species_summary.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n[Save] {csv_path}")

    # ── Save Markdown table ──
    md_path = out_root / "cross_species_summary.md"
    with open(md_path, "w") as f:
        f.write("# Cross-species v3 vs BLAST summary\n\n")
        f.write(f"Generated by cross_species_summary.py, n_species={len(df_all)}\n\n")
        f.write("| Species | Full name | Domain | Total | BLAST hits | v3_only | TP (UniProt) | v3_only real P | Improvement |\n")
        f.write("|---------|-----------|--------|-------|------------|---------|--------------|---------------|-------------|\n")
        for _, r in df_all.iterrows():
            imp = (f"{r['improvement_pct']:.1f}%" if pd.notna(r.get('improvement_pct'))
                   else "N/A")
            rp = (f"{r['real_precision_v3_only']*100:.1f}%"
                  if pd.notna(r.get('real_precision_v3_only')) else "N/A")
            f.write(f"| {r['species']} | {r['full_name']} | {r['domain']} | "
                    f"{r['total_proteins']} | {r['blast_hits']} | {r['v3_only']} | "
                    f"{r['blast_real_tp']} | {rp} | {imp} |\n")
    print(f"[Save] {md_path}")

    # ── Plot 1: v3_only vs BLAST bar ──
    fig, ax = plt.subplots(figsize=(11, 5.5))
    df_all_sorted = df_all.sort_values("v3_only", ascending=False)
    x = np.arange(len(df_all_sorted))
    width = 0.4
    colors = ["#ff7f0e" if d == "eukaryote" else "#1f77b4"
              for d in df_all_sorted["domain"]]
    bars1 = ax.bar(x - width/2, df_all_sorted["v3_only"], width,
                   label="v3_only (BLAST-missed)", color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, df_all_sorted["blast_hits"], width,
                   label="BLAST hits (top-1)", color="#1b9e77", alpha=0.85,
                   edgecolor="black", linewidth=0.5)
    for i, (_, r) in enumerate(df_all_sorted.iterrows()):
        ax.annotate(str(r["v3_only"]), (i - width/2, r["v3_only"]),
                    textcoords="offset points", xytext=(0, 4),
                    fontsize=8, ha="center", fontweight="bold")
        ax.annotate(str(r["blast_hits"]), (i + width/2, r["blast_hits"]),
                    textcoords="offset points", xytext=(0, 4),
                    fontsize=8, ha="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_all_sorted["species"], rotation=30, ha="right")
    ax.set_ylabel("# proteins")
    ax.set_title("v3_only (novel) vs BLAST hits per species")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    png1 = out_root / "species_bar.png"
    fig.savefig(png1, dpi=150)
    print(f"[Save] {png1}")

    # ── Plot 2: improvement bar (only species with UniProt GT) ──
    df_val = df_all.dropna(subset=["improvement_pct"]).copy()
    if len(df_val) > 0:
        df_val = df_val.sort_values("improvement_pct", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#ff7f0e" if d == "eukaryote" else "#1f77b4"
                  for d in df_val["domain"]]
        bars = ax.bar(df_val["species"], df_val["improvement_pct"],
                      color=colors, edgecolor="black", linewidth=0.5)
        for bar, imp in zip(bars, df_val["improvement_pct"]):
            ax.annotate(f"+{imp:.1f}%", (bar.get_x() + bar.get_width()/2, imp),
                        textcoords="offset points", xytext=(0, 4),
                        fontsize=10, ha="center", fontweight="bold")
        ax.set_ylabel("v3 (sim>=0.90) + BLAST improvement over BLAST alone (%)")
        ax.set_title("Improvement of v3 + BLAST over BLAST alone\n"
                     "UniProt-validated, higher = v3 more valuable")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        png2 = out_root / "improvement_bar.png"
        fig.savefig(png2, dpi=150)
        print(f"[Save] {png2}")

    # ── Plot 3: calibration overlay (Figure 5) ──
    if cal_data:
        cal_df = pd.DataFrame(cal_data).dropna(subset=["real_p"])
        fig, ax = plt.subplots(figsize=(10, 6))
        # One line per species
        cmap = plt.get_cmap("tab10")
        for i, species in enumerate(cal_df["species"].unique()):
            sub = cal_df[cal_df["species"] == species].sort_values("lo", ascending=False)
            xs = [(r["lo"] + r["hi"]) / 2 for _, r in sub.iterrows()]
            ys = list(sub["real_p"])
            sizes = [max(40, r["n_bin"] * 1.2) for _, r in sub.iterrows()]
            ax.plot(xs, ys, "o-", color=cmap(i), lw=1.5, ms=8,
                    label=f"{species}", alpha=0.85)
            for x, y, r in zip(xs, ys, sub.itertuples()):
                ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                            xytext=(0, 5), fontsize=7, ha="center",
                            color=cmap(i), alpha=0.7)
        ax.axhline(0.5, ls="--", color="gray", alpha=0.5, label="50% precision")
        ax.axhline(0.8, ls=":", color="green", alpha=0.5, label="80% precision")
        ax.set_xlabel("ESM2 similarity (bin mid-point)")
        ax.set_ylabel("Real precision (UniProt-validated)")
        ax.set_title(f"Calibration curve: similarity -> real precision\n"
                     f"({len(cal_df['species'].unique())} species, n_bins shown)")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout()
        png3 = out_root / "calibration_compare.png"
        fig.savefig(png3, dpi=150)
        print(f"[Save] {png3}")

    print(f"\n[DONE] All outputs in {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())