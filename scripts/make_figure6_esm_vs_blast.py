#!/usr/bin/env python3
"""
make_figure6_esm_vs_blast.py
Figure 6: ESM-2 cosine similarity vs diamond blastp identity (top 50 HMO candidates)
v3 - clean layout: ASCII digits, jittered E. coli self-hits, bottom table

Usage:
  python make_figure6_esm_vs_blast.py \
      --tsv /data2/chao/transport/outputs/uniprot_sprot/hmo_esm_vs_blast.tsv \
      --out /data2/chao/transport/outputs/uniprot_sprot/figure6_esm_vs_blast.png \
      --dpi 300
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


# 10 wet-lab candidates — assigned numbers 1-10 by ESM-2 sim descending
# (gene names filled in for the table key; rank by ESM-2 sim ensures stable mapping)
WETLAB_BY_RANK = [
    # (accession, gene, organism, tc, note, blast_missed)
    # 5 E. coli K12 literature-validated positive controls
    ("P11551", "fucP",     "E. coli K12",  "2.A.1.7.2",   "literature",  False),
    ("P02920", "lacY",     "E. coli K12",  "2.A.1.5.1",   "literature",  False),
    ("P23889", "kpsM",     "E. coli K12",  "3.A.1.101.1", "literature",  False),
    ("P23200", "mglC",     "E. coli K12",  "3.A.1.2.3",   "literature",  False),
    ("A4TMJ0", "nanX",     "E. coli K12",  "2.A.1.12.1",  "literature",  True),   # BLAST-missed
    # 5 BLAST-missed novel candidates
    ("Q9LU76", "SLC35D4",  "Arabidopsis",  "2.A.7.9.17",  "BLAST-missed", True),
    ("P39816", "ptsG",     "E. coli K12",  "4.A.1.1.3",   "BLAST-missed", True),
    ("P21906", "glf",      "Z. mobilis",   "2.A.1.1.4",   "BLAST-missed", True),
    ("Q9FX21", "-",        "Arabidopsis",  "9.B.142.3.14","BLAST-missed", True),
    ("P52887", "-",        "mouse",        "9.B.231.1.5", "BLAST-missed", True),
]
WETLAB_ACC_SET = {row[0] for row in WETLAB_BY_RANK}
# Numbering: 1..10 (assigned later by ESM-2 sim rank)
WETLAB_INFO = {row[0]: row for row in WETLAB_BY_RANK}


# TC family palette
TC_COLORS = {
    "2.A.1": "#1f77b4",   # MFS - blue
    "2.A.7": "#ff7f0e",   # GPH - orange
    "3.A.1": "#2ca02c",   # ABC - green
    "4.A.1": "#d62728",   # PTS - red
    "9.B":   "#9467bd",   # auxiliary - purple
    "other": "#7f7f7f",   # gray
}


def tc_root(tc):
    if not isinstance(tc, str) or not tc:
        return "other"
    for k in ("2.A.1", "2.A.7", "3.A.1", "4.A.1", "9.B"):
        if tc.startswith(k):
            return k
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t")
    print(f"[load] {len(df)} candidates from {args.tsv}")

    df["blast_id_pct"] = pd.to_numeric(df["blast_identity"], errors="coerce") * 100
    df["esm_sim"] = pd.to_numeric(df["esm_sim"], errors="coerce")
    df["is_no_hit"] = df["blast_id_pct"].isna()
    df["tc_root"] = df["tc_number"].apply(tc_root)
    df["is_wetlab"] = df["accession"].isin(WETLAB_ACC_SET)

    # Assign wetlab number 1-10 by ESM-2 sim descending
    wet = df[df["is_wetlab"]].sort_values("esm_sim", ascending=False).reset_index(drop=True)
    wet["wet_num"] = range(1, len(wet) + 1)
    acc_to_num = dict(zip(wet["accession"], wet["wet_num"]))
    df["wet_num"] = df["accession"].map(acc_to_num)

    n_total = len(df)
    n_no_hit = int(df["is_no_hit"].sum())
    n_hit = n_total - n_no_hit
    print(f"  no-hit: {n_no_hit}  hit: {n_hit}  wet-lab: {len(wet)}")

    # === Figure layout: main plot on top, table on bottom ===
    fig = plt.figure(figsize=(11, 7.5), dpi=args.dpi)
    gs = GridSpec(2, 1, height_ratios=[3.2, 1.0], hspace=0.05, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    # === (a) BLAST-found: scatter by TC family ===
    hit = df[~df["is_no_hit"]]
    for tc, sub in hit.groupby("tc_root"):
        ax.scatter(
            sub["blast_id_pct"], sub["esm_sim"],
            s=70, alpha=0.78, edgecolor="black", linewidth=0.5,
            color=TC_COLORS.get(tc, TC_COLORS["other"]),
            label=f"{tc} (n={len(sub)})", zorder=3,
        )

    # === (b) BLAST-missed: pink diamonds at x=0 ===
    no_hit = df[df["is_no_hit"]]
    ax.scatter(
        no_hit["blast_id_pct"].fillna(0),
        no_hit["esm_sim"],
        s=100, alpha=0.85, edgecolor="black", linewidth=0.5,
        marker="D", color="#e377c2",
        label=f"no BLAST hit (n={len(no_hit)})", zorder=4,
    )

    # === (c) Wet-lab candidate highlights ===
    # The 4 E. coli K12 self-hits (P11551/P02920/P23889/P23200) all have
    # x=100% and y in [0.99999, 1.00000] -- separated by <1e-5 in y, so
    # they COMPLETELY overlap in the main plot. We mark them with a single
    # shared red ring + count, then show a zoomed INSET for individual labels.
    wet_plot = df[df["is_wetlab"]].copy()
    wet_plot["plot_x"] = wet_plot["blast_id_pct"].fillna(0)
    wet_plot["plot_y"] = wet_plot["esm_sim"]
    ecoli_self_accs = ["P11551", "P02920", "P23889", "P23200"]
    ecoli_self_mask = wet_plot["accession"].isin(ecoli_self_accs)

    # 6 well-separated wet-lab candidates: 5 BLAST-missed (x=0) + A4TMJ0 (x=37.1)
    separable = wet_plot[~ecoli_self_mask]
    for _, r in separable.iterrows():
        ax.scatter(
            r["plot_x"], r["plot_y"],
            s=280, facecolor="none", edgecolor="red", linewidth=2.0,
            marker="o", zorder=6,
        )
        ax.text(
            r["plot_x"], r["plot_y"], f"{int(r['wet_num'])}",
            fontsize=11, fontweight="bold", color="red",
            ha="center", va="center", zorder=7,
        )

    # The 4 E. coli K12 self-hits: one shared red ring + count in main plot
    ecoli_self = wet_plot[ecoli_self_mask]
    ax.scatter(
        100, ecoli_self["esm_sim"].mean(),
        s=350, facecolor="none", edgecolor="red", linewidth=2.5,
        marker="o", zorder=6,
    )
    ax.text(
        100, ecoli_self["esm_sim"].mean(), "1-4",
        fontsize=9, fontweight="bold", color="red",
        ha="center", va="center", zorder=7,
    )
    # Connector line from main plot to inset
    ax.annotate(
        "",
        xy=(100, 0.999998), xytext=(0.5, 0.65),  # in axes fraction for xytext
        xycoords="data", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2, alpha=0.7),
    )

    # === Inset: zoom on the (99.5-100.5%, 0.99990-1.00001) region ===
    axins = ax.inset_axes(
        [0.62, 0.60, 0.35, 0.30],  # [left, bottom, w, h] in axes fraction
        xlim=(99.5, 100.5),
        ylim=(0.99990, 1.00001),
    )
    # Plot all data in inset (gray) for context
    for tc, sub in hit.groupby("tc_root"):
        axins.scatter(
            sub["blast_id_pct"], sub["esm_sim"],
            s=30, alpha=0.4, color=TC_COLORS.get(tc, TC_COLORS["other"]),
        )
    axins.scatter(
        no_hit["blast_id_pct"].fillna(0), no_hit["esm_sim"],
        s=30, alpha=0.3, color="#e377c2", marker="D",
    )
    # Plot the 4 E. coli K12 self-hits highlighted
    for _, r in ecoli_self.iterrows():
        axins.scatter(
            r["blast_id_pct"], r["esm_sim"],
            s=200, facecolor="none", edgecolor="red", linewidth=2.0,
            marker="o", zorder=10,
        )
        # Number to the right of the point with leader line
        axins.annotate(
            f"{int(r['wet_num'])}",
            xy=(r["blast_id_pct"], r["esm_sim"]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=11, fontweight="bold", color="red",
            ha="left", va="center", zorder=11,
            arrowprops=dict(arrowstyle="-", color="red", lw=0.6),
        )
    # Style inset
    axins.set_xticks([99.5, 100.0, 100.5])
    axins.set_yticks([0.99992, 0.99996, 1.00000])
    axins.tick_params(labelsize=7, colors="gray")
    axins.grid(True, alpha=0.3, linestyle=":", linewidth=0.4)
    for spine in axins.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(1.2)
    axins.set_title("zoom: 4 E. coli K12 self-hits", fontsize=8, color="darkred", pad=2)
    # Mark the inset region on the main axes
    ax.indicate_inset_zoom(axins, edgecolor="red", alpha=0.5, linewidth=1.0)

    # === Reference lines ===
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    ax.text(99.5, 0.9503, "TClass threshold  (ESM-2 sim = 0.95)",
            fontsize=8, color="gray", ha="right", va="bottom")
    ax.axvline(30, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, zorder=2)
    ax.text(30.3, 0.9785, "BLAST id 30%",
            fontsize=8, color="gray", rotation=90, va="bottom", ha="left")

    # === Discovery zone shading (top-left) ===
    ax.axvspan(0, 30, ymin=(0.95 - 0.978) / (1.001 - 0.978), ymax=1.0,
               facecolor="#e377c2", alpha=0.10, zorder=1)
    ax.text(15, 0.9996,
            "TClass-only discovery zone\n(ESM-2 >= 0.95, BLAST < 30% or no hit)",
            fontsize=8, ha="center", va="top", color="#a04080", style="italic")

    # === Axes ===
    ax.set_xlabel("BLAST identity to nearest HMO reference (%)", fontsize=11)
    ax.set_ylabel("ESM-2 cosine similarity to HMO reference set", fontsize=11)
    ax.set_xlim(-3, 102)
    ax.set_ylim(0.978, 1.001)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xticks([5, 15, 25, 35, 45, 55, 65, 75, 85, 95], minor=True)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_title(
        f"Figure 6.  ESM-2 similarity vs BLAST identity for the top-50 HMO candidates from Swiss-Prot\n"
        f"({n_no_hit} BLAST-missed at x=0, {n_hit} BLAST-detected; "
        f"10 wet-lab candidates marked with red rings -- 6 individually labeled, "
        f"4 E. coli K12 self-hits zoomed in the inset)",
        fontsize=10, pad=8,
    )

    # === Legend (TC family, no-hit) ===
    handles, labels = ax.get_legend_handles_labels()
    no_hit_idx = [i for i, l in enumerate(labels) if l.startswith("no BLAST")]
    tc_idx = [i for i, l in enumerate(labels) if not l.startswith("no BLAST")]
    order = no_hit_idx + tc_idx
    leg = ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower right", fontsize=8, framealpha=0.92, title="TC family",
        title_fontsize=9,
    )

    # === Stats box (top-left, inside axes, above any wet-lab markers) ===
    stats = (
        f"Top-50 stats:\n"
        f"  TClass-found:  {int((df['esm_sim'] >= 0.95).sum())}/{n_total}  (100%)\n"
        f"  BLAST-found:   {n_hit}/{n_total}  ({100*n_hit/n_total:.0f}%)\n"
        f"  BLAST-missed:  {n_no_hit}/{n_total}  ({100*n_no_hit/n_total:.0f}%)\n"
        f"  Wet-lab:       10/50 (5+5)"
    )
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=8,
            family="monospace", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="lightgray",
                      boxstyle="round,pad=0.3", alpha=0.95))

    # === Bottom table: wet-lab candidates (10 rows + 1 group divider) ===
    header = ["#", "Accession", "Gene", "Organism", "TC number", "Tier"]
    ecoli_rows = []
    missed_rows = []
    for _, r in wet.sort_values("wet_num").iterrows():
        acc = r["accession"]
        info = WETLAB_INFO[acc]
        gene, org, tc, note, is_missed = info[1], info[2], info[3], info[4], info[5]
        tier = "BLAST-missed" if is_missed else "literature"
        row = [str(int(r["wet_num"])), acc, gene, org, tc, tier]
        if is_missed:
            missed_rows.append(row)
        else:
            ecoli_rows.append(row)

    # Single header + group divider + 5 E. coli K12 + 5 BLAST-missed = 12 rows
    divider1 = ["E. coli K12 (literature controls)", "", "", "", "", ""]
    divider2 = ["BLAST-missed novel candidates",    "", "", "", "", ""]
    table_rows = (
        [header]
        + [divider1]
        + ecoli_rows
        + [divider2]
        + missed_rows
    )

    tbl = ax_tbl.table(
        cellText=table_rows,
        loc="upper center",
        cellLoc="left",
        colWidths=[0.04, 0.10, 0.08, 0.14, 0.14, 0.16],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.25)

    # Style cells
    n_ecoli = len(ecoli_rows)
    n_missed = len(missed_rows)
    divider1_idx = 1
    divider2_idx = 1 + 1 + n_ecoli  # = 7
    ecoli_range = range(2, 2 + n_ecoli)               # 2..6
    missed_range = range(divider2_idx + 1, divider2_idx + 1 + n_missed)  # 8..12

    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor("lightgray")
        if i == divider1_idx or i == divider2_idx:
            cell.set_facecolor("#f0e0ea")
            cell.set_text_props(weight="bold", color="#7a2050")
        elif i == 0:
            cell.set_facecolor("#d8d8d8")
            cell.set_text_props(weight="bold")
        elif i in ecoli_range:
            cell.set_facecolor("#eef3ff")
        elif i in missed_range:
            cell.set_facecolor("#fff0f5")
        cell.PAD = 0.04

    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight", format=args.fmt)
    print(f"[save] {args.out}  ({n_total} candidates, {n_no_hit} no-hit)")


if __name__ == "__main__":
    main()
