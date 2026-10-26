#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_distributions.py
=====================
读 final_report.csv, 自动生成一组分布图, 用于评估 Phase 1/Phase 2 的分类质量.

v2 改进:
  - 所有横坐标长标签的图改为水平柱状图 (避免重叠)
  - TC 大类只显示编号 (3/2/1...), 完整名称放图例
  - 图宽自适应, 避免拥挤

用法:
    python plot_distributions.py
    python plot_distributions.py -in /path/to/final_report.csv -out ./plots
"""

from __future__ import annotations
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings("ignore")

# ── 中文字体兜底 ─────────────────────────────────────────────────────────────
for cand in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei",
             "Microsoft YaHei", "PingFang SC", "DejaVu Sans"]:
    try:
        rcParams["font.sans-serif"] = [cand] + rcParams["font.sans-serif"]
        break
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False

# ── TCDB 大类名称 ────────────────────────────────────────────────────────────
TC_CLASS_NAME = {
    "1": "Channels/Pores",
    "2": "Electrochemical Potential-driven",
    "3": "Primary Active Transporters",
    "4": "Group Translocators",
    "5": "Transmembrane Electron Carriers",
    "8": "Accessory Factors",
    "9": "Incompletely Characterized",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [save] {path}")


def _tc_class(tc: str) -> str:
    if not isinstance(tc, str) or not tc:
        return ""
    return tc.split(".")[0].strip()


def _tc_subclass(tc: str) -> str:
    if not isinstance(tc, str) or not tc:
        return ""
    parts = tc.split(".")
    return ".".join(parts[:2]).strip() if len(parts) >= 2 else parts[0].strip()


# ── TC 大类分布 — 水平柱状图 + 完整名称作图例 ────────────────────────────────
def plot_tc_class(df: pd.DataFrame, out: Path) -> None:
    counts = df["tc_family"].dropna().map(_tc_class).value_counts().sort_index()
    if counts.empty:
        print("  [skip] TC class distribution: 无数据"); return

    # 反序: 让最大的在最上面 (barh 默认从下往上)
    counts = counts.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.7 * len(counts) + 1.5)))
    labels = [f"Class {c}" for c in counts.index]
    bars = ax.barh(labels, counts.values,
                  color="#4C72B0", edgecolor="black", linewidth=0.6)

    # 在柱右侧标 数值 + 占比
    for bar, c, v in zip(bars, counts.index, counts.values):
        pct = 100 * v / counts.sum()
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {v}  ({pct:.1f}%)",
                va="center", fontsize=9)

    ax.set_xlabel("Count")
    ax.set_title(f"TC Class Distribution (n={counts.sum()})")

    # 完整名称放图例 (用一个不可见的占位 bar)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="white", edgecolor="white",
                             label=f"Class {c}: {TC_CLASS_NAME.get(c, '?')}")
                       for c in counts.index]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=8, framealpha=0.95, title="TC Class 名")

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, counts.max() * 1.18)   # 给右侧文字留点空间
    _save(fig, out / "01_tc_class_distribution.png")


# ── TC 亚类 — 水平柱状图 ────────────────────────────────────────────────────
def plot_tc_subclass(df: pd.DataFrame, out: Path, top: int = 15) -> None:
    subs = df["tc_family"].dropna().map(_tc_subclass).value_counts()
    if subs.empty:
        print("  [skip] TC subclass distribution: 无数据"); return

    main = subs.head(top)
    other = subs.iloc[top:].sum()
    if other > 0:
        main["Other"] = other
    main = main.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(main) + 1.5)))
    bars = ax.barh(main.index, main.values,
                   color="#55A868", edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, main.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {v}", va="center", fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"TC Subclass Distribution (Top {top}, n={subs.sum()})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, main.max() * 1.12)
    _save(fig, out / "02_tc_subclass_distribution.png")


# ── Top N 高频家族 — 水平柱状图 ──────────────────────────────────────────────
def plot_top_families(df: pd.DataFrame, out: Path, top: int = 20) -> None:
    fam = df["tc_family"].dropna().value_counts().head(top).sort_values(ascending=True)
    if fam.empty:
        print("  [skip] top families: 无数据"); return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(fam) + 1.5)))
    bars = ax.barh(fam.index, fam.values,
                   color="#C44E52", edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, fam.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {v}", va="center", fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"Top {top} TC Families")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, fam.max() * 1.12)
    _save(fig, out / "03_top20_tc_families.png")


# ── Top-1 相似度直方图 ───────────────────────────────────────────────────────
def plot_similarity_hist(df: pd.DataFrame, out: Path,
                         cutoffs=(0.5, 0.7, 0.8, 0.9)) -> None:
    sim = df.loc[df["is_transport"] == "YES", "top1_similarity"].dropna()
    if sim.empty:
        print("  [skip] similarity hist: 无数据"); return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(sim, bins=30, color="#8172B2", edgecolor="black", linewidth=0.5)
    for c in cutoffs:
        n = (sim >= c).sum()
        ax.axvline(c, linestyle="--", linewidth=1)
        ax.text(c, ax.get_ylim()[1] * 0.95, f"≥{c}\n{n}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.set_xlabel("Top-1 similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Top-1 Similarity Distribution (n={len(sim)}, mean={sim.mean():.3f})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out / "04_top1_similarity_hist.png")


# ── 相似度累计曲线 ───────────────────────────────────────────────────────────
def plot_similarity_cdf(df: pd.DataFrame, out: Path,
                        cutoffs=(0.5, 0.7, 0.8, 0.9)) -> None:
    sim = np.sort(df.loc[df["is_transport"] == "YES", "top1_similarity"]
                  .dropna().values)
    if len(sim) == 0:
        return
    cdf = np.arange(1, len(sim) + 1) / len(sim)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sim, cdf, color="#4C72B0", linewidth=2)
    for c in cutoffs:
        if sim.min() <= c <= sim.max():
            n = (sim >= c).sum()
            ax.axvline(c, linestyle="--", color="grey", linewidth=1)
            ax.axhline(n / len(sim), linestyle="--", color="grey", linewidth=1)
            ax.scatter([c], [n / len(sim)], s=60, zorder=5, color="red")
            ax.annotate(f"≥{c}: keep {n} ({100*n/len(sim):.1f}%)",
                        xy=(c, n / len(sim)),
                        xytext=(c + 0.05, max(0, n / len(sim) - 0.12)),
                        fontsize=9, arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_xlabel("Top-1 similarity threshold")
    ax.set_ylabel("Fraction kept (CDF)")
    ax.set_title("Similarity CDF — Pick Your Cutoff")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    _save(fig, out / "05_similarity_cdf.png")


# ── Phase 1 概率分布 ────────────────────────────────────────────────────────
def plot_transport_prob(df: pd.DataFrame, out: Path) -> None:
    yes_p = df.loc[df["is_transport"] == "YES", "transport_prob"].dropna()
    no_p  = df.loc[df["is_transport"] == "NO",  "transport_prob"].dropna()
    if yes_p.empty and no_p.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(no_p,  bins=bins, alpha=0.55, label=f"NO  (n={len(no_p)})",
            color="#999999", edgecolor="black", linewidth=0.3)
    ax.hist(yes_p, bins=bins, alpha=0.65, label=f"YES (n={len(yes_p)})",
            color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set_xlabel("transport_prob (Phase 1 ensemble)")
    ax.set_ylabel("Count")
    ax.set_title("Phase 1 Probability — Actual Class")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out / "06_transport_prob_phase1.png")


# ── 序列长度分布 ────────────────────────────────────────────────────────────
def plot_seq_length(df: pd.DataFrame, out: Path) -> None:
    yes_l = df.loc[df["is_transport"] == "YES", "seq_length"].dropna()
    no_l  = df.loc[df["is_transport"] == "NO",  "seq_length"].dropna()
    if yes_l.empty and no_l.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(yes_l.max(), no_l.max())
                       if len(yes_l) + len(no_l) else 1, 50)
    ax.hist(no_l,  bins=bins, alpha=0.55,
            label=f"NO  (n={len(no_l)},  median={no_l.median():.0f})",
            color="#999999", edgecolor="black", linewidth=0.3)
    ax.hist(yes_l, bins=bins, alpha=0.65,
            label=f"YES (n={len(yes_l)}, median={yes_l.median():.0f})",
            color="#55A868", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Sequence length (aa)")
    ax.set_ylabel("Count")
    ax.set_title("Sequence Length — Actual Class")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out / "07_seq_length.png")


# ── 新增: ident / qcov 直方图 (如果有这些列) ─────────────────────────────────
def plot_identity(df: pd.DataFrame, out: Path) -> None:
    if "top1_seq_identity" not in df.columns:
        return
    ident = df.loc[
        (df["is_transport"] == "YES") & df["top1_seq_identity"].notna(),
        "top1_seq_identity"
    ]
    if ident.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(ident, bins=30, color="#937860", edgecolor="black", linewidth=0.5)
    for c in (0.2, 0.3, 0.5):
        n = (ident >= c).sum()
        ax.axvline(c, linestyle="--", linewidth=1)
        ax.text(c, ax.get_ylim()[1] * 0.95, f"≥{c}\n{n}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.set_xlabel("Top-1 sequence identity (Smith-Waterman)")
    ax.set_ylabel("Count")
    ax.set_title(f"Top-1 Sequence Identity (n={len(ident)}, "
                 f"median={ident.median():.2f})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out / "08_top1_identity.png")


def plot_qcov(df: pd.DataFrame, out: Path) -> None:
    if "top1_qcov" not in df.columns:
        return
    qcov = df.loc[
        (df["is_transport"] == "YES") & df["top1_qcov"].notna(),
        "top1_qcov"
    ]
    if qcov.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(qcov, bins=30, color="#DA8BC3", edgecolor="black", linewidth=0.5)
    for c in (0.3, 0.5, 0.8):
        n = (qcov >= c).sum()
        ax.axvline(c, linestyle="--", linewidth=1)
        ax.text(c, ax.get_ylim()[1] * 0.95, f"≥{c}\n{n}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.set_xlabel("Top-1 query coverage (Smith-Waterman)")
    ax.set_ylabel("Count")
    ax.set_title(f"Top-1 Query Coverage (n={len(qcov)}, "
                 f"median={qcov.median():.2f})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out / "09_top1_qcov.png")


# ── 2x2 总览 (dashboard) ────────────────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) TC class — 水平柱状图
    counts = df["tc_family"].dropna().map(_tc_class).value_counts().sort_values()
    if not counts.empty:
        labels = [f"Class {c}" for c in counts.index]
        axes[0, 0].barh(labels, counts.values,
                        color="#4C72B0", edgecolor="black", linewidth=0.4)
        axes[0, 0].set_title("TC Class")
        axes[0, 0].grid(axis="x", linestyle="--", alpha=0.4)
        for i, v in enumerate(counts.values):
            axes[0, 0].text(v, i, f" {v}", va="center", fontsize=8)

    # (2) similarity hist
    sim = df.loc[df["is_transport"] == "YES", "top1_similarity"].dropna()
    if not sim.empty:
        axes[0, 1].hist(sim, bins=25, color="#8172B2",
                        edgecolor="black", linewidth=0.4)
        axes[0, 1].axvline(0.7, color="red", linestyle="--", label="cutoff=0.7")
        axes[0, 1].set_title("Top-1 Similarity")
        axes[0, 1].legend()
        axes[0, 1].grid(axis="y", linestyle="--", alpha=0.4)

    # (3) Phase 1 prob
    for label, color in [("YES", "#4C72B0"), ("NO", "#999999")]:
        p = df.loc[df["is_transport"] == label, "transport_prob"].dropna()
        if not p.empty:
            axes[1, 0].hist(p, bins=30, alpha=0.55, label=label,
                            color=color, edgecolor="black", linewidth=0.3)
    axes[1, 0].axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
    axes[1, 0].set_title("Phase 1 Probability")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)

    # (4) top 10 families — 水平柱状图
    fam = df["tc_family"].dropna().value_counts().head(10).sort_values()
    if not fam.empty:
        axes[1, 1].barh(fam.index, fam.values,
                        color="#C44E52", edgecolor="black", linewidth=0.4)
        axes[1, 1].set_title("Top 10 TC Families")
        axes[1, 1].grid(axis="x", linestyle="--", alpha=0.4)
        for i, v in enumerate(fam.values):
            axes[1, 1].text(v, i, f" {v}", va="center", fontsize=8)

    fig.suptitle(f"Transport Classification Dashboard — n={len(df)}",
                 fontsize=14, fontweight="bold")
    _save(fig, out / "99_dashboard.png")


# ── 文本摘要 ────────────────────────────────────────────────────────────────
def write_summary(df: pd.DataFrame, out: Path) -> None:
    yes = df[df["is_transport"] == "YES"]
    sim = yes["top1_similarity"]
    lines = []
    lines.append("=" * 60)
    lines.append("  Distribution Summary")
    lines.append("=" * 60)
    lines.append(f"  Total rows                : {len(df)}")
    lines.append(f"  Predicted transporters    : {len(yes)}  ({100*len(yes)/len(df):.1f}%)")
    lines.append("")
    lines.append("  -- TC 大类分布 --")
    cls = yes["tc_family"].dropna().map(_tc_class).value_counts().sort_index()
    for k, v in cls.items():
        pct = 100 * v / cls.sum() if cls.sum() else 0
        lines.append(f"    Class {k:<3} ({TC_CLASS_NAME.get(k, '?'):<32})  "
                     f"{v:>5}  ({pct:.1f}%)")
    lines.append("")
    lines.append("  -- Top-1 similarity --")
    if not sim.empty:
        lines.append(f"    mean={sim.mean():.4f}   median={sim.median():.4f}   "
                     f"std={sim.std():.4f}")
        for c in (0.5, 0.7, 0.8, 0.9, 0.95):
            lines.append(f"    ≥{c:.2f}  ->  {(sim >= c).sum():>5}  "
                         f"({(sim >= c).mean()*100:5.1f}%)")

    if "top1_seq_identity" in df.columns:
        ident = yes["top1_seq_identity"].dropna()
        if not ident.empty:
            lines.append("")
            lines.append("  -- Top-1 sequence identity (SW) --")
            lines.append(f"    mean={ident.mean():.4f}   median={ident.median():.4f}")
            for c in (0.2, 0.3, 0.5, 0.7):
                lines.append(f"    ≥{c:.2f}  ->  {(ident >= c).sum():>5}  "
                             f"({(ident >= c).mean()*100:5.1f}%)")

    if "top1_qcov" in df.columns:
        qcov = yes["top1_qcov"].dropna()
        if not qcov.empty:
            lines.append("")
            lines.append("  -- Top-1 query coverage (SW) --")
            lines.append(f"    mean={qcov.mean():.4f}   median={qcov.median():.4f}")
            for c in (0.3, 0.5, 0.8):
                lines.append(f"    ≥{c:.2f}  ->  {(qcov >= c).sum():>5}  "
                             f"({(qcov >= c).mean()*100:5.1f}%)")

    lines.append("")
    lines.append("  -- Phase 1 transport_prob (ensemble) --")
    p_yes = yes["transport_prob"]
    p_no  = df.loc[df["is_transport"] == "NO", "transport_prob"]
    lines.append(f"    YES: mean={p_yes.mean():.3f}  median={p_yes.median():.3f}")
    lines.append(f"    NO : mean={p_no.mean():.3f}  median={p_no.median():.3f}")
    lines.append("")
    lines.append("=" * 60)

    txt = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"  [save] {out / 'summary.txt'}")


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="为 final_report.csv 自动出分布图")
    ap.add_argument("-in",  "--input",
                    default="/data2/chao/transport/results/final_report.csv")
    ap.add_argument("-out", "--output",
                    default="/data2/chao/transport/results/plots")
    args = ap.parse_args()

    csv_path = Path(args.input)
    out_dir  = Path(args.output)

    if not csv_path.exists():
        raise SystemExit(f"[ERROR] CSV 不存在: {csv_path}")

    print(f"[Load] {csv_path}")
    df = pd.read_csv(csv_path)

    expected = {"query", "seq_length", "transport_prob", "is_transport",
                "tc_family", "tc_number", "tc_description", "top1_similarity"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"[ERROR] CSV 缺少列: {missing}\n实际列: {list(df.columns)}")

    print(f"[Rows] {len(df)}   "
          f"(YES={sum(df['is_transport']=='YES')}, "
          f"NO={sum(df['is_transport']=='NO')})")
    print(f"[Out]  {out_dir}\n")

    print("[Plot] 01 TC class ...")
    plot_tc_class(df, out_dir)
    print("[Plot] 02 TC subclass ...")
    plot_tc_subclass(df, out_dir)
    print("[Plot] 03 top families ...")
    plot_top_families(df, out_dir)
    print("[Plot] 04 similarity hist ...")
    plot_similarity_hist(df, out_dir)
    print("[Plot] 05 similarity CDF ...")
    plot_similarity_cdf(df, out_dir)
    print("[Plot] 06 phase1 prob ...")
    plot_transport_prob(df, out_dir)
    print("[Plot] 07 seq length ...")
    plot_seq_length(df, out_dir)
    print("[Plot] 08 top1 identity ...")
    plot_identity(df, out_dir)
    print("[Plot] 09 top1 qcov ...")
    plot_qcov(df, out_dir)
    print("[Plot] 99 dashboard ...")
    plot_dashboard(df, out_dir)
    print("[Sum ] summary ...")
    write_summary(df, out_dir)

    print(f"\n[Done] 共 {len(list(out_dir.glob('*.png')))} 张图 + 1 份 summary")


if __name__ == "__main__":
    main()