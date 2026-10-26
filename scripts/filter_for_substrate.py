#!/usr/bin/env python3
"""
filter_for_substrate.py — 从 v3 候选里筛"可能跟某代谢物相关"的转运蛋白。

针对一个目标 substrate (e.g. squalene)，按以下信号打分/过滤:
  1. TC class/subclass 命中 (e.g. 3.A=ABC, 2.A.6=MFS lipid, 9.B=putative)
  2. TCDB 描述里有 substrate 相关关键词
  3. (可选) UniProt 注释里有该 substrate
  4. v3 confidence 阈值 (similarity / Phase 1 prob)

输出:
  - filtered CSV (后续 docking / 实验验证的候选)
  - summary stats (每个 filter 留多少人)
  - TC class 分布柱状图
  - 优先级排序 (TXT, paper 报告用)

用法:
  python filter_for_substrate.py \\
      --v3_csv /data2/chao/transport/blastout/sweep/per_threshold/thresh_0.30/validation/v3_only_detail.csv \\
      --annot  /data2/chao/transport/data/tcdb_annot.clean.tsv \\
      --substrate squalene
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


# ── Default keyword sets per substrate ─────────────────────────────────────
SUBSTRATE_KEYWORDS: dict[str, list[str]] = {
    "squalene": [
        "squalene", "squalene-hopene", "squalene synthase",
        "ergosterol", "sterol", "isoprenoid", "terpenoid", "terpene",
        "lipid", "fatty acid", "fatty-acid",
    ],
    "lipid": [
        "lipid", "fatty acid", "fatty-acid", "phospholipid", "sphingolipid",
        "sterol", "cholesterol", "ergosterol", "squalene", "ceramide",
        "isoprenoid", "terpenoid",
    ],
    "sterol": [
        "sterol", "cholesterol", "ergosterol", "sitosterol", "stigmasterol",
        "squalene", "steroid", "lanosterol",
    ],
    "sugar": [
        "sugar", "glucose", "fructose", "galactose", "mannose", "xylose",
        "hexose", "pentose", "monosaccharide", "disaccharide",
    ],
    "amino_acid": [
        "amino acid", "amino-acid", "amino acid transport",
        "leucine", "lysine", "arginine", "glutamate", "aspartate",
    ],
    "ion": [
        "ion", "cation", "anion", "proton", "sodium", "potassium",
        "calcium", "chloride", "metal", "iron", "zinc", "copper",
    ],
    "drug": [
        "drug", "antimicrobial", "multidrug", "MDR", "ABC transporter",
    ],
}


def load_annot_dict(path: Path) -> dict[str, str]:
    """tcdb_annot.clean.tsv -> {tc_number: description}"""
    annots: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                annots[parts[0].strip()] = parts[1].strip()
    return annots


def normalize_query(q: str) -> str:
    if not isinstance(q, str) or not q:
        return q
    import re as _re
    m = _re.match(r"^(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|", q)
    if m:
        return m.group(1)
    return q.split()[0] if " " in q else q


def load_uniprot_accessions(path: Path) -> set[str]:
    accs = set()
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.split("\t")
            if parts and parts[0].strip():
                accs.add(parts[0].strip())
    return accs


def tc_matches_prefix(tc: str, prefixes: list[str]) -> bool:
    """1.I.1.1 matches '1', '1.I', '1.I.1' (any of these prefixes)."""
    if not tc or not isinstance(tc, str):
        return False
    for p in prefixes:
        if tc == p or tc.startswith(p + "."):
            return True
    return False


def parse_queries(s: str) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Filter v3 candidates for a target substrate")
    ap.add_argument("--v3_csv", required=True,
                    help="v3 final_report.csv OR v3_only_detail.csv (must have query + tc_number cols)")
    ap.add_argument("--annot", required=True,
                    help="tcdb_annot.clean.tsv (for TC descriptions)")
    ap.add_argument("--substrate", required=True,
                    help="substrate name (must be a key in SUBSTRATE_KEYWORDS or use --keywords)")
    ap.add_argument("--keywords", default=None,
                    help="comma-separated extra keywords (in addition to substrate defaults)")
    ap.add_argument("--tc_classes", default=None,
                    help="comma-separated TC class prefixes to whitelist "
                         "(e.g. '3.A,2.A,9.B,1.I'). Empty = no whitelist.")
    ap.add_argument("--sim_min", type=float, default=0.0,
                    help="min v3 ESM2 top-1 similarity (default 0.0)")
    ap.add_argument("--prob_min", type=float, default=0.0,
                    help="min v3 Phase 1 transport_prob (default 0.0)")
    ap.add_argument("--uniprot_tsv", default=None,
                    help="(optional) UniProt accessions TSV; --substrate_annot_words "
                         "filters further by description. Usually not needed for substrate search.")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <v3_csv_dir>/substrate_<name>/)")
    args = ap.parse_args()

    v3_path = Path(args.v3_csv)
    out_dir = Path(args.out) if args.out else v3_path.parent / f"substrate_{args.substrate}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load v3 candidates ──
    print(f"[Load] {v3_path}")
    df = pd.read_csv(v3_path)
    n_total = len(df)
    print(f"  {n_total} input rows")
    # Detect columns: v3_only_detail.csv has 'v3_tc'; final_report.csv has 'tc_number'
    if "v3_tc" in df.columns:
        df = df.rename(columns={"v3_tc": "tc_number", "v3_tc_desc": "tc_description",
                                "top1_similarity": "similarity"})
    if "tc_number" not in df.columns:
        print("[Error] no tc_number / v3_tc column found", file=sys.stderr)
        return 1
    if "tc_description" not in df.columns:
        df["tc_description"] = ""

    # ── Load TC annotations (for description enrichment) ──
    print(f"[Load] {args.annot}")
    annots = load_annot_dict(Path(args.annot))
    print(f"  {len(annots)} TC descriptions")
    # Fill missing descriptions
    missing_mask = df["tc_description"].isna() | (df["tc_description"].astype(str).str.len() == 0)
    df.loc[missing_mask, "tc_description"] = (
        df.loc[missing_mask, "tc_number"].astype(str).map(annots).fillna("")
    )

    # ── Determine keywords ──
    substrate_key = args.substrate.lower()
    if substrate_key in SUBSTRATE_KEYWORDS:
        keywords = list(SUBSTRATE_KEYWORDS[substrate_key])
    else:
        keywords = [substrate_key]
    if args.keywords:
        keywords.extend(parse_queries(args.keywords))
    keywords = [k.lower() for k in keywords]
    print(f"\n[Keywords] {keywords}")

    # ── TC class whitelist ──
    tc_prefixes = parse_queries(args.tc_classes) if args.tc_classes else []
    if tc_prefixes:
        print(f"[TC whitelist] {tc_prefixes}")

    # ── UniProt accessions (optional) ──
    uniprot_accs: set[str] = set()
    if args.uniprot_tsv:
        uniprot_accs = load_uniprot_accessions(Path(args.uniprot_tsv))
        print(f"[UniProt] {len(uniprot_accs)} accessions")

    # ── Apply filters step by step ──
    print(f"\n{'=' * 60}")
    print(f" Filtering for substrate: {args.substrate}")
    print("=" * 60)
    n = len(df)
    print(f"  Start:                 {n:>5}")

    # 1. TC class whitelist
    if tc_prefixes:
        df = df[df["tc_number"].astype(str).apply(lambda x: tc_matches_prefix(x, tc_prefixes))]
        print(f"  After TC class filter:  {len(df):>5}  (kept prefixes {tc_prefixes})")

    # 2. Description keyword match
    desc_lower = df["tc_description"].fillna("").astype(str).str.lower()
    pattern = "|".join(keywords)  # OR-match
    keyword_mask = desc_lower.str.contains(pattern, regex=True, na=False)
    df_kw = df[keyword_mask].copy()
    print(f"  After keyword match:    {len(df_kw):>5}  (any of: {keywords[:5]}{'...' if len(keywords)>5 else ''})")

    # 3. v3 confidence: similarity
    if "similarity" in df_kw.columns and args.sim_min > 0:
        sim_col = "similarity"
    elif "top1_similarity" in df_kw.columns and args.sim_min > 0:
        sim_col = "top1_similarity"
    else:
        sim_col = None
    if sim_col and args.sim_min > 0:
        df_kw = df_kw[pd.to_numeric(df_kw[sim_col], errors="coerce").fillna(0) >= args.sim_min]
        print(f"  After sim >= {args.sim_min}:     {len(df_kw):>5}")

    # 4. Phase 1 prob
    if "transport_prob" in df_kw.columns and args.prob_min > 0:
        df_kw = df_kw[pd.to_numeric(df_kw["transport_prob"], errors="coerce").fillna(0) >= args.prob_min]
        print(f"  After prob >= {args.prob_min}:    {len(df_kw):>5}")

    # 5. UniProt
    if uniprot_accs and "query" in df_kw.columns:
        df_kw["acc"] = df_kw["query"].astype(str).map(normalize_query)
        before = len(df_kw)
        df_kw = df_kw[df_kw["acc"].isin(uniprot_accs)]
        print(f"  After UniProt match:    {len(df_kw):>5}  (was {before})")

    print()

    if len(df_kw) == 0:
        print("[Result] 0 candidates passed all filters. Loosen thresholds?")
        return 0

    # ── Rank and output ──
    # Rank by similarity (or transport_prob as fallback)
    rank_col = None
    for c in ("similarity", "top1_similarity", "transport_prob"):
        if c in df_kw.columns:
            rank_col = c
            break
    if rank_col:
        df_kw = df_kw.sort_values(rank_col, ascending=False)

    # Output filtered CSV
    out_csv = out_dir / "filtered_candidates.csv"
    df_kw.to_csv(out_csv, index=False)
    print(f"[Save] {out_csv}  ({len(df_kw)} candidates)")

    # Output priority list (top 20)
    top_n = min(20, len(df_kw))
    top = df_kw.head(top_n)
    prio_path = out_dir / "top_priority.txt"
    with open(prio_path, "w") as f:
        f.write(f"# Top {top_n} candidates for substrate: {args.substrate}\n")
        f.write(f"# Filters: TC={tc_prefixes or '(any)'}, sim>={args.sim_min}, prob>={args.prob_min}\n")
        f.write(f"# Keywords: {keywords}\n")
        f.write(f"# Source: {v3_path}\n\n")
        for i, (_, r) in enumerate(top.iterrows(), 1):
            tc = str(r.get("tc_number", ""))
            sim = r.get(rank_col, "")
            sim_str = f"{sim:.3f}" if pd.notna(sim) and isinstance(sim, (int, float, np.floating)) else str(sim)
            desc = str(r.get("tc_description", ""))[:60]
            f.write(f"{i:>2}. {str(r.get('query', '')):<35} TC={tc:<14} {rank_col}={sim_str}  | {desc}\n")
    print(f"[Save] {prio_path}")

    # ── Plot: TC class distribution of filtered set ──
    df_kw["tc_class"] = df_kw["tc_number"].fillna("").astype(str).str.split(".").str[0]
    cls_dist = df_kw["tc_class"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: TC class bar
    ax = axes[0]
    cls_dist.plot(kind="bar", ax=ax, color="#d95f02", edgecolor="black")
    for i, (k, v) in enumerate(cls_dist.items()):
        ax.annotate(str(v), (i, v), textcoords="offset points",
                    xytext=(0, 5), fontsize=10, ha="center", fontweight="bold")
    ax.set_xlabel("TC class")
    ax.set_ylabel("# candidates")
    ax.set_title(f"Filtered for {args.substrate}: TC class distribution\n"
                 f"(n={len(df_kw)})")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Right: top-K priority (TC family bar)
    ax2 = axes[1]
    df_kw["tc_family"] = df_kw["tc_number"].fillna("").astype(str).apply(
        lambda x: ".".join(x.split(".")[:3]) if x else "")
    fam_dist = df_kw["tc_family"].value_counts().head(10).sort_values()
    fam_dist.plot(kind="barh", ax=ax2, color="#1b9e77", edgecolor="black")
    for i, (k, v) in enumerate(fam_dist.items()):
        ax2.annotate(str(v), (v, i), textcoords="offset points",
                     xytext=(5, 0), fontsize=9, va="center")
    ax2.set_xlabel("# candidates")
    ax2.set_title("Top 10 TC families (3-segment)")
    ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    png = out_dir / "filter_summary.png"
    fig.savefig(png, dpi=150)
    print(f"[Save] {png}")

    print(f"\n[Done] {len(df_kw)} candidates for {args.substrate} in {out_dir}")
    print(f"\nTop 5 by {rank_col}:")
    for i, (_, r) in enumerate(df_kw.head(5).iterrows(), 1):
        tc = str(r.get("tc_number", ""))
        sim = r.get(rank_col, "")
        sim_str = f"{sim:.3f}" if pd.notna(sim) and isinstance(sim, (int, float, np.floating)) else str(sim)
        desc = str(r.get("tc_description", ""))[:60]
        print(f"  {i}. {str(r.get('query', '')):<30} TC={tc:<14} {rank_col}={sim_str}  {desc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
