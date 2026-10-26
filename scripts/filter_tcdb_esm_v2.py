#!/usr/bin/env python3
"""
filter_tcdb_esm_v2.py
合并 filter_for_substrate.py (TC class + TCDB description) + filter_tcdb_esm.py (ESM-2 cosine sim)。
3 阶段 pipeline:
  1. is_transport=YES + top1_sim >= 0.7         (v3 confidence)
  2. TC class 白名单 (e.g. 3.A,2.A,9.B)         (TCDB structural signal)
  3. TCDB description 关键词匹                   (TCDB functional signal, curated by Saier lab)
  4. ESM-2 cosine sim >= 阈值 to reference set   (sequence-level signal)

输出:
  <out_dir>/<target>_v2_candidates.csv
  <out_dir>/<target>_v2_top.csv
  <out_dir>/<target>_v2_summary.txt
  <out_dir>/<target>_v2_filter_summary.png   (TC class + TC family 分布柱状图)
  <out_dir>/<target>_v2_top_priority.txt     (top-N paper-ready list)

用法:
  # 预设 target
  python filter_tcdb_esm_v2.py --csv ... --emb ... --ref_npz ... \\
    --annot /data2/chao/transport/data/tcdb_annot.clean.tsv \\
    --target squalene --top_n 50

  # 自定义
  python filter_tcdb_esm_v2.py --csv ... --emb ... --ref_npz ... \\
    --annot /data2/chao/transport/data/tcdb_annot.clean.tsv \\
    --tc_classes "3.A,2.A,9.B" --keywords "squalene,hopanoid" --top_n 50
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# === 预设 target ===
# 沿用 filter_for_substrate.py 的 SUBSTRATE_KEYWORDS 思路，但每个 target 同时给
# (tc_classes 白名单, keywords 描述) — 都可被 CLI 覆盖
TARGET_PRESETS = {
    "hmo": {
        "tc_classes": ["2.A.1", "2.A.2", "2.A.7", "2.A.21", "2.A.3", "2.A.29",
                       "3.A.1", "3.D.1", "4.A.1", "9.B"],
        "keywords": ["fucose", "sialic", "lactose", "galactose", "glucose",
                     "GlcNAc", "oligosaccharide", "HMO", "sugar"],
    },
    "squalene": {
        # 2026-09-14: 9.B.* 子家族进一步纯化
        # 排除 9.B.115 (sterol reductase, 酶)、9.B.12 (prenyltransferase, 酶)、
        # 9.B.105 (squalene cyclase, 酶)、9.B.241 (polyprenyltransferase, 酶)、
        # 9.B.446 (lysophospholipid acyltransferase, 酶)
        # 只留 9.B.198 (LAM/YSP2，true lipid-binding auxiliary factor) + 2.A.6 (HpnN RND)
        # 3.A.2 → 3.A.2.1 缩窄：排除 3.A.2.2 (V-ATPase c 亚基，假阳性 flood)
        # 1.A.115 删除：那是 ERG1 角鲨烯环氧化酶分类（不是转运蛋白）
        "tc_classes": [
            # RND family: hopanoid exporter (e.g. HpnN)
            "2.A.6",
            # 9.B 只留 lipid-binding auxiliary factor (LAM5/LAM6/YSP2，GRAM domain)
            "9.B.198",
            # P-type ATPase: 脂质外排（可能）
            "3.D.1", "3.D.10",
            # H+/oligopeptide symporter (3.A.2.1，排除 V-ATPase 3.A.2.2)
            "3.A.2.1",
            # 其他（保留待评估）
            "1.C.59", "3.E.4",
        ],
        "keywords": ["squalene", "hopene", "hopanoid", "polyprenyl", "polyprenol",
                     "farnesyl", "isoprenoid", "terpenoid", "sterol",
                     "lanosterol", "ergosterol", "lipid"],
    },
}


# === 工具函数 ===
def get_acc_key(df):
    if "accession" in df.columns:
        return "accession"
    elif "query" in df.columns:
        return "query"
    raise KeyError("DataFrame has neither 'accession' nor 'query' column")


def load_tcdb_annot(path: Path) -> dict[str, str]:
    """tcdb_annot.clean.tsv -> {tc_number: description}.
    TSV 格式: tc_number\\tdescription (可能有 comment 行以 # 开头)
    """
    annots = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                annots[parts[0].strip()] = parts[1].strip()
    return annots


def tc_matches_any_prefix(tc: str, prefixes: list[str]) -> bool:
    """1.I.1.1 匹配 '1', '1.I', '1.I.1' 中的任意一个。
    空 prefixes 列表返回 True（不限制）"""
    if not prefixes:
        return True
    if not tc or not isinstance(tc, str):
        return False
    for p in prefixes:
        if tc == p or tc.startswith(p + "."):
            return True
    return False


def keyword_match(text: str, keywords: list[str]) -> bool:
    """Word-boundary 关键词匹 (case-insensitive)。
    keywords 是 substring, ≤3 字符的用 word-boundary 防 GLNT1 类假阳性。
    空 keywords 列表返回 True（不限制）。
    """
    if not keywords:
        return True
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    for kw in keywords:
        kw_esc = re.escape(kw.lower())
        if len(kw) <= 3:
            if re.search(rf"\b{kw_esc}\b", text_lower):
                return True
        else:
            if kw.lower() in text_lower:
                return True
    return False


def cosine_sim_matrix(Q, R):
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    return Qn @ Rn.T


# === 主函数 ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="v3 final_report.csv")
    ap.add_argument("--emb", required=True, help="v3 embeddings.npz")
    ap.add_argument("--ref_npz", required=True, help="reference embeddings npz")
    ap.add_argument("--annot", required=True, help="TCDB annot TSV (e.g. tcdb_annot.clean.tsv)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target", choices=list(TARGET_PRESETS.keys()) + ["custom"],
                    default="custom", help="target preset (or 'custom' to use --tc_classes/--keywords)")
    ap.add_argument("--tc_classes", default=None,
                    help="comma-separated TC class prefixes (e.g. '3.A,2.A,9.B'). Overrides preset.")
    ap.add_argument("--keywords", default=None,
                    help="comma-separated keywords. Overrides preset.")
    ap.add_argument("--tc_threshold", type=float, default=0.7)
    ap.add_argument("--sim_threshold", type=float, default=0.5)
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--annot_max_len", type=int, default=200,
                    help="truncate TCDB description to this many chars in output (default 200)")
    args = ap.parse_args()

    csv_path  = Path(args.csv)
    emb_path  = Path(args.emb)
    ref_path  = Path(args.ref_npz)
    annot_path = Path(args.annot)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Resolve target presets ===
    if args.target != "custom":
        preset = TARGET_PRESETS[args.target]
        tc_classes = preset["tc_classes"]
        keywords = preset["keywords"]
    else:
        tc_classes = [s.strip() for s in (args.tc_classes or "").split(",") if s.strip()]
        keywords   = [s.strip() for s in (args.keywords or "").split(",") if s.strip()]

    print(f"[filter_tcdb_esm_v2.py] VERSION: 2026-09-10-r1 (TC class + TCDB desc + ESM-2 sim)")
    print(f"  target={args.target}")
    print(f"  tc_classes={tc_classes}")
    print(f"  keywords={keywords}")
    print(f"  csv={csv_path}")
    print(f"  emb={emb_path}")
    print(f"  ref={ref_path}")
    print(f"  annot={annot_path}")

    # === Load v3 outputs ===
    print(f"\n[Load] v3 final_report.csv ...")
    df = pd.read_csv(csv_path, dtype=str)
    df["transport_prob"]   = pd.to_numeric(df["transport_prob"],   errors="coerce")
    df["top1_similarity"]  = pd.to_numeric(df["top1_similarity"],  errors="coerce")
    df["top_family_votes"] = pd.to_numeric(df["top_family_votes"], errors="coerce")
    print(f"  total rows: {len(df)}")
    print(f"  YES rows  : {(df['is_transport'] == 'YES').sum()}")
    acc_key = get_acc_key(df)
    print(f"  accession col: '{acc_key}'")

    # === Load embeddings ===
    print(f"\n[Load] v3 embeddings.npz ...")
    emb_data = np.load(emb_path)
    emb_accs = emb_data["accessions"]
    emb_mat  = emb_data["embeddings"]
    print(f"  shape: {emb_mat.shape}")
    df_idx_by_acc = {acc: i for i, acc in enumerate(emb_accs)}

    # === Load TCDB annotations ===
    print(f"\n[Load] TCDB annot ({annot_path}) ...")
    tcdb_annot = load_tcdb_annot(annot_path)
    print(f"  {len(tcdb_annot)} TC descriptions loaded")

    # === Load reference embeddings ===
    print(f"\n[Load] reference embeddings ({ref_path}) ...")
    ref_data = np.load(ref_path, allow_pickle=True)
    ref_accs  = ref_data["accessions"]
    ref_genes = ref_data["genes"]
    ref_subs  = ref_data.get("substrates", np.array(["?"] * len(ref_accs)))
    ref_mat   = ref_data["embeddings"]
    print(f"  {len(ref_accs)} reference proteins:")
    for a, g, s in zip(ref_accs, ref_genes, ref_subs):
        print(f"    {a}  {g:18s}  {s}")

    # === Filter 1: is_transport=YES + top1_sim >= threshold ===
    yes = df[df["is_transport"] == "YES"].copy()
    confident = yes[yes["top1_similarity"].fillna(0) >= args.tc_threshold].copy()
    print(f"\n[Filter 1] is_transport=YES + top1_sim >= {args.tc_threshold}: {len(confident)} proteins")

    # === Filter 2: TC class whitelist ===
    if tc_classes:
        mask_tc = confident["tc_number"].fillna("").astype(str).apply(
            lambda x: tc_matches_any_prefix(x, tc_classes))
        tc_match = confident[mask_tc].copy()
    else:
        tc_match = confident.copy()
    print(f"[Filter 2] TC class 白名单 {tc_classes or '(any)'}: {len(tc_match)} proteins")

    if len(tc_match) == 0:
        print("[WARN] No candidates after TC class filter. Loosen --tc_classes.")
        return 0

    # === Filter 3: TCDB description keyword match ===
    # 用 TCDB annot 给每个蛋白的 tc_number 补充 description
    tc_match["tc_description"] = tc_match["tc_number"].fillna("").astype(str).map(tcdb_annot).fillna("")
    if keywords:
        mask_kw = tc_match["tc_description"].apply(
            lambda x: keyword_match(x, keywords))
        kw_match = tc_match[mask_kw].copy()
    else:
        kw_match = tc_match.copy()
    print(f"[Filter 3] TCDB description 含 keywords {keywords or '(any)'}: {len(kw_match)} proteins")

    if len(kw_match) == 0:
        print("[WARN] No candidates after keyword filter. Loosen --keywords or --tc_classes.")
        return 0

    # === Filter 4: ESM-2 cosine similarity to references ===
    q_idx = [df_idx_by_acc.get(acc, -1) for acc in kw_match[acc_key]]
    valid_mask = np.array([i >= 0 for i in q_idx])
    if not valid_mask.all():
        n_missing = (~valid_mask).sum()
        print(f"  WARNING: {n_missing} candidates missing from embeddings.npz, dropping")
        kw_match = kw_match[valid_mask].reset_index(drop=True)
        q_idx = [i for i in q_idx if i >= 0]
    Q = emb_mat[q_idx]
    sim = cosine_sim_matrix(Q, ref_mat)
    max_sim = sim.max(axis=1)
    argmax_ref = sim.argmax(axis=1)
    kw_match = kw_match.copy()
    kw_match["max_ref_sim"] = max_sim
    kw_match["best_ref_acc"] = [ref_accs[i] for i in argmax_ref]
    kw_match["best_ref_gene"] = [ref_genes[i] for i in argmax_ref]
    sim_filtered = kw_match[kw_match["max_ref_sim"] >= args.sim_threshold].copy()
    print(f"[Filter 4] ESM-2 cosine sim >= {args.sim_threshold}: {len(sim_filtered)} proteins")

    # === Sort by max_ref_sim > transport_prob > top1_similarity ===
    sim_filtered = sim_filtered.sort_values(
        by=["max_ref_sim", "transport_prob", "top1_similarity"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # === Truncate TCDB description for output readability ===
    if "tc_description" in sim_filtered.columns:
        sim_filtered["tc_description_short"] = sim_filtered["tc_description"].astype(str).str[:args.annot_max_len]

    # === Write outputs ===
    cand_csv = out_dir / f"{args.target}_v2_candidates.csv"
    top_csv  = out_dir / f"{args.target}_v2_top.csv"
    summary  = out_dir / f"{args.target}_v2_summary.txt"
    plot_png = out_dir / f"{args.target}_v2_filter_summary.png"
    prio_txt = out_dir / f"{args.target}_v2_top_priority.txt"

    cols = [acc_key, "tc_number", "tc_description_short", "tc_description",
            "transport_prob", "is_transport", "top1_similarity", "top_family_votes",
            "max_ref_sim", "best_ref_acc", "best_ref_gene", "seq_length"]
    cols = list(dict.fromkeys(c for c in cols if c in sim_filtered.columns))
    sim_filtered[cols].to_csv(cand_csv, index=False)
    print(f"\n[Save] {cand_csv}  ({len(sim_filtered)} rows)")

    sim_filtered.head(args.top_n)[cols].to_csv(top_csv, index=False)
    print(f"[Save] {top_csv}  (top {args.top_n})")

    # === Priority list (paper-ready) ===
    with open(prio_txt, "w") as f:
        f.write(f"# Top {min(args.top_n, len(sim_filtered))} candidates for target: {args.target}\n")
        f.write(f"# Filters: TC={tc_classes or '(any)'}, keywords={keywords or '(any)'}, "
                f"sim>={args.sim_threshold}\n")
        f.write(f"# Source: {csv_path}\n\n")
        for i, (_, r) in enumerate(sim_filtered.head(args.top_n).iterrows(), 1):
            acc = str(r.get(acc_key, ""))
            tc = str(r.get("tc_number", ""))
            sim_v = r.get("max_ref_sim", 0.0)
            ref_g = r.get("best_ref_gene", "")
            desc = str(r.get("tc_description_short", ""))
            prob = r.get("transport_prob", 0.0)
            sim1 = r.get("top1_similarity", 0.0)
            f.write(f"{i:>2}. {acc:<20s} TC={tc:<14s} sim1={sim1:.3f} prob={prob:.3f} "
                    f"ref={ref_g:18s}({sim_v:.3f}) | {desc}\n")
    print(f"[Save] {prio_txt}")

    # === Plot: TC class + TC family distributions ===
    if len(sim_filtered) > 0:
        sim_filtered["tc_class"] = sim_filtered["tc_number"].fillna("").astype(str).str.split(".").str[0]
        cls_dist = sim_filtered["tc_class"].value_counts().sort_index()
        sim_filtered["tc_family"] = sim_filtered["tc_number"].fillna("").astype(str).apply(
            lambda x: ".".join(x.split(".")[:3]) if x else "")
        fam_dist = sim_filtered["tc_family"].value_counts().head(10).sort_values()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax = axes[0]
        cls_dist.plot(kind="bar", ax=ax, color="#d95f02", edgecolor="black")
        for i, (k, v) in enumerate(cls_dist.items()):
            ax.annotate(str(v), (i, v), textcoords="offset points",
                        xytext=(0, 5), fontsize=10, ha="center", fontweight="bold")
        ax.set_xlabel("TC class")
        ax.set_ylabel("# candidates")
        ax.set_title(f"{args.target}: TC class distribution (n={len(sim_filtered)})")
        ax.grid(alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=0)

        ax2 = axes[1]
        fam_dist.plot(kind="barh", ax=ax2, color="#1b9e77", edgecolor="black")
        for i, (k, v) in enumerate(fam_dist.items()):
            ax2.annotate(str(v), (v, i), textcoords="offset points",
                         xytext=(5, 0), fontsize=9, va="center")
        ax2.set_xlabel("# candidates")
        ax2.set_title("Top 10 TC families (3-segment)")
        ax2.grid(alpha=0.3, axis="x")

        fig.tight_layout()
        fig.savefig(plot_png, dpi=150)
        plt.close(fig)
        print(f"[Save] {plot_png}")

    # === Print top 20 ===
    print("\n" + "="*120)
    print(f"TOP 20 {args.target.upper()} CANDIDATES  (by max_ref_sim > prob > sim)")
    print("="*120)
    for i, row in sim_filtered.head(20).iterrows():
        acc_v = row.get(acc_key, row.get("query", "?"))
        ref_g = row.get("best_ref_gene", "?")
        ref_a = row.get("best_ref_acc", "?")
        tc_num = row.get("tc_number", "?")
        sim_v = row.get("max_ref_sim", 0.0)
        sim1 = row.get("top1_similarity", 0.0)
        desc = str(row.get("tc_description_short", ""))[:80]
        print(f"{i+1:2d}  {str(acc_v):14s} | TC {str(tc_num):14s} | "
              f"ref_sim {float(sim_v):.3f} -> {str(ref_g):18s} ({str(ref_a)}) | top1 {float(sim1):.3f}")
        print(f"     {desc}")

    # === Summary ===
    with open(summary, "w") as f:
        f.write(f"TClass {args.target} v2 filter (TC class + TCDB desc + ESM-2 sim)\n")
        f.write(f"============================================================\n\n")
        f.write(f"v3 CSV:        {csv_path}\n")
        f.write(f"v3 emb:        {emb_path}\n")
        f.write(f"ref npz:       {ref_path}\n")
        f.write(f"annot:         {annot_path}\n")
        f.write(f"Output:        {out_dir}\n\n")
        f.write(f"=== Pipeline ===\n")
        f.write(f"TC classes:    {tc_classes}\n")
        f.write(f"Keywords:      {keywords}\n")
        f.write(f"Total:           {len(df)}\n")
        f.write(f"YES:             {(df['is_transport']=='YES').sum()}\n")
        f.write(f"Confident:       {len(confident)}\n")
        f.write(f"TC class:        {len(tc_match)}\n")
        f.write(f"Keyword:         {len(kw_match)}\n")
        f.write(f"ESM-2 sim:       {len(sim_filtered)}\n\n")
        if len(sim_filtered) > 0:
            f.write(f"=== TC class distribution ===\n")
            f.write(sim_filtered["tc_class"].value_counts().to_string())
            f.write(f"\n\n=== Best ref match distribution ===\n")
            f.write(sim_filtered["best_ref_gene"].value_counts().to_string())
            f.write(f"\n\n=== Top {min(args.top_n, 20)} ===\n")
            f.write(sim_filtered.head(min(args.top_n, 20))[cols].to_string(index=False))
    print(f"\n[Save] {summary}")
    print(f"\nDone. Open {top_csv} / {prio_txt} / {plot_png}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
