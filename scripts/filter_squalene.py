#!/usr/bin/env python3
"""
filter_squalene.py
从 TClass 输出的 final_report.csv 筛选 squalene / 类萜 / hopene 相关转运蛋白候选。
基于 filter_hmo_candidates.py 同样的 4 级 pipeline + r4 hybrid word-boundary 匹配。

用法:
    python filter_squalene.py

输出:
    outputs/uniprot_sprot/squalene_candidates.csv
    outputs/uniprot_sprot/squalene_top_candidates.csv
    outputs/uniprot_sprot/squalene_summary.txt
"""
import re
import sys
from pathlib import Path
import pandas as pd

# === 配置（env var override，默认 Swiss-Prot run） ===
import os as _os
_INPUT_CSV  = _os.environ.get("SQ_INPUT_CSV",
    "/data2/chao/transport/outputs/uniprot_sprot/final_report.csv")
_OUTPUT_DIR = _os.environ.get("SQ_OUTPUT_DIR",
    "/data2/chao/transport/outputs/uniprot_sprot")
_FASTA_PATHS = _os.environ.get("SQ_FASTA_PATHS",
    "/data2/chao/transport/inputs/uniprot_sprot.fasta,"
    "/data2/chao/transport/inputs/uniprot_sprot_chunk_000.fasta,"
    "/data2/chao/transport/inputs/uniprot_sprot_chunk_001.fasta")

INPUT_CSV  = Path(_INPUT_CSV)
OUTPUT_DIR = Path(_OUTPUT_DIR)
CAND_CSV   = OUTPUT_DIR / "squalene_candidates.csv"
TOP_CSV    = OUTPUT_DIR / "squalene_top_candidates.csv"
SUMMARY    = OUTPUT_DIR / "squalene_summary.txt"

# === Auto-detect FASTA path from input CSV (if env var not set) ===
# If SQ_FASTA_PATHS is not explicitly set AND input_csv is in outputs/<DATASET>/,
# look for inputs/<DATASET>.fasta as fallback (saves user from typing)
if "SQ_FASTA_PATHS" not in _os.environ:
    # Extract dataset name from output dir, e.g. "outputs/max" -> "max"
    dataset = OUTPUT_DIR.name
    auto_fasta = Path(f"/data2/chao/transport/inputs/{dataset}.fasta")
    if auto_fasta.exists():
        _FASTA_PATHS = str(auto_fasta)
        print(f"[auto-detect] SQ_FASTA_PATHS not set; using {auto_fasta} based on OUTPUT_DIR name")

# 原始 FASTA（v3 脚本的 query 字段只存 accession，description 在原 FASTA header 里）
FASTA_PATHS = [Path(p) for p in _FASTA_PATHS.split(",") if p.strip()]
_UNIPROT_FULL_RE = re.compile(r"^>(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|(\S+)\s*(.*)$")
# 去除 GN= / PE= / SV= 标签
_TRAILING_TAG_RE = re.compile(r"\s+(GN=\S+|PE=\S+|SV=\S+)\s*")

# === Squalene 相关关键词（hybrid word-boundary: 短词 word-boundary，长词子串） ===
# SPECIFIC: squalene / hopene / 类萜直接相关
SQUALENE_SPECIFIC = [
    "squalene",                # 直系
    "hopene",                  # squalene-hopene cyclase 产物
    "hopanoid",                # squalene 衍生的五环三萜
    "triterpene", "triterpenoid",  # squalene 是三萜前体
    "farnesyl",                # FPP = squalene 直接前体
    "farnesyl-diphosphate", "farnesyl diphosphate",
    "presqualene",             # 中间体
    "polyterpene", "polyprenyl", "polyprenol",  # 多萜类
    "isoprenoid",              # 萜类大分类
    "sesquiterpene",           # FPP 也是倍半萜前体
    "squalestatin",            # squalene synthase 抑制剂
    "botryococcene",           # 某些菌用 squalene 合成
]
# 短词：3 字符以内的基因名缩写
SQUALENE_SHORT = [
    "SQS",   # squalene synthase
    "SQE",   # squalene epoxidase
    "SHC",   # squalene-hopene cyclase
    "ERG1",  # yeast squalene epoxidase
    "ERG7",  # yeast squalene synthase
    "ERG9",  # yeast squalene synthase (sometimes)
    "HpnC", "HpnD", "HpnE", "HpnF",  # hopanoid biosynthesis
]

# GENERIC: 通用脂质/转运描述（仅作打分，不作 hard filter）
SQUALENE_GENERIC = [
    "transporter", "permease", "pump", "carrier",
    "lipid", "terpenoid", "isoprenoid",  # 通用大类
    "ABC", "MFS",  # 常见 transporter family
]


# === Hybrid word-boundary 匹配（与 filter_hmo_candidates.py r4 一致） ===
def kw_match_mask(text_series, keywords):
    """短词 (≤3 chars) word-boundary 防子串假阳性；长词子串匹配保留化合物形式。"""
    mask = pd.Series([False] * len(text_series), index=text_series.index)
    text = text_series.fillna("").astype(str)
    for kw in keywords:
        if len(kw) <= 3:
            pattern = rf"\b{re.escape(kw)}\b"
        else:
            pattern = re.escape(kw)
        mask |= text.str.contains(pattern, case=False, na=False, regex=True)
    return mask


# === 加载 FASTA 反查 acc -> (gene, description) ===
print(f"[filter_squalene.py] VERSION: 2026-09-10-r3 (env var + auto-detect FASTA from OUTPUT_DIR name)")
print(f"Building accession -> (gene, description) map from FASTA ...")
acc_to_gene, acc_to_desc = {}, {}
for fasta in FASTA_PATHS:
    if not fasta.exists():
        continue
    n = 0
    with open(fasta) as f:
        for line in f:
            if not line.startswith(">"): continue
            m = _UNIPROT_FULL_RE.match(line)
            if not m: continue
            acc = m.group(1)
            gene = m.group(2)
            desc = _TRAILING_TAG_RE.sub(" ", m.group(3)).strip()
            acc_to_gene[acc] = gene
            acc_to_desc[acc] = desc
            n += 1
    print(f"  {fasta.name}: {n} entries")
print(f"  total acc -> (gene, desc) map: {len(acc_to_desc)}")


# === 加载 final_report.csv ===
print(f"\nLoading {INPUT_CSV} ...")
if not INPUT_CSV.exists():
    print(f"ERROR: {INPUT_CSV} not found.")
    sys.exit(1)
df = pd.read_csv(INPUT_CSV, dtype=str)
df["transport_prob"]   = pd.to_numeric(df["transport_prob"],   errors="coerce")
df["top1_similarity"]  = pd.to_numeric(df["top1_similarity"],  errors="coerce")
df["top_family_votes"] = pd.to_numeric(df["top_family_votes"], errors="coerce")
print(f"  total rows: {len(df)}")
print(f"  YES rows  : {(df['is_transport'] == 'YES').sum()}")
print(f"  NO rows   : {(df['is_transport'] == 'NO').sum()}")

# === 解析 query 提取 accession ===
def parse_query(q):
    s = str(q).lstrip(">")
    m = re.match(r"^(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|", s)
    if m: return m.group(1)
    m2 = re.match(r"^([A-Z][A-Z0-9]+(?:\-\d+)?)", s)
    if m2: return m2.group(1)
    return s

df["accession"] = df["query"].apply(parse_query)
df["description"] = df["accession"].map(acc_to_desc).fillna("")
df["gene_or_name"] = df["accession"].map(acc_to_gene).fillna("")
print(f"  parsed: {df['accession'].nunique()} unique accessions")
print(f"  descriptions found: {(df['description'] != '').sum()}/{len(df)}")


# === Filter 1: is_transport=YES ===
yes = df[df["is_transport"] == "YES"].copy()
print(f"\n[Filter 1] is_transport=YES: {len(yes)} proteins")


# === Filter 2: top1_similarity >= 0.70 ===
confident = yes[yes["top1_similarity"].fillna(0) >= 0.70].copy()
print(f"[Filter 2] top1_similarity >= 0.70: {len(confident)} proteins")


# === Filter 3: description 含 squalene / 类萜 关键词 ===
desc_only = confident["description"]
mask_specific = kw_match_mask(desc_only, SQUALENE_SPECIFIC)
mask_short    = kw_match_mask(desc_only, SQUALENE_SHORT)
mask_generic  = kw_match_mask(desc_only, SQUALENE_GENERIC)
print(f"[Filter 3a] squalene-specific 命中: {int(mask_specific.sum())} proteins")
print(f"[Filter 3b] squalene-short-gene 命中: {int(mask_short.sum())} proteins")
print(f"[Filter 3c] squalene-generic   命中: {int(mask_generic.sum())} proteins")


# === 打分：specific=3, short=2, generic=1 ===
confident["sq_specific_hits"] = mask_specific.astype(int) * 3
confident["sq_short_hits"]    = mask_short.astype(int) * 2
confident["sq_generic_hits"]  = mask_generic.astype(int)
confident["sq_score"] = (confident["sq_specific_hits"] + confident["sq_short_hits"]
                         + confident["sq_generic_hits"])

# 取 any-match
mask_any = mask_specific | mask_short | mask_generic
sq = confident[mask_any].copy()
print(f"[Filter 3] squalene any 命中: {len(sq)} proteins")
print(f"  其中 specific 命中: {int(sq['sq_specific_hits'].gt(0).sum())} proteins")
print(f"  其中 short 命中: {int(sq['sq_short_hits'].gt(0).sum())} proteins")


# === 排序：sq_score > transport_prob > top1_similarity ===
sq_sorted = sq.sort_values(
    by=["sq_score", "transport_prob", "top1_similarity"],
    ascending=[False, False, False]
).reset_index(drop=True)


# === 写主输出 ===
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cols = [
    "accession", "gene_or_name", "description",
    "transport_prob", "is_transport", "tc_family", "tc_number",
    "top1_similarity", "top_family_votes", "top_family_purity",
    "sq_score", "sq_specific_hits", "sq_short_hits", "sq_generic_hits",
    "query", "seq_length",
]
sq_sorted[cols].to_csv(CAND_CSV, index=False)
print(f"\n[Save] {CAND_CSV}  ({len(sq_sorted)} rows)")


# === Top 30 ===
top30 = sq_sorted.head(30)[
    ["accession", "gene_or_name", "description", "tc_family",
     "tc_number", "transport_prob", "top1_similarity",
     "sq_specific_hits", "sq_short_hits", "sq_generic_hits"]
]
top30.to_csv(TOP_CSV, index=False)
print(f"[Save] {TOP_CSV}  (top 30)")


# === 打印 top 20 到 stdout ===
print("\n" + "="*120)
print("TOP 20 SQUALENE-RELATED CANDIDATES  (by sq_score > prob > sim)")
print("="*120)
for i, row in sq_sorted.head(20).iterrows():
    marks = []
    if row["sq_specific_hits"] > 0: marks.append("**")
    if row["sq_short_hits"] > 0:    marks.append("#")
    mark = "".join(marks) if marks else "  "
    print(f"{i+1:2d}{mark}{row['accession']:12s} | {row['gene_or_name']:18s} | "
          f"TC {row['tc_number']:14s} | prob {row['transport_prob']:.3f} | "
          f"sim {row['top1_similarity']:.3f} | score {int(row['sq_score'])}")
    print(f"     {row['description'][:110]}")
print("    ** = specific  # = short-gene (SQS/SQE/SHC/ERG)")


# === 摘要统计 ===
with open(SUMMARY, "w") as f:
    f.write(f"TClass Swiss-Prot squalene-related candidate filter\n")
    f.write(f"====================================================\n\n")
    f.write(f"Input:           {INPUT_CSV}\n")
    f.write(f"Total proteins:  {len(df)}\n")
    f.write(f"YES proteins:    {(df['is_transport']=='YES').sum()}\n")
    f.write(f"Confident:       {len(confident)}\n")
    f.write(f"Squalene any:    {len(sq_sorted)} "
            f"(specific={int(sq_sorted['sq_specific_hits'].gt(0).sum())}, "
            f"short-gene={int(sq_sorted['sq_short_hits'].gt(0).sum())}, "
            f"generic-only={int((sq_sorted['sq_score']==1).sum())})\n\n")
    f.write(f"=== TC family distribution (top 20) ===\n")
    f.write(sq_sorted.head(20)["tc_family"].value_counts().to_string())
    f.write(f"\n\n=== Top 20 ===\n")
    f.write(top30.head(20).to_string(index=False))
    f.write(f"\n\n=== Per-organism (top 20) ===\n")
    orgs = sq_sorted.head(20)["description"].str.extract(
        r"\b(OS:([^=]+?)(?: OX=|$))", expand=False
    )[1].fillna("").str.strip()
    if (orgs == "").all():
        orgs = sq_sorted.head(20)["description"].str.split().str[0:3].str.join(" ")
    f.write(orgs.value_counts().head(20).to_string())
print(f"\n[Save] {SUMMARY}")
print(f"\nDone. Open {TOP_CSV} to review top 30 manually.")
