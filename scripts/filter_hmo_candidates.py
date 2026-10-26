#!/usr/bin/env python3
"""
filter_hmo_candidates.py
从 TClass 输出的 final_report.csv 筛选 HMO 相关转运蛋白候选。
输出:
  hmo_candidates.csv          全部命中关键词的候选
  hmo_top_candidates.csv      top 50 (transport_prob 排序, 供人工 review)
  hmo_summary.txt             统计摘要
"""
import re
import sys
from pathlib import Path
import pandas as pd

# === 配置（env var override，默认 Swiss-Prot run） ===
import os as _os
_INPUT_CSV  = _os.environ.get("HMO_INPUT_CSV",
    "/data2/chao/transport/outputs/uniprot_sprot/final_report.csv")
_OUTPUT_DIR = _os.environ.get("HMO_OUTPUT_DIR",
    "/data2/chao/transport/outputs/uniprot_sprot")
_FASTA_PATHS = _os.environ.get("HMO_FASTA_PATHS",
    "/data2/chao/transport/inputs/uniprot_sprot.fasta,"
    "/data2/chao/transport/inputs/uniprot_sprot_chunk_000.fasta,"
    "/data2/chao/transport/inputs/uniprot_sprot_chunk_001.fasta")

# === Auto-detect FASTA path from input CSV (if env var not set) ===
if "HMO_FASTA_PATHS" not in _os.environ:
    dataset = OUTPUT_DIR.name
    auto_fasta = Path(f"/data2/chao/transport/inputs/{dataset}.fasta")
    if auto_fasta.exists():
        _FASTA_PATHS = str(auto_fasta)
        print(f"[auto-detect] HMO_FASTA_PATHS not set; using {auto_fasta} based on OUTPUT_DIR name")

INPUT_CSV  = Path(_INPUT_CSV)
OUTPUT_DIR = Path(_OUTPUT_DIR)
HMO_CSV    = OUTPUT_DIR / "hmo_candidates.csv"
TOP_CSV    = OUTPUT_DIR / "hmo_top_candidates.csv"
SUMMARY    = OUTPUT_DIR / "hmo_summary.txt"

# 原始 FASTA（v3 脚本的 query 字段只存 accession，description 在原 FASTA header 里）
# 必须存在，否则 description 字段会全空 → HMO 关键词过滤得 0
FASTA_PATHS = [Path(p) for p in _FASTA_PATHS.split(",") if p.strip()]
_UNIPROT_FULL_RE = re.compile(r"^>(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|(\S+)\s*(.*)$")

# === TC 家族正则：糖转运相关 ===
SUGAR_TC_PATTERNS = [
    r"^2\.A\.1",    # MFS
    r"^2\.A\.2",    # SSS (Solute:Sodium Symporter)
    r"^3\.A\.1",    # ABC
    r"^2\.A\.7",    # GPH (Glycoside-Pentoside-Hexuronide)
    r"^4\.A\.1",    # PTS
    r"^2\.A\.21",   # SSS-like
    r"^2\.A\.3",    # APC (Amino acid-Polyamine-organoCation)
    r"^3\.D",       # PTS-EL
    r"^9\.B",       # Auxiliary factors (sphingolipid-related)
]

# === HMO 相关关键词（分两层） ===
# SPECIFIC: 直接命中 HMO 底物/结构（fucose/sialic/lactose/GlcNAc/HMO 名称等）
HMO_SPECIFIC = [
    "fucose", "fucosyl",
    "sialic", "sialyl", "Neu5Ac", "Neu5Gc",
    "oligosaccharide", "oligosacch",
    "lactose", "lactulose",
    "galactose", "galactos",
    "N-acetylglucosamine", "GlcNAc",
    "N-acetylneuraminic",
    "HMO",
    "2'-fucosyllactose", "2'-FL", "2'FL",
    "3-fucosyllactose", "3-FL", "3FL",
    "LNnT", "LNT",
    "3'-SL", "6'-SL", "3'SL", "6'SL",
    "milk sugar",
]
# GENERIC: 通用糖/转运描述（用作打分加成，不当 hard filter，避免把 triacylglyceride/riboflavin/spermidine 这种拉进来）
HMO_GENERIC = [
    "hexose", "pentose",
    "monosaccharide", "disaccharide",
    "transporter", "permease", "symporter", "antiporter",
    "sugar", "glycoside",
]

# === 加载 ===
print(f"Loading {INPUT_CSV} ...")
if not INPUT_CSV.exists():
    print(f"ERROR: {INPUT_CSV} not found.")
    print("Run TClass first: python scripts/predict_transport_family_v3.py "
          "-in inputs/uniprot_sprot.fasta -out outputs/uniprot_sprot/ --gpu 0 --no_plot")
    sys.exit(1)

# === 从原始 FASTA 反查 accession -> (gene, description) 映射 ===
# v3 脚本的 query 列只存了 accession（P71447），gene / description 在原 FASTA header 里
# 必须把 gene name 和 description 分开存 —— 否则 "LNT" 这种 3-char 关键词会匹配
# 到 gene name "GLNT1"（谷氨酰胺转运蛋白）或 "GN=lnt" 字段造成假阳性
print(f"[filter_hmo_candidates.py] VERSION: 2026-09-10-r6 (env var + auto-detect FASTA from OUTPUT_DIR name)")
print(f"Building accession -> (gene, description) map from FASTA ...")
acc_to_gene, acc_to_desc = {}, {}
# Strip GN=/PE=/SV= fields from description before storing (they're in acc_to_gene already)
# 否则 "LNT" 关键词会匹到 "GN=lnt" 子串
_TRAILING_TAG_RE = re.compile(r"\s+(GN=\S+|PE=\S+|SV=\S+)\s*")
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
            desc = m.group(3).strip()
            # 去掉 GN= / PE= / SV= 标签（基因名已存到 acc_to_gene）
            desc_clean = _TRAILING_TAG_RE.sub(" ", desc).strip()
            acc_to_gene[acc] = gene
            acc_to_desc[acc] = desc_clean
            n += 1
    print(f"  {fasta.name}: {n} entries")
print(f"  total acc -> (gene, desc) map: {len(acc_to_desc)}")

df = pd.read_csv(INPUT_CSV, dtype=str)
df["transport_prob"]   = pd.to_numeric(df["transport_prob"],   errors="coerce")
df["top1_similarity"]  = pd.to_numeric(df["top1_similarity"],  errors="coerce")
df["top_family_votes"] = pd.to_numeric(df["top_family_votes"], errors="coerce")
print(f"  total rows: {len(df)}")
print(f"  YES rows  : {(df['is_transport'] == 'YES').sum()}")
print(f"  NO rows   : {(df['is_transport'] == 'NO').sum()}")

# === 解析 query 提取 accession ===
# query 字段是 accession（如 P71447），用 FASTA map 拿到 description
def parse_query(q):
    s = str(q).lstrip(">")
    # Try UniProt full format first
    m = re.match(r"^(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|", s)
    if m:
        return m.group(1)
    # Fallback: bare accession (e.g. "P71447" or "P71447-2")
    m2 = re.match(r"^([A-Z][A-Z0-9]+(?:\-\d+)?)", s)
    if m2:
        return m2.group(1)
    return s

df["accession"] = df["query"].apply(parse_query)
df["description"] = df["accession"].map(acc_to_desc).fillna("")
df["gene_or_name"] = df["accession"].map(acc_to_gene).fillna("")
print(f"  parsed: {df['accession'].nunique()} unique accessions")
print(f"  descriptions found: {(df['description'] != '').sum()}/{len(df)}")

# === 过滤 1: is_transport=YES ===
yes = df[df["is_transport"] == "YES"].copy()
print(f"\n[Filter 1] is_transport=YES: {len(yes)} proteins")

# === 过滤 2: TC family = 糖转运相关 ===
mask_sugar = pd.Series([False] * len(yes), index=yes.index)
for pat in SUGAR_TC_PATTERNS:
    mask_sugar |= yes["tc_family"].fillna("").str.match(pat, case=False, na=False)
sugar = yes[mask_sugar].copy()
print(f"[Filter 2] sugar/oligosaccharide TC families: {len(sugar)} proteins")

# === 过滤 3: top1_similarity >= 0.70 (高 confidence) ===
confident = sugar[sugar["top1_similarity"].fillna(0) >= 0.70].copy()
print(f"[Filter 3] top1_similarity >= 0.70: {len(confident)} proteins")

# === 过滤 4: HMO 关键词 — 拆成 specific / generic 两层 ===
# 关键词匹配用 word-boundary 正则（避免 "LNT" 匹配 gene name "GLNT1" 之类假阳性）
# 只在 description 字段上匹配，不在 gene name 上匹配（gene name 单独存 acc_to_gene）
def kw_match_mask(text_series, keywords):
    """Hybrid keyword match:
    - 短关键词 (≤3 字符，如 LNT/HMO/FL/SL): word-boundary 正则，避免子串假阳性
      (e.g. "LNT" 不匹 "GLNT1", "HMO" 不匹 "HOMOLOG")
    - 长关键词 (>3 字符，如 sialic/lactose/fucose): 子串匹配，保留化合物形式
      (e.g. "sialic" 仍能匹 "polysialic", "lactose" 匹 "galactooligosaccharide" 中 "lactose" 误报不要紧)
    
    这样既解决了 GLNT1 假阳性，又不会把 "polysialic" 漏掉（之前 word-boundary 把 polysialic 干掉了）。
    """
    mask = pd.Series([False] * len(text_series), index=text_series.index)
    text = text_series.fillna("").astype(str)
    for kw in keywords:
        if len(kw) <= 3:
            # Short keyword: word-boundary required to avoid gene name false positives
            pattern = rf"\b{re.escape(kw)}\b"
        else:
            # Long keyword: substring match catches compound forms (polysialic → sialic)
            pattern = re.escape(kw)
        mask |= text.str.contains(pattern, case=False, na=False, regex=True)
    return mask

desc_only = confident["description"]
mask_specific = kw_match_mask(desc_only, HMO_SPECIFIC)
mask_generic  = kw_match_mask(desc_only, HMO_GENERIC)
print(f"[Filter 4a] HMO-specific 关键词命中: {int(mask_specific.sum())} proteins")
print(f"[Filter 4b] HMO-generic  关键词命中: {int(mask_generic.sum())} proteins")

# 加分：specific 命中 +2 分，generic 命中 +1 分（用于排序）
confident["hmo_specific_hits"] = mask_specific.astype(int) * 2
confident["hmo_generic_hits"]  = mask_generic.astype(int)
confident["hmo_score"] = confident["hmo_specific_hits"] + confident["hmo_generic_hits"]

# 取 any-match 当 hard filter（保底包含 generic 命中的，但要单独标记 specific）
mask_any = mask_specific | mask_generic
hmo = confident[mask_any].copy()
print(f"[Filter 4] HMO 关键词（any）命中: {len(hmo)} proteins")
print(f"  其中 specific 命中: {int(hmo['hmo_specific_hits'].gt(0).sum())} proteins")

# === 排序：specific 优先 > transport_prob > top1_similarity ===
hmo_sorted = hmo.sort_values(
    by=["hmo_specific_hits", "transport_prob", "top1_similarity"],
    ascending=[False, False, False]
).reset_index(drop=True)

# === 写主输出 ===
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cols = [
    "accession", "gene_or_name", "description",
    "transport_prob", "is_transport", "tc_family", "tc_number",
    "top1_similarity", "top_family_votes", "top_family_purity",
    "hmo_score", "hmo_specific_hits", "hmo_generic_hits",
    "query", "seq_length",
]
hmo_sorted[cols].to_csv(HMO_CSV, index=False)
print(f"\n[Save] {HMO_CSV}  ({len(hmo_sorted)} rows)")

# === Top 50 供人工 review ===
top50 = hmo_sorted.head(50)[
    ["accession", "gene_or_name", "description", "tc_family",
     "tc_number", "transport_prob", "top1_similarity",
     "hmo_specific_hits", "hmo_generic_hits"]
]
top50.to_csv(TOP_CSV, index=False)
print(f"[Save] {TOP_CSV}  (top 50, sorted by hmo_specific_hits > prob > sim)")

# === 打印 top 20 到 stdout ===
print("\n" + "="*120)
print("TOP 20 HMO-RELATED CANDIDATES  (specific matches FIRST, then by transport_prob)")
print("="*120)
for i, row in hmo_sorted.head(20).iterrows():
    spec_mark = "**" if row["hmo_specific_hits"] > 0 else "  "
    print(f"{i+1:2d}{spec_mark}{row['accession']:12s} | {row['gene_or_name']:18s} | "
          f"TC {row['tc_number']:14s} | prob {row['transport_prob']:.3f} | "
          f"sim {row['top1_similarity']:.3f} | spec={int(row['hmo_specific_hits'])}/gen={int(row['hmo_generic_hits'])}")
    print(f"     {row['description'][:110]}")

# === 摘要统计 ===
with open(SUMMARY, "w") as f:
    f.write(f"TClass Swiss-Prot HMO candidate filter\n")
    f.write(f"==========================================\n\n")
    f.write(f"Input:           {INPUT_CSV}\n")
    f.write(f"Total proteins:  {len(df)}\n")
    f.write(f"YES proteins:    {(df['is_transport']=='YES').sum()}\n")
    f.write(f"Sugar families:  {len(sugar)}\n")
    f.write(f"Confident:       {len(confident)}\n")
    f.write(f"HMO keywords:    {len(hmo_sorted)} "
            f"(specific={int(hmo_sorted['hmo_specific_hits'].gt(0).sum())}, "
            f"generic-only={int((hmo_sorted['hmo_specific_hits']==0).sum())})\n\n")
    f.write(f"=== TC family distribution ===\n")
    f.write(hmo_sorted["tc_family"].value_counts().to_string())
    f.write(f"\n\n=== Top 20 ===\n")
    f.write(top50.head(20).to_string(index=False))
    f.write(f"\n\n=== Per-organism distribution (top 20) ===\n")
    # 提取 OS 字段或从 description 猜 organism
    orgs = hmo_sorted["description"].str.extract(
        r"\b(OS:([^=]+?)(?: OX=|$))", expand=False
    )[1].fillna("").str.strip()
    if (orgs == "").all():
        orgs = hmo_sorted["description"].str.split().str[0:3].str.join(" ")
    f.write(orgs.value_counts().head(20).to_string())
print(f"\n[Save] {SUMMARY}")
print(f"\nDone. Open {TOP_CSV} to review top 50 manually.")
