#!/usr/bin/env python
"""
parse_blast_to_tcdb.py
======================
把 DIAMOND/blastp 的结果跟 TCDB 描述关联, 输出跟 v3 final_report 格式一致的 CSV.

输入:
  1. DIAMOND blastp -outfmt 6 结果 (tab 分隔, header 行可选)
     列: qseqid, sseqid, pident, length, mismatch, gapopen, qstart, qend,
          sstart, send, evalue, bitscore
  2. TCDB 注释 tsv (格式: tc_number<TAB>description), 如 tcdb_annot.clean.tsv
  3. (可选) v3 final_report.csv, 用于 v3 vs BLAST 直接对比

输出:
  blast_results.tsv (主文件, 每个 query 一行, 含 TC 号 + 描述)
  comparison.tsv   (如果给了 v3_report, 输出 v3 vs BLAST 的逐行对比)
  blast_stats.txt  (汇总: 命中率, TC 号分布, e-value 分布, 等)

用法:
    python3 parse_blast_to_tcdb.py \\
        --blast  max_results.tsv \\
        --annot /data2/chao/transport/tcdb_annot.clean.tsv \\
        --out   blast_output/

    # 带 v3 对比:
    python3 parse_blast_to_tcdb.py \\
        --blast  max_results.tsv \\
        --annot /data2/chao/transport/tcdb_annot.clean.tsv \\
        --v3     outputs/yli/final_report.csv \\
        --out    blast_output/
"""

import argparse
import re
from collections import Counter
from pathlib import Path


# DIAMOND -outfmt 6 列名 (按调用顺序)
DIAMOND_COLS = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore",
]
# 注: 你那次调用没加 qlen/slen/qcovhsp, 后面 coverage 会从 length 估


def normalize_query(q: str) -> str:
    """统一 query ID 格式: 去掉 UniProt 前缀, 只保留 accession.

    例: 'sp|W0T9X4|ATG1_KLUMD' -> 'W0T9X4'
        'tr|A0ABX6F2G7|A0ABX6F2G7_KLUMA' -> 'A0ABX6F2G7'
        'W0T9X4' -> 'W0T9X4' (已经是 accession, 原样返回)
    """
    if not isinstance(q, str) or not q:
        return q
    # 匹配 'sp|XXX|...' 或 'tr|XXX|...' 格式
    m = re.match(r"^(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|", q)
    if m:
        return m.group(1)
    # 已经是纯 accession (或别的), 原样返回
    return q.split()[0] if " " in q else q


def parse_tc_from_subject(sseqid: str) -> str | None:
    """从 sseqid 提取 TC 号. 格式: gnl|TC-DB|<acc>|<tc_number>
    例: gnl|TC-DB|P53104|9.A.15.1.1 -> '9.A.15.1.1'
    """
    if not isinstance(sseqid, str) or "|" not in sseqid:
        return None
    parts = sseqid.split("|")
    if len(parts) < 4:
        return None
    # TC 号在第 4 段
    tc_candidate = parts[3].strip()
    # 验证格式: X.X.X.X[.X]
    if re.match(r"^\d+\.[A-Z]+\.\d+\.\d+(?:\.\d+)?$", tc_candidate):
        return tc_candidate
    return None


def load_annot(annot_path: str) -> dict[str, str]:
    """读 tcdb_annot.clean.tsv -> {tc_number: description}"""
    annots: dict[str, str] = {}
    with open(annot_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                annots[parts[0].strip()] = parts[1]
    return annots


def load_blast(blast_path: str) -> "pd.DataFrame":
    """读 DIAMOND -outfmt 6, 跳过以 # 开头的注释行"""
    import pandas as pd
    df = pd.read_csv(
        blast_path, sep="\t", header=None, names=DIAMOND_COLS,
        comment="#", dtype={"qseqid": str, "sseqid": str},
    )
    return df


def annotate_blast(df, annots: dict[str, str]) -> "pd.DataFrame":
    """给 BLAST 结果加 TC 号 + 描述 + 估算 coverage"""
    import pandas as pd
    df = df.copy()
    df["tc_number"] = df["sseqid"].map(parse_tc_from_subject)
    df["tc_description"] = df["tc_number"].map(annots).fillna("(no description)")
    # 估算 query coverage: alignment length / query length (query length 不可知,
    # 用 alignment length 占位). 如果以后有 qlen 列就改用 qcovhsp
    df["aln_length"] = df["length"]
    return df


def best_hit_per_query(df):
    """每个 query 只保留一个最好的 hit (按 bitscore 降序, evalue 升序)"""
    import pandas as pd
    df = df.sort_values(["qseqid", "bitscore", "evalue"],
                        ascending=[True, False, True])
    return df.drop_duplicates("qseqid", keep="first")


def load_v3_report(v3_path: str) -> dict[str, dict]:
    """读 v3 final_report.csv -> {normalized_query: {tc_number, is_transport, raw_query}}"""
    import pandas as pd
    df = pd.read_csv(v3_path)
    out = {}
    for _, row in df.iterrows():
        raw_q = str(row["query"])
        q = normalize_query(raw_q)
        out[q] = {
            "tc_number": str(row.get("tc_number", "")),
            "is_transport": str(row.get("is_transport", "")),
            "raw_query": raw_q,
        }
    return out


def compare_v3_vs_blast(blast_df, v3_map, annots):
    """生成 v3 vs BLAST 对比表. query 用 normalized 后的 ID 匹配.

    遍历 BLAST top-1 ∪ v3 final_report 的**并集**，完整覆盖 5186 个 query。
    这样 v3 在 BLAST 没命中的 query 上预测的 YES 会被算作 v3_only (novel discovery)。

    分类 (match_level):
      yes       : v3 YES + BLAST TC 一致      (TP, 都同意)
      disagree  : v3 YES + BLAST TC 不一致    (FP, v3 有不同意见)
      v3_only   : v3 YES + BLAST 无命中       (FP, v3 novel 发现, paper 主打)
      blast_only: BLAST 命中 + v3 NO          (FN, BLAST 找到了 v3 漏)
      neither   : BLAST 无命中 + v3 NO        (TN, 都同意)
    """
    import pandas as pd
    rows = []
    blast_top = best_hit_per_query(blast_df)
    # 把 blast_top 转成 normalized qid -> Series 的 dict
    blast_map = {}
    for _, b in blast_top.iterrows():
        qid = normalize_query(b["qseqid"])
        blast_map[qid] = b
    # 并集: BLAST 命中的 query ∪ v3 final_report 所有 query
    all_qids = sorted(set(blast_map.keys()) | set(v3_map.keys()))
    n_in_both = sum(1 for q in all_qids if q in blast_map and q in v3_map)
    n_blast_only_q = sum(1 for q in all_qids if q in blast_map and q not in v3_map)
    n_v3_only_q = sum(1 for q in all_qids if q in v3_map and q not in blast_map)

    for qid in all_qids:
        b = blast_map.get(qid)
        v3 = v3_map.get(qid, {})
        raw_qid = b["qseqid"] if b is not None else v3.get("raw_query", qid)
        blast_tc = ""
        blast_evalue = None
        blast_pident = None
        blast_bitscore = None
        if b is not None and pd.notna(b.get("tc_number")):
            blast_tc = b["tc_number"]
            blast_evalue = b["evalue"]
            blast_pident = b["pident"]
            blast_bitscore = b["bitscore"]
        # v3 only contributes TC if Phase 1 said "YES"
        v3_tc = v3.get("tc_number", "") if v3.get("is_transport") == "YES" else ""

        if v3_tc and blast_tc:
            match_level = "yes" if v3_tc == blast_tc else "disagree"
        elif v3_tc and not blast_tc:
            match_level = "v3_only"           # v3 找到，BLAST 没有 → novel discovery
        elif blast_tc and not v3_tc:
            match_level = "blast_only"        # BLAST 找到，v3 没找到
        else:
            match_level = "neither"           # 都没找到
        rows.append({
            "query": qid,
            "raw_blast_query": raw_qid,
            "v3_tc": v3_tc,
            "v3_tc_desc": annots.get(v3_tc, ""),
            "v3_is_transport": v3.get("is_transport", ""),
            "blast_tc": blast_tc,
            "blast_tc_desc": annots.get(blast_tc, ""),
            "blast_evalue": blast_evalue,
            "blast_pident": blast_pident,
            "blast_bitscore": blast_bitscore,
            "match": match_level,
        })
    print(f"[Compare] union={len(all_qids)} | both={n_in_both} "
          f"| BLAST-only-q={n_blast_only_q} | v3-only-q={n_v3_only_q} "
          f"(iterated over {len(blast_top)} BLAST top-1 + {len(v3_map) - n_in_both} v3-no-blast queries)")
    return pd.DataFrame(rows)


def write_stats(df, v3_df, out_path):
    """写 BLAST 命中统计 + (如果给了 v3) 对比统计"""
    lines = []
    n_total = len(df)
    n_with_tc = df["tc_number"].notna().sum()
    n_unique_q = df["qseqid"].nunique()
    top = best_hit_per_query(df)
    n_top_with_tc = top["tc_number"].notna().sum()

    lines.append("=" * 60)
    lines.append("  BLAST vs TCDB - Hit Statistics")
    lines.append("=" * 60)
    lines.append(f"  Total blast hits (raw)     : {n_total}")
    lines.append(f"  Unique queries            : {n_unique_q}")
    lines.append(f"  Hits with parseable TC    : {n_with_tc} ({100*n_with_tc/max(1,n_total):.1f}%)")
    lines.append(f"  Top-1 hits                : {len(top)}")
    lines.append(f"  Top-1 hits with TC        : {n_top_with_tc} "
                 f"({100*n_top_with_tc/max(1,len(top)):.1f}%)")
    lines.append("")

    if n_with_tc:
        lines.append("  -- E-value distribution (top-1 hits) --")
        for thresh in (1e-100, 1e-50, 1e-30, 1e-20, 1e-10, 1e-5):
            n = (top["evalue"] <= thresh).sum()
            lines.append(f"    <= {thresh:.0e}  : {n:>5}")
        lines.append("")

        lines.append("  -- % identity distribution (top-1 hits) --")
        for thresh in (90, 70, 50, 40, 30):
            n = (top["pident"] >= thresh).sum()
            lines.append(f"    >= {thresh}%   : {n:>5}")
        lines.append("")

        lines.append("  -- TC class distribution (top-1 hits) --")
        cls = top["tc_number"].dropna().map(lambda x: x.split(".")[0]).value_counts().sort_index()
        for k, v in cls.items():
            lines.append(f"    Class {k:<3}: {v:>5}")
        lines.append("")

        lines.append("  -- Top 10 TC families (top-1 hits) --")
        fam = top["tc_number"].dropna().value_counts().head(10)
        for tc, v in fam.items():
            lines.append(f"    {tc:<15}: {v:>5}")
        lines.append("")

    if v3_df is not None and len(v3_df) > 0:
        lines.append("=" * 60)
        lines.append("  v3 vs BLAST Comparison")
        lines.append("=" * 60)
        match_counts = v3_df["match"].value_counts()
        for k, v in match_counts.items():
            lines.append(f"  {k:<15}: {v:>5}")
        lines.append("")
        if "yes" in match_counts:
            n_yes = match_counts["yes"]
            n_total_comp = len(v3_df)
            lines.append(f"  Agreement (exact TC match): "
                         f"{100*n_yes/max(1,n_total_comp):.1f}%  "
                         f"({n_yes}/{n_total_comp})")
        if "v3_only" in match_counts:
            lines.append(f"  v3 finds, BLAST misses: {match_counts['v3_only']}  "
                         f"← 这就是你的核心卖点 (远缘同源)")
        if "disagree" in match_counts:
            lines.append(f"  v3 finds, BLAST disagrees on TC: {match_counts['disagree']}  "
                         f"(v3+hit 但 TC 不同)")
        if "blast_only" in match_counts:
            lines.append(f"  BLAST finds, v3 misses: {match_counts['blast_only']}")
        # v3 预测的"自我一致性" — 在 v3 自己说 YES 的那些里，有多少跟 BLAST 吻合
        n_v3_yes = (match_counts.get("yes", 0) + match_counts.get("disagree", 0)
                    + match_counts.get("v3_only", 0))
        if n_v3_yes > 0:
            conf = match_counts.get("yes", 0) / n_v3_yes
            lines.append(f"  v3 self-consistency (yes / v3_YES): "
                         f"{100*conf:.1f}%  ({match_counts.get('yes',0)}/{n_v3_yes})")
        lines.append("")

    txt = "\n".join(lines)
    out_path.write_text(txt)
    print(txt)
    print(f"[save] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blast", required=True, help="DIAMOND -outfmt 6 result file")
    ap.add_argument("--annot", required=True, help="TCDB annotation tsv (e.g. tcdb_annot.clean.tsv)")
    ap.add_argument("--v3", help="(optional) v3 final_report.csv for comparison")
    ap.add_argument("--out", default="./blast_output", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] BLAST: {args.blast}")
    df = load_blast(args.blast)
    print(f"  {len(df)} raw hits, {df['qseqid'].nunique()} unique queries")

    print(f"[Load] Annot: {args.annot}")
    annots = load_annot(args.annot)
    print(f"  {len(annots)} TC descriptions")

    print("[Annotate] joining TC numbers + descriptions ...")
    df = annotate_blast(df, annots)
    n_with_tc = df["tc_number"].notna().sum()
    print(f"  {n_with_tc} hits have parseable TC numbers")

    # 主结果: 每个 query 取 top-1
    top = best_hit_per_query(df)
    out_csv = out_dir / "blast_top_hits.csv"
    top.to_csv(out_csv, index=False)
    print(f"[Save] {out_csv}  ({len(top)} queries)")

    # 也保存所有 hits (top-k)
    out_all = out_dir / "blast_all_hits.csv"
    df.to_csv(out_all, index=False)
    print(f"[Save] {out_all}  ({len(df)} hits)")

    # v3 对比
    v3_df = None
    if args.v3:
        print(f"[Load] v3 report: {args.v3}")
        v3_map = load_v3_report(args.v3)
        v3_df = compare_v3_vs_blast(df, v3_map, annots)
        out_cmp = out_dir / "v3_vs_blast.csv"
        v3_df.to_csv(out_cmp, index=False)
        print(f"[Save] {out_cmp}  ({len(v3_df)} queries compared)")

    # 统计
    write_stats(df, v3_df, out_dir / "blast_stats.txt")


if __name__ == "__main__":
    main()