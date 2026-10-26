#!/usr/bin/env python3
"""
merge_final_reports.py <out_dir> <p0_dir> <p1_dir> [...]
合并多个 TClass 输出的 final_report.csv 到一个目录。
用法: merge_final_reports.py outputs/uniprot_sprot outputs/uniprot_sprot_p0 outputs/uniprot_sprot_p1
"""
import sys
import shutil
from pathlib import Path
import pandas as pd

def main():
    if len(sys.argv) < 3:
        print("Usage: merge_final_reports.py <out_dir> <p0_dir> [p1_dir] ...")
        sys.exit(1)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    csvs = [Path(d) / "final_report.csv" for d in sys.argv[2:]]
    csvs = [c for c in csvs if c.exists()]
    if not csvs:
        print("ERROR: no final_report.csv found in any input dir")
        sys.exit(1)
    print(f"Merging {len(csvs)} files:")
    for c in csvs:
        print(f"  {c}  ({c.stat().st_size/1024/1024:.1f} MB)")

    dfs = [pd.read_csv(c) for c in csvs]
    merged = pd.concat(dfs, ignore_index=True)
    out_csv = out / "final_report.csv"
    merged.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}  ({len(merged)} rows, {out_csv.stat().st_size/1024/1024:.1f} MB)")

    # also merge phase1 / phase2
    for name in ["phase1_report.csv", "phase2_report.csv"]:
        sub_csvs = [Path(d) / name for d in sys.argv[2:]]
        sub_csvs = [c for c in sub_csvs if c.exists()]
        if sub_csvs:
            sub_dfs = [pd.read_csv(c) for c in sub_csvs]
            sub_merged = pd.concat(sub_dfs, ignore_index=True)
            sub_out = out / name
            sub_merged.to_csv(sub_out, index=False)
            print(f"  -> {sub_out}  ({len(sub_merged)} rows)")

if __name__ == "__main__":
    main()
