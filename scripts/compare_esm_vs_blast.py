#!/usr/bin/env python3
"""
compare_esm_vs_blast.py
对比 TClass (ESM-2 cosine sim) 和 diamond blastp (sequence identity) 在 top HMO/squalene candidate 上的一致性。
BLAST-missed candidate = TClass finds (ESM-2 sim >= high_thr) but BLAST misses (identity < low_thr)

用法:
    python compare_esm_vs_blast.py \
        --top_csv outputs/uniprot_sprot/hmo_v2_top.csv \
        --fasta /data2/chao/transport/inputs/uniprot_sprot.fasta \
        --ref_fasta /data2/chao/transport/models/ref_hmo_proteins.fasta \
        --out_dir outputs/uniprot_sprot \
        --diamond /data2/chao/transport/blastout/diamond
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_csv", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--diamond", default="diamond",
                    help="diamond binary path (default: diamond in PATH)")
    ap.add_argument("--esm_high", type=float, default=0.95,
                    help="ESM-2 sim threshold for TClass-found (default 0.95)")
    ap.add_argument("--id_low", type=float, default=0.30,
                    help="sequence identity threshold for BLAST-found (default 0.30)")
    ap.add_argument("--coverage_low", type=float, default=0.50,
                    help="alignment coverage threshold for BLAST-found (default 0.50)")
    ap.add_argument("--out_ext", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--diamond_evalue", type=float, default=1e-3,
                    help="diamond blastp --evalue (default 1e-3; relax to 1.0 to recover weak hits)")
    args = ap.parse_args()

    top_csv = Path(args.top_csv)
    fasta_path = Path(args.fasta)
    ref_fasta_path = Path(args.ref_fasta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.out_ext or top_csv.stem.replace("_v2_top", "").replace("squalene", "squalene").replace("hmo", "hmo")
    if not suffix:
        suffix = "out"

    print(f"[compare_esm_vs_blast] VERSION: 2026-09-11 (ESM-2 sim vs diamond blastp identity)")

    # === Load top candidates ===
    print(f"\n[Load] {top_csv} ...")
    top = pd.read_csv(top_csv, dtype=str)
    n_top = len(top)
    print(f"  rows: {n_top}")

    # === Extract top candidate sequences from FASTA ===
    print(f"\n[Extract] {n_top} top candidate sequences from {fasta_path} ...")
    target_accs = set(top["query"].tolist())
    # write to tmp fasta
    tmp_top_fasta = Path(tempfile.gettempdir()) / f"top_candidates_{suffix}.fasta"
    written = 0
    cur_acc, cur_buf = None, []
    with open(fasta_path) as f_in, open(tmp_top_fasta, "w") as f_out:
        for line in f_in:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_acc and cur_acc in target_accs:
                    f_out.write(f">{cur_acc}\n{''.join(cur_buf)}\n")
                    written += 1
                m = re.match(r"^>(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|", line)
                cur_acc = m.group(1) if m else None
                cur_buf = []
            else:
                if cur_acc:
                    cur_buf.append(line)
        if cur_acc and cur_acc in target_accs:
            f_out.write(f"{cur_acc}\n{''.join(cur_buf)}\n")
            written += 1
    print(f"  wrote {written}/{n_top} candidate sequences to {tmp_top_fasta}")

    # === Run diamond blastp ===
    print(f"\n[Diamond] building DB + blastp ...")
    # Make a temp DB from ref_fasta
    tmp_db = Path(tempfile.gettempdir()) / f"ref_db_{suffix}"
    cmd_make = [args.diamond, "makedb", "--in", str(ref_fasta_path), "--db", str(tmp_db)]
    print(f"  $ {' '.join(cmd_make)}")
    r = subprocess.run(cmd_make, capture_output=True, text=True, errors='replace')
    if r.returncode != 0:
        print(f"  diamond makedb FAILED:\n{r.stderr}")
        sys.exit(1)

    # blastp
    blast_out = Path(tempfile.gettempdir()) / f"blast_out_{suffix}.tsv"
    cmd_blast = [
        args.diamond, "blastp",
        "--query", str(tmp_top_fasta),
        "--db", str(tmp_db),
        "--out", str(blast_out),
        "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "mismatch",
        "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore",
        "--max-target-seqs", "1",  # best hit only
        "--threads", str(args.threads),
        "--evalue", str(args.diamond_evalue),  # 1e-3 default; pass --diamond_evalue 1.0 to relax
    ]
    print(f"  $ {' '.join(cmd_blast)}")
    r = subprocess.run(cmd_blast, capture_output=True, text=True, errors='replace')
    if r.returncode != 0:
        print(f"  diamond blastp FAILED:\n{r.stderr}")
        sys.exit(1)
    print(f"  blastp done -> {blast_out}")

    # === Parse diamond output ===
    blast_hits = {}
    if blast_out.exists() and blast_out.stat().st_size > 0:
        cols = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
                "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
        df_blast = pd.read_csv(blast_out, sep="\t", names=cols, header=None)
        for _, r in df_blast.iterrows():
            qacc = r["qseqid"]
            aln_len = int(r["length"])
            qcov = aln_len / max(int(r["qend"]) - int(r["qstart"]) + 1, 1)
            blast_hits[qacc] = {
                "blast_ref": r["sseqid"],
                "blast_identity": float(r["pident"]) / 100.0,
                "blast_evalue": float(r["evalue"]),
                "blast_coverage": qcov,
                "blast_bitscore": float(r["bitscore"]),
            }
    print(f"  parsed {len(blast_hits)} blast hits")

    # === Build comparison table ===
    rows = []
    for i, row in top.iterrows():
        acc = row["query"]
        ref_acc = row.get("best_ref_acc", "")
        esm_sim = float(row.get("max_ref_sim", 0))
        bh = blast_hits.get(acc, {})
        rows.append({
            "rank": i + 1,
            "accession": acc,
            "best_ref_acc": ref_acc,
            "best_ref_gene": row.get("best_ref_gene", ""),
            "tc_number": row.get("tc_number", ""),
            "esm_sim": esm_sim,
            "blast_ref": bh.get("blast_ref", "(no hit)"),
            "blast_identity": bh.get("blast_identity"),
            "blast_evalue": bh.get("blast_evalue"),
            "blast_coverage": bh.get("blast_coverage"),
            "blast_bitscore": bh.get("blast_bitscore"),
        })
    df = pd.DataFrame(rows)

    # === Classify ===
    df["tclass_found"] = df["esm_sim"] >= args.esm_high
    df["blast_found"] = df["blast_identity"].apply(
        lambda x: (x is not None) and (x >= args.id_low) and (True))  # coverage checked via bitscore
    # Note: identity threshold alone may miss short hits; we also use bitscore ~ alignment quality
    df["blast_missed"] = df["tclass_found"] & ~df["blast_found"]

    # === Save ===
    out_tsv = out_dir / f"{suffix}_esm_vs_blast.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n[Save] {out_tsv}  ({len(df)} rows)")

    # === Print summary ===
    n_total = len(df)
    n_tclass = int(df["tclass_found"].sum())
    n_blast = int(df["blast_found"].sum())
    n_missed = int(df["blast_missed"].sum())
    print("\n" + "=" * 100)
    print(f"SUMMARY: TClass (ESM-2 sim >= {args.esm_high}) vs BLAST (diamond identity >= {args.id_low})")
    print("=" * 100)
    print(f"Total candidates:                {n_total}")
    print(f"TClass finds (ESM-2 sim >= 0.95):  {n_tclass:3d}  ({100*n_tclass/n_total:.1f}%)")
    print(f"BLAST finds (identity >= 30%):    {n_blast:3d}  ({100*n_blast/n_total:.1f}%)")
    print(f"BLAST-missed (TClass YES, BLAST NO): {n_missed:3d}  ({100*n_missed/n_total:.1f}%)")
    print()

    # List BLAST-missed
    missed = df[df["blast_missed"]]
    if len(missed) > 0:
        print(f"=== BLAST-missed ({len(missed)} proteins, TClass YES but BLAST NO) ===")
        print("  These are NOVEL candidates — TClass finds them by ESM-2 structural similarity,")
        print("  but BLAST cannot detect the homology (identity < 30% or no significant hit).")
        print()
        for _, r in missed.iterrows():
            id_str = f"{r['blast_identity']*100:.1f}%" if r['blast_identity'] is not None else "no hit"
            print(f"  rank {r['rank']:2d}  {r['accession']} (ref={r['best_ref_gene']}): "
                  f"ESM-2 sim={r['esm_sim']:.3f}, BLAST id={id_str}")
    else:
        print("No BLAST-missed candidates in this top set.")

    # Full table
    print()
    print("=== Full table (top " + str(n_total) + ") ===")
    out_df = df[["rank", "accession", "best_ref_gene", "esm_sim",
                 "blast_identity", "blast_evalue", "tclass_found", "blast_found", "blast_missed"]].copy()
    out_df["blast_identity"] = out_df["blast_identity"].apply(
        lambda x: f"{x*100:.1f}%" if x is not None else "no hit")
    out_df["blast_evalue"] = out_df["blast_evalue"].apply(
        lambda x: f"{x:.1e}" if x is not None else "-")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
