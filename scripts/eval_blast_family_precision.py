#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_blast_family_precision.py
================================

Compare DIAMOND BLASTp (against TCDB) vs v3 on the same 8-species narrow GT
(UniProt ∩ TCDB cross-ref). Same precision/recall_fuzzy/recall_strict metrics
as eval_family_precision.py so numbers are directly comparable.

DIAMOND params (matching the original Section 3.8 protocol):
  e-value <= 1e-30
  identity >= 40%
  query-cover >= 80%
  max-target-seqs = 1

Inputs:
  /data2/chao/transport/inputs/<sp>.fasta
  /data2/chao/transport/data/tcdb.dmnd
  /data2/chao/transport/data/tcdb_annot.clean.tsv
  /data2/chao/transport/data/uniprot_with_tc/uniprot_<sp>_transporters_with_tc.tsv

Outputs:
  /data2/chao/transport/outputs/blast_eval/blast_<sp>.tsv   (per-species DIAMOND top hits)
  /data2/chao/transport/outputs/blast_eval/Table3_BLAST.csv   (aggregated metrics)

Usage:
    python eval_blast_family_precision.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


SPECIES = [
    # (3b_dir_name,        gt_fname,      fasta_fname,   blast_alias)
    ("E_coli_K12_MG1655",  "MG1655",      "MG1655.fasta", "MG1655"),
    ("S_cerevisiae_S288C", "s288c",       "s288c.fasta",  "s288c"),
    ("B_subtilis_168",     "bsub",        "bsub.fasta",   "bsub"),
    ("L_lactis",           "IL1403",      "lacto.fasta",  "lacto"),
    ("C_glutamicum",       "cgl",         "cgl.fasta",    "cgl"),
    ("A_niger",            "aniger",      "aniger.fasta", "aniger"),
    ("F_fujikuroi",        "fuji",        "fuji.fasta",   "fuji"),
    ("B_cinerea",          "bcin",        "bcin.fasta",   "bcin"),
]


def to_4digit(tc):
    if not isinstance(tc, str) or not tc:
        return ""
    s = tc.strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    parts = s.split(".")
    if len(parts) >= 4:
        return ".".join(parts[:4])
    return s


def parse_tc_from_sseqid(sseqid):
    """Extract TC number from a TCDB DIAMOND sseqid.

    TCDB sseqid format: 'gnl|TC-DB|<id>|<TC_number>'  e.g. gnl|TC-DB|1CN3_F|1.A.83.1.5
    The TC number (X.X.X.X or X.X.X.X.X) is embedded in the string and may be the
    last | segment. We use regex to robustly find it.
    """
    if not isinstance(sseqid, str) or not sseqid:
        return ""
    m = re.search(r"(\d+\.[A-Z]\.\d+\.\d+(?:\.\d+)?)", sseqid)
    return m.group(1) if m else ""


def extract_uniprot_acc(qseqid):
    """Extract a UniProt accession from a fasta header qseqid.

    Handles:
      sp|P0A123|name desc          -> P0A123
      tr|P0A123|name desc          -> P0A123
      gi|<num>|P0A123 desc        -> P0A123
      gnl|DB|<id>|P0A123|...       -> P0A123
      P0A123 some desc              -> P0A123
    """
    if not qseqid:
        return ""
    s = str(qseqid).strip()
    # 1) sp|ACC|... or tr|ACC|...
    m = re.match(r"^(?:sp|tr)\|([A-Z0-9]+)\|", s)
    if m:
        return m.group(1)
    # 2) gi|<num>|<acc>
    m = re.match(r"^gi\|\d+\|([A-Z0-9]+)", s)
    if m:
        return m.group(1)
    # 3) pipe-separated: scan parts, skip TC-like patterns, return first
    #    token that looks like a UniProt accession (>=5 chars, [A-Z][A-Z0-9]+)
    if "|" in s:
        for p in s.split("|"):
            p = p.strip()
            if not p or re.match(r"^\d+\.[A-Z]\.", p):
                continue
            if re.match(r"^[A-Z][A-Z0-9]{4,}$", p):
                return p
    # 4) plain: first whitespace-delimited token
    return s.split()[0] if s else ""


def run_diamond(fasta, db, out_tsv, threads=4):
    """Run DIAMOND BLASTp with the standard v3 paper protocol."""
    cmd = [
        "diamond", "blastp",
        "--query", fasta,
        "--db", db,
        "--out", out_tsv,
        "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "qcovhsp", "evalue",
        "--evalue", "1e-30",
        "--id", "40",
        "--query-cover", "80",
        "--max-target-seqs", "1",
        "--threads", str(threads),
        "--quiet",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERROR] diamond failed: {r.stderr}")
        return False
    return True


def main():
    base = Path("/data2/chao/transport")
    fasta_dir = base / "inputs"
    db = base / "data" / "tcdb.dmnd"
    gt_dir = base / "data" / "uniprot_with_tc"
    out_dir = base / "outputs" / "blast_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db.exists():
        sys.exit(f"[ERROR] TCDB DIAMOND db not found: {db}")

    # No need to pre-parse tcdb.fa: DIAMOND sseqid is the full header
    # (e.g. "gnl|TC-DB|1CN3_F|1.A.83.1.5") and contains the TC number
    # embedded. We extract TC number per hit via regex.

    # For each species: run DIAMOND, parse top hit, eval against GT
    rows = []
    for sp_dir, gt_fname, fasta_fname, sp_alias in SPECIES:
        fasta = fasta_dir / fasta_fname
        gt_tsv = gt_dir / f"uniprot_{gt_fname}_transporters_with_tc.tsv"
        blast_tsv = out_dir / f"blast_{sp_alias}.tsv"
        out_csv = out_dir / f"Table3_{sp_alias}.csv"

        if not fasta.exists():
            print(f"[SKIP] {sp_alias}: fasta not found ({fasta})")
            continue
        if not gt_tsv.exists():
            print(f"[SKIP] {sp_alias}: GT not found ({gt_tsv})")
            continue

        # Run DIAMOND
        print(f"\n[{sp_alias}] DIAMOND blastp {fasta.name} ...")
        if not run_diamond(str(fasta), str(db), str(blast_tsv)):
            print(f"[{sp_alias}] DIAMOND failed, skip")
            continue

        # Parse top hit per query
        blast = pd.read_csv(blast_tsv, sep="\t", header=None,
                            names=["qseqid", "sseqid", "pident", "length", "qcovhsp", "evalue"])
        # top hit = first row per qseqid (DIAMOND already sorts by score, --max-target-seqs 1)
        top_hits = blast.groupby("qseqid").first().reset_index()
        # Parse TC number from sseqid (embedded in DIAMOND header)
        top_hits["tc_number"] = top_hits["sseqid"].astype(str).apply(parse_tc_from_sseqid)
        top_hits["tc_family_4digit"] = top_hits["tc_number"].apply(to_4digit)
        n_with_tc = int((top_hits["tc_family_4digit"] != "").sum())
        print(f"  Top hits: {len(top_hits)} queries, {n_with_tc} mapped to TC family")

        # Load GT
        gt = pd.read_csv(gt_tsv, sep="\t")
        gt_lookup = {}
        for _, grow in gt.iterrows():
            acc = str(grow["Entry"])
            gt4 = to_4digit(str(grow["tc_family_4digit"]))
            if acc and gt4:
                gt_lookup[acc] = gt4
        n_gt = len(gt_lookup)
        print(f"  GT size: {n_gt} (4-digit)")

        # Eval: among BLAST top hits in GT, what fraction matches at 4-digit?
        # BLAST hit is "predict YES + family" if tc_family_4digit is non-empty.
        # Compare BLAST predictions vs GT the same way as v3.
        correct = 0
        n_eval = 0
        n_pred_yes = 0
        n_acc_mapped = 0
        for _, row in top_hits.iterrows():
            qacc = extract_uniprot_acc(row["qseqid"])
            if qacc:
                n_acc_mapped += 1
            pred4 = to_4digit(str(row["tc_family_4digit"]))
            if pred4:
                n_pred_yes += 1
                if qacc in gt_lookup:
                    n_eval += 1
                    if pred4 == gt_lookup[qacc]:
                        correct += 1
        print(f"  acc-mapped: {n_acc_mapped}/{len(top_hits)}")

        precision = correct / n_eval if n_eval > 0 else 0.0
        recall_fuzzy = n_eval / n_gt if n_gt > 0 else 0.0
        recall_strict = correct / n_gt if n_gt > 0 else 0.0
        n_proteins = len(top_hits)

        print(f"  [{sp_alias}] pred_yes={n_pred_yes} matched={n_eval} correct={correct} | "
              f"prec={precision:.4f} rec_f={recall_fuzzy:.4f} rec_s={recall_strict:.4f}")

        # save per-species row
        rows.append({
            "species": sp_alias,
            "n_proteins": n_proteins,
            "n_pred_yes": n_pred_yes,
            "n_matched_to_gt": n_eval,
            "n_correct": correct,
            "n_gt": n_gt,
            "precision_4digit": precision,
            "recall_fuzzy": recall_fuzzy,
            "recall_strict": recall_strict,
        })

    # Aggregate
    out_df = pd.DataFrame(rows)
    out_path = out_dir / "Table3_BLAST.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Save] {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
