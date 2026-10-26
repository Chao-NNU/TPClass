#!/usr/bin/env python3
"""
run_species_pipeline.py - End-to-end v3 + BLAST + UniProt calibration for any species.

Usage:
  python run_species_pipeline.py /path/to/genome.fasta
  python run_species_pipeline.py /path/to/genome.fasta --gpu 3
  python run_species_pipeline.py /path/to/genome.fasta --no-uniprot
  python run_species_pipeline.py /path/to/genome.fasta --taxid 83333

Automatic steps:
  1. Parse taxid from FASTA header (OX=xxxxx), species name from filename
  2. Run DIAMOND BLAST vs TCDB
  3. Run v3 prediction (Phase 1 + Phase 2)
  4. Run parse_blast_to_tcdb for v3 vs BLAST comparison
  5. (Optional) Download UniProt transporter annotations for this species as ground truth
  6. Run validate_v3_only.py (calibration + head-to-head)

All outputs go to /data2/chao/transport/outputs/<species>/ and /blastout/,
matching the manual workflow structure. s4pred cache lives at v3's default
location (<out_dir>/ss_results/), so existing .horiz files are reused.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ── Defaults (matches project layout) ───────────────────────────────────
TRANSPORT_ROOT = Path("/data2/chao/transport")
TCDB_ANNOT = TRANSPORT_ROOT / "data" / "tcdb_annot.clean.tsv"
TCDB_DMND = TRANSPORT_ROOT / "data" / "tcdb.dmnd"
DATA_DIR = TRANSPORT_ROOT / "data"
BLASTOUT_DIR = TRANSPORT_ROOT / "blastout"
OUTPUTS_DIR = TRANSPORT_ROOT / "outputs"

SCRIPTS_DIR = TRANSPORT_ROOT / "scripts"


def step(msg: str):
    """Print a banner-style step header."""
    print(f"\n{'=' * 70}\n {msg}\n{'=' * 70}", flush=True)


def run(cmd: list[str], timeout: int | None = None, env: dict | None = None) -> int:
    """Run a shell command, streaming output, return exit code."""
    print(f"\n[exec] {' '.join(cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    try:
        return subprocess.run(cmd, timeout=timeout, env=e).returncode
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT after {timeout}s]", flush=True)
        return 124


def parse_species(fasta: Path) -> str:
    """Derive species tag from filename: 'MG1655.fasta' -> 'MG1655', 'yli.fasta' -> 'yli'.
    Preserves case (only replaces spaces with underscores)."""
    stem = fasta.stem
    return stem.replace(" ", "_")


def parse_taxid(fasta: Path) -> int | None:
    """Parse taxid from the OX= field of the first FASTA header."""
    try:
        with open(fasta) as f:
            for line in f:
                if line.startswith(">"):
                    m = re.search(r"OX=(\d+)", line)
                    if m:
                        return int(m.group(1))
                    break
    except OSError:
        pass
    return None


def download_uniprot_transporters(taxid: int, out_path: Path) -> bool:
    """Download UniProt accessions annotated with GO:0022857 or KW-0813 for the given taxid."""
    if out_path.exists() and out_path.stat().st_size > 100:
        print(f"[Skip] {out_path} already exists", flush=True)
        return True
    url = (
        "https://rest.uniprot.org/uniprotkb/stream?"
        f"query=taxonomy_id%3A{taxid}+AND+%28go%3A0022857+OR+keyword%3AKW-0813%29"
        "&format=tsv&fields=accession,id,protein_name,go_id,keyword"
    )
    print(f"[curl] {url}", flush=True)
    rc = run(["curl", "-sL", url, "-o", str(out_path)], timeout=300)
    if rc != 0:
        print(f"[curl FAILED] rc={rc}", flush=True)
        return False
    # quick check
    with open(out_path) as f:
        lines = f.readlines()
    n_acc = max(0, len(lines) - 1)
    print(f"[OK] {n_acc} UniProt transporter accessions for taxid {taxid}", flush=True)
    return n_acc > 0


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end v3 + BLAST + UniProt pipeline for any species FASTA")
    ap.add_argument("fasta", help="input FASTA path")
    ap.add_argument("--gpu", default="3", help="CUDA device for v3 (default 3)")
    ap.add_argument("--taxid", type=int, default=None,
                    help="override taxid (otherwise parsed from FASTA header)")
    ap.add_argument("--no-uniprot", action="store_true",
                    help="skip UniProt download + validate_v3_only (just v3 + BLAST compare)")
    ap.add_argument("--sim-cutoff", type=float, default=0.90,
                    help="sim cutoff for validate_v3_only (default 0.90)")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="v3 Phase 1 threshold (default 0.30)")
    ap.add_argument("--threads", default="8", help="DIAMOND threads (default 8)")
    args = ap.parse_args()

    fasta = Path(args.fasta).resolve()
    if not fasta.exists():
        print(f"[Error] {fasta} not found", file=sys.stderr)
        return 1

    species = parse_species(fasta)
    taxid = args.taxid or parse_taxid(fasta)
    out_dir = OUTPUTS_DIR / species
    # All compare outputs (blast_top_hits, v3_vs_blast, blast_stats) go directly
    # into out_dir alongside final_report.csv, so validate_v3_only.py can find
    # both final_report.csv and v3_vs_blast.csv in one thresh_dir.
    blast_tsv = BLASTOUT_DIR / f"{species}_results.tsv"
    s4pred_dir = out_dir / "ss_results"

    out_dir.mkdir(parents=True, exist_ok=True)
    s4pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSpecies     : {species}")
    print(f"Taxid       : {taxid}")
    print(f"FASTA       : {fasta}")
    print(f"v3 output   : {out_dir}")
    print(f"BLAST output: {blast_tsv}")
    print(f"s4pred cache: {s4pred_dir}  (v3 default; reused if .horiz files exist)")
    print(f"UniProt GT  : {'yes' if not args.no_uniprot else 'no'}")
    print(f"GPU         : {args.gpu}")
    print(f"Threshold   : {args.threshold}")
    print(f"sim cutoff  : {args.sim_cutoff}")

    # ── 1. DIAMOND BLAST ───────────────────────────────────────────────
    step(f"Step 1: DIAMOND BLAST vs TCDB ({fasta.name})")
    if blast_tsv.exists() and blast_tsv.stat().st_size > 100:
        print(f"[Skip] {blast_tsv} already exists ({blast_tsv.stat().st_size} bytes)")
    else:
        cmd = [
            "diamond", "blastp",
            "--query", str(fasta),
            "--db", str(TCDB_DMND),
            "--evalue", "1e-30", "--id", "40", "--query-cover", "80",
            "--max-target-seqs", "1",
            "--outfmt", "6",
            "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
            "qstart", "qend", "sstart", "send", "evalue", "bitscore",
            "--threads", args.threads,
            "--out", str(blast_tsv),
        ]
        rc = run(cmd, timeout=1800)
        if rc != 0:
            print(f"[BLAST FAILED] rc={rc}", file=sys.stderr)
            return 1

    # quick stats
    with open(blast_tsv) as f:
        n_hits = sum(1 for _ in f)
    print(f"[BLAST] {n_hits} hits written to {blast_tsv}", flush=True)

    # ── 2. v3 prediction ───────────────────────────────────────────────
    step(f"Step 2: v3 prediction (threshold={args.threshold})")
    final_csv = out_dir / "final_report.csv"
    if final_csv.exists() and final_csv.stat().st_size > 100:
        print(f"[Skip] {final_csv} already exists")
    else:
        # Don't pass --s4pred_outdir: let v3 use its default (<out_dir>/ss_results)
        # so existing .horiz files in outputs/<species>/ss_results/ are reused.
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "predict_transport_family_v3.py"),
            "-in", str(fasta),
            "-out", str(out_dir),
            "--threshold", str(args.threshold),
            "--tcdb_annot", str(TCDB_ANNOT),
            "--no_plot",
        ]
        env = {"CUDA_VISIBLE_DEVICES": args.gpu}
        rc = run(cmd, timeout=7200, env=env)
        if rc != 0:
            print(f"[v3 FAILED] rc={rc}", file=sys.stderr)
            return 1

    # ── 3. v3 vs BLAST comparison ──────────────────────────────────────
    step("Step 3: v3 vs BLAST comparison")
    if (out_dir / "blast_stats.txt").exists():
        print(f"[Skip] {out_dir / 'blast_stats.txt'} already exists")
    else:
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "parse_blast_to_tcdb.py"),
            "--blast", str(blast_tsv),
            "--annot", str(TCDB_ANNOT),
            "--v3", str(final_csv),
            "--out", str(out_dir),
        ]
        rc = run(cmd, timeout=600)
        if rc != 0:
            print(f"[compare FAILED] rc={rc}", file=sys.stderr)
            return 1

    # ── 4. UniProt ground truth ────────────────────────────────────────
    if not args.no_uniprot and taxid:
        step(f"Step 4: UniProt ground truth (taxid {taxid})")
        uniprot_tsv = DATA_DIR / f"uniprot_{species}_transporters.tsv"
        ok = download_uniprot_transporters(taxid, uniprot_tsv)
        if ok:
            # ── 5. validate_v3_only (calibration + head-to-head) ───────
            step(f"Step 5: validate_v3_only (sim_cutoff={args.sim_cutoff})")
            validation_dir = out_dir / "validation"
            cmd = [
                sys.executable, str(SCRIPTS_DIR / "validate_v3_only.py"),
                "--thresh_dir", str(out_dir),
                "--annot", str(TCDB_ANNOT),
                "--uniprot_tsv", str(uniprot_tsv),
                "--sim_cutoff", str(args.sim_cutoff),
                "--out", str(validation_dir),
            ]
            rc = run(cmd, timeout=600)
            if rc != 0:
                print(f"[validate FAILED] rc={rc}", file=sys.stderr)
                return 1
        else:
            print(f"[Skip] Step 5 (UniProt download failed)", flush=True)
    else:
        if args.no_uniprot:
            print(f"\n[Skip] UniProt ground truth (--no-uniprot)")
        elif not taxid:
            print(f"\n[Skip] UniProt ground truth (no OX= in FASTA header)")

    # ── Final summary ──────────────────────────────────────────────────
    step("DONE - summary")
    print(f"Species         : {species}")
    print(f"v3 final_report : {final_csv}")
    print(f"BLAST output    : {blast_tsv}")
    print(f"v3 vs BLAST dir : {out_dir}")
    if (out_dir / "validation").exists():
        print(f"Validation dir  : {out_dir / 'validation'}")
        for f in (out_dir / "validation").iterdir():
            print(f"  - {f.name}")
    print(f"\nKey files to inspect:")
    print(f"  {out_dir / 'blast_stats.txt'}                          (category counts)")
    print(f"  {out_dir / 'v3_vs_blast.csv'}                          (per-row)")
    print(f"  {out_dir / 'validation' / 'v3_only_validation.png'}   (2x2 figure)")
    print(f"  {out_dir / 'validation' / 'v3_only_detail.csv'}        (per-row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())