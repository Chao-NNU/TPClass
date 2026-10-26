"""ESM-2 vs BLAST comparison.

Compares TPClass (ESM-2 cosine similarity) and DIAMOND blastp (sequence
identity + query coverage) on a set of candidate proteins to identify:

- TPClass finds + BLAST finds (both methods agree);
- TPClass finds + BLAST misses (TPClass-only candidates);
- BLAST finds + TPClass misses (TPClass false negatives).

Two details matter for the "BLAST missed" claim and are handled explicitly
here:

* **Query coverage is measured against the full query length**, not against the
  span of the alignment.  Using the alignment span makes the coverage of a
  short local hit indistinguishable from the coverage of a full-length hit.
* The ``coverage_low`` threshold is actually applied when deciding whether
  BLAST "found" a protein.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd

from tpclass.core.embeddings import read_fasta_ordered

logger = logging.getLogger(__name__)

#: DIAMOND output columns used by this module.
BLAST_COLUMNS = [
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
]


def query_coverage(qstart: int, qend: int, query_length: int) -> float:
    """Fraction of the **full query** covered by an alignment.

    The denominator is the length of the query sequence, not the span of the
    alignment: a local hit covering 100 residues of a 1,000-residue protein
    has a coverage of 0.1, not 1.0.

    Parameters
    ----------
    qstart, qend : int
        Alignment bounds on the query (1-based, inclusive).
    query_length : int
        Length of the query sequence.

    Returns
    -------
    float
        Coverage in ``[0, 1]``, or ``nan`` when the query length is unknown.
    """
    if query_length <= 0:
        return float("nan")
    span = max(int(qend) - int(qstart) + 1, 1)
    return span / query_length


def classify_hits(
    identity: pd.Series,
    coverage: pd.Series,
    hit_evalue: pd.Series,
    id_low: float,
    coverage_low: float,
    evalue_max: float,
) -> pd.Series:
    """Decide whether a BLAST hit counts as a hit.

    A hit passes when the sequence identity, the query coverage **and** the
    e-value all satisfy their thresholds.
    """
    ident = pd.to_numeric(identity, errors="coerce")
    cov = pd.to_numeric(coverage, errors="coerce")
    ev = pd.to_numeric(hit_evalue, errors="coerce")
    return (
        ident.notna()
        & (ident >= id_low)
        & cov.notna()
        & (cov >= coverage_low)
        & ev.notna()
        & (ev <= evalue_max)
    )


def compare_esm_vs_blast(
    candidates: str | Path,
    fasta: str | Path,
    ref_fasta: str | Path,
    out_tsv: str | Path,
    esm_high: float = 0.95,
    id_low: float = 0.30,
    coverage_low: float = 0.50,
    evalue: float = 1e-3,
    diamond: str = "diamond",
    threads: int = 4,
    max_target_seqs: int = 1,
) -> pd.DataFrame:
    """Compare TPClass (ESM-2 similarity) vs DIAMOND BLASTp on candidate proteins.

    Parameters
    ----------
    candidates : str or Path
        CSV of TPClass candidates.  Must contain a ``query`` column; the
        ``max_ref_sim`` column is used as the ESM-2 similarity when present.
    fasta : str or Path
        FASTA holding the candidate sequences (used for BLAST and for the
        coverage denominator).
    ref_fasta : str or Path
        Reference FASTA used as the BLAST database.
    out_tsv : str or Path
        Output TSV path.
    esm_high : float
        TPClass-found similarity threshold.
    id_low : float
        Minimum sequence identity (0-1) for a BLAST hit to count as a hit.
    coverage_low : float
        Minimum fraction of the **full query** covered by the alignment for a
        BLAST hit to count as a hit.
    evalue : float
        Maximum e-value of a BLAST hit.
    diamond : str
        Path to the DIAMOND binary.
    threads : int
        Number of DIAMOND threads.
    max_target_seqs : int
        Number of reference hits DIAMOND reports per query.

        The published protocol uses ``--max-target-seqs 1``, i.e. only the
        single best hit is considered.  Note that in that mode a query whose
        best hit fails the identity/coverage thresholds is reported as
        "BLAST missed" even when a weaker-but-passing hit exists.  Raise this
        value (e.g. 50) when screening for missed calls.

    Returns
    -------
    pd.DataFrame
        One row per candidate with the ESM-2 and BLAST evidence plus the
        ``tpclass_found`` / ``blast_found`` / ``blast_missed`` flags.
    """
    candidates = Path(candidates)
    fasta = Path(fasta)
    ref_fasta = Path(ref_fasta)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading candidates from {candidates}")
    cand = pd.read_csv(candidates, dtype=str)
    n_cand = len(cand)
    logger.info(f"  {n_cand} candidates")

    # ---- candidate sequences and their lengths (coverage denominator) -------
    logger.info(f"Reading sequences from {fasta}")
    _names, sequences = read_fasta_ordered(fasta)
    query_lengths: Dict[str, int] = {k: len(v) for k, v in sequences.items()}

    target_accs = set(cand["query"].tolist())
    tmp_top_fasta = Path(tempfile.gettempdir()) / f"top_candidates_{out_tsv.stem}.fasta"
    written = 0
    with open(tmp_top_fasta, "w") as f_out:
        for acc in target_accs:
            seq = sequences.get(acc)
            if seq is None:
                continue
            f_out.write(f">{acc}\n{seq}\n")
            written += 1
    logger.info(f"  wrote {written}/{n_cand} candidate sequences to {tmp_top_fasta}")
    if written == 0:
        raise ValueError(
            f"None of the {n_cand} candidate accessions were found in {fasta}; "
            "check that the two files use the same accession identifiers."
        )

    # ---- DIAMOND database ---------------------------------------------------
    tmp_db = Path(tempfile.gettempdir()) / f"ref_db_{out_tsv.stem}"
    cmd_make = [diamond, "makedb", "--in", str(ref_fasta), "--db", str(tmp_db)]
    logger.info(f"  $ {' '.join(cmd_make)}")
    r = subprocess.run(cmd_make, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"diamond makedb failed: {r.stderr}")

    # ---- BLASTp -------------------------------------------------------------
    blast_out = Path(tempfile.gettempdir()) / f"blast_out_{out_tsv.stem}.tsv"
    cmd_blast = [
        diamond,
        "blastp",
        "--query",
        str(tmp_top_fasta),
        "--db",
        str(tmp_db),
        "--out",
        str(blast_out),
        "--outfmt",
        "6",
        *BLAST_COLUMNS,
        "--max-target-seqs",
        str(max_target_seqs),
        "--threads",
        str(threads),
        "--evalue",
        str(evalue),
    ]
    logger.info(f"  $ {' '.join(cmd_blast)}")
    if max_target_seqs == 1:
        logger.warning(
            "max_target_seqs=1 keeps only the single best DIAMOND hit; a query whose "
            "best hit fails the identity/coverage thresholds is counted as BLAST-missed "
            "even if a weaker hit would have passed. Use a larger value for screening."
        )
    r = subprocess.run(cmd_blast, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"diamond blastp failed: {r.stderr}")

    # ---- parse hits ---------------------------------------------------------
    blast_hits: Dict[str, dict] = {}
    if blast_out.exists() and blast_out.stat().st_size > 0:
        df_blast = pd.read_csv(
            blast_out, sep="\t", header=None, names=BLAST_COLUMNS, dtype=str
        )
        for _, hit in df_blast.iterrows():
            qid = str(hit["qseqid"])
            q_len = query_lengths.get(qid, 0)
            if q_len <= 0:
                logger.warning(f"No length known for query {qid}; coverage set to NaN")
            # Coverage of the FULL query, not of the alignment span.
            qcov = query_coverage(int(hit["qstart"]), int(hit["qend"]), q_len)
            current = blast_hits.get(qid)
            if current is None or float(hit["bitscore"]) > current["blast_bitscore"]:
                blast_hits[qid] = {
                    "blast_ref": str(hit["sseqid"]),
                    "blast_identity": float(hit["pident"]) / 100.0,
                    "blast_evalue": float(hit["evalue"]),
                    "blast_coverage": qcov,
                    "blast_bitscore": float(hit["bitscore"]),
                }
    logger.info(f"  parsed {len(blast_hits)} blast hits")

    # ---- comparison table ---------------------------------------------------
    rows = []
    for i, row in cand.iterrows():
        acc = row["query"]
        ref_acc = row.get("best_ref_acc", "")
        esm_sim = float(row.get("max_ref_sim", 0) or 0)
        bh = blast_hits.get(acc, {})
        rows.append(
            {
                "rank": i + 1,
                "accession": acc,
                "best_ref_acc": ref_acc,
                "best_ref_gene": row.get("best_ref_gene", ""),
                "esm_sim": esm_sim,
                "blast_ref": bh.get("blast_ref", "(no hit)"),
                "blast_identity": bh.get("blast_identity"),
                "blast_evalue": bh.get("blast_evalue"),
                "blast_coverage": bh.get("blast_coverage"),
            }
        )
    df = pd.DataFrame(rows)

    # ---- classification -----------------------------------------------------
    df["tpclass_found"] = df["esm_sim"] >= esm_high
    df["blast_found"] = classify_hits(
        df["blast_identity"],
        df["blast_coverage"],
        df["blast_evalue"],
        id_low=id_low,
        coverage_low=coverage_low,
        evalue_max=evalue,
    )
    df["blast_missed"] = df["tpclass_found"] & ~df["blast_found"]

    df.to_csv(out_tsv, sep="\t", index=False)
    logger.info(f"Saved {out_tsv} ({len(df)} rows)")

    n_total = len(df)
    n_tpclass = int(df["tpclass_found"].sum())
    n_blast = int(df["blast_found"].sum())
    n_missed = int(df["blast_missed"].sum())
    logger.info("=" * 80)
    logger.info(
        f"SUMMARY: TPClass (ESM-2 sim >= {esm_high}) vs BLAST "
        f"(identity >= {id_low}, query coverage >= {coverage_low}, e-value <= {evalue})"
    )
    logger.info("=" * 80)
    logger.info(f"Total candidates:                       {n_total}")
    logger.info(f"TPClass finds:                          {n_tpclass} ({100*n_tpclass/n_total:.1f}%)")
    logger.info(f"BLAST finds:                            {n_blast} ({100*n_blast/n_total:.1f}%)")
    logger.info(f"BLAST-missed (TPClass YES, BLAST NO):  {n_missed} ({100*n_missed/n_total:.1f}%)")

    return df
