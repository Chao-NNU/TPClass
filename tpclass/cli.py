"""Command-line interface for TPClass."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  predict
# ══════════════════════════════════════════════════════════════════════════
def ensure_parent_dir(path) -> Path:
    """Create the parent directory of an output path if it is missing.

    The documented examples write to ``results/candidates.csv``; without this
    the write fails on a fresh checkout because ``results/`` does not exist.
    """
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    return Path(path)


def parse_phase1_weights(value):
    """Parse ``"mlp,xgb"`` into a weight pair; ``None`` keeps the defaults."""
    if not value:
        return None
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"--phase1_weights must be 'mlp_weight,xgb_weight', got {value!r}"
        )
    try:
        w_mlp, w_xgb = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"--phase1_weights must be numeric, got {value!r}"
        ) from exc
    if w_mlp < 0 or w_xgb < 0:
        raise ValueError(f"--phase1_weights must be non-negative, got {value!r}")
    return (w_mlp, w_xgb)


def cmd_predict(args):
    """Predict transporters in a FASTA file."""
    from tpclass.core.predictor import TPClassPredictor

    predictor = TPClassPredictor(
        phase1_3b_checkpoint=args.phase1_3b,
        phase1_650m_checkpoint=args.phase1_650m,
        phase1_xgb_3b=args.phase1_xgb_3b,
        phase1_xgb_650m=args.phase1_xgb_650m,
        phase2_3b_checkpoint=args.phase2_3b,
        phase2_650m_checkpoint=args.phase2_650m,
        esm2_3b_path=args.esm2_3b,
        esm2_650m_path=args.esm2_650m,
        esm2_phase2_3b=args.esm2_phase2_3b,
        esm2_phase2_650m=args.esm2_phase2_650m,
        ref_embeddings_3b=args.ref_emb_3b,
        ref_embeddings_650m=args.ref_emb_650m,
        ref_metadata=args.ref_metadata,
        tcdb_annot=args.tcdb_annot,
        s4pred_cache_dir=args.s4pred_cache,
        phase1_weights=parse_phase1_weights(args.phase1_weights),
        device=args.device,
    )
    df = predictor.predict(
        fasta=args.fasta,
        backbone=args.backbone,
        threshold=args.threshold,
        run_phase2=not args.no_phase2,
        top_k=args.top_k,
    )
    df.to_csv(ensure_parent_dir(args.out), index=False)
    n_yes = (df["is_transport"] == "YES").sum()
    print(
        f"Predicted {n_yes}/{len(df)} as transporters "
        f"(threshold={args.threshold}, backbone={args.backbone})"
    )
    print(f"Saved to {args.out}")


# ══════════════════════════════════════════════════════════════════════════
#  filter / three-layer
# ══════════════════════════════════════════════════════════════════════════
def _three_layer_filter(args, ref_embeddings=None, ref_set=None):
    from tpclass.filters.three_layer import ThreeLayerFilter

    tc_classes = args.tc_classes.split(",") if args.tc_classes else []
    keywords = args.keywords.split(",") if args.keywords else []
    flt = ThreeLayerFilter(
        ref_embeddings=ref_embeddings,
        ref_set=ref_set,
        esm2_path=getattr(args, "esm2", None),
        tc_classes=tc_classes,
        keywords=keywords,
        min_ref_sim=args.sim_min,
        device=getattr(args, "device", None) or "cuda",
    )
    df = flt.filter(args.v3_csv, query_embeddings=getattr(args, "query_embeddings", None))
    df.to_csv(ensure_parent_dir(args.out), index=False)
    print(f"After three-layer filter: {len(df)} candidates")
    print(f"Saved to {args.out}")


def cmd_filter(args):
    """Apply the substrate-specific three-layer filter."""
    _three_layer_filter(args, ref_set=args.ref_set)


def cmd_three_layer(args):
    """Apply the three-layer evidence filter with pre-computed embeddings."""
    _three_layer_filter(args, ref_embeddings=args.ref_embeddings)


# ══════════════════════════════════════════════════════════════════════════
#  compare-esm-blast
# ══════════════════════════════════════════════════════════════════════════
def cmd_compare(args):
    """Compare ESM-2 (TPClass) vs BLAST on candidates."""
    from tpclass.analysis.esm_vs_blast import compare_esm_vs_blast

    df = compare_esm_vs_blast(
        candidates=args.candidates,
        fasta=args.fasta,
        ref_fasta=args.ref_fasta,
        out_tsv=args.out,
        esm_high=args.esm_high,
        id_low=args.id_low,
        coverage_low=args.coverage_low,
        evalue=args.evalue,
        diamond=args.diamond,
        threads=args.threads,
        max_target_seqs=args.max_target_seqs,
    )
    print(f"Saved {args.out}")


# ══════════════════════════════════════════════════════════════════════════
#  compare-reports
# ══════════════════════════════════════════════════════════════════════════
def cmd_compare_reports(args):
    """Compare two prediction tables row by row."""
    from tpclass.analysis.compare_reports import compare_reports, format_report

    diff = compare_reports(
        a=args.a,
        b=args.b,
        key=args.key,
        tol=args.tol,
        show=args.show,
    )
    print(format_report(diff))
    if not diff.identical:
        raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════
#  paired-test
# ══════════════════════════════════════════════════════════════════════════
def cmd_paired_test(args):
    """Paired significance tests between two configurations."""
    import pandas as pd

    from tpclass.analysis.paired_tests import compare_config_groups

    df = pd.read_csv(args.preds)
    if args.pairs:
        pairs = []
        for spec in args.pairs:
            parts = [p.strip() for p in spec.split(",")]
            if len(parts) != 2:
                raise SystemExit(f"--pairs entry must be 'A,B', got {spec!r}")
            pairs.append((parts[0], parts[1], f"{parts[0]} vs {parts[1]}"))
    else:
        pairs = [
            ("3B_cached", "650M_cached", "3B vs 650M (cached)"),
            ("3B_fresh", "650M_fresh", "3B vs 650M (fresh)"),
            ("650M_fresh", "650M_cached", "650M cached -> fresh"),
            ("3B_fresh", "3B_cached", "3B cached -> fresh"),
        ]

    rows = compare_config_groups(df, pairs, n_boot=args.n_boot, seed=args.seed)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    for r in rows:
        print(
            f"{r['comparison']}: n={r['n']}  McNemar(correctness) "
            f"n01={r['mcnemar_n01']} n10={r['mcnemar_n10']} p={r['mcnemar_p']:.3e}  "
            f"| label-variant p={r['mcnemar_label_variant_p']:.3e}  "
            f"Jaccard={r['pos_jaccard']:.4f}"
        )
    print(f"Saved {args.out}")


# ══════════════════════════════════════════════════════════════════════════
#  family-metrics
# ══════════════════════════════════════════════════════════════════════════
def cmd_family_metrics(args):
    """Family-level precision / fuzzy recall / strict recall."""
    from tpclass.analysis.family_metrics import evaluate_from_files

    metrics = evaluate_from_files(
        pred_csv=args.pred_csv,
        gt_tsv=args.gt_tsv,
        out_csv=args.out,
        fuzzy_rule=args.fuzzy_rule,
        query_col=args.query_col,
        family_col=args.family_col,
        is_yes_col=args.is_yes_col,
        gt_acc_col=args.gt_acc_col,
        gt_family_col=args.gt_family_col,
        name=args.name,
    )
    row = metrics.as_row()
    print(f"{metrics.name}  (fuzzy rule: {metrics.fuzzy_rule})")
    print(f"  Phase 1 YES with TC:        {metrics.n_pred_yes_with_family}")
    print(f"  Matched GT with a family:    {metrics.n_matched}")
    print(f"  Predicted YES inside GT:     {metrics.n_yes_in_gt}")
    print(f"  Correct (4-digit match):    {metrics.n_correct}")
    print(f"  GT size (4-digit):           {metrics.n_gt}")
    print(f"  Precision (4-digit TC):      {metrics.precision_4digit:.4f}")
    print(f"    with_family denominator:    {metrics.precision_with_family:.4f}")
    print(f"    all_yes denominator:        {metrics.precision_all_yes:.4f}")
    print(f"  Coverage (TC assign):        {metrics.coverage_4digit:.4f}")
    print(f"  Recall fuzzy (selected):     {metrics.recall_fuzzy:.4f}")
    print(f"  Recall fuzzy (with_family):  {row['recall_fuzzy_matched']:.4f}")
    print(f"  Recall fuzzy (all_yes):      {row['recall_fuzzy_all_yes']:.4f}")
    print(f"  Recall strict:               {metrics.recall_strict:.4f}")
    if args.out:
        print(f"Saved {args.out}")


# ══════════════════════════════════════════════════════════════════════════
#  compute-refs
# ══════════════════════════════════════════════════════════════════════════
def cmd_compute_refs(args):
    """Compute reference embeddings from a FASTA."""
    from tpclass.core.refs import compute_reference_embeddings

    compute_reference_embeddings(
        fasta=args.fasta,
        esm2_path=args.esm2,
        out_npz=args.out,
        device=args.device or "cuda",
        max_len=args.max_len,
        batch_size=args.batch_size,
    )
    print(f"Saved {args.out}")


# ══════════════════════════════════════════════════════════════════════════
#  split / train
# ══════════════════════════════════════════════════════════════════════════
def cmd_split(args):
    """Split a TCDB FASTA into train/val/test."""
    from tpclass.train.dataset import load_tcdb_records
    from tpclass.train.split import (
        cluster_split,
        stratified_split,
        write_fasta,
        write_split_csv,
    )

    records = load_tcdb_records(args.input, negatives=args.negatives)
    if args.strategy == "cluster":
        splits = cluster_split(
            records,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            threshold=args.cluster_threshold,
        )
    else:
        splits = stratified_split(
            records,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        write_fasta(items, out_dir / f"{name}.fasta")
    write_split_csv(splits, out_dir / "split.csv")
    print(f"Saved splits to {out_dir}")


def cmd_train_phase1(args):
    """Train the Phase 1 MLP and XGBoost models."""
    from tpclass.train.phase1 import fit_isotonic, train_mlp, train_xgb

    data_dir = Path(args.data_dir)
    x_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "Y_train.npy")
    x_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "Y_val.npy")

    mlp_path = Path(args.out_dir) / "transport_classifier_best.pth"
    train_mlp(
        x_train, y_train, x_val, y_val, mlp_path,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    print(f"MLP -> {mlp_path}")

    if not args.skip_xgb:
        xgb_path = Path(args.out_dir) / "transport_xgb_best.json"
        train_xgb(x_train, y_train, x_val, y_val, xgb_path)
        print(f"XGBoost -> {xgb_path}")

    if args.calibrate:
        from tpclass.core.phase1 import ENSEMBLE_WEIGHTS

        import torch

        from tpclass.core.phase1 import TransportClassifier

        w_mlp, w_xgb = ENSEMBLE_WEIGHTS[args.backbone]
        model = TransportClassifier(input_dim=x_train.shape[1])
        model.load_state_dict(torch.load(mlp_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            mlp_proba = torch.sigmoid(model(torch.from_numpy(x_val).float())).numpy().flatten()
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(Path(args.out_dir) / "transport_xgb_best.json"))
        xgb_proba = booster.predict(xgb.DMatrix(x_val))
        raw = w_mlp * mlp_proba + w_xgb * xgb_proba
        calib_path = Path(args.out_dir) / "isotonic_calibrator.joblib"
        fit_isotonic(raw, y_val, calib_path)
        print(f"Calibrator -> {calib_path}")


def cmd_train_phase2(args):
    """Train the Phase 2 contrastive head."""
    import pandas as pd

    from tpclass.train.phase2 import train_contrastive

    df = pd.read_csv(args.data_csv)
    df = df.dropna(subset=["sequence", "tc_number"])
    sequences = df["sequence"].astype(str).tolist()
    families = df["tc_number"].astype(str).map(lambda t: ".".join(t.split(".")[:3])).tolist()
    path = train_contrastive(
        sequences=sequences,
        families=families,
        esm2_path=args.esm2,
        output_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device or "cuda",
        seed=args.seed,
    )
    print(f"Phase 2 checkpoint -> {path}")


# ══════════════════════════════════════════════════════════════════════════
#  parser
# ══════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tpclass",
        description="TPClass: A Two-Phase Predictor for Membrane Transporter Annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- predict -----------------------------------------------------------
    p = sub.add_parser("predict", help="Predict transporters in a FASTA file")
    p.add_argument("--fasta", required=True)
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--backbone", choices=["3B", "650M", "cascade"], default="3B")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument(
        "--phase1_weights",
        default=None,
        help='Phase 1 ensemble weights "mlp_w,xgb_w". Default: the per-backbone '
        "values (3B 0.8/0.2, 650M 0.4/0.6). The released script defaults to "
        '"0.8,0.2" for every backbone; pass it explicitly to reproduce a '
        "specific published configuration.",
    )
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--no_phase2", action="store_true")
    p.add_argument("--phase1_3b", help="3B Phase 1 MLP checkpoint (.pth)")
    p.add_argument("--phase1_650m", help="650M Phase 1 MLP checkpoint (.pth)")
    p.add_argument("--phase1_xgb_3b", help="3B Phase 1 XGBoost model (.json)")
    p.add_argument("--phase1_xgb_650m", help="650M Phase 1 XGBoost model (.json)")
    p.add_argument("--phase2_3b", help="3B Phase 2 contrastive checkpoint (.pth)")
    p.add_argument("--phase2_650m", help="650M Phase 2 contrastive checkpoint (.pth)")
    p.add_argument("--esm2_3b", help="ESM-2 3B checkpoint (.pt or HuggingFace directory)")
    p.add_argument("--esm2_650m", help="ESM-2 650M checkpoint (.pt or HuggingFace directory)")
    p.add_argument(
        "--esm2_phase2_3b",
        help="ESM-2 backbone of the 3B Phase 2 head (defaults to --esm2_3b)",
    )
    p.add_argument(
        "--esm2_phase2_650m",
        help="ESM-2 backbone of the 650M Phase 2 head (defaults to --esm2_650m)",
    )
    p.add_argument(
        "--ref_emb_3b",
        help="3B Phase 2 reference index (.npy); defaults to tcdb_embeddings.npy next to the checkpoint",
    )
    p.add_argument(
        "--ref_emb_650m",
        help="650M Phase 2 reference index (.npy); defaults to tcdb_embeddings.npy next to the checkpoint",
    )
    p.add_argument(
        "--ref_metadata",
        help="Reference metadata CSV; defaults to tcdb_index_metadata.csv next to the checkpoint",
    )
    p.add_argument(
        "--tcdb_annot",
        help=(
            "TCDB annotation table (tc_number -> description); when given, the TC "
            "description is looked up by TC number as the released pipeline does"
        ),
    )
    p.add_argument(
        "--s4pred_cache",
        help=(
            "Directory of cached s4pred .horiz files; the cached secondary "
            "structure is reused, so the features match the released run exactly"
        ),
    )
    p.add_argument("--device", help="Torch device (cuda / cpu)")
    p.set_defaults(func=cmd_predict)

    # ---- filter ------------------------------------------------------------
    p = sub.add_parser(
        "filter", help="Substrate-specific filter (TC class + keywords + reference similarity)"
    )
    p.add_argument("--v3_csv", required=True, help="Prediction table to filter")
    p.add_argument("--out", required=True)
    p.add_argument("--ref_set", required=True, help="Reference FASTA of known substrate transporters")
    p.add_argument("--esm2", help="ESM-2 checkpoint used to embed the reference set")
    p.add_argument("--tc_classes", help="Comma-separated TC class prefixes")
    p.add_argument("--keywords", help="Comma-separated description keywords")
    p.add_argument("--sim_min", type=float, default=0.5, help="Minimum reference cosine similarity")
    p.add_argument("--query_embeddings", help="Optional .npz of query embeddings")
    p.add_argument("--device", help="Torch device (cuda / cpu)")
    p.set_defaults(func=cmd_filter)

    # ---- three-layer -------------------------------------------------------
    p = sub.add_parser(
        "three-layer", help="Three-layer filter with pre-computed reference embeddings"
    )
    p.add_argument("--v3_csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ref_embeddings", required=True, help="Reference embeddings (.npz)")
    p.add_argument("--tc_classes")
    p.add_argument("--keywords")
    p.add_argument("--sim_min", type=float, default=0.5)
    p.add_argument("--query_embeddings", help="Optional .npz of query embeddings")
    p.set_defaults(func=cmd_three_layer)

    # ---- compare-esm-blast -------------------------------------------------
    p = sub.add_parser("compare-esm-blast", help="Compare ESM-2 (TPClass) vs BLAST on candidates")
    p.add_argument("--candidates", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--ref_fasta", required=True, help="Reference FASTA used as the BLAST database")
    p.add_argument("--out", required=True)
    p.add_argument("--esm_high", type=float, default=0.95)
    p.add_argument("--id_low", type=float, default=0.30, help="Minimum sequence identity (0-1)")
    p.add_argument(
        "--coverage_low",
        type=float,
        default=0.50,
        help="Minimum fraction of the full query covered by the alignment",
    )
    p.add_argument("--evalue", type=float, default=1e-3)
    p.add_argument(
        "--max_target_seqs",
        type=int,
        default=1,
        help="DIAMOND hits per query; 1 matches the published protocol, larger values "
        "avoid counting a query as missed when a weaker hit would have passed",
    )
    p.add_argument("--diamond", default="diamond")
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(func=cmd_compare)

    # ---- compare-reports ----------------------------------------------------
    p = sub.add_parser(
        "compare-reports",
        help="Compare two prediction tables row by row (package vs. released script)",
    )
    p.add_argument("--a", required=True, help="First CSV table")
    p.add_argument("--b", required=True, help="Second CSV table")
    p.add_argument("--key", default="query", help="Join column (default: query)")
    p.add_argument("--tol", type=float, default=1e-4, help="Numeric tolerance")
    p.add_argument("--show", type=int, default=5, help="Mismatching rows to print per column")
    p.set_defaults(func=cmd_compare_reports)

    # ---- paired-test --------------------------------------------------------
    p = sub.add_parser(
        "paired-test",
        help="Paired significance tests between two TPClass configurations",
    )
    p.add_argument("--preds", required=True, help="Long-format predictions CSV")
    p.add_argument("--out", default="paired_test.csv", help="Output CSV")
    p.add_argument(
        "--pairs",
        nargs="+",
        help='Comparisons as "A,B" (default: the four Table-1 configurations)',
    )
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_paired_test)

    # ---- family-metrics -----------------------------------------------------
    p = sub.add_parser(
        "family-metrics",
        help="Family-level (4-digit TC) precision / fuzzy recall / strict recall",
    )
    p.add_argument("--pred_csv", required=True, help="Prediction table (final_report.csv)")
    p.add_argument("--gt_tsv", required=True, help="Ground-truth TSV")
    p.add_argument("--out", help="Output summary CSV")
    p.add_argument(
        "--fuzzy_rule",
        choices=["with_family", "all_yes"],
        default="with_family",
        help=(
            "How fuzzy recall is counted: 'with_family' counts only predictions "
            "that received a TC family (the rule of eval_family_precision.py), "
            "'all_yes' counts every predicted positive (the rule of "
            "eval_hybrid_v3.py).  Both are always reported."
        ),
    )
    p.add_argument("--query_col", default="query")
    p.add_argument("--family_col", default="tc_number")
    p.add_argument("--is_yes_col", default="is_transport")
    p.add_argument("--gt_acc_col", default="accession")
    p.add_argument("--gt_family_col", default="tc_family_4digit")
    p.add_argument("--name", default="")
    p.set_defaults(func=cmd_family_metrics)

    # ---- compute-refs ------------------------------------------------------
    p = sub.add_parser("compute-refs", help="Compute mean-pooled ESM-2 embeddings for a FASTA")
    p.add_argument("--fasta", required=True)
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--esm2", required=True, help="ESM-2 checkpoint (.pt or HuggingFace directory)")
    p.add_argument("--device", help="Torch device (cuda / cpu)")
    p.add_argument("--max_len", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=8)
    p.set_defaults(func=cmd_compute_refs)

    # ---- split -------------------------------------------------------------
    p = sub.add_parser("split", help="Split a TCDB FASTA into train/val/test")
    p.add_argument("--input", required=True, help="TCDB protein FASTA")
    p.add_argument("--negatives", help="Optional FASTA of non-transporters")
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--strategy",
        choices=["stratified", "cluster"],
        default="cluster",
        help="'cluster' keeps homologous proteins in the same subset",
    )
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cluster_threshold", type=float, default=0.5)
    p.set_defaults(func=cmd_split)

    # ---- train-phase1 ------------------------------------------------------
    p = sub.add_parser("train-phase1", help="Train the Phase 1 MLP / XGBoost models")
    p.add_argument("--data_dir", required=True, help="Directory with X_*.npy and Y_*.npy")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone", choices=["3B", "650M"], default="650M")
    p.add_argument("--skip_xgb", action="store_true")
    p.add_argument("--calibrate", action="store_true", help="Also fit an isotonic calibrator")
    p.set_defaults(func=cmd_train_phase1)

    # ---- train-phase2 ------------------------------------------------------
    p = sub.add_parser("train-phase2", help="Train the Phase 2 contrastive head")
    p.add_argument("--data_csv", required=True, help="CSV with 'sequence' and 'tc_number' columns")
    p.add_argument("--esm2", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", help="Torch device (cuda / cpu)")
    p.set_defaults(func=cmd_train_phase2)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
