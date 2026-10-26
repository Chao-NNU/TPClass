#!/usr/bin/env python3
"""
sweep_thresholds.py — 在一组 threshold 上跑 v3 + 对比 BLAST，收集 PR 曲线。

每个 threshold = 一次 v3 + 一次 parse_blast_to_tcdb，结果写到：
  {out_dir}/per_threshold/thresh_X.XX/final_report.csv   (v3 输出)
  {out_dir}/per_threshold/thresh_X.XX/sweep_summary.txt  (本 threshold 的数字)
  {out_dir}/pr_curve.csv                                  (所有 threshold 汇总)
  {out_dir}/pr_curve.png                                  (PR 折线图)

streaming 模式：v3 的 print 实时流到当前终端，不会卡在 "无输出" 状态。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import subprocess


# 5 个 threshold 足够画 PR 曲线；想更细就 --thresholds 0.2 0.3 0.4 0.5 0.6 0.7 0.8
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# s4pred 缓存默认位置 —— 多个 threshold 共享，避免重复跑 s4pred (25k → 5k 次)
DEFAULT_S4PRED_CACHE = "/data2/chao/transport/cache/ss_results"


def run_v3(v3_path: str, fasta: str, out_dir: str, threshold: float,
           s4pred_cache: str | None = None):
    """跑一次 v3 在指定 threshold 下；流式输出，实时显示进度。
    s4pred_cache: 固定缓存目录（--s4pred_outdir 给 v3），所有 threshold 复用同一份 .horiz
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, v3_path,
        "-in", fasta,
        "-out", str(out_dir_p),
        "--threshold", str(threshold),
        "--no_plot",  # sweep 时不画图，加速
    ]
    if s4pred_cache:
        cmd += ["--s4pred_outdir", s4pred_cache]
    print(f"\n========== threshold = {threshold} ==========", flush=True)
    print(f"[v3] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    # 关键：不传 capture_output，让 v3 的 stdout/stderr 继承到当前终端
    res = subprocess.run(cmd, text=True, timeout=7200)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[v3] FAILED (exit {res.returncode}) after {dt:.1f}s", flush=True)
        return None
    print(f"[v3] OK in {dt:.1f}s -> {out_dir_p / 'final_report.csv'}", flush=True)
    return out_dir_p / "final_report.csv"


def run_compare(compare_path: str, blast: str, annot: str, v3_csv: Path, out_dir: str):
    """跑 parse_blast_to_tcdb，提取 PR 数字；流式输出"""
    cmd = [
        sys.executable, compare_path,
        "--blast", blast,
        "--annot", annot,
        "--v3", str(v3_csv),
        "--out", out_dir,
    ]
    print(f"[compare] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, text=True, timeout=600)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[compare] FAILED after {dt:.1f}s", flush=True)
        return None
    print(f"[compare] OK in {dt:.1f}s", flush=True)
    return Path(out_dir)


def parse_compare_summary(compare_out: Path) -> dict | None:
    """从 parse_blast_to_tcdb 输出的 blast_stats.txt 里抠出 match 分类并计算 PR。

    把 BLAST 当 ground truth:
        TP = yes  (v3 跟 BLAST 同 TC)
        FP = v3_only  (v3 有 TC，BLAST 没 hit)
        FN = blast_only  (BLAST 有 TC，v3 NO)
        TN = neither  (都 NO)
    """
    stats_path = compare_out / "blast_stats.txt"
    if not stats_path.exists():
        return None
    text = stats_path.read_text()

    # 抓 "  <key>            :    <count>" 这一段
    counts = {"yes": 0, "blast_only": 0, "v3_only": 0, "neither": 0}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith(tuple(counts.keys())):
            continue
        if ":" not in s:
            continue
        key, _, val = s.partition(":")
        key = key.strip()
        if key in counts:
            try:
                counts[key] = int(val.strip())
            except ValueError:
                pass

    if not any(counts.values()) and "yes" not in text:
        return None

    tp = counts["yes"]
    fp = counts["v3_only"]
    fn = counts["blast_only"]
    tn = counts["neither"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    # v3's "discovery rate" = how many v3 YES are beyond BLAST's reach
    n_v3_yes = tp + fp
    discovery = fp / max(1, n_v3_yes)

    return {
        "threshold_dir": str(compare_out),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_v3_yes": n_v3_yes,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "discovery_rate": round(discovery, 4),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Sweep v3 thresholds, collect PR curve vs BLAST baseline")
    ap.add_argument("--v3", required=True, help="path to predict_transport_family_v3.py")
    ap.add_argument("--fasta", required=True, help="input FASTA")
    ap.add_argument("--blast", required=True, help="BLAST -outfmt 6 result")
    ap.add_argument("--annot", required=True, help="TCDB annotation tsv")
    ap.add_argument("--compare", required=True, help="path to parse_blast_to_tcdb.py")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS,
                    help=f"threshold list (default: {' '.join(map(str, DEFAULT_THRESHOLDS))})")
    ap.add_argument("--s4pred_outdir", default=DEFAULT_S4PRED_CACHE,
                    help=f"shared s4pred .horiz cache dir across all thresholds "
                         f"(default: {DEFAULT_S4PRED_CACHE}). Set to '' to disable caching.")
    args = ap.parse_args()

    # "" means user wants to disable s4pred caching
    s4pred_cache = args.s4pred_outdir if args.s4pred_outdir else None

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # 先数 BLAST 里的 query 数 (informational)
    try:
        with open(args.blast) as f:
            n_blast = sum(1 for _ in f)
    except OSError:
        n_blast = -1
    print(f"[sweep] {n_blast} BLAST hits, will sweep {len(args.thresholds)} thresholds: "
          f"{args.thresholds}", flush=True)
    print(f"[sweep] output -> {out_root}", flush=True)
    print(f"[sweep] s4pred cache -> {s4pred_cache or '(disabled, will re-run each threshold)'}",
          flush=True)

    rows = []
    for th in args.thresholds:
        th_dir = out_root / "per_threshold" / f"thresh_{th:.2f}"
        v3_csv = run_v3(args.v3, args.fasta, str(th_dir), th, s4pred_cache=s4pred_cache)
        if v3_csv is None or not v3_csv.exists():
            print(f"[sweep] skip thresh={th} (v3 failed)", flush=True)
            continue
        compare_out = run_compare(args.compare, args.blast, args.annot, v3_csv, str(th_dir))
        if compare_out is None:
            print(f"[sweep] skip thresh={th} (compare failed)", flush=True)
            continue
        stats = parse_compare_summary(compare_out)
        if stats is None:
            print(f"[sweep] WARN: no blast_stats.txt in {compare_out}", flush=True)
            continue
        stats["threshold"] = th
        rows.append(stats)
        print(f"[sweep] thresh={th}: TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} "
              f"| P={stats['precision']:.3f} R={stats['recall']:.3f} "
              f"F1={stats['F1']:.3f} | v3_YES={stats['n_v3_yes']} "
              f"discovery={stats['discovery_rate']:.1%}", flush=True)

    if not rows:
        print("[sweep] NO successful thresholds — nothing to plot", flush=True)
        return 1

    # 写汇总 CSV
    csv_path = out_root / "pr_curve.csv"
    keys = ["threshold", "precision", "recall", "F1", "accuracy",
            "tp", "fp", "fn", "tn", "n_v3_yes", "discovery_rate"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n[sweep] saved -> {csv_path}", flush=True)

    # 画 PR 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = [r["threshold"] for r in rows]
        ps = [r.get("precision", 0) for r in rows]
        rs = [r.get("recall", 0) for r in rows]
        fs = [r.get("F1", 0) for r in rows]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(rs, ps, "o-", color="#2c7fb8", linewidth=2, markersize=8, label="Precision-Recall")
        for t, p, r in zip(ts, ps, rs):
            ax.annotate(f"{t:.2f}", (r, p), textcoords="offset points",
                        xytext=(5, 5), fontsize=9)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"v3 PR curve vs BLAST ({len(rows)} thresholds)")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        png = out_root / "pr_curve.png"
        fig.savefig(png, dpi=150)
        print(f"[sweep] saved -> {png}", flush=True)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(ts, ps, "o-", label="Precision")
        ax2.plot(ts, rs, "s-", label="Recall")
        ax2.plot(ts, fs, "^-", label="F1")
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Score")
        ax2.set_title("P/R/F1 vs threshold")
        ax2.set_ylim(-0.02, 1.02); ax2.grid(alpha=0.3); ax2.legend()
        fig2.tight_layout()
        png2 = out_root / "metrics_vs_threshold.png"
        fig2.savefig(png2, dpi=150)
        print(f"[sweep] saved -> {png2}", flush=True)
    except ImportError:
        print("[sweep] matplotlib not available, skipped plots", flush=True)

    print(f"\n[sweep] DONE. {len(rows)} thresholds summarized.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
