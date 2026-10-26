#!/usr/bin/env python
"""
eval_ablation_direct.py
=======================

Direct evaluation of v3 Phase 1 ablation variants on the balanced val set
(extracted by extract_features_for_retrain.py, 4785 sequences, 1:1 pos/neg).

This bypasses the v3 predict_transport_family_v3.py inference path because
that path filters out UniProt negatives (they have no TCDB entry). For
ablation we only need the Phase 1 binary output, not the full v3 pipeline.

Variants evaluated (5):
  - full_v3       : MLP × 0.4 + XGB × 0.6  (default ensemble)
  - no_s4pred     : MLP × 0.4 + XGB × 0.6, with s4pred 3-dim zeroed out
  - pure_mlp      : MLP × 1.0 + XGB × 0.0
  - pure_xgb      : MLP × 0.0 + XGB × 1.0
  - no_phase1     : Phase 1 disabled — fall back to Phase 2 ESM-2 similarity
                    (computed here from the val X[:, :1280] ESM-2 portion,
                    cosine to TCDB Phase 2 reference, threshold-tuned)

Outputs:
  - <out>.csv   per-variant metrics (Acc/BalAcc/Precision/Recall/F1/MCC + TP/TN/FP/FN)
  - <out>.md    markdown table (Bioinformatics style)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, precision_score, recall_score,
    confusion_matrix,
)

SCRIPTS_DIR = "/data2/chao/transport/scripts"
sys.path.insert(0, SCRIPTS_DIR)
import predict_transport_family_v3 as v3


def load_mlp(ckpt_path: str, device, input_dim: int = 1283):
    model = v3.TransportClassifier(input_dim=input_dim).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


def mlp_predict_proba(model, X, device, batch_size: int = 1024):
    probs = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.tensor(X[s:s + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy().flatten())
    return np.concatenate(probs)


def xgb_predict_proba(bst, X):
    return bst.predict(xgb.DMatrix(X))


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)
    return dict(TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
                accuracy=acc, balanced_accuracy=bacc,
                precision=prec, recall=rec, f1=f1, mcc=mcc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data2/chao/transport/data/features_v3_retrain")
    ap.add_argument("--mlp_ckpt", default="/data2/chao/transport/models/v3_retrained/phase1/transport_classifier_best.pth")
    ap.add_argument("--xgb_model", default="/data2/chao/transport/models/v3_retrained/phase1/transport_xgb_best.json")
    ap.add_argument("--phase2_emb", default="/data2/chao/transport/phase2_contrastive/tcdb_embeddings.npy")
    ap.add_argument("--phase2_meta", default="/data2/chao/transport/phase2_contrastive/tcdb_index_metadata.csv")
    ap.add_argument("--out", default="/data2/chao/transport/outputs/ablation/val_table_direct.csv")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    # ---- Load balanced val features ----
    X_val = np.load(os.path.join(args.data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(args.data_dir, "Y_val.npy"))
    print(f"[Load] X_val={X_val.shape}  Y_val={Y_val.shape}  pos={int(Y_val.sum())}  neg={int((1-Y_val).sum())}")

    # ---- Split features into ESM-2 (N-3) + s4pred (3) ----
    n_feat    = X_val.shape[1]
    n_esm2    = n_feat - 3  # last 3 dims are s4pred
    X_esm2    = X_val[:, :n_esm2]
    X_s4p     = X_val[:, n_esm2:n_feat]
    X_full    = X_val
    X_noss    = np.concatenate([X_esm2, np.zeros_like(X_s4p)], axis=1)
    print(f"[Feat] total={n_feat}  ESM-2: {X_esm2.shape}  s4pred: {X_s4p.shape}  s4pred-zeroed: {X_noss.shape}")

    # ---- Load models ----
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"[Device] {device}")

    # input_dim = total feature count (auto-inferred from X_val shape)
    input_dim = n_feat
    print(f"[Model] MLP input_dim = {input_dim}")
    mlp  = load_mlp(args.mlp_ckpt, device, input_dim=input_dim)
    bst  = xgb.Booster(); bst.load_model(args.xgb_model)

    # ---- Compute base probabilities ----
    print("\n[Predict] MLP (full features) ...")
    p_mlp_full = mlp_predict_proba(mlp, X_full, device)
    print("[Predict] MLP (s4pred zeroed) ...")
    p_mlp_noss = mlp_predict_proba(mlp, X_noss, device)

    print("[Predict] XGB (full features) ...")
    p_xgb_full = xgb_predict_proba(bst, X_full)
    print("[Predict] XGB (s4pred zeroed) ...")
    p_xgb_noss = xgb_predict_proba(bst, X_noss)

    # ---- Build 5 variant predictions ----
    def ens(p_m, p_x, w_m, w_x):
        return w_m * p_m + w_x * p_x

    rows = []
    # 1) Anchor variants (single rows in Table 1)
    variants = [
        # (name, mlp_probs, xgb_probs, w_mlp, w_xgb)
        # Full 2x2x2 ablation: s4pred {on, off} x classifier {MLP+XGB, MLP, XGB}
        ("full_v3",        p_mlp_full, p_xgb_full, 0.4, 0.6),
        ("no_s4pred",      p_mlp_noss, p_xgb_noss, 0.4, 0.6),
        ("pure_mlp",       p_mlp_full, p_xgb_full, 1.0, 0.0),
        ("mlp_no_s4pred",  p_mlp_noss, p_xgb_noss, 1.0, 0.0),
        ("pure_xgb",       p_mlp_full, p_xgb_full, 0.0, 1.0),
        ("xgb_no_s4pred",  p_mlp_noss, p_xgb_noss, 0.0, 1.0),
    ]

    for name, pm, px, wm, wx in variants:
        prob = ens(pm, px, wm, wx)
        pred = (prob > 0.5).astype(int)
        m = compute_metrics(Y_val, pred)
        m["variant"] = name
        m["w_mlp"] = wm
        m["w_xgb"] = wx
        rows.append(m)
        print(f"\n[{name}]  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}  "
              f"Acc={m['accuracy']:.4f}  BalAcc={m['balanced_accuracy']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}")

    # 2) Weight grid scan (dynamic ensemble ablation)
    # Scan w_mlp from 0.0 to 1.0 in 0.1 steps, both with and without s4pred.
    # Reports the optimum w_mlp, plus the default (0.4) and a flat 0.5 anchor.
    print("\n" + "=" * 80)
    print("[Weight scan] Scanning w_mlp from 0.0 to 1.0 in 0.1 steps ...")
    print("=" * 80)
    grid_rows = []
    for s4p_label, pm_base, px_base in [("s4pred_on", p_mlp_full, p_xgb_full),
                                          ("s4pred_off", p_mlp_noss, p_xgb_noss)]:
        print(f"\n--- {s4p_label} ---")
        for wm in np.arange(0.0, 1.01, 0.1):
            wx = 1.0 - wm
            prob = wm * pm_base + wx * px_base
            pred = (prob > 0.5).astype(int)
            m = compute_metrics(Y_val, pred)
            grid_rows.append({
                "feature_set": s4p_label,
                "w_mlp": round(wm, 2),
                "w_xgb": round(wx, 2),
                **m,
            })
            print(f"  w_mlp={wm:.1f}  Acc={m['accuracy']:.4f}  BalAcc={m['balanced_accuracy']:.4f}  "
                  f"F1={m['f1']:.4f}  MCC={m['mcc']:.4f}  "
                  f"TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")

    # Best by accuracy per feature set
    for s4p_label in ["s4pred_on", "s4pred_off"]:
        sub = [r for r in grid_rows if r["feature_set"] == s4p_label]
        best = max(sub, key=lambda r: r["accuracy"])
        print(f"\n[Best {s4p_label}]  w_mlp={best['w_mlp']:.2f}  w_xgb={best['w_xgb']:.2f}  "
              f"Acc={best['accuracy']:.4f}  F1={best['f1']:.4f}  MCC={best['mcc']:.4f}")

    # Append best-of-grid as additional rows
    for s4p_label in ["s4pred_on", "s4pred_off"]:
        sub = [r for r in grid_rows if r["feature_set"] == s4p_label]
        best = max(sub, key=lambda r: r["accuracy"])
        rows.append({
            "variant": f"best_grid_{s4p_label}",
            "w_mlp": best["w_mlp"],
            "w_xgb": best["w_xgb"],
            "TP": best["TP"], "TN": best["TN"], "FP": best["FP"], "FN": best["FN"],
            "accuracy": best["accuracy"],
            "balanced_accuracy": best["balanced_accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "mcc": best["mcc"],
        })

    # ---- no_phase1: skip Phase 1, use ESM-2 KNN classifier as Phase 2-only proxy ----
    # The Phase 2 encoder projects 1280-dim ESM-2 → 512-dim and uses contrastive
    # learning. Since we don't have raw val sequences in memory (X_val only has
    # pooled features), we approximate Phase 2-only classification with a
    # KNN classifier on the 1280-dim ESM-2 pooled features. This is the
    # natural "similarity-based" classifier that the Phase 2 retrieval
    # architecture collapses to when the MLP/XGB gating is removed.
    from sklearn.neighbors import KNeighborsClassifier
    X_train_full = np.load(os.path.join(args.data_dir, "X_train.npy"))
    Y_train_full = np.load(os.path.join(args.data_dir, "Y_train.npy"))
    X_train_esm2 = X_train_full[:, :n_esm2]
    print(f"\n[no_phase1] KNN on ESM-2 {n_esm2}-dim pooled features (Phase 2-only proxy)")
    print(f"[no_phase1] X_train_esm2={X_train_esm2.shape}, Y_train pos={int(Y_train_full.sum())} neg={int((1-Y_train_full).sum())}")
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs=-1)
    knn.fit(X_train_esm2, Y_train_full)
    pred_knn = knn.predict(X_esm2)
    m = compute_metrics(Y_val, pred_knn)
    m["variant"] = "no_phase1"
    m["w_mlp"] = 0.0
    m["w_xgb"] = 0.0
    rows.append(m)
    print(f"\n[no_phase1]  KNN(k=10, cosine)  "
          f"TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}  "
          f"Acc={m['accuracy']:.4f}  BalAcc={m['balanced_accuracy']:.4f}  "
          f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}")
    print(f"  (Phase 1 disabled; ESM-2 similarity KNN as Phase 2-only proxy)")

    # ---- Save CSV / Markdown ----
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    cols = ["variant", "w_mlp", "w_xgb", "TP", "TN", "FP", "FN",
            "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc"]
    df = df[cols]
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[Save] {out_csv}")

    # ---- Save weight grid scan separately ----
    grid_csv = out_csv.with_name("weight_scan.csv")
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    print(f"[Save] {grid_csv}  ({len(grid_df)} rows)")

    md = ["# Direct Balanced Held-out Ablation Results (Phase 1 only)",
          "",
          "Balanced val (TCDB 2401 pos + UniProt 2384 neg, 1:1) evaluated directly with retrained Phase 1 models. ",
          "Variants that use Phase 2 (e.g. `no_phase1`) bypass the v3 inference pipeline because that pipeline filters out non-TCDB entries.",
          "",
          "| Variant | w_mlp | w_xgb | TP | TN | FP | FN | Acc | BalAcc | Precision | Recall | F1 | MCC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['w_mlp']} | {r['w_xgb']} | "
            f"{r['TP']} | {r['TN']} | {r['FP']} | {r['FN']} | "
            f"{r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['mcc']:.4f} |"
        )
    md_path = out_csv.with_suffix(".md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[Save] {md_path}")


if __name__ == "__main__":
    main()
