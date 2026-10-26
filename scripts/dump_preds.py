#!/usr/bin/env python
"""
dump_preds.py
=============
Dump per-protein Phase 1 predictions for the balanced TCDB held-out set
(n = 4,785), so that paired significance tests can be computed afterwards.

Cached mode (features already extracted by the training pipeline):
  python dump_preds.py \
      --features /data2/chao/transport/data/features_v3_retrain/X_val.npy \
      --mlp_ckpt  /data2/chao/transport/models/v3_retrained/phase1/transport_classifier_best.pth \
      --xgb_model /data2/chao/transport/models/v3_retrained/phase1/transport_xgb_best.json \
      --fasta    /data2/chao/transport/data/tcdb_val_balanced.fasta \
      --w_mlp 0.4 --w_xgb 0.6 --config 650M_cached --out preds/650M_cached.csv

Fresh mode: use --features pointing at the features written by the independent
end-to-end release-script run (same X_val layout, same row order).

Ensemble weights (from Methods 4.1):
  650M backbone : 0.4 * MLP + 0.6 * XGBoost
  3B   backbone : 0.8 * MLP + 0.2 * XGBoost

Output columns: protein_id, config, y_true, y_pred, transport_prob
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

SCRIPTS_DIR = "/data2/chao/transport/scripts"
sys.path.insert(0, SCRIPTS_DIR)
import predict_transport_family_v3 as v3  # noqa: E402


def read_fasta_ids(path: str) -> list:
    """First whitespace-separated token of each header, in file order."""
    ids = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                ids.append(line[1:].split()[0].strip())
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="X_val.npy of the config being dumped")
    ap.add_argument("--mlp_ckpt", required=True)
    ap.add_argument("--xgb_model", required=True)
    ap.add_argument("--fasta", required=True, help="FASTA whose order matches X_val rows")
    ap.add_argument("--w_mlp", type=float, required=True)
    ap.add_argument("--w_xgb", type=float, required=True)
    ap.add_argument("--config", required=True,
                    help="650M_cached | 650M_fresh | 3B_cached | 3B_fresh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    X = np.load(args.features)
    Y = np.load(os.path.join(os.path.dirname(args.features), "Y_val.npy"))
    ids = read_fasta_ids(args.fasta)

    if len(ids) != len(Y):
        raise SystemExit(f"fasta ids ({len(ids)}) != Y_val ({len(Y)}): order mismatch, cannot align")
    if len(X) != len(Y):
        raise SystemExit(f"X_val rows ({len(X)}) != Y_val ({len(Y)})")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"[Device] {device}   feat_dim={X.shape[1]}   n={len(X)}")

    model = v3.TransportClassifier(input_dim=X.shape[1]).to(device)
    model.load_state_dict(torch.load(args.mlp_ckpt, map_location=device, weights_only=True))
    model.eval()

    probs = []
    with torch.no_grad():
        for s in range(0, len(X), 1024):
            xb = torch.tensor(X[s:s + 1024], dtype=torch.float32, device=device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy().flatten())
    p_mlp = np.concatenate(probs)

    bst = xgb.Booster()
    bst.load_model(args.xgb_model)
    p_xgb = bst.predict(xgb.DMatrix(X))

    prob = args.w_mlp * p_mlp + args.w_xgb * p_xgb
    pred = (prob > 0.5).astype(int)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame({
        "protein_id": ids,
        "config": args.config,
        "y_true": Y.astype(int),
        "y_pred": pred,
        "transport_prob": prob,
    }).to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[Save] {args.out}  ({len(ids)} rows)  pos_pred={int(pred.sum())}  pos_true={int(Y.sum())}")


if __name__ == "__main__":
    main()
