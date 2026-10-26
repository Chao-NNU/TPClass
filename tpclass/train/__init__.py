"""Training utilities for TPClass.

* :mod:`tpclass.train.split` -- dataset splitting (stratified and
  homology-aware cluster split).
* :mod:`tpclass.train.dataset` -- dataset construction and feature extraction.
* :mod:`tpclass.train.phase1` -- Phase 1 MLP / XGBoost / isotonic calibration.
* :mod:`tpclass.train.phase2` -- Phase 2 contrastive training.
"""
from tpclass.train.dataset import build_training_dataset, load_tcdb_records
from tpclass.train.phase1 import extract_features, fit_isotonic, train_mlp, train_xgb
from tpclass.train.phase2 import SupConLoss, train_contrastive
from tpclass.train.split import (
    cluster_split,
    stratified_split,
    write_fasta,
    write_split_csv,
)

__all__ = [
    "build_training_dataset",
    "load_tcdb_records",
    "extract_features",
    "fit_isotonic",
    "train_mlp",
    "train_xgb",
    "SupConLoss",
    "train_contrastive",
    "cluster_split",
    "stratified_split",
    "write_fasta",
    "write_split_csv",
]
