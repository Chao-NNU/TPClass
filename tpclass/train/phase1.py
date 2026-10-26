"""Phase 1 training: MLP, XGBoost and isotonic calibration.

The feature extraction used here goes through exactly the same code path as
inference (:class:`tpclass.core.phase1.Phase1Classifier`), so a model trained
with this module sees the same representation it will see at prediction time.

Hyper-parameters follow the released training script
(``train_mlp.py`` / ``train_xgb.py``): AdamW with weight decay 1e-4, cosine
annealing over 100 epochs, batch size 1024, and an XGBoost model evaluated by
custom F1.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4


# ══════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ══════════════════════════════════════════════════════════════════════════
def extract_features(
    fasta: str | Path,
    esm2_path: str | Path,
    device: str = "cuda",
    max_len: int = 800,
    batch_size: int = 8,
) -> Tuple[list, np.ndarray]:
    """Build the Phase 1 feature matrix for a FASTA file.

    Returns
    -------
    (accessions, features)
        ``features`` has shape ``(n, embed_dim + 3)``.
    """
    from tpclass.core.phase1 import Phase1Classifier

    classifier = Phase1Classifier(
        esm2_path=esm2_path,
        checkpoint="unused",  # only the encoder is needed here
        backbone="650M",
        device=device,
        max_len=max_len,
        batch_size=batch_size,
    )
    names: list = []
    matrices: list = []
    for batch_names, features, _, _ in classifier.iter_batches(fasta):
        names.extend(batch_names)
        matrices.append(features)
    if not matrices:
        raise ValueError(f"No features could be extracted from {fasta}")
    return names, np.concatenate(matrices, axis=0)


# ══════════════════════════════════════════════════════════════════════════
#  MLP
# ══════════════════════════════════════════════════════════════════════════
def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    output_path: str | Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    device: Optional[str] = None,
    seed: int = 42,
) -> Path:
    """Train the Phase 1 MLP and write the best checkpoint.

    The checkpoint is a plain ``torch`` state dict for
    :class:`tpclass.core.phase1.TransportClassifier`, which is what
    :class:`tpclass.core.phase1.Phase1Classifier` expects at inference time.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from tpclass.core.phase1 import TransportClassifier

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

    model = TransportClassifier(input_dim=int(x_train.shape[1])).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * batch_x.size(0)
            correct += int(((logits > 0).float() == batch_y).sum().item())
            total += batch_x.size(0)
        scheduler.step()

        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                v_correct += int(((logits > 0).float() == batch_y.to(device)).sum().item())
                v_total += batch_x.size(0)
        val_acc = v_correct / max(v_total, 1)
        train_acc = correct / max(total, 1)
        logger.info(
            f"epoch {epoch:3d}/{epochs}  loss={loss_sum/max(total,1):.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            logger.info(f"  new best val_acc={val_acc:.4f} -> {output_path}")

    logger.info(f"MLP training done. Best val accuracy {best_val_acc:.4f} -> {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════
#  XGBoost
# ══════════════════════════════════════════════════════════════════════════
def train_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    output_path: str | Path,
    num_boost_round: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    seed: int = 42,
) -> Path:
    """Train the Phase 1 XGBoost model and write it as a ``.json``."""
    import xgboost as xgb
    from sklearn.metrics import f1_score

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_train = np.asarray(y_train).astype(int)
    y_val = np.asarray(y_val).astype(int)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    def f1_eval(preds, dmatrix):
        y = dmatrix.get_label()
        y_hat = (preds > 0.5).astype(int)
        return "f1", float(f1_score(y, y_hat))

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": max_depth,
        "eta": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "seed": seed,
        "tree_method": "hist",
    }
    evals = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        custom_metric=f1_eval,
        early_stopping_rounds=30,
        verbose_eval=25,
        evals_result=evals,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    logger.info(f"XGBoost training done -> {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════
#  Calibration
# ══════════════════════════════════════════════════════════════════════════
def fit_isotonic(
    raw_val_probs: np.ndarray,
    y_val: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Fit an isotonic calibrator on the validation split.

    The calibrator maps the raw ensemble probability onto the observed
    positive rate, which is what makes a single ``transport_prob`` threshold
    comparable across proteomes.
    """
    import joblib
    from sklearn.isotonic import IsotonicRegression

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.asarray(raw_val_probs, dtype=float), np.asarray(y_val).astype(int))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, output_path)
    logger.info(f"Isotonic calibrator -> {output_path}")
    return output_path
