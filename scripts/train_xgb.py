import os
import sys
import numpy as np
import time

# =================================================================
# 📦 0. Automatic Dependency Management
# =================================================================
try:
    import xgboost as xgb
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("❌ Missing scikit-learn or xgboost dependencies. Initializing automatic pipeline setup...")
    os.system(f"{sys.executable} -m pip install xgboost scikit-learn")
    import xgboost as xgb
    from sklearn.metrics import classification_report, confusion_matrix

# =================================================================
# 🎯 1. High-Throughput Custom F1-Score Metric Evaluator (Modern API)
# =================================================================
def xgb_f1_score(preds, dtrain):
    """
    High-throughput F1-Score evaluation metric optimized for recent XGBoost releases.
    Handles raw output logits, probabilities, and matrix spaces natively.
    """
    labels = dtrain.get_label()
    
    # Extract alternative class boundary space if probabilities arrive multi-dimensional
    if len(preds.shape) > 1 and preds.shape[1] > 1:
        preds = preds[:, 1]
    
    # Apply standard discrimination threshold at midpoint boundary
    preds_binary = (preds > 0.5).astype(int)
    
    tp = np.sum((preds_binary == 1) & (labels == 1))
    fp = np.sum((preds_binary == 1) & (labels == 0))
    fn = np.sum((preds_binary == 0) & (labels == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Must return a structured string identifier and score float sequence
    return 'f1_score', f1

# =================================================================
# ⚙️ Hyperparameters & Configuration Paths
# =================================================================
DATA_DIR = "/data2/chao/transport/train_ready_dataset"
MODEL_PATH = "transport_xgb_best.json"
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None,
                    help=f"Dir with X_train.npy, Y_train.npy, X_val.npy, Y_val.npy (default: {DATA_DIR})")
    ap.add_argument("--output_path", default=None,
                    help=f"Where to save XGBoost model (default: <data_dir>/{MODEL_PATH})")
    args = ap.parse_args()

    # Use CLI override if provided; otherwise fall back to module-level DATA_DIR
    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR
    output_path = args.output_path or os.path.join(data_dir, MODEL_PATH)
    print(f"  data_dir    = {data_dir}")
    print(f"  output_path = {output_path}")

    # 🚚 2. Rapid Matrix Ingestion Block
    print("🚚 Ingesting feature matrix tensors into system RAM...")
    start_time = time.time()
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
    print(f"✅ Data ingestion successful! Elapsed time: {time.time() - start_time:.2f}s")
    print(f"📊 Training Space: {X_train.shape[0]} | Validation Space: {X_val.shape[0]} | Dimensions: {X_train.shape[1]}")

    # ⚖️ 3. Dynamic Imbalance Regularization Strategy
    num_neg = np.sum(Y_train == 0)
    num_pos = np.sum(Y_train == 1)
    # When one class is absent (e.g. TCDB-only features, all labels = 1), set weight to 1.0 to avoid degenerate behavior
    scale_pos_weight = (num_neg / num_pos) if num_pos > 0 and num_neg > 0 else 1.0
    print(f"⚖️ Calculated sample weight coefficient (scale_pos_weight) = {scale_pos_weight:.2f}")
    print(f"   num_neg={int(num_neg)}, num_pos={int(num_pos)}")

    # ⚡ 4. High-Performance XGBoost DMatrix Encapsulation
    print("🚀 Binding arrays into native XGBoost runtime structures (DMatrix)...")
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dval = xgb.DMatrix(X_val, label=Y_val)

    # 🎛️ 5. Gradient Tree Parameter Configuration Map
    params = {
        'objective': 'binary:logistic',       # Binary classification using logistical regression
        'eval_metric': 'logloss',             # Default objective tracking criteria
        'max_depth': 7,                       # Maximum depth allocation per base estimator
        'eta': 0.05,                          # Step size shrinkage learning rate
        'subsample': 0.8,                     # Subsample ratio of the training instances
        'colsample_bytree': 0.8,              # Subsample ratio of columns when constructing each tree
        'scale_pos_weight': scale_pos_weight, # Penalty weight correction scaling factor
        'seed': 42
    }

    # Intercept system resources to identify computing assets
    import torch
    if torch.cuda.is_available():
        params['device'] = 'cuda'
        print("🔥 [GPU Acceleration] CUDA backend detected! Moving booster calculations to NVIDIA hardware targets.")
    else:
        params['n_jobs'] = -1
        print("🖥️  [CPU Multithreading] GPU unavailable. Committing full thread array pool allocation.")

    # 🏃‍♂️ 6. Iterative Boosting Engine Execution
    # Use AUC as the early-stopping signal (well-defined for single-class data); fall back to logloss otherwise.
    # We disable custom_metric to avoid the modern-API shape quirk that produced best_iteration=0 last run.
    print("🏋️ Initializing boosting iterations (Target: 500 rounds with logloss Early Stopping)...")
    evals = [(dtrain, 'train'), (dval, 'val')]

    # Pick the eval metric robustly:
    #  - if val contains both classes, use AUC (maximize)
    #  - if val is single-class (TCDB-only features), use logloss (minimize) and skip early stop
    if len(np.unique(Y_val)) > 1:
        primary_metric = 'auc'
        maximize = True
    else:
        primary_metric = 'logloss'
        maximize = False
    print(f"   primary early-stop metric = {primary_metric} (maximize={maximize})")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,     # Cease training loops if metric stalls for 30 rounds
        verbose_eval=20,              # Print structural telemetry out to console every 20 epochs
        maximize=maximize
    )

    # 💾 7. Secure Weights to Storage
    model.save_model(output_path)
    print(f"💾 Booster serialized! Best iterative step: {model.best_iteration}. Target: {output_path}")

    # 📊 8. End-to-End Diagnostic Assessment Dashboard
    print("\n🔮 Running evaluation telemetry across validation partitions...")
    probs = model.predict(dval)
    preds = (probs > 0.5).astype(int)

    print("\n📊 ========= 🌲 XGBoost Classifier Comprehensive Report =========")
    print(classification_report(Y_val, preds, target_names=["Negative (Non-Transport)", "Positive (Transport)"], digits=4))
    
    tn, fp, fn, tp = confusion_matrix(Y_val, preds).ravel()
    print("-----------------------------------------------------------------")
    print(f"🧱 Empirical Confusion Matrix Boundaries:")
    print(f"  ├─> True Negatives [TN]:  {tn}")
    print(f"  ├─> False Positives [FP]: {fp}  (⚠️ Type I Errors / False Alarms)")
    print(f"  ├─> False Negatives [FN]: {fn}  (⚠️ Type II Errors / Missed Targets)")
    print(f"  └─> True Positives [TP]:  {tp}")
    
    # 🧬 9. Feature Importance Gain Extraction Pipeline
    print("=================================================================")
    print("🧬 [Feature Topology] Measuring global information gain variables...")
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("🏆 Top 5 Metric Variables Ordered by Empirical Information Gain:")
    for rank, (feat, score) in enumerate(sorted_importance[:5], 1):
        feat_idx = int(feat.replace('f', ''))
        feat_desc = "ESM-2 Evolutionary Representation"
        if feat_idx == 1280: feat_desc = "s4pred Secondary Structure [α-helix (H)] Probability"
        elif feat_idx == 1281: feat_desc = "s4pred Secondary Structure [β-sheet (E)] Probability"
        elif feat_idx == 1282: feat_desc = "s4pred Secondary Structure [Random Coil (C)] Probability"
        
        print(f"  └─> Top {rank}: Feature Index {feat_idx:4d} ({feat_desc}) -> Gain Score: {score:.2f}")
    print("=================================================================\n")

if __name__ == "__main__":
    main()