# train_models_extended.py
# ==========================================
# Train multiple models, handle imbalance, SMOTE option, threshold tuning
# FIXED: Explicitly removes x1 and x2 features before training
# ==========================================

import os
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Try importing optional libraries; set flags if unavailable
HAS_SMOTE = True
HAS_XGB = True
HAS_LGB = True
HAS_CAT = True
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    HAS_SMOTE = False

try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
except Exception:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
except Exception:
    HAS_CAT = False

# -------------------------------
# Directories
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------
# Config
# -------------------------------
TARGET_COL = "fraud_bool"  # used only to locate columns removed earlier
SMOTE_APPLY = True if HAS_SMOTE else False  # set to True to enable SMOTE (if imblearn installed)
DO_SAMPLE = False            # set True to sample for faster dev/troubleshooting
SAMPLE_FRAC = 0.2            # fraction to sample if DO_SAMPLE True
THRESHOLDS = np.linspace(0.01, 0.99, 99)  # thresholds to search for best F1

# ✅ CRITICAL: Features to remove
FEATURES_TO_REMOVE = ['x1', 'x2']

# -------------------------------
# Models dictionary: name -> constructor (with useful defaults)
# -------------------------------
models = {}

models["LogisticRegression"] = lambda: LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
models["RandomForest"] = lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced', random_state=42)
models["ExtraTrees"] = lambda: ExtraTreesClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced', random_state=42)
models["AdaBoost"] = lambda: AdaBoostClassifier(n_estimators=100, random_state=42)

if HAS_XGB:
    models["XGBoost"] = lambda: XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, scale_pos_weight=1)
if HAS_LGB:
    models["LightGBM"] = lambda: LGBMClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced')
if HAS_CAT:
    models["CatBoost"] = lambda: CatBoostClassifier(n_estimators=200, verbose=0, auto_class_weights='Balanced')

if not (HAS_XGB or HAS_LGB or HAS_CAT):
    print("⚠️ Warning: None of XGBoost/LightGBM/CatBoost are available. You can install them for better models.")

# -------------------------------
# Helper functions
# -------------------------------
def best_threshold_and_metrics(clf, X_val, y_val, thresholds=THRESHOLDS):
    """Given classifier with predict_proba or decision_function, find threshold that maximizes F1 on validation set.
       Returns dict of metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC) and best threshold.
    """
    # Get probabilities or scores
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_val)[:,1]
    else:
        # fallback to decision_function (e.g., some classifiers)
        scores = clf.decision_function(X_val)
        # convert to probabilities via sigmoid might be possible but we'll treat as scores
        # we'll compare thresholds on the raw scores
    best_f1 = -1.0
    best_t = 0.5
    best_metrics = {}
    # compute global ROC-AUC and PR-AUC
    try:
        roc_auc = roc_auc_score(y_val, scores)
    except Exception:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_val, scores)
    except Exception:
        pr_auc = np.nan

    for t in thresholds:
        y_pred_t = (scores >= t).astype(int)
        # avoid zero division issues; zero_division=0 used below in precision/recall
        prec = precision_score(y_val, y_pred_t, zero_division=0)
        rec = recall_score(y_val, y_pred_t, zero_division=0)
        f1 = f1_score(y_val, y_pred_t, zero_division=0)
        acc = accuracy_score(y_val, y_pred_t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    # attach aucs
    best_metrics["ROC_AUC"] = roc_auc
    best_metrics["PR_AUC"] = pr_auc
    return best_t, best_metrics

# -------------------------------
# MAIN: iterate through processed datasets
# -------------------------------
results = []

train_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith("_X_train.csv")])
if not train_files:
    raise FileNotFoundError("No processed datasets found. Run preprocessing first and ensure files are in data/processed")

for train_file in train_files:
    prefix = train_file.replace("_X_train.csv", "")
    print(f"\n{'='*60}")
    print(f"Dataset: {prefix}")
    print(f"{'='*60}")
    
    # load
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, f"{prefix}_X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, f"{prefix}_y_train.csv")).values.ravel()
    X_val = pd.read_csv(os.path.join(PROCESSED_DIR, f"{prefix}_X_val.csv"))
    y_val = pd.read_csv(os.path.join(PROCESSED_DIR, f"{prefix}_y_val.csv")).values.ravel()

    print(f"Original shapes - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # ✅ CRITICAL FIX: Remove x1 and x2 if they exist
    columns_to_drop = [col for col in FEATURES_TO_REMOVE if col in X_train.columns]
    if columns_to_drop:
        print(f"⚠️  Removing features: {columns_to_drop}")
        X_train = X_train.drop(columns=columns_to_drop)
        X_val = X_val.drop(columns=columns_to_drop)
        print(f"✓ New shapes - Train: {X_train.shape}, Val: {X_val.shape}")
    else:
        print(f"✓ No x1/x2 features found - data is clean")
    
    # Store final feature names
    final_feature_names = X_train.columns.tolist()
    print(f"✓ Training with {len(final_feature_names)} features")

    # optional sampling for dev speed
    if DO_SAMPLE:
        print(f"⚡ Sampling for faster dev run (frac = {SAMPLE_FRAC})")
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=SAMPLE_FRAC, stratify=y_train, random_state=42)

    # Optionally apply SMOTE to training set (only on training data)
    if SMOTE_APPLY:
        if not HAS_SMOTE:
            print("⚠️ imbalanced-learn not installed; skipping SMOTE.")
        else:
            print("🔄 Applying SMOTE to training set (balance classes)...")
            sm = SMOTE(random_state=42)
            X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
            print(f"   Before SMOTE: {X_train.shape}, After SMOTE: {X_train_bal.shape}")
    else:
        X_train_bal, y_train_bal = X_train, y_train

    for model_name, model_ctor in models.items():
        print(f"\n→ Training model: {model_name}")
        clf = model_ctor()
        t0 = time.time()
        try:
            clf.fit(X_train_bal, y_train_bal)
        except Exception as e:
            print(f"❌ Training failed for {model_name}: {e}")
            continue
        train_time = time.time() - t0
        
        # threshold tuning on validation
        try:
            best_t, metrics = best_threshold_and_metrics(clf, X_val, y_val)
        except Exception as e:
            print(f"❌ Threshold tuning failed for {model_name}: {e}")
            best_t = 0.5
            # fallback metrics
            y_pred = clf.predict(X_val)
            metrics = {
                "Accuracy": accuracy_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred, zero_division=0),
                "Recall": recall_score(y_val, y_pred, zero_division=0),
                "F1": f1_score(y_val, y_pred, zero_division=0),
                "ROC_AUC": np.nan,
                "PR_AUC": np.nan
            }

        # Save model and threshold WITH feature names
        model_filename = os.path.join(MODELS_DIR, f"{prefix}_{model_name}.pkl")
        model_bundle = {
            "model": clf, 
            "threshold": float(best_t),
            "feature_names": final_feature_names  # ✅ Save feature names
        }
        joblib.dump(model_bundle, model_filename)

        # append results
        results.append({
            "Dataset": prefix,
            "Model": model_name,
            "Accuracy": metrics.get("Accuracy", np.nan),
            "Precision": metrics.get("Precision", np.nan),
            "Recall": metrics.get("Recall", np.nan),
            "F1": metrics.get("F1", np.nan),
            "ROC_AUC": metrics.get("ROC_AUC", np.nan),
            "PR_AUC": metrics.get("PR_AUC", np.nan),
            "BestThreshold": float(best_t),
            "TrainTimeSec": train_time,
            "NumFeatures": len(final_feature_names)
        })
        print(f"✅ Trained {model_name} (time: {train_time:.1f}s)")
        print(f"   Best threshold: {best_t:.4f}")
        print(f"   F1: {metrics.get('F1', np.nan):.4f}")
        print(f"   Precision: {metrics.get('Precision', np.nan):.4f}")
        print(f"   Recall: {metrics.get('Recall', np.nan):.4f}")
        print(f"   Saved to: {model_filename}")

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Dataset", "F1"], ascending=[True, False])
results_path = os.path.join(MODELS_DIR, "model_comparison_extended.csv")
results_df.to_csv(results_path, index=False)

print("\n" + "="*60)
print("📊 TRAINING COMPLETE")
print("="*60)
print(f"\nResults saved to: {results_path}")
print("\nTop 5 models by F1 score:")
print(results_df.nlargest(5, 'F1')[['Dataset', 'Model', 'F1', 'Precision', 'Recall', 'NumFeatures']])

print("\n✅ All models trained WITHOUT x1 and x2 features!")
print("\nNext steps:")
print("1. Run: python create_scaler.py  (to create matching scaler)")
print("2. Update your API to use the new model")
print("3. Restart API: uvicorn fraud_detection_api.api.main:app --reload")