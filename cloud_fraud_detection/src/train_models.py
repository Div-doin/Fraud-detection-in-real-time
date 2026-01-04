# ==========================================
# STEP 3: TRAIN AND EVALUATE FRAUD DETECTION MODELS
# ==========================================

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# -------------------------------
# Define directories
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------
# Helper function: evaluate model
# -------------------------------
def evaluate_model(model, X_val, y_val):
    """Evaluate a trained model on validation data and return key metrics."""
    y_pred = model.predict(X_val)
    return {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1": f1_score(y_val, y_pred, zero_division=0)
    }

# -------------------------------
# Train models for all dataset variants
# -------------------------------
results = []

# Get all *_X_train.csv files from processed data
train_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_X_train.csv")]
if not train_files:
    raise FileNotFoundError("❌ No processed datasets found. Run preprocess_all.py first!")

for file in train_files:
    prefix = file.split("_X_train.csv")[0]
    print(f"\n🔄 Training models for dataset: {prefix} ...")

    # Load train/val data
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, f"{prefix}_X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, f"{prefix}_y_train.csv")).values.ravel()
    X_val = pd.read_csv(os.path.join(PROCESSED_DIR, f"{prefix}_X_val.csv"))
    y_val = pd.read_csv(os.path.join(PROCESSED_DIR, f"{prefix}_y_val.csv")).values.ravel()

    # -------------------------------
    # 1️⃣ Logistic Regression Model
    # -------------------------------
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    log_reg_scores = evaluate_model(log_reg, X_val, y_val)
    joblib.dump(log_reg, os.path.join(MODELS_DIR, f"{prefix}_logreg.pkl"))

    # -------------------------------
    # 2️⃣ Random Forest Model
    # -------------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_scores = evaluate_model(rf, X_val, y_val)
    joblib.dump(rf, os.path.join(MODELS_DIR, f"{prefix}_rf.pkl"))

    # -------------------------------
    # Store results
    # -------------------------------
    results.append({
        "Dataset": prefix,
        "Model": "LogisticRegression",
        **log_reg_scores
    })
    results.append({
        "Dataset": prefix,
        "Model": "RandomForest",
        **rf_scores
    })

    print(f"✅ Models for {prefix} trained and saved!")

# -------------------------------
# Save all model comparison results
# -------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)

print("\n📊 Model comparison saved to models/model_comparison.csv")
print("\n✅ All models trained successfully!\n")
print(results_df)
