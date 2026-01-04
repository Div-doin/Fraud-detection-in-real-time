import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import time

print("="*60)
print("QUICK TRAINING: CatBoost (Variant III) ONLY")
print("="*60)

# Load data
print("\n📂 Loading data...")
X_train = pd.read_csv('cloud_fraud_detection/data/processed/Base_X_train.csv')
y_train = pd.read_csv('cloud_fraud_detection/data/processed/Base_y_train.csv')
X_test = pd.read_csv('cloud_fraud_detection/data/processed/Base_X_test.csv')
y_test = pd.read_csv('cloud_fraud_detection/data/processed/Base_y_test.csv')

print(f"✓ Training data: {X_train.shape}")
print(f"✓ Test data: {X_test.shape}")

# ✅ CRITICAL: Remove x1 and x2 if they exist
columns_to_remove = ['x1', 'x2']
original_cols = X_train.columns.tolist()

X_train = X_train.drop(columns=[col for col in columns_to_remove if col in X_train.columns], errors='ignore')
X_test = X_test.drop(columns=[col for col in columns_to_remove if col in X_test.columns], errors='ignore')

removed = [col for col in columns_to_remove if col in original_cols]
if removed:
    print(f"\n⚠️  Removed columns: {removed}")
    print(f"✓ New training shape: {X_train.shape}")
    print(f"✓ New test shape: {X_test.shape}")

# Convert to numpy arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Get feature names
feature_names = X_train.columns.tolist()
print(f"\n✓ Final features: {len(feature_names)}")

# Train CatBoost (Variant III)
print("\n" + "="*60)
print("🚀 TRAINING: CatBoost (Variant III)")
print("="*60)

start_time = time.time()

model = CatBoostClassifier(
    iterations=200,           # Reduced from 500 for faster training
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    border_count=128,
    thread_count=-1,          # Use all CPU cores
    verbose=50,               # Show progress every 50 iterations
    random_seed=42
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=50)
train_time = time.time() - start_time

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)
print(f"\nTraining Time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Save model
model_bundle = {
    'model': model,
    'feature_names': feature_names,
    'n_features': len(feature_names),
    'variant': 'Variant III',
    'algorithm': 'CatBoost',
    'train_time': train_time,
    'roc_auc': roc_auc
}

output_path = 'fraud_detection_api/models/Variant III_CatBoost.pkl'
joblib.dump(model_bundle, output_path)

print(f"\n✅ Model saved to: {output_path}")
print(f"✅ Features in model: {len(feature_names)}")
print(f"✅ NO x1 or x2 included!")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Recreate scaler: python create_scaler.py")
print("2. Restart API: uvicorn fraud_detection_api.api.main:app --reload")
print("3. Test at: http://127.0.0.1:8000/static/index.html")