# retrain_and_save.py
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- LOAD DATA ---
df = pd.read_csv("training_data.csv")  # adapt path
target_col = "is_fraud"                # adapt if needed

# --- DEFINE FEATURES: remove x1 and x2 here ---
all_cols = [c for c in df.columns if c != target_col]
# explicitly remove x1/x2 if present
for c in ["x1", "x2"]:
    if c in all_cols:
        all_cols.remove(c)

X = df[all_cols]
y = df[target_col]

# --- TRAIN / PIPELINE (simple example) ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),         # or your actual preprocessor
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# --- Save pipeline ---
MODEL_PATH = "model.joblib"
joblib.dump(pipeline, MODEL_PATH)
print("Saved model to", MODEL_PATH)

# --- Persist expected_features in the exact order the API requires ---
# If you used a ColumnTransformer or transformer that expands features (OHE), compute expanded names accordingly.
expected_features = list(X.columns)  # order preserved

with open("features.json", "w") as f:
    json.dump(expected_features, f, indent=2)
print("Saved expected features list to features.json")
