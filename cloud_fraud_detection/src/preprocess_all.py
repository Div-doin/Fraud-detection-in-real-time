# ==========================================
# PREPROCESS MULTIPLE FRAUD DATASET VARIANTS
# ==========================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define paths (make sure these paths match your folder)
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# List all CSV files inside raw_dir
files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
print("🔍 Found datasets:", files)

# Loop through and preprocess each file
for file in files:
    print(f"\n🔄 Processing {file} ...")
    file_path = os.path.join(RAW_DIR, file)
    df = pd.read_csv(file_path)

    target_col = "fraud_bool"

    # Replace invalid values (-1) with NaN if those columns exist
    for col in ['prev_address_months_count', 'current_address_months_count']:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    # Fill missing numeric values with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Scale numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col)
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Split into train/val/test
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Save processed files
    prefix = file.split(".")[0]
    X_train.to_csv(f"{PROCESSED_DIR}/{prefix}_X_train.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/{prefix}_y_train.csv", index=False)
    X_val.to_csv(f"{PROCESSED_DIR}/{prefix}_X_val.csv", index=False)
    y_val.to_csv(f"{PROCESSED_DIR}/{prefix}_y_val.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/{prefix}_X_test.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/{prefix}_y_test.csv", index=False)

    print(f"✅ Finished {file} → Saved to {PROCESSED_DIR}/")

print("\n🎉 All dataset variants processed successfully!")
