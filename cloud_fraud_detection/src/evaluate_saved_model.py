import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# -------------------------------
# Paths
# -------------------------------
import  sys
os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.getcwd())

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Variant III_CatBoost.pkl")  # change to your best model
DATA_X = os.path.join(BASE_DIR, "data", "processed", "Variant III_X_test.csv")
DATA_Y = os.path.join(BASE_DIR, "data", "processed", "Variant III_y_test.csv")

# -------------------------------
# Load model + data
# -------------------------------
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = bundle["threshold"]

X_test = pd.read_csv(DATA_X)
y_test = pd.read_csv(DATA_Y).values.ravel()

# -------------------------------
# Predict probabilities and labels
# -------------------------------
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# -------------------------------
# Metrics
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

print("\n✅ Evaluation complete!")
print("Threshold used:", threshold)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# -------------------------------
# Plot ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Plot fraud probability distribution
# -------------------------------
plt.figure(figsize=(7,5))
plt.hist(y_proba[y_test==0], bins=50, alpha=0.6, label="Legit Transactions")
plt.hist(y_proba[y_test==1], bins=50, alpha=0.6, label="Fraud Transactions")
plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.2f}")
plt.title("Distribution of Predicted Fraud Probabilities")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()
# idk what is problem but this is not giving output by running the play button but in terminal down should run first pwd and then 
# C:\Python313\python.exe src\evaluate_saved_model.py