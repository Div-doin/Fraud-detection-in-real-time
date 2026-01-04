import joblib
import pandas as pd

bundle = joblib.load("models/Variant III_CatBoost.pkl")
model = bundle["model"]
threshold = bundle["threshold"]

# Example new transaction(s)
X_new = pd.read_csv("data/processed/Variant III_X_test.csv").head(150)
proba = model.predict_proba(X_new)[:,1]
pred = (proba >= threshold).astype(int)
print(pred)
import matplotlib.pyplot as plt

plt.hist(proba, bins=50)
plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.2f}")
plt.xlabel("Predicted Fraud Probability")
plt.ylabel("Number of Transactions")
plt.title("Distribution of Fraud Probabilities")
plt.legend()
plt.show()
