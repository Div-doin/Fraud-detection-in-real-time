import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "models", "model_comparison_extended.csv")
charts_dir = os.path.join(BASE_DIR, "models", "charts")
os.makedirs(charts_dir, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print("✅ Loaded results:")
print(df.head())

metrics = ["F1", "Recall", "PR_AUC"]
for metric in metrics:
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="Dataset", y=metric, hue="Model")
    plt.title(f"{metric} Comparison Across Datasets and Models")
    plt.ylabel(metric)
    plt.xlabel("Dataset Variant")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.savefig(os.path.join(charts_dir, f"{metric.lower()}_comparison.png"))
    plt.show()

print(f"📊 Charts saved in: {charts_dir}")
