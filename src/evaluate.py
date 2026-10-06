"""
Stage 'evaluate' du pipeline DVC.
Responsabilite unique : evaluer le modele et exporter les metriques.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():
    test_df = pd.read_csv("data/processed/test.csv")

    target_col = [c for c in test_df.columns if "Churn" in c][0]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    model = joblib.load("models/model.pkl")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("[evaluate] Metriques :")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("[evaluate] Metriques sauvegardees dans : metrics/metrics.json")


if __name__ == "__main__":
    main()
