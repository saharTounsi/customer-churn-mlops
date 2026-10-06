"""
Stage 'train' du pipeline DVC.
Responsabilite unique : entrainer le modele RandomForest et sauvegarder l'artefact.
"""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_df = pd.read_csv("data/processed/train.csv")

    target_col = [c for c in train_df.columns if "Churn" in c][0]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    print(
        f"[train] Entrainement sur {len(X_train)} exemples "
        f"avec {len(X_train.columns)} features"
    )

    model = RandomForestClassifier(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
        min_samples_split=params["train"]["min_samples_split"],
        random_state=params["train"]["random_state"],
    )

    model.fit(X_train, y_train)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)

    print(f"[train] Modele sauvegarde avec succes dans : {model_path}")
    print(f"[train] Score Train Accuracy : {model.score(X_train, y_train):.4f}")


if __name__ == "__main__":
    main()
