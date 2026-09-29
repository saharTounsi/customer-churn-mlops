"""Pipeline d'entraînement et d'évaluation du modèle de churn client."""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Permet l'import de 'src' quel que soit le dossier d'exécution
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent),
)

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.load_data import load_raw_data
from src.features.preprocess import clean_raw_dataframe, encode_features, split_data


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Calcule les métriques de performance sur l'ensemble de test."""

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }


def run_pipeline(
    config_path: str = "configs/config.yaml",
) -> Tuple[Any, Dict[str, float]]:
    """Exécute le pipeline complet MLOps.

    Pipeline :
    Ingestion -> Features -> Train -> Eval -> Save
    """

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Ingestion sécurisée
    df_raw = load_raw_data(cfg["data"]["raw_path"])

    # 2. Préparation et Feature Engineering
    df_clean = clean_raw_dataframe(
        df_raw,
        cfg["features"]["id_column"],
        cfg["features"]["target_column"],
    )

    df_encoded = encode_features(
        df_clean,
        cfg["features"]["binary_columns"],
        cfg["features"]["categorical_columns"],
        cfg["features"]["numerical_columns"],
    )

    X_train, X_test, y_train, y_test = split_data(
        df_encoded,
        cfg["features"]["target_column"],
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    # 3. Entraînement du modèle
    model = LogisticRegression(
        C=cfg["model"]["regularization_c"],
        max_iter=cfg["model"]["max_iter"],
        random_state=cfg["model"]["random_state"],
    )

    model.fit(X_train, y_train)

    # 4. Évaluation sur les données de test
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("=" * 50)
    print("RÉSULTATS DE L'ÉVALUATION EN PRODUCTION")
    print("=" * 50)

    for k, v in metrics.items():
        print(f"* {k.upper():<12} : {v:.4f}")

    print("=" * 50)

    # 5. Sérialisation de l'artefact modèle
    save_path = Path(cfg["model"]["save_path"])

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modèle sauvegardé avec succès dans : " f"{save_path.resolve()}")

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline MLOps Churn Prediction")

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Chemin vers le fichier YAML",
    )

    args = parser.parse_args()

    run_pipeline(args.config)
