import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Permet l'exécution directe sous Windows/Linux sans variable d'environnement
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402

from src.data.load_data import load_raw_data  # noqa: E402
from src.features.preprocess import (  # noqa: E402
    clean_raw_dataframe,
    encode_features,
    split_data,
)


def load_yaml_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Charge le fichier de configuration YAML du projet.

    Args:
        config_path: Chemin vers le fichier YAML.

    Returns:
        Dict[str, Any]: Paramètres du projet.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    reg_c: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Entraîne un classifieur de régression logistique.

    Args:
        X_train: Features d'entraînement.
        y_train: Cible d'entraînement.
        reg_c: Inverse du coefficient de régularisation L2.
        max_iter: Nombre maximal d'itérations du solver.
        random_state: Graine aléatoire.

    Returns:
        LogisticRegression: Modèle ajusté.
    """
    model = LogisticRegression(C=reg_c, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Calcule les métriques clés de performance sur le jeu de test.

    Args:
        model: Modèle entraîné compatible Scikit-Learn.
        X_test: Features du jeu de test.
        y_test: Cibles réelles de test.

    Returns:
        Dict[str, float]: Dictionnaire contenant accuracy, f1_score et roc_auc.
    """
    preds = model.predict(X_test)
    probs = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }
    return metrics


def save_model_artifact(model: Any, output_path: str) -> None:
    """Sérialise le modèle entraîné sur disque.

    Args:
        model: Objet modèle à sauvegarder.
        output_path: Chemin du fichier .pkl cible.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé avec succès dans : {path.resolve()}")


def run_pipeline(
    config_path: str = "configs/config.yaml",
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """Exécute l'intégralité du pipeline MLOps complet.

    Étapes : Ingestion -> Features -> Train -> Eval -> Save.

    Args:
        config_path: Chemin vers le fichier de configuration.

    Returns:
        Tuple[LogisticRegression, Dict[str, float]]: Modèle et métriques.
    """
    config = load_yaml_config(config_path)

    # 1. Ingestion
    print(f"[1/4] Ingestion des données depuis {config['data']['raw_path']}...")
    df_raw = load_raw_data(config["data"]["raw_path"])

    # 2. Nettoyage & Encodage
    print("[2/4] Preprocessing et Feature Engineering...")
    df_clean = clean_raw_dataframe(
        df_raw,
        id_col=config["features"]["id_column"],
        target_col=config["features"]["target_column"],
    )
    df_encoded = encode_features(
        df_clean,
        binary_cols=config["features"]["binary_columns"],
        categorical_cols=config["features"]["categorical_columns"],
        numerical_cols=config["features"]["numerical_columns"],
    )

    X_train, X_test, y_train, y_test = split_data(
        df_encoded,
        target_col=config["features"]["target_column"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # 3. Entraînement
    print(f"[3/4] Entraînement du modèle {config['model']['type']}...")
    model = train_logistic_regression(
        X_train,
        y_train,
        reg_c=config["model"]["regularization_c"],
        max_iter=config["model"]["max_iter"],
        random_state=config["model"]["random_state"],
    )

    # 4. Évaluation & Sauvegarde
    print("[4/4] Évaluation des performances...")
    metrics = evaluate_model(model, X_test, y_test)
    print("--- RÉSULTATS DU MODÈLE ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    save_model_artifact(model, config["model"]["save_path"])
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline MLOps d'entraînement du modèle Churn."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Chemin vers le fichier de configuration YAML.",
    )
    args = parser.parse_args()
    run_pipeline(args.config)
