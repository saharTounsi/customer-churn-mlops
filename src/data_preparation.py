"""
Stage 'prepare' du pipeline DVC.
Responsabilite unique : charger les donnees brutes,
nettoyer et produire train.csv et test.csv.
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main():
    # 1. Charger les parametres de preparation
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    test_size = params["prepare"]["test_size"]
    random_state = params["prepare"]["random_state"]
    raw_path = params["prepare"]["raw_data_path"]

    # 2. Lecture du dataset brut
    df = pd.read_csv(raw_path)
    print(
        f"[prepare] Dataset charge : " f"{len(df)} lignes, {len(df.columns)} colonnes"
    )

    # 3. Nettoyage des identifiants et colonnes inutiles
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 4. Traitement des valeurs manquantes numeriques
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 5. Encodage One-Hot des variables categorielles
    df = pd.get_dummies(df, drop_first=True)

    # Convertir les colonnes booleennes en entiers
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # 6. Identification de la colonne cible
    target_col = [c for c in df.columns if "Churn" in c][0]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 7. Partitionnement stratifie Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 8. Sauvegarde des datasets traites
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(
        f"[prepare] Train: {len(train_df)} lignes, "
        f"Test: {len(test_df)} lignes sauvegardes avec succes."
    )


if __name__ == "__main__":
    main()
