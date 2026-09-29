"""Module de feature engineering et de préparation des données."""

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_raw_dataframe(df: pd.DataFrame, id_col: str, target_col: str) -> pd.DataFrame:
    """Supprime l'identifiant technique et convertit la cible en binaire.

    Args:
        df: DataFrame brut d'entrée.
        id_col: Nom de la colonne identifiant technique à supprimer.
        target_col: Nom de la colonne cible métier (ex: Churn).

    Returns:
        pd.DataFrame: DataFrame nettoyé sans effet de bord.
    """
    data = df.copy()

    # Suppression de l'identifiant technique
    if id_col in data.columns:
        data = data.drop(columns=[id_col])

    # Conversion de la variable cible en binaire
    if target_col in data.columns:
        data[target_col] = data[target_col].map({"Yes": 1, "No": 0})

    return data


def encode_features(
    df: pd.DataFrame,
    binary_cols: List[str],
    categorical_cols: List[str],
    numerical_cols: List[str],
) -> pd.DataFrame:
    """Applique l'encodage binaire, One-Hot et la normalisation Min-Max.

    Args:
        df: DataFrame nettoyé.
        binary_cols: Liste des colonnes binaires (Yes/No, Male/Female).
        categorical_cols: Liste des variables catégorielles nominales.
        numerical_cols: Liste des variables quantitatives continues.

    Returns:
        pd.DataFrame: Jeu de données entièrement transformé et encodé.
    """
    data = df.copy()

    # 1. Encodage binaire déterministe
    binary_map = {
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0,
    }

    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map(binary_map).fillna(0)

    # 2. Encodage One-Hot des colonnes catégorielles
    existing_cat = [col for col in categorical_cols if col in data.columns]

    if existing_cat:
        data = pd.get_dummies(
            data,
            columns=existing_cat,
            drop_first=True,
            dtype=int,
        )

    # 3. Imputation et normalisation Min-Max des variables numériques
    for col in numerical_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())

            c_min = data[col].min()
            c_max = data[col].max()

            if c_max > c_min:
                data[col] = (data[col] - c_min) / (c_max - c_min)
            else:
                data[col] = 0.0

    return data


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Découpe le dataset en ensembles train/test avec stratification.

    Args:
        df: DataFrame prêt pour la modélisation.
        target_col: Nom de la variable cible binaire.
        test_size: Proportion de l'échantillon de test.
        random_state: Graine aléatoire garantissant la reproductibilité.

    Returns:
        Tuple: X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
