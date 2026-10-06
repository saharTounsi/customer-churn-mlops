import numpy as np
import pandas as pd

from src.features.preprocess import clean_raw_dataframe, encode_features, split_data


def test_clean_raw_dataframe():
    """Vérifie la suppression de la colonne ID et l'encodage binaire de la cible."""
    df = pd.DataFrame(
        {"customerID": ["001", "002"], "Churn": ["Yes", "No"], "feature1": [10, 20]}
    )

    cleaned = clean_raw_dataframe(df, id_col="customerID", target_col="Churn")
    assert "customerID" not in cleaned.columns
    assert list(cleaned["Churn"]) == [1, 0]


def test_encode_features_normalization_bounds():
    """Vérifie que les variables numériques sont
    strictement normalisées entre 0 et 1.
    """
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male"],
            "MonthlyCharges": [20.0, 50.0, 100.0],
            "Contract": ["Month-to-month", "One year", "Two year"],
        }
    )

    encoded = encode_features(
        df,
        binary_cols=["gender"],
        categorical_cols=["Contract"],
        numerical_cols=["MonthlyCharges"],
    )

    assert encoded["MonthlyCharges"].min() >= 0.0
    assert encoded["MonthlyCharges"].max() <= 1.0
    assert not encoded.isnull().values.any()


def test_split_data_proportions():
    """Vérifie les proportions de train et test."""
    df = pd.DataFrame({"feat1": np.random.randn(100), "Churn": [0, 1] * 50})
    X_train, X_test, y_train, y_test = split_data(
        df, target_col="Churn", test_size=0.25
    )
    assert len(X_train) == 75
    assert len(X_test) == 25
    assert len(y_train) == 75
    assert len(y_test) == 25
