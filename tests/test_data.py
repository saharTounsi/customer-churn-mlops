import pandas as pd
import pytest

from src.data.load_data import load_raw_data


def test_load_raw_data_valid(tmp_path):
    """Vérifie le chargement correct d'un fichier CSV valide."""
    dummy_csv = tmp_path / "sample.csv"
    dummy_csv.write_text("col1,col2\n1,2\n3,4")

    df = load_raw_data(dummy_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_raw_data_not_found():
    """Vérifie la levée d'exception pour un fichier inexistant."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("fichier_totalement_inexistant_9999.csv")
