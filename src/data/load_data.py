"""Module d’ingestion et de validation des données brutes."""

from pathlib import Path
from typing import Union

import pandas as pd


def load_raw_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Charge un fichier CSV de données brutes et valide son intégrité.

    Args:
        file_path: Chemin relatif ou absolu vers le fichier CSV.

    Returns:
        pd.DataFrame: Jeu de données chargé sous forme de DataFrame Pandas.

    Raises:
        FileNotFoundError: Si le fichier spécifié est introuvable sur le disque.
        ValueError: Si le fichier existe mais ne contient aucune ligne (vide).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {path.resolve()}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Le fichier {path} est vide.")

    return df
