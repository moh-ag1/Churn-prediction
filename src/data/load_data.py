from pathlib import Path

import pandas as pd


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the Telco Customer Churn CSV file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, got: {path.suffix}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df