from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import (
    build_features,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_processed.csv"
)


def main():

    print("📥 Loading raw dataset...")

    df = load_data(
        INPUT_PATH
    )

    print("🧹 Preprocessing...")

    df = preprocess_data(
        df,
        target_col="Churn",
    )

    print("⚙️ Building features...")

    processed = build_features(
        df,
        target_col="Churn",
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"✅ Saved processed data to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"Shape: {processed.shape}"
    )


if __name__ == "__main__":
    main()