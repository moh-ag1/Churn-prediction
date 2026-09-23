from pathlib import Path
import sys

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

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


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)


def main():

    print("=" * 60)
    print("PHASE 1 PIPELINE TEST")
    print("=" * 60)

    # Load
    df = load_data(
        DATA_PATH
    )

    print(
        f"\nRaw shape: {df.shape}"
    )

    # Preprocess
    df = preprocess_data(
        df,
        target_col="Churn",
    )

    print(
        f"After preprocessing: "
        f"{df.shape}"
    )

    # Features
    processed = build_features(
        df,
        target_col="Churn",
    )

    print(
        f"After feature engineering: "
        f"{processed.shape}"
    )

    # Checks
    assert "Churn" in processed.columns

    assert processed["Churn"].isin(
        [0, 1]
    ).all()

    feature_columns = processed.drop(
        columns=["Churn"]
    )

    assert not feature_columns.isna().any().any()

    assert feature_columns.select_dtypes(
        include=["object", "string"]
    ).shape[1] == 0

    print(
        "\n✅ PHASE 1 PASSED"
    )


if __name__ == "__main__":
    main()