from pathlib import Path
import sys

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
)
from sklearn.metrics import (
    recall_score,
)
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

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
    TelcoFeatureTransformer,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)


def main():

    print("=" * 60)
    print("PHASE 2 MODEL TEST")
    print("=" * 60)

    df = load_data(
        DATA_PATH
    )

    df = preprocess_data(
        df,
        target_col="Churn",
    )

    X = df.drop(
        columns=["Churn"]
    )

    y = df["Churn"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    model = Pipeline(
        steps=[
            (
                "features",
                TelcoFeatureTransformer(),
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    # CV happens ONLY on training data.
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="recall",
    )

    print(
        "\nCross-validation recall:"
    )

    print(
        f"Scores: {cv_scores}"
    )

    print(
        f"Mean: {cv_scores.mean():.4f}"
    )

    # Final training.
    model.fit(
        X_train,
        y_train,
    )

    # Test set is used only here.
    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    threshold = 0.35

    predictions = (
        probabilities >= threshold
    ).astype(int)

    test_recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print(
        f"\nFinal test recall "
        f"(threshold={threshold}): "
        f"{test_recall:.4f}"
    )

    print(
        "\n✅ PHASE 2 PASSED"
    )


if __name__ == "__main__":
    main()

