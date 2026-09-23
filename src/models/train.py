from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.build_features import (
    TelcoFeatureTransformer,
)


class ThresholdChurnModel:
    """
    Complete production model.

    It contains:
        feature preprocessing
        +
        XGBoost
        +
        production probability threshold
    """

    def __init__(
        self,
        model_params: Dict,
        threshold: float = 0.35,
    ):
        self.model_params = dict(model_params)
        self.threshold = float(threshold)

        self.pipeline = Pipeline(
            steps=[
                (
                    "features",
                    TelcoFeatureTransformer(),
                ),
                (
                    "classifier",
                    XGBClassifier(
                        **self.model_params
                    ),
                ),
            ]
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
    ):
        return self.pipeline.predict_proba(X)

    def predict(
        self,
        X: pd.DataFrame,
    ):
        probabilities = self.predict_proba(X)[:, 1]

        return (
            probabilities >= self.threshold
        ).astype(int)

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        predictions = self.predict(X)

        return float(
            np.mean(predictions == np.asarray(y))
        )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Dict,
    threshold: float = 0.35,
) -> ThresholdChurnModel:

    model = ThresholdChurnModel(
        model_params=model_params,
        threshold=threshold,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model