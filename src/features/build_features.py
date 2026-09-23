from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


BINARY_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]


MULTI_CATEGORY_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]


BINARY_MAPPINGS = {
    "gender": {
        "Female": 0,
        "Male": 1,
    },
    "Partner": {
        "No": 0,
        "Yes": 1,
    },
    "Dependents": {
        "No": 0,
        "Yes": 1,
    },
    "PhoneService": {
        "No": 0,
        "Yes": 1,
    },
    "PaperlessBilling": {
        "No": 0,
        "Yes": 1,
    },
}


def _make_one_hot_encoder():
    """
    Create a version-compatible OneHotEncoder.

    Newer sklearn uses sparse_output.
    Older sklearn uses sparse.
    """

    try:
        return OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
    except TypeError:
        return OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float64,
        )


class TelcoFeatureTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Deterministic feature transformer for the Telco dataset.

    This object is fitted during training and reused unchanged
    during inference.
    """

    def __init__(self):
        self.encoder = _make_one_hot_encoder()
        self.feature_columns_: List[str] = []

    def _prepare_dataframe(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        X = X.copy()

        # Remove target if somebody accidentally supplies it.
        if "Churn" in X.columns:
            X = X.drop(columns=["Churn"])

        # Remove customer ID.
        for column in [
            "customerID",
            "CustomerID",
            "customer_id",
        ]:
            if column in X.columns:
                X = X.drop(columns=[column])

        # Strip strings.
        object_columns = X.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in object_columns:
            X[column] = (
                X[column]
                .astype("string")
                .str.strip()
            )

        # Numeric values.
        for column in NUMERIC_COLUMNS:
            if column in X.columns:
                X[column] = pd.to_numeric(
                    X[column],
                    errors="coerce",
                ).fillna(0)

        # Binary values.
        for column in BINARY_COLUMNS:
            if column not in X.columns:
                X[column] = 0
                continue

            mapping = BINARY_MAPPINGS[column]

            X[column] = (
                X[column]
                .map(mapping)
                .fillna(0)
                .astype(float)
            )

        return X

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        X = self._prepare_dataframe(X)

        # Ensure expected numeric columns exist.
        for column in NUMERIC_COLUMNS:
            if column not in X.columns:
                X[column] = 0.0

        # Ensure expected categorical columns exist.
        for column in MULTI_CATEGORY_COLUMNS:
            if column not in X.columns:
                X[column] = "Unknown"

        self.encoder.fit(
            X[MULTI_CATEGORY_COLUMNS]
        )

        encoded_names = list(
            self.encoder.get_feature_names_out(
                MULTI_CATEGORY_COLUMNS
            )
        )

        self.feature_columns_ = (
            NUMERIC_COLUMNS
            + BINARY_COLUMNS
            + encoded_names
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ):
        X = self._prepare_dataframe(X)

        # Ensure columns exist.
        for column in NUMERIC_COLUMNS:
            if column not in X.columns:
                X[column] = 0.0

        for column in BINARY_COLUMNS:
            if column not in X.columns:
                X[column] = 0.0

        for column in MULTI_CATEGORY_COLUMNS:
            if column not in X.columns:
                X[column] = "Unknown"

        numeric_values = (
            X[NUMERIC_COLUMNS]
            .astype(float)
            .to_numpy()
        )

        binary_values = (
            X[BINARY_COLUMNS]
            .astype(float)
            .to_numpy()
        )

        categorical_values = (
            self.encoder
            .transform(
                X[MULTI_CATEGORY_COLUMNS]
            )
        )

        output = np.hstack(
            [
                numeric_values,
                binary_values,
                categorical_values,
            ]
        )

        return output.astype(np.float64)

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        if not self.feature_columns_:
            raise RuntimeError(
                "Transformer has not been fitted."
            )

        return np.asarray(
            self.feature_columns_,
            dtype=object,
        )


def build_features(
    df: pd.DataFrame,
    target_col: str = "Churn",
) -> pd.DataFrame:
    """
    Backwards-compatible helper.

    Fits the feature transformer and returns a fully encoded DataFrame.
    """

    transformer = TelcoFeatureTransformer()

    X = df.drop(
        columns=[target_col],
        errors="ignore",
    )

    transformer.fit(X)

    transformed = transformer.transform(X)

    result = pd.DataFrame(
        transformed,
        columns=transformer.get_feature_names_out(),
        index=df.index,
    )

    if target_col in df.columns:
        result[target_col] = df[target_col].values

    return result