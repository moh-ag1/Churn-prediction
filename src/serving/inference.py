from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import mlflow.pyfunc
import pandas as pd


MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "./artifacts/model",
)


def load_model():
    """
    Load the complete MLflow model.

    The model artifact already contains:
        preprocessing
        feature encoding
        XGBoost
        prediction threshold
    """

    model_path = Path(MODEL_DIR)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: "
            f"{model_path.resolve()}"
        )

    return mlflow.pyfunc.load_model(
        str(model_path)
    )


MODEL = load_model()


def predict(
    input_data: Dict,
) -> Dict:

    if not isinstance(input_data, dict):
        raise TypeError(
            "input_data must be a dictionary."
        )

    df = pd.DataFrame([input_data])

    prediction = MODEL.predict(df)

    # pyfunc may return a pandas Series,
    # numpy array, or list.
    prediction_value = int(
        prediction[0]
    )

    # Try to retrieve probability from the
    # underlying model if available.
    probability = None

    try:
        # The MLflow pyfunc wrapper may expose
        # the underlying model differently depending
        # on MLflow version, so probability is optional.
        underlying = getattr(
            MODEL,
            "_model_impl",
            None,
        )

        if underlying is not None:
            python_model = getattr(
                underlying,
                "python_model",
                None,
            )

            if python_model is not None:
                probability = float(
                    python_model
                    .predict_proba(df)[0][1]
                )
    except Exception:
        probability = None

    result = {
        "prediction": (
            "Likely to churn"
            if prediction_value == 1
            else "Not likely to churn"
        ),
        "churn": prediction_value,
    }

    if probability is not None:
        result["probability"] = round(
            probability,
            4,
        )

    return result