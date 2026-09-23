from __future__ import annotations

import argparse
import json

import optuna
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import (
    TelcoFeatureTransformer,
)


def tune_model(
    X,
    y,
    n_trials: int = 30,
):
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                700,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                8,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.7,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.7,
                1.0,
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),
            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                2.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                0.0,
                1.0,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                0.5,
                3.0,
            ),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        pipeline = Pipeline(
            steps=[
                (
                    "features",
                    TelcoFeatureTransformer(),
                ),
                (
                    "classifier",
                    XGBClassifier(
                        **params
                    ),
                ),
            ]
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="recall",
            n_jobs=1,
        )

        return scores.mean()

    sampler = optuna.samplers.TPESampler(
        seed=42
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    return study.best_params


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/raw/Telco-Customer-Churn.csv",
    )

    parser.add_argument(
        "--target",
        default="Churn",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--output",
        default="artifacts/best_params.json",
    )

    args = parser.parse_args()

    df = load_data(args.input)

    df = preprocess_data(
        df,
        target_col=args.target,
    )

    X = df.drop(
        columns=[args.target]
    )

    y = df[args.target]

    best_params = tune_model(
        X,
        y,
        n_trials=args.trials,
    )

    # Ensure parameters needed by XGBoost.
    best_params.update({
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    })

    output_path = args.output

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_params,
            file,
            indent=2,
        )

    print("\n✅ Best parameters:")
    print(
        json.dumps(
            best_params,
            indent=2,
        )
    )

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()