from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
import argparse
import json
import time

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.evaluate import evaluate_model
from src.models.train import train_model
from src.utils.validate_data import validate_telco_data


DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
}


def load_best_params(
    path: str | None,
):
    if not path:
        return DEFAULT_PARAMS.copy()

    params_path = Path(path)

    if not params_path.exists():
        raise FileNotFoundError(
            f"Parameter file not found: "
            f"{params_path}"
        )

    with open(
        params_path,
        "r",
        encoding="utf-8",
    ) as file:
        params = json.load(file)

    params.setdefault(
        "random_state",
        42,
    )

    params.setdefault(
        "n_jobs",
        -1,
    )

    params.setdefault(
        "eval_metric",
        "logloss",
    )

    return params


def main():

    parser = argparse.ArgumentParser(
        description="Telco Customer Churn ML pipeline"
    )

    parser.add_argument(
        "--input",
        default=(
            "data/raw/"
            "Telco-Customer-Churn.csv"
        ),
    )

    parser.add_argument(
        "--target",
        default="Churn",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--best-params",
        default=None,
        help=(
            "Optional JSON file produced by tune.py."
        ),
    )

    parser.add_argument(
        "--mlflow-uri",
        default=None,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Directories
    # ---------------------------------------------------------

    artifacts_dir = PROJECT_ROOT / "artifacts"
    processed_dir = (
        PROJECT_ROOT / "data" / "processed"
    )
    model_dir = artifacts_dir / "model"

    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # MLflow
    # ---------------------------------------------------------

    if args.mlflow_uri:
        tracking_uri = args.mlflow_uri
    else:
        mlflow_db = PROJECT_ROOT / "mlflow.db"
        tracking_uri = f"sqlite:///{mlflow_db}"
        
    mlflow.set_tracking_uri(tracking_uri)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    print("\n📥 Loading dataset...")

    raw_df = load_data(
        PROJECT_ROOT / args.input
    )

    print(
        f"Loaded {len(raw_df):,} rows "
        f"and {len(raw_df.columns)} columns."
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    valid, failures = (
        validate_telco_data(raw_df)
    )

    if not valid:
        raise ValueError(
            "Dataset validation failed."
        )

    # ---------------------------------------------------------
    # Preprocess
    # ---------------------------------------------------------

    print("\n🧹 Preprocessing...")

    df = preprocess_data(
        raw_df,
        target_col=args.target,
    )

    clean_path = (
        processed_dir
        / "telco_churn_clean.csv"
    )

    df.to_csv(
        clean_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    X = df.drop(
        columns=[args.target]
    )

    y = df[args.target]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=42,
            stratify=y,
        )
    )

    print(
        f"\nTrain samples: {len(X_train):,}"
    )

    print(
        f"Test samples:  {len(X_test):,}"
    )

    # ---------------------------------------------------------
    # Class imbalance
    # ---------------------------------------------------------

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "No positive churn examples in training data."
        )

    scale_pos_weight = (
        negative_count / positive_count
    )

    # ---------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------

    model_params = load_best_params(
        args.best_params
    )

    model_params[
        "scale_pos_weight"
    ] = scale_pos_weight

    # ---------------------------------------------------------
    # MLflow run
    # ---------------------------------------------------------

    with mlflow.start_run() as run:

        mlflow.log_params({
            "test_size": args.test_size,
            "threshold": args.threshold,
            "scale_pos_weight": scale_pos_weight,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })

        mlflow.log_params({
            f"model_{key}": value
            for key, value in model_params.items()
        })

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        print("\n🤖 Training model...")

        start_time = time.perf_counter()

        model = train_model(
            X_train,
            y_train,
            model_params=model_params,
            threshold=args.threshold,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Training completed in "
            f"{training_time:.2f}s"
        )

        mlflow.log_metric(
            "training_time_seconds",
            training_time,
        )

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        print("\n📊 Evaluating model...")

        start_time = time.perf_counter()

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        prediction_time = (
            time.perf_counter()
            - start_time
        )

        mlflow.log_metrics({
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
            and key not in {
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
            }
        })

        mlflow.log_metrics({
            "true_negative": metrics[
                "true_negative"
            ],
            "false_positive": metrics[
                "false_positive"
            ],
            "false_negative": metrics[
                "false_negative"
            ],
            "true_positive": metrics[
                "true_positive"
            ],
            "prediction_time_seconds": (
                prediction_time
            ),
        })

        print("\n📈 Metrics:")

        for key, value in metrics.items():
            print(
                f"  {key}: {value}"
            )

        # -----------------------------------------------------
        # Save processed feature dataset
        # -----------------------------------------------------

        from src.features.build_features import (
            build_features,
        )

        processed_features = build_features(
            df,
            target_col=args.target,
        )

        processed_path = (
            processed_dir
            / "telco_churn_processed.csv"
        )

        processed_features.to_csv(
            processed_path,
            index=False,
        )

        # -----------------------------------------------------
        # Save model locally
        # -----------------------------------------------------

        if model_dir.exists():
            import shutil

            shutil.rmtree(model_dir)

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # MLflow model
        # -----------------------------------------------------

        print(
            "\n💾 Saving MLflow model..."
        )

        signature = None

        try:
            from mlflow.models import (
                infer_signature,
            )

            sample_input = X_train.head(5)

            sample_output = model.predict(
                sample_input
            )

            signature = infer_signature(
                sample_input,
                sample_output,
            )

        except Exception as exc:
            print(
                "⚠️ Could not infer MLflow "
                f"signature: {exc}"
            )

        mlflow.sklearn.save_model(
            model,
            path=str(model_dir),
            signature=signature,
            serialization_format="cloudpickle",
        )

        # Log model to MLflow.
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            serialization_format="cloudpickle",
        )

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------

        metadata = {
            "threshold": args.threshold,
            "target": args.target,
            "features": list(X.columns),
            "model_params": model_params,
            "metrics": metrics,
            "run_id": run.info.run_id,
        }

        metadata_path = (
            artifacts_dir
            / "model_metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
                default=str,
            )

        mlflow.log_artifact(
            str(metadata_path),
            artifact_path="metadata",
        )

        print(
            "\n✅ Pipeline completed successfully."
        )

        print(
            f"MLflow run ID: "
            f"{run.info.run_id}"
        )

        print(
            f"Model saved to: "
            f"{model_dir}"
        )


if __name__ == "__main__":
    main()