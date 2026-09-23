from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained churn model and return metrics.
    """

    # Get predictions
    y_pred = model.predict(X_test)

    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
    ).ravel()

    # Print results
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    # IMPORTANT: return the dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }