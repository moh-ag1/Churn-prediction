from typing import List, Tuple

import pandas as pd


REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


EXPECTED_VALUES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {
        "Month-to-month",
        "One year",
        "Two year",
    },
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


def validate_telco_data(
    df: pd.DataFrame,
) -> Tuple[bool, List[str]]:
    """
    Validate the raw Telco Customer Churn dataset.

    Returns:
        (success, list_of_failed_checks)
    """

    print("🔍 Starting data validation...")

    failures: List[str] = []

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        failures.append(
            f"Missing columns: {missing_columns}"
        )

    # Stop here if schema is fundamentally wrong.
    if missing_columns:
        print("❌ Data validation FAILED.")
        for failure in failures:
            print(f"   - {failure}")

        return False, failures

    # ---------------------------------------------------------
    # Empty dataset
    # ---------------------------------------------------------

    if df.empty:
        failures.append("Dataset is empty.")

    # ---------------------------------------------------------
    # Customer ID
    # ---------------------------------------------------------

    if df["customerID"].isna().any():
        failures.append(
            "customerID contains null values."
        )

    if df["customerID"].duplicated().any():
        failures.append(
            "customerID contains duplicate values."
        )

    # ---------------------------------------------------------
    # Categorical values
    # ---------------------------------------------------------

    for column, allowed_values in EXPECTED_VALUES.items():

        if column not in df.columns:
            continue

        values = set(
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        unexpected = values - allowed_values

        if unexpected:
            failures.append(
                f"{column} contains unexpected values: "
                f"{sorted(unexpected)}"
            )

    # ---------------------------------------------------------
    # Numeric columns
    # ---------------------------------------------------------

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    numeric_copy = df.copy()

    for column in numeric_columns:
        numeric_copy[column] = pd.to_numeric(
            numeric_copy[column],
            errors="coerce",
        )

    for column in numeric_columns:
        invalid_count = numeric_copy[column].isna().sum()

        # TotalCharges can legitimately have blank values
        # in the original Telco dataset, so don't fail on it.
        if column != "TotalCharges" and invalid_count > 0:
            failures.append(
                f"{column} contains "
                f"{invalid_count} non-numeric/null values."
            )

    # ---------------------------------------------------------
    # Ranges
    # ---------------------------------------------------------

    if (
        numeric_copy["SeniorCitizen"]
        .dropna()
        .isin([0, 1])
        .all()
        is False
    ):
        failures.append(
            "SeniorCitizen must contain only 0 or 1."
        )

    if (
        numeric_copy["tenure"].dropna() < 0
    ).any():
        failures.append(
            "tenure contains negative values."
        )

    if (
        numeric_copy["MonthlyCharges"].dropna() < 0
    ).any():
        failures.append(
            "MonthlyCharges contains negative values."
        )

    if (
        numeric_copy["TotalCharges"].dropna() < 0
    ).any():
        failures.append(
            "TotalCharges contains negative values."
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    total_checks = 1 + len(EXPECTED_VALUES) + len(numeric_columns)

    if failures:
        print(
            f"❌ Data validation FAILED: "
            f"{len(failures)} issue(s) found."
        )

        for failure in failures:
            print(f"   - {failure}")

        return False, failures

    print(
        f"✅ Data validation PASSED."
    )

    return True, []