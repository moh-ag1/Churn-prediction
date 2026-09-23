import pandas as pd


CUSTOMER_ID_COLUMNS = [
    "customerID",
    "CustomerID",
    "customer_id",
]


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
) -> pd.DataFrame:
    """
    Clean the raw Telco dataset.

    Responsibilities:
    - Strip column names.
    - Strip string values.
    - Remove customer ID.
    - Convert target Yes/No to 1/0.
    - Convert numeric columns.
    - Handle missing numeric values.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    df = df.copy()

    # Clean column names.
    df.columns = df.columns.astype(str).str.strip()

    # Strip whitespace from object/string columns.
    object_columns = df.select_dtypes(include=["object"]).columns

    for col in object_columns:
        df[col] = df[col].astype("string").str.strip()

    # Remove customer ID.
    for col in CUSTOMER_ID_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Target processing.
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found."
        )

    if df[target_col].dtype == "object" or str(df[target_col].dtype) == "string":
        target = df[target_col].astype("string").str.strip()

        invalid_targets = sorted(
            target.dropna().loc[~target.dropna().isin(["Yes", "No"])].unique()
        )

        if invalid_targets:
            raise ValueError(
                f"Unexpected target values: {invalid_targets}"
            )

        df[target_col] = target.map({
            "No": 0,
            "Yes": 1,
        })

    # Force target to numeric.
    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="coerce",
    )

    # TotalCharges is commonly read as object because of blank values.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

    # Known numeric columns.
    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # Fill numeric missing values.
    numeric_columns_in_df = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns_in_df] = (
        df[numeric_columns_in_df]
        .fillna(0)
    )

    # SeniorCitizen should be integer.
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = (
            df["SeniorCitizen"]
            .astype(int)
        )

    # Final target validation.
    if df[target_col].isna().any():
        raise ValueError(
            f"Target column '{target_col}' contains missing values."
        )

    df[target_col] = df[target_col].astype(int)

    invalid_target_values = set(
        df[target_col].unique()
    ) - {0, 1}

    if invalid_target_values:
        raise ValueError(
            f"Target must contain only 0/1. "
            f"Found: {invalid_target_values}"
        )

    return df