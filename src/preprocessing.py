"""Reusable data cleaning, encoding, and feature engineering for the Telco
Customer Churn dataset.

The logic here mirrors the decisions documented and prototyped in
notebooks/exploration.ipynb:
- TotalCharges blank strings (0-tenure customers) are coerced to numeric
  and imputed with 0.
- customerID is dropped (unique identifier, no predictive signal / leakage risk).
- Binary Yes/No columns are mapped to 0/1 directly.
- Nominal multi-level categoricals are one-hot encoded.
- Two derived features are engineered: tenure_bucket and total_services.
"""
import pandas as pd

BINARY_YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NOMINAL_COLS = [
    "gender",
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

TENURE_BUCKET_BINS = [-1, 12, 24, 48, float("inf")]
TENURE_BUCKET_LABELS = ["0-12", "13-24", "25-48", "49+"]


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning + feature engineering to a raw feature dataframe
    (no `Churn` column required). Shared by both training
    (`load_and_preprocess`) and inference (`api/main.py`), so the exact
    same logic runs in both places and there's no train/serve skew.
    """
    df = df.copy()

    # --- Missing values: TotalCharges blank strings -> NaN -> 0 for
    # customers with 0 tenure (not yet billed, not a data error). ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- Leakage: drop unique identifier, if present. ---
    df = df.drop(columns=["customerID"], errors="ignore")

    # --- Feature engineering ---
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=TENURE_BUCKET_BINS, labels=TENURE_BUCKET_LABELS
    ).astype(str)

    df["total_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)

    # --- Binary Yes/No columns -> 0/1 ---
    for col in BINARY_YES_NO_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal categoricals (including engineered tenure_bucket)."""
    nominal_cols_present = NOMINAL_COLS + ["tenure_bucket"]
    return pd.get_dummies(df, columns=nominal_cols_present, drop_first=False)


def load_and_preprocess(raw_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the raw Telco churn CSV and return (X, y) ready for modeling.

    Parameters
    ----------
    raw_path : str
        Path to the raw Telco-Customer-Churn.csv file.

    Returns
    -------
    X : pd.DataFrame
        Cleaned, encoded, feature-engineered predictors.
    y : pd.Series
        Binary target (1 = churned, 0 = retained).
    """
    df = pd.read_csv(raw_path)

    y = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["Churn"])

    df = clean_and_engineer(df)
    X = encode_features(df)

    return X, y


def preprocess_for_inference(raw_features: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Run the exact training-time cleaning/encoding on a raw single- or
    multi-row feature dataframe (as received by the API, no `Churn`
    column), then align the resulting columns to the feature set the
    trained model expects — one-hot encoding a single row can only produce
    columns for the categories present in that row, so missing dummy
    columns are added back as 0 and column order is fixed to match
    training.
    """
    df = clean_and_engineer(raw_features)
    X = encode_features(df)
    return X.reindex(columns=feature_names, fill_value=0)


if __name__ == "__main__":
    X, y = load_and_preprocess("data/raw/Telco-Customer-Churn.csv")
    print("X shape:", X.shape)
    print("y distribution:\n", y.value_counts(normalize=True))

    processed = X.copy()
    processed["Churn"] = y
    processed.to_csv("data/processed/churn_processed.csv", index=False)
    print("Saved to data/processed/churn_processed.csv")
