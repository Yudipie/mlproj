"""Load the raw Telco churn dataset and print a profile summary."""
import pandas as pd

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("DTYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("MISSING VALUES PER COLUMN")
    print("=" * 60)
    # TotalCharges is stored as object with some blank strings for
    # customers with 0 tenure — coerce to numeric to surface those as NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values detected (post TotalCharges coercion)")
    print("\nFull missing count per column:")
    print(missing)

    print("\n" + "=" * 60)
    print("TARGET CLASS DISTRIBUTION (Churn)")
    print("=" * 60)
    print(df["Churn"].value_counts())
    print(df["Churn"].value_counts(normalize=True).round(4) * 100, "%")


if __name__ == "__main__":
    main()
