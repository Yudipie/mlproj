"""Fetch the Telco Customer Churn dataset into data/raw/.

Tries the Kaggle API first (if credentials are configured), then falls
back to a public CSV mirror.
"""
import os
import subprocess
import sys
import urllib.request

RAW_DIR = "data/raw"
TARGET_PATH = os.path.join(RAW_DIR, "Telco-Customer-Churn.csv")
MIRROR_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def have_kaggle_credentials() -> bool:
    return os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))


def fetch_via_kaggle() -> bool:
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "blastchar/telco-customer-churn",
                "-p", RAW_DIR, "--unzip",
            ],
            check=True,
        )
        downloaded = os.path.join(RAW_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        if os.path.exists(downloaded):
            os.replace(downloaded, TARGET_PATH)
        return os.path.exists(TARGET_PATH)
    except Exception as e:
        print(f"Kaggle download failed: {e}", file=sys.stderr)
        return False


def fetch_via_mirror() -> bool:
    try:
        urllib.request.urlretrieve(MIRROR_URL, TARGET_PATH)
        return os.path.exists(TARGET_PATH)
    except Exception as e:
        print(f"Mirror download failed: {e}", file=sys.stderr)
        return False


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    if have_kaggle_credentials():
        print("Kaggle credentials found, attempting Kaggle API download...")
        if fetch_via_kaggle():
            print(f"Downloaded via Kaggle API to {TARGET_PATH}")
            return
        print("Falling back to public mirror...")
    else:
        print("No Kaggle credentials found, using public mirror...")

    if fetch_via_mirror():
        print(f"Downloaded via mirror to {TARGET_PATH}")
    else:
        print("Failed to fetch dataset from both sources.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
