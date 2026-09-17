# Churn Prediction

Portfolio project for a Machine Learning Engineering Intern application,
demonstrating the full ML lifecycle: data preparation, model evaluation,
and building an ML service — not just training a model in a notebook.

See [PROBLEM.md](PROBLEM.md) for the business problem statement and dataset
description.

## Project Structure

```
data/raw/          raw dataset (gitignored, fetched via script)
data/processed/    cleaned/feature-engineered datasets
notebooks/         exploration, cleaning, EDA
src/               reusable pipeline code (data prep, training, evaluation)
api/               FastAPI service that serves the trained model
tests/             pytest unit tests for the API and pipeline
models/            saved/trained model artifacts
scripts/           standalone utility scripts (env check, etc.)

Dockerfile, .dockerignore, docker-compose.yml — containerize and run the API
```

## Setup

On a fresh machine:

```
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
.venv\Scripts\activate        # Windows (cmd/PowerShell)
source .venv/bin/activate     # macOS/Linux

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Verify the environment (imports every dependency, prints versions)
python scripts/check_env.py
```

`requirements.txt` pins exact versions (pandas, numpy, scikit-learn,
xgboost, torch, fastapi, uvicorn[standard], pydantic, pytest, httpx,
jupyter, matplotlib, seaborn) for reproducibility.

## Get the data

```
python src/fetch_data.py
```

Downloads to `data/raw/Telco-Customer-Churn.csv` (via Kaggle API if
`kaggle.json` credentials are configured, otherwise from a public CSV
mirror).

## EDA & Cleaning Summary

Full reasoning lives in [notebooks/exploration.ipynb](notebooks/exploration.ipynb);
the short version:

- **Missing values**: `TotalCharges` looks complete under a naive `isna()`
  check because its 11 missing values are stored as blank strings, not
  `NaN`. All 11 belong to customers with `tenure == 0` (not yet billed), so
  they're coerced to numeric and imputed with `0` rather than dropped.
- **Class balance**: ~73.5% No-churn / ~26.5% Churn — imbalanced enough that
  plain accuracy is misleading (predicting "No churn" for everyone scores
  ~73.5% while catching zero at-risk customers), so modeling uses
  stratified splits and precision/recall/F1/ROC-AUC.
- **Key signal**: churned customers skew toward low `tenure` and higher
  `MonthlyCharges`; `Contract_Month-to-month` is the strongest single
  correlate with churn (no long-term commitment = free to leave anytime),
  while `Contract_Two year` and long tenure are the strongest signals
  *against* churn.
- **Leakage**: `customerID` is dropped (unique identifier, no signal). No
  other column is derived from the outcome itself.
- **Encoding**: true binary Yes/No columns are mapped to 0/1 directly;
  nominal multi-level categoricals (contract type, internet service,
  payment method, etc.) are one-hot encoded rather than label-encoded, to
  avoid inventing a false ordinal relationship.
- **Engineered features**: `tenure_bucket` (0-12/13-24/25-48/49+ months)
  and `total_services` (count of subscribed add-on services, as an
  "engagement depth" proxy).
- **Reusable pipeline**: all of the above is implemented in
  `src/preprocessing.py` as `load_and_preprocess(raw_path)`, producing
  `data/processed/churn_processed.csv` (7,043 rows × 46 columns after
  encoding).

## Running the API

Train a model first if `models/best_model.joblib` doesn't exist yet:

```
python -m src.train
```

Then start the API server:

```
uvicorn api.main:app --reload
```

Docs are auto-generated at `http://127.0.0.1:8000/docs`.

### Example request

```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

Response:

```json
{"churn_probability":0.8113,"churn_predicted":true,"model_used":"LogisticRegression"}
```

(This is a real response from a locally running instance — a new,
month-to-month, low-tenure customer is exactly the profile EDA flagged as
highest-risk.)

`GET /health` returns `{"status":"ok","model_loaded":true,"model_name":"LogisticRegression"}`.

### No train/serve skew

The API does not duplicate any cleaning/encoding logic. `api/main.py` calls
`src.preprocessing.preprocess_for_inference()`, which runs the exact same
`clean_and_engineer()` and `encode_features()` functions that
`load_and_preprocess()` uses during training (`src/preprocessing.py`), then
aligns the resulting one-hot columns to `models/feature_names.json` — a
single request can only produce dummy columns for the categories present in
that one row, so missing columns are added back as zeros in training's
column order before the model sees them.

### Tests

```
pytest tests/test_api.py -v
```

Covers: a valid request returns 200 with the expected response shape, a
request missing a required field returns 422, an out-of-range value (e.g.
negative `tenure`) returns 422, an invalid categorical value returns 422,
and `/health` returns 200.

## Run with Docker

Requires a trained model already present in `models/` (`python -m src.train`)
— the image copies `models/` in at build time rather than training inside
the container.

### Build

```
docker build -t churn-api .
```

### Run

```
docker run -p 8000:8000 churn-api
```

### Test the running container

```
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
  }'
```

Expected: `/health` returns 200 with `"model_loaded":true`; `/predict`
returns 200 with a `churn_probability`, matching the values shown in the
[Example request](#example-request) section above (same model, same input —
the container just serves it on a different host).

> **Note:** these commands have not been executed against a live Docker
> daemon in this environment (Docker isn't installed here) — verify them
> locally after installing Docker Desktop / the Docker CLI.

### Optional: run with MLflow via Docker Compose

`docker-compose.yml` starts the API alongside a local MLflow tracking
server (`ghcr.io/mlflow/mlflow`) on port 5000, so both can be started
together:

```
docker compose up
```

This brings up the API on `:8000` and MLflow's UI/tracking server on
`:5000`. Note the training pipeline (`src/train.py`) does not currently log
runs to MLflow — this compose file only provisions the tracking server as
infrastructure for that to be wired in later.

## Roadmap

- [x] Project scaffolding + dataset ingestion
- [x] Data cleaning & feature engineering (missing values, encoding, leakage checks)
- [x] EDA (class balance, key feature distributions)
- [x] Baseline model (logistic regression) + stratified k-fold evaluation
- [x] XGBoost + optional PyTorch MLP, model comparison, error analysis (see [RESULTS.md](RESULTS.md))
- [x] FastAPI service (`/predict`, `/health`) wrapping the best model
- [x] pytest unit tests for the API
- [x] Containerize the API with Docker (build/run verification pending local Docker install — see [Run with Docker](#run-with-docker))
