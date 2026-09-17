"""FastAPI service serving the trained churn prediction model."""
import json
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import ChurnFeatures, HealthResponse, PredictionResponse
from src.preprocessing import preprocess_for_inference

MODELS_DIR = "models"

# Populated at startup by load_model_artifacts().
_state = {"model": None, "scaler": None, "feature_names": None, "model_name": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_artifacts()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability from account/service attributes.",
    version="1.0.0",
    lifespan=lifespan,
)


def load_model_artifacts():
    model_name_path = os.path.join(MODELS_DIR, "best_model_name.txt")
    if not os.path.exists(model_name_path):
        return  # no trained model yet — /health will report model_loaded=False

    with open(model_name_path) as f:
        model_name = f.read().strip()

    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"best_model_name.txt says '{model_name}' but {model_path} is missing. "
            "(PyTorch MLP models are saved as best_model.pt and are not yet supported "
            "by this serving code.)"
        )
    model = joblib.load(model_path)

    _state["model"] = model
    _state["scaler"] = scaler
    _state["feature_names"] = feature_names
    _state["model_name"] = model_name


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Return a clean, readable 422 instead of FastAPI's default verbose
    payload or an unhandled stack trace.
    """
    errors = [
        {"field": ".".join(str(p) for p in err["loc"][1:]), "message": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input.", "errors": errors},
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
        model_name=_state["model_name"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ChurnFeatures):
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py to train and save a model first.",
        )

    try:
        raw_df = pd.DataFrame([features.model_dump()])
        # Same clean/encode pipeline used at training time (src/preprocessing.py) —
        # this is what keeps train-time and serve-time feature construction in sync.
        X = preprocess_for_inference(raw_df, _state["feature_names"])

        model_name = _state["model_name"]
        if model_name == "XGBoost":
            proba = _state["model"].predict_proba(X.values)[:, 1][0]
        else:
            X_scaled = _state["scaler"].transform(X.values)
            proba = _state["model"].predict_proba(X_scaled)[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process input: {e}")

    return PredictionResponse(
        churn_probability=round(float(proba), 4),
        churn_predicted=bool(proba >= 0.5),
        model_used=model_name,
    )
