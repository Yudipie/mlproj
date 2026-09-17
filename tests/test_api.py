"""Tests for the churn prediction FastAPI service."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    # Using TestClient as a context manager runs the app's lifespan
    # (startup) handler, which loads the trained model artifacts.
    with TestClient(app) as c:
        yield c

VALID_PAYLOAD = {
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
    "TotalCharges": 29.85,
}


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_valid_request_returns_expected_shape(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()

    assert "churn_probability" in body
    assert "churn_predicted" in body
    assert "model_used" in body

    assert isinstance(body["churn_probability"], float)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_predicted"], bool)
    assert isinstance(body["model_used"], str)


def test_predict_missing_required_field_returns_422(client):
    incomplete_payload = VALID_PAYLOAD.copy()
    del incomplete_payload["tenure"]

    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert any(err["field"] == "tenure" for err in body["errors"])


def test_predict_invalid_field_value_returns_422(client):
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["tenure"] = -5  # violates ge=0 constraint

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_invalid_categorical_value_returns_422(client):
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["Contract"] = "Lifetime"  # not one of the allowed Literal values

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
