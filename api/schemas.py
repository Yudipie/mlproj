"""Pydantic request/response schemas for the churn prediction API."""
from typing import Literal

from pydantic import BaseModel, Field


class ChurnFeatures(BaseModel):
    """Raw customer attributes, matching the Telco churn dataset's feature
    columns (everything except customerID and the Churn target). This is
    the same shape of data the training pipeline reads from the raw CSV —
    it gets run through the identical `clean_and_engineer` / `encode_features`
    functions from src/preprocessing.py before hitting the model.
    """

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 = senior citizen, 0 = not")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, description="Current monthly charge in dollars")
    TotalCharges: float = Field(ge=0, description="Total amount charged to date")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(description="Predicted probability of churn (0-1)")
    churn_predicted: bool = Field(description="Predicted label at a 0.5 decision threshold")
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
