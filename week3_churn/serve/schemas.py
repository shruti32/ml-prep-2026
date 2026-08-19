"""
schemas.py — Pydantic v2 request/response models for the Churn Prediction API
"""
from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int          # 0 or 1
    Partner: str                # Yes / No
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool      # True if probability >= threshold
    threshold: float
    model_version: str
