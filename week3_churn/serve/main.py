"""
main.py — FastAPI serving layer for the Telco Churn model
---------------------------------------------------------
Endpoints:
  GET  /health    → liveness check
  POST /predict   → churn probability for a single customer

Usage (local, without Docker):
    cd week3_churn/serve
    DATABASE_URL=postgresql://churn:churn123@localhost:5432/churn_predictions \
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload

Usage (Docker Compose):
    docker-compose up --build churn-api
"""
from __future__ import annotations

import pathlib
import pickle
import sys
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

# Allow importing db/schemas when running from this directory directly
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from db import init_db, log_prediction
from schemas import ChurnResponse, CustomerFeatures

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH    = pathlib.Path(__file__).parents[1] / "artifacts" / "best_pipeline.pkl"
MODEL_VERSION = "1.0.0"
THRESHOLD     = 0.5   # tune via threshold_selection.py (Week 2 exercise)

pipeline = None


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    # Initialise DB table
    init_db()
    # Load model
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    print(f"✓ Model loaded from {MODEL_PATH}  (version={MODEL_VERSION})")
    yield
    # Nothing to clean up


app = FastAPI(
    title="Telco Churn Prediction API",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — returns 200 when model is loaded."""
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=ChurnResponse)
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    - Runs the full sklearn pipeline (custom transformers + OHE + scaler + classifier).
    - Logs the result to Postgres.
    - Returns probability, binary decision, and the threshold used.
    """
    # Build a single-row DataFrame — pipeline expects a DataFrame, not a dict
    df = pd.DataFrame([customer.model_dump()])

    try:
        prob = float(pipeline.predict_proba(df)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    prediction = prob >= THRESHOLD

    # Fire-and-forget DB log (synchronous is fine for this volume)
    try:
        log_prediction(
            customer_id=customer.customerID,
            probability=prob,
            prediction=prediction,
            model_version=MODEL_VERSION,
        )
    except Exception as exc:
        # Don't let a DB write failure break the API response
        print(f"⚠️  DB log failed: {exc}")

    return ChurnResponse(
        customer_id=customer.customerID,
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        threshold=THRESHOLD,
        model_version=MODEL_VERSION,
    )
