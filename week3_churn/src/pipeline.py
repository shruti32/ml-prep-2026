"""
pipeline.py — Week 3 Telco Churn: sklearn Pipeline + Custom Transformers
-------------------------------------------------------------------------
Defines:
  • TotalChargesFixup   — coerces TotalCharges object column to float
  • TenureGrouper       — bins tenure into lifecycle groups (new/growing/established/loyal)
  • ChargesRatioAdder   — engineers MonthlyCharges / TotalCharges ratio feature
  • build_pipeline()    — assembles the full sklearn Pipeline for any classifier

Usage:
    from week3_churn.src.pipeline import build_pipeline
    from xgboost import XGBClassifier

    pipeline = build_pipeline(XGBClassifier())
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict_proba(X_test)[:, 1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Column groups ─────────────────────────────────────────────────────────────
# Defined here so they're a single source of truth for both the pipeline
# and any tests.

DROP_COLS = ["customerID"]

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_ratio",      # engineered
]

# SeniorCitizen is 0/1 int but semantically categorical — treat as category.
CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",       # engineered
]


# ── Custom Transformers ───────────────────────────────────────────────────────

class TotalChargesFixup(BaseEstimator, TransformerMixin):
    """
    Coerces TotalCharges from object → float.

    The IBM dataset stores TotalCharges as a string column. New customers
    (tenure = 0) have a single space " " instead of "0". pd.to_numeric with
    errors='coerce' turns those into NaN, which we then fill with 0.

    Must run BEFORE any numeric processing.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "TotalChargesFixup":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["TotalCharges"] = (
            pd.to_numeric(X["TotalCharges"].astype(str).str.strip(), errors="coerce")
            .fillna(0.0)
        )
        return X


class TenureGrouper(BaseEstimator, TransformerMixin):
    """
    Bins the continuous `tenure` column into lifecycle groups.

        new         0–12 months   (highest churn risk ~50%)
        growing    13–24 months
        established 25–48 months
        loyal      49–72 months   (lowest churn risk ~10%)

    Why bucket instead of leaving as numeric?
    - The churn-vs-tenure relationship is non-linear and step-like.
    - Bucketing makes this pattern explicit and adds a categorical signal
      that complements the raw numeric tenure column.
    - Tree models can learn this anyway, but OHE of the bucket boosts
      weaker learners and makes SHAP explanations cleaner.

    Adds `tenure_group` column; leaves `tenure` untouched.
    """

    BINS   = [0, 12, 24, 48, 72]
    LABELS = ["new", "growing", "established", "loyal"]

    def fit(self, X: pd.DataFrame, y=None) -> "TenureGrouper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=self.BINS,
            labels=self.LABELS,
            include_lowest=True,
        ).astype(str)   # cast to str so OHE treats it as a regular category
        return X


class ChargesRatioAdder(BaseEstimator, TransformerMixin):
    """
    Adds `charges_ratio` = MonthlyCharges / TotalCharges.

    Intuition:
    - A high ratio means TotalCharges is still low relative to monthly spend
      → the customer is recent (new customers also churn a lot).
    - A low ratio means TotalCharges has accumulated → long-tenure customer.
    - This captures a different signal than raw tenure because it reflects
      spend trajectory, not just time.

    Division by zero is handled by replacing 0 with 1 before dividing.
    Run AFTER TotalChargesFixup so TotalCharges is already float.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ChargesRatioAdder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        safe_total = X["TotalCharges"].replace(0.0, 1.0)
        X["charges_ratio"] = (X["MonthlyCharges"] / safe_total).astype(np.float32)
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Drops columns that should not enter the feature matrix (e.g. customerID).
    Accepts a list of column names; silently skips any that don't exist.
    """

    def __init__(self, columns: list[str] = DROP_COLS):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "DropColumns":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in self.columns if c in X.columns]
        return X.drop(columns=to_drop)


# ── Pipeline factory ──────────────────────────────────────────────────────────

def build_pipeline(classifier) -> Pipeline:
    """
    Assemble the full sklearn Pipeline.

    Preprocessing flow
    ------------------
    Raw DataFrame
        → DropColumns          (remove customerID)
        → TotalChargesFixup    (coerce TotalCharges to float)
        → ChargesRatioAdder    (add charges_ratio feature)
        → TenureGrouper        (add tenure_group feature)
        → ColumnTransformer
              numeric branch:  StandardScaler on NUMERIC_COLS
              category branch: OneHotEncoder on CATEGORICAL_COLS
        → classifier

    Parameters
    ----------
    classifier : sklearn-compatible estimator
        Any classifier with fit/predict_proba interface (XGBClassifier,
        LGBMClassifier, RandomForestClassifier, …).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """

    # Numeric: scale so regularisation-sensitive models benefit.
    # Tree ensembles don't need scaling but it doesn't hurt them either.
    numeric_transformer = StandardScaler()

    # Categorical: OHE with handle_unknown='ignore' so unseen categories
    # at inference time don't crash the pipeline.
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",      # drop any columns not listed above
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            ("drop_id",      DropColumns(DROP_COLS)),
            ("fix_charges",  TotalChargesFixup()),
            ("add_ratio",    ChargesRatioAdder()),
            ("add_tenure_grp", TenureGrouper()),
            ("preprocessor", preprocessor),
            ("clf",          classifier),
        ]
    )

    # Enable pandas output on the ColumnTransformer only (not the full pipeline).
    # Custom transformers already return DataFrames; the classifier accepts
    # numpy arrays from ColumnTransformer just fine.
    pipeline.named_steps["preprocessor"].set_output(transform="pandas")

    return pipeline


# ── Label encoder helper ──────────────────────────────────────────────────────

def encode_target(series: pd.Series) -> np.ndarray:
    """Convert 'Yes'/'No' Churn column to 1/0 numpy array."""
    return (series.str.strip() == "Yes").astype(int).values


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib

    from sklearn.dummy import DummyClassifier

    DATA_PATH = pathlib.Path(__file__).parents[2] / "data" / "telco_churn.csv"

    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=["Churn"])
    y  = encode_target(df["Churn"])

    pipe = build_pipeline(DummyClassifier(strategy="most_frequent"))
    pipe.fit(X, y)
    preds = pipe.predict(X)

    n_features = pipe.named_steps["preprocessor"].transform(
        pipe[:-1].transform(X)
    ).shape[1]

    print("✓ Pipeline smoke-test passed")
    print(f"  Input shape  : {X.shape}")
    print(f"  Feature count after preprocessing: {n_features}")
    print(f"  Prediction sample: {preds[:5]}")
    print(f"  Unique predictions: {set(preds)}")
