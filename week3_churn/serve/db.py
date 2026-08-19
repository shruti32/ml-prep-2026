"""
db.py — Postgres connection + prediction logging
"""

import os

import psycopg2

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://churn:churn123@localhost:5432/churn_predictions",
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                SERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    customer_id       TEXT NOT NULL,
    churn_probability FLOAT NOT NULL,
    churn_prediction  BOOLEAN NOT NULL,
    model_version     TEXT NOT NULL
);
"""


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def init_db() -> None:
    """Create the predictions table if it doesn't exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()


def log_prediction(
    customer_id: str,
    probability: float,
    prediction: bool,
    model_version: str,
) -> None:
    """Insert one prediction row into the predictions table."""
    sql = """
    INSERT INTO predictions (customer_id, churn_probability, churn_prediction, model_version)
    VALUES (%s, %s, %s, %s)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (customer_id, probability, prediction, model_version))
        conn.commit()
