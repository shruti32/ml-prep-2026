"""
retrain_quick.py — Retrain churn model without MLflow server.
Saves best_pipeline.pkl to week3_churn/artifacts/.

Usage:
    cd week3_churn/src
    poetry run python retrain_quick.py
"""
import pathlib
import pickle

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
import pandas as pd

from pipeline import build_pipeline, encode_target

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH    = pathlib.Path(__file__).parents[1] / "data" / "telco_churn.csv"
ARTIFACT_DIR = pathlib.Path(__file__).parents[1] / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=["Churn"])
y  = encode_target(df["Churn"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    model_name = trial.suggest_categorical("model", ["xgb", "lgbm"])
    neg_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    if model_name == "xgb":
        clf = XGBClassifier(
            n_estimators     = trial.suggest_int("n_estimators", 100, 400),
            max_depth        = trial.suggest_int("max_depth", 3, 7),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            scale_pos_weight = neg_pos,
            eval_metric      = "logloss",
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        )
    else:
        clf = LGBMClassifier(
            n_estimators     = trial.suggest_int("n_estimators", 100, 400),
            max_depth        = trial.suggest_int("max_depth", 3, 7),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves       = trial.suggest_int("num_leaves", 20, 100),
            class_weight     = "balanced",
            random_state     = 42,
            n_jobs           = -1,
            verbose          = -1,
        )

    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        pipe = build_pipeline(clf)
        pipe.fit(X_train.iloc[train_idx], y_train[train_idx])
        prob = pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y_train[val_idx], prob))
    return float(np.mean(scores))

print("Running 5-trial Optuna search (no MLflow)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5, show_progress_bar=True)

print(f"Best CV ROC-AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Retrain on full training set with best params
best = study.best_params
model_name = best.pop("model")
if model_name == "xgb":
    neg_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = XGBClassifier(**best, scale_pos_weight=neg_pos, eval_metric="logloss",
                        random_state=42, n_jobs=-1, verbosity=0)
else:
    clf = LGBMClassifier(**best, class_weight="balanced",
                         random_state=42, n_jobs=-1, verbose=-1)

pipeline = build_pipeline(clf)
pipeline.fit(X_train, y_train)

test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
print(f"Test ROC-AUC: {test_auc:.4f}")

out = ARTIFACT_DIR / "best_pipeline.pkl"
with open(out, "wb") as f:
    pickle.dump(pipeline, f)
print(f"Saved → {out}")
