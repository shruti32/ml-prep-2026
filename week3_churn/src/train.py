"""
train.py — Week 3 Telco Churn: Optuna Hyperparameter Search + MLflow Tracking
------------------------------------------------------------------------------
Runs an Optuna study that searches over XGBoost and LightGBM hyperparameters,
evaluates each trial via 5-fold stratified CV, prunes poor trials early, and
logs everything to a local MLflow tracking server.

Usage:
    # Make sure MLflow is running first:
    #   docker-compose up mlflow
    #
    python week3_churn/src/train.py [--trials 30] [--experiment telco-churn]

After running, open http://localhost:5000 to explore results.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from pipeline import build_pipeline, encode_target

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH    = pathlib.Path(__file__).parents[1] / "data" / "telco_churn.csv"
ARTIFACT_DIR = pathlib.Path(__file__).parents[1] / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load Telco Churn CSV and return (X, y)."""
    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=["Churn"])
    y  = encode_target(df["Churn"])
    log.info("Loaded data: %d rows, %d features, %.1f%% churn",
             len(df), X.shape[1], y.mean() * 100)
    return X, y


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X_train: pd.DataFrame, y_train: np.ndarray):
    """
    Returns the Optuna objective function (closure over training data).

    Why a closure?
    The objective must accept only `trial` as its argument (Optuna's API),
    but it also needs access to X_train and y_train. A closure captures
    those at definition time cleanly, without globals.

    Pruning strategy
    ----------------
    We run StratifiedKFold(n_splits=5) and report each fold's ROC-AUC to
    Optuna via trial.report(). MedianPruner compares the intermediate values
    against completed trials and prunes if the trial looks unpromising.
    n_warmup_steps=2 means the first 2 folds are never pruned (we need at
    least some signal before pruning).
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:

        # ── 1. Suggest model type ────────────────────────────────────────────
        model_name = trial.suggest_categorical("model", ["xgb", "lgbm"])

        # ── 2. Suggest hyperparameters conditional on model type ─────────────
        if model_name == "xgb":
            # scale_pos_weight compensates for class imbalance.
            # Rule of thumb: n_negative / n_positive
            neg_pos_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

            clf = XGBClassifier(
                n_estimators        = trial.suggest_int("n_estimators", 100, 800),
                max_depth           = trial.suggest_int("max_depth", 3, 9),
                learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample           = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree    = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight    = trial.suggest_int("min_child_weight", 1, 10),
                reg_alpha           = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda          = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                scale_pos_weight    = trial.suggest_float(
                    "scale_pos_weight", neg_pos_ratio * 0.5, neg_pos_ratio * 1.5
                ),
                eval_metric         = "logloss",
                random_state        = 42,
                n_jobs              = -1,
                verbosity           = 0,
            )

        else:  # lgbm
            clf = LGBMClassifier(
                n_estimators        = trial.suggest_int("n_estimators", 100, 800),
                max_depth           = trial.suggest_int("max_depth", 3, 9),
                learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves          = trial.suggest_int("num_leaves", 20, 150),
                min_child_samples   = trial.suggest_int("min_child_samples", 5, 100),
                subsample           = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree    = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_alpha           = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda          = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                class_weight        = "balanced",
                random_state        = 42,
                n_jobs              = -1,
                verbose             = -1,
            )

        # ── 3. Build full pipeline with this classifier ──────────────────────
        pipeline = build_pipeline(clf)

        # ── 4. Run CV fold-by-fold so we can report intermediate values ──────
        fold_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val   = X_train.iloc[val_idx]
            y_fold_val   = y_train[val_idx]

            pipeline.fit(X_fold_train, y_fold_train)
            probs = pipeline.predict_proba(X_fold_val)[:, 1]
            score = roc_auc_score(y_fold_val, probs)
            fold_scores.append(score)

            # Report intermediate value — this is what enables pruning
            trial.report(score, step=fold_idx)

            # Prune if this trial looks worse than the median of completed trials
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """
    Compute a full set of classification metrics on a held-out test set.
    Returns a dict suitable for mlflow.log_metrics().
    """
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "test_roc_auc":          round(roc_auc_score(y_test, probs), 4),
        "test_avg_precision":    round(average_precision_score(y_test, probs), 4),
        "test_f1":               round(f1_score(y_test, preds), 4),
        "test_precision":        round(precision_score(y_test, preds), 4),
        "test_recall":           round(recall_score(y_test, preds), 4),
    }


# ── MLflow callback ───────────────────────────────────────────────────────────

def mlflow_callback(study: optuna.Study, trial: optuna.Trial) -> None:
    """
    Optuna callback: logs each completed trial as a child MLflow run.

    Pattern: one parent run for the whole study, one child run per trial.
    This keeps the MLflow UI clean — you can compare all trials in one view.
    """
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return  # don't log pruned or failed trials as runs

    with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("cv_roc_auc", trial.value)
        mlflow.log_metric("trial_number", trial.number)


# ── Main training loop ────────────────────────────────────────────────────────

def train(n_trials: int = 30, experiment_name: str = "telco-churn") -> None:

    # ── Setup ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    X, y = load_data()

    # Hold out 20% as a final test set — never seen during Optuna search
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    log.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ── Optuna study ─────────────────────────────────────────────────────────
    #
    # MedianPruner:
    #   n_startup_trials=5  → don't prune until we have 5 completed trials
    #                          (need enough data to compute a median)
    #   n_warmup_steps=2    → never prune before fold 2 within a trial
    #                          (first 2 folds don't count toward pruning)
    #
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study  = optuna.create_study(direction="maximize", pruner=pruner)

    objective = make_objective(X_train, y_train)

    # ── Parent MLflow run wraps the entire study ──────────────────────────────
    with mlflow.start_run(run_name="optuna_study") as parent_run:

        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("churn_rate_train", round(y_train.mean(), 3))

        log.info("Starting Optuna study — %d trials", n_trials)
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[mlflow_callback],
            show_progress_bar=True,
        )

        # ── Study summary ────────────────────────────────────────────────────
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

        log.info("Study finished: %d completed, %d pruned", len(completed), len(pruned))
        log.info("Best trial #%d  CV ROC-AUC: %.4f", study.best_trial.number, study.best_value)
        log.info("Best params: %s", study.best_params)

        mlflow.log_metric("best_cv_roc_auc", study.best_value)
        mlflow.log_metric("trials_completed", len(completed))
        mlflow.log_metric("trials_pruned", len(pruned))
        mlflow.log_param("best_trial_number", study.best_trial.number)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

        # ── Retrain best model on full training set ───────────────────────────
        log.info("Retraining best model on full training set...")

        best_model_name = study.best_params["model"]
        best_params     = {k: v for k, v in study.best_params.items() if k != "model"}

        if best_model_name == "xgb":
            best_clf = XGBClassifier(
                **best_params,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        else:
            best_clf = LGBMClassifier(
                **best_params,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        best_pipeline = build_pipeline(best_clf)
        best_pipeline.fit(X_train, y_train)

        # ── Evaluate on held-out test set ─────────────────────────────────────
        test_metrics = evaluate_model(best_pipeline, X_test, y_test)
        mlflow.log_metrics(test_metrics)

        log.info("Test set metrics:")
        for k, v in test_metrics.items():
            log.info("  %-25s %.4f", k, v)

        # ── Log model artifact ────────────────────────────────────────────────
        # Logged to MLflow so you can load it later with:
        #   mlflow.sklearn.load_model("runs:/<run_id>/best_model")
        mlflow.sklearn.log_model(
            sk_model       = best_pipeline,
            artifact_path  = "best_model",
            registered_model_name = "telco-churn-pipeline",
            input_example  = X_test.head(5),
        )

        # Also save a local pickle for quick loading during Sunday's exercise
        local_path = ARTIFACT_DIR / "best_pipeline.pkl"
        with open(local_path, "wb") as f:
            pickle.dump(best_pipeline, f)
        mlflow.log_artifact(str(local_path))
        log.info("Model saved locally → %s", local_path)

        log.info(
            "MLflow parent run: %s  (view at http://localhost:5000/#/experiments)",
            parent_run.info.run_id[:8],
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telco Churn — Optuna + MLflow training")
    parser.add_argument("--trials",     type=int, default=30,            help="Number of Optuna trials")
    parser.add_argument("--experiment", type=str, default="telco-churn", help="MLflow experiment name")
    args = parser.parse_args()

    train(n_trials=args.trials, experiment_name=args.experiment)
