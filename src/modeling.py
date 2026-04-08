from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    LGBM_PARAMS,
    METRICS_PATH,
    MODEL_PATH,
    SPOILAGE_COST,
    STOCKOUT_COST,
    TARGET_COL,
)

def total_cost(pred: np.ndarray, actual: np.ndarray) -> float:
    spoilage = np.maximum(0, pred - actual) * SPOILAGE_COST
    stockout = np.maximum(0, actual - pred) * STOCKOUT_COST
    return float((spoilage + stockout).sum())


def train_lgbm_model(demand: pd.DataFrame) -> tuple[lgb.LGBMRegressor, dict, dict]:
    X = demand[FEATURE_COLS].copy()
    y = demand[TARGET_COL].copy()

    X["time_bucket"] = X["time_bucket"].astype("category")
    X["department"] = X["department"].astype("category")

    test_mask = X["order_dow"] == 6
    X_train = X[~test_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[~test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
        categorical_feature=CATEGORICAL_FEATURES,
    )

    y_pred = np.maximum(0, model.predict(X_test))
    y_baseline = X_test["demand_lag_1w"].values

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "bias": round(float(np.mean(y_pred - y_test)), 2),
        "baseline_mae": round(float(mean_absolute_error(y_test, y_baseline)), 2),
        "baseline_rmse": round(float(np.sqrt(mean_squared_error(y_test, y_baseline))), 2),
        "model_cost_total": round(float(total_cost(y_pred, y_test.values)), 2),
        "baseline_cost_total": round(float(total_cost(y_baseline, y_test.values)), 2),
    }
    metrics["savings_pct"] = round(
        ((metrics["baseline_cost_total"] - metrics["model_cost_total"]) / metrics["baseline_cost_total"] * 100)
        if metrics["baseline_cost_total"] > 0
        else 0.0,
        2,
    )

    artifacts = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_baseline": y_baseline,
    }

    return model, metrics, artifacts


def save_model_and_metrics(model: lgb.LGBMRegressor, metrics: dict) -> None:
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
