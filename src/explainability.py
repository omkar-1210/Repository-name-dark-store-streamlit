from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import shap

from .config import MODEL_PATH, SHAP_CACHE

def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 500):
    explainer = shap.TreeExplainer(model)
    X_sample = X.sample(n=max_samples, random_state=42) if len(X) > max_samples else X
    return explainer(X_sample)

def get_shap_for_row(model, X: pd.DataFrame, row_index: int):
    explainer = shap.TreeExplainer(model)
    return explainer(X.iloc[[row_index]])[0]

def get_cached_shap(model, X: pd.DataFrame, max_samples: int = 500):
    if SHAP_CACHE.exists() and MODEL_PATH.exists():
        if SHAP_CACHE.stat().st_mtime > MODEL_PATH.stat().st_mtime:
            return joblib.load(SHAP_CACHE)

    shap_values = compute_shap_values(model, X, max_samples=max_samples)
    joblib.dump(shap_values, SHAP_CACHE)
    return shap_values
