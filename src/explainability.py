import shap
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from src.model import load_processed_data, prepare_features

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "cache"

def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 500):
    """Compute SHAP values using TreeExplainer.
    
    TreeExplainer is the fastest SHAP method for tree-based models (LightGBM).
    We subsample to max_samples for performance — full dataset is 9.8K rows.
    
    Args:
        model: Trained LGBMRegressor
        X: Feature DataFrame (must have correct dtypes — categoricals as 'category')
        max_samples: Cap on rows to explain (500 is plenty for beeswarm)
    
    Returns:
        shap.Explanation object with .values, .base_values, .data, .feature_names
    """
    explainer = shap.TreeExplainer(model)
    
    # Subsample for performance
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X
    
    shap_values = explainer(X_sample)
    
    return shap_values

def get_shap_for_row(model, X: pd.DataFrame, row_index: int):
    """Compute SHAP values for a single data point (for waterfall plot).
    
    Args:
        model: Trained LGBMRegressor
        X: Full feature DataFrame
        row_index: Index of the row to explain
    
    Returns:
        shap.Explanation for a single row
    """
    explainer = shap.TreeExplainer(model)
    single_row = X.iloc[[row_index]]
    shap_value = explainer(single_row)
    return shap_value[0]  # Return single explanation, not array

def get_cached_shap(model, X: pd.DataFrame, max_samples: int = 500):
    """Load SHAP values from cache, or compute and cache them.
    
    Cache key: cache/shap_values.pkl
    The cache is invalidated if the model file is newer than the cache.
    
    Returns:
        shap.Explanation object
    """
    cache_path = _CACHE_DIR / "shap_values.pkl"
    model_path = config.MODEL_PATH
    
    # Check if cache is fresh
    if cache_path.exists() and model_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        model_mtime = model_path.stat().st_mtime
        if cache_mtime > model_mtime:
            return joblib.load(cache_path)
    
    # Compute and cache
    shap_values = compute_shap_values(model, X, max_samples)
    joblib.dump(shap_values, cache_path)
    
    return shap_values

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 5: SHAP Explainability")
    print("=" * 60)
    
    # Load model and data
    model = joblib.load(config.MODEL_PATH)
    df = load_processed_data()
    X, y = prepare_features(df)
    
    # Test cached SHAP
    print("\n[1/3] Computing SHAP values (500 samples)...")
    shap_vals = get_cached_shap(model, X)
    print(f"  Shape: {shap_vals.values.shape}")
    print(f"  Base value: {shap_vals.base_values[0]:.2f}")
    print(f"  Features: {shap_vals.feature_names}")
    
    # Test single row
    print("\n[2/3] Getting SHAP for row 0...")
    single = get_shap_for_row(model, X, 0)
    print(f"  Prediction contribution breakdown:")
    for name, val in zip(single.feature_names, single.values):
        print(f"    {name:20s} → {val:+.2f}")
    
    # Test that cache works
    print("\n[3/3] Testing cache hit...")
    shap_vals_2 = get_cached_shap(model, X)
    assert np.allclose(shap_vals.values, shap_vals_2.values), "Cache returned different values"
    print("  ✅ Cache hit — values identical")
    
    print("\n✅ Phase 5A Complete!")
