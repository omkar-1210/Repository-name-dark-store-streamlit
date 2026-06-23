import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os

# Adjust sys.path to find config.py at project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "cache"
_MODEL_DIR = _PROJECT_ROOT / "models"

def load_processed_data() -> pd.DataFrame:
    path = _CACHE_DIR / "processed_data.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Run feature_engineering.py first."
        )
    return pd.read_pickle(path)

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[config.FEATURE_COLS].copy()
    y = df[config.TARGET_COL].copy()
    
    X["time_bucket"] = X["time_bucket"].astype("category")
    X["department"] = X["department"].astype("category")
    
    return X, y

def train_test_split_temporal(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    test_mask = X["order_dow"] == 6
    
    X_train = X[~test_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[~test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
        categorical_feature=config.CATEGORICAL_FEATURES,
    )
    
    return model

def evaluate_model(model: lgb.LGBMRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_pred = np.maximum(0, y_pred)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    bias = float(np.mean(y_pred - y_test))
    
    baseline_pred = X_test["demand_lag_1w"].values
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    
    def cost_fn(predicted, actual):
        spoilage = np.maximum(0, predicted - actual) * config.SPOILAGE_COST_PER_UNIT
        stockout = np.maximum(0, actual - predicted) * config.STOCKOUT_COST_PER_UNIT
        return spoilage + stockout
    
    model_cost = cost_fn(y_pred, y_test.values).sum()
    baseline_cost = cost_fn(baseline_pred, y_test.values).sum()
    
    savings_pct = (baseline_cost - model_cost) / baseline_cost * 100 if baseline_cost > 0 else 0.0
    
    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "bias": round(bias, 2),
        "baseline_mae": round(baseline_mae, 2),
        "baseline_rmse": round(baseline_rmse, 2),
        "model_cost_total": round(model_cost, 2),
        "baseline_cost_total": round(baseline_cost, 2),
        "savings_pct": round(savings_pct, 2),
    }
    
    return metrics

def save_model(model: lgb.LGBMRegressor, metrics: dict) -> Path:
    import json
    
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = config.MODEL_PATH
    joblib.dump(model, model_path)
    
    metrics_path = config.METRICS_PATH
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    
    return model_path

def get_feature_importance(model: lgb.LGBMRegressor, X: pd.DataFrame) -> pd.DataFrame:
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    return feat_imp

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2: LightGBM Model Training & Evaluation")
    print("=" * 60)
    
    print("\n[1/6] Loading processed data...")
    df = load_processed_data()
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    
    print("\n[2/6] Preparing features...")
    X, y = prepare_features(df)
    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  Categoricals: {[c for c in X.columns if X[c].dtype.name == 'category']}")
    
    print("\n[3/6] Splitting train/test (temporal: DOW 6 = test)...")
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y)
    print(f"  Train: {X_train.shape[0]} rows ({X_train.shape[0]/X.shape[0]:.0%})")
    print(f"  Test:  {X_test.shape[0]} rows ({X_test.shape[0]/X.shape[0]:.0%})")
    
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0], "Split lost rows"
    assert 6 not in X_train["order_dow"].values, "DOW 6 leaked into training"
    
    print("\n[4/6] Training LightGBM model...")
    model = train_model(X_train, y_train, X_test, y_test)
    print(f"  Best iteration: {model.best_iteration_}")
    
    print("\n[5/6] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n" + "─" * 40)
    print("  MODEL EVALUATION RESULTS")
    print("─" * 40)
    print(f"  Model MAE:      {metrics['mae']:>12,.2f}")
    print(f"  Baseline MAE:   {metrics['baseline_mae']:>12,.2f}")
    print(f"  Model RMSE:     {metrics['rmse']:>12,.2f}")
    print(f"  Baseline RMSE:  {metrics['baseline_rmse']:>12,.2f}")
    print(f"  Model Bias:     {metrics['bias']:>12,.2f}")
    print(f"  Model Cost:     ₹{metrics['model_cost_total']:>12,.2f}")
    print(f"  Baseline Cost:  ₹{metrics['baseline_cost_total']:>12,.2f}")
    print(f"  Savings:        {metrics['savings_pct']:>11.2f}%")
    print("─" * 40)
    
    if metrics["savings_pct"] > 0:
        print(f"  ✅ Model BEATS baseline by {metrics['savings_pct']:.1f}%")
    else:
        print(f"  ⚠️  Model WORSE than baseline by {abs(metrics['savings_pct']):.1f}%")
        print(f"  (This is NOT the -677% bug — the split is correct. May need hyperparameter tuning.)")
    
    feat_imp = get_feature_importance(model, X_train)
    print("\n  Top 5 Features:")
    for _, row in feat_imp.head(5).iterrows():
        print(f"    {row['feature']:<20s} {row['importance']:>6d}")
    
    print("\n[6/6] Saving model and metrics...")
    save_model(model, metrics)
    
    print("\n✅ Phase 2 Complete!")
