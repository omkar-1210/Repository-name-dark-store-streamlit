import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"

PERISHABLE_DEPARTMENTS = ["produce", "dairy eggs", "meat seafood", "bakery", "deli"]

FEATURE_COLS = [
    "store_id", "order_dow", "time_bucket", "department", 
    "demand_lag_1w", "demand_roll_4w", "demand_std_4w",
    "is_perishable", "is_weekend", "is_morning", "is_evening"
]

TARGET_COL = "units_sold"

# ─── Cost Constants (₹) ─────────────────────────────────
SPOILAGE_COST_PER_UNIT = 25
STOCKOUT_COST_PER_UNIT = 15

# ─── Replenishment ──────────────────────────────────────
SERVICE_LEVEL_Z = 1.65    # 95% service level
LEAD_TIME_DAYS = 1

# ─── LightGBM Hyperparameters ───────────────────────────
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "n_estimators": 200,
    "num_leaves": 15,
    "min_child_samples": 50,
    "random_state": 42,
    "verbose": -1,
    "force_col_wise": True,
}
CATEGORICAL_FEATURES = ["store_id", "time_bucket", "department"]

# ─── Simulation Scenarios ───────────────────────────────
SCENARIOS = {
    "Normal Day":        {"_default": 1.0},
    "Rainy Day":         {"_default": 0.7},
    "IPL Match Evening": {"snacks": 1.5, "beverages": 1.5, "_default": 1.0},
    "Heatwave":          {"produce": 1.3, "beverages": 1.3, "dairy eggs": 1.2, "_default": 1.0},
    "Festival/Sale":     {"_default": 1.4},
}

# ─── Model Paths ────────────────────────────────────────
MODEL_PATH = BASE_DIR / "models" / "lgbm_demand.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"
