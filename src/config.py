from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
CACHE_DIR = ARTIFACTS_DIR / "cache"

CACHE_PATH = CACHE_DIR / "processed_data.pkl"
MODEL_PATH = MODELS_DIR / "lgbm_demand.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
SHAP_CACHE = CACHE_DIR / "shap_values.pkl"
SCALER_PATH = MODELS_DIR / "store_scaler.joblib"
KMEANS_PATH = MODELS_DIR / "store_kmeans.joblib"

PERISHABLE_DEPARTMENTS = ["produce", "dairy eggs", "meat seafood", "bakery", "deli"]

FEATURE_COLS = [
    "store_id",
    "order_dow",
    "time_bucket",
    "department",
    "demand_lag_1w",
    "demand_roll_4w",
    "demand_std_4w",
    "is_perishable",
    "is_weekend",
    "is_morning",
    "is_evening",
]

TARGET_COL = "units_sold"

SPOILAGE_COST = 25
STOCKOUT_COST = 15

SERVICE_LEVEL_Z = 1.65
LEAD_TIME_DAYS = 1

N_STORES = 20
RANDOM_STATE = 42

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "n_estimators": 200,
    "num_leaves": 15,
    "min_child_samples": 50,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "force_col_wise": True,
}

CATEGORICAL_FEATURES = ["store_id", "time_bucket", "department"]

SCENARIOS = {
    "Normal Day": {"_default": 1.0},
    "Rainy Day": {"_default": 0.7},
    "IPL Match Evening": {"snacks": 1.5, "beverages": 1.5, "_default": 1.0},
    "Heatwave": {"produce": 1.3, "beverages": 1.3, "dairy eggs": 1.2, "_default": 1.0},
    "Festival/Sale": {"_default": 1.4},
}

TIME_BINS = [-1, 3, 7, 11, 15, 19, 23]
TIME_LABELS = ["midnight", "early_morning", "morning", "afternoon", "evening", "night"]
DOW_NAMES = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}
TIME_ORDER = ["midnight", "early_morning", "morning", "afternoon", "evening", "night"]
