from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import (
    CACHE_PATH,
    CATEGORICAL_FEATURES,
    DATA_DIR,
    KMEANS_PATH,
    LEAD_TIME_DAYS,
    MODEL_PATH,
    N_STORES,
    PERISHABLE_DEPARTMENTS,
    RANDOM_STATE,
    SCALER_PATH,
    SERVICE_LEVEL_Z,
    TARGET_COL,
    TIME_BINS,
    TIME_LABELS,
    FEATURE_COLS,
    ARTIFACTS_DIR,
)

RAW_FILES = {
    "aisles": ("aisles.csv", {"aisle_id": "int16"}),
    "departments": ("departments.csv", {"department_id": "int16"}),
    "products": ("products.csv", {"product_id": "int32", "aisle_id": "int16", "department_id": "int16"}),
    "orders": (
        "orders.csv",
        {
            "order_id": "int32",
            "user_id": "int32",
            "order_number": "int16",
            "order_dow": "int8",
            "order_hour_of_day": "int8",
        },
    ),
    "order_products_prior": (
        "order_products__prior.csv",
        {"order_id": "int32", "product_id": "int32", "add_to_cart_order": "int16", "reordered": "int8"},
    ),
    "order_products_train": (
        "order_products__train.csv",
        {"order_id": "int32", "product_id": "int32", "add_to_cart_order": "int16", "reordered": "int8"},
    ),
}


def ensure_artifact_dirs(artifacts_dir: Path = ARTIFACTS_DIR) -> None:
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "cache").mkdir(parents=True, exist_ok=True)


def load_raw_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    missing = []
    for key, (filename, _) in RAW_FILES.items():
        if not (data_dir / filename).exists():
            missing.append(filename)
    if missing:
        raise FileNotFoundError(
            "Missing input files in data/raw: " + ", ".join(missing)
        )

    data = {}
    for key, (filename, dtypes) in RAW_FILES.items():
        data[key] = pd.read_csv(data_dir / filename, dtype=dtypes)
    return data


def build_user_store_mapping(orders: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, KMeans]:
    prior_orders = orders[orders["eval_set"] == "prior"].copy()

    user_features = prior_orders.groupby("user_id").agg(
        avg_hour=("order_hour_of_day", "mean"),
        avg_dow=("order_dow", "mean"),
        order_count=("order_id", "count"),
        avg_gap=("days_since_prior_order", "mean"),
    ).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_features)

    kmeans = KMeans(n_clusters=N_STORES, random_state=RANDOM_STATE, n_init=10)
    user_features["store_id"] = kmeans.fit_predict(X_scaled)

    user_features = user_features.reset_index()
    return user_features, scaler, kmeans


def build_demand_table(
    merged: pd.DataFrame,
    user_features: pd.DataFrame,
) -> pd.DataFrame:
    merged_with_store = merged.merge(user_features[["user_id", "store_id"]], on="user_id", how="inner")

    merged_with_store["time_bucket"] = pd.cut(
        merged_with_store["order_hour_of_day"],
        bins=TIME_BINS,
        labels=TIME_LABELS,
    )

    demand = (
        merged_with_store.groupby(
            ["store_id", "order_dow", "time_bucket", "department"],
            observed=True,
        )
        .size()
        .reset_index(name=TARGET_COL)
    )

    demand = demand[demand[TARGET_COL] > 0].copy()
    demand = demand.sort_values(
        ["store_id", "department", "time_bucket", "order_dow"]
    ).reset_index(drop=True)

    grouped = demand.groupby(
        ["store_id", "department", "time_bucket"],
        observed=True,
    )[TARGET_COL]

    demand["demand_lag_1w"] = grouped.shift(1)
    demand["demand_roll_4w"] = grouped.transform(lambda x: x.rolling(4).mean())
    demand["demand_std_4w"] = grouped.transform(lambda x: x.rolling(4).std())

    demand["is_perishable"] = demand["department"].isin(PERISHABLE_DEPARTMENTS).astype(int)
    demand["is_weekend"] = demand["order_dow"].isin([0, 6]).astype(int)
    demand["is_morning"] = (demand["time_bucket"] == "morning").astype(int)
    demand["is_evening"] = (demand["time_bucket"] == "evening").astype(int)

    demand = demand.dropna(subset=["demand_lag_1w", "demand_roll_4w", "demand_std_4w"])
    demand = demand.reset_index(drop=True)

    return demand


def load_and_prepare(data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    ensure_artifact_dirs(artifacts_dir)

    datasets = load_raw_data(data_dir)

    merged = datasets["order_products_prior"].merge(datasets["orders"], on="order_id", how="inner")
    merged["days_since_prior_order"] = merged["days_since_prior_order"].fillna(15.0)
    merged = merged.merge(datasets["products"], on="product_id", how="inner")
    merged = merged.merge(datasets["departments"], on="department_id", how="inner")

    user_features, scaler, kmeans = build_user_store_mapping(datasets["orders"])

    demand = build_demand_table(merged, user_features)

    demand.to_pickle(CACHE_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, KMEANS_PATH)

    return demand


def build_replenishment_frame(
    demand: pd.DataFrame,
    model=None,
    use_baseline: bool = False,
) -> pd.DataFrame:
    out = demand.copy()

    if use_baseline or model is None:
        out["predicted_demand"] = out["demand_lag_1w"].values
        out["prediction_source"] = "baseline"
    else:
        X = out[FEATURE_COLS].copy()
        X["time_bucket"] = X["time_bucket"].astype("category")
        X["department"] = X["department"].astype("category")
        out["predicted_demand"] = np.maximum(0, model.predict(X))
        out["prediction_source"] = "model"

    out["safety_stock"] = np.maximum(
        0,
        SERVICE_LEVEL_Z * out["demand_std_4w"].fillna(0) * np.sqrt(LEAD_TIME_DAYS),
    )

    group_mean = out.groupby(["store_id", "department"])["units_sold"].transform("mean")
    out["simulated_inventory"] = (group_mean * 0.8).round()

    out["order_qty"] = np.maximum(
        0,
        out["predicted_demand"] + out["safety_stock"] - out["simulated_inventory"],
    )
    out["order_qty_rounded"] = np.ceil(out["order_qty"]).astype(int)

    pred = out["predicted_demand"]
    order = out["order_qty"]
    perishable = out["is_perishable"] == 1

    out["risk_level"] = "Low"
    out["risk_reason"] = "Non-perishable item"

    mask_low = perishable & (order <= pred * 1.1)
    mask_med = perishable & (order > pred * 1.1) & (order <= pred * 1.3)
    mask_high = perishable & (order > pred * 1.3)

    out.loc[mask_low, "risk_level"] = "Low"
    out.loc[mask_low, "risk_reason"] = "Order closely matches demand"
    out.loc[mask_med, "risk_level"] = "Medium"
    out.loc[mask_med, "risk_reason"] = "Order 10-30% above demand"
    out.loc[mask_high, "risk_level"] = "High"
    out.loc[mask_high, "risk_reason"] = "Order >30% above demand - high spoilage risk"

    return out
