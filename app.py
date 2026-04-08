from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.config import (
    ARTIFACTS_DIR,
    CACHE_PATH,
    DOW_NAMES,
    FEATURE_COLS,
    METRICS_PATH,
    MODEL_PATH,
    SCENARIOS,
    SHAP_CACHE,
    TIME_ORDER,
    TIME_BINS,
    TIME_LABELS,
)
from src.data_pipeline import build_replenishment_frame
from src.explainability import get_cached_shap, get_shap_for_row
from src.simulation import apply_scenario, compare_scenarios

st.set_page_config(page_title="Dark Store Demand Dashboard", layout="wide")
st.title("Dark Store Demand Forecasting and Replenishment")

def load_artifacts():
    if not CACHE_PATH.exists():
        return None, None, None

    demand = pd.read_pickle(CACHE_PATH)

    model = None
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return demand, model, metrics


@st.cache_data
def load_cached_outputs():
    demand, model, metrics = load_artifacts()
    return demand, metrics


demand, metrics = load_cached_outputs()
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

if demand is None:
    st.error("Processed data not found. Run `python train.py` first to create artifacts.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    use_baseline = st.checkbox("Use baseline forecast", value=False)
    selected_store = st.selectbox("Store", sorted(demand["store_id"].unique().tolist()))
    scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()))

replenishment_df = build_replenishment_frame(
    demand,
    model=model,
    use_baseline=use_baseline,
)

selected_store_df = replenishment_df[replenishment_df["store_id"] == selected_store].copy()

tab_overview, tab_store, tab_scenarios, tab_explain = st.tabs(
    ["Overview", "Store analysis", "Scenario simulation", "Explainability"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed rows", f"{len(demand):,}")
    c2.metric("Stores", f"{demand['store_id'].nunique():,}")
    c3.metric("Departments", f"{demand['department'].nunique():,}")
    c4.metric("High-risk rows", f"{(replenishment_df['risk_level'] == 'High').sum():,}")

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model MAE", f"{metrics.get('mae', 0):,.2f}")
        m2.metric("Baseline MAE", f"{metrics.get('baseline_mae', 0):,.2f}")
        m3.metric("Model cost", f"₹{metrics.get('model_cost_total', 0):,.2f}")
        m4.metric("Savings", f"{metrics.get('savings_pct', 0):+.2f}%")

    st.subheader("Processed demand data")
    st.dataframe(demand.head(100), use_container_width=True)

    if "predicted_demand" in replenishment_df.columns:
        st.subheader("Top risk rows")
        risk_view = replenishment_df.sort_values(
            ["risk_level", "predicted_demand"],
            ascending=[False, False],
        )[[
            "store_id",
            "department",
            "order_dow",
            "time_bucket",
            "units_sold",
            "predicted_demand",
            "order_qty_rounded",
            "risk_level",
            "risk_reason",
        ]].head(50)
        st.dataframe(risk_view, use_container_width=True)

with tab_store:
    st.subheader(f"Store #{selected_store} analysis")
    store_df = selected_store_df

    k1, k2, k3 = st.columns(3)
    k1.metric("Rows", f"{len(store_df):,}")
    k2.metric("Total actual demand", f"{store_df['units_sold'].sum():,}")
    k3.metric("Total predicted demand", f"{store_df['predicted_demand'].sum():,.0f}")

    dept_summary = store_df.groupby("department").agg(
        total_demand=("units_sold", "sum"),
        total_predicted=("predicted_demand", "sum"),
        total_order_qty=("order_qty_rounded", "sum"),
        is_perishable=("is_perishable", "first"),
    ).reset_index()

    dept_costs = store_df.groupby("department").apply(
        lambda g: pd.Series(
            {
                "spoilage_units": np.maximum(0, g["predicted_demand"] - g["units_sold"]).sum(),
                "stockout_units": np.maximum(0, g["units_sold"] - g["predicted_demand"]).sum(),
            }
        )
    ).reset_index()

    dept_summary = dept_summary.merge(dept_costs, on="department", how="left")
    dept_summary["spoilage_cost"] = dept_summary["spoilage_units"] * 25
    dept_summary["stockout_cost"] = dept_summary["stockout_units"] * 15
    dept_summary["total_cost"] = dept_summary["spoilage_cost"] + dept_summary["stockout_cost"]
    dept_summary = dept_summary.sort_values("total_cost", ascending=False)

    st.write("Department summary")
    st.dataframe(dept_summary, use_container_width=True)

    heatmap = store_df.pivot_table(
        index="order_dow",
        columns="time_bucket",
        values="units_sold",
        aggfunc="sum",
    )
    heatmap = heatmap.reindex(columns=[t for t in TIME_ORDER if t in heatmap.columns])
    heatmap.index = heatmap.index.map(DOW_NAMES)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(heatmap.values, aspect="auto")
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_title(f"Demand heatmap - Store #{selected_store}")
    for r in range(heatmap.shape[0]):
        for c in range(heatmap.shape[1]):
            val = heatmap.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{int(val):,}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Units sold")
    st.pyplot(fig)

    st.write("Store-level rows")
    st.dataframe(store_df.head(100), use_container_width=True)

with tab_scenarios:
    st.subheader("Scenario comparison")

    scenario_df = compare_scenarios(selected_store_df)
    st.dataframe(scenario_df, use_container_width=True)

    simulated = apply_scenario(selected_store_df, scenario_name)
    dept_impact = simulated.groupby("department").agg(
        original=("original_demand", "sum"),
        adjusted=("predicted_demand", "sum"),
    ).reset_index()

    dept_impact["change_pct"] = np.where(
        dept_impact["original"] > 0,
        (dept_impact["adjusted"] - dept_impact["original"]) / dept_impact["original"] * 100,
        0,
    )
    dept_impact = dept_impact.sort_values("change_pct")
    colors = ["#16a34a" if x <= 0 else "#dc2626" for x in dept_impact["change_pct"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(dept_impact["department"], dept_impact["change_pct"], color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Demand change (%)")
    ax.set_title(f'Scenario impact: {scenario_name} | Store #{selected_store}')
    for i, val in enumerate(dept_impact["change_pct"]):
        ax.text(val + (0.3 if val >= 0 else -0.3), i, f"{val:+.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

with tab_explain:
    st.subheader("Model explainability")

    if model is None:
        st.info("Model not found. Run `python train.py` first.")
    else:
        X_shap = demand[FEATURE_COLS].copy()
        X_shap["time_bucket"] = X_shap["time_bucket"].astype("category")
        X_shap["department"] = X_shap["department"].astype("category")

        shap_values = get_cached_shap(model, X_shap, max_samples=500)
        st.write(f"SHAP sample size: {shap_values.values.shape[0]} rows")

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.beeswarm(shap_values, show=False, plot_size=None)
        plt.tight_layout()
        st.pyplot(fig)

        row_index = int(demand[demand["store_id"] == selected_store].index.min())
        single_row = X_shap.iloc[[row_index]]
        single_shap = get_shap_for_row(model, X_shap, row_index)

        st.write("One-row explanation")
        st.write(
            {
                "store_id": int(demand.loc[row_index, "store_id"]),
                "department": str(demand.loc[row_index, "department"]),
                "actual_demand": float(demand.loc[row_index, "units_sold"]),
                "predicted_demand": float(np.maximum(0, model.predict(single_row)[0])),
            }
        )

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(single_shap, show=False)
        plt.tight_layout()
        st.pyplot(fig2)
