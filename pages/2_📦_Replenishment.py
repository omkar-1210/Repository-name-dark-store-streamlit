import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.replenishment import (
    get_department_breakdown, 
    calculate_cost_comparison,
    calculate_store_kpis,
)

st.header("📦 Replenishment Panel")

DOW_NAMES = {3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}

data = st.session_state.get("pipeline_data")
store_id = st.session_state.get("selected_store_id", 0)

if data is None:
    st.error("Pipeline data not loaded. Please return to the main page.")
    st.stop()

store_data = data[data["store_id"] == store_id]

# ─── Filters Row ────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    selected_dow = st.selectbox(
        "Day of Week",
        options=["All"] + [DOW_NAMES[d] for d in sorted(store_data["order_dow"].unique())],
        key="replenish_dow"
    )

with col_f2:
    risk_filter = st.multiselect(
        "Risk Level",
        options=["🟢 Low", "🟡 Medium", "🔴 High"],
        default=["🟢 Low", "🟡 Medium", "🔴 High"],
        key="replenish_risk"
    )

with col_f3:
    perishable_only = st.toggle("Perishable Only", value=False, key="replenish_perish")

# Apply filters
filtered = store_data.copy()

if selected_dow != "All":
    dow_reverse = {v: k for k, v in DOW_NAMES.items()}
    filtered = filtered[filtered["order_dow"] == dow_reverse[selected_dow]]

if risk_filter:
    filtered = filtered[filtered["risk_level"].isin(risk_filter)]

if perishable_only:
    filtered = filtered[filtered["is_perishable"] == 1]

# ─── Order Summary Table ────────────────────────────────
st.subheader("📋 Order Recommendations")

display_cols = [
    "department", "time_bucket", "order_dow",
    "predicted_demand", "safety_stock", "simulated_inventory",
    "order_qty_rounded", "risk_level", "risk_reason"
]
display_df = filtered[display_cols].copy()

display_df["order_dow"] = display_df["order_dow"].map(DOW_NAMES)
display_df.columns = [
    "Department", "Time Bucket", "Day",
    "Predicted Demand", "Safety Stock", "Current Inventory",
    "Order Qty", "Risk", "Risk Reason"
]

# Round floats for display
for col in ["Predicted Demand", "Safety Stock", "Current Inventory"]:
    display_df[col] = display_df[col].round(0).astype(int)

st.dataframe(
    display_df.sort_values(["Risk", "Order Qty"], ascending=[True, False]),
    width="stretch",
    hide_index=True,
    height=400,
)
st.caption(f"Showing {len(display_df)} items for Store #{store_id}")

# ─── Cost Comparison Section ────────────────────────────
st.markdown("---")
st.subheader("💰 Model vs Baseline Cost Comparison")

cost = calculate_cost_comparison(store_data)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model Total Cost",
        f"₹{cost['model']['total']:,.0f}",
        delta=f"Spoilage: ₹{cost['model']['spoilage_cost']:,.0f}"
    )

with col2:
    st.metric(
        "Baseline Total Cost",
        f"₹{cost['baseline']['total']:,.0f}",
        delta=f"Spoilage: ₹{cost['baseline']['spoilage_cost']:,.0f}"
    )

with col3:
    savings_color = "normal" if cost['savings']['percentage'] >= 0 else "inverse"
    st.metric(
        "Savings",
        f"₹{cost['savings']['absolute']:,.0f}",
        delta=f"{cost['savings']['percentage']:+.1f}%",
        delta_color=savings_color
    )

# Cost breakdown chart (model vs baseline side by side)
cost_compare_df = pd.DataFrame({
    "Type": ["Spoilage", "Stockout", "Spoilage", "Stockout"],
    "Source": ["Model", "Model", "Baseline", "Baseline"],
    "Cost (₹)": [
        cost["model"]["spoilage_cost"],
        cost["model"]["stockout_cost"],
        cost["baseline"]["spoilage_cost"],
        cost["baseline"]["stockout_cost"],
    ]
})

fig_cost = px.bar(
    cost_compare_df,
    x="Source",
    y="Cost (₹)",
    color="Type",
    barmode="group",
    color_discrete_map={"Spoilage": "#dc2626", "Stockout": "#d97706"},
    text_auto=True,
)

fig_cost.update_layout(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#374151"),
    height=350,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_cost.update_traces(textfont_size=11, textposition="outside")
st.plotly_chart(fig_cost, width="stretch")

# ─── Expiry Risk Summary ────────────────────────────────
st.markdown("---")
st.subheader("⚠️ Expiry Risk Summary")

risk_counts = store_data["risk_level"].value_counts()

col1, col2, col3 = st.columns(3)
for col, level in zip([col1, col2, col3], ["🟢 Low", "🟡 Medium", "🔴 High"]):
    count = risk_counts.get(level, 0)
    pct = count / len(store_data) * 100 if len(store_data) > 0 else 0
    with col:
        st.metric(level, f"{count}", delta=f"{pct:.1f}% of items")

# High-risk items table
high_risk = store_data[store_data["risk_level"] == "🔴 High"]

if len(high_risk) > 0:
    st.warning(f"⚠️ {len(high_risk)} items at HIGH spoilage risk")
    st.dataframe(
        high_risk[["department", "time_bucket", "order_dow", "predicted_demand", 
                    "order_qty_rounded", "risk_reason"]]
        .rename(columns={"order_dow": "Day"})
        .assign(Day=lambda df: df["Day"].map(DOW_NAMES)),
        width="stretch",
        hide_index=True,
    )
else:
    st.success("✅ No high-risk items for this store!")
