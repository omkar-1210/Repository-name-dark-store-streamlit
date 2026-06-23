import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.replenishment import calculate_store_kpis, get_department_breakdown

st.header("🏪 Store Dashboard")

DOW_NAMES = {3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}

# Get shared state
data = st.session_state.get("pipeline_data")
store_id = st.session_state.get("selected_store_id", 0)

if data is None:
    st.error("Pipeline data not loaded. Please return to the main page.")
    st.stop()

store_data = data[data["store_id"] == store_id]

# ─── KPI Cards Row ──────────────────────────────────────
kpis = calculate_store_kpis(data, store_id=store_id)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Spoilage Rate", f"{kpis['spoilage_rate']}%")
with col2:
    st.metric("Fill Rate", f"{kpis['fill_rate']}%")
with col3:
    cost = kpis['cost_saved']
    if cost >= 0:
        st.metric("Cost Saved vs Baseline", f"₹{cost:,.0f}")
    else:
        st.metric("Extra Cost vs Baseline", f"₹{abs(cost):,.0f}", 
                  delta="Model costs more", delta_color="inverse")

st.markdown("---")

# ─── Hourly Demand Heatmap ──────────────────────────────
st.subheader("📊 Demand Heatmap — Day × Time Bucket")

# Pivot: rows=DOW, cols=time_bucket, values=sum(units_sold)
heatmap_data = store_data.pivot_table(
    index="order_dow", 
    columns="time_bucket", 
    values="units_sold", 
    aggfunc="sum"
)

# Reorder time buckets logically
time_order = ["midnight", "early_morning", "morning", "afternoon", "evening", "night"]
heatmap_data = heatmap_data.reindex(columns=[t for t in time_order if t in heatmap_data.columns])

# Map DOW numbers to names for display
heatmap_data.index = heatmap_data.index.map(DOW_NAMES)

fig_heat = px.imshow(
    heatmap_data,
    labels=dict(x="Time of Day", y="Day", color="Units Sold"),
    color_continuous_scale="Blues",
    aspect="auto",
    text_auto=True,
)

fig_heat.update_layout(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#374151"),
    height=350,
    margin=dict(l=20, r=20, t=30, b=20),
)
fig_heat.update_traces(textfont_size=11)

st.plotly_chart(fig_heat, width="stretch")

# ─── Department Breakdown ────────────────────────────────
st.subheader("🏷️ Department-Level Breakdown")

dept_df = get_department_breakdown(data, store_id=store_id)

# Horizontal bar chart sorted by total_cost
fig_dept = go.Figure()

fig_dept.add_trace(go.Bar(
    y=dept_df["department"],
    x=dept_df["spoilage_cost"],
    name="Spoilage Cost (₹)",
    orientation="h",
    marker_color="#dc2626",       # Red for spoilage
    marker_line=dict(width=0),
))

fig_dept.add_trace(go.Bar(
    y=dept_df["department"],
    x=dept_df["stockout_cost"],
    name="Stockout Cost (₹)",
    orientation="h",
    marker_color="#d97706",       # Amber for stockout
    marker_line=dict(width=0),
))

fig_dept.update_layout(
    barmode="stack",
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#374151"),
    height=550,
    margin=dict(l=120, r=20, t=10, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12),
    ),
    xaxis_title="Cost (₹)",
    yaxis=dict(autorange="reversed"),
)

st.plotly_chart(fig_dept, width="stretch")

# ─── Detailed Table (expandable) ────────────────────────
with st.expander("📋 View Full Department Data Table"):
    display_df = dept_df[[
        "department", "total_demand", "total_predicted", 
        "total_order_qty", "spoilage_cost", "stockout_cost", 
        "total_cost", "risk_level"
    ]].copy()
    
    display_df.columns = [
        "Department", "Actual Demand", "Predicted Demand",
        "Order Qty", "Spoilage Cost (₹)", "Stockout Cost (₹)",
        "Total Cost (₹)", "Risk"
    ]
    
    st.dataframe(
        display_df, 
        width="stretch",
        hide_index=True,
        height=500,
    )
