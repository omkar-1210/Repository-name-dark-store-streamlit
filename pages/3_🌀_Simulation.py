import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.simulation import apply_scenario, apply_custom_multipliers, compare_scenarios
from src.replenishment import calculate_store_kpis, calculate_order_quantities, flag_expiry_risk
import config

st.header("🌀 Scenario Simulation")

data = st.session_state.get("pipeline_data")
store_id = st.session_state.get("selected_store_id", 0)

if data is None:
    st.error("Pipeline data not loaded. Please return to the main page.")
    st.stop()

store_data = data[data["store_id"] == store_id].copy()

# ─── Scenario Selector ──────────────────────────────────
st.subheader("🎛️ Choose a Scenario")

scenario_tab, custom_tab = st.tabs(["📋 Preset Scenarios", "🔧 Custom Multipliers"])

with scenario_tab:
    selected_scenario = st.selectbox(
        "Select Scenario",
        options=list(config.SCENARIOS.keys()),
        key="sim_scenario"
    )
    
    # Show what multipliers this scenario applies
    scenario_details = config.SCENARIOS[selected_scenario]
    with st.expander("View scenario multipliers"):
        for dept, mult in scenario_details.items():
            label = "All other departments" if dept == "_default" else dept.title()
            color = "🔴" if mult > 1.2 else "🟢" if mult < 0.9 else "🟡"
            st.markdown(f"{color} **{label}**: ×{mult}")
    
    # Apply scenario
    simulated = apply_scenario(store_data, selected_scenario)

with custom_tab:
    st.markdown("Adjust demand multipliers per department:")
    
    departments = sorted(store_data["department"].unique())
    custom_mults = {}
    
    # Create slider grid — 3 columns
    cols = st.columns(3)
    for i, dept in enumerate(departments):
        with cols[i % 3]:
            mult = st.slider(
                dept.title(),
                min_value=0.0, max_value=3.0, value=1.0, step=0.1,
                key=f"sim_slider_{dept}"
            )
            if mult != 1.0:
                custom_mults[dept] = mult
    
    if custom_mults:
        simulated = apply_custom_multipliers(store_data, custom_mults)
    else:
        simulated = store_data.copy()
        simulated["original_demand"] = simulated["predicted_demand"]
        simulated["multiplier_applied"] = 1.0
        simulated["scenario"] = "No changes"

# ─── Before/After Comparison ────────────────────────────
st.markdown("---")
st.subheader("📊 Demand Impact")

total_before = simulated["original_demand"].sum()
total_after = simulated["predicted_demand"].sum()
change_pct = ((total_after - total_before) / total_before * 100) if total_before > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Original Demand", f"{total_before:,.0f}")
with col2:
    st.metric("Adjusted Demand", f"{total_after:,.0f}", 
              delta=f"{change_pct:+.1f}%")
with col3:
    st.metric("Impact Direction", 
              "📈 Increase" if change_pct > 0 else "📉 Decrease" if change_pct < 0 else "➡️ No Change")

# Per-department impact chart
dept_impact = simulated.groupby("department").agg(
    original=("original_demand", "sum"),
    adjusted=("predicted_demand", "sum"),
).reset_index()

dept_impact["change_pct"] = ((dept_impact["adjusted"] - dept_impact["original"]) / dept_impact["original"] * 100)
dept_impact = dept_impact.sort_values("change_pct", ascending=True)

fig_impact = go.Figure()

# Color: green for decrease (less spoilage risk), red for increase
colors = ["#16a34a" if x <= 0 else "#dc2626" for x in dept_impact["change_pct"]]

fig_impact.add_trace(go.Bar(
    y=dept_impact["department"],
    x=dept_impact["change_pct"],
    orientation="h",
    marker_color=colors,
    text=[f"{x:+.1f}%" for x in dept_impact["change_pct"]],
    textposition="outside",
))

fig_impact.update_layout(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#374151"),
    height=500,
    margin=dict(l=120, r=60, t=10, b=40),
    xaxis_title="Demand Change (%)",
    xaxis=dict(zeroline=True, zerolinecolor="#d1d5db"),
)

st.plotly_chart(fig_impact, width="stretch")

# ─── Updated Replenishment Under Scenario ────────────────
st.markdown("---")
st.subheader("📦 Updated Replenishment (Under Scenario)")

# Recalculate order quantities and risks with adjusted demand
simulated_replenish = calculate_order_quantities(simulated)
simulated_replenish = flag_expiry_risk(simulated_replenish)

sim_kpis = calculate_store_kpis(simulated_replenish, store_id=store_id)
orig_kpis = calculate_store_kpis(store_data, store_id=store_id)

col1, col2, col3 = st.columns(3)
with col1:
    delta_spoil = sim_kpis["spoilage_rate"] - orig_kpis["spoilage_rate"]
    st.metric("Spoilage Rate", f"{sim_kpis['spoilage_rate']}%", 
              delta=f"{delta_spoil:+.1f}% vs normal", delta_color="inverse")
with col2:
    delta_fill = sim_kpis["fill_rate"] - orig_kpis["fill_rate"]
    st.metric("Fill Rate", f"{sim_kpis['fill_rate']}%", 
              delta=f"{delta_fill:+.1f}% vs normal")
with col3:
    st.metric("Adjusted Orders", f"{simulated_replenish['order_qty_rounded'].sum():,}")

# ─── All Scenarios Comparison Table ──────────────────────
st.markdown("---")
st.subheader("📋 All Scenarios at a Glance")

comparison = compare_scenarios(store_data)

comparison["demand_change_pct"] = comparison["demand_change_pct"].apply(lambda x: f"{x:+.1f}%")
comparison["total_cost"] = comparison["total_cost"].apply(lambda x: f"₹{x:,.0f}")
comparison["spoilage_cost"] = comparison["spoilage_cost"].apply(lambda x: f"₹{x:,.0f}")
comparison["stockout_cost"] = comparison["stockout_cost"].apply(lambda x: f"₹{x:,.0f}")
comparison["total_demand"] = comparison["total_demand"].apply(lambda x: f"{x:,}")

comparison.columns = ["Scenario", "Total Demand", "Demand Δ", "Spoilage", "Stockout", "Total Cost"]
st.dataframe(comparison, width="stretch", hide_index=True)
