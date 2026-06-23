import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import sys, os
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.model import get_feature_importance, load_processed_data, prepare_features
from src.explainability import get_cached_shap, get_shap_for_row

st.header("🔍 Model Explainability")

# ─── Load Model ─────────────────────────────────────────
if not config.MODEL_PATH.exists():
    st.error("No trained model found. Please train the model first (Phase 2).")
    st.stop()

model = joblib.load(config.MODEL_PATH)

# ─── Feature Importance ─────────────────────────────────
st.subheader("📊 Feature Importance (LightGBM Gain)")

df = load_processed_data()
X, y = prepare_features(df)
feat_imp = get_feature_importance(model, X)

fig_imp = px.bar(
    feat_imp,
    x="importance",
    y="feature",
    orientation="h",
    color="importance",
    color_continuous_scale="Blues",
)

fig_imp.update_layout(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#374151"),
    height=400,
    margin=dict(l=140, r=20, t=10, b=40),
    xaxis_title="Importance (Split Count)",
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False,
    showlegend=False,
)

st.plotly_chart(fig_imp, width="stretch")

# ─── Model Metrics ──────────────────────────────────────
st.subheader("📈 Model Performance Metrics")
import json

metrics_path = config.METRICS_PATH
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE", f"{metrics['mae']:,.2f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:,.2f}")
    with col3:
        st.metric("Baseline MAE", f"{metrics['baseline_mae']:,.2f}")
    with col4:
        savings_delta = "better" if metrics['savings_pct'] > 0 else "worse"
        st.metric("Cost Savings", f"{metrics['savings_pct']:+.1f}%",
                  delta=f"vs baseline ({savings_delta})")

# ─── SHAP Analysis ─────────────────────────────────────
st.markdown("---")
st.subheader("🧠 SHAP Analysis")

# Compute/load cached SHAP values
with st.spinner("Computing SHAP values (first load may take a few seconds)..."):
    shap_values = get_cached_shap(model, X)

# Beeswarm plot
st.markdown("### Global Feature Impact (Beeswarm)")
st.caption("Each dot is one data point. Position shows impact on prediction. Color shows feature value (red=high, blue=low).")

fig_beeswarm, ax_bee = plt.subplots(figsize=(10, 5))
shap.plots.beeswarm(shap_values, show=False, plot_size=None)
ax_bee = plt.gca()
ax_bee.set_facecolor("white")
fig_beeswarm = plt.gcf()
fig_beeswarm.patch.set_facecolor("white")
st.pyplot(fig_beeswarm)
plt.close('all')

st.markdown("### Per-Prediction Breakdown (Waterfall)")
st.caption("Select a data point to see how each feature pushed the prediction up or down.")

# Let user pick a row to explain
col_w1, col_w2, col_w3 = st.columns(3)
with col_w1:
    selected_store = st.selectbox("Store", options=sorted(X["store_id"].unique()), key="shap_store")
with col_w2:
    store_mask = X["store_id"] == selected_store
    available_depts = sorted(X.loc[store_mask, "department"].unique())
    selected_dept = st.selectbox("Department", options=available_depts, key="shap_dept")
with col_w3:
    dept_mask = store_mask & (X["department"] == selected_dept)
    row_indices = X[dept_mask].index.tolist()
    selected_row = st.selectbox("Row Index", options=row_indices[:20], key="shap_row",
                                help="Select a specific data point to explain")

# Show waterfall for selected row
if selected_row is not None:
    single_shap = get_shap_for_row(model, X, selected_row)
    
    fig_waterfall, ax_wf = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(single_shap, show=False)
    ax_wf = plt.gca()
    ax_wf.set_facecolor("white")
    fig_waterfall = plt.gcf()
    fig_waterfall.patch.set_facecolor("white")
    st.pyplot(fig_waterfall)
    plt.close('all')
    
    # Show the actual data values for context
    with st.expander("View raw feature values for this data point"):
        row_data = X.iloc[selected_row]
        actual = y.iloc[selected_row]
        predicted = model.predict(X.iloc[[selected_row]])[0]
        
        st.markdown(f"**Actual demand:** {actual:,.0f} units")
        st.markdown(f"**Predicted demand:** {predicted:,.0f} units")
        st.dataframe(
            pd.DataFrame({"Feature": row_data.index, "Value": row_data.values}),
            hide_index=True,
            width="stretch",
        )
