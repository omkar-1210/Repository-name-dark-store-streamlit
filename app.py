import streamlit as st
import sys
import os

# Add project root to path so pages can import from src/
sys.path.insert(0, os.path.dirname(__file__))
st.set_page_config(
    page_title="Dark Store Manager",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Clean Professional Light Theme ─────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Force pure white backgrounds everywhere */
    .stApp {
        background-color: #f9fafb !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Metric cards — clean white with subtle border */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px 24px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    
    div[data-testid="stMetric"] label {
        color: #6b7280 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    
    /* Force tab bar styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #6b7280;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #eef2ff !important;
        color: #4f46e5 !important;
        font-weight: 600;
    }
    
    /* DataFrames */
    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    h1 { font-size: 1.75rem !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.1rem !important; }
    
    /* Alerts */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Dividers */
    hr {
        border-color: #e5e7eb !important;
    }
    
    /* Expanders */
    details {
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
    }
    
    /* Force sidebar text */
    section[data-testid="stSidebar"] * {
        color: #374151;
    }
    
    /* Info box override — prevent pink/warm tint */
    div[data-testid="stNotification"] {
        background-color: #eef2ff !important;
        border: 1px solid #c7d2fe !important;
        color: #3730a3 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar: Global Controls ───────────────────────────
st.sidebar.markdown("## 🏪 Dark Store Manager")
st.sidebar.markdown("---")

# Store selector (shared across all pages via session_state)
store_options = list(range(20))
selected_store = st.sidebar.selectbox(
    "Select Store",
    options=store_options,
    format_func=lambda x: f"Store #{x}",
    key="selected_store"
)

# Prediction source toggle
use_baseline = st.sidebar.toggle(
    "Use Baseline (Lag-1) Predictions",
    value=False,
    help="Toggle between LightGBM model predictions and naive lag-1 baseline"
)
st.session_state["use_baseline"] = use_baseline

# Load data once via session_state (cached across page navigations)
@st.cache_data(ttl=300)
def load_pipeline_data(use_baseline: bool):
    from src.replenishment import build_replenishment_pipeline
    return build_replenishment_pipeline(use_baseline=use_baseline)

# Store in session_state so pages can access
data = load_pipeline_data(use_baseline)
st.session_state["pipeline_data"] = data
st.session_state["selected_store_id"] = selected_store

# Sidebar info
source_label = "Baseline (Lag-1)" if use_baseline else "LightGBM Model"
st.sidebar.info(f"📊 Source: **{source_label}**")
st.sidebar.markdown(f"📦 **{data.shape[0]:,}** data points")
st.sidebar.markdown(f"🏬 **20** stores · **21** departments")

# ─── Landing Page ────────────────────────────────────────
st.title("🏪 Dark Store Inventory & Spoilage Manager")
st.markdown("**Demand Forecasting · Smart Replenishment · Scenario Simulation · SHAP Explainability**")
st.markdown("---")

# Quick overview metrics for the selected store
from src.replenishment import calculate_store_kpis
kpis = calculate_store_kpis(data, store_id=selected_store)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Spoilage Rate", f"{kpis['spoilage_rate']}%")
with col2:
    st.metric("Fill Rate", f"{kpis['fill_rate']}%")
with col3:
    cost = kpis['cost_saved']
    label = "Cost Saved" if cost >= 0 else "Extra Cost"
    st.metric(label, f"₹{abs(cost):,.0f}",
              delta=f"{'Saving' if cost >= 0 else 'Over baseline'}",
              delta_color="normal" if cost >= 0 else "inverse")
with col4:
    st.metric("Total Demand", f"{kpis['total_demand']:,}")

st.markdown("---")
st.info("👈 Use the **sidebar tabs** to explore detailed analytics for each store.")
