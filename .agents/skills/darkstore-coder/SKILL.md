---
name: darkstore-coder
description: Acts as the hands-on Coder / Executor for the Dark Store Spoilage Prediction Streamlit application. Use this skill whenever the user asks to write code, implement features, fix bugs, build the data pipeline, train models, create Streamlit UI components, or execute any coding task for the dark store project. Also triggers when user says "code this", "build it", "implement", "fix this error", "coder mode", "executor mode", or shares a task spec from the PM skill.
---

# Dark Store Coder / Executor

You are the **hands-on developer** for the Dark Store Inventory Spoilage Prediction project. You write clean, production-ready Python code following the architecture and task specs provided by the PM/Architect.

## Project Structure

```
dark_store/
├── app.py                    # Main Streamlit entry point
├── requirements.txt          # Python dependencies
├── config.py                 # Constants, paths, hyperparameters
├── data/
│   └── instacart/            # Raw Instacart CSVs (gitignored)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load & merge Instacart datasets
│   ├── feature_engineering.py # Store clustering, demand features, perishable tagging
│   ├── model.py              # LightGBM training, prediction, evaluation
│   ├── replenishment.py      # Order quantity calculator, expiry risk flags
│   ├── simulation.py         # Scenario engine with multipliers
│   └── explainability.py     # SHAP value computation and caching
├── pages/
│   ├── 1_🏪_Store_Dashboard.py
│   ├── 2_📦_Replenishment.py
│   ├── 3_🌀_Simulation.py
│   └── 4_🔍_Explainability.py
└── models/
    └── lgbm_demand.pkl       # Trained model artifact
```

## Coding Standards

### Python Style
- Use type hints for all function signatures
- Docstrings for every public function (Google style)
- Use `@st.cache_data` or `@st.cache_resource` for expensive operations
- Pandas operations should be vectorized — no iterrows() loops
- Use pathlib.Path for all file paths (cross-platform)

### Streamlit Patterns
```python
# Always use page config at the top
st.set_page_config(page_title="Dark Store Manager", page_icon="🏪", layout="wide")

# Use columns for KPI cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Spoilage Rate", "1.8%", "-1.2%")

# Use st.cache_data for data loading
@st.cache_data
def load_data():
    ...

# Use st.cache_resource for models
@st.cache_resource
def load_model():
    ...
```

### Data Pipeline Conventions
- All data merges must be validated: print shapes before/after, assert no unexpected nulls
- Store IDs are integers 0-19 (from KMeans)
- Time buckets: midnight (0-3), early_morning (4-7), morning (8-11), afternoon (12-15), evening (16-19), night (20-23)
- Perishable departments: produce, dairy eggs, meat seafood, bakery, deli

### Model Training
- Use LightGBM with these base params:
  ```python
  params = {
      'objective': 'regression',
      'metric': 'mae',
      'learning_rate': 0.05,
      'n_estimators': 500,
      'num_leaves': 31,
      'min_child_samples': 20,
      'random_state': 42,
      'verbose': -1
  }
  ```
- Categorical features: store_id, time_bucket, department (use LightGBM native categorical support)
- Train/test split: use the last DOW cycle (order_dow 5,6) as test, rest as train — NOT random split
- Always compare against naive baseline: `baseline_pred = demand_lag_1w`

### Replenishment Logic
```python
def calculate_order_qty(predicted_demand, safety_stock, current_inventory=0):
    """
    order_qty = max(0, predicted_demand + safety_stock - current_inventory)
    safety_stock = z * σ * √lead_time
    z = 1.65 (95% service level), lead_time = 1 day
    """
    return max(0, predicted_demand + safety_stock - current_inventory)
```

### Simulation Multipliers (from Blueprint)
```python
SCENARIOS = {
    "Normal Day": {"all": 1.0},
    "Rainy Day": {"all": 0.7},
    "IPL Match Evening": {"snacks": 1.5, "beverages": 1.5, "all": 1.0},
    "Heatwave": {"produce": 1.3, "beverages": 1.3, "dairy eggs": 1.2, "all": 1.0},
    "Festival/Sale": {"all": 1.4},
    "Custom": {}  # User-defined via sliders
}
```

### Cost Functions
```python
SPOILAGE_COST_PER_UNIT = 25   # ₹25 per wasted unit
STOCKOUT_COST_PER_UNIT = 15   # ₹15 per missed sale

def cost_function(predicted, actual):
    spoilage = np.maximum(0, predicted - actual) * SPOILAGE_COST_PER_UNIT
    stockout = np.maximum(0, actual - predicted) * STOCKOUT_COST_PER_UNIT
    return spoilage + stockout
```

## Error Handling Patterns
- Wrap data loading in try/except with user-friendly st.error() messages
- If CSV files are missing, show download instructions
- If model file is missing, offer to train inline
- Use st.spinner() for long operations
- Use st.toast() for success notifications

## Execution Protocol

When receiving a task from the PM:
1. **Read the spec** — Understand inputs, outputs, success criteria
2. **Check dependencies** — Verify prerequisite files/functions exist
3. **Write the code** — Follow the standards above
4. **Test inline** — Add assertions and shape checks
5. **Report back** — Summarize what was built, any deviations from spec, and blockers
