import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_predictions(
    df: pd.DataFrame, 
    model=None, 
    use_baseline: bool = False
) -> pd.DataFrame:
    result = df.copy()
    
    if use_baseline:
        result["predicted_demand"] = result["demand_lag_1w"].values
        result["prediction_source"] = "baseline"
    else:
        if model is None:
            raise ValueError("Model must be provided when use_baseline=False")
        
        X = result[config.FEATURE_COLS].copy()
        X["time_bucket"] = X["time_bucket"].astype("category")
        X["department"] = X["department"].astype("category")
        
        result["predicted_demand"] = model.predict(X)
        result["prediction_source"] = "model"
    
    # Demand is always non-negative
    result["predicted_demand"] = np.maximum(0, result["predicted_demand"])
    
    return result

def calculate_safety_stock(
    demand_std: pd.Series,
    z: float = None,
    lead_time: int = None
) -> pd.Series:
    if z is None:
        z = config.SERVICE_LEVEL_Z
    if lead_time is None:
        lead_time = config.LEAD_TIME_DAYS
        
    safety = z * demand_std * np.sqrt(lead_time)
    return np.maximum(0, safety.fillna(0))

def calculate_order_quantities(predictions_df: pd.DataFrame) -> pd.DataFrame:
    result = predictions_df.copy()
    
    result["safety_stock"] = calculate_safety_stock(result["demand_std_4w"])
    
    group_means = result.groupby(
        ["store_id", "department"]
    )["units_sold"].transform("mean")
    result["simulated_inventory"] = (group_means * 0.8).round()
    
    result["order_qty"] = np.maximum(
        0, 
        result["predicted_demand"] + result["safety_stock"] - result["simulated_inventory"]
    )
    result["order_qty_rounded"] = np.ceil(result["order_qty"]).astype(int)
    
    return result

def flag_expiry_risk(replenishment_df: pd.DataFrame) -> pd.DataFrame:
    result = replenishment_df.copy()
    
    pred = result["predicted_demand"]
    order = result["order_qty"]
    is_perish = result["is_perishable"] == 1
    
    # Default: Low risk
    result["risk_level"] = "🟢 Low"
    result["risk_reason"] = "Non-perishable item"
    
    # Perishable items: assess overstocking risk
    perish_low = is_perish & (order <= pred * 1.1)
    perish_med = is_perish & (order > pred * 1.1) & (order <= pred * 1.3)
    perish_high = is_perish & (order > pred * 1.3)
    
    result.loc[perish_low, "risk_level"] = "🟢 Low"
    result.loc[perish_low, "risk_reason"] = "Order quantity closely matches expected demand"
    
    result.loc[perish_med, "risk_level"] = "🟡 Medium"
    result.loc[perish_med, "risk_reason"] = "Order exceeds demand by 10-30% — monitor freshness"
    
    result.loc[perish_high, "risk_level"] = "🔴 High"
    result.loc[perish_high, "risk_reason"] = "Order exceeds demand by >30% — high spoilage risk"
    
    return result

def calculate_cost_comparison(replenishment_df: pd.DataFrame) -> dict:
    actual = replenishment_df["units_sold"].values
    predicted = replenishment_df["predicted_demand"].values
    baseline = replenishment_df["demand_lag_1w"].values
    
    def breakdown(pred, act):
        spoilage = np.maximum(0, pred - act) * config.SPOILAGE_COST_PER_UNIT
        stockout = np.maximum(0, act - pred) * config.STOCKOUT_COST_PER_UNIT
        return {
            "spoilage_cost": round(float(spoilage.sum()), 2),
            "stockout_cost": round(float(stockout.sum()), 2),
            "total": round(float((spoilage + stockout).sum()), 2),
        }
    
    model_costs = breakdown(predicted, actual)
    baseline_costs = breakdown(baseline, actual)
    
    savings_abs = baseline_costs["total"] - model_costs["total"]
    savings_pct = (savings_abs / baseline_costs["total"] * 100) if baseline_costs["total"] > 0 else 0
    
    return {
        "model": model_costs,
        "baseline": baseline_costs,
        "savings": {
            "absolute": round(savings_abs, 2),
            "percentage": round(savings_pct, 2),
        }
    }

def calculate_store_kpis(
    replenishment_df: pd.DataFrame, 
    store_id: int = None
) -> dict:
    df = replenishment_df.copy()
    if store_id is not None:
        df = df[df["store_id"] == store_id]
    
    if len(df) == 0:
        return {
            "spoilage_rate": 0.0, "fill_rate": 0.0, "cost_saved": 0.0,
            "total_ordered": 0, "total_demand": 0, "total_spoilage_units": 0,
        }
    
    predicted = df["predicted_demand"].values
    actual = df["units_sold"].values
    baseline = df["demand_lag_1w"].values
    
    # Spoilage: units we over-ordered
    spoilage_units = np.maximum(0, predicted - actual).sum()
    total_ordered = predicted.sum()
    spoilage_rate = (spoilage_units / total_ordered * 100) if total_ordered > 0 else 0
    
    # Fill rate: how much demand we satisfied
    fulfilled = np.minimum(predicted, actual).sum()
    total_demand = actual.sum()
    fill_rate = (fulfilled / total_demand * 100) if total_demand > 0 else 0
    
    # Cost saved vs baseline
    model_cost = (
        np.maximum(0, predicted - actual) * config.SPOILAGE_COST_PER_UNIT +
        np.maximum(0, actual - predicted) * config.STOCKOUT_COST_PER_UNIT
    ).sum()
    baseline_cost = (
        np.maximum(0, baseline - actual) * config.SPOILAGE_COST_PER_UNIT +
        np.maximum(0, actual - baseline) * config.STOCKOUT_COST_PER_UNIT
    ).sum()
    cost_saved = baseline_cost - model_cost
    
    return {
        "spoilage_rate": round(float(spoilage_rate), 1),
        "fill_rate": round(float(fill_rate), 1),
        "cost_saved": round(float(cost_saved), 2),
        "total_ordered": int(total_ordered),
        "total_demand": int(total_demand),
        "total_spoilage_units": int(spoilage_units),
    }

def get_department_breakdown(
    replenishment_df: pd.DataFrame,
    store_id: int = None
) -> pd.DataFrame:
    df = replenishment_df.copy()
    if store_id is not None:
        df = df[df["store_id"] == store_id]
    
    dept_summary = df.groupby("department").agg(
        total_demand=("units_sold", "sum"),
        total_predicted=("predicted_demand", "sum"),
        total_order_qty=("order_qty_rounded", "sum"),
        is_perishable=("is_perishable", "first"),
    ).reset_index()
    
    # Calculate costs per department
    dept_costs = df.groupby("department").apply(
        lambda g: pd.Series({
            "spoilage_units": np.maximum(0, g["predicted_demand"] - g["units_sold"]).sum(),
            "stockout_units": np.maximum(0, g["units_sold"] - g["predicted_demand"]).sum(),
        })
    ).reset_index()
    
    dept_summary = dept_summary.merge(dept_costs, on="department")
    dept_summary["spoilage_cost"] = (dept_summary["spoilage_units"] * config.SPOILAGE_COST_PER_UNIT).round(2)
    dept_summary["stockout_cost"] = (dept_summary["stockout_units"] * config.STOCKOUT_COST_PER_UNIT).round(2)
    dept_summary["total_cost"] = dept_summary["spoilage_cost"] + dept_summary["stockout_cost"]
    
    # Most common risk level per department
    risk_mode = df.groupby("department")["risk_level"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "🟢 Low"
    ).reset_index()
    dept_summary = dept_summary.merge(risk_mode, on="department")
    
    return dept_summary.sort_values("total_cost", ascending=False).reset_index(drop=True)

def build_replenishment_pipeline(use_baseline: bool = False) -> pd.DataFrame:
    # Load data
    df = pd.read_pickle(config.CACHE_DIR / "processed_data.pkl")
    
    # Load model (even if using baseline, for comparison)
    model = None
    if not use_baseline and config.MODEL_PATH.exists():
        model = joblib.load(config.MODEL_PATH)
    elif not use_baseline:
        print("⚠️ Model not found, falling back to baseline")
        use_baseline = True
    
    # Generate predictions
    result = generate_predictions(df, model=model, use_baseline=use_baseline)
    
    # Calculate order quantities
    result = calculate_order_quantities(result)
    
    # Flag expiry risks
    result = flag_expiry_risk(result)
    
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3: Replenishment Pipeline")
    print("=" * 60)
    
    # Test with baseline (reliable)
    print("\n--- Testing with BASELINE predictions ---")
    result_baseline = build_replenishment_pipeline(use_baseline=True)
    print(f"Shape: {result_baseline.shape}")
    print(f"Columns: {list(result_baseline.columns)}")
    
    # KPIs
    kpis = calculate_store_kpis(result_baseline)
    print(f"\nGlobal KPIs (baseline):")
    print(f"  Spoilage Rate: {kpis['spoilage_rate']}%")
    print(f"  Fill Rate: {kpis['fill_rate']}%")
    print(f"  Cost Saved vs baseline: ₹{kpis['cost_saved']:,.2f}")
    
    # Per-store KPIs
    print(f"\nPer-store KPIs (store 0):")
    store_kpis = calculate_store_kpis(result_baseline, store_id=0)
    for k, v in store_kpis.items():
        print(f"  {k}: {v}")
    
    # Department breakdown
    dept = get_department_breakdown(result_baseline, store_id=0)
    print(f"\nDepartment breakdown (store 0):")
    print(dept[["department", "total_demand", "total_order_qty", "total_cost", "risk_level"]].to_string(index=False))
    
    # Cost comparison
    cost = calculate_cost_comparison(result_baseline)
    print(f"\nCost Comparison:")
    print(f"  Model:    ₹{cost['model']['total']:,.2f}")
    print(f"  Baseline: ₹{cost['baseline']['total']:,.2f}")
    print(f"  Savings:  ₹{cost['savings']['absolute']:,.2f} ({cost['savings']['percentage']:.1f}%)")
    
    # Risk distribution
    risk_dist = result_baseline["risk_level"].value_counts()
    print(f"\nRisk Distribution:")
    for level, count in risk_dist.items():
        print(f"  {level}: {count} ({count/len(result_baseline)*100:.1f}%)")
    
    # Test with model
    print("\n\n--- Testing with MODEL predictions ---")
    result_model = build_replenishment_pipeline(use_baseline=False)
    model_kpis = calculate_store_kpis(result_model)
    print(f"Model KPIs:")
    print(f"  Spoilage Rate: {model_kpis['spoilage_rate']}%")
    print(f"  Fill Rate: {model_kpis['fill_rate']}%")
    
    # Validation assertions
    print("\n--- Validation ---")
    assert result_baseline.shape[0] == 9827, f"Expected 9827 rows, got {result_baseline.shape[0]}"
    assert (result_baseline["order_qty"] >= 0).all(), "Negative order quantities found"
    assert result_baseline["risk_level"].notna().all(), "Missing risk flags"
    assert "predicted_demand" in result_baseline.columns
    assert "safety_stock" in result_baseline.columns
    assert "order_qty_rounded" in result_baseline.columns
    print("✅ All validations passed!")
    
    print("\n✅ Phase 3 Complete!")
