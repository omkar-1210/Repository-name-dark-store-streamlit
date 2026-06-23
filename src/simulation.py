import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

def apply_scenario(
    predictions_df: pd.DataFrame,
    scenario_name: str
) -> pd.DataFrame:
    """Apply a predefined scenario's demand multipliers.
    
    Each scenario has a dict of {department: multiplier}.
    "_default" applies to all departments not explicitly listed.
    
    Args:
        predictions_df: DataFrame with 'predicted_demand' and 'department' columns
        scenario_name: Key from config.SCENARIOS (e.g., "Rainy Day")
    
    Returns:
        DataFrame with updated 'predicted_demand' and new columns:
        - scenario: str (name of the applied scenario)
        - multiplier_applied: float (the multiplier used per row)
        - original_demand: float (pre-adjustment demand)
    """
    if scenario_name not in config.SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {list(config.SCENARIOS.keys())}"
        )
    
    result = predictions_df.copy()
    scenario = config.SCENARIOS[scenario_name]
    default_mult = scenario.get("_default", 1.0)
    
    # Save original demand for before/after comparison
    result["original_demand"] = result["predicted_demand"].copy()
    
    # Apply per-department multipliers
    result["multiplier_applied"] = default_mult
    for dept, mult in scenario.items():
        if dept == "_default":
            continue
        mask = result["department"] == dept
        result.loc[mask, "multiplier_applied"] = mult
    
    # Apply multiplier to demand
    result["predicted_demand"] = result["original_demand"] * result["multiplier_applied"]
    result["scenario"] = scenario_name
    
    return result

def apply_custom_multipliers(
    predictions_df: pd.DataFrame,
    multipliers: dict[str, float]
) -> pd.DataFrame:
    """Apply user-defined custom multipliers (from UI sliders).
    
    Args:
        predictions_df: DataFrame with predicted_demand and department
        multipliers: Dict of {department_name: float_multiplier}
            e.g., {"produce": 1.3, "beverages": 0.8}
            Departments not in the dict get multiplier=1.0
    
    Returns:
        DataFrame with adjusted predicted_demand + metadata columns
    """
    result = predictions_df.copy()
    result["original_demand"] = result["predicted_demand"].copy()
    result["multiplier_applied"] = 1.0
    
    for dept, mult in multipliers.items():
        mask = result["department"] == dept
        result.loc[mask, "multiplier_applied"] = mult
    
    result["predicted_demand"] = result["original_demand"] * result["multiplier_applied"]
    result["scenario"] = "Custom"
    
    return result

def compare_scenarios(
    predictions_df: pd.DataFrame,
    scenario_names: list[str] = None,
) -> pd.DataFrame:
    """Run all (or selected) scenarios and return a summary comparison.
    
    Returns:
        Summary DataFrame with one row per scenario, columns:
        - scenario, total_demand, demand_change_pct, 
          estimated_spoilage_cost, estimated_stockout_cost, total_cost
    """
    if scenario_names is None:
        scenario_names = list(config.SCENARIOS.keys())
    
    summaries = []
    
    # Get baseline demand for comparison
    base_demand = predictions_df["predicted_demand"].sum()
    
    for name in scenario_names:
        adjusted = apply_scenario(predictions_df, name)
        new_demand = adjusted["predicted_demand"].sum()
        change_pct = ((new_demand - base_demand) / base_demand * 100) if base_demand > 0 else 0
        
        # Estimate costs under this scenario
        actual = adjusted["units_sold"].values
        predicted = adjusted["predicted_demand"].values
        spoilage = float(np.maximum(0, predicted - actual).sum() * config.SPOILAGE_COST_PER_UNIT)
        stockout = float(np.maximum(0, actual - predicted).sum() * config.STOCKOUT_COST_PER_UNIT)
        
        summaries.append({
            "scenario": name,
            "total_demand": round(new_demand),
            "demand_change_pct": round(change_pct, 1),
            "spoilage_cost": round(spoilage, 2),
            "stockout_cost": round(stockout, 2),
            "total_cost": round(spoilage + stockout, 2),
        })
    
    return pd.DataFrame(summaries)

if __name__ == "__main__":
    from replenishment import build_replenishment_pipeline
    
    print("=" * 60)
    print("PHASE 3B: Simulation Engine")
    print("=" * 60)
    
    # Get base predictions
    base_df = build_replenishment_pipeline(use_baseline=True)
    base_demand = base_df["predicted_demand"].sum()
    print(f"Base total demand: {base_demand:,.0f}")
    
    # Test each scenario
    for name in config.SCENARIOS:
        adjusted = apply_scenario(base_df, name)
        new_demand = adjusted["predicted_demand"].sum()
        change = (new_demand - base_demand) / base_demand * 100
        print(f"  {name:20s} → demand: {new_demand:>12,.0f} ({change:+.1f}%)")
    
    # Test custom multipliers
    print("\nCustom scenario (produce×2.0, beverages×0.5):")
    custom = apply_custom_multipliers(base_df, {"produce": 2.0, "beverages": 0.5})
    print(f"  Total demand: {custom['predicted_demand'].sum():,.0f}")
    
    # Compare all scenarios
    print("\nScenario comparison table:")
    comparison = compare_scenarios(base_df)
    print(comparison.to_string(index=False))
    
    # Validation
    print("\n--- Validation ---")
    rainy = apply_scenario(base_df, "Rainy Day")
    assert (rainy["multiplier_applied"] == 0.7).all(), "Rainy day should apply 0.7 to ALL depts"
    
    ipl = apply_scenario(base_df, "IPL Match Evening")
    snack_mult = ipl[ipl["department"] == "snacks"]["multiplier_applied"].unique()
    other_mult = ipl[ipl["department"] == "frozen"]["multiplier_applied"].unique()
    assert 1.5 in snack_mult, f"Snacks should be 1.5x, got {snack_mult}"
    assert 1.0 in other_mult, f"Non-snack depts should be 1.0x, got {other_mult}"
    
    normal = apply_scenario(base_df, "Normal Day")
    assert np.allclose(normal["predicted_demand"].values, normal["original_demand"].values), \
        "Normal Day should not change demand"
    
    print("✅ All scenario validations passed!")
    print("\n✅ Phase 3B Complete!")
