from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SCENARIOS, SPOILAGE_COST, STOCKOUT_COST

def apply_scenario(df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    scenario = SCENARIOS[scenario_name]
    default_mult = scenario.get("_default", 1.0)

    out = df.copy()
    out["original_demand"] = out["predicted_demand"].copy()
    out["multiplier_applied"] = default_mult

    for department, multiplier in scenario.items():
        if department == "_default":
            continue
        out.loc[out["department"] == department, "multiplier_applied"] = multiplier

    out["predicted_demand"] = out["original_demand"] * out["multiplier_applied"]
    out["scenario"] = scenario_name
    return out


def apply_custom_multipliers(df: pd.DataFrame, multipliers: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["original_demand"] = out["predicted_demand"].copy()
    out["multiplier_applied"] = 1.0

    for department, multiplier in multipliers.items():
        out.loc[out["department"] == department, "multiplier_applied"] = multiplier

    out["predicted_demand"] = out["original_demand"] * out["multiplier_applied"]
    out["scenario"] = "Custom"
    return out


def compare_scenarios(predictions_df: pd.DataFrame, scenario_names: list[str] | None = None) -> pd.DataFrame:
    if scenario_names is None:
        scenario_names = list(SCENARIOS.keys())

    base_demand = predictions_df["predicted_demand"].sum()
    rows = []

    for name in scenario_names:
        adjusted = apply_scenario(predictions_df, name)
        new_demand = adjusted["predicted_demand"].sum()
        change_pct = ((new_demand - base_demand) / base_demand * 100) if base_demand > 0 else 0.0

        actual = adjusted["units_sold"].values
        pred = adjusted["predicted_demand"].values
        spoilage = float(np.maximum(0, pred - actual).sum() * SPOILAGE_COST)
        stockout = float(np.maximum(0, actual - pred).sum() * STOCKOUT_COST)

        rows.append(
            {
                "scenario": name,
                "total_demand": round(new_demand),
                "demand_change_pct": round(change_pct, 1),
                "spoilage_cost": round(spoilage, 2),
                "stockout_cost": round(stockout, 2),
                "total_cost": round(spoilage + stockout, 2),
            }
        )

    return pd.DataFrame(rows)
