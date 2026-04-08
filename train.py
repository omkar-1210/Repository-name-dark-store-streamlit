from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ARTIFACTS_DIR, CACHE_PATH, DATA_DIR, METRICS_PATH, MODEL_PATH
from src.data_pipeline import ensure_artifact_dirs, load_and_prepare
from src.modeling import save_model_and_metrics, train_lgbm_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Train dark store demand model")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Folder containing raw Instacart CSV files")
    parser.add_argument("--artifacts-dir", type=str, default=str(ARTIFACTS_DIR), help="Folder for processed data and model artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)

    ensure_artifact_dirs(artifacts_dir)
    demand = load_and_prepare(data_dir=data_dir, artifacts_dir=artifacts_dir)

    model, metrics, _ = train_lgbm_model(demand)
    save_model_and_metrics(model, metrics)

    print("\nTraining complete")
    print(f"Processed data: {CACHE_PATH}")
    print(f"Model file    : {MODEL_PATH}")
    print(f"Metrics file  : {METRICS_PATH}")

if __name__ == "__main__":
    main()
