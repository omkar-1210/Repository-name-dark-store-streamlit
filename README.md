# Dark Store Demand Forecasting Streamlit Project

This project converts the notebook into a clean, modular Python project.

## Folder structure

```text
dark_store_streamlit_project/
├── app.py
├── train.py
├── requirements.txt
├── data/
│   └── raw/
├── artifacts/
│   ├── models/
│   └── cache/
└── src/
```

## Required input files

Place these files inside `data/raw/`:

- `aisles.csv`
- `departments.csv`
- `products.csv`
- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`

## Install

```bash
pip install -r requirements.txt
```

## Train the pipeline

```bash
python train.py
```

This will create:

- `artifacts/cache/processed_data.pkl`
- `artifacts/models/lgbm_demand.pkl`
- `artifacts/models/model_metrics.json`
- `artifacts/models/store_scaler.joblib`
- `artifacts/models/store_kmeans.joblib`
- `artifacts/cache/shap_values.pkl` when needed

## Run Streamlit locally

```bash
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this folder to GitHub.
2. Keep `app.py`, `src/`, `requirements.txt`, and `artifacts/` in the repo.
3. Make sure `artifacts/` already contains the trained model and processed pickle.
4. In Streamlit Cloud, point the app to `app.py`.

## Notes

- Training is done once with `python train.py`.
- The Streamlit app only reads saved artifacts.
- If you change the dataset or model code, run `python train.py` again.
