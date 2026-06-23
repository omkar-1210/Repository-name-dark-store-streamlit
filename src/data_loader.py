import pandas as pd
import os
import gc

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'instacart')

def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load all 6 Instacart CSVs into a dict keyed by name."""
    
    dtype_orders = {
        "order_id": "int32",
        "user_id": "int32",
        "order_number": "int16",
        "order_dow": "int8",
        "order_hour_of_day": "int8",
    }
    
    dtype_order_products = {
        "order_id": "int32",
        "product_id": "int32",
        "add_to_cart_order": "int16",
        "reordered": "int8"
    }

    raw = {}
    
    file_map = {
        "orders": ("orders.csv", dtype_orders),
        "order_products_prior": ("order_products__prior.csv", dtype_order_products),
        "order_products_train": ("order_products__train.csv", dtype_order_products),
        "products": ("products.csv", {"product_id": "int32", "aisle_id": "int16", "department_id": "int16"}),
        "departments": ("departments.csv", {"department_id": "int16"}),
        "aisles": ("aisles.csv", {"aisle_id": "int16"})
    }

    for key, (filename, dtypes) in file_map.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            raw[key] = pd.read_csv(filepath, dtype=dtypes)
        else:
            print(f"File not found: {filepath}")

    return raw

def merge_all(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge pipeline:
    1. order_products_prior + orders → on order_id (inner)
    2. result + products → on product_id (inner)
    3. result + departments → on department_id (inner)
    
    Returns: DataFrame with columns:
        order_id, product_id, add_to_cart_order, reordered,
        user_id, eval_set, order_number, order_dow,
        order_hour_of_day, days_since_prior_order,
        product_name, aisle_id, department_id, department
    """
    
    print(f"Shape before merge 1 - order_products_prior: {raw['order_products_prior'].shape}, orders: {raw['orders'].shape}")
    merged = pd.merge(raw['order_products_prior'], raw['orders'], on='order_id', how='inner')
    print(f"Shape after merge 1: {merged.shape}")
    
    merged['days_since_prior_order'] = merged['days_since_prior_order'].fillna(15.0)
    
    print(f"Shape before merge 2 - current: {merged.shape}, products: {raw['products'].shape}")
    merged = pd.merge(merged, raw['products'], on='product_id', how='inner')
    print(f"Shape after merge 2: {merged.shape}")
    
    print(f"Shape before merge 3 - current: {merged.shape}, departments: {raw['departments'].shape}")
    merged = pd.merge(merged, raw['departments'], on='department_id', how='inner')
    print(f"Shape after merge 3: {merged.shape}")

    return merged

if __name__ == '__main__':
    raw_data = load_raw_data()
    
    if len(raw_data) > 0:
        merged = merge_all(raw_data)
        
        print("\n--- Validation Assertions ---")
        try:
            assert merged.shape[0] > 30_000_000, f"Expected 32M+ rows, got {merged.shape[0]}"
            assert merged.shape[1] == 14, f"Expected 14 cols, got {merged.shape[1]}"
            assert merged["days_since_prior_order"].isna().sum() == 0, "NaNs remain in days_since_prior_order"
            assert "department" in merged.columns
            print("All validations passed successfully!")
            print(f"Final dataset shape: {merged.shape}")
        except AssertionError as e:
            print(f"Validation FAILED: {e}")
    else:
        print("No raw data found to process.")
