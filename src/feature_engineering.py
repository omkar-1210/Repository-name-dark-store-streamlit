import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

def create_store_clusters(orders: pd.DataFrame, n_stores: int = 20) -> pd.DataFrame:
    """
    1. Aggregate user-level features:
       - avg_hour = mean(order_hour_of_day)
       - avg_dow = mean(order_dow)
       - order_count = count(order_id)
       - avg_gap = mean(days_since_prior_order)
    2. StandardScaler on these 4 features
    3. KMeans(n_clusters=20, random_state=42, n_init=10)
    4. Assign store_id = cluster labels
    
    Returns: DataFrame indexed by user_id with columns:
        [avg_hour, avg_dow, order_count, avg_gap, store_id]
    """
    # Filter to prior orders
    prior_orders = orders[orders['eval_set'] == 'prior']
    
    # Aggregate features
    user_features = prior_orders.groupby('user_id').agg(
        avg_hour=('order_hour_of_day', 'mean'),
        avg_dow=('order_dow', 'mean'),
        order_count=('order_id', 'count'),
        avg_gap=('days_since_prior_order', 'mean')
    ).fillna(0)
    
    # Standard scale
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(user_features)
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_stores, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    user_features['store_id'] = cluster_labels
    
    # Store models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'store_scaler.joblib'))
    joblib.dump(kmeans, os.path.join(models_dir, 'store_kmeans.joblib'))
    
    return user_features

def add_time_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map order_hour_of_day → time_bucket using config.TIME_BUCKETS.
    
    Buckets:
        0-3  → midnight
        4-7  → early_morning
        8-11 → morning
        12-15 → afternoon
        16-19 → evening
        20-23 → night
    """
    bins = [-1, 3, 7, 11, 15, 19, 23]
    labels = ["midnight", "early_morning", "morning", "afternoon", "evening", "night"]
    df["time_bucket"] = pd.cut(df["order_hour_of_day"], bins=bins, labels=labels)
    return df

def aggregate_demand(merged_df: pd.DataFrame, user_features: pd.DataFrame) -> pd.DataFrame:
    """
    1. Join merged_df with user_features on user_id → adds store_id
    2. Add time_bucket
    3. GroupBy [store_id, order_dow, time_bucket, department] → count = units_sold
    
    Returns: DataFrame with columns:
        [store_id, order_dow, time_bucket, department, units_sold]
    Shape: ~17,000-18,000 rows (20 stores × 21 depts × 7 days × 6 buckets, minus sparse combos)
    """
    merged_with_store = merged_df.merge(user_features[['store_id']], on='user_id', how='inner')
    merged_with_time = add_time_bucket(merged_with_store)
    
    demand = merged_with_time.groupby(['store_id', 'order_dow', 'time_bucket', 'department'], observed=True).size().reset_index(name="units_sold")
    demand = demand[demand["units_sold"] > 0]
    demand = demand.sort_values(["store_id", "department", "time_bucket", "order_dow"]).reset_index(drop=True)
    return demand

def add_lag_features(demand: pd.DataFrame) -> pd.DataFrame:
    """
    For each group of (store_id, department, time_bucket), compute:
    
    1. demand_lag_1w = shift(units_sold, 1)
       → "What was demand for this same store×dept×timeslot last DOW?"
       
    2. demand_roll_4w = rolling(units_sold, 4).mean()
       → "Average demand over the last 4 DOW periods"
       
    3. demand_std_4w = rolling(units_sold, 4).std()
       → "Demand volatility over last 4 DOW periods"
    
    These are the primary features for the LightGBM model.
    """
    grouped = demand.groupby(["store_id", "department", "time_bucket"], observed=True)["units_sold"]
    
    demand["demand_lag_1w"] = grouped.shift(1)
    demand["demand_roll_4w"] = grouped.transform(lambda x: x.rolling(4).mean())
    demand["demand_std_4w"] = grouped.transform(lambda x: x.rolling(4).std())
    
    return demand

def add_binary_features(demand: pd.DataFrame) -> pd.DataFrame:
    """
    Add 4 binary indicator columns:
    
    1. is_perishable: 1 if department in PERISHABLE_DEPARTMENTS, else 0
    2. is_weekend: 1 if order_dow in [0, 6], else 0
    3. is_morning: 1 if time_bucket == "morning", else 0
    4. is_evening: 1 if time_bucket == "evening", else 0
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        import config
        perishables = config.PERISHABLE_DEPARTMENTS
    except (ImportError, AttributeError):
        perishables = ["produce", "dairy eggs", "meat seafood", "bakery", "deli"]

    demand["is_perishable"] = demand["department"].isin(perishables).astype(int)
    demand["is_weekend"] = demand["order_dow"].isin([0, 6]).astype(int)
    demand["is_morning"] = (demand["time_bucket"] == "morning").astype(int)
    demand["is_evening"] = (demand["time_bucket"] == "evening").astype(int)
    
    return demand

def finalize_and_cache(demand: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop rows where lag/rolling features are NaN
       (first 1-3 rows per group — expected from rolling windows)
    2. Reset index
    3. Save to cache/processed_data.pkl
    4. Print final shape and column summary
    
    Returns: Clean DataFrame ready for LightGBM
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import config
    
    demand = demand.dropna(subset=["demand_lag_1w", "demand_roll_4w", "demand_std_4w"])
    demand = demand.reset_index(drop=True)
    
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    save_path = config.CACHE_DIR / "processed_data.pkl"
    demand.to_pickle(save_path)
    
    return demand

if __name__ == '__main__':
    from data_loader import load_raw_data, merge_all
    
    raw = load_raw_data()
    orders = raw['orders']
    
    print("Creating store clusters...")
    user_features = create_store_clusters(orders)
    
    print("\n--- Validation Assertions ---")
    try:
        assert user_features["store_id"].nunique() == 20, f"Expected 20 unique stores, got {user_features['store_id'].nunique()}"
        assert user_features.shape[0] > 200_000, f"Expected >200K users, got {user_features.shape[0]}"
        assert user_features["store_id"].min() == 0, f"Expected min store_id 0, got {user_features['store_id'].min()}"
        assert user_features["store_id"].max() == 19, f"Expected max store_id 19, got {user_features['store_id'].max()}"
        
        print("Testing time buckets...")
        orders_with_time = add_time_bucket(orders.copy())
        assert orders_with_time["time_bucket"].isna().sum() == 0, "Found NaNs in time bucket"
        assert orders_with_time["time_bucket"].nunique() == 6, f"Expected 6 time buckets, got {orders_with_time['time_bucket'].nunique()}"
        
        print("Test merging data for demand aggregation...")
        merged_df = merge_all(raw)
        
        print("Testing aggregate_demand...")
        demand = aggregate_demand(merged_df, user_features)
        assert demand.shape[0] > 15_000, f"Expected ~17K rows, got {demand.shape[0]}"
        assert demand.shape[1] == 5, f"Expected 5 cols, got {demand.shape[1]}"
        assert demand["store_id"].nunique() == 20, "Expected 20 stores in demand"
        assert demand["units_sold"].min() >= 1, "Expected no zero-count groups"
        assert demand.sort_values(["store_id","department","time_bucket","order_dow"]).equals(demand), "DataFrame must be sorted for lag features"
        
        print("Testing add_lag_features...")
        demand = add_lag_features(demand)
        
        # Spot check validation
        group = demand[
            (demand["store_id"] == 0) & 
            (demand["department"] == "alcohol") & 
            (demand["time_bucket"] == "afternoon")
        ].copy()
        
        if len(group) >= 2:
            assert group.iloc[1]["demand_lag_1w"] == group.iloc[0]["units_sold"]
        else:
            # Fallback spot-check if group is missing or sparse
            store, dept, tb = demand.iloc[0][["store_id", "department", "time_bucket"]]
            group = demand[
                (demand["store_id"] == store) & 
                (demand["department"] == dept) & 
                (demand["time_bucket"] == tb)
            ].copy()
            if len(group) >= 2:
                assert group.iloc[1]["demand_lag_1w"] == group.iloc[0]["units_sold"]
        
        print("Testing add_binary_features...")
        demand = add_binary_features(demand)
        assert demand["is_perishable"].sum() > 0
        assert demand[demand["department"] == "produce"]["is_perishable"].all()
        assert demand[demand["department"] == "alcohol"]["is_perishable"].sum() == 0
        assert demand["is_weekend"].dtype in [int, 'int32', 'int64']
        
        print("Testing finalize_and_cache...")
        demand = finalize_and_cache(demand)
        
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        import config
        
        final = pd.read_pickle(config.CACHE_DIR / "processed_data.pkl")
        assert final.isna().sum().sum() == 0, "No NaN allowed in final dataset"
        assert final.shape[0] > 8_000, f"Expected ~10K rows after dropping, got {final.shape[0]}"
        assert set(config.FEATURE_COLS + [config.TARGET_COL]).issubset(final.columns)
        print(f"✅ Phase 1 Complete: {final.shape[0]} rows × {final.shape[1]} cols")
        print(f"   Stores: {final['store_id'].nunique()}")
        print(f"   Departments: {final['department'].nunique()}")
        print(f"   Perishable rows: {final['is_perishable'].sum()} ({final['is_perishable'].mean():.1%})")
    except AssertionError as e:
        print(f"Validation FAILED: {e}")
