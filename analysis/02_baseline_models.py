"""Baseline models for Italian electricity load forecasting."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_parquet("data/processed/italy_load_hourly.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

print("=== BASELINE MODELS ===\n")
print(f"Total rows: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")

# Time-series split: 70% train, 15% val, 15% test
n_total = len(df)
train_size = int(0.7 * n_total)
val_size = int(0.15 * n_total)

train = df[:train_size].copy()
val = df[train_size:train_size + val_size].copy()
test = df[train_size + val_size:].copy()

print(f"Train: {len(train)} rows ({train['timestamp'].min().date()} to {train['timestamp'].max().date()})")
print(f"Val:   {len(val)} rows ({val['timestamp'].min().date()} to {val['timestamp'].max().date()})")
print(f"Test:  {len(test)} rows ({test['timestamp'].min().date()} to {test['timestamp'].max().date()})\n")

# Combine train+val for training baselines (common practice)
train_val = pd.concat([train, val]).reset_index(drop=True)
###################################################################################
# Sanity check: verify lag-168 makes sense
print("=== DIAGNOSTIC: Lag-168 Check ===")
test_diagnostic = test.copy()
test_diagnostic["lag1"] = test_diagnostic["load_mw"].shift(1)
test_diagnostic["lag168"] = test_diagnostic["load_mw"].shift(168)

# Show a few rows where both lag1 and lag168 are not NaN
sample = test_diagnostic.dropna().head(10)
print("\nSample of lag-1 vs lag-168 (should match same day last week):")
print(sample[["timestamp", "load_mw", "lag1", "lag168"]])

# Verify lag-168 is actually ~7 days earlier
print("\nTime difference check (should be ~7 days = 168 hours):")
test_diagnostic_clean = test_diagnostic.dropna()
if len(test_diagnostic_clean) > 0:
    idx = test_diagnostic_clean.index[100]  # Pick a random row
    lag168_time_diff = test_diagnostic_clean.loc[idx, "timestamp"] - test.loc[idx-168, "timestamp"]
    print(f"Row {idx}: time diff = {lag168_time_diff}")
print()
####################################################################################

def evaluate(y_true, y_pred, model_name):
    """Compute MAE, RMSE, MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"{model_name:25s} | MAE: {mae:8.0f} MW | RMSE: {rmse:8.0f} MW | MAPE: {mape:6.2f}%")
    return {"model": model_name, "mae": mae, "rmse": rmse, "mape": mape}


results = []

# === BASELINE 1: NAIVE (yesterday's value) ===
print("=== BASELINE 1: NAIVE (lag-1) ===")
# y_pred[t] = y[t-1]
test_naive = test.copy()
test_naive["pred"] = test_naive["load_mw"].shift(1)
test_naive = test_naive.dropna()

results.append(evaluate(test_naive["load_mw"], test_naive["pred"], "Naive (lag-1)"))
print()

# === BASELINE 2: SEASONAL NAIVE (same hour last week) ===
print("=== BASELINE 2: SEASONAL NAIVE (lag-168) ===")
# y_pred[t] = y[t-168] (168 hours = 1 week)
test_seasonal = test.copy()
test_seasonal["pred"] = test_seasonal["load_mw"].shift(168)
test_seasonal = test_seasonal.dropna()

results.append(evaluate(test_seasonal["load_mw"], test_seasonal["pred"], "Seasonal Naive (lag-168)"))
print()

# === BASELINE 3: MOVING AVERAGE ===
print("=== BASELINE 3: MOVING AVERAGE (7-day) ===")
# y_pred[t] = mean(y[t-7*24:t]) (7-day rolling mean)
# Compute on entire dataset, evaluate on test
df_ma = df.copy()
df_ma["pred"] = df_ma["load_mw"].rolling(window=7*24, min_periods=1).mean()
test_ma = df_ma[df_ma.index >= train_size + val_size].copy()
test_ma = test_ma.dropna()

results.append(evaluate(test_ma["load_mw"], test_ma["pred"], "Moving Avg (7-day)"))
print()

# === BASELINE 4: LINEAR REGRESSION ===
print("=== BASELINE 4: LINEAR REGRESSION ===")
# Simple: predict load as f(hours_since_start, hour_of_day)
train_val_lr = train_val.copy()
train_val_lr["hours_since_start"] = (train_val_lr["timestamp"] - train_val_lr["timestamp"].min()).dt.total_seconds() / 3600
train_val_lr["hour_of_day"] = train_val_lr["timestamp"].dt.hour

test_lr = test.copy()
test_lr["hours_since_start"] = (test_lr["timestamp"] - train_val_lr["timestamp"].min()).dt.total_seconds() / 3600
test_lr["hour_of_day"] = test_lr["timestamp"].dt.hour

# Fit on train_val
X_train = train_val_lr[["hours_since_start", "hour_of_day"]]
y_train = train_val_lr["load_mw"]
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predict on test
X_test = test_lr[["hours_since_start", "hour_of_day"]]
test_lr["pred"] = model_lr.predict(X_test)

results.append(evaluate(test_lr["load_mw"], test_lr["pred"], "Linear Regression"))
print()

# === SUMMARY ===
print("=" * 80)
print("RESULTS SUMMARY - RANKED BY MAPE (lower is better)")
print("=" * 80)
results_df = pd.DataFrame(results).sort_values("mape").reset_index(drop=True)
print(results_df.to_string(index=False))
print()
print(f"BEST BASELINE: {results_df.iloc[0]['model']}")
print(f"   MAE:  {results_df.iloc[0]['mae']:.0f} MW")
print(f"   RMSE: {results_df.iloc[0]['rmse']:.0f} MW")
print(f"   MAPE: {results_df.iloc[0]['mape']:.2f}%")
print()
print("Use this as your reference benchmark.")
print("Your ML models (XGBoost, LightGBM) must beat this to be useful.")