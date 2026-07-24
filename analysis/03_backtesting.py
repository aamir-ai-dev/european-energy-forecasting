"""Backtesting framework for time-series forecasting models."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_parquet("data/processed/italy_load_hourly.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

print("=== BACKTESTING FRAMEWORK ===\n")
print(f"Total rows: {len(df)}")
print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}\n")


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def naive_lag1(train_df, test_df):
    """Predict each hour using the previous hour's value."""
    preds = test_df["load_mw"].shift(1).bfill()
    return preds.values


def seasonal_naive_lag168(train_df, test_df):
    """Predict each hour using the same hour from one week ago."""
    # Combine train + test so we can look back 168 hours from the start of test
    combined = pd.concat([train_df, test_df]).reset_index(drop=True)
    test_start_idx = len(train_df)
    preds = []
    for i in range(len(test_df)):
        idx = test_start_idx + i
        lag_idx = idx - 168
        if lag_idx >= 0:
            preds.append(combined.loc[lag_idx, "load_mw"])
        else:
            preds.append(combined.loc[test_start_idx + i, "load_mw"])
    return np.array(preds)


# === WALK-FORWARD BACKTESTING ===
#
# Instead of one single train/test split, we:
# 1. Start with an initial training window
# 2. Evaluate on the NEXT fixed window (test fold)
# 3. Slide both windows forward by one fold size
# 4. Repeat until we reach the end of data
# 5. Average the metrics across all folds
#
# This gives a much more honest estimate of real-world performance
# because the model is tested on MANY different time periods,
# not just one lucky (or unlucky) slice.

INITIAL_TRAIN_HOURS = 8760   # 1 year of initial training data
FOLD_SIZE_HOURS     = 720    # evaluate on 30-day windows (~1 month)
N_FOLDS             = 10     # number of sliding windows to test

print(f"Backtesting configuration:")
print(f"  Initial training window : {INITIAL_TRAIN_HOURS} hours ({INITIAL_TRAIN_HOURS//24} days)")
print(f"  Fold size               : {FOLD_SIZE_HOURS} hours ({FOLD_SIZE_HOURS//24} days)")
print(f"  Number of folds         : {N_FOLDS}")
print(f"  Total data evaluated    : {N_FOLDS * FOLD_SIZE_HOURS} hours ({N_FOLDS * FOLD_SIZE_HOURS // 24} days)\n")

models = {
    "Naive (lag-1)":          naive_lag1,
    "Seasonal Naive (lag-168)": seasonal_naive_lag168,
}

all_results = {name: [] for name in models}

print(f"{'Fold':<6} {'Test Period':<35} {'Naive MAE':>10} {'Naive MAPE':>11} {'Seasonal MAE':>13} {'Seasonal MAPE':>14}")
print("-" * 95)

for fold in range(N_FOLDS):
    train_end   = INITIAL_TRAIN_HOURS + fold * FOLD_SIZE_HOURS
    test_start  = train_end
    test_end    = test_start + FOLD_SIZE_HOURS

    if test_end > len(df):
        break

    train_df = df.iloc[:train_end].copy()
    test_df  = df.iloc[test_start:test_end].copy()

    test_start_date = test_df["timestamp"].iloc[0].strftime("%Y-%m-%d")
    test_end_date   = test_df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
    period_str      = f"{test_start_date} to {test_end_date}"

    fold_metrics = {}
    for name, model_fn in models.items():
        preds   = model_fn(train_df, test_df)
        y_true  = test_df["load_mw"].values
        metrics = compute_metrics(y_true, preds)
        fold_metrics[name] = metrics
        all_results[name].append({
            "fold": fold + 1,
            "test_start": test_df["timestamp"].iloc[0],
            "test_end":   test_df["timestamp"].iloc[-1],
            **metrics
        })

    naive_m    = fold_metrics["Naive (lag-1)"]
    seasonal_m = fold_metrics["Seasonal Naive (lag-168)"]
    print(f"  {fold+1:<4} {period_str:<35} "
          f"{naive_m['mae']:>9.0f}  "
          f"{naive_m['mape']:>9.2f}%  "
          f"{seasonal_m['mae']:>11.0f}  "
          f"{seasonal_m['mape']:>12.2f}%")

# === AGGREGATE RESULTS ===
print("\n" + "=" * 60)
print("AGGREGATE RESULTS ACROSS ALL FOLDS")
print("=" * 60)

summary_rows = []
for name, fold_list in all_results.items():
    maes  = [f["mae"]  for f in fold_list]
    rmses = [f["rmse"] for f in fold_list]
    mapes = [f["mape"] for f in fold_list]
    summary_rows.append({
        "Model":     name,
        "MAE mean":  np.mean(maes),
        "MAE std":   np.std(maes),
        "RMSE mean": np.mean(rmses),
        "MAPE mean": np.mean(mapes),
        "MAPE std":  np.std(mapes),
        "MAPE min":  np.min(mapes),
        "MAPE max":  np.max(mapes),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("MAPE mean")
print(summary_df.to_string(index=False, float_format="{:.2f}".format))

print("\n")
for row in summary_rows:
    print(f"{row['Model']}:")
    print(f"  MAE:  {row['MAE mean']:,.0f} MW  (+/- {row['MAE std']:,.0f} MW across folds)")
    print(f"  MAPE: {row['MAPE mean']:.2f}%  (+/- {row['MAPE std']:.2f}%  |  range: {row['MAPE min']:.2f}% - {row['MAPE max']:.2f}%)")
    print()

# === PLOTS ===
print("=== GENERATING PLOTS ===")

fig, axes = plt.subplots(2, 1, figsize=(14, 9))
fig.suptitle("Backtesting Results — MAPE by Fold", fontsize=13, fontweight="bold")

colors = {"Naive (lag-1)": "steelblue", "Seasonal Naive (lag-168)": "darkorange"}

for name, fold_list in all_results.items():
    folds = [f["fold"] for f in fold_list]
    mapes = [f["mape"] for f in fold_list]
    maes  = [f["mae"]  for f in fold_list]

    axes[0].plot(folds, mapes, marker="o", label=name, color=colors[name], linewidth=2)
    axes[1].plot(folds, maes,  marker="o", label=name, color=colors[name], linewidth=2)

axes[0].set_title("MAPE per Fold (lower is better)", fontsize=11)
axes[0].set_ylabel("MAPE (%)", fontsize=10)
axes[0].set_xlabel("Fold", fontsize=10)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range(1, N_FOLDS + 1))

axes[1].set_title("MAE per Fold (lower is better)", fontsize=11)
axes[1].set_ylabel("MAE (MW)", fontsize=10)
axes[1].set_xlabel("Fold", fontsize=10)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(1, N_FOLDS + 1))

plt.tight_layout()
plt.savefig("analysis/03_backtesting.png", dpi=100, bbox_inches="tight")
print("Saved: analysis/03_backtesting.png")

print("\n=== KEY INSIGHTS ===")
best = summary_df.iloc[0]
worst = summary_df.iloc[-1]
print(f"1. Best model on average: {best['Model']} with MAPE {best['MAPE mean']:.2f}%")
print(f"2. Variance in performance: MAPE ranges {best['MAPE min']:.2f}% to {best['MAPE max']:.2f}% across folds")
print(f"3. This tells you how stable the model is — high variance means it is sensitive to which time period you test on")
print(f"4. Your ML models must beat {best['MAPE mean']:.2f}% MAPE consistently across folds, not just on one test window")
