"""Exploratory Data Analysis of Italian electricity load."""

import pandas as pd
import matplotlib.pyplot as plt

# Load the clean data
df = pd.read_parquet("data/processed/italy_load_hourly.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

print("=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Load range: {df['load_mw'].min():.0f} MW to {df['load_mw'].max():.0f} MW")
print(f"Mean load: {df['load_mw'].mean():.0f} MW")
print(f"Std dev: {df['load_mw'].std():.0f} MW")

# Check for missing values at high level
print(f"\nMissing values: {df.isna().sum().sum()}")

# === DATA VALIDATION ===
print("\n=== DATA VALIDATION CHECKS ===\n")

# 1. Check for gaps (missing hours)
df_sorted = df.sort_values("timestamp").reset_index(drop=True)
time_diffs = df_sorted["timestamp"].diff()

expected_interval = pd.Timedelta(hours=1)
unexpected = time_diffs[time_diffs != expected_interval]

if len(unexpected) > 0:
    print("WARNING: GAPS OR UNEXPECTED INTERVALS DETECTED:")
    print(f"  Found {len(unexpected)} non-1-hour intervals")
    print("\n  Details of anomalies:")
    for idx, diff in unexpected.items():
        if idx > 0:
            timestamp = df_sorted.loc[idx, "timestamp"]
            prev_timestamp = df_sorted.loc[idx-1, "timestamp"]
            print(f"    {prev_timestamp} -> {timestamp}: {diff}")
else:
    print("OK: No gaps. All intervals are exactly 1 hour.")

# 2. Check for duplicates
duplicates = df[df.duplicated(subset=["timestamp"], keep=False)]
if len(duplicates) > 0:
    print(f"\nWARNING: DUPLICATES DETECTED: {len(duplicates)} rows")
    print(duplicates)
else:
    print("OK: No duplicate timestamps.")

# 3. Check for null/NaN values
null_counts = df.isna().sum()
if null_counts.sum() > 0:
    print(f"\nWARNING: MISSING VALUES DETECTED:")
    print(null_counts[null_counts > 0])
else:
    print("OK: No missing values.")

# 4. Check for outliers (values way outside normal range)
Q1 = df["load_mw"].quantile(0.25)
Q3 = df["load_mw"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR  # 3x IQR is aggressive
upper_bound = Q3 + 3 * IQR

outliers = df[(df["load_mw"] < lower_bound) | (df["load_mw"] > upper_bound)]
if len(outliers) > 0:
    print(f"\nWARNING: OUTLIERS DETECTED ({len(outliers)} rows outside 3*IQR):")
    print(outliers[["timestamp", "load_mw"]])
else:
    print("OK: No extreme outliers.")

# 5. Timestamp ordering
if df["timestamp"].is_monotonic_increasing:
    print("OK: Timestamps are sorted and in order.")
else:
    print("WARNING: Timestamps are NOT in order. This will break lag features.")

print(f"\n=== SUMMARY ===")
print(f"Total rows: {len(df)}")
print(f"Date span: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Expected rows (3 years * 365.25 * 24): ~26,280")
print(f"Actual rows: {len(df)}")
print(f"Difference: {len(df) - 26280} (DST adjustments are OK)")

# === VISUALIZATIONS ===
print("\n=== GENERATING PLOTS ===\n")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 1. Time series
axes[0].plot(df["timestamp"], df["load_mw"], linewidth=0.5, color="steelblue")
axes[0].set_title("Italian Hourly Load (2023-2026)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Load (MW)", fontsize=11)
axes[0].grid(True, alpha=0.3)

# 2. Distribution
axes[1].hist(df["load_mw"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
axes[1].set_title("Distribution of Hourly Load", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Load (MW)", fontsize=11)
axes[1].grid(True, alpha=0.3, axis="y")

# 3. Seasonal pattern (avg by hour of day)
df["hour"] = df["timestamp"].dt.hour
hourly_avg = df.groupby("hour")["load_mw"].mean()
axes[2].plot(hourly_avg.index, hourly_avg.values, marker="o", linewidth=2, 
             markersize=6, color="steelblue")
axes[2].set_title("Average Load by Hour of Day (UTC)", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Hour (UTC)", fontsize=11)
axes[2].set_ylabel("Average Load (MW)", fontsize=11)
axes[2].set_xticks(range(0, 24, 2))
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis/01_eda.png", dpi=100, bbox_inches="tight")
print("Plots saved to analysis/01_eda.png\n")

# Additional insights
print("=== KEY INSIGHTS ===")
print(f"Lowest average load hour: {hourly_avg.idxmin()}:00 UTC ({hourly_avg.min():.0f} MW)")
print(f"Highest average load hour: {hourly_avg.idxmax()}:00 UTC ({hourly_avg.max():.0f} MW)")
print(f"Daily range: {hourly_avg.max() - hourly_avg.min():.0f} MW")
print(f"Coefficient of variation: {(df['load_mw'].std() / df['load_mw'].mean()) * 100:.1f}%")