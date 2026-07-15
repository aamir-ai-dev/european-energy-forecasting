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

# Check for missing values
print(f"\nMissing values: {df.isna().sum().sum()}")

# Simple plots
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Time series
axes[0].plot(df["timestamp"], df["load_mw"], linewidth=0.5)
axes[0].set_title("Italian Hourly Load (2023-2026)")
axes[0].set_ylabel("Load (MW)")
axes[0].grid(True, alpha=0.3)

# Distribution
axes[1].hist(df["load_mw"], bins=50, edgecolor="black", alpha=0.7)
axes[1].set_title("Distribution of Hourly Load")
axes[1].set_xlabel("Load (MW)")
axes[1].grid(True, alpha=0.3)

# Seasonal pattern (avg by hour of day)
df["hour"] = df["timestamp"].dt.hour
hourly_avg = df.groupby("hour")["load_mw"].mean()
axes[2].plot(hourly_avg.index, hourly_avg.values, marker="o")
axes[2].set_title("Average Load by Hour of Day")
axes[2].set_xlabel("Hour (UTC)")
axes[2].set_ylabel("Average Load (MW)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis/01_eda.png", dpi=100)
print("\nPlots saved to analysis/01_eda.png")