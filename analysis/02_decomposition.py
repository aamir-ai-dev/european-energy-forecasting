"""Time-series decomposition of Italian electricity load."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL

df = pd.read_parquet("data/processed/italy_load_hourly.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

print("=== TIME-SERIES DECOMPOSITION ===\n")
print(f"Total rows: {len(df)}")
print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}\n")

# STL works on a Series with a DatetimeIndex
series = df.set_index("timestamp")["load_mw"]

# === STL DECOMPOSITION ===
# period=168 means one full weekly cycle (24 hours x 7 days)
# seasonal=25 controls smoothness of the seasonal component (must be odd)
print("Running STL decomposition (period=168 hours = 1 week)...")
stl = STL(series, period=168, seasonal=25, robust=True)
result = stl.fit()

trend    = result.trend
seasonal = result.seasonal
residual = result.resid

print("Decomposition complete.\n")

# === SUMMARY STATISTICS ===
print("=== COMPONENT SUMMARY ===\n")

total_variance = np.var(series)
trend_variance    = np.var(trend)
seasonal_variance = np.var(seasonal)
residual_variance = np.var(residual)

print(f"Total load variance:    {total_variance:,.0f} MW^2")
print(f"Trend variance:         {trend_variance:,.0f} MW^2  ({100*trend_variance/total_variance:.1f}% of total)")
print(f"Seasonal variance:      {seasonal_variance:,.0f} MW^2  ({100*seasonal_variance/total_variance:.1f}% of total)")
print(f"Residual variance:      {residual_variance:,.0f} MW^2  ({100*residual_variance/total_variance:.1f}% of total)")

print(f"\nTrend range:    {trend.min():,.0f} MW to {trend.max():,.0f} MW  (swing: {trend.max()-trend.min():,.0f} MW)")
print(f"Seasonal range: {seasonal.min():,.0f} MW to {seasonal.max():,.0f} MW  (swing: {seasonal.max()-seasonal.min():,.0f} MW)")
print(f"Residual range: {residual.min():,.0f} MW to {residual.max():,.0f} MW")
print(f"Residual std:   {residual.std():,.0f} MW")

# === DAILY SEASONAL PATTERN ===
print("\n=== DAILY PATTERN (avg seasonal component by hour) ===\n")
seasonal_df = seasonal.reset_index()
seasonal_df.columns = ["timestamp", "seasonal"]
seasonal_df["hour"] = seasonal_df["timestamp"].dt.hour
hourly_seasonal = seasonal_df.groupby("hour")["seasonal"].mean()

print("Hour (UTC) | Avg Seasonal Component")
print("-" * 35)
for hour, val in hourly_seasonal.items():
    bar = "#" * int(abs(val) / 200)
    sign = "+" if val >= 0 else "-"
    print(f"  {hour:02d}:00    | {sign}{abs(val):6.0f} MW  {bar}")

# === PLOTS ===
print("\n=== GENERATING PLOTS ===")

fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
fig.suptitle("STL Decomposition — Italian Electricity Load (2023-2025)",
             fontsize=14, fontweight="bold", y=0.98)

# Plot only 6 months for readability on the decomposition plot
mask = (series.index >= "2024-01-01") & (series.index < "2024-07-01")

axes[0].plot(series[mask].index, series[mask].values, color="steelblue", linewidth=0.6)
axes[0].set_ylabel("Observed (MW)", fontsize=10)
axes[0].set_title("Observed", fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(trend[mask].index, trend[mask].values, color="darkorange", linewidth=1.2)
axes[1].set_ylabel("Trend (MW)", fontsize=10)
axes[1].set_title("Trend (long-term direction)", fontsize=11)
axes[1].grid(True, alpha=0.3)

axes[2].plot(seasonal[mask].index, seasonal[mask].values, color="green", linewidth=0.6)
axes[2].set_ylabel("Seasonal (MW)", fontsize=10)
axes[2].set_title("Seasonal (weekly cycle)", fontsize=11)
axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[2].grid(True, alpha=0.3)

axes[3].plot(residual[mask].index, residual[mask].values, color="red", linewidth=0.6, alpha=0.7)
axes[3].set_ylabel("Residual (MW)", fontsize=10)
axes[3].set_title("Residual (unexplained noise)", fontsize=11)
axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[3].grid(True, alpha=0.3)

axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[3].xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("analysis/02_decomposition.png", dpi=100, bbox_inches="tight")
print("Saved: analysis/02_decomposition.png")

# === TREND OVER FULL 3 YEARS ===
fig2, ax = plt.subplots(figsize=(14, 5))
ax.plot(series.index, series.values, color="lightsteelblue", linewidth=0.4, alpha=0.6, label="Observed")
ax.plot(trend.index, trend.values, color="darkorange", linewidth=2, label="Trend")
ax.set_title("Full 3-Year Trend vs Observed Load", fontsize=12, fontweight="bold")
ax.set_ylabel("Load (MW)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("analysis/02_trend.png", dpi=100, bbox_inches="tight")
print("Saved: analysis/02_trend.png")

print("\n=== KEY INSIGHTS ===")
print(f"1. Seasonal component explains {100*seasonal_variance/total_variance:.1f}% of total variance — weekly cycle dominates")
print(f"2. Trend explains {100*trend_variance/total_variance:.1f}% — long-term direction is relatively stable")
print(f"3. Residual explains {100*residual_variance/total_variance:.1f}% — this is what models must learn to reduce")
print(f"4. Peak seasonal hour: {hourly_seasonal.idxmax():02d}:00 UTC (+{hourly_seasonal.max():,.0f} MW above trend)")
print(f"5. Trough seasonal hour: {hourly_seasonal.idxmin():02d}:00 UTC ({hourly_seasonal.min():,.0f} MW below trend)")
