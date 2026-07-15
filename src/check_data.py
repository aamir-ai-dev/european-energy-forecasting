""" import pandas as pd

df = pd.read_parquet("data/processed/italy_load_hourly.parquet")
print(f"Shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Timezone: {df['timestamp'].dtype}")
print(f"Any NaN: {df.isna().sum()}") """
import pandas as pd
# Diagnostics
df = pd.read_parquet("data/processed/italy_load_hourly.parquet")
print("\n=== DATA QUALITY CHECKS ===")
print(f"Unique intervals: {df['timestamp'].diff().value_counts().head(10)}")
print(f"Rows with 1-hour intervals: {(df['timestamp'].diff() == pd.Timedelta(hours=1)).sum()}")
print(f"Duplicates in timestamp column: {df['timestamp'].duplicated().sum()}")
print(f"\nExpected ~26,280 hourly rows for 3 years; got {len(df)}")