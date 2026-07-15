"""Fetch Italian hourly electricity load from ENTSO-E."""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()

# Config
API_TOKEN = os.environ["ENTSOE_API_TOKEN"]
COUNTRY_CODE = "IT"  # Italy
START = pd.Timestamp("2023-01-01", tz="Europe/Rome")
END = pd.Timestamp("2026-01-01", tz="Europe/Rome")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_load():
    """Pull hourly load data for Italy."""
    client = EntsoePandasClient(api_key=API_TOKEN)
    print(f"Fetching load data for {COUNTRY_CODE} from {START.date()} to {END.date()}...")
    load = client.query_load(COUNTRY_CODE, start=START, end=END)
    print(f"Received {len(load)} rows.")
    return load


def main():
    load = fetch_load()

    # Save raw CSV for inspection
    raw_path = RAW_DIR / f"entsoe_{COUNTRY_CODE}_load_{START.date()}_{END.date()}.csv"
    load.to_csv(raw_path)
    print(f"Saved raw CSV: {raw_path}")

    # Convert to clean DataFrame
    df = load.reset_index()
    df.columns = ["timestamp", "load_mw"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Filter to hourly data only (drop 15-min, 30-min, 45-min intervals)
    # Keep only rows where minute == 0
    df = df[df["timestamp"].dt.minute == 0].copy()
    df = df.reset_index(drop=True)

    print(f"\nAfter filtering to hourly only: {len(df)} rows")
    print(f"Expected ~26,280 for 3 years. Difference: {26280 - len(df)}")

    # Save as Parquet
    processed_path = PROCESSED_DIR / "italy_load_hourly.parquet"
    df.to_parquet(processed_path, index=False)
    print(f"Saved processed Parquet: {processed_path}")
    
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
    print(f"\nStats:\n{df['load_mw'].describe()}")


if __name__ == "__main__":
    main()
    