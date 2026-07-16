# Italian Electricity Load Forecasting

End-to-end electricity load forecasting for Italy using public ENTSO-E data.

## Project Goal

Build a defensible, production-ready forecasting pipeline for Italian grid load. Not a tutorial. Something to walk through in technical depth.

## Data

Source: ENTSO-E Transparency Platform API (entsoe-py)
Target: Italian total hourly load
Range: 2021-01-01 to 2024-01-01 (3 years, 26,302 hourly observations)
Format: Raw CSV (audit trail) + Parquet (working data)

## Week 1 Complete

- [x] Data ingestion pipeline (`src/fetch_data.py`)
- [x] Data validation and cleaning (3 year-boundary gaps interpolated)
- [x] EDA with plots (`analysis/01_eda.py`)
- [x] Project structure (src/, analysis/, data/, models/)

### Key Insights

- Strong seasonal pattern: 20k-50k MW swing between summer and winter
- Clear intraday pattern: peak at 18:00 UTC, trough at 02:00 UTC
- No outliers, no duplicates, clean data ready for modeling

## Next: Week 2 Baseline Models

- Naive, seasonal naive, moving average, linear regression
- Evaluation framework (MAE, RMSE, MAPE)
- Backtesting setup