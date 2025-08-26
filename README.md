# LightGBM Consumption Forecasting

## Overview
This script (`forecast_consumption.py`) forecasts electricity consumption using LightGBM models for system-wide, consumer-specific, and group-specific predictions. It processes meter reading and temperature data to generate daily consumption forecasts for 100 days into the future, clustering consumers into three groups based on average consumption.

## Purpose
- **System-wide Forecasting**: Predicts average daily consumption across all consumers.
- **Consumer-specific Forecasting**: Predicts daily consumption for individual consumers.
- **Group-specific Forecasting**: Predicts average daily consumption for consumer clusters (low, medium, high consumption).
- **Use Case**: Supports energy planning, billing, and demand forecasting for utility providers.

## Inputs
### Data Sources
- **Meter Readings**:
  - Directory: `csv/`
  - Files: CSV files matching `*reading*.csv`
  - Columns:
    - `consumer_id`: Unique identifier for each consumer (int64).
    - `reading_date`: Date of meter reading (YYYY-MM-DD).
    - `reading`: Meter reading in Wh (converted to kWh by dividing by 1000).
  - Format: CSV, one or more files.
- **Temperature Data**:
  - File: `csv/temperature.csv`
  - Columns:
    - `date`: Date of temperature reading (YYYY-MM-DD).
    - `tavg`: Average daily temperature (°C).
  - Format: CSV, optional (defaults to 20.0°C if missing).

### Data Requirements
- Reading dates must be parseable as `YYYY-MM-DD`.
- Consumers must have ≥500 unique days of readings to be included.
- Consumption is capped at 0–100 kWh/day.
- Negative consumption values are removed.
- Outliers are removed per consumer group using IQR (1.5 × IQR rule).

## Outputs
### Forecast Files
- **System-wide**:
  - `lightgbm_system_forecast.csv`: 100-day forecast of average daily consumption.
    - Columns: `ds` (date), `yhat` (predicted kWh).
- **Consumer-specific**:
  - `lightgbm_consumer_forecast.csv`: 100-day forecast for each consumer.
    - Columns: `ds` (date), `consumer_id`, `group`, `yhat` (predicted kWh).
  - `lightgbm_consumer_avg_forecast.csv`: Average of consumer-specific forecasts.
    - Columns: `ds` (date), `yhat_avg` (average predicted kWh).
- **Group-specific**:
  - `lightgbm_group_{0,1,2}_forecast.csv`: 100-day forecast for each consumer group.
    - Columns: `ds` (date), `yhat` (predicted kWh).

### Visualizations
- `lightgbm_system_forecast.png`: Plot of historical and forecasted system-wide and consumer-specific average consumption.
- `lightgbm_group_forecast.png`: Plot of historical and forecasted group-specific consumption.

### Model Information
- `lightgbm_system_model_info.csv`: System-wide model metrics and metadata.
  - Columns: `rmse`, `mae`, `mae_median_ratio`, `features_used`, `data_points`, `date_range`, `params`.
- `lightgbm_consumer_model_info.csv`: Consumer-specific model metrics and metadata.
  - Columns: Same as above.
- `lightgbm_group_{0,1,2}_model_info.csv`: Group-specific model metrics and metadata.
  - Columns: `group`, `rmse`, `mae`, `mae_median_ratio`, `features_used`, `data_points`, `date_range`, `params`.

### Saved Models
- `lightgbm_system_model_{timestamp}.txt`: System-wide LightGBM model.
- `lightgbm_consumer_model_{timestamp}.txt`: Consumer-specific LightGBM model.
- `lightgbm_group_{0,1,2}_model_{timestamp}.txt`: Group-specific LightGBM models.
- Timestamp format: `YYYYMMDD_HHMMSS`.

### Logs
- `lightgbm_training.log`: Detailed execution log (data shapes, metrics, errors).
- `model_performance.log`: Summary of final metrics for monitoring.

## Features Used
- `avg_temp`: Average daily temperature (°C).
- `year`, `month`, `day_of_week`, `day_of_year`, `week_of_year`: Temporal features.
- `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`: Rolling mean consumption (7, 14, 30 days).
- `consumer_id`, `group`: Categorical features for consumer-specific and group-specific models.

## Assumptions
- Meter readings are in Wh (converted to kWh).
- Consumption is capped at 0–100 kWh/day.
- Consumers with <500 days of data are excluded.
- K-means clustering with 3 clusters (random_state=42) for consumer grouping.
- Temperature data is optional; defaults to 20.0°C if missing.
- Missing rolling means are filled with current consumption.
- Outliers are removed per group using IQR.

## Setup
### Prerequisites
- Python 3.8+
- Libraries: `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `dask[dataframe]`, `matplotlib`
- Install dependencies:
  ```bash
  pip install pandas numpy lightgbm scikit-learn dask[dataframe] matplotlib