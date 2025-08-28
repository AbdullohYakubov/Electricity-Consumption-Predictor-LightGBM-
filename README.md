# ⚡ LightGBM Consumption Forecasting

## 📖 Overview
This repository contains **`forecast_consumption.py`**, a forecasting pipeline using **LightGBM** to predict electricity consumption at three levels:

- **System-wide** – Average daily consumption across all consumers.  
- **Consumer-specific** – Daily consumption for individual consumers.  
- **Group-specific** – Average daily consumption for consumer clusters (low, medium, high).  

The pipeline processes **meter readings** and **temperature data**, generates **100-day forecasts**, and clusters consumers into **3 groups** based on average daily consumption.  

---

## 🎯 Purpose
- **System-wide Forecasting** – Energy planning for utilities.  
- **Consumer-specific Forecasting** – Personalized billing and demand analysis.  
- **Group-specific Forecasting** – Identifying group-level consumption trends.  

💡 **Use Case**: Supports utility providers in **energy planning, billing, and demand forecasting**.  

---

## 📂 Inputs

### 🔌 Meter Readings
- **Directory**: `csv/`  
- **Files**: `*reading*.csv` (e.g., `01.04.2023-20.07.2023(reading).csv`)  
- **Columns**:
  - `consumer_id` – Unique ID (int64)  
  - `reading_date` – Date (YYYY-MM-DD)  
  - `reading` – Meter reading in Wh (converted to kWh)  

---

### 🌡️ Temperature Data
- **File**: `csv/temperature.csv`  
- **Columns**:
  - `date` – Date (YYYY-MM-DD)  
  - `tavg` – Average daily temperature (°C)  

---

### ✅ Data Requirements
- Dates must be parseable (`YYYY-MM-DD`).  
- Consumers must have **≥365 unique days** of readings.  
- **Consumption capped at 0–100 kWh/day** (outlier handling).  
- Negative consumption interpolated with median per consumer.  
- Outliers removed via **IQR method (1.5 × IQR rule)**.  

---

## 📤 Outputs

### 📑 Forecast Files
- **System-wide**: `lightgbm_system_forecast.csv`  
- **Consumer-specific**:  
  - `lightgbm_consumer_forecast.csv` – All consumers, 100 days  
  - `lightgbm_consumer_avg_forecast.csv` – Consumer average  
- **Group-specific**: `lightgbm_group_{0,1,2}_forecast.csv`  

---

### 📊 Visualizations
- `lightgbm_system_forecast.png` – System + consumer average forecasts  
- `lightgbm_group_forecast.png` – Group-level forecasts  

---

### 🧾 Model Information
- `lightgbm_system_model_info.csv`  
- `lightgbm_consumer_model_info.csv`  
- `lightgbm_group_{0,1,2}_model_info.csv`  

Each includes: **rmse, mae, mae/median ratio, features, data points, date range, params**.  

---

### 💾 Saved Models
- `lightgbm_system_model_{timestamp}.txt`  
- `lightgbm_consumer_model_{timestamp}.txt`  
- `lightgbm_group_{0,1,2}_model_{timestamp}.txt`  

---

### 📝 Logs
- `lightgbm_training.log` – Detailed execution log.  
- `model_performance.log` – Summary metrics for monitoring.  

---

## 🧩 Features Used
- **Weather**: `avg_temp` (filled with median if missing).  
- **Date**: `year, month, day_of_week, day_of_year, week_of_year`.  
- **Rolling stats**: `rolling_mean_7, rolling_mean_14, rolling_mean_30`.  
- **IDs**: `consumer_id, group`.  

---

## ⚙️ Model Details

### 🔹 System-wide
- **Objective**: Regression  
- **Params**: `num_leaves=31, learning_rate=0.05, ...`  
- **Performance**: RMSE = 0.27, MAE = 0.21, MAE/Median = 3.2%  

### 🔹 Consumer-specific
- **Objective**: Tweedie (`tweedie_variance_power=1.5`)  
- **Params**: `num_leaves=64, learning_rate=0.03, ...`  
- **Performance**: RMSE = 2.59, MAE = 1.33, MAE/Median = 33.2%  

### 🔹 Group-specific
- **Group 0 (Low, ~4.9 kWh/day)**: Regression, RMSE = 0.15  
- **Group 1 (High, ~51.3 kWh/day)**: Tweedie, RMSE = 3.61  
- **Group 2 (Medium, ~16.3 kWh/day)**: Regression, RMSE = 0.76  

---

## 📌 Assumptions
- Meter readings in Wh (converted to kWh).  
- Forecast horizon: **100 days** after last known date.  
- Consumers with <365 days excluded.  
- Clustering via **KMeans (3 groups, random_state=42)**.  
- Missing values filled via median (temperature, rolling means).  

---

## ⚡ Setup

### 🖥️ Prerequisites
- Python 3.8+  
- Libraries: `pandas, numpy, lightgbm, scikit-learn, dask[dataframe], matplotlib`  

```bash
pip install pandas numpy lightgbm scikit-learn dask[dataframe] matplotlib
```

---

### ▶️ Running
1. Place meter readings in `csv/`  
2. Place temperature file in `csv/temperature.csv`  
3. Run:

```bash
python forecast_consumption.py
```

4. Forecasts, models, plots, and logs will be generated in the working directory.  

---

## ⏱️ Performance

- **Data Size**:  
  - Input: ~9.4M readings, 11,357 consumers  
  - Processed: ~12.3M daily rows  
- **Runtime**: ~31 min for 100 days (consumer-level training = bottleneck)  
- **Latest Run (2025-08-27)**:  
  - System-wide: RMSE = 0.27, MAE = 0.21 (3.2% error)  
  - Consumer: RMSE = 2.59, MAE = 1.33 (33.2% error)  
  - Group 0: RMSE = 0.15 (2.9%)  
  - Group 1: RMSE = 3.61 (7.0%)  
  - Group 2: RMSE = 0.76 (4.0%)  

---

## 🚀 Key Improvements
- ✅ **Non-Negative Predictions** via Tweedie objective.  
- ✅ **Data Quality Handling**: Negative readings interpolated, outliers removed.  
- ✅ **Robust Validation**: All 11,357 consumers validated across 100 days.  

---

## 🏭 Production Notes
- **Data Quality**: Implement upstream validation to reduce anomalies.  
- **Runtime**: Consider parallelizing consumer forecasts.  
- **Retraining**: Periodically retrain (data covers 2022–2024).  
- **Feature Expansion**: Add weather/demographic features for Group 1.  
- **Monitoring**: Track `model_performance.log` over time for drift.  