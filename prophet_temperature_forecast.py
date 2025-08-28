# temperature_forecast.py
import pandas as pd
import numpy as np
from datetime import timedelta
from prophet import Prophet  # Using Prophet as primary due to SARIMAX limitations

# --- Load and Prepare Data ---
try:
    df = pd.read_csv("csv/temperature.csv")
    df = df.rename(columns={'date': 'ds', 'tavg': 'y'})  # Prophet requires 'ds' and 'y'
    df['ds'] = pd.to_datetime(df['ds'])
    df = df[['ds', 'y']].dropna()  # Keep only ds and y, drop NA
except FileNotFoundError:
    print("Error: 'csv/temperature.csv' not found. Please ensure the file exists.")
    exit(1)
except KeyError:
    print("Error: 'date' or 'tavg' column not found in temperature.csv. Check column names.")
    exit(1)

# --- Train Prophet Model ---
model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
model.fit(df)

# --- Forecast Daily (e.g., for 730 days ~2 years) ---
future = model.make_future_dataframe(periods=730)
forecast = model.predict(future)

# --- Extract Forecasted Temperatures ---
future_temps = forecast[['ds', 'yhat']].tail(730)  # Last 730 days
future_temps.columns = ['ds', 'avg_temp']  # Rename for consistency

# --- Save to CSV ---
future_temps.to_csv('future_temps.csv', index=False)
print("Future temperatures forecasted and saved to 'future_temps.csv'.")
print(future_temps.head())