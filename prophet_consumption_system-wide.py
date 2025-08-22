from prophet import Prophet
import pandas as pd
import glob
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
warnings.filterwarnings('ignore')

def prepare_aggregate_consumption(df_readings):
    # 1. Сортируем и считаем consumption для каждого consumer_id
    df_readings = df_readings.sort_values(['consumer_id', 'reading_date'])
    df_readings['prev_reading'] = df_readings.groupby('consumer_id')['reading'].shift(1)
    df_readings['consumption'] = df_readings['reading'] - df_readings['prev_reading']
    # 2. Убираем отрицательные и NaN значения
    df_readings = df_readings[df_readings['consumption'] >= 0]
    df_readings = df_readings.dropna(subset=['consumption', 'reading_date'])
    # 3. Группируем по дате и суммируем consumption
    df_agg_avg = df_readings.groupby('reading_date')['consumption'].median().reset_index()
    df_agg_avg.columns = ['ds', 'y']

    # Calculate IQR bounds
    Q1 = df_agg_avg['y'].quantile(0.25)
    Q3 = df_agg_avg['y'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    df_agg_clean = df_agg_avg[(df_agg_avg['y'] >= lower_bound) & (df_agg_avg['y'] <= upper_bound)]

    print(f"Removed {len(df_agg_avg) - len(df_agg_clean)} outlier days")

    print("Original date range:", df_agg_avg['ds'].min(), "to", df_agg_avg['ds'].max())
    print("Cleaned date range:", df_agg_clean['ds'].min(), "to", df_agg_clean['ds'].max())
    print("Original y range:", df_agg_avg['y'].min(), "to", df_agg_avg['y'].max())
    print("Cleaned y range:", df_agg_clean['y'].min(), "to", df_agg_clean['y'].max())

    return df_agg_clean

def main():
    print("Step 1: Loading and merging reading data...")
    reading_files = glob.glob("csv/*reading*.csv")  # Fayllaringiz joylashgan papkani moslang
    print(f"Found {len(reading_files)} reading files: {reading_files}")
    df_readings = pd.concat([pd.read_csv(f) for f in reading_files], ignore_index=True)
    df_readings['reading_date'] = pd.to_datetime(df_readings['reading_date'])
    df_readings['reading'] = df_readings['reading'] / 1000
    df_readings = df_readings.drop_duplicates(subset=['consumer_id', 'reading_date'])
    df_readings = df_readings.dropna(subset=['reading_date'])
    df_readings = df_readings.sort_values(['consumer_id', 'reading_date']).reset_index(drop=True)
    print(f"Total readings: {len(df_readings)}")
    print(f"Unique consumers: {df_readings['consumer_id'].nunique()}")

    print("Step 2: Preparing aggregate consumption data...")
    df_agg = prepare_aggregate_consumption(df_readings)
    print(f"Aggregate consumption data points: {len(df_agg)}")
    # print(df_agg.head())

    print("Step 3: Fitting Prophet model...")
    # --- Calculate metrics (MSE, MAE) using train/test split ---
    split_idx = int(len(df_agg) * 0.8)
    train = df_agg.iloc[:split_idx]
    test = df_agg.iloc[split_idx:]
    metric_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    metric_model.fit(train)
    future_metric = metric_model.make_future_dataframe(periods=len(test), freq='D')
    forecast_metric = metric_model.predict(future_metric)
    forecast_test = forecast_metric.iloc[split_idx:split_idx+len(test)]
    actual = test['y'].values
    predicted = forecast_test['yhat'].values
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    print(f"Test MSE: {mse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Median actual consumption in test set: {np.median(actual):.2f}")
    # --- Fit on all data for final forecast and plotting ---
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.fit(df_agg)

    print("Step 4: Forecasting future consumption...")
    periods = 100  # Qancha kun oldinga bashorat qilish
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    print("Step 5: Plotting results...")
    plt.figure(figsize=(14, 7))
    plt.plot(df_agg['ds'], df_agg['y'], 'b.', label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], 'r-', label='Forecast')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2, label='Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Total Consumption')
    plt.title('System-wide Consumption Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('aggregate_prophet_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'aggregate_prophet_forecast.png'")

    # Save forecast to CSV
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('aggregate_prophet_forecast.csv', index=False)
    print("Forecast saved as 'aggregate_prophet_forecast.csv'")

if __name__ == "__main__":
    main()