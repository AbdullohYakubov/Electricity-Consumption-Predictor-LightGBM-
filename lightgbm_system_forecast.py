import pandas as pd
import glob
import numpy as np
from datetime import timedelta
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')

def calculate_recent_payments(df_readings, df_payments):
    # Sort payments and readings dataframes
    df_payments = df_payments.sort_values(['consumer_id', 'payment_date'])
    df_readings = df_readings.sort_values(['consumer_id', 'reading_date'])
    
    # Initialize result list for recent payments
    result = []
    
    # Group readings by consumer_id for efficiency
    for cid, group in df_readings.groupby('consumer_id'):
        # Filter payments for this consumer
        payments = df_payments[df_payments['consumer_id'] == cid]
        if payments.empty:
            continue
        
        # For each reading date, calculate sum of payments within 30 days
        for date in group['reading_date']:
            mask = (payments['payment_date'] <= date) & (payments['payment_date'] > date - pd.Timedelta(days=30))
            total_payment = payments.loc[mask, 'amount'].sum()
            result.append({'consumer_id': cid, 'reading_date': date, 'recent_payments': total_payment})
    
    # Convert result to DataFrame
    df_recent_payments = pd.DataFrame(result)
    
    return df_recent_payments

def prepare_all_daily_consumption(df_readings):
    # Similar to your prepare_consumer_data_with_regressors, but for all users
    agg = df_readings.groupby(['consumer_id', 'reading_date'])['reading'].agg(
        max_reading='max',
        min_reading='min'
    ).reset_index()
    agg = agg.sort_values(['consumer_id', 'reading_date'])
    agg['prev_reading'] = agg.groupby('consumer_id')['min_reading'].shift(1)
    agg['prev_date'] = agg.groupby('consumer_id')['reading_date'].shift(1)
    agg = agg.dropna(subset=['prev_reading'])
    agg['consumption'] = agg['max_reading'] - agg['prev_reading']
    agg['days_diff'] = (agg['reading_date'] - agg['prev_date']).dt.days
    agg = agg[agg['days_diff'] > 0]
    agg['daily_consumption'] = agg['consumption'] / agg['days_diff']

    all_rows = []
    for _, row in agg.iterrows():
        for i in range(1, int(row['days_diff']) + 1):
            day = row['prev_date'] + pd.Timedelta(days=i)
            all_rows.append({
                'consumer_id': row['consumer_id'],
                'reading_date': day,
                'consumption': row['daily_consumption']
            })

    return pd.DataFrame(all_rows)

def remove_iqr_outliers(df, group_col='group', value_col='consumption'):
    """
    Removes outliers from `value_col` based on IQR, computed per `group_col`.
    """
    def iqr_filter(group_df):
        Q1 = group_df[value_col].quantile(0.25)
        Q3 = group_df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return group_df[(group_df[value_col] >= lower) & (group_df[value_col] <= upper)]

    return df.groupby(group_col, group_keys=False).apply(iqr_filter).reset_index(drop=True)

def prepare_system_data(df_readings, df_temperature, df_recent_payments):
    """
    Prepare system-wide aggregated data with features for LightGBM
    """
    print("Preparing system-wide aggregated data...")
    
    # Aggregate consumption by date (system-wide average)
    daily_consumption = df_readings.groupby('reading_date')['consumption'].mean().reset_index()
    daily_consumption.columns = ['ds', 'y']
    
    # Merge temperature data
    if df_temperature is not None:
        daily_consumption = daily_consumption.merge(df_temperature, left_on='ds', right_on='reading_date', how='left')
        if 'tavg' in daily_consumption.columns:
            daily_consumption = daily_consumption.rename(columns={'tavg': 'avg_temp'})
        daily_consumption['avg_temp'] = daily_consumption['avg_temp'].fillna(daily_consumption['avg_temp'].median())
    else:
        daily_consumption['avg_temp'] = 20.0
    
    # Merge recent payments (system-wide average)
    if df_recent_payments is not None:
        daily_payments = df_recent_payments.groupby('reading_date')['recent_payments'].mean().reset_index()
        daily_consumption = daily_consumption.merge(daily_payments, left_on='ds', right_on='reading_date', how='left')
        daily_consumption['recent_payments'] = daily_consumption['recent_payments'].fillna(0.0)
    else:
        daily_consumption['recent_payments'] = 0.0
    
    # Add time-based features
    daily_consumption['year'] = daily_consumption['ds'].dt.year
    daily_consumption['month'] = daily_consumption['ds'].dt.month
    daily_consumption['day_of_week'] = daily_consumption['ds'].dt.dayofweek
    daily_consumption['day_of_year'] = daily_consumption['ds'].dt.dayofyear
    daily_consumption['week_of_year'] = daily_consumption['ds'].dt.isocalendar().week
    
    # Add rolling features
    daily_consumption = daily_consumption.sort_values('ds')
    daily_consumption['rolling_mean_7'] = daily_consumption['y'].rolling(window=7, min_periods=1).mean()
    daily_consumption['rolling_mean_14'] = daily_consumption['y'].rolling(window=14, min_periods=1).mean()
    daily_consumption['rolling_mean_30'] = daily_consumption['y'].rolling(window=30, min_periods=1).mean()
    
    # Shift rolling means to avoid look-ahead bias
    daily_consumption['rolling_mean_7'] = daily_consumption['rolling_mean_7'].shift(1)
    daily_consumption['rolling_mean_14'] = daily_consumption['rolling_mean_14'].shift(1)
    daily_consumption['rolling_mean_30'] = daily_consumption['rolling_mean_30'].shift(1)
    
    # Fill NaN values
    daily_consumption['rolling_mean_7'] = daily_consumption['rolling_mean_7'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_14'] = daily_consumption['rolling_mean_14'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_30'] = daily_consumption['rolling_mean_30'].fillna(daily_consumption['y'])
    
    # Remove rows with NaN in target
    daily_consumption = daily_consumption.dropna(subset=['y'])
    
    # Ensure positive consumption
    daily_consumption = daily_consumption[daily_consumption['y'] >= 0]
    
    print(f"System data shape: {daily_consumption.shape}")
    print(f"Date range: {daily_consumption['ds'].min()} to {daily_consumption['ds'].max()}")
    print(f"Consumption range: {daily_consumption['y'].min():.2f} to {daily_consumption['y'].max():.2f}")

    print(daily_consumption.columns.tolist())
    
    return daily_consumption

def train_lightgbm_model(data, test_size=0.2):
    """
    Train LightGBM model with time series cross-validation
    """
    print("Training LightGBM model...")
    
    # Prepare features
    feature_cols = ['avg_temp', 'recent_payments', 'year', 'month', 'day_of_week', 
                   'day_of_year', 'week_of_year', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30']
    
    # Filter available features
    available_features = [col for col in feature_cols if col in data.columns]
    print(f"Using features: {available_features}")
    
    # Split data
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Prepare LightGBM datasets
    X_train = train_data[available_features]
    y_train = train_data['y']
    X_test = test_data[available_features]
    y_test = test_data['y']
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Create datasets
    train_dataset = lgb.Dataset(X_train, label=y_train)
    test_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)
    
    # Train model
    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[test_dataset],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test MAE/Median: {(mae/y_test.median())*100:.1f}%")
    
    return model, available_features, rmse, mae

def forecast_future(model, data, features, periods=100):
    """
    Forecast future consumption
    """
    print(f"Forecasting next {periods} days...")
    
    # Get last date and create future dates
    last_date = data['ds'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    # Create future dataframe
    future_data = pd.DataFrame({'ds': future_dates})
    
    # Add time features
    future_data['year'] = future_data['ds'].dt.year
    future_data['month'] = future_data['ds'].dt.month
    future_data['day_of_week'] = future_data['ds'].dt.dayofweek
    future_data['day_of_year'] = future_data['ds'].dt.dayofyear
    future_data['week_of_year'] = future_data['ds'].dt.isocalendar().week
    
    # Use last known values for other features
    last_row = data.iloc[-1]
    for feature in features:
        if feature not in ['year', 'month', 'day_of_week', 'day_of_year', 'week_of_year']:
            future_data[feature] = last_row[feature]
    
    # Make predictions
    X_future = future_data[features]
    future_predictions = model.predict(X_future)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions
    })
    
    return forecast_df

def main():
    start_time = time.time()
    print("Starting LightGBM System-Wide Consumption Forecasting")
    print("=" * 60)

    # 1. Read and merge all reading files
    print("Step 1: Loading and merging reading data...")
    reading_files = glob.glob("csv/*reading*.csv")
    print(f"Found {len(reading_files)} reading files: {reading_files}")

    df_readings = pd.concat([pd.read_csv(f) for f in reading_files], ignore_index=True)
    df_readings['reading_date'] = pd.to_datetime(df_readings['reading_date'])
    df_readings = df_readings.dropna(subset=['reading_date'])

    # Basic stats
    df_readings = df_readings.sort_values(['consumer_id', 'reading_date']).reset_index(drop=True)
    print(f"Total readings: {len(df_readings)}")
    print(f"Unique consumers: {df_readings['consumer_id'].nunique()}")

    # Convert from W/h to kW/h
    df_readings['reading'] = df_readings['reading'] / 1000

    # 2. Read temperature
    print("Step 2: Loading temperature data...")
    try:
        df_temperature = pd.read_csv("csv/temperature.csv")
        df_temperature['date'] = pd.to_datetime(df_temperature['date'])
        df_temperature = df_temperature.dropna(subset=['date'])
        df_temperature = df_temperature.rename(columns={'date': 'reading_date'})
        print("Temperature data loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load temperature data: {e}")
        df_temperature = None

    # 3. Read payments and calculate recent payments
    print("Step 3: Loading payment data...")
    try:
        df_payments = pd.read_csv("csv/confirmed_payment.csv")
        df_payments['payment_date'] = pd.to_datetime(df_payments['payment_date'])
        df_payments = df_payments.dropna(subset=['payment_date'])
        df_recent_payments = calculate_recent_payments(df_readings, df_payments)
        print("Payment data processed successfully")
    except Exception as e:
        print(f"Warning: Could not load payment data: {e}")
        df_recent_payments = None

    print(f"Date range: {df_readings['reading_date'].min()} to {df_readings['reading_date'].max()}")

    # 4. Calculate daily consumption for all consumers
    print("Step 4: Calculating daily consumption...")
    temp_df = prepare_all_daily_consumption(df_readings)

    # Filter out negative consumption
    print(f"Before removing negative consumption: {len(temp_df)} rows")
    temp_df = temp_df[temp_df['consumption'] >= 0]
    print(f"After removing negative consumption: {len(temp_df)} rows")

    # 5. Clustering consumers (optional - for data quality)
    print("Step 5: Clustering consumers for data quality...")
    user_mean = temp_df.groupby('consumer_id')['consumption'].mean()
    user_mean = user_mean[user_mean > 0]
    print(f"Consumers with positive mean consumption: {len(user_mean)}")

    # Perform clustering
    from sklearn.cluster import KMeans
    X = user_mean.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    user_groups = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("Group centers (average daily consumption):")
    for i, center in enumerate(centroids):
        print(f"Group {i}: {center[0]:.2f} kWh/day")

    # Assign group labels
    user_group_df = pd.DataFrame({
        'consumer_id': user_mean.index,
        'mean_consumption': user_mean.values,
        'group': user_groups
    })

    # Merge group labels back into the daily consumption data
    temp_df = temp_df.merge(user_group_df[['consumer_id', 'group']], on='consumer_id', how='left')

    # 6. Remove outliers per group using IQR
    print("Step 6: Removing group-wise outliers using IQR...")
    temp_df = remove_iqr_outliers(temp_df, group_col='group', value_col='consumption')

    # 7. Prepare system-wide data
    print("Step 7: Preparing system-wide aggregated data...")
    system_data = prepare_system_data(temp_df, df_temperature, df_recent_payments)

    # 8. Train LightGBM model
    print("Step 8: Training LightGBM model...")
    model, features, rmse, mae = train_lightgbm_model(system_data, test_size=0.2)

    # 9. Forecast future
    print("Step 9: Forecasting future consumption...")
    forecast_df = forecast_future(model, system_data, features, periods=100)

    # 10. Create visualization
    print("Step 10: Creating visualization...")
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(system_data['ds'], system_data['y'], 'b.', alpha=0.6, label='Historical', markersize=2)
    
    # Plot forecast
    plt.plot(forecast_df['ds'], forecast_df['yhat'], 'r-', linewidth=2, label='Forecast')
    
    plt.title('System-Wide Consumption Forecast (LightGBM)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Daily Consumption (kWh)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('lightgbm_system_forecast.png', dpi=300, bbox_inches='tight')
    print("Forecast plot saved as 'lightgbm_system_forecast.png'")

    # 11. Save results
    print("Step 11: Saving results...")
    
    # Save forecast
    forecast_df.to_csv('lightgbm_system_forecast.csv', index=False)
    print("Forecast saved as 'lightgbm_system_forecast.csv'")
    
    # Save model info
    model_info = {
        'rmse': rmse,
        'mae': mae,
        'mae_median_ratio': (mae/system_data['y'].median())*100,
        'features_used': features,
        'data_points': len(system_data),
        'date_range': f"{system_data['ds'].min()} to {system_data['ds'].max()}"
    }
    
    model_info_df = pd.DataFrame([model_info])
    model_info_df.to_csv('lightgbm_system_model_info.csv', index=False)
    print("Model info saved as 'lightgbm_system_model_info.csv'")

    print("\nLightGBM System-Wide Forecasting completed!")
    print("=" * 60)
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
    print(f"Final RMSE: {rmse:.2f}")
    print(f"Final MAE: {mae:.2f}")
    print(f"MAE/Median ratio: {(mae/system_data['y'].median())*100:.1f}%")

if __name__ == "__main__":
    main() 