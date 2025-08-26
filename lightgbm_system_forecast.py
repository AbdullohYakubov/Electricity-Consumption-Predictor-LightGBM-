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
import logging
from sklearn.cluster import KMeans
import dask.dataframe as dd

warnings.filterwarnings('ignore')

# Set up logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lightgbm_training.log'),
        logging.StreamHandler()
    ]
)

def calculate_recent_payments(df_readings, df_payments):
    logging.info("Calculating recent payments with Dask...")
    if df_payments.empty:
        logging.warning("Payment DataFrame is empty, returning empty recent payments")
        return pd.DataFrame(columns=['consumer_id', 'reading_date', 'recent_payments'])
    
    dd_payments = dd.from_pandas(df_payments, npartitions=10)
    dd_readings = dd.from_pandas(df_readings, npartitions=10)
    
    def compute_payments(cid, group):
        payments = dd_payments[dd_payments['consumer_id'] == cid].compute()
        if payments.empty:
            return pd.DataFrame()
        result = []
        for date in group['reading_date']:
            mask = (payments['payment_date'] <= date) & (payments['payment_date'] > date - pd.Timedelta(days=30))
            total_payment = payments.loc[mask, 'amount'].sum()
            result.append({'consumer_id': cid, 'reading_date': date, 'recent_payments': total_payment})
        return pd.DataFrame(result)
    
    result = dd_readings.groupby('consumer_id').apply(compute_payments, meta={'consumer_id': 'int64', 'reading_date': 'datetime64[ns]', 'recent_payments': 'float64'}).compute()
    df_recent_payments = result.reset_index(drop=True)
    logging.info(f"Recent payments DataFrame shape: {df_recent_payments.shape}")
    logging.info(f"Recent payments columns: {df_recent_payments.columns.tolist()}")
    return df_recent_payments

def prepare_all_daily_consumption(df_readings):
    """
    Calculate daily consumption for all consumers
    """
    logging.info("Calculating daily consumption...")
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

def prepare_system_wide_data(df_readings, df_temperature, df_recent_payments):
    """
    Prepare system-wide aggregated data with features for LightGBM (average consumption)
    """
    logging.info("Preparing system-wide aggregated data...")
    
    daily_consumption = df_readings.groupby('reading_date')['consumption'].mean().reset_index()
    daily_consumption.columns = ['ds', 'y']
    
    if df_temperature is not None:
        daily_consumption = daily_consumption.merge(df_temperature, left_on='ds', right_on='reading_date', how='left')
        if 'tavg' in daily_consumption.columns:
            daily_consumption = daily_consumption.rename(columns={'tavg': 'avg_temp'})
        daily_consumption['avg_temp'] = daily_consumption['avg_temp'].fillna(daily_consumption['avg_temp'].median())
    else:
        daily_consumption['avg_temp'] = 20.0
    
    if df_recent_payments is not None and not df_recent_payments.empty and 'reading_date' in df_recent_payments.columns:
        df_recent_payments = df_recent_payments.rename(columns={'reading_date': 'ds'})
        daily_payments = df_recent_payments.groupby('ds')['recent_payments'].mean().reset_index()
        daily_consumption = daily_consumption.merge(daily_payments, on='ds', how='left')
        daily_consumption['recent_payments'] = daily_consumption['recent_payments'].fillna(0.0)
    else:
        logging.warning("No valid recent payments data available, setting recent_payments to 0.0")
        daily_consumption['recent_payments'] = 0.0
    
    daily_consumption['year'] = daily_consumption['ds'].dt.year
    daily_consumption['month'] = daily_consumption['ds'].dt.month
    daily_consumption['day_of_week'] = daily_consumption['ds'].dt.dayofweek
    daily_consumption['day_of_year'] = daily_consumption['ds'].dt.dayofyear
    daily_consumption['week_of_year'] = daily_consumption['ds'].dt.isocalendar().week
    
    daily_consumption = daily_consumption.sort_values('ds')
    daily_consumption['rolling_mean_7'] = daily_consumption['y'].rolling(window=7, min_periods=1).mean()
    daily_consumption['rolling_mean_14'] = daily_consumption['y'].rolling(window=14, min_periods=1).mean()
    daily_consumption['rolling_mean_30'] = daily_consumption['y'].rolling(window=30, min_periods=1).mean()
    
    daily_consumption['rolling_mean_7'] = daily_consumption['rolling_mean_7'].shift(1)
    daily_consumption['rolling_mean_14'] = daily_consumption['rolling_mean_14'].shift(1)
    daily_consumption['rolling_mean_30'] = daily_consumption['rolling_mean_30'].shift(1)
    
    daily_consumption['rolling_mean_7'] = daily_consumption['rolling_mean_7'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_14'] = daily_consumption['rolling_mean_14'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_30'] = daily_consumption['rolling_mean_30'].fillna(daily_consumption['y'])
    
    daily_consumption = daily_consumption.dropna(subset=['y'])
    daily_consumption = daily_consumption[daily_consumption['y'] >= 0]
    
    logging.info(f"System data shape: {daily_consumption.shape}")
    logging.info(f"Date range: {daily_consumption['ds'].min()} to {daily_consumption['ds'].max()}")
    logging.info(f"Consumption range: {daily_consumption['y'].min():.2f} to {daily_consumption['y'].max():.2f}")
    
    return daily_consumption

def prepare_consumer_specific_data(df_readings, df_temperature, df_recent_payments):
    """
    Prepare consumer-specific data with features for LightGBM (individual consumption)
    """
    logging.info("Preparing consumer-specific data...")
    
    daily_consumption = df_readings.copy()
    daily_consumption['ds'] = daily_consumption['reading_date']
    daily_consumption['y'] = daily_consumption['consumption'].clip(lower=0.0, upper=100.0)  # Ensure cap is applied
    daily_consumption['group'] = daily_consumption['group'].astype(int)
    
    if df_temperature is not None:
        daily_consumption = daily_consumption.merge(df_temperature, left_on='ds', right_on='reading_date', how='left')
        if 'tavg' in daily_consumption.columns:
            daily_consumption = daily_consumption.rename(columns={'tavg': 'avg_temp'})
        daily_consumption['avg_temp'] = daily_consumption['avg_temp'].fillna(daily_consumption['avg_temp'].median())
    else:
        daily_consumption['avg_temp'] = 20.0
    
    if df_recent_payments is not None and not df_recent_payments.empty and 'reading_date' in df_recent_payments.columns:
        df_recent_payments = df_recent_payments.rename(columns={'reading_date': 'ds'})
        logging.info(f"Columns in df_recent_payments after rename: {df_recent_payments.columns}")
        daily_consumption = daily_consumption.merge(df_recent_payments, on=['consumer_id', 'ds'], how='left')
        daily_consumption['recent_payments'] = daily_consumption['recent_payments'].fillna(0.0)
    else:
        logging.warning("No valid recent payments data available, setting recent_payments to 0.0")
        daily_consumption['recent_payments'] = 0.0
    
    daily_consumption['year'] = daily_consumption['ds'].dt.year
    daily_consumption['month'] = daily_consumption['ds'].dt.month
    daily_consumption['day_of_week'] = daily_consumption['ds'].dt.dayofweek
    daily_consumption['day_of_year'] = daily_consumption['ds'].dt.dayofyear
    daily_consumption['week_of_year'] = daily_consumption['ds'].dt.isocalendar().week
    
    daily_consumption = daily_consumption.sort_values(['consumer_id', 'ds'])
    daily_consumption['rolling_mean_7'] = daily_consumption.groupby('consumer_id')['y'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    daily_consumption['rolling_mean_14'] = daily_consumption.groupby('consumer_id')['y'].rolling(window=14, min_periods=1).mean().reset_index(0, drop=True)
    daily_consumption['rolling_mean_30'] = daily_consumption.groupby('consumer_id')['y'].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
    
    daily_consumption['rolling_mean_7'] = daily_consumption.groupby('consumer_id')['rolling_mean_7'].shift(1)
    daily_consumption['rolling_mean_14'] = daily_consumption.groupby('consumer_id')['rolling_mean_14'].shift(1)
    daily_consumption['rolling_mean_30'] = daily_consumption.groupby('consumer_id')['rolling_mean_30'].shift(1)
    
    daily_consumption['rolling_mean_7'] = daily_consumption['rolling_mean_7'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_14'] = daily_consumption['rolling_mean_14'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_30'] = daily_consumption['rolling_mean_30'].fillna(daily_consumption['y'])
    
    daily_consumption = daily_consumption.dropna(subset=['y'])
    daily_consumption = daily_consumption[daily_consumption['y'] >= 0]
    
    logging.info(f"Consumer-specific data shape: {daily_consumption.shape}")
    logging.info(f"Date range: {daily_consumption['ds'].min()} to {daily_consumption['ds'].max()}")
    logging.info(f"Consumption range: {daily_consumption['y'].min():.2f} to {daily_consumption['y'].max():.2f}")
    
    return daily_consumption

def prepare_group_data(df_readings, df_temperature, df_recent_payments, group_id):
    """
    Prepare data for a specific group with features for LightGBM (average consumption per group)
    """
    logging.info(f"Preparing data for group {group_id}...")
    
    group_data = df_readings[df_readings['group'] == group_id].copy()
    daily_consumption = group_data.groupby('ds')['y'].mean().reset_index()
    daily_consumption.columns = ['ds', 'y']
    
    if df_temperature is not None:
        daily_consumption = daily_consumption.merge(df_temperature, left_on='ds', right_on='reading_date', how='left')
        if 'tavg' in daily_consumption.columns:
            daily_consumption = daily_consumption.rename(columns={'tavg': 'avg_temp'})
        daily_consumption['avg_temp'] = daily_consumption['avg_temp'].fillna(daily_consumption['avg_temp'].median())
    else:
        daily_consumption['avg_temp'] = 20.0
    
    if df_recent_payments is not None and not df_recent_payments.empty and 'reading_date' in df_recent_payments.columns:
        group_payments = df_recent_payments[df_recent_payments['consumer_id'].isin(group_data['consumer_id'].unique())]
        group_payments = group_payments.rename(columns={'reading_date': 'ds'})
        daily_payments = group_payments.groupby('ds')['recent_payments'].mean().reset_index()
        daily_consumption = daily_consumption.merge(daily_payments, on='ds', how='left')
        daily_consumption['recent_payments'] = daily_consumption['recent_payments'].fillna(0.0)
    else:
        logging.warning(f"No valid recent payments data for group {group_id}, setting recent_payments to 0.0")
        daily_consumption['recent_payments'] = 0.0
    
    daily_consumption['year'] = daily_consumption['ds'].dt.year
    daily_consumption['month'] = daily_consumption['ds'].dt.month
    daily_consumption['day_of_week'] = daily_consumption['ds'].dt.dayofweek
    daily_consumption['day_of_year'] = daily_consumption['ds'].dt.dayofyear
    daily_consumption['week_of_year'] = daily_consumption['ds'].dt.isocalendar().week
    
    daily_consumption = daily_consumption.sort_values('ds')
    daily_consumption['rolling_mean_7'] = daily_consumption['y'].rolling(window=7, min_periods=1).mean()
    daily_consumption['rolling_mean_14'] = daily_consumption['y'].rolling(window=14, min_periods=1).mean()
    daily_consumption['rolling_mean_30'] = daily_consumption['y'].rolling(window=30, min_periods=1).mean()
    
    daily_consumption['rolling_mean_7'] = daily_consumption['rolling_mean_7'].shift(1)
    daily_consumption['rolling_mean_14'] = daily_consumption['rolling_mean_14'].shift(1)
    daily_consumption['rolling_mean_30'] = daily_consumption['rolling_mean_30'].shift(1)
    
    daily_consumption['rolling_mean_7'] = daily_consumption['rolling_mean_7'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_14'] = daily_consumption['rolling_mean_14'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_30'] = daily_consumption['rolling_mean_30'].fillna(daily_consumption['y'])
    
    daily_consumption = daily_consumption.dropna(subset=['y'])
    daily_consumption = daily_consumption[daily_consumption['y'] >= 0]
    
    logging.info(f"Group {group_id} data shape: {daily_consumption.shape}")
    logging.info(f"Date range: {daily_consumption['ds'].min()} to {daily_consumption['ds'].max()}")
    logging.info(f"Consumption range: {daily_consumption['y'].min():.2f} to {daily_consumption['y'].max():.2f}")
    
    return daily_consumption

def train_lightgbm_model(data, test_size=0.2, params=None):
    """
    Train LightGBM model with given parameters
    """
    logging.info("Training LightGBM model...")
    logging.info(f"Parameters: {params}")
    
    feature_cols = ['avg_temp', 'recent_payments', 'year', 'month', 'day_of_week', 
                    'day_of_year', 'week_of_year', 'rolling_mean_7', 'rolling_mean_14', 
                    'rolling_mean_30']
    
    if 'consumer_id' in data.columns:
        feature_cols.append('consumer_id')
    if 'group' in data.columns:
        feature_cols.append('group')
    
    available_features = [col for col in feature_cols if col in data.columns]
    logging.info(f"Using features: {available_features}")
    
    categorical_feature = [col for col in ['consumer_id', 'group'] if col in available_features]
    
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    logging.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    X_train = train_data[available_features]
    y_train = train_data['y']
    X_test = test_data[available_features]
    y_test = test_data['y']
    
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feature)
    test_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset, categorical_feature=categorical_feature)
    
    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[test_dataset],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    if 'group' in test_data.columns:
        test_data['y_pred'] = y_pred
        error_by_group = test_data.groupby('group').apply(
            lambda x: pd.Series({
                'RMSE': np.sqrt(mean_squared_error(x['y'], x['y_pred'])),
                'MAE': mean_absolute_error(x['y'], x['y_pred']),
                'Count': len(x)
            })
        ).reset_index()
        logging.info(f"Error by cluster group:\n{error_by_group}")
    
    logging.info(f"Train RMSE: {train_rmse:.2f}")
    logging.info(f"Train MAE: {train_mae:.2f}")
    logging.info(f"Test RMSE: {rmse:.2f}")
    logging.info(f"Test MAE: {mae:.2f}")
    logging.info(f"Test MAE/Median: {(mae/y_test.median())*100:.1f}%")
    
    return model, available_features, rmse, mae

def forecast_system_future(model, data, features, periods=100):
    """
    Forecast future system-wide consumption
    """
    logging.info(f"Forecasting next {periods} days (system-wide)...")
    
    last_date = data['ds'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    future_data = pd.DataFrame({'ds': future_dates})
    future_data['year'] = future_data['ds'].dt.year
    future_data['month'] = future_data['ds'].dt.month
    future_data['day_of_week'] = future_data['ds'].dt.dayofweek
    future_data['day_of_year'] = future_data['ds'].dt.dayofyear
    future_data['week_of_year'] = future_data['ds'].dt.isocalendar().week
    
    last_row = data.iloc[-1]
    for feature in features:
        if feature not in ['year', 'month', 'day_of_week', 'day_of_year', 'week_of_year']:
            future_data[feature] = last_row[feature]
    
    X_future = future_data[features]
    future_predictions = model.predict(X_future)
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions
    })
    
    return forecast_df

def forecast_consumer_future(model, data, features, periods=100):
    """
    Forecast future consumption for each consumer
    """
    logging.info(f"Forecasting next {periods} days (consumer-specific)...")
    
    last_date = data['ds'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    future_data = []
    for consumer_id in data['consumer_id'].unique():
        consumer_data = data[data['consumer_id'] == consumer_id].iloc[-1]
        for date in future_dates:
            row = {
                'ds': date,
                'consumer_id': consumer_id,
                'group': consumer_data['group'],
                'year': date.year,
                'month': date.month,
                'day_of_week': date.dayofweek,
                'day_of_year': date.dayofyear,
                'week_of_year': date.isocalendar().week,
                'avg_temp': consumer_data['avg_temp'],
                'recent_payments': consumer_data['recent_payments'],
                'rolling_mean_7': consumer_data['rolling_mean_7'],
                'rolling_mean_14': consumer_data['rolling_mean_14'],
                'rolling_mean_30': consumer_data['rolling_mean_30']
            }
            future_data.append(row)
    
    future_data = pd.DataFrame(future_data)
    X_future = future_data[features]
    future_predictions = model.predict(X_future)
    
    forecast_df = pd.DataFrame({
        'ds': future_data['ds'],
        'consumer_id': future_data['consumer_id'],
        'group': future_data['group'],
        'yhat': future_predictions
    })
    
    return forecast_df

def forecast_group_future(model, data, features, periods=100):
    """
    Forecast future average consumption for a specific group
    """
    logging.info(f"Forecasting next {periods} days for group...")
    
    last_date = data['ds'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    future_data = pd.DataFrame({'ds': future_dates})
    future_data['year'] = future_data['ds'].dt.year
    future_data['month'] = future_data['ds'].dt.month
    future_data['day_of_week'] = future_data['ds'].dt.dayofweek
    future_data['day_of_year'] = future_data['ds'].dt.dayofyear
    future_data['week_of_year'] = future_data['ds'].dt.isocalendar().week
    
    last_row = data.iloc[-1]
    for feature in features:
        if feature not in ['year', 'month', 'day_of_week', 'day_of_year', 'week_of_year']:
            future_data[feature] = last_row[feature]
    
    X_future = future_data[features]
    future_predictions = model.predict(X_future)
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions
    })
    
    return forecast_df

def main():
    start_time = time.time()
    logging.info("Starting LightGBM System-Wide, Consumer-Specific, and Group-Specific Consumption Forecasting")
    logging.info("=" * 60)

    system_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'force_col_wise': True
    }
    
    consumer_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.03,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'verbose': -1,
        'force_col_wise': True
    }

    group_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'force_col_wise': True
    }

    group1_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.03,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'verbose': -1,
        'force_col_wise': True
    }

    # 1. Read and merge all reading files
    logging.info("Step 1: Loading and merging reading data...")
    reading_files = glob.glob("csv/*reading*.csv")
    logging.info(f"Found {len(reading_files)} reading files: {reading_files}")

    df_readings = pd.concat([pd.read_csv(f) for f in reading_files], ignore_index=True)
    df_readings['reading_date'] = pd.to_datetime(df_readings['reading_date'])
    df_readings = df_readings.dropna(subset=['reading_date'])

    # # Select a small fraction (first 100 unique consumers) for testing
    # unique_consumers = df_readings['consumer_id'].unique()[:100]
    # df_readings = df_readings[df_readings['consumer_id'].isin(unique_consumers)]
    # logging.info(f"Selected {len(unique_consumers)} unique consumers for testing.")

    # Filter consumers with sufficient data
    days_per_consumer = df_readings.groupby('consumer_id')['reading_date'].nunique()
    logging.info(f"Min unique days per consumer: {days_per_consumer.min()}")
    logging.info(f"Max unique days per consumer: {days_per_consumer.max()}")
    min_days = 500
    valid_consumers = days_per_consumer[days_per_consumer >= min_days].index
    df_readings = df_readings[df_readings['consumer_id'].isin(valid_consumers)]
    logging.info(f"Filtered to {len(valid_consumers)} consumers with >= {min_days} days")

    # Save original consumer IDs for payment matching
    original_consumer_ids = df_readings['consumer_id'].unique()
    
    # Map consumer IDs
    consumer_id_map = {cid: i for i, cid in enumerate(original_consumer_ids)}
    df_readings['consumer_id'] = df_readings['consumer_id'].map(consumer_id_map)
    logging.info(f"Mapped {len(consumer_id_map)} unique consumer_id values to 0-{len(consumer_id_map)-1}")

    days_per_consumer = df_readings.groupby('consumer_id')['reading_date'].nunique()
    logging.info(f"Average unique days per consumer: {days_per_consumer.mean():.2f}")

    df_readings = df_readings.sort_values(['consumer_id', 'reading_date']).reset_index(drop=True)
    logging.info(f"Total readings: {len(df_readings)}")
    logging.info(f"Unique consumers: {df_readings['consumer_id'].nunique()}")

    df_readings['reading'] = df_readings['reading'] / 1000

    # 2. Read temperature
    logging.info("Step 2: Loading temperature data...")
    try:
        df_temperature = pd.read_csv("csv/temperature.csv")
        df_temperature['date'] = pd.to_datetime(df_temperature['date'])
        df_temperature = df_temperature.dropna(subset=['date'])
        df_temperature = df_temperature.rename(columns={'date': 'reading_date'})
        logging.info("Temperature data loaded successfully")
    except Exception as e:
        logging.warning(f"Could not load temperature data: {e}")
        df_temperature = None

    # 3. Read payments and calculate recent payments
    logging.info("Step 3: Loading payment data...")
    try:
        # Explicitly specify required columns to avoid parsing issues
        df_payments = pd.read_csv("csv/confirmed_payment.csv", usecols=['consumer_id', 'amount', 'payment_date'])
        df_payments['payment_date'] = pd.to_datetime(df_payments['payment_date'])
        df_payments = df_payments.dropna(subset=['payment_date'])
        logging.info(f"Raw payments DataFrame shape: {df_payments.shape}")
        logging.info(f"Raw payments columns: {df_payments.columns.tolist()}")
        logging.info(f"Sample payments data:\n{df_payments.head().to_string()}")
        
        # Filter payments for valid consumer IDs before mapping
        df_payments = df_payments[df_payments['consumer_id'].isin(original_consumer_ids)]
        df_payments['consumer_id'] = df_payments['consumer_id'].map(consumer_id_map)
        df_payments = df_payments.dropna(subset=['consumer_id'])
        
        logging.info(f"Filtered payments DataFrame shape: {df_payments.shape}")
        logging.info(f"Payments columns: {df_payments.columns.tolist()}")
        df_recent_payments = calculate_recent_payments(df_readings, df_payments)
        logging.info("Payment data processed successfully")
    except Exception as e:
        logging.warning(f"Could not load payment data: {e}")
        df_recent_payments = pd.DataFrame(columns=['consumer_id', 'reading_date', 'recent_payments'])

    logging.info(f"Date range: {df_readings['reading_date'].min()} to {df_readings['reading_date'].max()}")

    # 4. Calculate daily consumption for all consumers
    logging.info("Step 4: Calculating daily consumption...")
    temp_df = prepare_all_daily_consumption(df_readings)

    logging.info(f"Average days per consumer after consumption calculation: {len(temp_df) / temp_df['consumer_id'].nunique():.2f}")

    logging.info(f"Before removing negative consumption: {len(temp_df)} rows")
    temp_df = temp_df[temp_df['consumption'] >= 0]
    logging.info(f"After removing negative consumption: {len(temp_df)} rows")

    # 5. Clustering consumers for data quality...
    logging.info("Step 5: Clustering consumers for data quality...")
    user_mean = temp_df.groupby('consumer_id')['consumption'].mean()
    user_mean = user_mean[user_mean > 0]
    logging.info(f"Consumers with positive mean consumption: {len(user_mean)}")

    X = user_mean.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)  # Fixed random_state for consistency
    user_groups = kmeans.labels_
    centroids = kmeans.cluster_centers_

    logging.info("Group centers (average daily consumption):")
    for i, center in enumerate(centroids):
        logging.info(f"Group {i}: {center[0]:.2f} kWh/day")

    user_group_df = pd.DataFrame({
        'consumer_id': user_mean.index,
        'mean_consumption': user_mean.values,
        'group': user_groups
    })

    temp_df = temp_df.merge(user_group_df[['consumer_id', 'group']], on='consumer_id', how='left')

    # 6. Remove outliers per group using IQR
    logging.info("Step 6: Removing group-wise outliers using IQR...")
    temp_df = remove_iqr_outliers(temp_df, group_col='group', value_col='consumption')

    # 7. Prepare system-wide, consumer-specific, and group-specific data
    logging.info("Step 7: Preparing aggregated system-wide data...")
    system_wide_data = prepare_system_wide_data(temp_df, df_temperature, df_recent_payments)
    
    logging.info("Step 7: Preparing consumer-specific data...")
    consumer_specific_data = prepare_consumer_specific_data(temp_df, df_temperature, df_recent_payments)

    group_data = {}
    for group_id in sorted(consumer_specific_data['group'].unique()):
        logging.info(f"Step 7: Preparing data for group {group_id}...")
        group_data[group_id] = prepare_group_data(consumer_specific_data, df_temperature, df_recent_payments, group_id)

    # 8. Train LightGBM models
    logging.info("Step 8: Training system-wide LightGBM model...")
    system_model, system_features, system_rmse, system_mae = train_lightgbm_model(system_wide_data, test_size=0.2, params=system_params)
    
    logging.info("Step 8: Training consumer-specific LightGBM model...")
    consumer_model, consumer_features, consumer_rmse, consumer_mae = train_lightgbm_model(consumer_specific_data, test_size=0.2, params=consumer_params)

    group_models = {}
    group_features = {}
    group_rmse = {}
    group_mae = {}
    for group_id, g_data in group_data.items():
        logging.info(f"Step 8: Training group {group_id} LightGBM model...")
        params = group1_params if group_id == 1 else group_params
        g_model, g_features, g_rmse, g_mae = train_lightgbm_model(g_data, test_size=0.2, params=params)
        group_models[group_id] = g_model
        group_features[group_id] = g_features
        group_rmse[group_id] = g_rmse
        group_mae[group_id] = g_mae

    # 9. Forecast future
    logging.info("Step 9: Forecasting future consumption (system-wide)...")
    system_forecast_df = forecast_system_future(system_model, system_wide_data, system_features, periods=100)
    
    logging.info("Step 9: Forecasting future consumption (consumer-specific)...")
    consumer_forecast_df = forecast_consumer_future(consumer_model, consumer_specific_data, consumer_features, periods=100)

    group_forecasts = {}
    for group_id, g_model in group_models.items():
        logging.info(f"Step 9: Forecasting future consumption for group {group_id}...")
        group_forecasts[group_id] = forecast_group_future(g_model, group_data[group_id], group_features[group_id], periods=100)

    consumer_avg_forecast = consumer_forecast_df.groupby('ds')['yhat'].mean().reset_index(name='yhat_avg')

    # 10. Create visualization
    logging.info("Step 10: Creating visualization...")
    plt.figure(figsize=(15, 8))
    plt.plot(system_wide_data['ds'], system_wide_data['y'], 'b.', alpha=0.6, label='Historical System-Wide', markersize=2)
    plt.plot(system_forecast_df['ds'], system_forecast_df['yhat'], 'r-', linewidth=2, label='System-Wide Forecast')
    plt.plot(consumer_avg_forecast['ds'], consumer_avg_forecast['yhat_avg'], 'g--', linewidth=2, label='Consumer-Specific Avg Forecast')
    plt.title('System-Wide and Consumer-Specific Average Consumption Forecast (LightGBM)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Daily Consumption (kWh)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('lightgbm_system_forecast.png', dpi=300, bbox_inches='tight')
    logging.info("Forecast plot saved as 'lightgbm_system_forecast.png'")

    plt.figure(figsize=(15, 8))
    for group_id, g_data in group_data.items():
        plt.plot(g_data['ds'], g_data['y'], '.', alpha=0.6, label=f'Historical Group {group_id}', markersize=2)
        plt.plot(group_forecasts[group_id]['ds'], group_forecasts[group_id]['yhat'], linewidth=2, label=f'Group {group_id} Forecast')
    plt.title('Group-Specific Average Consumption Forecasts (LightGBM)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Daily Consumption (kWh)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('lightgbm_group_forecast.png', dpi=300, bbox_inches='tight')
    logging.info("Group forecast plot saved as 'lightgbm_group_forecast.png'")

    # 11. Save results
    logging.info("Step 11: Saving results...")
    system_forecast_df.to_csv('lightgbm_system_forecast.csv', index=False)
    logging.info("System-wide forecast saved as 'lightgbm_system_forecast.csv'")
    
    consumer_forecast_df.to_csv('lightgbm_consumer_forecast.csv', index=False)
    logging.info("Consumer-specific forecast saved as 'lightgbm_consumer_forecast.csv'")
    
    consumer_avg_forecast.to_csv('lightgbm_consumer_avg_forecast.csv', index=False)
    logging.info("Consumer-specific average forecast saved as 'lightgbm_consumer_avg_forecast.csv'")
    
    model_info = {
        'rmse': system_rmse,
        'mae': system_mae,
        'mae_median_ratio': (system_mae/system_wide_data['y'].median())*100,
        'features_used': system_features,
        'data_points': len(system_wide_data),
        'date_range': f"{system_wide_data['ds'].min()} to {system_wide_data['ds'].max()}",
        'params': [system_params]
    }
    model_info_df = pd.DataFrame([model_info])
    model_info_df.to_csv('lightgbm_system_model_info.csv', index=False)
    logging.info("System-wide model info saved as 'lightgbm_system_model_info.csv'")
    
    consumer_model_info = {
        'rmse': consumer_rmse,
        'mae': consumer_mae,
        'mae_median_ratio': (consumer_mae/consumer_specific_data['y'].median())*100,
        'features_used': consumer_features,
        'data_points': len(consumer_specific_data),
        'date_range': f"{consumer_specific_data['ds'].min()} to {consumer_specific_data['ds'].max()}",
        'params': [consumer_params]
    }
    consumer_model_info_df = pd.DataFrame([consumer_model_info])
    consumer_model_info_df.to_csv('lightgbm_consumer_model_info.csv', index=False)
    logging.info("Consumer-specific model info saved as 'lightgbm_consumer_model_info.csv'")

    for group_id, g_model in group_models.items():
        g_forecast = group_forecasts[group_id]
        g_forecast.to_csv(f'lightgbm_group_{group_id}_forecast.csv', index=False)
        logging.info(f"Group {group_id} forecast saved as 'lightgbm_group_{group_id}_forecast.csv'")
        
        g_model_info = {
            'group': group_id,
            'rmse': group_rmse[group_id],
            'mae': group_mae[group_id],
            'mae_median_ratio': (group_mae[group_id]/group_data[group_id]['y'].median())*100,
            'features_used': group_features[group_id],
            'data_points': len(group_data[group_id]),
            'date_range': f"{group_data[group_id]['ds'].min()} to {group_data[group_id]['ds'].max()}",
            'params': [group1_params if group_id == 1 else group_params]
        }
        g_model_info_df = pd.DataFrame([g_model_info])
        g_model_info_df.to_csv(f'lightgbm_group_{group_id}_model_info.csv', index=False)
        logging.info(f"Group {group_id} model info saved as 'lightgbm_group_{group_id}_model_info.csv'")

    logging.info("\nLightGBM System-Wide, Consumer-Specific, and Group-Specific Forecasting completed!")
    logging.info("=" * 60)
    logging.info(f"Total runtime: {time.time() - start_time:.2f} seconds")
    logging.info(f"System-wide Final RMSE: {system_rmse:.2f}")
    logging.info(f"System-wide Final MAE: {system_mae:.2f}")
    logging.info(f"System-wide MAE/Median ratio: {(system_mae/system_wide_data['y'].median())*100:.1f}%")
    logging.info(f"Consumer-specific Final RMSE: {consumer_rmse:.2f}")
    logging.info(f"Consumer-specific Final MAE: {consumer_mae:.2f}")
    logging.info(f"Consumer-specific MAE/Median ratio: {(consumer_mae/consumer_specific_data['y'].median())*100:.1f}%")
    for group_id in group_rmse:
        logging.info(f"Group {group_id} Final RMSE: {group_rmse[group_id]:.2f}")
        logging.info(f"Group {group_id} Final MAE: {group_mae[group_id]:.2f}")
        logging.info(f"Group {group_id} MAE/Median ratio: {(group_mae[group_id]/group_data[group_id]['y'].median())*100:.1f}%")

if __name__ == "__main__":
    main()