import pandas as pd
import dask.dataframe as dd
import glob
import numpy as np
from datetime import timedelta, datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
import time
import logging

warnings.filterwarnings('ignore')

# ------------------ Logging Setup ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lightgbm_training.log'),
        logging.StreamHandler()
    ]
)

future_temps = pd.read_csv('future_temps.csv')
future_temps['ds'] = pd.to_datetime(future_temps['ds'])

# ------------------ Data Preparation Functions ------------------
def prepare_all_daily_consumption(df_readings):
    """Calculate daily consumption for all consumers, handling negative consumptions."""
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
    
    # Handle negative consumptions
    negative_rows = agg[agg['consumption'] < 0][['consumer_id', 'reading_date', 'consumption', 'prev_reading', 'max_reading']]
    if not negative_rows.empty:
        negative_ids = negative_rows['consumer_id'].unique().tolist()
        logging.info(f"Negative consumption detected for {len(negative_rows)} rows across {len(negative_ids)} consumer_ids: {negative_ids}")
        # Interpolate negative consumptions with median for the consumer
        median_consumption = agg[agg['consumption'] >= 0].groupby('consumer_id')['consumption'].median().to_dict()
        agg['consumption'] = agg.apply(
            lambda x: median_consumption.get(x['consumer_id'], 0) if x['consumption'] < 0 else x['consumption'], axis=1
        )
    
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
    result = pd.DataFrame(all_rows)
    logging.info(f"Daily consumption calculated: {len(result)} rows")
    return result

def remove_iqr_outliers(df, group_col='group', value_col='consumption'):
    """Remove outliers from value_col based on IQR, computed per group_col."""
    def iqr_filter(group_df):
        if len(group_df) < 4:
            return group_df
        Q1 = group_df[value_col].quantile(0.25)
        Q3 = group_df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return group_df[(group_df[value_col] >= lower) & (group_df[value_col] <= upper)]
    
    return df.groupby(group_col, group_keys=False).apply(iqr_filter).reset_index(drop=True)

def prepare_system_wide_data(df_readings, df_temperature):
    """Prepare system-wide aggregated data with features for LightGBM."""
    logging.info("Preparing system-wide aggregated data...")
    
    daily_consumption = df_readings.groupby('reading_date')['consumption'].mean().reset_index()
    daily_consumption.columns = ['ds', 'y']
    
    if df_temperature is not None:
        daily_consumption = daily_consumption.merge(df_temperature, left_on='ds', right_on='reading_date', how='left')
        if 'tavg' in daily_consumption.columns:
            daily_consumption = daily_consumption.rename(columns={'tavg': 'avg_temp'})
        daily_consumption['avg_temp'] = daily_consumption['avg_temp'].fillna(daily_consumption['avg_temp'].median())
    
    daily_consumption['year'] = daily_consumption['ds'].dt.year
    daily_consumption['month'] = daily_consumption['ds'].dt.month
    daily_consumption['day_of_week'] = daily_consumption['ds'].dt.dayofweek
    daily_consumption['day_of_year'] = daily_consumption['ds'].dt.dayofyear
    daily_consumption['week_of_year'] = daily_consumption['ds'].dt.isocalendar().week
    
    daily_consumption = daily_consumption.sort_values('ds')
    daily_consumption['rolling_mean_7'] = daily_consumption['y'].rolling(window=7, min_periods=1).mean().shift(1)
    daily_consumption['rolling_mean_14'] = daily_consumption['y'].rolling(window=14, min_periods=1).mean().shift(1)
    daily_consumption['rolling_mean_30'] = daily_consumption['y'].rolling(window=30, min_periods=1).mean().shift(1)
    
    daily_consumption['rolling_mean_7'] = daily_consumption['rolling_mean_7'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_14'] = daily_consumption['rolling_mean_14'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_30'] = daily_consumption['rolling_mean_30'].fillna(daily_consumption['y'])
    
    daily_consumption = daily_consumption.dropna(subset=['y'])
    daily_consumption = daily_consumption[daily_consumption['y'] >= 0]
    
    logging.info(f"System data shape: {daily_consumption.shape}")
    logging.info(f"Date range: {daily_consumption['ds'].min()} to {daily_consumption['ds'].max()}")
    logging.info(f"Consumption range: {daily_consumption['y'].min():.2f} to {daily_consumption['y'].max():.2f}")
    
    return daily_consumption

def prepare_consumer_specific_data(df_readings, df_temperature):
    """Prepare consumer-specific data with features for LightGBM."""
    logging.info("Preparing consumer-specific data...")
    
    daily_consumption = df_readings.copy()
    daily_consumption['ds'] = daily_consumption['reading_date']
    daily_consumption['y'] = daily_consumption['consumption'].clip(lower=0.0, upper=100.0)
    daily_consumption['group'] = daily_consumption['group'].astype(int)
    
    if df_temperature is not None:
        daily_consumption = daily_consumption.merge(df_temperature, left_on='ds', right_on='reading_date', how='left')
        if 'tavg' in daily_consumption.columns:
            daily_consumption = daily_consumption.rename(columns={'tavg': 'avg_temp'})
        daily_consumption['avg_temp'] = daily_consumption['avg_temp'].fillna(daily_consumption['avg_temp'].median())
    
    daily_consumption['year'] = daily_consumption['ds'].dt.year
    daily_consumption['month'] = daily_consumption['ds'].dt.month
    daily_consumption['day_of_week'] = daily_consumption['ds'].dt.dayofweek
    daily_consumption['day_of_year'] = daily_consumption['ds'].dt.dayofyear
    daily_consumption['week_of_year'] = daily_consumption['ds'].dt.isocalendar().week
    
    daily_consumption = daily_consumption.sort_values(['consumer_id', 'ds'])
    daily_consumption['rolling_mean_7'] = daily_consumption.groupby('consumer_id')['y'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True).shift(1)
    daily_consumption['rolling_mean_14'] = daily_consumption.groupby('consumer_id')['y'].rolling(window=14, min_periods=1).mean().reset_index(0, drop=True).shift(1)
    daily_consumption['rolling_mean_30'] = daily_consumption.groupby('consumer_id')['y'].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True).shift(1)
    
    daily_consumption['rolling_mean_7'] = daily_consumption['rolling_mean_7'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_14'] = daily_consumption['rolling_mean_14'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_30'] = daily_consumption['rolling_mean_30'].fillna(daily_consumption['y'])
    
    daily_consumption = daily_consumption.dropna(subset=['y'])
    daily_consumption = daily_consumption[daily_consumption['y'] >= 0]
    
    logging.info(f"Consumer-specific data shape: {daily_consumption.shape}")
    logging.info(f"Date range: {daily_consumption['ds'].min()} to {daily_consumption['ds'].max()}")
    logging.info(f"Consumption range: {daily_consumption['y'].min():.2f} to {daily_consumption['y'].max():.2f}")
    
    return daily_consumption

def prepare_group_data(df_readings, df_temperature, group_id):
    """Prepare data for a specific group with features for LightGBM."""
    logging.info(f"Preparing data for group {group_id}...")
    
    group_data = df_readings[df_readings['group'] == group_id].copy()
    daily_consumption = group_data.groupby('ds')['y'].mean().reset_index()
    daily_consumption.columns = ['ds', 'y']
    
    if df_temperature is not None:
        daily_consumption = daily_consumption.merge(df_temperature, left_on='ds', right_on='reading_date', how='left')
        if 'tavg' in daily_consumption.columns:
            daily_consumption = daily_consumption.rename(columns={'tavg': 'avg_temp'})
        daily_consumption['avg_temp'] = daily_consumption['avg_temp'].fillna(daily_consumption['avg_temp'].median())
    
    daily_consumption['year'] = daily_consumption['ds'].dt.year
    daily_consumption['month'] = daily_consumption['ds'].dt.month
    daily_consumption['day_of_week'] = daily_consumption['ds'].dt.dayofweek
    daily_consumption['day_of_year'] = daily_consumption['ds'].dt.dayofyear
    daily_consumption['week_of_year'] = daily_consumption['ds'].dt.isocalendar().week
    
    daily_consumption = daily_consumption.sort_values('ds')
    daily_consumption['rolling_mean_7'] = daily_consumption['y'].rolling(window=7, min_periods=1).mean().shift(1)
    daily_consumption['rolling_mean_14'] = daily_consumption['y'].rolling(window=14, min_periods=1).mean().shift(1)
    daily_consumption['rolling_mean_30'] = daily_consumption['y'].rolling(window=30, min_periods=1).mean().shift(1)
    
    daily_consumption['rolling_mean_7'] = daily_consumption['rolling_mean_7'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_14'] = daily_consumption['rolling_mean_14'].fillna(daily_consumption['y'])
    daily_consumption['rolling_mean_30'] = daily_consumption['rolling_mean_30'].fillna(daily_consumption['y'])
    
    daily_consumption = daily_consumption.dropna(subset=['y'])
    daily_consumption = daily_consumption[daily_consumption['y'] >= 0]
    
    logging.info(f"Group {group_id} data shape: {daily_consumption.shape}")
    logging.info(f"Date range: {daily_consumption['ds'].min()} to {daily_consumption['ds'].max()}")
    logging.info(f"Consumption range: {daily_consumption['y'].min():.2f} to {daily_consumption['y'].max():.2f}")
    
    return daily_consumption

# ------------------ Model Training and Forecasting Functions ------------------
def train_lightgbm_model(data, test_size=0.2, params=None):
    """Train LightGBM model with given parameters and evaluate performance."""
    logging.info("Training LightGBM model...")
    logging.info(f"Parameters: {params}")
    
    feature_cols = ['avg_temp', 'year', 'month', 'day_of_week', 'day_of_year', 'week_of_year', 
                    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30']
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
    
    try:
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
    except Exception as e:
        logging.error(f"Failed to train LightGBM model: {e}")
        raise

def forecast_system_future(model, data, features, periods=100):  # Increased periods to 100 for consistency
    """Forecast future system-wide consumption for the specified periods."""
    logging.info(f"Forecasting next {periods} days (system-wide)...")
    
    try:
        last_date = data['ds'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        # Merge with forecasted temperatures
        future_data = pd.DataFrame({
            'ds': future_dates,
            'year': future_dates.year,  # Use .year directly on DatetimeIndex
            'month': future_dates.month,  # Use .month directly
            'day_of_week': future_dates.dayofweek,  # Use .dayofweek directly
            'day_of_year': future_dates.dayofyear,  # Use .dayofyear directly
            'week_of_year': future_dates.isocalendar().week  # Use .isocalendar().week directly
        })
        future_data = future_data.merge(future_temps, on='ds', how='left')
        future_data['avg_temp'] = future_data['avg_temp'].fillna(future_data['avg_temp'].mean())  # Fallback

        # Initialize rolling means with last historical values
        last_row = data.iloc[-1]
        future_predictions = np.zeros(periods)
        last_7 = list(data['y'].tail(7))
        last_14 = list(data['y'].tail(14))
        last_30 = list(data['y'].tail(30))

        # Iterative forecasting
        for i, date in enumerate(future_dates):
            row = {
                'ds': date,
                'year': date.year,
                'month': date.month,
                'day_of_week': date.dayofweek,
                'day_of_year': date.dayofyear,
                'week_of_year': date.isocalendar().week,
                'avg_temp': future_data.loc[future_data['ds'] == date, 'avg_temp'].values[0],
                'rolling_mean_7': np.mean(last_7) if last_7 else 0,
                'rolling_mean_14': np.mean(last_14) if last_14 else 0,
                'rolling_mean_30': np.mean(last_30) if last_30 else 0
            }
            X_future = pd.DataFrame([row])[features]
            pred = model.predict(X_future)[0]
            future_predictions[i] = pred

            # Update rolling means
            last_7.append(pred); last_7 = last_7[-7:]
            last_14.append(pred); last_14 = last_14[-14:]
            last_30.append(pred); last_30 = last_30[-30:]

        # Validate predictions
        if (future_predictions < 0).any():
            logging.warning(f"Negative predictions detected in system-wide forecast: {future_predictions[future_predictions < 0]}")
            future_predictions = np.clip(future_predictions, 0, None)
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': future_predictions
        })
        
        return forecast_df
    except Exception as e:
        logging.error(f"Failed to forecast system-wide: {e}")
        raise

def forecast_consumer_future(model, data, features, valid_consumers, periods=100):  # Increased periods to 100
    """Forecast future consumption for each consumer."""
    logging.info(f"Forecasting next {periods} days (consumer-specific)...")
    
    try:
        last_date = data['ds'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        future_data = []
        for consumer_id in valid_consumers:
            consumer_history = data[data['consumer_id'] == consumer_id]
            if consumer_history.empty:
                logging.warning(f"No valid data for consumer {consumer_id}, assigning zero forecast")
                for date in future_dates:
                    row = {
                        'ds': date,
                        'consumer_id': consumer_id,
                        'group': 0,
                        'year': date.year,
                        'month': date.month,
                        'day_of_week': date.dayofweek,
                        'day_of_year': date.dayofyear,
                        'week_of_year': date.isocalendar().week
                    }
                    future_data.append(row)
            else:
                last_row = consumer_history.iloc[-1]
                # Initialize rolling means
                last_7 = list(consumer_history['y'].tail(7)) if not consumer_history.empty else [0]*7
                last_14 = list(consumer_history['y'].tail(14)) if not consumer_history.empty else [0]*14
                last_30 = list(consumer_history['y'].tail(30)) if not consumer_history.empty else [0]*30

                for date in future_dates:
                    # Merge with future temperatures
                    temp_value = future_temps.loc[future_temps['ds'] == date, 'avg_temp'].values
                    avg_temp = temp_value[0] if len(temp_value) > 0 else last_row['avg_temp']
                    row = {
                        'ds': date,
                        'consumer_id': consumer_id,
                        'group': last_row['group'],
                        'year': date.year,
                        'month': date.month,
                        'day_of_week': date.dayofweek,
                        'day_of_year': date.dayofyear,
                        'week_of_year': date.isocalendar().week,
                        'avg_temp': avg_temp,
                        'rolling_mean_7': np.mean(last_7) if last_7 else 0,
                        'rolling_mean_14': np.mean(last_14) if last_14 else 0,
                        'rolling_mean_30': np.mean(last_30) if last_30 else 0
                    }
                    future_data.append(row)

        future_data = pd.DataFrame(future_data)
        X_future = future_data[features]
        future_predictions = model.predict(X_future)

        # Update rolling means iteratively after all predictions
        updated_predictions = np.zeros(len(future_dates) * len(valid_consumers))
        start_idx = 0
        for consumer_idx, consumer_id in enumerate(valid_consumers):
            consumer_start_idx = consumer_idx * len(future_dates)
            consumer_end_idx = (consumer_idx + 1) * len(future_dates)
            consumer_preds = future_predictions[consumer_start_idx:consumer_end_idx]
            
            consumer_history = data[data['consumer_id'] == consumer_id]
            last_7 = list(consumer_history['y'].tail(7)) if not consumer_history.empty else [0]*7
            last_14 = list(consumer_history['y'].tail(14)) if not consumer_history.empty else [0]*14
            last_30 = list(consumer_history['y'].tail(30)) if not consumer_history.empty else [0]*30

            for i, pred in enumerate(consumer_preds):
                updated_predictions[consumer_start_idx + i] = pred
                if i < len(future_dates) - 1:  # Update for next iteration within consumer
                    last_7.append(pred); last_7 = last_7[-7:]
                    last_14.append(pred); last_14 = last_14[-14:]
                    last_30.append(pred); last_30 = last_30[-30:]

        # Validate predictions
        if (updated_predictions < 0).any():
            logging.warning(f"Negative predictions detected in consumer-specific forecast: {updated_predictions[updated_predictions < 0]}")
            updated_predictions = np.clip(updated_predictions, 0, None)
        
        forecast_df = pd.DataFrame({
            'ds': future_data['ds'],
            'consumer_id': future_data['consumer_id'],
            'group': future_data['group'],
            'yhat': updated_predictions
        })
        
        return forecast_df
    except Exception as e:
        logging.error(f"Failed to forecast consumer-specific: {e}")
        raise

def forecast_group_future(model, data, features, group_id, periods=100):  # Increased periods to 100
    """Forecast future average consumption for a specific group."""
    logging.info(f"Forecasting next {periods} days for group {group_id}...")
    
    try:
        last_date = data['ds'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        # Merge with forecasted temperatures
        future_data = pd.DataFrame({
            'ds': future_dates,
            'year': future_dates.year,  # Use .year directly on DatetimeIndex
            'month': future_dates.month,  # Use .month directly
            'day_of_week': future_dates.dayofweek,  # Use .dayofweek directly
            'day_of_year': future_dates.dayofyear,  # Use .dayofyear directly
            'week_of_year': future_dates.isocalendar().week  # Use .isocalendar().week directly
        })
        future_data = future_data.merge(future_temps, on='ds', how='left')
        future_data['avg_temp'] = future_data['avg_temp'].fillna(future_data['avg_temp'].mean())  # Fallback

        # Initialize rolling means with last historical values
        last_row = data.iloc[-1]
        future_predictions = np.zeros(periods)
        last_7 = list(data['y'].tail(7))
        last_14 = list(data['y'].tail(14))
        last_30 = list(data['y'].tail(30))

        # Iterative forecasting
        for i, date in enumerate(future_dates):
            row = {
                'ds': date,
                'year': date.year,
                'month': date.month,
                'day_of_week': date.dayofweek,
                'day_of_year': date.dayofyear,
                'week_of_year': date.isocalendar().week,
                'avg_temp': future_data.loc[future_data['ds'] == date, 'avg_temp'].values[0],
                'rolling_mean_7': np.mean(last_7) if last_7 else 0,
                'rolling_mean_14': np.mean(last_14) if last_14 else 0,
                'rolling_mean_30': np.mean(last_30) if last_30 else 0
            }
            X_future = pd.DataFrame([row])[features]
            pred = model.predict(X_future)[0]
            future_predictions[i] = pred

            # Update rolling means
            last_7.append(pred); last_7 = last_7[-7:]
            last_14.append(pred); last_14 = last_14[-14:]
            last_30.append(pred); last_30 = last_30[-30:]

        # Validate predictions
        if (future_predictions < 0).any():
            logging.warning(f"Negative predictions detected in group {group_id} forecast: {future_predictions[future_predictions < 0]}")
            future_predictions = np.clip(future_predictions, 0, None)
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': future_predictions
        })
        
        return forecast_df
    except Exception as e:
        logging.error(f"Failed to forecast group-specific: {e}")
        raise

def cluster_consumers(daily_consumption, n_clusters=3):
    """Cluster consumers based on mean daily consumption, retaining all consumers."""
    logging.info("Clustering consumers for data quality...")
    consumer_means = daily_consumption.groupby('consumer_id')['consumption'].mean().reset_index()
    valid_consumers = consumer_means['consumer_id']
    clustering_consumers = consumer_means[consumer_means['consumption'] >= 0]['consumer_id']
    logging.info(f"Consumers with non-negative mean consumption: {len(clustering_consumers)}")
    X = consumer_means[consumer_means['consumer_id'].isin(clustering_consumers)]['consumption'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()
    logging.info("Group centers (average daily consumption):")
    for i, center in enumerate(centers):
        logging.info(f"Group {i}: {center:.2f} kWh/day")
    cluster_labels = pd.Series(-1, index=valid_consumers)
    cluster_labels[clustering_consumers] = labels
    user_group_df = pd.DataFrame({
        'consumer_id': consumer_means['consumer_id'],
        'mean_consumption': consumer_means['consumption'],
        'group': cluster_labels.values
    })
    daily_consumption = daily_consumption.merge(user_group_df[['consumer_id', 'group']], on='consumer_id', how='left')
    return daily_consumption, centers, valid_consumers

# ------------------ Main Execution ------------------
def main():
    """Main function to execute consumption forecasting pipeline."""
    start_time = time.time()
    logging.info("Starting LightGBM System-Wide, Consumer-Specific, and Group-Specific Consumption Forecasting")
    logging.info("=" * 60)

    # Model parameters
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
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
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
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 128,
        'learning_rate': 0.02,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'verbose': -1,
        'force_col_wise': True
    }

    # Step 1: Load and preprocess reading data
    logging.info("Step 1: Loading and merging reading data...")
    try:
        reading_files = glob.glob("csv/*reading*.csv")
        if not reading_files:
            raise FileNotFoundError("No reading files found in 'csv/' directory")
        logging.info(f"Found {len(reading_files)} reading files: {reading_files}")
        df_readings = dd.concat([dd.read_csv(f) for f in reading_files]).compute()
        df_readings['reading_date'] = pd.to_datetime(df_readings['reading_date'], errors='coerce')
        if df_readings['reading_date'].isna().any():
            logging.warning(f"Found {df_readings['reading_date'].isna().sum()} invalid reading dates, dropping...")
            df_readings = df_readings.dropna(subset=['reading_date'])
        if df_readings['consumer_id'].isna().any():
            logging.warning(f"Found {df_readings['consumer_id'].isna().sum()} missing consumer IDs, dropping...")
            df_readings = df_readings.dropna(subset=['consumer_id'])
    except Exception as e:
        logging.error(f"Failed to load reading files: {e}")
        raise

    # Select a small fraction (first 100 unique consumers) for testing
    # unique_consumers = df_readings['consumer_id'].unique()[:100]
    # df_readings = df_readings[df_readings['consumer_id'].isin(unique_consumers)]
    # logging.info(f"Selected {len(unique_consumers)} unique consumers for testing.")

    # Filter consumers with sufficient data
    min_days = 365
    days_per_consumer = df_readings.groupby('consumer_id')['reading_date'].nunique()
    logging.info(f"Min unique days per consumer: {days_per_consumer.min()}")
    logging.info(f"Max unique days per consumer: {days_per_consumer.max()}")
    valid_consumers = days_per_consumer[days_per_consumer >= min_days].index
    excluded_consumers = days_per_consumer[days_per_consumer < min_days].index
    logging.info(f"Excluded {len(excluded_consumers)} consumers due to insufficient data")
    df_readings = df_readings[df_readings['consumer_id'].isin(valid_consumers)]
    logging.info(f"Filtered to {len(valid_consumers)} consumers with >= {min_days} days")

    # Map consumer IDs
    consumer_id_map = {cid: i for i, cid in enumerate(df_readings['consumer_id'].unique())}
    df_readings['consumer_id'] = df_readings['consumer_id'].map(consumer_id_map)
    logging.info(f"Mapped {len(consumer_id_map)} unique consumer_id values to 0-{len(consumer_id_map)-1}")

    logging.info(f"Average unique days per consumer: {days_per_consumer.mean():.2f}")
    df_readings = df_readings.sort_values(['consumer_id', 'reading_date']).reset_index(drop=True)
    logging.info(f"Total readings: {len(df_readings)}")
    logging.info(f"Unique consumers: {df_readings['consumer_id'].nunique()}")
    df_readings['reading'] = df_readings['reading'] / 1000

    # Step 2: Load temperature data
    logging.info("Step 2: Loading temperature data...")
    try:
        df_temperature = pd.read_csv("csv/temperature.csv")
        df_temperature['date'] = pd.to_datetime(df_temperature['date'], errors='coerce')
        df_temperature = df_temperature.dropna(subset=['date'])
        df_temperature = df_temperature.rename(columns={'date': 'reading_date'})
        logging.info("Temperature data loaded successfully")
    except Exception as e:
        logging.warning(f"Could not load temperature data: {e}")
        df_temperature = None

    # Step 3: Calculate daily consumption
    logging.info("Step 3: Calculating daily consumption...")
    temp_df = prepare_all_daily_consumption(df_readings)
    logging.info(f"Average days per consumer after consumption calculation: {len(temp_df) / temp_df['consumer_id'].nunique():.2f}")

    # Step 4: Cluster consumers
    logging.info("Step 4: Clustering consumers for data quality...")
    temp_df, centroids, valid_consumers = cluster_consumers(temp_df, n_clusters=3)

    # Step 5: Remove outliers
    logging.info("Step 5: Removing group-wise outliers using IQR...")
    temp_df = remove_iqr_outliers(temp_df, group_col='group', value_col='consumption')

    # Step 6: Prepare data for models
    logging.info("Step 6: Preparing aggregated system-wide data...")
    system_wide_data = prepare_system_wide_data(temp_df, df_temperature)
    
    logging.info("Step 6: Preparing consumer-specific data...")
    consumer_specific_data = prepare_consumer_specific_data(temp_df, df_temperature)

    group_data = {}
    for group_id in sorted(consumer_specific_data['group'].unique()):
        logging.info(f"Step 6: Preparing data for group {group_id}...")
        group_data[group_id] = prepare_group_data(consumer_specific_data, df_temperature, group_id)

    # Step 7: Train models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.info("Step 7: Training system-wide LightGBM model...")
    system_model, system_features, system_rmse, system_mae = train_lightgbm_model(system_wide_data, test_size=0.2, params=system_params)
    system_model.save_model(f'lightgbm_system_model_{timestamp}.txt')
    
    logging.info("Step 7: Training consumer-specific LightGBM model...")
    consumer_model, consumer_features, consumer_rmse, consumer_mae = train_lightgbm_model(consumer_specific_data, test_size=0.2, params=consumer_params)
    consumer_model.save_model(f'lightgbm_consumer_model_{timestamp}.txt')

    group_models = {}
    group_features = {}
    group_rmse = {}
    group_mae = {}
    for group_id, g_data in group_data.items():
        logging.info(f"Step 7: Training group {group_id} LightGBM model...")
        params = group1_params if group_id == 1 else group_params
        g_model, g_features, g_rmse, g_mae = train_lightgbm_model(g_data, test_size=0.2, params=params)
        group_models[group_id] = g_model
        group_features[group_id] = g_features
        group_rmse[group_id] = g_rmse
        group_mae[group_id] = g_mae
        g_model.save_model(f'lightgbm_group_{group_id}_model_{timestamp}.txt')

    # Step 8: Forecast future consumption
    logging.info("Step 8: Forecasting future consumption (system-wide)...")
    system_forecast_df = forecast_system_future(system_model, system_wide_data, system_features, periods=100)
    
    logging.info("Step 8: Forecasting future consumption (consumer-specific)...")
    consumer_forecast_df = forecast_consumer_future(consumer_model, consumer_specific_data, consumer_features, valid_consumers, periods=100)

    group_forecasts = {}
    for group_id, g_model in group_models.items():
        logging.info(f"Step 8: Forecasting future consumption for group {group_id}...")
        group_forecasts[group_id] = forecast_group_future(g_model, group_data[group_id], group_features[group_id], group_id, periods=100)

    # Step 9: Validate outputs
    logging.info("Step 9: Validating output forecasts...")
    try:
        assert consumer_forecast_df['consumer_id'].nunique() == len(valid_consumers), \
            f"Expected {len(valid_consumers)} consumers, got {consumer_forecast_df['consumer_id'].nunique()}"
        assert consumer_forecast_df.shape[0] == len(valid_consumers) * 100, \
            f"Expected {len(valid_consumers) * 100} rows, got {consumer_forecast_df.shape[0]}"
        if (consumer_forecast_df['yhat'] < 0).any():
            logging.warning(f"Negative predictions in consumer forecast: {consumer_forecast_df[consumer_forecast_df['yhat'] < 0]['yhat'].values}")
            consumer_forecast_df['yhat'] = consumer_forecast_df['yhat'].clip(lower=0)
        logging.info("Consumer-specific forecast validation passed")
    except AssertionError as e:
        logging.error(f"Validation failed: {e}")
        raise

    # Step 10: Create visualizations
    logging.info("Step 10: Creating visualizations...")
    consumer_avg_forecast = consumer_forecast_df.groupby('ds')['yhat'].mean().reset_index(name='yhat_avg')
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
    logging.info("System forecast plot saved as 'lightgbm_system_forecast.png'")

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

    # Step 11: Save results
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

    # Step 12: Log final metrics
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

    # Step 13: Monitor performance
    with open('model_performance.log', 'a') as f:
        f.write(f"{datetime.now()}: System RMSE={system_rmse:.2f}, MAE={system_mae:.2f}, MAE/Median={(system_mae/system_wide_data['y'].median())*100:.1f}%\n")
        f.write(f"{datetime.now()}: Consumer RMSE={consumer_rmse:.2f}, MAE={consumer_mae:.2f}, MAE/Median={(consumer_mae/consumer_specific_data['y'].median())*100:.1f}%\n")
        for group_id in group_rmse:
            f.write(f"{datetime.now()}: Group {group_id} RMSE={group_rmse[group_id]:.2f}, MAE={group_mae[group_id]:.2f}, MAE/Median={(group_mae[group_id]/group_data[group_id]['y'].median())*100:.1f}%\n")

if __name__ == "__main__":
    main()