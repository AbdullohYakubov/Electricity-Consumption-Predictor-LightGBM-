import pandas as pd
import glob
import numpy as np
from datetime import timedelta
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import warnings
import time
warnings.filterwarnings('ignore')
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output

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

def prepare_consumer_data_with_regressors(df_readings, consumer_id, df_temp=None):
    # Filter by consumer
    consumer_data = df_readings[df_readings['consumer_id'] == consumer_id].copy()
    consumer_data = consumer_data.dropna(subset=['consumption'])

    if len(consumer_data) == 0:
        return None
    
     # Merge temperature (if provided)
    if df_temp is not None:
        try:
            # Check if temperature column exists and rename if needed
            if 'tavg' in df_temp.columns:
                df_temp_renamed = df_temp.rename(columns={'tavg': 'avg_temp'})
            elif 'temperature' in df_temp.columns:
                df_temp_renamed = df_temp.rename(columns={'temperature': 'avg_temp'})
            elif 'avg_temp' in df_temp.columns:
                df_temp_renamed = df_temp
            else:
                print(f"Warning: No temperature column found in temperature data. Available columns: {df_temp.columns.tolist()}")
                df_temp_renamed = None
            
            if df_temp_renamed is not None:
                consumer_data = consumer_data.merge(df_temp_renamed, on='reading_date', how='left')
                # Fill missing temperature values with median
                if 'avg_temp' in consumer_data.columns:
                    consumer_data['avg_temp'] = consumer_data['avg_temp'].fillna(consumer_data['avg_temp'].median())
                else:
                    consumer_data['avg_temp'] = 20.0  # Default temperature if merge fails
            else:
                consumer_data['avg_temp'] = 20.0  # Default temperature
        except Exception as e:
            print(f"Warning: Temperature merge failed for consumer {consumer_id}: {e}")
            consumer_data['avg_temp'] = 20.0  # Default temperature
    else:
        consumer_data['avg_temp'] = 20.0  # Default temperature if no temperature data provided

    # Check which columns exist before selecting
    available_columns = ['reading_date', 'consumption', 'recent_payments', 'avg_temp']
    existing_columns = [col for col in available_columns if col in consumer_data.columns]
    
    prophet_data = consumer_data[existing_columns].copy()
    
    # Calculate rolling means with proper handling of NaN values
    prophet_data['rolling_mean_7'] = prophet_data['consumption'].rolling(window=7, min_periods=1).mean()
    prophet_data['rolling_mean_14'] = prophet_data['consumption'].rolling(window=14, min_periods=1).mean()
    prophet_data['rolling_mean_30'] = prophet_data['consumption'].rolling(window=30, min_periods=1).mean()
    
    # Shift rolling means to avoid look-ahead bias (use previous values)
    prophet_data['rolling_mean_7'] = prophet_data['rolling_mean_7'].shift(1)
    prophet_data['rolling_mean_14'] = prophet_data['rolling_mean_14'].shift(1)
    prophet_data['rolling_mean_30'] = prophet_data['rolling_mean_30'].shift(1)
    
    # Fill NaN values in rolling means with the consumption value itself
    prophet_data['rolling_mean_7'] = prophet_data['rolling_mean_7'].fillna(prophet_data['consumption'])
    prophet_data['rolling_mean_14'] = prophet_data['rolling_mean_14'].fillna(prophet_data['consumption'])
    prophet_data['rolling_mean_30'] = prophet_data['rolling_mean_30'].fillna(prophet_data['consumption'])
    
    # Rename columns to Prophet format
    column_mapping = {
        'reading_date': 'ds',
        'consumption': 'y',
        'recent_payments': 'recent_payments',
        'avg_temp': 'avg_temp',
        'rolling_mean_7': 'rolling_mean_7',
        'rolling_mean_14': 'rolling_mean_14',
        'rolling_mean_30': 'rolling_mean_30'
    }
    
    prophet_data.columns = [column_mapping.get(col, col) for col in prophet_data.columns]
    prophet_data = prophet_data[prophet_data['y'] >= 0]
    
    # Drop rows with NaN in required columns
    required_cols = ['recent_payments', 'avg_temp']
    prophet_data = prophet_data.dropna(subset=required_cols)

    MIN_CONSUMPTION_POINTS = 30 # Minimum number of valid consumption points required

    if len(prophet_data) < MIN_CONSUMPTION_POINTS:
        print(f"Consumer {consumer_id}: only {len(prophet_data)} valid points, skipping (threshold = {MIN_CONSUMPTION_POINTS})")
        return None
    
    return prophet_data

def fit_and_forecast_prophet(prophet_data, periods=100):
    try:
        # Ensure sorted unique dates and clean targets
        prophet_data = prophet_data.sort_values('ds').copy()  # Removed drop_duplicates to keep all readings
        prophet_data = prophet_data[np.isfinite(prophet_data['y'])]

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )

        # Add regressors only if they exist in the data
        available_regressors = []
        for reg in ['recent_payments', 'avg_temp', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30']:
            if reg in prophet_data.columns:
                model.add_regressor(reg)
                available_regressors.append(reg)

        model.fit(prophet_data)
        
        # Debug: Check data structure
        print(f"Prophet data shape: {prophet_data.shape}")
        print(f"Date range: {prophet_data['ds'].min()} to {prophet_data['ds'].max()}")
        print(f"Y values range: {prophet_data['y'].min():.2f} to {prophet_data['y'].max():.2f}")
        print(f"Any NaN in y: {prophet_data['y'].isna().any()}")
        
        # Additional debugging for data quality
        zero_count = (prophet_data['y'] == 0).sum()
        zero_percentage = (zero_count / len(prophet_data)) * 100
        print(f"Zero consumption values: {zero_count}/{len(prophet_data)} ({zero_percentage:.1f}%)")
        print(f"Y statistics: mean={prophet_data['y'].mean():.2f}, std={prophet_data['y'].std():.2f}, median={prophet_data['y'].median():.2f}")
        
        # Check for suspicious patterns (too many identical values)
        unique_values = prophet_data['y'].nunique()
        print(f"Unique Y values: {unique_values}/{len(prophet_data)} ({unique_values/len(prophet_data)*100:.1f}% unique)")
        
        # Make future predictions
        future = model.make_future_dataframe(periods=periods)
        
        # Set future values for available regressors
        for reg in available_regressors:
            last_val = prophet_data[reg].iloc[-1]
            future[reg] = last_val

        forecast = model.predict(future)
    
        # Calculate metrics if we have enough data
        metrics = {}
        try:
            # Use simpler, more reliable cross-validation parameters
            data_length_days = (prophet_data['ds'].max() - prophet_data['ds'].min()).days
            print(f"Consumer data length: {data_length_days} days, {len(prophet_data)} data points")
            
            # Use fixed parameters that worked before
            if data_length_days >= 365:
                initial_period = '365 days'  # 1 year instead of 730 days
                period = '90 days'
                horizon = '90 days'  # Shorter horizon
                print(f"Using standard CV: initial={initial_period}, period={period}, horizon={horizon}")
            else:
                # Skip cross-validation for very short time series
                raise ValueError(f"Data too short for cross-validation: {data_length_days} days")
            
            print(f"Starting cross-validation...")
            df_cv = cross_validation(
                model,
                initial=initial_period,
                period=period,
                horizon=horizon
            )
            print(f"Cross-validation completed with {len(df_cv)} folds")
            df_metrics = performance_metrics(df_cv)
            metrics['rmse'] = df_metrics['rmse'].mean()
            metrics['mae'] = df_metrics['mae'].mean()
            metrics['mape'] = df_metrics['mape'].mean()
            print(f"CV metrics calculated: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")
        except Exception as e:
            print(f"Cross-validation failed for consumer: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            metrics = {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}

        # Fallback: manual train/test if metrics are NaN
        if (not metrics) or any(np.isnan([metrics.get('rmse', np.nan), metrics.get('mae', np.nan)])):
            print("Falling back to manual train/test evaluation...")
            n = len(prophet_data)
            test_size = max(60, int(0.2 * n)) if n > 120 else max(30, int(0.15 * n))
            if n - test_size < 30:
                test_size = max(15, n - 30)
            train_df = prophet_data.iloc[: n - test_size].copy()
            test_df = prophet_data.iloc[n - test_size :].copy()

            # Refit a fresh model on train only
            fallback_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            for reg in available_regressors:
                fallback_model.add_regressor(reg)
            fallback_model.fit(train_df)

            # Predict on the test horizon using known regressors
            required_cols = ['ds'] + available_regressors
            future_eval = test_df[required_cols].copy() if available_regressors else test_df[['ds']].copy()
            pred = fallback_model.predict(future_eval)
            y_true = test_df['y'].values
            y_pred = pred['yhat'].values
            rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
            mae = np.abs(y_true - y_pred).mean()
            metrics['rmse'] = rmse
            metrics['mae'] = mae
            print(f"Fallback metrics: RMSE={rmse:.2f}, MAE={mae:.2f}")

        return {
            'model': model,
            'forecast': forecast,
            'metrics': metrics,
            'data_points': len(prophet_data)
        }
    
    except Exception as e:
        print(f"Error fitting Prophet model: {e}")
        return None

def process_single_consumer(args):
    consumer_id, df_readings, df_temperature, periods = args
    prophet_data = prepare_consumer_data_with_regressors(df_readings, consumer_id, df_temperature)
    if prophet_data is None:
        return {
            'consumer_id': consumer_id,
            'status': 'insufficient_data',
            'data_points': len(prophet_data) if prophet_data is not None else 0
        }
    
    else:
        result = fit_and_forecast_prophet(prophet_data, periods)

    if result is None:
        return {
            'consumer_id': consumer_id,
            'status': 'model_failed',
            'data_points': len(prophet_data)
        }
    
    return {
        'consumer_id': consumer_id,
        'status': 'success',
        'model': result['model'],
        'forecast': result['forecast'],
        'metrics': result['metrics'],
        'data_points': result['data_points']
    }

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

def main():
    start_time = time.time()
    print("Starting Prophet Time Series Forecasting for Utility Consumption")
    print("=" * 60)

    # 1. Read and merge all reading files
    print("Step 1: Loading and merging reading data...")
    reading_files = glob.glob("csv/*reading*.csv")
    print(f"Found {len(reading_files)} reading files: {reading_files}")

    df_readings = pd.concat([pd.read_csv(f) for f in reading_files], ignore_index=True)
    df_readings['reading_date'] = pd.to_datetime(df_readings['reading_date'])
    df_readings = df_readings.dropna(subset=['reading_date'])

    #. Printing out basic stats
    df_readings = df_readings.sort_values(['consumer_id', 'reading_date']).reset_index(drop=True)
    print(f"Total readings: {len(df_readings)}")
    print(f"Unique consumers: {df_readings['consumer_id'].nunique()}")

    # Converting from W/h to kW/h
    df_readings['reading'] = df_readings['reading'] / 1000

    # 3. Read temperature
    df_temperature = pd.read_csv("csv/temperature.csv")
    df_temperature['date'] = pd.to_datetime(df_temperature['date'])
    df_temperature = df_temperature.dropna(subset=['date'])
    # Rename 'date' to 'reading_date' to match the merge column
    df_temperature = df_temperature.rename(columns={'date': 'reading_date'})

    
  
    # 2. Remove duplicates and sort
    # Find duplicate rows (excluding the first occurrence)
    # duplicates = df_readings.duplicated(subset=['consumer_id', 'reading_date'], keep=False)

    # Show all duplicates
    # duplicate_rows = df_readings[duplicates]
    # print(f"Number of duplicate (consumer_id, reading_date) pairs: {duplicate_rows.shape[0]}")
    # print(duplicate_rows)
    # df_readings = df_readings.drop_duplicates(subset=['consumer_id', 'reading_date'])

    # 3. Read payments and calculate recent payments
    df_payments = pd.read_csv("csv/confirmed_payment.csv")
    df_payments['payment_date'] = pd.to_datetime(df_payments['payment_date'])
    df_payments = df_payments.dropna(subset=['payment_date'])

    print("Available columns:", df_readings.columns.tolist())

    # first_100_consumers = df_readings['consumer_id'].drop_duplicates().head(100)
    # df_readings = df_readings[df_readings['consumer_id'].isin(first_100_consumers)]

    df_recent_payments = calculate_recent_payments(df_readings, df_payments)
    print(f"Date range: {df_readings['reading_date'].min()} to {df_readings['reading_date'].max()}")

    print("Available columns:", df_readings.columns.tolist())

    # df_readings = df_readings.merge(
    #     df_recent_payments,
    #     on=['consumer_id', 'reading_date'],
    #     how='left'
    # )

    # Step 2.5: KMeans Clustering of Consumers by Average Daily Consumption
    print("\nStep 2.5: Clustering consumers based on average daily consumption...")

    # Calculate daily consumption per consumer (you already expanded it in prepare_consumer_data_with_regressors)
    temp_df = prepare_all_daily_consumption(df_readings)

    # Filter out negative consumption before clustering
    temp_df = temp_df[temp_df['consumption'] >= 0]
    print(f"After removing negative consumption: {len(temp_df)} rows")

    # Merge recent_payments into temp_df
    temp_df = temp_df.merge(
        df_recent_payments[['consumer_id', 'reading_date', 'recent_payments']],
        on=['consumer_id', 'reading_date'],
        how='left'
    )
    temp_df['recent_payments'] = temp_df['recent_payments'].fillna(0.0)

    # Calculate mean daily consumption per consumer
    user_mean = temp_df.groupby('consumer_id')['consumption'].mean()
    
    # Filter out consumers with negative or zero mean consumption
    user_mean = user_mean[user_mean > 0]
    print(f"Consumers with positive mean consumption: {len(user_mean)}")

    # Reshape for clustering
    X = user_mean.values.reshape(-1, 1)

    # Perform clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    user_groups = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Print cluster centroids
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

    # Step 2.6: Remove outliers per group using IQR
    print("Step 2.6: Removing group-wise outliers using IQR...")
    temp_df = remove_iqr_outliers(temp_df, group_col='group', value_col='consumption')

    first_10_consumers = temp_df['consumer_id'].drop_duplicates().head(10)
    df_readings = temp_df[temp_df['consumer_id'].isin(first_10_consumers)]

    # 4. Get list of consumers with sufficient data
    print("\nStep 2: Identifying consumers with sufficient data...")
    consumer_counts = df_readings.groupby('consumer_id').size()
    consumers_with_data = consumer_counts[consumer_counts >= 5].index.tolist()
    print(f"Consumers with 5+ readings: {len(consumers_with_data):,}")

    # # Step 5: Forecasting for each group
    # print("\nStep 5: Group-level forecasting...")

    # for group_id in sorted(user_group_df['group'].unique()):
    #     print(f"\nProcessing Group {group_id}...")

    #     group_data = temp_df[temp_df['group'] == group_id]
    #     daily_avg = group_data.groupby('reading_date')['consumption'].mean().reset_index()
    #     daily_avg.columns = ['ds', 'y']
        
    #     if len(daily_avg) < 30:
    #         print(f"Group {group_id} has too little data, skipping.")
    #         continue

    #     # Fit Prophet
    #     m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    #     m.fit(daily_avg)
    #     future = m.make_future_dataframe(periods=100)
    #     forecast = m.predict(future)

    #     # Plot
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=daily_avg['ds'], y=daily_avg['y'], mode='lines', name='Historical'))
    #     fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    #     fig.update_layout(
    #         title=f"Group {group_id} Consumption Forecast",
    #         xaxis_title="Date",
    #         yaxis_title="Avg Daily Consumption (kWh)"
    #     )
    #     fig.write_html(f"group_{group_id}_forecast.html")
    #     print(f"Group {group_id} forecast saved as group_{group_id}_forecast.html")

    # 5. Process consumers (with parallel processing)
    print("\nStep 3: Fitting Prophet models and forecasting...")
    args_list = [(consumer_id, df_readings, df_temperature, 100) for consumer_id in consumers_with_data]
    results = []
    with ProcessPoolExecutor() as executor:
        for i, result in enumerate(executor.map(process_single_consumer, args_list)):
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(consumers_with_data)} consumers")
    
    # 7. Analyze results
    print("\nStep 4: Analyzing results...")
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] != 'success']
    print(f"Successful models: {len(successful_results)}")
    print(f"Failed models: {len(failed_results)}")
    if successful_results:
        rmse_values = [r['metrics'].get('rmse', np.nan) for r in successful_results if 'rmse' in r['metrics']]
        mae_values = [r['metrics'].get('mae', np.nan) for r in successful_results if 'mae' in r['metrics']]
        
        # Convert None values to np.nan for proper handling
        rmse_values = [np.nan if val is None else val for val in rmse_values]
        mae_values = [np.nan if val is None else val for val in mae_values]
        
        # Check if we have any valid metrics
        valid_rmse = [val for val in rmse_values if not np.isnan(val)]
        valid_mae = [val for val in mae_values if not np.isnan(val)]
        
        if valid_rmse:
            print(f"\nAverage RMSE: {np.mean(valid_rmse):.2f}")
        else:
            print(f"\nAverage RMSE: No valid RMSE values available")
            
        if valid_mae:
            print(f"Average MAE: {np.mean(valid_mae):.2f}")
        else:
            print(f"Average MAE: No valid MAE values available")
            
        print("Mean consumption:", df_readings['consumption'].mean())
        print("Median consumption:", df_readings['consumption'].median())
        print("Max consumption:", df_readings['consumption'].max())
        print("Min consumption:", df_readings['consumption'].min())

        # 8. Visualize results for a few consumers
        print("\nStep 5: Creating visualizations...")
        fig, axes = plt.subplots(5, 1, figsize=(12, 15))
        for i, result in enumerate(successful_results[:5]):
            consumer_id = result['consumer_id']
            forecast = result['forecast']
            prophet_data = prepare_consumer_data_with_regressors(df_readings, consumer_id, df_temperature)
            ax = axes[i]
            ax.plot(prophet_data['ds'], prophet_data['y'], 'b.', label='Actual', alpha=0.7)
            ax.plot(forecast['ds'], forecast['yhat'], 'r-', label='Forecast', alpha=0.8)
            ax.fill_between(forecast['ds'],
                            forecast['yhat_lower'],
                            forecast['yhat_upper'],
                            alpha=0.3, color='red', label='Confidence Interval')
            ax.set_title(f'Consumer {consumer_id}: Consumption Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Consumption')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('prophet_forecasts.png', dpi=300, bbox_inches='tight')
        print("Forecast plots saved as 'prophet_forecasts.png'")

        # 9. Save results
        print("\nStep 6: Saving results...")
        summary_df = pd.DataFrame([
            {
                'consumer_id': r['consumer_id'],
                'status': r['status'],
                'data_points': r['data_points'],
                'rmse': r.get('metrics', {}).get('rmse', np.nan),
                'mae': r.get('metrics', {}).get('mae', np.nan),
            }
            for r in results
        ])
        summary_df.to_csv('prophet_results_summary.csv', index=False)
        print("Results summary saved as 'prophet_results_summary.csv'")
        all_forecasts = []
        for result in successful_results:
            forecast_df = result['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_df['consumer_id'] = result['consumer_id']
            all_forecasts.append(forecast_df)
        if all_forecasts:
            combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
            combined_forecasts.to_csv('prophet_forecasts.csv', index=False)
            print("Detailed forecasts saved as 'prophet_forecasts.csv'")
    print("\nProphet forecasting completed!")
    print("=" * 60)

    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
    # print(forecast['ds'].tail(10))

    # empty_forecasts = 0
    # for result in successful_results:
    #     if result['forecast'].empty:
    #         empty_forecasts += 1
    # print(f"Пустых прогнозов: {empty_forecasts}")

    df = pd.read_csv('prophet_forecasts.csv')
    print(df['consumer_id'].nunique())

    # Prompt user to launch dashboard
    launch = input("\nDo you want to launch the interactive dashboard? (y/n): ")
    if launch.strip().lower() == 'y':
        run_dashboard()
        return

def run_dashboard():
    # Load results files
    summary_df = pd.read_csv('prophet_results_summary.csv')
    forecast_df = pd.read_csv('prophet_forecasts.csv')
    summary_df['consumer_id'] = summary_df['consumer_id'].astype(str)
    forecast_df['consumer_id'] = forecast_df['consumer_id'].astype(str)
    consumer_ids = summary_df['consumer_id'].unique()

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Individual Consumer Consumption Forecast Dashboard"),
        html.Label("Select Consumer:"),
        dcc.Dropdown(
            id='consumer-dropdown',
            options=[{'label': str(cid), 'value': str(cid)} for cid in consumer_ids],
            value=str(consumer_ids[0])
        ),
        html.Div(id='metrics-output'),
        dcc.Graph(id='forecast-graph')
    ])

    @app.callback(
        [Output('metrics-output', 'children'),
         Output('forecast-graph', 'figure')],
        [Input('consumer-dropdown', 'value')]
    )
    def update_dashboard(consumer_id):
        row = summary_df[summary_df['consumer_id'] == consumer_id]
        if row.empty:
            return [html.P("No data for this consumer.")], go.Figure()
        row = row.iloc[0]
        metrics = [
            html.P(f"RMSE: {row['rmse']:.2f}"),
            html.P(f"MAE: {row['mae']:.2f}"),
            html.P(f"Data Points: {row['data_points']}")
        ]
        consumer_forecast = forecast_df[forecast_df['consumer_id'] == consumer_id]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=consumer_forecast['ds'],
            y=consumer_forecast['yhat'],
            mode='lines',
            name='Forecast'
        ))
        fig.add_trace(go.Scatter(
            x=consumer_forecast['ds'],
            y=consumer_forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(dash='dash', color='rgba(255,0,0,0.3)')
        ))
        fig.add_trace(go.Scatter(
            x=consumer_forecast['ds'],
            y=consumer_forecast['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            line=dict(dash='dash', color='rgba(0,0,255,0.3)')
        ))
        fig.update_layout(
            title=f"Forecast for Consumer {consumer_id}",
            xaxis_title="Date",
            yaxis_title="Consumption",
            legend=dict(x=0, y=1)
        )
        return metrics, fig

    app.run(debug=False)

if __name__ == "__main__":
    main() 