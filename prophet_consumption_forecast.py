import pandas as pd
import glob
import numpy as np
from datetime import timedelta
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import warnings
import time
warnings.filterwarnings('ignore')

def calculate_recent_payments(df_readings, df_payments):
    # Sort payments and readings dataframes
    df_payments = df_payments.sort_values(['consumer_id', 'payment_date'])
    df_readings = df_readings.sort_values(['consumer_id', 'reading_date'])
    # Initialize recent payments at 0
    df_readings['recent_payments'] = 0.0

    # Group by consumer for efficiency
    for cid, group in df_readings.groupby('consumer_id'):
        payments = df_payments[df_payments['consumer_id'] == cid]
        if payments.empty:
            continue
        payments = payments.sort_values('payment_date')
        idx = group.index
        vals = []
        for date in group['reading_date']:
            mask = (payments['payment_date'] <= date) & (payments['payment_date'] > date - pd.Timedelta(days=30))
            vals.append(payments.loc[mask, 'amount'].sum())
        df_readings.loc[idx, 'recent_payments'] = vals
    return df_readings

def prepare_consumer_data_with_regressors(df_readings, consumer_id):
    consumer_data = df_readings[df_readings['consumer_id'] == consumer_id].copy()
    if len(consumer_data) < 2:
        return None
    consumer_data = consumer_data.sort_values('reading_date')
    consumer_data['prev_reading'] = consumer_data['reading'].shift(1)
    consumer_data['consumption'] = consumer_data['reading'] - consumer_data['prev_reading']
    consumer_data = consumer_data.dropna(subset=['consumption'])
    if len(consumer_data) == 0:
        return None
    prophet_data = consumer_data[['reading_date', 'consumption', 'recent_payments']].copy()
    prophet_data.columns = ['ds', 'y', 'recent_payments']
    prophet_data = prophet_data[prophet_data['y'] >= 0]
    # Drop rows with NaNs in regressors
    prophet_data = prophet_data.dropna(subset=['recent_payments'])
    if len(prophet_data) < 5:
        print(f"Consumer {consumer_id}: less than 5 valid rows after all cleaning")
        return None
    return prophet_data

def fit_and_forecast_prophet(prophet_data, periods=100):
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )

        model.add_regressor('recent_payments')
        model.fit(prophet_data)
        # Make future predictions
        future = model.make_future_dataframe(periods=periods)
        for reg in ['recent_payments']:
            last_val = prophet_data[reg].iloc[-1]
            future[reg] = last_val
        forecast = model.predict(future)
        # Calculate metrics if we have enough data
        metrics = {}
        if len(prophet_data) >= 10:
            split_idx = int(len(prophet_data) * 0.8)
            actual = prophet_data.iloc[split_idx:]['y'].values
            predicted = forecast.iloc[split_idx:len(prophet_data)]['yhat'].values
            if len(actual) > 0 and len(predicted) > 0:
                min_len = min(len(actual), len(predicted))
                actual = actual[:min_len]
                predicted = predicted[:min_len]
                metrics['rmse'] = np.sqrt(mean_squared_error(actual, predicted))
                metrics['mae'] = mean_absolute_error(actual, predicted)
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
    consumer_id, df_readings, periods = args
    prophet_data = prepare_consumer_data_with_regressors(df_readings, consumer_id)
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

    # Find duplicate rows (excluding the first occurrence)
    # duplicates = df_readings.duplicated(subset=['consumer_id', 'reading_date'], keep=False)

    # Show all duplicates
    # duplicate_rows = df_readings[duplicates]
    # print(f"Number of duplicate (consumer_id, reading_date) pairs: {duplicate_rows.shape[0]}")
    # print(duplicate_rows)
  
    # 2. Remove duplicates and sort
    df_readings = df_readings.drop_duplicates(subset=['consumer_id', 'reading_date'])
    df_readings = df_readings.dropna(subset=['reading_date'])
    df_readings = df_readings.sort_values(['consumer_id', 'reading_date']).reset_index(drop=True)
    print(f"Total readings: {len(df_readings)}")
    print(f"Unique consumers: {df_readings['consumer_id'].nunique()}")

    # 3. Read payments and calculate recent payments
    df_payments = pd.read_csv("csv/confirmed_payment.csv")
    df_payments['payment_date'] = pd.to_datetime(df_payments['payment_date'])
    df_readings = calculate_recent_payments(df_readings, df_payments)
    print(f"Date range: {df_readings['reading_date'].min()} to {df_readings['reading_date'].max()}")

    # 4. Get list of consumers with sufficient data
    print("\nStep 2: Identifying consumers with sufficient data...")
    consumer_counts = df_readings.groupby('consumer_id').size()
    consumers_with_data = consumer_counts[consumer_counts >= 5].index.tolist()
    print(f"Consumers with 5+ readings: {len(consumers_with_data):,}")

    # 5. Process consumers (with parallel processing)
    print("\nStep 3: Fitting Prophet models and forecasting...")
    args_list = [(consumer_id, df_readings, 100) for consumer_id in consumers_with_data]
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
        print(f"\nAverage RMSE: {np.nanmean(rmse_values):.2f}")
        print(f"Average MAE: {np.nanmean(mae_values):.2f}")
        print("Mean consumption:", df_readings['reading'].mean())
        print("Median consumption:", df_readings['reading'].median())
        print("Max consumption:", df_readings['reading'].max())
        print("Min consumption:", df_readings['reading'].min())

        # 8. Visualize results for a few consumers
        print("\nStep 5: Creating visualizations...")
        fig, axes = plt.subplots(5, 1, figsize=(12, 15))
        for i, result in enumerate(successful_results[:5]):
            consumer_id = result['consumer_id']
            forecast = result['forecast']
            prophet_data = prepare_consumer_data_with_regressors(df_readings, consumer_id)
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
    print(forecast['ds'].tail(10))

    empty_forecasts = 0
    for result in successful_results:
        if result['forecast'].empty:
            empty_forecasts += 1
    print(f"Пустых прогнозов: {empty_forecasts}")

    df = pd.read_csv('prophet_forecasts.csv')
    print(df['consumer_id'].nunique())



if __name__ == "__main__":
    main() 