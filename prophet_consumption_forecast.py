import pandas as pd
import glob
import numpy as np
from datetime import timedelta
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

def calculate_recent_payments(df_readings, df_payments):
    # Efficiently calculate recent payments for each reading
    df_payments = df_payments.sort_values(['consumer_id', 'payment_date'])
    df_readings = df_readings.sort_values(['consumer_id', 'reading_date'])
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
    prophet_data = consumer_data[['reading_date', 'consumption', 'balance_in', 'balance_out', 'recent_payments']].copy()
    prophet_data.columns = ['ds', 'y', 'balance_in', 'balance_out', 'recent_payments']
    prophet_data = prophet_data[prophet_data['y'] >= 0]
    # Drop rows with NaNs in regressors
    prophet_data = prophet_data.dropna(subset=['balance_in', 'balance_out', 'recent_payments'])
    return prophet_data

def fit_and_forecast_prophet(prophet_data, periods=30):
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.add_regressor('balance_in')
        model.add_regressor('balance_out')
        model.add_regressor('recent_payments')
        model.fit(prophet_data)
        # Make future predictions
        future = model.make_future_dataframe(periods=periods)
        for reg in ['balance_in', 'balance_out', 'recent_payments']:
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
                metrics['mape'] = np.mean(np.abs((actual - predicted) / actual)) * 100
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
    if prophet_data is None or len(prophet_data) < 5:
        return {
            'consumer_id': consumer_id,
            'status': 'insufficient_data',
            'data_points': len(prophet_data) if prophet_data is not None else 0
        }
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
    print("Starting Prophet Time Series Forecasting for Utility Consumption")
    print("=" * 60)
    # 1. Read and merge all reading files
    print("Step 1: Loading and merging reading data...")
    reading_files = glob.glob("*reading*.csv")
    print(f"Found {len(reading_files)} reading files: {reading_files}")
    df_readings = pd.concat([pd.read_csv(f) for f in reading_files], ignore_index=True)
    df_readings['reading_date'] = pd.to_datetime(df_readings['reading_date'])
    # 2. Merge all balance files
    balance_files = glob.glob("*balance*.csv")
    df_balances = pd.concat([pd.read_csv(f) for f in balance_files], ignore_index=True)
    df_balances['period'] = pd.to_datetime(df_balances['period'])
    # Sort and remove duplicates
    df_readings = df_readings.drop_duplicates(subset=['consumer_id', 'reading_date'])
    df_balances = df_balances.drop_duplicates(subset=['consumer_id', 'period'])
    df_readings = df_readings.dropna(subset=['reading_date'])
    df_balances = df_balances.dropna(subset=['period'])
    df_readings = df_readings.sort_values(['consumer_id', 'reading_date']).reset_index(drop=True)
    df_balances = df_balances.sort_values(['consumer_id', 'period']).reset_index(drop=True)

    for cid, group in df_readings.groupby('consumer_id'):
        if not group['reading_date'].is_monotonic_increasing:
            print(f"Not sorted for consumer_id {cid}")
    # 3. Merge most recent balance info
    print("df_readings:")
    print(df_readings[['consumer_id', 'reading_date']].head(20))
    print(df_readings[['consumer_id', 'reading_date']].tail(20))
    print(df_readings[['consumer_id', 'reading_date']].sample(20))
    print("df_balances:")
    print(df_balances[['consumer_id', 'period']].head(20))
    print("Duplicates in df_readings:", df_readings.duplicated(subset=['consumer_id', 'reading_date']).sum())
    print("Duplicates in df_balances:", df_balances.duplicated(subset=['consumer_id', 'period']).sum())
    cid = df_readings['consumer_id'].iloc[0]
    df_readings_cid = df_readings[df_readings['consumer_id'] == cid].sort_values('reading_date').reset_index(drop=True)
    df_balances_cid = df_balances[df_balances['consumer_id'] == cid].sort_values('period').reset_index(drop=True)

    print(df_readings_cid)
    print(df_balances_cid)

    df_readings = pd.merge_asof(
        df_readings_cid,
        df_balances_cid,
        by='consumer_id',
        left_on='reading_date',
        right_on='period',
        direction='backward'
    )

    # df_readings = df_readings.drop(columns=['period'])

    missing_in_balances = set(df_readings['consumer_id']) - set(df_balances['consumer_id'])
    print(f"Consumers in readings but not in balances: {len(missing_in_balances)}")
    
    print("Columns after merge:", df_readings.columns)
    print(df_readings[['consumer_id', 'reading_date', 'period', 'balance_in', 'balance_out']].head(30))
    print(df_readings.isnull().sum())
    # 4. Read payments and calculate recent payments
    df_payments = pd.read_csv("confirmed_payment.csv")
    df_payments['payment_date'] = pd.to_datetime(df_payments['payment_date'])
    df_readings = calculate_recent_payments(df_readings, df_payments)
    print(f"Total readings loaded: {len(df_readings):,}")
    print(f"Date range: {df_readings['reading_date'].min()} to {df_readings['reading_date'].max()}")
    print(f"Unique consumers: {df_readings['consumer_id'].nunique():,}")
    # 5. Get list of consumers with sufficient data
    print("\nStep 2: Identifying consumers with sufficient data...")
    consumer_counts = df_readings.groupby('consumer_id').size()
    consumers_with_data = consumer_counts[consumer_counts >= 5].index.tolist()
    print(f"Consumers with 5+ readings: {len(consumers_with_data):,}")
    # For testing, use a subset of consumers
    test_consumers = consumers_with_data[:100]  # Start with 100 consumers
    print(f"Testing with first {len(test_consumers)} consumers")
    # 6. Process consumers (with parallel processing)
    print("\nStep 3: Fitting Prophet models and forecasting...")
    args_list = [(consumer_id, df_readings, 30) for consumer_id in test_consumers]
    results = []
    with ProcessPoolExecutor() as executor:
        for i, result in enumerate(executor.map(process_single_consumer, args_list)):
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_consumers)} consumers")
    # 7. Analyze results
    print("\nStep 4: Analyzing results...")
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] != 'success']
    print(f"Successful models: {len(successful_results)}")
    print(f"Failed models: {len(failed_results)}")
    if successful_results:
        rmse_values = [r['metrics'].get('rmse', np.nan) for r in successful_results if 'rmse' in r['metrics']]
        mae_values = [r['metrics'].get('mae', np.nan) for r in successful_results if 'mae' in r['metrics']]
        mape_values = [r['metrics'].get('mape', np.nan) for r in successful_results if 'mape' in r['metrics']]
        print(f"\nAverage RMSE: {np.nanmean(rmse_values):.2f}")
        print(f"Average MAE: {np.nanmean(mae_values):.2f}")
        print(f"Average MAPE: {np.nanmean(mape_values):.2f}%")
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
                'rmse': r['metrics'].get('rmse', np.nan),
                'mae': r['metrics'].get('mae', np.nan),
                'mape': r['metrics'].get('mape', np.nan)
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

    # Check sort per group

if __name__ == "__main__":
    main() 