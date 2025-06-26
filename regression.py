import pandas as pd
import glob
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Read and merge all readings
reading_files = glob.glob("*reading*.csv")
df_readings = pd.concat([pd.read_csv(f) for f in reading_files], ignore_index=True)
df_readings['reading_date'] = pd.to_datetime(df_readings['reading_date'])

# 2. Calculate consumption for each consumer
df_readings = df_readings.sort_values(['consumer_id', 'reading_date'])
df_readings['prev_reading'] = df_readings.groupby('consumer_id')['reading'].shift(1)
df_readings['consumption'] = df_readings['reading'] - df_readings['prev_reading']

# Remove first readings (no previous reading to compare)
df_readings = df_readings.dropna(subset=['consumption'])

# 3. Read and merge all balances
balance_files = glob.glob("*balance*.csv")
df_balances = pd.concat([pd.read_csv(f) for f in balance_files], ignore_index=True)
df_balances['period'] = pd.to_datetime(df_balances['period'])

# Remove duplicates in merge keys
dupes = df_readings.duplicated(subset=['consumer_id', 'reading_date'])
print("Number of duplicate (consumer_id, reading_date):", dupes.sum())
if dupes.sum() > 0:
    print(df_readings[dupes])

df_readings = df_readings.drop_duplicates(subset=['consumer_id', 'reading_date'])
df_balances = df_balances.drop_duplicates(subset=['consumer_id', 'period'])

# Drop NaT/NaN in date columns
df_readings = df_readings.dropna(subset=['reading_date'])
df_balances = df_balances.dropna(subset=['period'])

# Ensure datetime types
df_readings['reading_date'] = pd.to_datetime(df_readings['reading_date'])
df_balances['period'] = pd.to_datetime(df_balances['period'])

print(df_readings['reading_date'].isna().sum())
print(df_balances['period'].isna().sum())

print(df_readings.duplicated(subset=['consumer_id', 'reading_date']).sum())

print(df_readings['reading_date'].dtype)
print(df_balances['period'].dtype)

print(df_readings[['consumer_id', 'reading_date']].head(20))
print(df_readings[['consumer_id', 'reading_date']].tail(20))
print(df_readings['consumer_id'].is_monotonic_increasing)

# Sort by both keys and reset index
df_readings = df_readings.sort_values(['consumer_id', 'reading_date']).reset_index(drop=True)
df_balances = df_balances.sort_values(['consumer_id', 'period']).reset_index(drop=True)

# Print to check sorting
# print("Readings head:")
# print(df_readings[['consumer_id', 'reading_date']].head(10))
# print("Balances head:")
# print(df_balances[['consumer_id', 'period']].head(10))
# print("reading_date dtype:", df_readings['reading_date'].dtype)
# print("period dtype:", df_balances['period'].dtype)
print(df_balances[['consumer_id', 'period']].head(20))
print(df_balances[['consumer_id', 'period']].tail(20))

print(df_balances['consumer_id'].is_monotonic_increasing)

print(df_balances['consumer_id'].isna().sum())
print(df_balances['period'].isna().sum())

# 4. Merge latest balance before reading_date for each consumer
# df_readings = pd.merge_asof(
#     df_readings,
#     df_balances,
#     by='consumer_id',
#     left_on='reading_date',
#     right_on='period',
#     direction='backward'
# )

# test_merge = pd.merge_asof(
#     df_readings.head(100),
#     df_balances,
#     by='consumer_id',
#     left_on='reading_date',
#     right_on='period',
#     direction='backward'
# )
# print(test_merge.head())
# cid = df_readings['consumer_id'].iloc[0]
# df_readings_cid = df_readings[df_readings['consumer_id'] == cid]
# df_balances_cid = df_balances[df_balances['consumer_id'] == cid]

# test_merge = pd.merge_asof(
#     df_readings_cid,
#     df_balances_cid,
#     by='consumer_id',
#     left_on='reading_date',
#     right_on='period',
#     direction='backward'
# )
# print(test_merge.head())
df_readings = pd.merge_asof(
    df_readings,
    df_balances,
    by='consumer_id',
    left_on='reading_date',
    right_on='period',
    direction='backward'
)

# 5. Read payments
df_payments = pd.read_csv("confirmed_payment.csv")
df_payments['payment_date'] = pd.to_datetime(df_payments['payment_date'])

# 6. Feature: sum of payments in last 30 days for each reading
def sum_recent_payments(row):
    cid = row['consumer_id']
    date = row['reading_date']
    mask = (df_payments['consumer_id'] == cid) & \
           (df_payments['payment_date'] <= date) & \
           (df_payments['payment_date'] > date - timedelta(days=30))
    return df_payments.loc[mask, 'amount'].sum()

df_readings['recent_payments'] = df_readings.apply(sum_recent_payments, axis=1)

# 7. Feature engineering: add time features
df_readings['month'] = df_readings['reading_date'].dt.month
df_readings['dayofweek'] = df_readings['reading_date'].dt.dayofweek

# 8. Drop rows with missing values (optional)
df_readings = df_readings.dropna()

# 9. Prepare features and target
features = ['balance_in', 'balance_out', 'recent_payments', 'month', 'dayofweek']
X = df_readings[features]
y = df_readings['consumption']

# 10. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 12. Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# 13. Save model (optional)
import joblib
joblib.dump(model, "consumption_model.pkl")

# 14. Predict for new data (example)
# new_X = ... # DataFrame with same features as X
# predictions = model.predict(new_X)
