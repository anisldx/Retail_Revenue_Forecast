import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv('Air_solution_sales.csv')

df = df.rename(columns={'Timestamp': 'ds', 'target': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

if 'Promotion' in df.columns:
    df['Promotion'] = df['Promotion'].astype(float)  

# Train-test split
train_df = df.iloc[:335]  # First 335 days for training
test_df = df.iloc[335:]   # Last 30 days for testing


m = Prophet()
m.add_regressor('Promotion')  # Add Promotion as a past covariate

m.fit(train_df)

future = m.make_future_dataframe(periods=30)

future = future.merge(df[['ds', 'Promotion']], on='ds', how='left').fillna(0)

forecast = m.predict(future)

forecast_test = forecast.iloc[-30:][['ds', 'yhat']]
actual_test = test_df[['ds', 'y']]

rmse = root_mean_squared_error(actual_test['y'], forecast_test['yhat'])
mape = np.mean(np.abs((actual_test['y'] - forecast_test['yhat']) / actual_test['y'])) * 100  # MAPE in %

# Print results
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
