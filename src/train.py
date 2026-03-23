import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb


# Data loading and parsing
gen_df = pd.read_csv('data/Plant_1_Generation_Data.csv')
sensor_df = pd.read_csv('data/Plant_1_Weather_Sensor_Data.csv')

# Parse dates
# Note: The format in the CSV is dd-mm-yyyy hh:mm
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'])
sensor_df['DATE_TIME'] = pd.to_datetime(sensor_df['DATE_TIME'])

df = pd.merge(gen_df, sensor_df, on='DATE_TIME', how='left')
print("Merged Shape:", df.shape)

# Drop redundant columns if any (Plant_ID check)
df = df.drop(columns=['PLANT_ID_y'])
df = df.rename(columns={'PLANT_ID_x': 'PLANT_ID', 'SOURCE_KEY_x': 'INVERTER_ID', 'SOURCE_KEY_y': 'SENSOR_ID'})

# Drop Missing
df = df.dropna()
print("Final Shape:", df.shape)

# Extract Time Features
df['hour'] = df['DATE_TIME'].dt.hour
df['minute'] = df['DATE_TIME'].dt.minute
df['month'] = df['DATE_TIME'].dt.month
df['day_of_year'] = df['DATE_TIME'].dt.dayofyear

# Select Features and Target
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'hour', 'minute', 'month', 'day_of_year']
target = 'AC_POWER'

X = df[features]
y = df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training Tuned XGBoost

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3,6]
}

grid_xgb = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid_xgb, cv=3, scoring='neg_root_mean_squared_error')
grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
preds_xgb_tuned = best_xgb.predict(X_test)
rmse_xgb_tuned = np.sqrt(mean_squared_error(y_test, preds_xgb_tuned))
r2_xgb_tuned = r2_score(y_test, preds_xgb_tuned)

print(f"XGB Params: {grid_xgb.best_params_}")
print(f"XGBoost (Tuned) - RMSE: {rmse_xgb_tuned:.4f}kW, R2: {r2_xgb_tuned:.4f}")

# Save the model
with open('xgb_tuned.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)


print(f"Model saved as 'xgb_tuned.pkl'")
