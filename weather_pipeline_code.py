# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:28:28 2025

@author: rahul
"""

# Import required libraries
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import boto3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# --- 1. Data Ingestion ---
# Connect to AWS RDS (MySQL)
db_user = 'your_username'
db_password = 'your_password'
db_host = 'your_rds_endpoint'
db_name = 'sales_db'
engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}')

# Query transactional data (3 years)
query = """
SELECT date, store_id, sku, sales_qty, revenue
FROM transactions
WHERE date BETWEEN '2020-01-01' AND '2023-12-31'
AND state IN ('MA', 'AZ')
"""
sales_df = pd.read_sql(query, engine)

# Connect to S3 for NOAA weather data
s3_client = boto3.client('s3')
bucket_name = 'your-weather-bucket'
weather_file = 'noaa_weather_data.csv'
s3_client.download_file(bucket_name, weather_file, '/tmp/weather_data.csv')

# Load weather data
weather_df = pd.read_csv('/tmp/weather_data.csv')

# --- 2. Data Preprocessing and Feature Engineering ---
# Merge sales and weather data
merged_df = pd.merge(
    sales_df,
    weather_df[['date', 'location', 'temperature', 'precipitation', 'humidity']],
    left_on=['date', 'store_id'],
    right_on=['date', 'location'],
    how='left'
)

# Handle missing values
merged_df['temperature'].fillna(merged_df['temperature'].mean(), inplace=True)
merged_df['precipitation'].fillna(0, inplace=True)
merged_df['humidity'].fillna(merged_df['humidity'].mean(), inplace=True)

# Feature engineering
# Moving averages for weather variables
merged_df['temp_ma7'] = merged_df.groupby('store_id')['temperature'].transform(lambda x: x.rolling(7, min_periods=1).mean())
merged_df['precip_ma7'] = merged_df.groupby('store_id')['precipitation'].transform(lambda x: x.rolling(7, min_periods=1).mean())

# Lag features for sales
merged_df['sales_qty_lag1'] = merged_df.groupby(['store_id', 'sku'])['sales_qty'].shift(1)
merged_df['sales_qty_lag7'] = merged_df.groupby(['store_id', 'sku'])['sales_qty'].shift(7)

# Normalize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ['temperature', 'precipitation', 'humidity', 'temp_ma7', 'precip_ma7', 'sales_qty_lag1', 'sales_qty_lag7']
merged_df[numerical_cols] = scaler.fit_transform(merged_df[numerical_cols])

# --- 3. Exploratory Data Analysis ---
# Correlation analysis
corr_matrix = merged_df[['sales_qty', 'temperature', 'precipitation', 'humidity']].corr()
print("Correlation Matrix:\n", corr_matrix)

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title('Correlation Heatmap: Sales vs Weather Variables')
plt.savefig('/tmp/corr_heatmap.png')  # Save for Tableau import
plt.close()

# Region-specific trends
ma_sales = merged_df[merged_df['store_id'].str.contains('MA')]
az_sales = merged_df[merged_df['store_id'].str.contains('AZ')]

print("MA Sales vs Temperature Correlation:", ma_sales[['sales_qty', 'temperature']].corr().iloc[0, 1])
print("AZ Sales vs Precipitation Correlation:", az_sales[['sales_qty', 'precipitation']].corr().iloc[0, 1])

# --- 4. Model Development ---
# Prepare features and target
features = ['temperature', 'precipitation', 'humidity', 'temp_ma7', 'precip_ma7', 'sales_qty_lag1', 'sales_qty_lag7']
target = 'sales_qty'
X = merged_df[features].fillna(0)
y = merged_df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"Random Forest - MSE: {rf_mse:.2f}, R2: {rf_r2:.2f}")

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print(f"Linear Regression - MSE: {lr_mse:.2f}, R2: {lr_r2:.2f}")

# Feature importance (Random Forest)
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("Feature Importance:\n", feature_importance)

# Save feature importance plot
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xticks(rotation=45)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('/tmp/feature_importance.png')  # Save for Tableau import
plt.close()



# --- Save the Best Model ---
# Compare models based on MSE (or R2)
if rf_mse < lr_mse:
    best_model = rf_model
    best_model_name = 'random_forest'
    best_mse = rf_mse
    best_r2 = rf_r2
else:
    best_model = lr_model
    best_model_name = 'linear_regression'
    best_mse = lr_mse
    best_r2 = lr_r2

print(f"Best Model: {best_model_name}, MSE: {best_mse:.2f}, R2: {best_r2:.2f}")

# Save the best model locally
model_path = f'/tmp/{best_model_name}_model.joblib'
joblib.dump(best_model, model_path)

# Upload to S3
s3_client.upload_file(model_path, bucket_name, f'models/{best_model_name}_model.joblib')
print(f"Best model ({best_model_name}) saved to s3://{bucket_name}/models/{best_model_name}_model.joblib")
# --- 5. Visualization Preparation ---
# Prepare data for Tableau
output_df = merged_df[['date', 'store_id', 'sku', 'sales_qty', 'temperature', 'precipitation', 'humidity']].copy()
output_df['rf_pred'] = rf_model.predict(merged_df[features].fillna(0))
output_df.to_csv('/tmp/sales_forecast.csv', index=False)

# Upload to S3 for Tableau access
s3_client.upload_file('/tmp/sales_forecast.csv', bucket_name, 'sales_forecast.csv')
s3_client.upload_file('/tmp/corr_heatmap.png', bucket_name, 'corr_heatmap.png')
s3_client.upload_file('/tmp/feature_importance.png', bucket_name, 'feature_importance.png')

print("Data and visualizations uploaded to S3 for Tableau integration.")