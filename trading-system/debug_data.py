import pandas as pd
import numpy as np

# Load the data
train_df = pd.read_parquet('/home/admin1/project9/trading-system/data/XAUUSD_1h_train.parquet')
val_df = pd.read_parquet('/home/admin1/project9/trading-system/data/XAUUSD_1h_val.parquet')

print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
print("\nColumn names:", train_df.columns.tolist())
print("\nFirst 5 rows:")
print(train_df.head())

# Check for missing values
print("\nMissing values in train:")
print(train_df.isnull().sum())

# Basic statistics
print("\nTrain close price stats:")
print(train_df['close'].describe())