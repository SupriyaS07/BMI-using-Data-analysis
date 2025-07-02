# p2.py

import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

# Step 1: Load dataset
print("Loading dataset...")
df = pd.read_csv('bankmarketing.csv')
print(f"Dataset shape: {df.shape}")

# Step 2: Handle missing values
print(" Checking for missing values...")
missing = df.isnull().sum()
print(missing[missing > 0])

# Drop rows with missing values
df = df.dropna()
print(f"After dropping missing rows: {df.shape}")

# Step 3: Encode categorical columns
print("Encoding categorical columns using one-hot encoding...")
categorical_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, drop_first=True)

print(f"New shape after encoding: {df_encoded.shape}")

# Step 4: Save cleaned and preprocessed data
output_file = 'bankmarketing_cleaned.csv'
df_encoded.to_csv(output_file, index=False)
print(f"✅ Preprocessed data saved to: {output_file}")
