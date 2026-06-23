"""
Inspect landmark dataset.
"""

import pandas as pd


df = pd.read_csv(
    "data/pose_landmarks.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(len(df.columns))

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nFirst 5 Rows:")
print(df.head())