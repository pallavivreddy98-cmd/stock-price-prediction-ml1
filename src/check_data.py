import pandas as pd

# Load AAPL stock data
data = pd.read_csv("data/AAPL.csv")

print("\n===== FIRST 5 ROWS =====")
print(data.head())

print("\n===== COLUMNS =====")
print(data.columns.tolist())

print("\n===== DATASET SIZE =====")
print(data.shape)

print("\n===== MISSING VALUES =====")
print(data.isnull().sum())
