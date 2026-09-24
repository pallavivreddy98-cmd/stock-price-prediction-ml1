import pandas as pd

# Load dataset
data = pd.read_csv("data/AAPL.csv")

print("Original shape:", data.shape)

# Convert Date column
data["Date"] = pd.to_datetime(data["Date"])

# Sort by date
data = data.sort_values("Date")

# Remove duplicate rows
data = data.drop_duplicates()

# Handle missing values
data = data.dropna()

# Reset index
data = data.reset_index(drop=True)

print("\nAfter preprocessing:")
print("Shape:", data.shape)

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nFirst 5 rows:")
print(data.head())

# Save cleaned dataset
data.to_csv("data/AAPL_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
