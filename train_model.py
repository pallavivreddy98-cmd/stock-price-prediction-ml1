import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv("data/AAPL_cleaned.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)


# ==========================================
# 2. CREATE RETURN-BASED FEATURES
# ==========================================

# Today's return
data["Daily_Return"] = data["Close"].pct_change()

# Previous day's return
data["Previous_Return"] = data["Daily_Return"].shift(1)

# Moving averages
data["MA7"] = data["Close"].rolling(7).mean()
data["MA21"] = data["Close"].rolling(21).mean()

# Moving average ratios
data["MA7_Ratio"] = data["Close"] / data["MA7"]
data["MA21_Ratio"] = data["Close"] / data["MA21"]

# Intraday price movement
data["High_Low_Range"] = (
    data["High"] - data["Low"]
) / data["Close"]

data["Open_Close_Range"] = (
    data["Close"] - data["Open"]
) / data["Open"]

# Volume change
data["Volume_Change"] = data["Volume"].pct_change()

# TARGET:
# Tomorrow's percentage return
data["Tomorrow_Return"] = data["Close"].shift(-1) / data["Close"] - 1


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "Daily_Return",
    "Previous_Return",
    "MA7_Ratio",
    "MA21_Ratio",
    "High_Low_Range",
    "Open_Close_Range",
    "Volume_Change"
]

data = data.dropna().reset_index(drop=True)

X = data[features]
y = data["Tomorrow_Return"]


# ==========================================
# 4. TIME-BASED TRAIN/TEST SPLIT
# ==========================================

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# ==========================================
# 5. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ==========================================
# 6. PREDICT RETURNS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n==========================================")
print("     AAPL STOCK RETURN PREDICTION MODEL")
print("==========================================")

print("Model              : Random Forest Regression")
print(f"Training Records   : {len(X_train)}")
print(f"Testing Records    : {len(X_test)}")
print(f"Number of Features : {len(features)}")

print("\n----- MODEL PERFORMANCE -----")
print(f"MAE                : {mae:.6f}")
print(f"RMSE               : {rmse:.6f}")
print(f"R2 Score           : {r2:.4f}")


# ==========================================
# 8. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/stock_price_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/stock_price_model.pkl")


# ==========================================
# 9. ACTUAL VS PREDICTED RETURN GRAPH
# ==========================================

os.makedirs("visualization", exist_ok=True)

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values,
    label="Actual Return"
)

plt.plot(
    y_pred,
    label="Predicted Return"
)

plt.title("AAPL Actual vs Predicted Daily Return")

plt.xlabel("Test Data Points")

plt.ylabel("Daily Return")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "visualization/aapl_actual_vs_predicted.png",
    dpi=300
)

plt.close()

print("Actual vs Predicted graph saved successfully!")

print("\n==========================================")
print("             TRAINING COMPLETE")
print("==========================================")