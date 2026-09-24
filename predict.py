import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/stock_price_model.pkl")

# Load data
data = pd.read_csv("data/AAPL_cleaned.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

# Create features
data["Daily_Return"] = data["Close"].pct_change()
data["Previous_Return"] = data["Daily_Return"].shift(1)

data["MA7"] = data["Close"].rolling(7).mean()
data["MA21"] = data["Close"].rolling(21).mean()

data["MA7_Ratio"] = data["Close"] / data["MA7"]
data["MA21_Ratio"] = data["Close"] / data["MA21"]

data["High_Low_Range"] = (
    data["High"] - data["Low"]
) / data["Close"]

data["Open_Close_Range"] = (
    data["Close"] - data["Open"]
) / data["Open"]

data["Volume_Change"] = data["Volume"].pct_change()

data = data.dropna().reset_index(drop=True)

# Latest row
latest = data.iloc[-1]

# Prepare features
X_new = pd.DataFrame({
    "Daily_Return": [latest["Daily_Return"]],
    "Previous_Return": [latest["Previous_Return"]],
    "MA7_Ratio": [latest["MA7_Ratio"]],
    "MA21_Ratio": [latest["MA21_Ratio"]],
    "High_Low_Range": [latest["High_Low_Range"]],
    "Open_Close_Range": [latest["Open_Close_Range"]],
    "Volume_Change": [latest["Volume_Change"]]
})

# Predict tomorrow's return
predicted_return = model.predict(X_new)[0]

# Convert predicted return to predicted price
latest_close = latest["Close"]

predicted_price = latest_close * (1 + predicted_return)

change = predicted_price - latest_close

percentage = predicted_return * 100

print("======================================")
print("       AAPL STOCK PRICE PREDICTION")
print("======================================")

print(f"Latest Date               : {latest['Date'].date()}")
print(f"Today's Close             : ${latest_close:.2f}")
print(f"Predicted Tomorrow Close  : ${predicted_price:.2f}")
print(f"Expected Change           : ${change:.2f}")
print(f"Expected Change %         : {percentage:.2f}%")

if predicted_price > latest_close:
    print("Prediction                : UP")
else:
    print("Prediction                : DOWN")

print("======================================")