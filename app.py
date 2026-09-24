import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)


# ==========================================
# LOAD DATA AND MODEL
# ==========================================

data = pd.read_csv("data/AAPL_cleaned.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

model = joblib.load("models/stock_price_model.pkl")


# ==========================================
# CREATE FEATURES
# ==========================================

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


# ==========================================
# LATEST DATA
# ==========================================

latest = data.dropna().iloc[-1]


# ==========================================
# PREDICTION
# ==========================================

X_new = pd.DataFrame({
    "Daily_Return": [latest["Daily_Return"]],
    "Previous_Return": [latest["Previous_Return"]],
    "MA7_Ratio": [latest["MA7_Ratio"]],
    "MA21_Ratio": [latest["MA21_Ratio"]],
    "High_Low_Range": [latest["High_Low_Range"]],
    "Open_Close_Range": [latest["Open_Close_Range"]],
    "Volume_Change": [latest["Volume_Change"]]
})

predicted_return = model.predict(X_new)[0]

latest_close = latest["Close"]

predicted_price = latest_close * (1 + predicted_return)

change = predicted_price - latest_close

percentage = predicted_return * 100


# ==========================================
# TITLE
# ==========================================

st.title("📈 Stock Price Prediction & Analysis")

st.subheader("AAPL — Apple Inc.")

st.markdown(
    "Machine Learning based stock price analysis and "
    "next-day stock price prediction."
)


# ==========================================
# MAIN METRICS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Latest Close Price",
        f"${latest_close:.2f}"
    )

with col2:
    st.metric(
        "Predicted Next Close",
        f"${predicted_price:.2f}"
    )

with col3:
    st.metric(
        "Expected Change",
        f"${change:.2f}",
        f"{percentage:.2f}%"
    )


# ==========================================
# PREDICTION STATUS
# ==========================================

if predicted_price > latest_close:

    st.success(
        f"📈 Model Prediction: UP — "
        f"Expected next close: ${predicted_price:.2f}"
    )

else:

    st.warning(
        f"📉 Model Prediction: DOWN — "
        f"Expected next close: ${predicted_price:.2f}"
    )


# ==========================================
# HISTORICAL STOCK PRICE
# ==========================================

st.subheader("📊 Historical Stock Price")

st.line_chart(
    data.set_index("Date")["Close"]
)


# ==========================================
# MOVING AVERAGE ANALYSIS
# ==========================================

st.subheader("📈 Moving Average Analysis")

ma_data = data.set_index("Date")[
    ["Close", "MA7", "MA21"]
].dropna()

st.line_chart(ma_data)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Random Forest Regression"
    )

with col2:
    st.metric(
        "Input Features",
        "7"
    )

with col3:
    st.metric(
        "Prediction Target",
        "Next-Day Return"
    )


# ==========================================
# FEATURE INFORMATION
# ==========================================

st.subheader("🔍 Features Used by the Model")

feature_names = [
    "Daily Return",
    "Previous Return",
    "7-Day Moving Average Ratio",
    "21-Day Moving Average Ratio",
    "High-Low Range",
    "Open-Close Range",
    "Volume Change"
]

st.write(", ".join(feature_names))


# ==========================================
# RECENT DATA
# ==========================================

st.subheader("📋 Recent Stock Data")

display_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

st.dataframe(
    data[display_columns].tail(10),
    use_container_width=True
)


# ==========================================
# DISCLAIMER
# ==========================================

st.info(
    "⚠️ This prediction is for educational and project "
    "purposes only. It should not be considered financial advice."
)