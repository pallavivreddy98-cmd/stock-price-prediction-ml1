# 📈 Stock Price Prediction & Analysis Using Machine Learning

## 📌 Project Overview

This project is a machine learning-based web application that analyzes historical stock market data and predicts the next day's stock price.

The project uses historical Apple Inc. (AAPL) stock data and a Random Forest Regression model.

The predicted return is converted into an estimated next-day closing price. The results are displayed through an interactive Streamlit dashboard.

## 🎯 Objectives

- Analyze historical stock prices
- Perform data preprocessing
- Calculate moving averages
- Identify stock price trends
- Train a machine learning model
- Predict the next day's stock return
- Estimate the next day's closing price
- Compare actual and predicted results
- Display results through a web dashboard

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Random Forest Regression
- Joblib
- Streamlit

## 📊 Dataset

The project uses historical AAPL stock market data containing:

- Date
- Open
- High
- Low
- Close
- Volume

## 🤖 Machine Learning Model

### Random Forest Regression

A Random Forest Regression model is used to predict the **next-day stock return**.

The model uses the following features:

- Daily Return
- Previous Return
- 7-Day Moving Average Ratio
- 21-Day Moving Average Ratio
- High-Low Range
- Open-Close Range
- Volume Change

The predicted return is then converted into an estimated next-day closing price.

## 📈 Data Analysis

The project includes:

- Historical Stock Price Analysis
- 7-Day Moving Average
- 21-Day Moving Average
- Actual vs Predicted Return Analysis
- Stock price trend visualization

## 📊 Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The evaluation is performed using a chronological train-test split.

## 🌐 Streamlit Dashboard

The dashboard displays:

- Latest closing price
- Predicted next closing price
- Expected price change
- UP/DOWN model prediction
- Historical stock price chart
- Moving average analysis
- Model information
- Features used by the model
- Recent stock data

## 📁 Project Structure

```text
stock_price_prediction/
│
├── data/
│   ├── AAPL.csv
│   └── AAPL_cleaned.csv
│
├── src/
│   ├── check_data.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   └── stock_price_model.pkl
│
├── visualization/
│   ├── aapl_price_trend.png
│   ├── aapl_moving_averages.png
│   └── aapl_actual_vs_predicted.png
│
├── app/
│   └── app.py
│
├── notebooks/
├── requirements.txt
└── README.md
