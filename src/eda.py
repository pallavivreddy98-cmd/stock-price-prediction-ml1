import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned data
data = pd.read_csv("data/AAPL_cleaned.csv")

# Convert Date to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Create figure
plt.figure(figsize=(12, 6))

# Plot closing price
plt.plot(data["Date"], data["Close"], linewidth=1.5)

# Add title and labels
plt.title("Apple (AAPL) Stock Price Trend", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Closing Price ($)")

# Add grid
plt.grid(True, alpha=0.3)

# Improve layout
plt.tight_layout()

# Save chart
os.makedirs("visualization", exist_ok=True)
plt.savefig("visualization/aapl_price_trend.png", dpi=300)

# Show chart
plt.show()
# Calculate moving averages
data["MA7"] = data["Close"].rolling(window=7).mean()
data["MA21"] = data["Close"].rolling(window=21).mean()

# Create moving average chart
plt.figure(figsize=(12, 6))

plt.plot(data["Date"], data["Close"], label="Actual Close Price")
plt.plot(data["Date"], data["MA7"], label="7-Day Moving Average")
plt.plot(data["Date"], data["MA21"], label="21-Day Moving Average")

plt.title("AAPL Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price ($)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save chart
plt.savefig("visualization/aapl_moving_averages.png", dpi=300)

plt.show()
