#!/usr/bin/env python3
#
# chronos.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import json
from chronos import ChronosPipeline
import requests
from datetime import datetime, timedelta

api_key = "EVUGT1xHMbAy753xwjj9ROdF8WSt3Wpg"

# Example function to fetch stock data from Polygon.io
def fetch_stock_data(symbol, from_date, to_date):
    """
    Fetch stock data using Polygon REST API with caching.

    :param symbol: Stock ticker symbol (e.g., "AAPL")
    :param from_date: Start date for data retrieval in 'YYYY-MM-DD' format
    :param to_date: End date for data retrieval in 'YYYY-MM-DD' format
    :return: DataFrame containing the stock data
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if 'results' in data:
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            return df
        else:
            print("No data in the response. Please check the symbol and date range.")
            return pd.DataFrame()
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return pd.DataFrame()

# Set the date range
to_date = datetime.now().strftime('%Y-%m-%d')
from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

# Fetch stock data
symbol = "AAPL"
df = fetch_stock_data(symbol, from_date, to_date)

if not df.empty:
    # Prepare the data for the Chronos model
    context = torch.tensor(df['close'].values)  # Using the 'close' price for prediction
    prediction_length = 12

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]

    # Generate the forecast data
    forecast_index = pd.date_range(df['timestamp'].iloc[-1], periods=prediction_length + 1, freq='D')[1:]
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    # Create a dictionary to hold the forecast data
    forecast_data = {
        "historical_data": df["close"].tolist(),
        "timestamps": df["timestamp"].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        "forecast": [
            {
                "timestamp": str(forecast_index[i].date()),
                "low": float(low_val),
                "median": float(median_val),
                "high": float(high_val)
            }
            for i, (low_val, median_val, high_val) in enumerate(zip(low, median, high))
        ]
    }

    # Convert the dictionary to a pretty-printed JSON object and print it
    forecast_json = json.dumps(forecast_data, indent=4)
    print(forecast_json)

    # Prepare data for plotting
    last_hour = df[df['timestamp'] >= (datetime.now() - timedelta(hours=1))]

    # Create a two-subplot figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # Top plot: Last hour of data
    axs[0].plot(last_hour['timestamp'], last_hour['close'], color="royalblue", label="Last hour data")
    axs[0].set_title("Last Hour Stock Data (1-minute intervals)")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Stock Price")
    axs[0].legend()
    axs[0].grid()

    # Bottom plot: Full historical data and forecast
    axs[1].plot(df['timestamp'], df["close"], color="royalblue", label="historical data")
    axs[1].plot(forecast_index, median, color="tomato", label="median forecast")
    axs[1].fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
    axs[1].set_title("Full Historical Stock Data with Forecast")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Stock Price")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()
else:
    print("No data to display or forecast.")

