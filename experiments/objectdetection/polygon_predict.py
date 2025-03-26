#!/usr/bin/env python3
#
# polygon_predict.py

import requests
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import datetime as dt
import time
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Configuration
api_key = "EVUGT1xHMbAy753xwjj9ROdF8WSt3Wpg"  # Replace with your actual Polygon API key
symbol = "AAPL"  # Stock ticker symbol
days_back = 30  # Number of days back to fetch data
future_minutes = 120  # Number of minutes into the future to predict
update_interval = 12000  # Update interval in milliseconds (12 seconds)
graph_window_minutes = 60  # Time window for the top graph in minutes

# Initialize global variables
df = pd.DataFrame()
forecast = pd.DataFrame()

def fetch_stock_data(symbol, from_date, to_date):
    """
    Fetch stock data using Polygon REST API.

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

def prepare_data_for_prophet(df):
    """
    Prepare data for the Prophet model.

    :param df: DataFrame containing historical stock data
    :return: DataFrame formatted for Prophet
    """
    prophet_df = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    return prophet_df

def predict_future_price_with_prophet(df, future_minutes):
    """
    Predict the future stock price using the Prophet model.

    :param df: DataFrame containing historical stock data
    :param future_minutes: Number of minutes into the future to predict
    :return: DataFrame with predictions
    """
    model = Prophet(interval_width=0.95)
    model.fit(df)

    future_dates = pd.date_range(start=df['ds'].iloc[-1], periods=future_minutes+1, freq='T')[1:]
    future = pd.DataFrame({'ds': future_dates})

    forecast = model.predict(future)
    return forecast

def create_dual_graphs(df, forecast, graph_window_minutes):
    """
    Create two graphs: one for short-term prediction and one for 24-hour historical view.

    :param df: DataFrame containing historical stock data
    :param forecast: DataFrame with Prophet predictions
    :param graph_window_minutes: Time window for the top graph in minutes
    :return: Plotly figure object
    """
    # Create subplot structure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1,
                        subplot_titles=("Short-term Prediction", "24-Hour Historical View"))

    # Filter data for the top graph (short-term prediction)
    graph_start_time = df['timestamp'].max() - pd.Timedelta(minutes=graph_window_minutes)
    df_short_term = df[df['timestamp'] >= graph_start_time]

    # Top graph: Short-term prediction
    fig.add_trace(go.Scatter(x=df_short_term['timestamp'], y=df_short_term['close'], mode='lines', name='Historical Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Prediction', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), fillcolor='rgba(255, 0, 0, 0.2)', fill='tonexty', showlegend=False), row=1, col=1)

    # Filter data for the bottom graph (24-hour historical view)
    historical_start_time = df['timestamp'].max() - pd.Timedelta(hours=24)
    df_historical = df[df['timestamp'] >= historical_start_time]

    # Bottom graph: 24-hour historical view
    fig.add_trace(go.Scatter(x=df_historical['timestamp'], y=df_historical['close'], mode='lines', name='24-Hour Historical Data', line=dict(color='blue')), row=2, col=1)

    # Update layout
    fig.update_layout(height=1000, title_text=f"Real-Time Stock Data for {symbol}")
    fig.update_xaxes(title_text="Timestamp", row=1, col=1)
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)

    return fig

# Initialize the Dash app
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(
        id='graph-update',
        interval=update_interval,
        n_intervals=0
    )
])

@app.callback(Output('live-graph', 'figure'),
              Input('graph-update', 'n_intervals'))
def update_graph(n):
    global df, forecast

    to_date = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    from_date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_back)).strftime('%Y-%m-%d')

    df = fetch_stock_data(symbol, from_date, to_date)

    if df.empty:
        print("No data fetched. Please check the symbol and date range.")
        return go.Figure()

    current_price = df['close'].iloc[-1]

    # Prepare the data for the Prophet model
    prophet_df = prepare_data_for_prophet(df)

    # Predict the future price using the Prophet model
    forecast = predict_future_price_with_prophet(prophet_df, future_minutes)

    predicted_price = forecast['yhat'].iloc[-1]
    future_timestamp = forecast['ds'].iloc[-1]

    # Print out the current symbol, current price, and prediction
    print(f"Symbol: {symbol}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Prophet Predicted Price in {future_minutes} minutes: ${predicted_price:.2f}")

    # Create the dual graphs
    fig = create_dual_graphs(df, forecast, graph_window_minutes)

    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
