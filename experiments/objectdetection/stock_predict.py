#!/usr/bin/env python3
#
# stock_predict.py

import requests
import json
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import datetime as dt
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import yfinance as yf
from functools import lru_cache

# Configuration
api_key = "EVUGT1xHMbAy753xwjj9ROdF8WSt3Wpg"  # Replace with your actual Polygon API key
default_symbol = "AAPL"  # Default stock ticker symbol
days_back = 30  # Number of days back to fetch data
future_minutes = 15  # Number of minutes into the future to predict
update_interval = 20000  # Update interval in milliseconds (60 seconds)
graph_window_minutes = 60  # Time window for the top graph in minutes

# Initialize global variables
df = pd.DataFrame()
forecast = pd.DataFrame()

@lru_cache(maxsize=32)
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

    # Generate future dates and make predictions
    future_dates = pd.date_range(start=df['ds'].iloc[-1], periods=future_minutes + 1, freq='min')
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)

    # Insert the last actual data point to ensure continuity
    last_actual = df.iloc[[-1]].copy()
    last_actual['yhat'] = last_actual['y']
    last_actual['yhat_lower'] = last_actual['y']
    last_actual['yhat_upper'] = last_actual['y']
    forecast = pd.concat([last_actual, forecast.iloc[1:]])

    return forecast

def create_dual_graphs(df, forecast, graph_window_minutes):
    """
    Create two graphs: one for short-term prediction and one for 24-hour historical view.

    :param df: DataFrame containing historical stock data
    :param forecast: DataFrame with Prophet predictions
    :param graph_window_minutes: Time window for the top graph in minutes
    :return: Plotly figure object
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1,
                        subplot_titles=("Short-term Prediction", "24-Hour Historical View"))

    # Ensure 1-minute resolution for the top graph
    graph_start_time = df['timestamp'].max() - pd.Timedelta(minutes=graph_window_minutes)
    df_short_term = df[df['timestamp'] >= graph_start_time].set_index('timestamp').resample('min').ffill().reset_index()

    fig.add_trace(go.Scatter(x=df_short_term['timestamp'], y=df_short_term['close'], mode='lines', name='Historical Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Prediction', line=dict(color='#00ff00')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), fillcolor='rgba(0, 255, 0, 0.2)', fill='tonexty', showlegend=False), row=1, col=1)

    # Historical data for the 24-hour view
    historical_start_time = df['timestamp'].max() - pd.Timedelta(hours=24)
    df_historical = df[df['timestamp'] >= historical_start_time]

    fig.add_trace(go.Scatter(x=df_historical['timestamp'], y=df_historical['close'], mode='lines', name='24-Hour Historical Data', line=dict(color='#00ffff')), row=2, col=1)

    fig.update_layout(
        height=800,
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        font=dict(color='#ffffff'),
    )
    fig.update_xaxes(title_text="Timestamp", row=1, col=1, gridcolor='#333333')
    fig.update_xaxes(title_text="Timestamp", row=2, col=1, gridcolor='#333333')
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor='#333333')
    fig.update_yaxes(title_text="Price", row=2, col=1, gridcolor='#333333')

    return fig

def get_stock_info(symbol):
    """
    Get additional stock information using yfinance.

    :param symbol: Stock ticker symbol
    :return: Dictionary containing stock information
    """
    stock = yf.Ticker(symbol)
    info = stock.info
    return {
        'name': info.get('longName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A'),
        'pe_ratio': info.get('trailingPE', 'N/A'),
        'dividend_yield': info.get('dividendYield', 'N/A'),
        '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
        '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
    }

# Initialize the Dash app with a dark theme
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Custom CSS for additional styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .card {
                background-color: #2c2c2c;
                margin-bottom: 10px;
                font-size: 0.9rem;
            }
            .card-header {
                background-color: #3c3c3c;
                font-size: 1rem;
            }
            .card-title {
                font-size: 1.1rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Input(id="stock-input", type="text", placeholder="Enter stock symbol", value=default_symbol, className="mb-2"),
        ], width=8),
        dbc.Col([
            dbc.Button("Update", id="submit-button", color="primary", className="mb-4"),
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='live-graph', animate=True),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id="stock-info"),
            html.Div(id="price-info"),
            html.Div(id="prediction-info"),
        ], width=12, className="d-flex justify-content-around mb-4"),
    ]),
    dcc.Interval(id='graph-update', interval=update_interval, n_intervals=0),
], fluid=True, style={'backgroundColor': '#1e1e1e', 'color': '#ffffff'})

@app.callback(
    [Output('live-graph', 'figure'),
     Output('stock-info', 'children'),
     Output('price-info', 'children'),
     Output('prediction-info', 'children')],
    [Input('graph-update', 'n_intervals'),
     Input('submit-button', 'n_clicks')],
    [State('stock-input', 'value')]
)
def update_graph(n, n_clicks, symbol):
    global df, forecast

    if not symbol:
        raise PreventUpdate

    to_date = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    from_date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_back)).strftime('%Y-%m-%d')

    df = fetch_stock_data(symbol, from_date, to_date)

    if df.empty:
        print("No data fetched. Please check the symbol and date range.")
        raise PreventUpdate

    current_price = df['close'].iloc[-1]
    previous_close = df['close'].iloc[-2]
    price_change = current_price - previous_close
    price_change_percent = (price_change / previous_close) * 100

    prophet_df = prepare_data_for_prophet(df)
    forecast = predict_future_price_with_prophet(prophet_df, future_minutes)

    predicted_price = forecast['yhat'].iloc[-1]
    prediction_change = predicted_price - current_price
    prediction_change_percent = (prediction_change / current_price) * 100

    fig = create_dual_graphs(df, forecast, graph_window_minutes)

    stock_info = get_stock_info(symbol)

    stock_info_component = dbc.Card([
        dbc.CardHeader(html.H4(stock_info['name'], className="card-title")),
        dbc.CardBody([
            html.P(f"Sector: {stock_info['sector']}", className="card-text"),
            html.P(f"Market Cap: ${stock_info['market_cap']:,}", className="card-text"),
            html.P(f"P/E Ratio: {stock_info['pe_ratio']:.2f}", className="card-text"),
            html.P(f"Dividend Yield: {stock_info['dividend_yield']:.2%}", className="card-text"),
            html.P(f"52 Week High: ${stock_info['52_week_high']:.2f}", className="card-text"),
            html.P(f"52 Week Low: ${stock_info['52_week_low']:.2f}", className="card-text"),
        ])
    ])

    price_info_component = dbc.Card([
        dbc.CardHeader(html.H4("Current Price Information")),
        dbc.CardBody([
            html.P(f"Current Price: ${current_price:.2f}", className="card-text"),
            html.P(f"Price Change: ${price_change:.2f} ({price_change_percent:.2f}%)",
                   className="card-text text-success" if price_change >= 0 else "card-text text-danger"),
            html.P(f"Day's High: ${df['high'].max():.2f}", className="card-text"),
            html.P(f"Day's Low: ${df['low'].min():.2f}", className="card-text"),
            html.P(f"Volume: {df['volume'].sum():,}", className="card-text"),
        ])
    ])

    prediction_info_component = dbc.Card([
        dbc.CardHeader(html.H4("Price Prediction")),
        dbc.CardBody([
            html.P(f"Predicted Price in {future_minutes} minutes: ${predicted_price:.2f}", className="card-text"),
            html.P(f"Predicted Change: ${prediction_change:.2f} ({prediction_change_percent:.2f}%)",
                   className="card-text text-success" if prediction_change >= 0 else "card-text text-danger"),
        ])
    ])

    return fig, stock_info_component, price_info_component, prediction_info_component

if __name__ == "__main__":
    app.run_server(debug=True)

