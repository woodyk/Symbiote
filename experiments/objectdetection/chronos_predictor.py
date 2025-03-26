#!/usr/bin/env python3
#
# predictions.py

import pandas as pd
import numpy as np
import datetime
from io import StringIO
import torch
import plotly.graph_objs as go
from chronos import ChronosPipeline

class DataHandler:
    def load_data(self, source) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            df = source
        elif isinstance(source, str):
            if source.startswith("http://") or source.startswith("https://"):
                df = pd.read_csv(source)
            else:
                df = pd.read_csv(source)
        else:
            raise ValueError("Input should be a pandas DataFrame, a file path, or a URL.")
        return df

    def validate_data(self, df: pd.DataFrame) -> None:
        if df.shape[1] > 2:
            raise ValueError(f"Input data has {df.shape[1]} columns, but only 1 or 2 columns are allowed. Columns found: {df.columns.tolist()}")

    def process_data(self, df: pd.DataFrame) -> torch.Tensor:
        self.validate_data(df)
        if df.shape[1] == 1:
            context = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32)
        else:
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            context = torch.tensor(y, dtype=torch.float32)
        return context

class ChronosPredictor:
    def __init__(self, model_name: str = "amazon/chronos-t5-tiny", device: str = "cpu"):
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

    def predict(self, context: torch.Tensor, prediction_length: int) -> torch.Tensor:
        forecast = self.pipeline.predict(context, prediction_length)
        return forecast

class PlotlyVisualizer:
    def plot_forecast(self, df: pd.DataFrame, forecast: np.ndarray, prediction_length: int) -> None:
        # Detect if the x-axis is datetime or integer
        x_column = df.iloc[:, 0]
        if pd.api.types.is_datetime64_any_dtype(x_column):
            forecast_index = pd.date_range(start=x_column.iloc[-1], periods=prediction_length + 1, freq=pd.infer_freq(x_column))[1:]
        else:
            forecast_index = list(range(len(df), len(df) + prediction_length))

        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x_column, y=df.iloc[:, 1], mode='lines', name='Historical Data', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=forecast_index, y=median, mode='lines', name='Median Forecast', line=dict(color='tomato')))
        fig.add_trace(go.Scatter(x=forecast_index, y=low, fill=None, mode='lines', line=dict(color='tomato', width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_index, y=high, fill='tonexty', mode='lines', line=dict(color='tomato', width=0), showlegend=False, opacity=0.3))

        fig.update_layout(title='Forecast Visualization', xaxis_title='Time' if pd.api.types.is_datetime64_any_dtype(x_column) else 'Index', yaxis_title='Values')
        fig.show()

class ErrorHandler:
    def raise_error(self, error_message: str, details: dict) -> None:
        detailed_message = f"{error_message}. Details: {details}"
        raise ValueError(detailed_message)

class ForecastingConfig:
    def __init__(self, prediction_length: int):
        self.prediction_length = prediction_length

    def get_config(self) -> dict:
        return {"prediction_length": self.prediction_length}

def generate_synthetic_time_series(resolution: str, count: int, start_value: float = 100.0, trend: float = 0.0, noise_std: float = 1.0):
    """
    Generates synthetic time series data with a specified resolution and count.

    Parameters:
    - resolution (str): The time interval between data points (e.g., '20S' for 20 seconds, '1T' for 1 minute, '1D' for 1 day).
    - count (int): The number of data points to generate.
    - start_value (float): The starting value for the series.
    - trend (float): The amount by which the value increases or decreases per data point on average.
    - noise_std (float): The standard deviation of the random noise added to the values.

    Returns:
    - DataFrame: A DataFrame containing the date and value columns.
    """
    # Generate date range with the specified resolution and count
    start_date = datetime.datetime.now()
    date_range = pd.date_range(start=start_date, periods=count, freq=resolution)

    # Generate synthetic values
    noise = np.random.normal(0, noise_std, count)  # Random noise for the values
    trend_values = trend * np.arange(count)  # Trend component
    series_values = start_value + trend_values + noise  # Combine the start value, trend, and noise

    # Create a DataFrame
    data = pd.DataFrame({'date': date_range, 'value': series_values})
    print(data)

    return data

if __name__ == "__main__":
    dataset = generate_synthetic_time_series(resolution='1T', count=1440, start_value = 100.0, trend=0.05, noise_std=2.0)

    #dataset = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    # Step 1: Load Data
    data_handler = DataHandler()
    df = data_handler.load_data(dataset)

    # Step 2: Process Data
    processed_data = data_handler.process_data(df)

    # Step 3: Configure Forecasting
    config = ForecastingConfig(prediction_length=64)

    # Step 4: Make Predictions
    predictor = ChronosPredictor(model_name="amazon/chronos-t5-tiny")
    forecast = predictor.predict(context=processed_data, prediction_length=config.get_config()["prediction_length"])

    # Step 5: Visualize Results
    visualizer = PlotlyVisualizer()
    visualizer.plot_forecast(df, forecast, config.get_config()["prediction_length"])

