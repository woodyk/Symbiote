#!/usr/bin/env python3
#
# PredictTimeseries.py

import pandas as pd
import numpy as np
import datetime
import torch
import plotly.graph_objs as go
import plotly.io as pio
from chronos import ChronosPipeline
import json
import warnings
import os
import contextlib

# Suppress specific warnings from the huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Suppress all stderr output (e.g., from libraries like huggingface_hub)
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stderr(fnull):
            yield

class TimeSeriesModel:
    def __init__(self):
        pass

    def generate_synthetic_time_series(self, resolution: str, count: int, start_value: float = 100.0, trend: float = 0.0, noise_std: float = 1.0) -> pd.DataFrame:
        """
        Generates synthetic time series data with a specified resolution and count.

        Parameters:
        - resolution (str): The time interval between data points (e.g., '20S' for 20 seconds, '1min' for 1 minute, '1D' for 1 day).
        - count (int): The number of data points to generate.
        - start_value (float): The starting value for the series.
        - trend (float): The amount by which the value increases or decreases per data point on average.
        - noise_std (float): The standard deviation of the random noise added to the values.

        Returns:
        - DataFrame: A DataFrame containing the date and value columns, or None if an error occurs.
        """
        try:
            # Generate date range with the specified resolution and count
            start_date = datetime.datetime.now()
            date_range = pd.date_range(start=start_date, periods=count, freq=resolution)

            # Generate synthetic values
            noise = np.random.normal(0, noise_std, count)  # Random noise for the values
            trend_values = trend * np.arange(count)  # Trend component
            series_values = start_value + trend_values + noise  # Combine the start value, trend, and noise

            # Create a DataFrame
            data = pd.DataFrame({'date': date_range, 'value': series_values})

            return data
        except Exception as e:
            print(f"Error generating synthetic time series: {e}")
            return None

    def predict_timeseries(self, dataset, prediction_length: int = 64, plot: bool = True) -> dict:
        """
        Simplified function to predict time series data using the ChronosPipeline.

        Parameters:
        - dataset (pd.DataFrame, str): Input dataset, which can be a pandas DataFrame, a URL, or a file path.
        - prediction_length (int): Number of time steps to predict into the future. Default is 64. Max allowed is 64.
        - plot (bool): Whether to render the plot. Default is True.

        Returns:
        - dict: A dictionary containing the original time series data and the predicted time series data, or None if an error occurs.
        """
        try:
            # Check and limit prediction length
            if prediction_length > 64:
                print("Warning: prediction_length cannot be greater than 64. Setting prediction_length to 64.")
                prediction_length = 64

            # Load the dataset
            if isinstance(dataset, pd.DataFrame):
                df = dataset
            elif isinstance(dataset, str):
                if dataset.startswith("http://") or dataset.startswith("https://"):
                    df = pd.read_csv(dataset)
                else:
                    df = pd.read_csv(dataset)
            else:
                raise ValueError("Input should be a pandas DataFrame, a file path, or a URL.")

            # Validate and process the data
            if df.shape[1] > 2:
                raise ValueError(f"Input data has {df.shape[1]} columns, but only 1 or 2 columns are allowed. Columns found: {df.columns.tolist()}")

            # Check if the first column is already in Unix timestamp format
            if not pd.api.types.is_integer_dtype(df.iloc[:, 0]):
                # If not, convert it to Unix timestamp
                try:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]).astype('int64') // 10**9  # Convert to Unix timestamp in seconds
                except (ValueError, TypeError):
                    pass  # If parsing fails, treat it as a regular x-axis column

            x_column = df.iloc[:, 0]
            y_column = df.iloc[:, 1]
            context = torch.tensor(y_column.values, dtype=torch.float32)

            # Initialize the Chronos pipeline and make predictions within the suppressed context
            with suppress_stderr():
                pipeline = ChronosPipeline.from_pretrained(
                    pretrained_model_name_or_path="amazon/chronos-t5-base",
                    device_map="cpu",
                    torch_dtype=torch.bfloat16,
                )
                forecast = pipeline.predict(context, prediction_length)

            # Prepare forecast data for return
            if pd.api.types.is_datetime64_any_dtype(pd.to_datetime(x_column, unit='s', errors='coerce')):
                freq = pd.infer_freq(pd.to_datetime(x_column, unit='s'))
                forecast_index = pd.date_range(start=pd.to_datetime(x_column.iloc[-1], unit='s'), periods=prediction_length + 1, freq=freq)[1:]
                forecast_index = [int(ts.timestamp()) for ts in forecast_index]  # Convert to Unix timestamp
            else:
                forecast_index = list(range(len(df), len(df) + prediction_length))

            low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

            # Convert DataFrame columns to Unix timestamp for JSON serialization
            original_data = {int(ts): val for ts, val in zip(x_column, y_column)}
            forecast_data = {int(ts): val for ts, val in zip(forecast_index, median)}

            # Plot the forecast only if plot is True
            if plot:
                pio.templates.default = "plotly_dark"  # Set dark mode
                fig = go.Figure()

                # Convert Unix timestamps to datetime for correct labeling
                x_column_datetime = pd.to_datetime(x_column, unit='s')
                forecast_index_datetime = pd.to_datetime(forecast_index, unit='s')

                # Plot original data in blue
                fig.add_trace(go.Scatter(x=x_column_datetime, y=y_column, mode='lines', name='Original Data', line=dict(color='blue')))
                
                # Plot predicted data in red, connected to original data
                fig.add_trace(go.Scatter(x=forecast_index_datetime, y=median, mode='lines', name='Predicted Data', line=dict(color='red')))
                
                # Plot the low and high range around predictions
                fig.add_trace(go.Scatter(x=forecast_index_datetime, y=low, fill=None, mode='lines', line=dict(color='red', width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=forecast_index_datetime, y=high, fill='tonexty', mode='lines', line=dict(color='red', width=0), showlegend=False, opacity=0.3))
                
                fig.update_layout(
                    title='Forecast Visualization',
                    xaxis_title='Time',
                    yaxis_title='Values',
                    xaxis=dict(type='date')  # Ensure x-axis is treated as dates
                )
                fig.show()

            # Create the output dictionary
            result = {
                "original_data": original_data,
                "predicted_data": forecast_data
            }

            return result

        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

# Example usage
if __name__ == "__main__":
    ts_model = TimeSeriesModel()

    # Generate synthetic data
    dataset = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    #dataset = ts_model.generate_synthetic_time_series(resolution='1min', count=1440, start_value=100.0, trend=0.05, noise_std=2.0)

    # Predict and visualize the time series, and get the result as a dictionary
    if dataset is not None:
        result = ts_model.predict_timeseries(dataset=dataset, prediction_length=64)  # You can set plot=False to skip plotting
        if result is not None:
            result_json = json.dumps(result, indent=4)
            print(result_json)

