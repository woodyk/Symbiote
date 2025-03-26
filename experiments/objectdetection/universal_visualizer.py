#!/usr/bin/env python3
#
# universal_visualizer.py

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import requests
from io import StringIO
from plotly.subplots import make_subplots

# Set plotly to dark mode
pio.templates.default = "plotly_dark"

def load_csv(source):
    """Load CSV from a file path or URL."""
    if source.startswith('http://') or source.startswith('https://'):
        response = requests.get(source)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            return pd.read_csv(csv_data)
        else:
            raise ValueError("Unable to fetch data from URL.")
    else:
        return pd.read_csv(source)

def generate_graphs(df):
    """Generate all the required graph types and return a Plotly figure."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Pie Chart"),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "pie"}, None]]
    )

    # Line Chart
    if df.select_dtypes(include=['number']).shape[1] >= 2:
        fig.add_trace(go.Scatter(x=df[df.columns[0]], y=df[df.columns[1]], mode='lines', name='Line Chart'), row=1, col=1)

    # Bar Chart
    if df.select_dtypes(include=['number']).shape[1] >= 1:
        fig.add_trace(go.Bar(x=df[df.columns[0]], y=df[df.columns[1]], name='Bar Chart'), row=1, col=2)

    # Scatter Plot
    if df.select_dtypes(include=['number']).shape[1] >= 2:
        fig.add_trace(go.Scatter(x=df[df.columns[0]], y=df[df.columns[1]], mode='markers', name='Scatter Plot'), row=2, col=1)

    # Histogram
    if df.select_dtypes(include=['number']).shape[1] >= 1:
        fig.add_trace(go.Histogram(x=df[df.columns[0]], name='Histogram'), row=2, col=2)

    # Pie Chart
    if df.select_dtypes(include=['object', 'category']).shape[1] >= 1 and df.select_dtypes(include=['number']).shape[1] >= 1:
        fig.add_trace(go.Pie(labels=df[df.columns[0]], values=df[df.columns[1]], name='Pie Chart'), row=3, col=1)

    fig.update_layout(height=900, width=1200, title_text="Data Visualization Dashboard")
    return fig

def visualize_csv(source):
    """Main function to load CSV from a file or URL and visualize it."""
    try:
        df = load_csv(source)
        fig = generate_graphs(df)
        fig.show()
    except Exception as e:
        print(f"Error loading or processing the file: {e}")

if __name__ == "__main__":
    # Example usage: visualize_csv('your_file.csv') or visualize_csv('http://example.com/data.csv')
    source = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    visualize_csv(source)

