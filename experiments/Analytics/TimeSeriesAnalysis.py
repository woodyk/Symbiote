#!/usr/bin/env python3
#
# timeseries.py

class TimeSeriesAnalysis:
    def __init__(self, data, date_column, target_column):
        self.data = data
        self.date_column = date_column
        self.target_column = target_column
        
    def plot_timeseries(self):
        ts_data = self.data[[self.date_column, self.target_column]]
        ts_data.set_index(self.date_column, inplace=True)
        ts_data[self.target_column].plot(figsize=(14, 7))

    def compute_rolling_statistics(self, statistic, window):
        ts_data = self.data[[self.date_column, self.target_column]]
        ts_data.set_index(self.date_column, inplace=True)
        rolled = ts_data[self.target_column].rolling(window=window)
        aggregated = rolled.agg([statistic])
        return aggregated

    def decompose_timeseries(self):
        ts_data = self.data[[self.date_column, self.target_column]]
        ts_data.set_index(self.date_column, inplace=True)
        decomposed = seasonal_decompose(ts_data[self.target_column], model='multiplicative')
        return decomposed

    def auto_arima(self):
        ts_data = self.data[[self.date_column, self.target_column]]
        ts_data.set_index(self.date_column, inplace=True)
        model = AutoArima(ts_data[self.target_column], trace=True, suppress_warnings=True)
        model_fit = model.fit()
        return model_fit
