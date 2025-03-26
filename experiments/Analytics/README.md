## Data Analytics Library Readme

#### Introduction

This library offers various classes to facilitate data analysis, enabling you to clean, preprocess, analyze, visualize, and interpret your data efficiently. There are eight classes currently available:

1. **DataManipulation**: Handle basic data manipulation tasks.
2. **TimeSeriesAnalysis**: Perform time series analysis and forecasting.
3. **AnomalyDetection**: Discover anomalies and outliers in your data.
4. **DescriptiveStats**: Obtain descriptive statistics and summary statistics for better insight.
5. **MachineLearning**: Apply machine learning algorithms for supervised and unsupervised learning.
6. **DataVisualization**: Visualize your data in informative ways.
7. **DiagnosticAnalytics**: Perform diagnostic analytics, including Chi-Square tests and regression diagnostics.
8. **PrescriptiveAnalytics**: Carry out prescriptive analytics, such as Optimal Experimental Design and Simulation Optimization.

#### Class Details

##### 1. DataManipulation

Perform basic data manipulation tasks, such as merging, joining, filtering, aggregating, and melting dataframes.

###### Methods

| Method              | Description                          |
|---------------------|--------------------------------------|
| merge_datasets      | Merge two datasets based on keys       |
| join_datasets       | Join datasets horizontally           |
| concatenate_datasets| Combine datasets vertically           |
| drop_duplicate_rows | Remove duplicate rows                 |
| fill_na_with_constant | Fill missing values with a constant |
| aggregate_by_group | Aggregate data based on groups        |
| filter_data         | Filter data based on conditions       |
| replace_values      | Replace specific values              |

##### 2. TimeSeriesAnalysis

Perform time series analysis, including stationarity checks, smoothing, differencing, and seasonality assessment. Also, conduct ARIMA and ETS modeling for forecasting.

###### Methods

| Method             | Description                                       |
|--------------------|---------------------------------------------------|
| adfuller_test      | Run Augmented Dickey–Fuller Test for stationarity  |
| kpss_test         | Run Kwiatkowski–Phillips–Schmidt–Shin Test for stationarity |
| moving_avg_smooth  | Smoothen data using moving averages                 |
| exponentially_smooth | Smoothen data using exponential smoothing       |
| diff               | Difference data                                   |
| seasonal_decompose| Decompose time series data                         |
| arima_model        | Build ARIMA model and forecast                      |
| ets_model         | Build ETS model and forecast                       |

##### 3. AnomalyDetection

Discover anomalies and outliers in your data using Local Outlier Factor, Histogram-Based Outlier Score, Angle-Based Outlier Detection, and Longitudinal Outlier Factor.

###### Methods

| Method            | Description                                       |
|-------------------|---------------------------------------------------|
| local_outlier_factor | Detect outliers using Local Outlier Factor        |
| hbos              | Detect outliers using Histogram-Based Outlier Score|
| abod              | Detect outliers using Angle-Based Outlier Detection|
| longitudinal_outlier_factor | Detect longitudinal outliers using LoF |

##### 4. DescriptiveStats

Obtain descriptive statistics and summary statistics for better insight.

###### Methods

| Method           | Description                              |
|------------------|------------------------------------------|
| measure_center   | Measures of Central Tendency             |
| measure_spread   | Measures of Dispersion                   |
| measure_shape    | Measures of Shape                        |
| quantiles        | Quantiles                                 |
| correlation_matrix | Display correlation matrix for visual inspection |
| contingency_table | Display contingency tables for categorical variables |

##### 5. MachineLearning

Apply machine learning algorithms for supervised and unsupervised learning. Train, validate, and test algorithms using various scoring strategies.

###### Methods

| Method              | Description                                       |
|---------------------|---------------------------------------------------|
| train_model        | Train a machine learning model                    |
| validation_curve    | Plot validation curve for a model                   |
| score_model         | Score a machine learning model using various metrics|
| confusion_matrix    | Get Confusion Matrix for a model                   |
| roc_auc             | ROC Curve and Area Under the Curve                |
| silhouette_score   | Silhouette Coefficient for measuring cluster cohesion and separation|

##### 6. DataVisualization

Visualize your data in informative ways using Pandas Profiling, Heatmaps, Scatterplots, and more.

###### Methods

| Method          | Description                              |
|-----------------|------------------------------------------|
| profile_report  | Generate interactive HTML profiling report |
| heatmap         | Show heatmap for correlation matrix         |
| scatterplot     | Scatterplot for selected columns           |
| piechart        | Draw piechart for categorical data         |
| jointplot       | Jointplot for paired data                 |
| swarmplot      | Swarmplot for categorical data             |

##### 7. DiagnosticAnalytics

Perform diagnostic analytics, including Chi-Square tests, ANOVA, and regression diagnostics.

###### Methods

| Method               | Description                                       |
|----------------------|---------------------------------------------------|
| chi_square_test      | Run Chi-Square Test for independence                |
| oneway_anova         | Run One-Way ANOVA                                 |
| regression_diagnostics | Perform regression diagnostics                  |

##### 8. PrescriptiveAnalytics

Carry out prescriptive analytics, such as Optimal Experimental Design and Simulation Optimization.

###### Methods

| Method                 | Description                                       |
|------------------------|---------------------------------------------------|
| optimal_experimental_design | Plan experiments using Optimal Design          |
| simulation_optimization  | Optimize simulations using Bayesian Optimization  |
| decision_tree_analysis  | Construct decision trees for classification or regression|
| mixed_integer_programming | Solve problems involving discrete decisions with linear algebra |

### Example Usages

#### Example 1: Data Manipulation and Visualization

```python
import pandas as pd
from data_analytics_library import DataManipulation, DataVisualization

# Sample data
data = {'Category': ['A', 'B', 'A', 'A', 'B', 'B'],
        'Value': [1, 2, 3, 4, 5, 6]}

df = pd.DataFrame(data)

# Instantiate DataManipulation and DataVisualization
dm = DataManipulation(df)
dv = DataVisualization(df)

# Merge datasets
merged_df = dm.merge_datasets(df, df, ('Category'))

# Filter data
filtered_df = dm.filter_data(df, lambda row: row['Value'] > 3)

# Visualize data
dv.profile_report(df)
dv.scatterplot(df, 'Category', 'Value')
Example 2: Time Series Analysis and Forecasting
import pandas as pd
from data_analytics_library import TimeSeriesAnalysis

# Sample time series data
data = {'Date': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01'],
        'Value': [1, 2, 3, 4, 5, 6]}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Instantiate TimeSeriesAnalysis
tsa = TimeSeriesAnalysis(df)

# Stationarity check
stationarity = tsa.adfuller_test(df)

# Seasonality assessment
seasonality = tsa.seasonal_decompose(df)

# Model and forecast
arima_model = tsa.arima_model(df)
forecast = arima_model.predict(start=len(df), end=len(df)+3, typ='levels')
Example 3: Anomaly Detection
import pandas as pd
from data_analytics_library import AnomalyDetection

# Sample data
data = {'SensorID': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'Reading': [10, 11, 12, 100, 101, 102, 20, 21, 22]}
df = pd.DataFrame(data)

# Instantiate AnomalyDetection
ada = AnomalyDetection(df)

# Detect anomalies using Local Outlier Factor
local_outlier_factor = ada.local_outlier_factor(df, 'SensorID', 'Reading')

# Detect anomalies using Histogram-Based Outlier Score
hist_outlier_score = ada.histogram_based_outlier_score(df, 'SensorID', 'Reading')

# Detect anomalies using Angle-Based Outlier Detection
angle_based_outlier_score = ada.angle_based_outlier_detect(df, 'SensorID', 'Reading')
Remember to adjust the example usages to fit your data and desired goals. Happy analyzing!

