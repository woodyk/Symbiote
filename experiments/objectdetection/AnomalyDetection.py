#!/usr/bin/env python3
#
# AnomalyDetection.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from statsmodels.tsa.arima.model import ARIMA

# Generate synthetic data
def generate_data(n_samples=200, n_outliers=20):
    np.random.seed(42)
    X = 0.3 * np.random.randn(n_samples, 2)
    X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))
    X = np.concatenate([X, X_outliers], axis=0)
    return X

# Anomaly Detection Functions (without plotting)

def gaussian_anomaly_detection(X):
    clf = EllipticEnvelope(support_fraction=1., contamination=0.1)
    clf.fit(X)
    y_pred = clf.predict(X)
    return X, y_pred

def elliptic_envelope_anomaly_detection(X):
    clf = EllipticEnvelope(contamination=0.1)
    clf.fit(X)
    y_pred = clf.predict(X)
    return X, y_pred

def knn_anomaly_detection(X):
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X)
    distances, _ = knn.kneighbors(X)
    anomaly_score = np.mean(distances, axis=1)
    threshold = np.percentile(anomaly_score, 90)
    y_pred = np.where(anomaly_score > threshold, -1, 1)
    return X, y_pred, anomaly_score

def local_outlier_factor_anomaly_detection(X):
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred = clf.fit_predict(X)
    return X, y_pred

def isolation_forest_anomaly_detection(X):
    clf = IsolationForest(contamination=0.1)
    clf.fit(X)
    y_pred = clf.predict(X)
    anomaly_score = clf.decision_function(X)
    return X, y_pred, anomaly_score

def kmeans_anomaly_detection(X):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    y_pred = np.ones(len(X))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    y_pred[closest] = -1
    return X, y_pred, kmeans.cluster_centers_

def dbscan_anomaly_detection(X):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    y_pred = dbscan.fit_predict(X)
    y_pred = np.where(y_pred == -1, -1, 1)
    return X, y_pred

def arima_anomaly_detection():
    np.random.seed(42)
    # Generate synthetic time series data
    n_samples = 200
    X = np.sin(np.linspace(0, 50, n_samples)) + np.random.normal(scale=0.5, size=n_samples)
    X[150:] += 3  # Inject an anomaly

    model = ARIMA(X, order=(5, 1, 0))
    model_fit = model.fit()
    residuals = model_fit.resid
    anomaly_score = np.abs(residuals)

    threshold = np.percentile(anomaly_score, 95)
    y_pred = np.where(anomaly_score > threshold, -1, 1)
    return X, y_pred

def random_cut_forest_anomaly_detection(X):
    from amazon_kclpy import rcf
    rcf = rcf.RandomCutForest(dimensions=X.shape[1])
    anomaly_scores = np.array([rcf.score(point) for point in X])
    threshold = np.percentile(anomaly_scores, 90)
    y_pred = np.where(anomaly_scores > threshold, -1, 1)
    return X, y_pred, anomaly_scores

# Graphing Functions

def scatter_plot(X, y_pred, title="Scatter Plot"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', label='Data points')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction')
    plt.show()

def time_series_plot(X, y_pred, title="Time Series Plot"):
    plt.figure(figsize=(10, 6))
    plt.plot(X, label='Time Series')
    plt.scatter(np.where(y_pred == -1), X[y_pred == -1], c='red', label='Anomalies')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def histogram(X, title="Histogram"):
    plt.figure(figsize=(8, 6))
    plt.hist(X, bins=30, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def box_plot(X, title="Box Plot"):
    plt.figure(figsize=(8, 6))
    plt.boxplot(X)
    plt.title(title)
    plt.ylabel('Value')
    plt.show()

def heatmap(matrix, title="Heatmap"):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()

def bar_chart(y_pred, title="Bar Chart"):
    counts = np.bincount(y_pred + 1)  # +1 to handle -1 labels
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(counts)), counts, color='green')
    plt.title(title)
    plt.xlabel('Anomaly/Normal')
    plt.ylabel('Count')
    plt.show()

def density_plot(X, title="Density Plot"):
    plt.figure(figsize=(8, 6))
    density = plt.hist(X, bins=30, density=True, alpha=0.5, color='g')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

def contour_plot(X, title="Contour Plot"):
    plt.figure(figsize=(8, 6))
    plt.tricontourf(X[:, 0], X[:, 1], np.arange(len(X)), cmap='RdBu')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Density')
    plt.show()

# Main function to run all examples and visualize using different graphing methods
def run_anomaly_detection_visualizations():
    X = generate_data()

    # Gaussian Example
    X_gaussian, y_pred_gaussian = gaussian_anomaly_detection(X)
    scatter_plot(X_gaussian, y_pred_gaussian, title="Gaussian - Scatter Plot")
    box_plot(X_gaussian[:, 0], title="Gaussian - Box Plot")
    histogram(X_gaussian[:, 0], title="Gaussian - Histogram")
    density_plot(X_gaussian[:, 0], title="Gaussian - Density Plot")

    # Elliptic Envelope Example
    X_elliptic, y_pred_elliptic = elliptic_envelope_anomaly_detection(X)
    scatter_plot(X_elliptic, y_pred_elliptic, title="Elliptic Envelope - Scatter Plot")
    box_plot(X_elliptic[:, 0], title="Elliptic Envelope - Box Plot")
    histogram(X_elliptic[:, 0], title="Elliptic Envelope - Histogram")
    density_plot(X_elliptic[:, 0], title="Elliptic Envelope - Density Plot")

    # K-Nearest Neighbors (KNN) Example
    X_knn, y_pred_knn, anomaly_score_knn = knn_anomaly_detection(X)
    scatter_plot(X_knn, y_pred_knn, title="KNN - Scatter Plot")
    bar_chart(y_pred_knn, title="KNN - Bar Chart")
    density_plot(anomaly_score_knn, title="KNN - Density Plot")

    # Local Outlier Factor (LOF) Example
    X_lof, y_pred_lof = local_outlier_factor_anomaly_detection(X)
    scatter_plot(X_lof, y_pred_lof, title="LOF - Scatter Plot")
    bar_chart(y_pred_lof, title="LOF - Bar Chart")

    # Isolation Forest Example
    X_iso, y_pred_iso, anomaly_score_iso = isolation_forest_anomaly_detection(X)
    scatter_plot(X_iso, y_pred_iso, title="Isolation Forest - Scatter Plot")
    bar_chart(y_pred_iso, title="Isolation Forest - Bar Chart")
    density_plot(anomaly_score_iso, title="Isolation Forest - Density Plot")

    # K-Means Clustering Example
    X_kmeans, y_pred_kmeans, centroids = kmeans_anomaly_detection(X)
    scatter_plot(X_kmeans, y_pred_kmeans, title="K-Means - Scatter Plot")
    contour_plot(X_kmeans, title="K-Means - Contour Plot")
    bar_chart(y_pred_kmeans, title="K-Means - Bar Chart")

    # DBSCAN Example
    X_dbscan, y_pred_dbscan = dbscan_anomaly_detection(X)
    scatter_plot(X_dbscan, y_pred_dbscan, title="DBSCAN - Scatter Plot")
    bar_chart(y_pred_dbscan, title="DBSCAN - Bar Chart")

    # ARIMA Example
    X_arima, y_pred_arima = arima_anomaly_detection()
    time_series_plot(X_arima, y_pred_arima, title="ARIMA - Time Series Plot")

    # Random Cut Forest (RCF) Example
    X_rcf, y_pred_rcf, anomaly_score_rcf = random_cut_forest_anomaly_detection(X)
    scatter_plot(X_rcf, y_pred_rcf, title="Random Cut Forest (RCF) - Scatter Plot")
    bar_chart(y_pred_rcf, title="Random Cut Forest (RCF) - Bar Chart")
    density_plot(anomaly_score_rcf, title="Random Cut Forest (RCF) - Density Plot")

if __name__ == "__main__":
    run_anomaly_detection_visualizations()

