#!/usr/bin/env python3
#
# anomaly.py

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope

class AnomalyDetection:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column

    def detect_z_score(self, threshold=3):
        mu = self.data[self.target_column].mean()
        sigma = self.data[self.target_column].std()
        filtered_data = self.data[np.abs((self.data[self.target_column] - mu) / sigma) > threshold]
        return filtered_data

    def detect_modified_z_score(self, threshold=3):
        mu = self.data[self.target_column].mean()
        sigma = self.data[self.target_column].std()
        mAD = np.mean(np.absolute(pd.Series(self.data[self.target_column]).diff()))
        filtered_data = self.data[(np.abs((self.data[self.target_column] - mu) / sigma) > threshold) | ((np.absolute(pd.Series(self.data[self.target_column]).diff()) > 2 * mAD))]
        return filtered_data

    def detect_elliptic_envelope(self, contamination=0.1):
        clf = EllipticEnvelope(contamination=contamination)
        clf.fit(self.data[self.target_column].values.reshape(-1, 1))
        scores_pred = clf.decision_function(self.data[self.target_column].values.reshape(-1, 1))
        filtered_data = self.data[scores_pred < -clf.threshold]
        return filtered_data

