#!/usr/bin/env python3
#
# machinelearning.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class MachineLearning:
    def __init__(self, data, target_column, feature_columns):
        self.data = data
        self.target_column = target_column
        self.feature_columns = feature_columns

    def split_data(self, test_size=0.3, random_seed=42):
        X = self.data[self.feature_columns]
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        return X_train, X_test, y_train, y_test

    def train_linear_regression(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_logistic_regression(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def train_decision_tree_classifier(self, X_train, y_train):
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        return model

    def train_decision_tree_regressor(self, X_train, y_train):
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        return model

    def train_random_forest_classifier(self, X_train, y_train):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model

    def train_random_forest_regressor(self, X_train, y_train):
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        return model
