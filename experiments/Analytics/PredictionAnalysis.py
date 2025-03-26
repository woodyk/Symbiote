#!/usr/bin/env python3
#
# prdictions.py

class PredictionAnalysis:
    def __init__(self, data, target_variable, feature_variables):
        self.data = data
        self.target_variable = target_variable
        self.feature_variables = feature_variables
        
    def linear_regression(self):
        X = self.data[self.feature_variables].values
        y = self.data[self.target_variable].values
        model = LinearRegression().fit(X, y)
        predictions = model.predict(X)
        return model, predictions
    
    def polynomial_regression(self, degree):
        X = self.data[self.feature_variables].values
        y = self.data[self.target_variable].values
        model = PolynomialRegression(degree=degree).fit(X, y)
        predictions = model.predict(X)
        return model, predictions
    
    def decision_tree_regression(self):
        X = self.data[self.feature_variables].values
        y = self.data[self.target_variable].values
        model = DecisionTreeRegressor().fit(X, y)
        predictions = model.predict(X)
        return model, predictions
    
    def random_forest_regression(self, num_trees=100):
        X = self.data[self.feature_variables].values
        y = self.data[self.target_variable].values
        model = RandomForestRegressor(n_estimators=num_trees).fit(X, y)
        predictions = model.predict(X)
        return model, predictions
    
    def gradient_boosting_regression(self, num_iter=100, learning_rate=0.1):
        X = self.data[self.feature_variables].values
        y = self.data[self.target_variable].values
        model = GradientBoostingRegressor(n_estimators=num_iter, learning_rate=learning_rate).fit(X, y)
        predictions = model.predict(X)
        return model, predictions
