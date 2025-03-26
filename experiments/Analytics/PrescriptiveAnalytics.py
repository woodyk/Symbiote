#!/usr/bin/env python3
#
# PrescriptiveAnalytics.py

import pyDOE2
import optuna
import odo
import gurobipy
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class PrescriptiveAnalytics:
    def __init__(self, data):
        self.data = data

    def optimal_experimental_design(self, design_space, criteria):
        try:
            design = pyDOE2.generateDesignMatrix(design_space, criteria)
            return design
        except Exception as e:
            print("Error while performing Optimal Experimental Design:", e)

    def simulation_optimization(self, objective_function, bounds, num_samples):
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_function, n_trials=num_samples, bounds=bounds)
            best_params = study.best_params
            return best_params
        except Exception as e:
            print("Error while performing Simulation Optimization:", e)

    def decision_tree_analysis(self, X, y, criterion, max_depth, min_samples_split, min_samples_leaf):
        try:
            if isinstance(y, int) or isinstance(y, float):
                model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            else:
                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

            model.fit(X, y)
            return model
        except Exception as e:
            print("Error while performing Decision Tree Analysis:", e)

    def mixed_integer_programming(self, obj_fun, constr, lb, ub, integer_var):
        try:
            m = gurobipy.Model("MIP Problem")
            m.Params.OutputFlag = 0
            m.setObjective(obj_fun, sense=gurobipy.GRB.MAXIMIZE)

            for j in range(len(lb)):
                var = m.addVar(vtype=gurobipy.GRB.CONTINUOUS if not integer_var[j] else gurobipy.GRB.INTEGER, lb=lb[j], ub=ub[j], name="x{}".format(j))
                m.update()

            for con in constr:
                m.addConstr(con)

            m.optimize()

            solution = []
            for v in m.getVars():
                solution.append(v.x)

            return solution
        except Exception as e:
            print("Error while solving Mixed Integer Programming problem:", e)
