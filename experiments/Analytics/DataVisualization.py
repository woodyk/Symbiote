#!/usr/bin/env python3
#
# DataVisualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

class DataVisualization:
    def __init__(self, data):
        self.data = data

    def scatter_matrix(self, columns, figsize=(12, 12)):
        try:
            corr = self.data[columns].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            f, ax = plt.subplots(figsize=figsize)
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5})
            xticks = ()
            yticks = ()
            if len(columns) >= 5:
                xticks = np.array(range(len(columns)))[:, None] * 2 + 1
                yticks = xticks
            elif len(columns) >= 3:
                xticks = np.array(range(len(columns))) + 1
                yticks = xticks
            plt.xticks(yticks)
            plt.yticks(xticks)
            plt.show()
        except Exception as e:
            print("Error while creating the matrix:", e)

    def pairplot(self, columns):
        try:
            sns.pairplot(self.data[columns], height=3.5)
            plt.show()
        except Exception as e:
            print("Error while creating the pairplot:", e)

    def countplot(self, column):
        try:
            sns.countplot(self.data[column])
            plt.show()
        except Exception as e:
            print("Error while creating the countplot:", e)

    def violinplot(self, columns):
        try:
            sns.violinplot(x=self.data[columns[0]], y=self.data[columns[1]])
            plt.show()
        except Exception as e:
            print("Error while creating the violinplot:", e)

    def boxplot(self, columns):
        try:
            sns.boxplot(x=self.data[columns[0]], y=self.data[columns[1]])
            plt.show()
        except Exception as e:
            print("Error while creating the boxplot:", e)

    def barplot(self, x_column, y_column):
        try:
            sns.barplot(x=self.data[x_column], y=self.data[y_column])
            plt.show()
        except Exception as e:
            print("Error while creating the barplot:", e)

