#!/usr/bin/env python3
#
# DiagnosticAnalytics.py

class DiagnosticAnalytics:
    def __init__(self, data):
        self.data = data

    def chi_square_test(self, observations, expectations):
        try:
            chisq_test = ss.chi2_contingency(observations)
            print("Chi-Square Test Results:\n")
            print(f"Chi-Square Statistic: {chisq_test.statistic}")
            print(f"p-value: {chisq_test.pvalue}")
            print(f"Degrees of Freedom: {chisq_test.df}")
            print(f"Expected Values: \n{expectations}\n")
        except Exception as e:
            print("Error while performing Chi-Square Test:", e)

    def oneway_anova(self, dependent_variable, group_variable):
        try:
            f_oneway = f_oneway(self.data[dependent_variable], self.data[group_variable])
            print(f"One-Way ANOVA Results:\n")
            print(f"F Value: {f_oneway[0]}")
            print(f"p-value: {f_oneway[1]}")
            print(f"Critical Value: {f_oneway[2]}")
        except Exception as e:
            print("Error while performing One-Way ANOVA:", e)

    def regression_diagnostics(self, X, y, model, title=None):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            regplot(X, y, label="Observed", scatter_kws={"color": "blue"}, ax=axes[0][0])
            regplot(X, model.predict(X), label="Fitted", scatter_kws={"color": "green"}, ax=axes[0][0])
            axes[0][0].legend()

            res_plot = ResidualsPlot(model)
            res_plot.fit(X, y)
            res_plot.poof(ax=axes[0][1])

            r2_adj = 1 - (1 - model.score(X, y)) * (len(X) - 1) / (len(X) - X.shape[1] - 1)
            print(f"Adjusted R-squared: {r2_adj}")

            if title:
                axes[0][0].set_title(title)

            plt.suptitle("Diagnostic Analytics - Regression Diagnostics", fontsize=18, y=0.96)
            plt.show()

        except Exception as e:
            print("Error while generating regression diagnostics:", e)
