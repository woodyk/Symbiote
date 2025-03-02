#!/usr/bin/env python3
#
# Reporting.py

import seaborn as sns
import matplotlib.pyplot as plt

class ReportGenerator:
    """
    Generator of textual and graphical reports.
    """

    @staticmethod
    def generate_textual_report(title: str, content: str) -> str:
        """
        Creates a simple textual report.

        :param title: Textual report title.
        :param content: Body content of the report.
        :return: Complete textual report string.
        """
        report = f"\n========== {title.upper()} REPORT ==========\n\n{content}\n======================================="

        return report

    @staticmethod
    def generate_graphical_report(title: str, dataframe: pd.DataFrame, x_column: str, y_column: str, plot_kind: str = "scatterplot") -> None:
        """
        Creates a graphical report using seaborn plots.

        :param title: Graphical report title.
        :param dataframe: DataFrame containing data for graphing.
        :param x_column: Column name used for x-axis.
        :param y_column: Column name used for y-axis.
        :param plot_kind: Kind of plot generated ("scatterplot", "lineplot", or "barplot"). Default: "scatterplot".
        :return: None
        """
        sns.set_theme()

        switcher = {
            "scatterplot": sns.scatterplot,
            "lineplot": sns.lineplot,
            "barplot": sns.barplot
        }

        plot = switcher.get(plot_kind, sns.scatterplot)

        fig, ax = plt.subplots()
        plot(data=dataframe, x=x_column, y=y_column, ax=ax)
        ax.set(title=title)
        plt.savefig(f"{title}_graph.png")
        plt.close()

        print(f"Generated {title}_graph.png")
