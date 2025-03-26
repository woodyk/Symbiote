#!/usr/bin/env python3
#
# CorrelationAnalysis.py

import scipy.stats as stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class CorrelationAnalysis:
    """
    A class encompassing various correlation analysis techniques.
    """

    @staticmethod
    def pearson_coefficients(dataset: pd.DataFrame) -> dict:
        """
        Establishes Pearson's Correlation Coefficients for input dataset.

        :param dataset: Source tabular data containing target variables.
        :return: Dictionary mapping variable names to computed coefficients.
        """
        coefficients = {}
        correlation_matrix = dataset.corr(method="pearson").abs()
        upper_triangular = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

        for column in upper_triangular.columns:
            coeff = upper_triangular[column].tolist()[::-1]
            coefficients[column] = coeff

        return coefficients

    @staticmethod
    def spearman_rank(dataset: pd.DataFrame) -> dict:
        """
        Derives Spearman's Rank Correlation for input dataset.

        :param dataset: Source tabular data housing candidate variables.
        :return: Dictionary containing variable names mapped to obtained coefficients.
        """
        coefficients = {}
        correlation_matrix = dataset.corr(method="spearman").abs()
        upper_triangular = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

        for column in upper_triangular.columns:
            coeff = upper_triangular[column].tolist()[::-1]
            coefficients[column] = coeff

        return coefficients

    @staticmethod
    def kendalls_tau(dataset: pd.DataFrame) -> dict:
        """
        Yields Kendall's Tau Correlation for input dataset.

        :param dataset: Source tabular data holding prospective variables.
        :return: Dictionary comprising variable names coupled with calculated coefficients.
        """
        coefficients = {}
        _, kendall_tau_matrix = stats.kendalltau(dataset.T)
        upper_triangular = kendall_tau_matrix.where(np.triu(np.ones(kendall_tau_matrix.shape), k=1).astype(np.bool))

        for column in upper_triangular.columns:
            coeff = upper_triangular[column].tolist()[::-1]
            coefficients[column] = coeff

        return coefficients

    @staticmethod
    def heatmap(correlation_matrix: pd.DataFrame, cmap: str = "coolwarm") -> None:
        """
        Renders a heatmap displaying the correlation matrix.

        :param correlation_matrix: Input correlation matrix derived from preceding techniques.
        :param cmap: Colormap scheme governing the visual appearance.
        :return: N/A
        """
        sns.set_style("whitegrid")
        sns.heatmap(correlation_matrix, cmap=cmap, annot=True, fmt=".2f", linewidths=.05)
        plt.title("Heatmap: Correlation Matrix")
        plt.show()

    @staticmethod
    def point_biserial(continuous_data: pd.Series, binary_data: pd.Series) -> float:
        """
        Computes Point-Biserial Correlation for input datasets.

        :param continuous_data: Source continuous data exhibiting normalcy.
        :param binary_data: Binary data assuming solely two possible values.
        :return: Scalar Point-Biserial Correlation coefficient.
        """
        return stats.pointbiserialr(binary_data, continuous_data).statistic

    @staticmethod
    def phi_coef(dataset_a: pd.Series, dataset_b: pd.Series) -> float:
        """
        Evaluates Phi Coefficient for provided datasets.

        :param dataset_a: Attribute A assumed to exhibit nominal qualities.
        :param dataset_b: Attribute B hypothesized to demonstrate nominal traits.
        :return: Normalized scalar Phi Coefficient indicating association.
        """
        return stats.phipcc(dataset_a, dataset_b)

    @staticmethod
    def cramers_v(dataset_a: pd.Series, dataset_b: pd.Series) -> float:
        """
        Determines Cramér's V for supplied datasets.

        :param dataset_a: Feature A presumed to possess ordinal or nominal nature.
        :param dataset_b: Entity B postulated to conform to ordinal or nominal conventions.
        :return: Standardized Cramér's V conveying dependency.
        """
        confusion_matrix = pd.crosstab(dataset_a, dataset_b, margins=False)
        chi_square, degrees_freedom, expected_values, approximate_correction = stats.chi2_contingency(confusion_matrix)
        n = sum(sum(confusion_matrix))
        phi = math.sqrt(chi_square / n)
        minimum_expected_value = min(min(expected_values))
        maximum_expected_value = max(max(expected_values))
        correction = math.sqrt((n - 1) / (n - minimum_expected_value - 1))
        cramer = phi * correction
        return cramer * math.sqrt(((dataset_a.nunique() - 1) * (dataset_b.nunique() - 1)) / degrees_freedom)

    @staticmethod
    def lambda_coef(dataset_a: pd.Series, dataset_b: pd.Series) -> float:
        """
        Ascertains Lambda Coefficient for designated datasets.

        :param dataset_a: Attribute A purportedly exemplifying nominal traits.
        :param dataset_b: Characteristic B suspected to mirror nominal tenets.
        :return: Scalar Lambda Coefficient indicative of connection.
        """
        empty_series = pd.Series([])
        lambda_a = stats.association(empty_series, dataset_b, "nominal", "interval")
        lambda_b = stats.association(dataset_a, empty_series, "interval", "nominal")
        return max(lambda_a.statistic, lambda_b.statistic)

    @staticmethod
    def contingency_coef(dataset_a: pd.Series, dataset_b: pd.Series) -> float:
        """
        Computes Contingency Coefficient for provided datasets.

        :param dataset_a: Attribute A assumed to exhibit nominal qualities.
        :param dataset_b: Attribute B hypothesized to demonstrate nominal traits.
        :return: Normalized scalar Contingency Coefficient indicating association.
        """
        confusion_matrix = pd.crosstab(dataset_a, dataset_b, margins=False)
        chi_square, _, _, _ = stats.chi2_contingency(confusion_matrix)
        n = sum(sum(confusion_matrix))
        contingency_coef = math.sqrt(chi_square / (chi_square + n))
        return contingency_coef
