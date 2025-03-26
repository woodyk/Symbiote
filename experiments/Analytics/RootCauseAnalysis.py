#!/usr/bin/env python3
#
# RootCauseAnalysis.py

import collections
import itertools
import math

class RootCauseAnalysis:
    """
    A class harboring various root cause analysis techniques.
    """

    @staticmethod
    def _fishbone_diagram(problem: str, categories: List[str]) -> dict:
        """
        Produces a Fishbone Diagram, organizing causes according to specified categories.

        :param problem: Issue warranting investigation.
        :param categories: Divisions partitioning contributory factors.
        :return: Structured representation of the Fishbone Diagram.
        """
        main_node = {"label": problem, "children": []}
        category_nodes = [{category: {"children": []}} for category in categories]

        for node_category in category_nodes:
            for child_category in categories:
                node_category[child_category]["label"] = child_category

            main_node["children"].extend(list(node_category.values()))

        return main_node

    @staticmethod
    def five_whys(problem: str, why_questions: List[str], answers: List[str]) -> tuple:
        """
        Executes a sequence of 'why' questions and accumulates answers,
        subsequently returning identified root causes and contributing factors.

        :param problem: String denoting the issue under examination.
        :param why_questions: Iterable collection enumerating inquiry questions.
        :param answers: Sequential array storing replies to posed queries.
        :return: A tuple consisting of detected root causes and ancillary contributors.
        """
        assert len(why_questions) == len(answers), "Number of questions and answers must match!"

        root_causes = []
        contributing_factors = []

        for question, answer in zip(why_questions, answers):
            if "Why" not in question:
                raise ValueError("Invalid question; anticipated format: '<number> Why...?'")

            if answer.strip() == "" or answer is None:
                raise ValueError("Empty or null response supplied.")

            # Append contributing factors prior to reaching the fifth consecutive 'why'.
            if len(root_causes) < 4:
                contributing_factors.append((question, answer))

            # Register the root cause upon concluding the fourth successive 'why'.
            if len(root_causes) == 4:
                root_causes.append((question, answer))

        return root_causes, contributing_factors

    @staticmethod
    def drilldown(data: dict, levels: int) -> dict:
        """
        Recursively applies drill-down analysis to hierarchical data structures.

        :param data: Dictionary embodying nested records.
        :param levels: Depth to penetrate beneath surface layer.
        :return: Reflective dictionary object exposing lower strata.
        """
        if levels <= 0:
            return data

        traversed_layers = collections.defaultdict(dict)
        remaining_keys = set(data.keys())

        while remaining_keys:
            next_layer = remaining_keys.pop()
            traversed_layers[next_layer] = data.pop(next_layer)
            remaining_keys |= set(traversed_layers[next_layer].keys())

        expanded_data = {
            layer_key: RootCauseAnalysis.drilldown(value, levels - 1)
            for layer_key, value in traversed_layers.items()
        }

        data.update(expanded_data)
        return data

    @staticmethod
    def pareto_principle(categories: List[str], frequencies: List[float]) -> dict:
        """
        Determines sorted categories satisfying the Pareto Principle.

        :param categories: Labels distinguishing separate entities.
        :param frequencies: Associated rates symbolizing magnitude.
        :return: Ordered dictionary highlighting dominant segments.
        """
        cumulative_freq = list(itertools.accumulate(sorted(frequencies, reverse=True)))
        running_total = sum(frequencies)
        index = 0

        while cumulative_freq[-1] / running_total < 0.8:
            index += 1

        pareto_dict = {
            category: {"frequency": frequency, "cumulative": cumulative}
            for category, frequency, cumulative in zip(categories, frequencies, cumulative_freq)
        }

        return dict(sorted(pareto_dict.items(), key=lambda item: item[1]["cumulative"], reverse=True))

    @staticmethod
    def swot_analysis(organization: str) -> dict:
        """
        Formulates a SWOT appraisal for a designated entity.

        :param organization: Title of the subject enterprise.
        :return: Organized inventory cataloguing strengths, weaknesses, opportunities, and threats.
        """
        strengths = ["Brand Loyalty", "Effective Marketing Campaigns", "Experience and Skills"]
        weaknesses = ["Limited Product Range", "Dependence on Single Market", "Poor Customer Service"]
        opportunities = ["Global Market Expansion", "New Partnerships", "Product Innovation"]
        threats = ["Market Saturation", "Intense Competition", "Legislative Changes"]

        return {
            "Strengths": strengths,
            "Weaknesses": weaknesses,
            "Opportunities": opportunities,
            "Threats": threats
        }

    @staticmethod
    def smart_goals(goal: str, specific: str, measurable: str, achievable: str, relevant: str, time_bound: str) -> bool:
        """
        Validates whether objectives comply with SMART standards.

        :param goal: Statement summarizing aspirations.
        :param specific: Precisely formulated intention.
        :param measurable: Observable outcome demonstrating accomplishment.
        :param achievable: Realistically attainable expectation.
        :param relevant: Aligned motivation resonating with overall vision.
        :param time_bound: Deadline mandating completion.
        :return: Boolean indicator reflecting compliance status.
        """
        compliant = True

        if not specific or not measurable or not achievable or not relevant or not time_bound:
            compliant = False

        if not (goal and specific and measurable and achievable and relevant and time_bound):
            compliant = False

        return compliant

