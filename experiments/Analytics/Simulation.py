#!/usr/bin/env python3
#
# Simulation.py

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity

class Simulation:
    """
    A class representing various simulation techniques.
    """

    @staticmethod
    def monte_carlo_simulation(sample_size: int, distribution: type, *, mean: float = None, std_dev: float = None, func: callable = None) -> tuple:
        """
        Executes Monte Carlo Simulation.

        :param sample_size: Number of draws from the probability distribution.
        :param distribution: Distribution from which to draw samples.
        :param mean: Optional mean of the distribution (for custom distribution).
        :param std_dev: Optional standard deviation of the distribution (for custom distribution).
        :param func: Function to apply to sampled data (optional).
        :return: Tuple containing results of the simulation.
        """
        if mean is None or std_dev is None:
            if issubclass(distribution, stats._ContinuousDist):
                params = {"loc": 0, "scale": 1}
            elif issubclass(distribution, stats._DiscreteDist):
                params = {}
            else:
                raise TypeError("Unsupported distribution type.")
        else:
            params = {"mu": mean, "sigma": std_dev}

        random_draws = distribution.rvs(sample_size, loc=params.get("loc"), scale=params.get("scale"))
        estimated_result = func(random_draws) if func else random_draws.mean()

        return estimated_result, random_draws

    @staticmethod
    def bootstrap_sampling(data: pd.Series, *, sample_size: int = None, replications: int = 1000, func: callable = None, alpha: float = 0.05) -> tuple:
        """
        Performs Bootstrapped Sampling.

        :param data: Original dataset.
        :param sample_size: Size of each bootstrap sample (defaults to original data size).
        :param replications: Number of bootstrap repetitions.
        :param func: Function to apply to each bootstrap sample (optional).
        :param alpha: Significance level for constructing confidence intervals (default: 0.05).
        :return: Tuple containing results of the bootstrapped sampling.
        """
        sample_size = sample_size or len(data)
        if sample_size > len(data):
            raise ValueError("Sample size cannot exceed the original data size.")

        bootstrap_results = [data.sample(sample_size, replace=True) for _ in range(replications)]
        bootstrap_diffs = []

        if func:
            for sample in bootstrap_results:
                bootstrap_diffs.append(func(sample))
        else:
            bootstrap_diffs = [sample.mean() for sample in bootstrap_results]

        ci_lower, ci_upper = np.percentile(bootstrap_diffs, (alpha / 2) * 100), np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

        return bootstrap_diffs, ci_lower, ci_upper

    @staticmethod
    def jackknife_sampling(data: pd.Series, *, replications: int = 1000, func: callable = None, alpha: float = 0.05) -> tuple:
        """
        Performs Jackknife Resampling.

        :param data: Original dataset.
        :param replications: Number of jackknife repetitions.
        :param func: Function to apply to each jackknife sample (optional).
        :param alpha: Significance level for constructing confidence intervals (default: 0.05).
        :return: Tuple containing results of the jackknife resampling.
        """
        sample_size = len(data)
        jackknife_results = []

        for idx in range(sample_size):
            resample = data.copy()
            resample.drop(idx, axis=0, inplace=True)
            jackknife_results.append(resample)

        jackknife_diffs = []

        if func:
            for sample in jackknife_results:
                jackknife_diffs.append(func(sample))
        else:
            jackknife_diffs = [sample.mean() for sample in jackknife_results]

        bias_hat = (sample_size - 1) * (jackknife_diffs.mean() - data.mean())
        hat_deltas = [jackknife_diffs[i] - bias_hat for i in range(sample_size)]
        se_hat = np.sqrt(sample_size / (sample_size - 1) * np.var(hat_deltas))

        ci_lower, ci_upper = data.mean() - stats.norm.ppf(1 - alpha / 2) * se_hat, data.mean() + stats.norm.ppf(1 - alpha / 2) * se_hat

        return jackknife_diffs, ci_lower, ci_upper

    @staticmethod
    def gravnet_simulation(agent_count: int, timesteps: int, dt: float, softening_radius: float = 0.1) -> pd.DataFrame:
        """
        Executes a simple GravNet simulation.

        :param agent_count: Number of agents in the system.
        :param timesteps: Total number of steps to run the simulation.
        :param dt: Timestep duration.
        :param softening_radius: Softening radius for force calculation (default: 0.1).
        :return: A DataFrame containing agent positions at each timestep.
        """
        agents = [Agent(np.random.uniform(1, 2), np.random.normal(size=(2,)), np.random.normal(size=(2,))) for _ in range(agent_count)]

        positions = np.stack([agent.position for agent in agents], axis=0)
        velocities = np.stack([agent.velocity for agent in agents], axis=0)
        masses = np.array([agent.mass for agent in agents])

        trajectories = np.zeros((timesteps, agent_count, 2))
        trajectories[0] = positions

        for t in range(1, timesteps):
            net_force = Simulation._gravnet_force(positions, masses, softening_radius)
            accelerations = net_force / masses
            velocities += accelerations * dt
            positions += velocities * dt

            trajectories[t] = positions

        agent_trajectories = [[{"Position": tuple(row)} for row in frame] for frame in trajectories]
        df = pd.DataFrame(agent_trajectories)

        return df

    @staticmethod
    def _gravnet_force(positions: np.ndarray, masses: np.ndarray, softening_radius: float) -> np.ndarray:
        """
        Computes the net force vector acting on each agent.

        :param positions: Array of shape (N, 2) containing agent positions.
        :param masses: Vector of length N containing agent masses.
        :param softening_radius: Softening radius for force calculation.
        :return: Net force vectors array of shape (N, 2).
        """
        distances = np.expand_dims(np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1), axis=-1)
        radii = distances + softening_radius * np.eye(positions.shape[0])
        unit_vectors = (positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) / radii
        gravity_term = G * masses[np.newaxis, :] * masses[:, np.newaxis] / radii ** 3
        repulsion_term = kernel_repulsion * masses[np.newaxis, :] * masses[:, np.newaxis] / radii ** 3

        force = np.sum((unit_vectors * (gravity_term - repulsion_term)).swapaxes(1, 2), axis=0)

        return force
