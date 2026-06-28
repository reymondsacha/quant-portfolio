import numpy as np
from src.physics.simulator import GBMSimulator

class MonteCarloGreeks:
    def __init__(self, K, T, r, sigma, dt, n_paths, epsilon=1e-4):
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.dt = dt
        self.n_paths = n_paths
        self.epsilon = epsilon

    def calculate_delta(self, S, seed):
        """
        Calculates MC Delta using central finite differences.
        """
        sim_up = GBMSimulator(seed)
        path_up = sim_up.simulate_paths(s0= S + self.epsilon, mu=self.r, sigma=self.sigma, T=self.T, dt=self.dt, n_paths=self.n_paths)

        sim_down = GBMSimulator(seed)
        path_down = sim_down.simulate_paths(s0= S - self.epsilon, mu=self.r, sigma=self.sigma, T=self.T, dt=self.dt, n_paths=self.n_paths)

        payoffs_up = np.maximum(path_up[-1] - self.K, 0)
        payoffs_down = np.maximum(path_down[-1] - self.K, 0)

        discounted_payoffs_up = np.exp(-self.r * self.T) * payoffs_up
        discounted_payoffs_down = np.exp(-self.r * self.T) * payoffs_down

        pathwise_delta = (discounted_payoffs_up - discounted_payoffs_down) / (2*self.epsilon)

        mean_delta = np.mean(pathwise_delta)

        standard_error = pathwise_delta.std(ddof=1) / np.sqrt(self.n_paths)

        return mean_delta, standard_error

    def calculate_gamma(self, S, seed):
        """
        Calculates MC Gamma using second-order central finite differences.
        """
        sim_up = GBMSimulator(seed)
        path_up = sim_up.simulate_paths(s0=S + self.epsilon, mu=self.r, sigma=self.sigma, T=self.T, dt=self.dt, n_paths=self.n_paths)

        # 2. Base simulation (S)
        sim_base = GBMSimulator(seed)
        path_base = sim_base.simulate_paths(s0=S, mu=self.r, sigma=self.sigma, T=self.T, dt=self.dt, n_paths=self.n_paths)

        # 3. Down-bump simulation (S - epsilon)
        sim_down = GBMSimulator(seed)
        path_down = sim_down.simulate_paths(s0=S - self.epsilon, mu=self.r, sigma=self.sigma, T=self.T, dt=self.dt, n_paths=self.n_paths)

        payoffs_up = np.maximum(path_up[-1] - self.K, 0)
        payoffs_base = np.maximum(path_base[-1] - self.K, 0)
        payoffs_down = np.maximum(path_down[-1] - self.K, 0)

        discount_factor = np.exp(-self.r * self.T)
        discounted_up = discount_factor * payoffs_up
        discounted_base = discount_factor * payoffs_base
        discounted_down = discount_factor * payoffs_down

        pathwise_gamma = (discounted_up - 2 * discounted_base + discounted_down) / (self.epsilon ** 2)

        mean_gamma = np.mean(pathwise_gamma)
        standard_error = pathwise_gamma.std(ddof=1) / np.sqrt(self.n_paths)

        return mean_gamma, standard_error





