import numpy as np

class GBMSimulator:
    def __init__(self, seed: int = None):
        """
        Initializes the simulator with a random seed.
        """
        self.rng = np.random.default_rng(seed)

    def simulate_paths(
        self, 
        s0: float, 
        mu: float, 
        sigma: float, 
        T: float, 
        dt: float, 
        n_paths: int
    ) -> np.ndarray:
        """
        Generates simulated paths for a Geometric Brownian Motion.
        """
        n_steps = int(T / dt)

        # discrete drift term for the log-returns
        drift = (mu - 0.5 * sigma**2) * dt

        # discrete diffusion coefficient
        diffusion = sigma * np.sqrt(dt)

        # 2D array of standard normal variables (Z)
        # Shape: (rows = n_steps, columns = n_paths)
        Z = self.rng.standard_normal((n_steps, n_paths))


        # Matrix of log-returns
        log_returns = drift + diffusion * Z

        # Cumulate the log-returns over time (along the n_steps axis)
        cumulative_log_returns = np.cumsum(log_returns, axis=0)

        # Prepend zeros for t=0 (so the exponentiated starting price is s0 * 1)
        # And exponentiate to get price levels
        cumulative_log_returns = np.vstack([np.zeros(n_paths), cumulative_log_returns])
        paths = s0 * np.exp(cumulative_log_returns)

        return paths





