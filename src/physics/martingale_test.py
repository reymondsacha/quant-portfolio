import numpy as np

class MartingaleAuditor:
    def __init__(self):
        pass

    def run_test(self, simulator, S0, r, T, n_paths, **sim_kwargs):
        """
        Audits a simulator for the Martingale property.
        
        Parameters:
        simulator: callable, returns (paths, time_grid) where paths is shape (n_paths, n_steps+1)
        S0: float, initial spot price
        r: float, risk-free rate
        T: float, time to maturity
        n_paths: int, number of simulation paths
        sim_kwargs: additional arguments for the simulator (e.g., drift, vol, n_steps)
        
        Returns:
        dict containing time_grid, expected_values, martingale_errors, and std_errors
        """
        # Generate paths from the provided simulator
        paths, time_grid = simulator(S0=S0, T=T, n_paths=n_paths, **sim_kwargs)
        
        # Vectorized discounting
        discount_factors = np.exp(-r * time_grid)
        
        # Broadcasting discount_factors (shape: n_steps+1) across paths (shape: n_paths, n_steps+1)
        discounted_paths = paths * discount_factors
        
        # Calculate expectations and statistics across the path axis (axis=0)
        expected_values = np.mean(discounted_paths, axis=0)
        martingale_errors = expected_values - S0
        
        # Standard Error of the mean: std / sqrt(N)
        std_devs = np.std(discounted_paths, axis=0, ddof=1)
        std_errors = std_devs / np.sqrt(n_paths)
        
        return {
            'time_grid': time_grid,
            'expected_values': expected_values,
            'martingale_errors': martingale_errors,
            'std_errors': std_errors
        }