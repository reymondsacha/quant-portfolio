import numpy as np
import numpy.typing as npt
import pandas as pd
import logging
from scipy.optimize import minimize
from typing import Dict

logger = logging.getLogger("Optimizer")

class InfeasibleConstraintError(Exception):
    """
    Raised when the target return is outside the feasible range [MinVar_Return, MaxReturn]
    """
    pass

class MeanVarianceOptimizer:
    mu: npt.NDArray[np.floating]
    S: npt.NDArray[np.floating]

    def __init__(self, returns: pd.DataFrame):
        """
        Initializes with a DataFrame of returns where columns are assets
        and rows are timestamps.
        """
        self.returns = returns
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)

        # Calculate Sample Mean (mu) and Sample Covariance (S)
        self.mu = np.asarray(returns.mean())
        self.S = np.asarray(returns.cov())
        self.bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))
        self.init_guess = np.repeat(1.0 / self.n_assets, self.n_assets)

    def _apply_ledoit_wolf_shrinkage(self, delta) -> np.ndarray:
        """
        Lifts the eigenvalue spectrum by shrinking S toward the Identity target.
        """
        # Average Variance (1/n * Tr(S))
        avg_var = np.trace(self.S) / self.n_assets

        # Target  Matrix F = avg_var * I
        F = avg_var * np.eye(self.n_assets)

        # Shrunk Matrix Sigma_stable = (1-delta) * S + delta * F
        return (1 - delta) * self.S + delta * F

    def check_feasibility(self, target_return: float, sigma: npt.NDArray[np.floating]) -> bool:
        """
        Feasibility Check: The target return must be between the return of the
        Global Minimum Variance (GMV) portfolio and the maximum individual asset.
        """
        # 1. Calculate GMV weights analytically: w = (S^-1 * 1) / (1^T * S^-1 * 1)
        inv_sigma = np.linalg.inv(sigma)
        ones = np.ones(self.n_assets)
        w_gmv = inv_sigma @ ones / (ones.T @ inv_sigma @ ones)

        gmv_return = np.dot(w_gmv, self.mu)
        max_asset_return = np.max(self.mu)
        if target_return < gmv_return:
            raise InfeasibleConstraintError(
                f"Target {target_return:.4f} is below GMV return {gmv_return:.4f}."
            )
            return gmv_return
        
        if target_return > max_asset_return:
            raise InfeasibleConstraintError(
                f"Target {target_return:.4f} exceeds max asset return {max_asset_return:.4f}."
            )

        return target_return

    def calculate_risk_decomposition(self, weights_dict: Dict[str, float], delta: float)->pd.DataFrame:
        """
        Implements the Euler Identity: sigma_p = sum(w_i * MCR_i).
        """
        w = np.array([weights_dict[ticker] for ticker in self.tickers])
        sigma = self._apply_ledoit_wolf_shrinkage(delta)

        #Portfolio Volatility: sqrt(w^T * sigma * w)
        port_vol = np.sqrt(w.T @ sigma @ w)

        #Marginal Contribution to Risk (MCR): (Sigma * w) / sigma_p
        mcr = (sigma @ w) / port_vol

        #Percentage Contribution to Risk (PCR) (%CR): (w_i * MCR_i) / sigma_p
        p_cr = (w * mcr) / port_vol

        return pd.DataFrame({
            'Weight': w,
            'Marginal_Risk': mcr,
            'Percent_Contribution': p_cr
        }, index=self.tickers)


    def get_optimal_weights(self, target_return: float, delta: float, l2_lambda: float=0.01) -> Dict[str, float]:
        """
        Solves the KKT system for a Long Only Portfolio.
        """
        # 1. Get the Stabilized Covariance Matrix
        sigma_stable = self._apply_ledoit_wolf_shrinkage(delta)
        self.check_feasibility(target_return, sigma_stable)

        # 2. Define the Quadratic Risk Objective (1/2 * w^T * Sigma_stable * w)
        def portfolio_variance(w):
            return 0.5 * np.dot(w.T, np.dot(sigma_stable, w))

        def portfolio_variance_jacobian(w):
            return np.dot(sigma_stable, w)

        # 3. Equality Constraint: h(w) = 0
        constraints = [
            # Budgets Constraints : sum(w) - 1 = 0
            {"type": "eq", "fun": lambda w: np.sum(w) - 1, "jac": lambda w: np.ones(self.n_assets)},
            # Return Constraint: w^T * mu - target_return = 0
            {"type": "eq", "fun": lambda w: np.dot(w.T, self.mu) - target_return, "jac": lambda w: self.mu}
        ]

        # 4. Inequality Constraint (Bounds) : 0 <= w_i <= 1
        # Scipy's 'bounds' handles the KKT 'z' multiplier internally
        

        # 5. Initial Guess (Equal Weights)
        w0 = np.ones(self.n_assets) / self.n_assets

        # 6. Optimization using SQLP (Sequential Least Squares Programming)
        # This algorithm is designed to solve KKT problems
        result = minimize(
            fun=portfolio_variance,
            x0=w0,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
            jac=portfolio_variance_jacobian,
            tol=1e-10
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Return as a clean dictionary
        return dict(zip(self.tickers, result.x))

    def calculate_optimal_delta(self):
        T, N = self.returns.values.shape
        S = self.S
        avg_var = np.trace(self.S) / self.n_assets
        F = avg_var * np.eye(self.n_assets)

        gamma = np.linalg.norm(self.S - F, "fro") ** 2

        Y = (self.returns - self.returns.mean()).values

        pi_matrix = []
        for t in range(T):
            realized_cov = np.outer(Y[t], Y[t])
            pi_matrix.append((realized_cov - S) ** 2)

        pi = np.sum(np.mean(pi_matrix, axis=0))

        rho = 0
        for i in range(N):
            rho += np.mean(
                [(Y[t, i] ** 2 - S[i, i]) * (avg_var - S[i, i]) for t in range(T)]
            )

        delta_star = (1 / T) * (pi - rho) / gamma
        return max(0, min(delta_star, 1))


    def find_tangency_portfolio(self, risk_free_rate: float=0.0424):
        """
        Solves for the Max Sharpe Ratio using the Charne-Cooper Transformation.
        Minimizes y.T @ Sigma @ y s.t. (mu - rf1).T @ y = 1
        """
        mu_excess =  self.mu - risk_free_rate
        delta = self.calculate_optimal_delta()
        sigma = self._apply_ledoit_wolf_shrinkage(delta=delta)

        def objective(y):
            return np.dot(y.T, np.dot(sigma, y))

        def obj_jacobian(y):
            return 2 * np.dot(sigma, y)

        constraints = [
            {'type': 'eq', 'fun': lambda y: np.dot(mu_excess.T, y) - 1, 'jac': lambda y : mu_excess}
        ]

        y_bounds = tuple((0, None) for _ in range(self.n_assets))

        res = minimize(
            fun=objective,
            x0=self.init_guess,
            method="SLSQP",
            constraints=constraints,
            jac=obj_jacobian,
            bounds=y_bounds
        )
 
        if not res.success:
            raise ValueError("CCT Optimization failed to converge.")

        y_star = res.x
        weights = y_star / np.sum(y_star)

        return{
            'weights': pd.Series(weights, index=self.tickers),
            'return': np.dot(self.mu, weights),
            'volatility': np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        }

    def generate_frontier(self, n_points: int=50):
        """
        Maps the Efficient Frontier using a sweep of target returns.
        Uses the 'Feasibility Check' logic to bound the search.
        """
        min_ret = np.min(self.mu)
        max_ret = np.max(self.mu)
        delta = self.calculate_optimal_delta()

        target_returns = np.linspace(min_ret * 1.01, max_ret * 0.99, n_points)
        frontier_vols = []
        frontier_rets = []
        weights_list = []
        sigma = self._apply_ledoit_wolf_shrinkage(delta=delta)

        for target in target_returns:
            try:
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1, 'jac': lambda w: np.ones(self.n_assets)},
                    {'type': 'eq', 'fun': lambda w: np.dot(w.T, self.mu) - target, 'jac': lambda w: self.mu}
                ]
     
                res = minimize(
                    lambda w: np.dot(w.T, np.dot(sigma, w)),
                    x0=self.init_guess,
                    method="SLSQP",
                    constraints=constraints,
                    bounds=self.bounds
                )

                w_dict = self.get_optimal_weights(target, delta=delta)
                w_array= np.array([w_dict[t] for t in self.tickers])

                vol = np.sqrt(np.dot(w_array.T, np.dot(sigma, w_array)))


                if res.success:
                    frontier_vols.append(vol)
                    frontier_rets.append(target)
                    weights_list.append(res.x)
            except InfeasibleConstraintError:
                continue
        return np.array(frontier_vols), np.array(frontier_rets), np.array(weights_list)
            
        


    

