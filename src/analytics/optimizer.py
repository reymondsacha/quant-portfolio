import numpy as np
import numpy.typing as npt
import pandas as pd
import logging
from scipy.optimize import minimize
from typing import Dict, Optional

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

        Units: Assumes returns are in a single period frequency (e.g. daily).
        self.mu and self.S are then in that same frequency (daily if rows are days).
        See UNITS.md for the full convention.
        """
        self.returns = returns
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)

        # Calculate Sample Mean (mu) and Sample Covariance (S)
        self.mu = np.asarray(returns.mean())
        self.S = np.asarray(returns.cov())
        self.bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))
        self.init_guess = np.repeat(1.0 / self.n_assets, self.n_assets)

    def solve(
        self,
        *,
        expected_returns: Optional[pd.Series] = None,
        covariance: Optional[pd.DataFrame] = None,
        risk_aversion: float = 1.0,
        max_weights: float = 0.5,
        long_only: bool = True,
        tol: float = 1e-12,
    ) -> pd.Series:
        r"""
        Solve the canonical mean-variance utility problem:

        \[
            \\max_w \\; \\mu^T w - \\frac{\\lambda}{2} w^T \\Sigma w
            \\quad \\text{s.t.} \\quad \\mathbf{1}^T w = 1
        \\]

        This formulation is the correct "equilibrium check" for Black–Litterman:
        if \\(\\Pi = \\lambda \\Sigma w_{mkt}\\) and there are no binding constraints,
        the optimizer recovers \\(w_{mkt}\\).

        Args:
            expected_returns: Expected returns \\(\\mu\\) as a Series indexed by tickers.
                If None, uses the sample mean from `self.returns`.
                Units: Must match covariance (both daily or both annual). Do not mix.
            covariance: Covariance matrix \\(\\Sigma\\) as a DataFrame with matching
                index/columns. If None, uses the sample covariance from `self.returns`.
                Units: Must match expected_returns (both daily or both annual). Do not mix.
            risk_aversion: Risk aversion \\(\\lambda > 0\\).
            long_only: If True, enforce \\(0 \\le w_i \\le 1\\).
            tol: Numerical tolerance passed to SLSQP.

        Returns:
            pd.Series: Optimal weights indexed by tickers.
        """
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive.")

        if expected_returns is None:
            mu_s = pd.Series(self.mu, index=self.tickers, name="mu")
        else:
            mu_s = expected_returns.reindex(self.tickers).astype(float)
            if mu_s.isna().any():
                missing = list(mu_s[mu_s.isna()].index)
                raise ValueError(f"expected_returns missing required tickers: {missing}")

        if covariance is None:
            sigma_df = pd.DataFrame(self.S, index=self.tickers, columns=self.tickers)
        else:
            if not isinstance(covariance, pd.DataFrame):
                raise TypeError("covariance must be a pandas DataFrame.")
            sigma_df = covariance.reindex(index=self.tickers, columns=self.tickers).astype(float)
            if sigma_df.isna().any().any():
                raise ValueError("covariance missing required tickers (NaNs after reindex).")

        if max_weights * self.n_assets < 1.0:
            raise ValueError(f"Infeasible:  max_weights {max_weights} is too low for {self.n_assets} assets.")

        sigma = np.asarray(sigma_df.values, dtype=float)
        mu = np.asarray(mu_s.values, dtype=float)

        # Minimize the convex equivalent:
        #   min_w (lambda/2) w^T Sigma w - mu^T w
        def objective(w: npt.NDArray[np.floating]) -> float:
            return float(0.5 * risk_aversion * (w.T @ sigma @ w) - (mu.T @ w))

        def objective_jacobian(w: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return (risk_aversion * (sigma @ w) - mu).astype(float)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(self.n_assets)},
        ]

        bounds = tuple((0, max_weights) for _ in range(self.n_assets)) if long_only else tuple((None, None) for _ in range(self.n_assets))
        w0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            fun=objective,
            x0=w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            jac=objective_jacobian,
            tol=tol,
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        return pd.Series(result.x, index=self.tickers, name="weights")

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


    def get_optimal_weights(self, target_return: float, delta: float) -> Dict[str, float]:
        """
        Solves the KKT system for a Long Only Portfolio.

        Units: target_return must be in the same frequency as self.mu (e.g. daily).
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


    def find_tangency_portfolio(
        self,
        risk_free_rate: float = 0.04,
        risk_free_rate_is_annual: bool = True,
    ):
        """
        Solves for the Max Sharpe Ratio using the Charnes-Cooper Transformation.
        Minimizes y.T @ Sigma @ y s.t. (mu - rf1).T @ y = 1

        Args:
            risk_free_rate: Risk-free rate. Interpreted as **annual** (e.g. 0.04)
                when risk_free_rate_is_annual is True, else same frequency as self.mu.
            risk_free_rate_is_annual: If True, risk_free_rate is in annual terms;
                it is converted to daily via (1 + r_ann)^(1/252) - 1 to match self.mu.
        """
        if risk_free_rate_is_annual:
            # Convert annual to daily: r_daily = (1 + r_ann)^(1/252) - 1
            rf = float((1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0)
        else:
            rf = float(risk_free_rate)
        mu_excess = self.mu - rf
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
            bounds=y_bounds,
            tol=1e-7
        )
 
        if not res.success:
            raise ValueError("CCT Optimization failed to converge.")

        y_star = res.x
        weights = y_star / np.sum(y_star)

        # Return and volatility are in same frequency as self.mu (e.g. daily)
        return {
            'weights': pd.Series(weights, index=self.tickers),
            'return': np.dot(self.mu, weights),
            'volatility': np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        }

    def generate_frontier(self, n_points: int=50):
        """
        Maps the Efficient Frontier using a sweep of target returns.
        Uses the 'Feasibility Check' logic to bound the search.

        Units: frontier_rets and frontier_vols are in same frequency as self.mu (e.g. daily).
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
                # Capture target in closure (default arg) so each iteration uses its own target
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1, 'jac': lambda w: np.ones(self.n_assets)},
                    {'type': 'eq', 'fun': lambda w, t=target: np.dot(w.T, self.mu) - t, 'jac': lambda w: self.mu}
                ]

                res = minimize(
                    lambda w: np.dot(w.T, np.dot(sigma, w)),
                    x0=self.init_guess,
                    method="SLSQP",
                    constraints=constraints,
                    bounds=self.bounds
                )

                w_dict = self.get_optimal_weights(target, delta=delta)
                w_array = np.array([w_dict[t] for t in self.tickers])

                vol = np.sqrt(np.dot(w_array.T, np.dot(sigma, w_array)))

                if res.success:
                    frontier_vols.append(vol)
                    frontier_rets.append(target)
                    # Use get_optimal_weights so graph matches run_analytics / compare_weights
                    weights_list.append(w_array)
            except InfeasibleConstraintError:
                continue
        return np.array(frontier_vols), np.array(frontier_rets), np.array(weights_list)
            
        

    def compute_active_share(self, market_weights: pd.Series, portfolio_weights: pd.Series) -> float:
        """
        Computes the Active Share of a portfolio compared to the market.
        """
        active_weights = portfolio_weights - market_weights
        return (1/2) * np.sum(np.abs(active_weights))

    def compute_active_error(self, S, market_weights: pd.Series, portfolio_weights: pd.Series) -> float:
        """
        Computes the Active Error of a portfolio compared to the market.
        """
        active_weights = portfolio_weights - market_weights
        return np.sqrt(np.dot(active_weights.T, np.dot(S, active_weights)))

    

