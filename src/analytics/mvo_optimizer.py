import numpy as np
import numpy.typing as npt
import pandas as pd
import logging
from scipy.optimize import minimize, linprog
from typing import Dict, Optional, cast

logger = logging.getLogger("Optimizer")


class InfeasibleConstraintError(Exception):
    """
    Raised when the target return is outside the feasible range [MinVar_Return, MaxReturn]
    """

    pass


class MeanVarianceOptimizer:
    mu: npt.NDArray[np.floating]
    S: npt.NDArray[np.floating]

    def __init__(self, returns: pd.DataFrame, frequency: int = 252):
        """
        Initializes with a DataFrame of returns where columns are assets
        and rows are timestamps.

        Units:
            - Assumes `returns` are in a single-period frequency (e.g. **daily**).
            - `self.mu` and `self.S` are in that same frequency (e.g. daily).
            - `self.frequency` is the number of periods per year (default 252).
              It is only used for unit diagnostics and documentation, it does
              not rescale inputs automatically.

        """
        self.returns = returns
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)
        self.frequency: int = int(frequency)

        # Calculate Sample Mean (mu) and Sample Covariance (S)
        self.mu = np.asarray(returns.mean())
        self.S = np.asarray(returns.cov())
        self.bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))
        self.init_guess = np.repeat(1.0 / self.n_assets, self.n_assets)
        self.delta = None

    def _validate_units(self, mu: pd.Series, sigma_df: pd.DataFrame) -> None:
        """
        Lightweight diagnostic to catch obvious unit mismatches between
        expected returns and covariance.

        Heuristic:
            - If |mu| is "annual-like" (~5–50%) and
              diag(cov) is "daily-like" (~1e-6–1e-3), we raise.
            - Conversely, if |mu| is "daily-like" (~0–1%) and
              diag(cov) is "annual-like" (> 1e-2), we also raise.
        """
        if sigma_df.shape[0] != sigma_df.shape[1]:
            raise ValueError("covariance must be square for unit diagnostics.")

        mean_abs_mu = float(mu.abs().mean())
        diag_var = np.diag(sigma_df.values.astype(float))
        mean_var = float(np.mean(diag_var))

        # Basic sanity: non-negative variance
        if mean_var < 0:
            raise ValueError("covariance has negative average variance, invalid input.")

        # Guard: annual mu with daily cov
        if 0.05 <= mean_abs_mu <= 0.5 and 0.0 < mean_var <= 1e-3:
            raise ValueError(
                "Unit mismatch detected: expected_returns look annual "
                "while covariance looks daily-scale. Rescale one of them "
                f"so they share a common frequency (e.g. both annual or both {self.frequency}-period)."
            )

        # Guard: daily mu with annual cov
        if 0.0 < mean_abs_mu <= 0.01 and mean_var >= 1e-2:
            raise ValueError(
                "Unit mismatch detected: expected_returns look single-period "
                "while covariance looks annual-scale. Rescale one of them "
                f"so they share a common frequency (e.g. both annual or both {self.frequency}-period)."
            )

    def solve(
        self,
        *,
        expected_returns: Optional[pd.Series] = None,
        covariance: Optional[pd.DataFrame] = None,
        risk_aversion: float = 1.0,
        max_weights: float = 0.5,
        long_only: bool = True,
        tol: float = 1e-12,
        factor_exposure: Optional[pd.DataFrame] = None,
        factor_limits: Optional[dict[int, float]] = None
    ) -> pd.Series:
        r"""
        Solve the canonical mean-variance utility problem.


        Returns:
            pd.Series: Optimal weights indexed by tickers.
        """
        if factor_exposure is not None and factor_limits is not None:
            self.check_factor_feasibility(factor_exposure, factor_limits, max_weights)
        
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive.")

        if expected_returns is None:
            mu_s = pd.Series(self.mu, index=self.tickers, name="mu")
        else:
            mu_s = expected_returns.reindex(self.tickers).astype(float)
            if mu_s.isna().any():
                missing = list(cast(pd.Index, mu_s.index[mu_s.isna()]))
                raise ValueError(
                    f"expected_returns missing required tickers: {missing}"
                )

        if covariance is None:
            sigma_df = pd.DataFrame(
                self.S,
                index=pd.Index(self.tickers),
                columns=pd.Index(self.tickers),
            )
        else:
            if not isinstance(covariance, pd.DataFrame):
                raise TypeError("covariance must be a pandas DataFrame.")
            sigma_df = covariance.reindex(
                index=self.tickers, columns=self.tickers
            ).astype(float)
            if bool(sigma_df.isna().to_numpy().any()):
                raise ValueError(
                    "covariance missing required tickers (NaNs after reindex)."
                )

        # Guard-rail against daily/annual mismatches
        self._validate_units(mu_s, sigma_df)

        if max_weights * self.n_assets < 1.0:
            raise ValueError(
                f"Infeasible:  max_weights {max_weights} is too low for {self.n_assets} assets."
            )

        sigma = np.asarray(sigma_df.values, dtype=float)
        mu = np.asarray(mu_s.values, dtype=float)

        # Minimize the convex equivalent:
        #   min_w (lambda/2) w^T Sigma w - mu^T w
        def objective(w: npt.NDArray[np.floating]) -> float:
            return float(0.5 * risk_aversion * (w.T @ sigma @ w) - (mu.T @ w))

        def objective_jacobian(w: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return (risk_aversion * (sigma @ w) - mu).astype(float)

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones(self.n_assets),
            },
        ]

        if factor_exposure is not None:
            if factor_limits is None:
                raise ValueError("factor_limits must be provided if factor_exposure is provided.")
            for k, limit in factor_limits.items():
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, k=k, lim=limit: (factor_exposure[:, k] @ w) + lim,
                    "jac": lambda w, k=k: factor_exposure[:, k]
                })
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, k=k, lim=limit: lim - (w @ factor_exposure[:, k]),
                    "jac": lambda w, k=k: - factor_exposure[:, k]
                })

        bounds = (
            tuple((0, max_weights) for _ in range(self.n_assets))
            if long_only
            else tuple((None, None) for _ in range(self.n_assets))
        )
        w0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            fun=objective,
            x0=w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            jac=objective_jacobian,
            tol=tol,
            options={"maxiter": 1000},
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        return pd.Series(result.x, index=self.tickers, name="weights")

    def _apply_ledoit_wolf_shrinkage(self) -> np.ndarray:
        """
        Lifts the eigenvalue spectrum by shrinking S toward the Identity target.
        """
        if self.delta is None:
            self.calculate_optimal_delta()
        # Average Variance (1/n * Tr(S))
        avg_var = np.trace(self.S) / self.n_assets

        # Target  Matrix F = avg_var * I
        F = avg_var * np.eye(self.n_assets)

        # Shrunk Matrix Sigma_stable = (1-delta) * S + delta * F
        return (1 - self.delta) * self.S + self.delta * F

    def check_feasibility(
        self, target_return: float, sigma: npt.NDArray[np.floating]
    ) -> bool:
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

        if target_return > max_asset_return:
            raise InfeasibleConstraintError(
                f"Target {target_return:.4f} exceeds max asset return {max_asset_return:.4f}."
            )

        return True


    def check_factor_feasibility(
        self, 
        factor_exposure: pd.DataFrame, 
        factor_limits: Dict[int, float], 
        max_weight: float = 1.0
    ) -> bool:
        """
        Checks if a feasible portfolio exists given the factor constraints.
        Returns True if feasible, raises InfeasibleConstraintError otherwise.
        """
        n_assets = self.n_assets
        # Objective: We don't care about the result, just feasibility. 
        # We use a dummy objective (all zeros).
        c = np.zeros(n_assets)

        # Equality Constraints: sum(w) = 1
        A_eq = [np.ones(n_assets)]
        b_eq = [1.0]

        # Inequality Constraints: factor_exposure @ w <= limit AND -factor_exposure @ w <= limit
        A_ub = []
        b_ub = []
    
        for k, limit in factor_limits.items():
            v_k = factor_exposure[:, k]
            A_ub.append(v_k)        # v_k @ w <= limit
            b_ub.append(limit)
            A_ub.append(-v_k)       # -v_k @ w <= limit (same as v_k @ w >= -limit)
            b_ub.append(limit)

        # Bounds: 0 <= w_i <= max_weight
        bounds = [(0, max_weight) for _ in range(n_assets)]

        # Solve the LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not res.success:
            # Diagnostic: Find which factor is the likely culprit
            problematic_factors = []
            for k, limit in factor_limits.items():
                v_k = factor_exposure[:, k]
                # If the best an asset can do is higher than the limit, it's a conflict
                if np.min(v_k) > limit or np.max(v_k) < -limit:
                    problematic_factors.append(k)
        
            raise InfeasibleConstraintError(
                f"Factor constraints are too tight. No feasible weights found. "
                f"Check factors: {problematic_factors}"
            )
    
        return True

    def calculate_risk_decomposition(
        self, weights_dict: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Implements the Euler Identity: sigma_p = sum(w_i * MCR_i).
        """   
        w = np.array([weights_dict[ticker] for ticker in self.tickers])
        sigma = self._apply_ledoit_wolf_shrinkage()

        # Portfolio Volatility: sqrt(w^T * sigma * w)
        port_vol = np.sqrt(w.T @ sigma @ w)

        # Marginal Contribution to Risk (MCR): (Sigma * w) / sigma_p
        mcr = (sigma @ w) / port_vol

        # Percentage Contribution to Risk (PCR) (%CR): (w_i * MCR_i) / sigma_p
        p_cr = (w * mcr) / port_vol

        return pd.DataFrame(
            {"Weight": w, "Marginal_Risk": mcr, "Percent_Contribution": p_cr},
            index=pd.Index(self.tickers),
        )

    def calculate_idiosyncratic_risk(self, weights_dict: Dict[str, float], eigenvectors: npt.NDArray[np.floating], k: int = 3, covariance: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculates the factor variance attribution using the Systematic Projection method.
        This prevents negative idiosyncratic risk by ensuring both components derive from the same Sigma.
        """
        # 1. Handle Covariance Source
        if covariance is None:
            Sigma = self._apply_ledoit_wolf_shrinkage()
        else:
            Sigma = np.asarray(covariance.values, dtype=float)

        w = np.array([weights_dict[ticker] for ticker in self.tickers])
        
        # 2. Total Portfolio Variance
        total_risk = w.T @ Sigma @ w
        
        # 3. Systematic Risk via Projection
        # We project the Covariance matrix itself onto the top K eigenvectors
        # Systematic_Sigma = V @ V.T @ Sigma @ V @ V.T
        V = eigenvectors[:, :k]
        P = V @ V.T  # Projection Matrix
        systematic_sigma = P @ Sigma @ P
        
        factor_risk = w.T @ systematic_sigma @ w
        idiosyncratic_risk = total_risk - factor_risk

        return pd.DataFrame(
            {"Factor_Risk": factor_risk, "Idiosyncratic_Risk": max(0, idiosyncratic_risk)},
            index=["Variance"]
        )
        
    def get_optimal_weights(
        self,
        target_return: float,
        covariance: Optional[pd.DataFrame] = None,
        factor_exposure: Optional[pd.DataFrame] = None,
        factor_limits: Optional[dict[int, float]] = None
    ) -> pd.Series:
        """
        Solves the KKT system for a Long Only Portfolio.

        Units:
            - `target_return` must be in the same frequency as `self.mu`
              and `covariance` (e.g. all daily or all annual).
            - If `covariance` is provided it is used as-is and no
              internal shrinkage is computed.
        """
        if factor_exposure is not None and factor_limits is not None:
            self.check_factor_feasibility(factor_exposure, factor_limits)
        
        # 1. Get the Stabilized Covariance Matrix (or trust injected covariance)
        if covariance is None:
            sigma_stable = self._apply_ledoit_wolf_shrinkage()
            sigma_df = pd.DataFrame(
                sigma_stable, index=self.tickers, columns=self.tickers
            )
        else:
            if not isinstance(covariance, pd.DataFrame):
                raise TypeError("covariance must be a pandas DataFrame.")
            sigma_df = covariance.reindex(
                index=self.tickers, columns=self.tickers
            ).astype(float)
            if bool(sigma_df.isna().to_numpy().any()):
                raise ValueError(
                    "covariance missing required tickers (NaNs after reindex)."
                )
            sigma_stable = np.asarray(sigma_df.values, dtype=float)

        # Unit diagnostics
        mu_s = pd.Series(self.mu, index=self.tickers, name="mu")
        self._validate_units(mu_s, sigma_df)
        self.check_feasibility(target_return, sigma_stable)

        # 2. Define the Quadratic Risk Objective (1/2 * w^T * Sigma_stable * w)
        def portfolio_variance(w):
            return 0.5 * np.dot(w.T, np.dot(sigma_stable, w))

        def portfolio_variance_jacobian(w):
            return np.dot(sigma_stable, w)

        # 3. Equality Constraint: h(w) = 0
        constraints = [
            # Budgets Constraints : sum(w) - 1 = 0
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1,
                "jac": lambda w: np.ones(self.n_assets),
            },
            # Return Constraint: w^T * mu - target_return = 0
            {
                "type": "eq",
                "fun": lambda w: np.dot(w.T, self.mu) - target_return,
                "jac": lambda w: self.mu,
            },
        ]

        if factor_exposure is not None:
            if factor_limits is None:
                raise ValueError("factor_limits must be provided if factor_exposure is provided.")
            for k, limit in factor_limits.items():
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, k=k, lim=limit: lim - (w @ factor_exposure[:, k]),
                    "jac": lambda w, k=k: - factor_exposure[:, k]
                })

                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, k=k, lim=limit: lim + (w @ factor_exposure[:, k]),
                    "jac": lambda w, k=k: factor_exposure[:, k]
                })

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
            tol=1e-10,
            options={"maxiter": 1000},
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Return as a pd.Series with canonical ticker order
        return pd.Series(result.x, index=self.tickers, name="weights")

    def calculate_optimal_delta(self):
        T, N = self.returns.values.shape
        S = self.S
        avg_var = np.trace(self.S) / self.n_assets
        F = avg_var * np.eye(self.n_assets)

        gamma = np.linalg.norm(self.S - F, "fro") ** 2

        Y = (self.returns - self.returns.mean()).values  # (T, N)

        # pi: mean over t of sum of (realized_cov_t - S)^2; vectorized via (T, N, N)
        realized_cov_all = Y[:, :, np.newaxis] * Y[:, np.newaxis, :]  # (T, N, N)
        pi = np.sum(np.mean((realized_cov_all - S) ** 2, axis=0))

        # rho: sum over i of mean over t of (Y[t,i]^2 - S[i,i]) * (avg_var - S[i,i])
        diag_S = np.diag(S)
        rho = np.sum(np.mean((Y**2 - diag_S) * (avg_var - diag_S), axis=0))

        delta_star = (1 / T) * (pi - rho) / gamma
        self.delta = max(0, min(delta_star, 1))
        return self.delta

    def find_tangency_portfolio(
        self,
        risk_free_rate: float = 0.04,
        risk_free_rate_is_annual: bool = True,
        covariance: Optional[pd.DataFrame] = None,
        factor_exposure: Optional[pd.DataFrame] = None,
        factor_limits: Optional[dict[int, float]] = None
    ):
        r"""
        Solves for the Max Sharpe Ratio using the Charnes-Cooper Transformation.
        Minimizes y.T @ Sigma @ y s.t. (mu - rf1).T @ y = 1

        Args:
            risk_free_rate: Risk-free rate. Interpreted as annual (e.g. 0.04)
                when risk_free_rate_is_annual is True, else same frequency as self.mu.
            risk_free_rate_is_annual: If True, risk_free_rate is in annual terms;
                it is converted to daily via (1 + r_ann)^(1/252) - 1 to match self.mu.
            covariance: Optional covariance matrix Sigma as a DataFrame.
                If provided, it is trusted as-is and no internal shrinkage is
                computed. Its units must match `self.mu` and the risk-free rate
                after any conversion.
        """
        if factor_exposure is not None and factor_limits is not None:
            self.check_factor_feasibility(factor_exposure, factor_limits)

        if risk_free_rate_is_annual:
            # Convert annual to daily: r_daily = (1 + r_ann)^(1/252) - 1
            rf = float((1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0)
        else:
            rf = float(risk_free_rate)
        mu_excess = self.mu - rf

        if covariance is None:
            sigma_df = pd.DataFrame(
                self._apply_ledoit_wolf_shrinkage(),
                index=self.tickers,
                columns=self.tickers,
            )
        else:
            if not isinstance(covariance, pd.DataFrame):
                raise TypeError("covariance must be a pandas DataFrame.")
            sigma_df = covariance.reindex(
                index=self.tickers, columns=self.tickers
            ).astype(float)
            if bool(sigma_df.isna().to_numpy().any()):
                raise ValueError(
                    "covariance missing required tickers (NaNs after reindex)."
                )

        # Unit diagnostics for excess returns vs covariance
        mu_excess_s = pd.Series(mu_excess, index=self.tickers, name="mu_excess")
        self._validate_units(mu_excess_s, sigma_df)

        sigma = np.asarray(sigma_df.values, dtype=float)

        def objective(y):
            return np.dot(y.T, np.dot(sigma, y))

        def obj_jacobian(y):
            return 2 * np.dot(sigma, y)

        constraints = [
            {
                "type": "eq",
                "fun": lambda y: np.dot(mu_excess.T, y) - 1,
                "jac": lambda y: mu_excess,
            }
        ]

        if factor_exposure is not None:
            if factor_limits is None:
                raise ValueError("factor_limits must be provided if factor_exposure is provided.")
            for k, limit in factor_limits.items():
                constraints.append({
                    "type": "ineq",
                    "fun": lambda y, k=k, lim=limit: lim * y.sum() - (y.T @ factor_exposure[:, k]),
                    "jac": lambda y, k=k, lim=limit: lim * np.ones(self.n_assets) - factor_exposure[:, k]
                })

                constraints.append({
                    "type": "ineq",
                    "fun": lambda y, k=k, lim=limit: (y.T @ factor_exposure[:, k]) + lim * y.sum(),
                    "jac": lambda y, k=k, lim=limit: lim * np.ones(self.n_assets) + factor_exposure[:, k]
                })

        y_bounds = tuple((0, None) for _ in range(self.n_assets))

        res = minimize(
            fun=objective,
            x0=self.init_guess,
            method="SLSQP",
            constraints=constraints,
            jac=obj_jacobian,
            bounds=y_bounds,
            tol=1e-7,
            options={"maxiter": 1000},
        )

        if not res.success:
            raise ValueError("CCT Optimization failed to converge.")

        y_star = res.x
        weights = y_star / np.sum(y_star)

        # Return and volatility are in same frequency as self.mu (e.g. daily)
        return {
            "weights": pd.Series(weights, index=self.tickers),
            "return": np.dot(self.mu, weights),
            "volatility": np.sqrt(np.dot(weights.T, np.dot(sigma, weights))),
        }

    def generate_frontier(self, n_points: int = 50):
        """
        Maps the Efficient Frontier using a sweep of target returns.
        Uses the 'Feasibility Check' logic to bound the search.

        Units: frontier_rets and frontier_vols are in same frequency as self.mu (e.g. daily).
        """
        min_ret = np.min(self.mu)
        max_ret = np.max(self.mu)
        self.calculate_optimal_delta()

        target_returns = np.linspace(min_ret * 1.01, max_ret * 0.99, n_points)
        frontier_vols = []
        frontier_rets = []
        weights_list = []
        sigma = self._apply_ledoit_wolf_shrinkage()

        sigma_df = pd.DataFrame(sigma, index=self.tickers, columns=self.tickers)
        mu_s = pd.Series(self.mu, index=self.tickers, name="mu")
        self._validate_units(mu_s, sigma_df)

        for target in target_returns:
            try:
                # Get optimal weights using the consistent method from run_analytics
                w_series = self.get_optimal_weights(target)
                w_array = w_series.reindex(self.tickers).to_numpy(dtype=float)

                # Calculate volatility using the shrunk covariance matrix
                vol = np.sqrt(np.dot(w_array.T, np.dot(sigma, w_array)))

                frontier_vols.append(vol)
                frontier_rets.append(target)
                weights_list.append(w_array)
            except (InfeasibleConstraintError, ValueError) as e:
                # Skip infeasible points (outside the feasible range)
                continue
        return np.array(frontier_vols), np.array(frontier_rets), np.array(weights_list)

    def compute_active_share(
        self, market_weights: pd.Series, portfolio_weights: pd.Series
    ) -> float:
        """
        Computes the Active Share of a portfolio compared to the market.
        """
        active_weights = portfolio_weights - market_weights
        return (1 / 2) * np.sum(np.abs(active_weights))

    def compute_active_error(
        self, S, market_weights: pd.Series, portfolio_weights: pd.Series
    ) -> float:
        """
        Computes the Active Error of a portfolio compared to the market.
        """
        active_weights = portfolio_weights - market_weights
        return np.sqrt(np.dot(active_weights.T, np.dot(S, active_weights)))
