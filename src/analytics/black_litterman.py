from typing import cast

import numpy as np
import pandas as pd


class BlackLittermanModel:
    def __init__(
        self, 
        market_caps: pd.Series | None = None,
        benchmark_return: float | None = None,
        risk_free_rate: float = 0.04, 
        frequency: int = 252
    ) -> None:
        """
        Black–Litterman model for portfolio optimization with investor views.

        Parameters:
            market_caps: Market capitalization for each asset (optional, can be set later)
            benchmark_return: Annual benchmark return (optional, defaults to 0.08 if not provided)
            risk_free_rate: Annual risk-free rate (default: 0.04)
            frequency: Number of periods per year (default: 252 for daily data)

        Units:
            - `self.rf` is treated as **annual** (e.g. 0.04).
            - All other inputs/outputs (benchmark_return, benchmark_var, covariance,
              implied returns, posterior returns) must share a common frequency
              (typically annual). See UNITS.md.
            - `self.frequency` is the number of periods per year (default 252),
              exposed for consistency with other optimizers and documentation.
        
        Usage:
            # Simple usage - just add views and call get_posterior_returns
            bl = BlackLittermanModel(market_caps=market_caps, benchmark_return=0.10)
            bl.add_view({'AAPL': 1.0}, target_return=0.15, confidence_score=0.5)
            posterior_returns = bl.get_posterior_returns(sigma_ann, tau=0.025)
        """
        self.rf: float = float(risk_free_rate)
        self.frequency: int = int(frequency)
        self.views = []
        self.P = None
        self.Q = None
        self.Omega = None
        self.Pi = None
        self.market_weights = None
        self.lambda_reg = None
        self._market_caps = market_caps
        self._benchmark_return = benchmark_return if benchmark_return is not None else 0.08
        self.benchmark_var = None

    def get_market_weights(self, market_caps: pd.Series) -> pd.Series:
        """
        Convert raw market caps into a weight vector (summing to 1)
        """
        self.market_weights = market_caps / market_caps.sum()
        return self.market_weights

    def calculate_risk_aversion(
        self, benchmark_return: float, market_caps: pd.Series, sigma_ann: pd.DataFrame
    ) -> float:
        """
        Estimate lambda (Risk Aversion) from benchmark performance.

        Units: benchmark_return and benchmark_var must be annual. self.rf is annual.
        """
        if self.market_weights is None:
                self.get_market_weights(market_caps)
        w_mkt_arr = self.market_weights.to_numpy(dtype=float)
        self.benchmark_var = float(w_mkt_arr.T @ sigma_ann @ w_mkt_arr)
        if self.benchmark_var <= 0:
            raise ValueError("benchmark_var must be positive to compute risk aversion.")
        self.lambda_reg = float((benchmark_return - self.rf) / self.benchmark_var)
        return self.lambda_reg
        

    def calculate_implied_returns(
        self, covariance: pd.DataFrame, market_weights: pd.Series | None = None
    ) -> pd.Series:
        """
        Derive Equilibrium Returns: Pi = lambda * Sigma * w_mkt.

        Units: covariance and returned Pi are in the same frequency (e.g. both annual).
        """
        if market_weights is None:
            if self.market_weights is None:
                if self._market_caps is not None:
                    market_weights = self.get_market_weights(self._market_caps)
                else:
                    raise ValueError("market_weights must be provided or computed via get_market_weights()")
            else: 
                market_weights = self.market_weights

        if self.lambda_reg is None:
            if self._market_caps is None:
                raise ValueError("Cannot calculate risk aversion without market_caps")
            self.calculate_risk_aversion(
                benchmark_return=self._benchmark_return, 
                market_caps=self._market_caps, 
                sigma_ann=covariance
            )
            
        if not isinstance(covariance, pd.DataFrame):
            raise TypeError(
                "covariance must be a pandas DataFrame with matching index/columns."
            )
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be square.")
        if not covariance.index.equals(covariance.columns):
            raise ValueError(
                "covariance index and columns must match (same assets, same order)."
            )

        w_aligned = market_weights.reindex(covariance.index).astype(float)
        if w_aligned.isna().any():
            missing = list(cast(pd.Index, w_aligned.index[w_aligned.isna()]))
            raise ValueError(
                f"market_weights missing assets required by covariance: {missing}"
            )

        # Pandas-safe multiplication with explicit alignment:
        # Pi = lambda * Sigma * w_mkt
        self.Pi = float(self.lambda_reg) * (covariance @ w_aligned)
        self.Pi.name = "Implied Returns"
        return self.Pi

    def add_view(
        self, weights: dict[str, float], target_return: float, confidence_score: float
    ):
        """
        Add an investor view to the model.
        
        Parameters:
            weights: Dictionary mapping tickers to weights (e.g., {'AAPL': 1.0} for absolute view,
                    {'AAPL': 1.0, 'MSFT': -1.0} for relative view)
            target_return: Expected return for this view (annual)
            confidence_score: Confidence in the view, between 0 and 1 (exclusive)
                            Higher values = more confident
        
        Example:
            # Absolute view: AAPL will return 15% annually
            bl.add_view({'AAPL': 1.0}, target_return=0.15, confidence_score=0.5)
            
            # Relative view: AAPL will outperform MSFT by 5%
            bl.add_view({'AAPL': 1.0, 'MSFT': -1.0}, target_return=0.05, confidence_score=0.7)
        """
        if not 0 < confidence_score < 1:
            raise ValueError("confidence_score must be between 0 and 1 (exclusive)")
        
        self.views.append(
            {"weights": weights, "q": target_return, "confidence": confidence_score}
        )
        # Reset matrices so they'll be recomputed with new view
        self.P = None
        self.Q = None
        self.Omega = None

    def reset(self):
        """
        Reset all computed values. Call this if you want to change market_caps,
        benchmark_return, or start with fresh views.
        """
        self.views = []
        self.P = None
        self.Q = None
        self.Omega = None
        self.Pi = None
        self.market_weights = None
        self.lambda_reg = None
    
    def _build_matrices(self, tickers: list[str]):
        K = len(self.views)
        self.P = np.zeros((K, (len(tickers))))
        self.Q = np.zeros((K, 1))

        for i, view in enumerate(self.views):
            for ticker, weight in view["weights"].items():
                try:
                    idx = tickers.index(ticker)
                except ValueError:
                    raise ValueError(f"Ticker {ticker} not found in tickers list")
                self.P[i, idx] = weight
            self.Q[i, 0] = view["q"]

        return self.P, self.Q

    def compute_omega(self, P, sigma, tau):
        P = np.array(P)
        K = P.shape[0]
        omega_diag = []
        for i in range(K):
            confidence_score = self.views[i]["confidence"]
            try:
                omega_diag.append(
                    ((1 - confidence_score) / confidence_score)
                    * tau
                    * P[i]
                    @ sigma
                    @ P[i].T
                )
            except ZeroDivisionError:
                raise ValueError("confidence_score can't be 0 or 1")
        return np.diag(omega_diag)

    def get_posterior_returns(
        self, 
        Sigma: pd.DataFrame, 
        tau: float = 0.025,
        market_caps: pd.Series | None = None,
        benchmark_return: float | None = None
    ) -> pd.Series:
        """
        Compute posterior returns incorporating investor views.
        
        This method automatically computes all intermediate steps if not already done:
        - Market weights (from market caps)
        - Risk aversion coefficient
        - Implied equilibrium returns
        - View matrices (P, Q, Omega)
        
        Parameters:
            Sigma: Covariance matrix (must be annual, matching units in UNITS.md)
            tau: Uncertainty in the prior (default: 0.025)
            market_caps: Market capitalization for each asset (optional if provided in __init__)
            benchmark_return: Annual benchmark return (optional if provided in __init__)
        
        Returns:
            pd.Series: Posterior expected returns for each asset
        
        Units: 
            Pi, Sigma, Omega and returned E must all be in the same frequency (e.g. annual).
        """
        # Update market_caps and benchmark_return if provided
        if market_caps is not None:
            self._market_caps = market_caps
        if benchmark_return is not None:
            self._benchmark_return = benchmark_return
            
        # Get list of tickers from the covariance matrix
        tickers = list(Sigma.index)
        
        # Step 1: Compute market weights if not already done
        if self.market_weights is None:
            if self._market_caps is None:
                raise ValueError(
                    "market_caps must be provided either in __init__ or get_posterior_returns()"
                )
            self.get_market_weights(self._market_caps)
        
        # Step 2: Compute risk aversion if not already done
        if self.lambda_reg is None:
            if self._market_caps is None:
                raise ValueError(
                    "market_caps required to compute risk aversion. "
                    "Provide in __init__ or get_posterior_returns()"
                )
            self.calculate_risk_aversion(
                benchmark_return=self._benchmark_return,
                market_caps=self._market_caps,
                sigma_ann=Sigma
            )
        
        # Step 3: Compute implied returns if not already done
        if self.Pi is None:
            self.calculate_implied_returns(Sigma, self.market_weights)
        
        # Step 4: Build view matrices if not already done
        if self.P is None or self.Q is None:
            if len(self.views) == 0:
                raise ValueError(
                    "No views have been added. Use add_view() to add at least one view."
                )
            self.P, self.Q = self._build_matrices(tickers)
        
        # Step 5: Compute Omega (uncertainty in views) if not already done
        if self.Omega is None:
            self.Omega = self.compute_omega(self.P, Sigma, tau)
        
        # Step 6: Compute posterior returns using Black-Litterman formula
        Sigma_np = Sigma.to_numpy()
        market_precision = np.linalg.inv(tau * Sigma_np)
        view_precision = self.P.T @ np.linalg.inv(self.Omega) @ self.P
        posterior_cov = np.linalg.inv(market_precision + view_precision)
        
        combined_return_vector = (
            market_precision @ self.Pi.values.reshape(-1, 1) 
            + self.P.T @ np.linalg.inv(self.Omega) @ self.Q
        )
        E = posterior_cov @ combined_return_vector
        
        return pd.Series(E.flatten(), index=self.Pi.index, name="Posterior Returns")
