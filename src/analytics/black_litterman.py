import numpy as np
import pandas as pd

class BlackLittermanModel:
    def __init__(self, risk_free_rate: float = 0.04) -> None:
        """
        Black–Litterman helper methods for computing the market prior.

        Units: self.rf is treated as annual (e.g. 0.04). All other inputs/outputs
        (benchmark_return, benchmark_var, covariance, implied returns) must be
        in the same frequency as each other (e.g. all annual). See UNITS.md.
        """
        self.rf: float = float(risk_free_rate)
        self.views = []
    
    def get_market_weights(self, market_caps: pd.Series) -> pd.Series:
        """
        Convert raw market caps into a weight vector (summing to 1)
        """
        return market_caps / market_caps.sum()

    def calculate_risk_aversion(self, benchmark_return: float, benchmark_var: float) -> float:
        """
        Estimate lambda (Risk Aversion) from benchmark performance.

        Units: benchmark_return and benchmark_var must be annual. self.rf is annual.
        """
        if benchmark_var <= 0:
            raise ValueError("benchmark_var must be positive to compute risk aversion.")
        return float((benchmark_return - self.rf) / benchmark_var)

    def calculate_implied_returns(
        self,
        covariance: pd.DataFrame,
        market_weights: pd.Series,
        risk_aversion: float
    ) -> pd.Series:
        """
        Derive Equilibrium Returns: Pi = lambda * Sigma * w_mkt.

        Units: covariance and returned Pi are in the same frequency (e.g. both annual).
        """
        if not isinstance(covariance, pd.DataFrame):
            raise TypeError("covariance must be a pandas DataFrame with matching index/columns.")
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be square.")
        if not covariance.index.equals(covariance.columns):
            raise ValueError("covariance index and columns must match (same assets, same order).")

        w_aligned = market_weights.reindex(covariance.index).astype(float)
        if w_aligned.isna().any():
            missing = list(w_aligned[w_aligned.isna()].index)
            raise ValueError(f"market_weights missing assets required by covariance: {missing}")

        # Pandas-safe multiplication with explicit alignment:
        # Pi = lambda * Sigma * w_mkt
        pi = float(risk_aversion) * (covariance @ w_aligned)
        pi.name = "Implied Returns"
        return pi

    def add_view(self, weights: dict[str, float], target_return: float, confidence_score: float):
        self.views.append({
            'weights':weights,
            'q':target_return,
            'confidence':confidence_score
        })

    def _build_matrices(self, tickers: list[str]):
            K = len(self.views)
            P = np.zeros((K,(len(tickers))))
            Q = np.zeros((K,1))

            for i, view in enumerate(self.views):
                for ticker, weight in view['weights'].items():
                    try:
                        idx = tickers.index(ticker)
                    except ValueError:
                        raise ValueError(f"Ticker {ticker} not found in tickers list")
                    P[i,idx] = weight
                Q[i, 0] = view['q']

            return P, Q

    def compute_omega(self, P, sigma, tau):
        P = np.array(P)
        k = P.shape[0]
        omega_diag = []
        for i in range(k):
            confidence_score = self.views[i]['confidence']
            try :
                omega_diag.append(((1 - confidence_score )/ confidence_score) * tau * P[i] @ sigma @ P[i].T)         
            except ZeroDivisionError:
                raise ValueError("confidence_score can't be 0 or 1")
        return np.diag(omega_diag)

    def get_posterior_returns(self, pi, Sigma, P, Q, Omega, tau):
        """
        Units: pi, Sigma, Omega and returned E must all be in the same frequency (e.g. annual).
        """
        market_precision = np.linalg.inv(tau*Sigma)
        view_precision = P.T @ np.linalg.inv(Omega) @ P
        posterior_cov = np.linalg.inv(market_precision + view_precision)
        combined_return_vector = market_precision @ pi.values.reshape(-1,1) + P.T @ np.linalg.inv(Omega) @ Q
        E = posterior_cov @ combined_return_vector
        return pd.Series(E.flatten(), index=pi.index)



