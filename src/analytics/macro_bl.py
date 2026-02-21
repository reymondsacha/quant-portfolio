import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.analytics.factor_model import FactorEngine
from src.analytics.black_litterman import BlackLittermanModel

class MacroBLModel:
    """
    Orchestrates Bayesian view injection in PCA factor space.
    """

    def __init__(self, returns: pd.DataFrame, covariance: pd.DataFrame, market_caps: pd.Series):
        self.covariance = covariance
        self.market_caps = market_caps
        self.returns = returns
        self.tickers = returns.columns
        self.start_date = returns.index[0]
        self.end_date = returns.index[-1]
        self.bl_helper = BlackLittermanModel(market_caps=self.market_caps)
        self.factor_engine = FactorEngine()
        self.injected_views = []
        self.factors_computed = False

    def ensure_factors(self, n_components=5):
        """
        Load PCA factors if not already computed.
        """
        if not self.factors_computed:
            _, self.loading, self.eigenvalues, self.eigenvectors = self.factor_engine.compute_pca_factors(
                tickers=list(self.tickers), 
                start_date=self.start_date, 
                end_date=self.end_date, 
                n_components=n_components
            )
            self.factors_computed = True

    def inject_factor_view(self, factor_index, relative_return, confidence, tau=0.025):
        """
        Translates a Factor-level thesis into a Bayesion Prior shift.
        """
        self.ensure_factors()
        eigenvalue = self.eigenvalues[factor_index]
        eigenvector = self.eigenvectors[:, factor_index]
        
        conf = np.clip(confidence, 1e-6, 0.999)
        omega_k = ((1 - confidence) / confidence) * tau * eigenvalue

        view = {
            'P_row': eigenvector,
            'Q_val': relative_return,
            'Omega_val': omega_k
        }

        self.injected_views.append(view)

    def compute_posterior(self, tau=0.025):
        """
        Executes the Black Litterman master formula to compute the posterior returns.
        """

        if not self.injected_views:
            raise ValueError("No views injected yet. Use inject_factor_view() first.")

        self.P = np.array([v['P_row'] for v in self.injected_views])
        self.Q = np.array([v['Q_val'] for v in self.injected_views]).reshape(-1, 1)
        self.Omega = np.diag([v['Omega_val'] for v in self.injected_views])
        
        Pi = self.bl_helper.calculate_implied_returns(covariance=self.covariance)

        sigma_np = self.covariance.to_numpy()
        Pi_np = Pi.to_numpy().reshape(-1, 1)

        market_precision = np.linalg.inv(tau * sigma_np)
        view_precision = self.P.T @ np.linalg.inv(self.Omega) @ self.P

        posterior_cov = np.linalg.inv(market_precision + view_precision)

        combined_return_vector = (
            market_precision @ Pi_np + view_precision @ self.P.T @ self.Q
        )

        E = posterior_cov @ combined_return_vector

        return pd.Series(E.flatten(), index=self.tickers, name="Posterior Factor-Adjusted Returns")
        



