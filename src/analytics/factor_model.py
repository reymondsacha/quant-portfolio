import numpy as np
import pandas as pd
from src.data.data_manager import DataManager
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger("FactorEngine")

class FactorEngine:
    def __init__(self):
        self.dm = DataManager()
        self.loading = None
        self.factor_returns = None
        self.pca_objects = None

    def compute_pca_factors(self, tickers, start_date, end_date, n_components):
        """
        """
        returns = self.dm.load_returns(tickers=tickers, start_date=start_date, end_date=end_date).dropna()
        
        std = returns.std()
        mean = returns.mean()
        std_returns = (returns - mean) / std

        self.pca_model = PCA(n_components=n_components)
        pca_factors = self.pca_model.fit_transform(std_returns)

        eigenvectors = self.pca_model.components_.T

        cov_matrix = returns.cov().values
        eigenvalues = np.diag(eigenvectors.T @ cov_matrix @ eigenvectors)

        factor_df = pd.DataFrame(
            pca_factors,
            index=returns.index,
            columns=[f"Factor_{i+1}" for i in range(n_components)]
        )

        self.loading = pd.DataFrame(
            eigenvectors,
            index=returns.columns,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )


        logger.info(f"PCA Factors computed. Top PC Variance: {eigenvalues[0]:.6f}")

        return factor_df, self.loading, eigenvalues, eigenvectors

    def get_explained_variance(self):
        if self.pca_model:
            return self.pca_model.explained_variance_ratio_
        return None

    def calculate_portfolio_betas(self, weights, eigenvectors):
        """
        Project portfolio weights onto the factor space.

        weights: pd.Series(index=tickers)
        eigenvectors: np.ndarray(index=tickers, columns=[PC1, PC2, ...])
        """
        w_normalized = weights / weights.sum()
        w_aligned = w_normalized.reindex(eigenvectors.index).fillna(0.0)
        betas = w_aligned.dot(eigenvectors)
        return betas





        
