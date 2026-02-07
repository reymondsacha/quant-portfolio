import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


class HrpOptimizer:
    def __init__(self, returns: pd.DataFrame, frequency: int = 252):
        """
        Hierarchical Risk Parity optimizer.

        Units:
            - `returns` are assumed to be in a single-period frequency (e.g. daily).
            - `self.frequency` is the number of periods per year (default 252) and
              is provided for documentation and diagnostics; HRP itself is scale
              invariant with respect to covariance units.
        """
        self.returns = returns
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)
        self.frequency: int = int(frequency)
        self.dist_matrix = None
        self.linkage_matrix = None

    def get_correlation_distance(self) -> pd.DataFrame:
        """
        Get the correlation distance matrix.
        """
        corr = self.returns.corr()
        dist = np.sqrt(np.clip(0.5 * (1 - corr), 0.0, None))
        self.dist_matrix = dist
        return dist

    def compute_linkage(self, method="single"):
        """
        Computes the hierarchical linkage matrix.
        """
        if self.dist_matrix is None:
            self.get_correlation_distance()
        assert self.dist_matrix is not None
        condensed_dist = squareform(self.dist_matrix, checks=False)
        self.linkage_matrix = sch.linkage(condensed_dist, method=method)
        return self.linkage_matrix

    def get_quasi_diag(self):
        """
        Reorders the covariance matrix so that similar assets are grouped together.
        'Tree traversal' step
        """
        if self.linkage_matrix is None:
            self.compute_linkage()
        link = self.linkage_matrix
        assert link is not None

        sort_indices = [2 * self.n_assets - 2]
        while max(sort_indices) >= self.n_assets:
            new_indices = []
            for idx in sort_indices:
                if idx >= self.n_assets:
                    # this is a cluster, not an asset
                    left_child = int(link[idx - self.n_assets, 0])
                    right_child = int(link[idx - self.n_assets, 1])

                    new_indices.append(left_child)
                    new_indices.append(right_child)
                else:
                    new_indices.append(idx)
            sort_indices = new_indices

        return sort_indices

    def get_sorted_tickers(self):
        """
        Returns the lit of tickers in quasi-diagonal order.
        """
        indices = self.get_quasi_diag()
        return [self.tickers[i] for i in indices]


    def get_cluster_var(self, covariance_matrix, cluster_indices):
        """
        Computes the variance of a cluster of assets.
        """
        new_cov = covariance_matrix.iloc[cluster_indices, cluster_indices].values
        diag_array = np.diag(new_cov)
        inv_var = 1/diag_array
        weights = inv_var/np.sum(inv_var)
        return np.dot(weights.T, np.dot(new_cov,weights))

    def get_rec_bisection(self, covariance_matrix, sorted_indices):
        """
        Recursive bisection of the covariance matrix.

        Returns:
            pd.Series: HRP weights indexed by **sorted tickers** in quasi-diagonal
            order corresponding to `sorted_indices`. Use `get_sorted_tickers()`
            if you need the explicit ticker order.
        """
        # Ensure covariance has rows/columns in self.tickers order (sorted_indices are
        # positional indices into this order). Prevents wrong-ticker assignment when
        # caller passes a DataFrame with different column order.
        if isinstance(covariance_matrix, pd.DataFrame):
            cov = covariance_matrix.reindex(
                index=self.tickers, columns=self.tickers
            ).astype(float)
            if cov.isna().any().any():
                raise ValueError(
                    "covariance_matrix missing required tickers or wrong column order."
                )
        else:
            cov = pd.DataFrame(
                covariance_matrix, index=self.tickers, columns=self.tickers
            )
        weights = pd.Series(1.0, index=sorted_indices)

        def recurse(cluster_indices):
            if len(cluster_indices) <= 1:
              return

            mid_point = len(cluster_indices)//2

            left_indices = cluster_indices[:mid_point]
            right_indices = cluster_indices[mid_point:]

            var_left = self.get_cluster_var(covariance_matrix=cov, cluster_indices=left_indices)
            var_right = self.get_cluster_var(covariance_matrix=cov, cluster_indices=right_indices)
            alpha = 1 - (var_left / (var_left + var_right))
            weights.loc[left_indices] *= alpha
            weights.loc[right_indices] *= 1 - alpha

            recurse(left_indices)
            recurse(right_indices)
        
        recurse(sorted_indices)

        if np.abs(np.sum(weights) - 1) > 1e-6:
            raise ValueError("Weights do not sum to 1")

        # Map from integer indices to ticker labels, preserving quasi-diagonal order.
        weights.index = [self.tickers[i] for i in weights.index]

        return weights


    