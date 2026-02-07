import numpy as np
import pytest
import pandas as pd
from src.analytics.mvo_optimizer import MeanVarianceOptimizer


@pytest.fixture
def optimizer():
    """
    Provides a robust MeanVarianceOptimizer instance with 6 stocks.
    """
    np.random.seed(42)  # For deterministic tests

    # Generate 100 days of data for 6 stocks
    # This ensures T > N and creates a realistic feasible region
    tickers = ["AAPL", "MSFT", "GOOGL", "JNJ", "PG", "KO"]

    # Create random returns with different means and volatilities
    data = {}
    for t in tickers:
        mean_ret = np.random.uniform(0.0001, 0.001)  # 0.01% to 0.1%
        vol = np.random.uniform(0.01, 0.03)
        data[t] = np.random.normal(mean_ret, vol, 100)

    df_returns = pd.DataFrame(data)
    return MeanVarianceOptimizer(df_returns)


def test_kkt_constraints(optimizer):
    target = 0.0016
    weights = optimizer.get_optimal_weights(target_return=target, delta=0.2)
    w_array = weights.reindex(optimizer.tickers).to_numpy(dtype=float)
    mu = optimizer.mu
    returns = np.dot(w_array, mu)
    assert returns >= (target - 1e-7)

    assert np.isclose(np.sum(w_array), 1.0, atol=1e-5)

    assert np.all(w_array >= -1e-7)


def test_ledoit_wolf_shrinkage(optimizer):
    delta = optimizer.calculate_optimal_delta()
    sigma_stable = optimizer._apply_ledoit_wolf_shrinkage(delta)
    assert np.allclose(sigma_stable, sigma_stable.T)


