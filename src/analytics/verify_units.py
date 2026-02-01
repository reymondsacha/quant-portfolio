"""
Unit consistency checks for src/analytics: detect daily vs annual mix-ups.

Run from project root: python -m src.analytics.verify_units
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Typical magnitude bounds (order of magnitude)
DAILY_RETURN_ABS_MAX = 0.05   # |daily return| rarely > 5%
ANNUAL_RETURN_ABS_MAX = 2.0   # |annual return| rarely > 200%
DAILY_VAR_MAX = 0.01          # daily variance rarely > 1%
ANNUAL_VAR_MAX = 5.0           # annual variance can be large


def _looks_daily_returns(series: pd.Series | np.ndarray) -> bool:
    """True if magnitudes look like daily returns (e.g. mean abs < 0.02)."""
    x = np.asarray(series).ravel()
    return np.abs(x).max() <= DAILY_RETURN_ABS_MAX and np.median(np.abs(x)) <= 0.01


def _looks_annual_returns(series: pd.Series | np.ndarray) -> bool:
    """True if magnitudes look like annual returns (e.g. typical 0.05--0.5)."""
    x = np.asarray(series).ravel()
    return np.any(np.abs(x) >= 0.01) and np.abs(x).max() <= ANNUAL_RETURN_ABS_MAX


def _looks_daily_covariance(matrix: np.ndarray | pd.DataFrame) -> bool:
    """True if diagonal looks like daily variance (small)."""
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    diag = np.diag(m)
    return np.all(diag >= 0) and np.max(diag) <= DAILY_VAR_MAX


def _looks_annual_covariance(matrix: np.ndarray | pd.DataFrame) -> bool:
    """True if diagonal looks like annual variance (larger)."""
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    diag = np.diag(m)
    return np.all(diag >= 0) and np.max(diag) >= 0.01 and np.max(diag) <= ANNUAL_VAR_MAX


def check_optimizer_units(optimizer) -> list[str]:
    """
    Check MeanVarianceOptimizer internal state and return list of warnings/errors.
    Assumes optimizer was built from a returns DataFrame (e.g. daily).
    """
    issues = []

    # self.mu should look like period returns (e.g. daily)
    if not _looks_daily_returns(optimizer.mu):
        if _looks_annual_returns(optimizer.mu):
            issues.append(
                "MeanVarianceOptimizer.mu looks ANNUAL (e.g. 0.1) but optimizer "
                "is assumed to be built from period returns (e.g. daily). "
                "Did you pass annualized returns as the DataFrame?"
            )
        else:
            issues.append(
                "MeanVarianceOptimizer.mu has unexpected scale. "
                "Expected period returns (e.g. daily ~1e-3). Check input returns."
            )

    # self.S should look like period covariance (e.g. daily)
    if not _looks_daily_covariance(optimizer.S):
        if _looks_annual_covariance(optimizer.S):
            issues.append(
                "MeanVarianceOptimizer.S looks ANNUAL covariance but optimizer "
                "is assumed to be built from period returns. Do not mix with self.mu (daily)."
            )
        else:
            issues.append(
                "MeanVarianceOptimizer.S has unexpected scale. "
                "Expected period covariance (e.g. daily). Check input returns."
            )

    return issues


def check_solve_inputs(
    expected_returns: pd.Series | None,
    covariance: pd.DataFrame | None,
    optimizer_mu_scale: str = "daily",
) -> list[str]:
    """
    Check that expected_returns and covariance are consistent (same frequency).
    optimizer_mu_scale: "daily" or "annual" when both are None (internal use).
    """
    issues = []

    if expected_returns is None and covariance is None:
        return issues  # Both from optimizer; assume consistent

    if expected_returns is not None and covariance is None:
        # Covariance will come from optimizer (e.g. daily)
        if _looks_annual_returns(expected_returns) and optimizer_mu_scale == "daily":
            issues.append(
                "solve(): expected_returns looks ANNUAL but covariance will be "
                "taken from optimizer (daily). Do not mix. Pass covariance=Sigma_ann "
                "if expected_returns is annual."
            )
        return issues

    if expected_returns is None and covariance is not None:
        # Expected returns from optimizer (e.g. daily)
        if _looks_annual_covariance(np.asarray(covariance)) and optimizer_mu_scale == "daily":
            issues.append(
                "solve(): covariance looks ANNUAL but expected_returns will be "
                "taken from optimizer (daily). Do not mix. Pass expected_returns "
                "in same frequency as covariance."
            )
        return issues

    # Both provided: must be same scale
    ret_daily = _looks_daily_returns(expected_returns)
    ret_annual = _looks_annual_returns(expected_returns)
    cov_daily = _looks_daily_covariance(covariance)
    cov_annual = _looks_annual_covariance(covariance)

    if ret_daily and cov_annual:
        issues.append(
            "solve(): expected_returns looks DAILY but covariance looks ANNUAL. "
            "Use same frequency for both."
        )
    if ret_annual and cov_daily:
        issues.append(
            "solve(): expected_returns looks ANNUAL but covariance looks DAILY. "
            "Use same frequency for both."
        )

    return issues


def check_black_litterman_inputs(
    benchmark_return: float,
    benchmark_var: float,
    covariance: pd.DataFrame | None,
    implied_returns: pd.Series | None,
) -> list[str]:
    """Check BL inputs/outputs for unit consistency (all annual expected)."""
    issues = []

    if np.abs(benchmark_return) > 0.5 and np.abs(benchmark_return) < 10:
        pass  # Could be annual (0.08)
    elif np.abs(benchmark_return) < 0.01:
        issues.append(
            "BlackLitterman: benchmark_return looks DAILY (e.g. 0.0005). "
            "calculate_risk_aversion expects annual (e.g. 0.08)."
        )

    if covariance is not None and implied_returns is not None:
        cov_annual = _looks_annual_covariance(covariance)
        pi_daily = _looks_daily_returns(implied_returns)
        pi_annual = _looks_annual_returns(implied_returns)
        if cov_annual and pi_daily:
            issues.append(
                "BlackLitterman: covariance looks ANNUAL but implied returns (Pi) "
                "look DAILY. calculate_implied_returns expects both in same frequency (e.g. annual)."
            )
        if not cov_annual and pi_annual:
            issues.append(
                "BlackLitterman: covariance looks DAILY but implied returns look ANNUAL. "
                "Use same frequency (e.g. annual Sigma and annual Pi)."
            )

    return issues


def run_all_checks(optimizer=None, df_returns: pd.DataFrame | None = None):
    """
    Run unit checks. If optimizer/df_returns not provided, use synthetic daily data.
    Returns (passed: bool, messages: list[str]).
    """
    if df_returns is None:
        np.random.seed(42)
        n_days, n_assets = 252, 5
        df_returns = pd.DataFrame(
            np.random.randn(n_days, n_assets) * 0.01,
            columns=[f"A{i}" for i in range(n_assets)],
        )

    if optimizer is None:
        from src.analytics.optimizer import MeanVarianceOptimizer
        optimizer = MeanVarianceOptimizer(df_returns)

    messages = []
    issues = check_optimizer_units(optimizer)
    messages.extend(issues)

    # Check solve() with annual Pi + annual Sigma (as in BL notebook) is consistent
    n = len(optimizer.tickers)
    pi_ann = pd.Series(np.full(n, 0.10), index=optimizer.tickers)
    sigma_ann = pd.DataFrame(0.05 * np.eye(n) + 0.01, index=optimizer.tickers, columns=optimizer.tickers)
    issues_solve = check_solve_inputs(pi_ann, sigma_ann, optimizer_mu_scale="daily")
    messages.extend(issues_solve)

    # Check BL risk_aversion with annual inputs
    from src.analytics.black_litterman import BlackLittermanModel
    bl = BlackLittermanModel(risk_free_rate=0.04)
    lam = bl.calculate_risk_aversion(benchmark_return=0.08, benchmark_var=0.04)
    pi_bl = bl.calculate_implied_returns(sigma_ann, pd.Series(1.0 / n, index=optimizer.tickers), lam)
    issues_bl = check_black_litterman_inputs(0.08, 0.04, sigma_ann, pi_bl)
    messages.extend(issues_bl)

    passed = len(messages) == 0
    if passed:
        messages.insert(0, "All unit consistency checks passed.")
    return passed, messages


if __name__ == "__main__":
    passed, messages = run_all_checks()
    for m in messages:
        print(m)
    raise SystemExit(0 if passed else 1)
