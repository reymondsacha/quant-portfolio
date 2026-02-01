# Unit convention: daily vs annualized

All analytics code must be consistent: **either use daily returns/variance everywhere, or annual everywhere** in a single computation. This file documents the convention and where each module stands.

## Convention

| Quantity              | In optimizer (internal) | In Black–Litterman / callers |
|-----------------------|-------------------------|------------------------------|
| Returns (μ, Π, E[R])  | **Daily** (from `returns.mean()`) | Caller-dependent; **must match** covariance |
| Covariance (Σ)        | **Daily** (`returns.cov()`)       | Caller-dependent; **must match** returns |
| Risk-free rate        | **Annual** when `risk_free_rate_is_annual=True`; converted to daily inside | **Annual** (e.g. 0.04) |
| Volatility (σ)        | **Daily** (sqrt of daily variance) | Annual = daily × √252 when plotting |

**Rule:** In any single formula (e.g. Π = λ Σ w, or `solve(expected_returns=..., covariance=...)`), **returns and covariance must be in the same frequency** (both daily or both annual). Do not mix.

---

## MeanVarianceOptimizer (`optimizer.py`)

- **Input:** `returns: pd.DataFrame` — rows = dates, columns = assets. **Assumed to be daily returns** (e.g. from `prices.pct_change()`).
- **Internal state:**
  - `self.mu` = sample mean of returns → **daily**
  - `self.S` = sample covariance of returns → **daily**
- **Methods:**
  - `solve(expected_returns, covariance, ...)`: If you pass **annual** Π and **annual** Σ (e.g. from a BL notebook), both must be annual. If you omit one, the other is taken from `self.mu` / `self.S` (**daily**). So **do not** pass annual expected_returns with `covariance=None` (that would mix annual μ with daily Σ).
  - `get_optimal_weights(target_return, delta)`: `target_return` must be **daily** (same as `self.mu`).
  - `find_tangency_portfolio(risk_free_rate, risk_free_rate_is_annual)`: When `risk_free_rate_is_annual=True`, `risk_free_rate` is **annual** and is converted to daily to match `self.mu`. Returned `'return'` and `'volatility'` are **daily**.
  - `generate_frontier()`: `frontier_rets` and `frontier_vols` are **daily**.
- **Plotting:** Multiply returns by 252 and volatilities by √252 to get annualized for axes.

---

## BlackLittermanModel (`black_litterman.py`)

- **Internal:** `self.rf` — **annual** risk-free rate (e.g. 0.04).
- **Methods:**
  - `calculate_risk_aversion(benchmark_return, benchmark_var)`: Both must be **annual** (e.g. benchmark return 0.08, variance of market portfolio return in annual units).
  - `calculate_implied_returns(covariance, market_weights, risk_aversion)`: `covariance` and the returned Π are in the **same units** (e.g. both annual). Caller must pass annual Σ if they want annual Π.
  - `get_posterior_returns(pi, Sigma, P, Q, Omega, tau)`: Π, Σ, Omega, and returned E must all be in the **same units** (e.g. all annual).

---

## visualize_frontier.py

- **Input:** `df_returns` — **daily** (from `run_analytics`).
- **Frontier:** `vols`, `rets` from `generate_frontier()` are **daily**; multiplied by √252 and 252 for plot axes (annual).
- **Tangency:** `tangency['volatility']` and `tangency['return']` are **daily**; multiplied by √252 and 252 for the scatter point.

---

## Quick check before calling

1. **Optimizer built from daily returns** → `self.mu` and `self.S` are daily. Use daily target returns in `get_optimal_weights`; use `risk_free_rate_is_annual=True` in `find_tangency_portfolio`.
2. **BL / equilibrium check** → Use **annual** Σ and **annual** Π (and annual λ from annual benchmark return and annual variance). Pass both to `solve(expected_returns=Π, covariance=Σ)` so no internal daily data is used.
3. **Plotting** → Always convert to annual for axis labels: return × 252, volatility × √252.

---

## Verification

Run the unit-consistency checks from project root:

```bash
poetry run python -m src.analytics.verify_units
```

This checks that optimizer internal state looks like period (daily) data and that
common call patterns (e.g. solve() with annual Pi and Sigma) are consistent.
