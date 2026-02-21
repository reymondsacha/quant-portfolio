import pandas as pd
import logging
import numpy as np
from src.data.data_manager import DataManager
from src.analytics.mvo_optimizer import MeanVarianceOptimizer

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    handlers=[
        logging.FileHandler("logs/analytics.log", mode="w"),
        logging.StreamHandler(),  # Console output
    ],
)


def run_analytics():
    # 1. Initialize your existing DataManager
    # Assuming your data is in "data/raw"
    dm = DataManager(base_dir="data/raw")

    # 2. Use your existing discovery method
    tickers = dm.list_existing_tickers()
    logging.info(f"Found {len(tickers)} tickers: {tickers}")

    all_data = []
    for ticker in tickers:
        try:
            # 3. Use your existing loading logic
            df_temp = dm.load_ticker(ticker)

            # Reset index to ensure 'Date' is available as a column if it was the index
            if "Date" not in df_temp.columns:
                df_temp = df_temp.reset_index()

            # Manual injection of ticker column (since Hive partitioning wasn't detected automatically)
            df_temp["ticker"] = ticker
            all_data.append(df_temp[["Date", "Close", "ticker"]])

        except Exception as e:
            print(f"Error loading {ticker}: {e}")

    if not all_data:
        raise RuntimeError("No data loaded")

    # 4. Prepare the matrix for math
    df_raw = pd.concat(all_data, ignore_index=True)
    df_prices = df_raw.pivot(index="Date", columns="ticker", values="Close")

    # Forward fill to handle mismatched market holidays
    df_returns = df_prices.sort_index().ffill().pct_change().dropna()
    logging.info(f"Optimization Matrix Ready: {df_returns.shape}")

    coverage = df_prices.notna().mean()
    min_coverage = coverage.min() if isinstance(coverage, pd.Series) else float(coverage)
    if min_coverage < 0.8:
        logging.warning("Warning: Some assets have less than 80% data coverage.")

    # 5. Run the Optimizer
    optimizer = MeanVarianceOptimizer(df_returns)
    tangency = optimizer.find_tangency_portfolio(
        risk_free_rate=0.04, risk_free_rate_is_annual=True
    )
    target_daily = tangency["return"]
    compare_weights(optimizer, target_daily, tangency_weights=tangency["weights"])
    logging.info(f"Optimal Delta: {optimizer.delta:.2f}")

    # 1. Reconstruct the Gradient of the Risk at the solution (use tangency weights)
    # Risk Gradient = Sigma * w
    w_array = np.array([tangency["weights"][ticker] for ticker in optimizer.tickers])
    grad_risk = np.dot(optimizer._apply_ledoit_wolf_shrinkage(), w_array)

    # 2. Check Complementary Slackness: w_i * z_i should be 0
    # We can't see z directly easily, but we know that if w_i > 0,
    # then the marginal risk must be a linear combination of return and budget.
    # For all assets where weight > 1e-4:
    active_indices = np.where(w_array > 1e-4)[0]

    if len(active_indices) >= 2:
        # Pick two active assets (i and j)
        i, j = int(active_indices[0]), int(active_indices[1])
        # At the optimum, for any two active assets:
        # (Marginal Risk_i - Marginal Risk_j) should be proportional to (mu_i - mu_j)
        risk_diff = grad_risk[i] - grad_risk[j]
        mu_diff = optimizer.mu[i] - optimizer.mu[j]
        logging.info(f"KKT Verification - Risk/Mu Ratio: {risk_diff / mu_diff:.6f}")

    # Compute the condition number of the Raw vs Shrunk matrix
    cond_raw = np.linalg.cond(optimizer.S)
    cond_shrunk = np.linalg.cond(
        (1 - optimizer.delta) * optimizer.S
        + optimizer.delta
        * (np.trace(optimizer.S) / optimizer.n_assets)
        * np.eye(optimizer.n_assets)
    )
    logging.info(f"Condition Number (Raw): {cond_raw:.2f}")
    logging.info(f"Condition Number (Shrunk): {cond_shrunk:.2f}")

    # Annualized Portfolio Volatility: sqrt(w.T * Sigma * w) * sqrt(252)
    w_dict = optimizer.get_optimal_weights(target_daily)
    w_array = np.array([w_dict[ticker] for ticker in optimizer.tickers])
    sigma_shrunk = optimizer._apply_ledoit_wolf_shrinkage()
    port_variance = np.dot(w_array.T, np.dot(sigma_shrunk, w_array))
    port_vol = np.sqrt(port_variance) * np.sqrt(252)
    logging.info(f"Expected Annual Volatility: {port_vol:.2%}")

    return df_returns, optimizer, optimizer.delta


def compare_weights(optimizer, target_return, tangency_weights=None):
    """
    Print weight comparison. If tangency_weights is provided (e.g. from
    find_tangency_portfolio), the Shrunk column uses tangency weights so the
    table matches the tangency star on the frontier plot.
    """
    # 1. Raw Markowitz (δ = 0)
    try:
        sigma_raw_df = pd.DataFrame(optimizer.S, index=optimizer.tickers, columns=optimizer.tickers)
        w_raw = optimizer.get_optimal_weights(target_return, covariance=sigma_raw_df)
    except ValueError:
        w_raw = "FAILED TO CONVERGE"
        logging.error("Raw Weights: Failed to converge")

    # 2. Tangency / min-var at target (use tangency weights when at tangency return)
    if tangency_weights is not None:
        w_shrunk = {t: float(tangency_weights[t]) for t in optimizer.tickers}
    else:
        w_shrunk = optimizer.get_optimal_weights(target_return)

    print(f"\n{'Ticker':<10} | {'Raw Weights (%)':<15} | {'Shrunk/Tangency (%)':<15}")
    print("-" * 50)
    for t in optimizer.tickers:
        if isinstance(w_raw, (dict, pd.Series)) and t in w_raw:
            raw_val = f"{w_raw[t]:.2%}"
        else:
            raw_val = "N/A"
        shrunk_val = f"{w_shrunk[t]:.2%}"
        print(f"{t:<10} | {raw_val:<15} | {shrunk_val:<15}")


if __name__ == "__main__":
    # 1. Run the math
    result = run_analytics()
    if result is None:
        raise RuntimeError("run_analytics() returned None")
    df_returns, optimizer, delta = result

    # 2. Import the visualization function here
    from src.analytics.visualize_frontier import plot_comprehensive_frontier

    # 3. Execute the plots
    logging.info("Generating Frontier Visualizations...")
    plot_comprehensive_frontier(df_returns, optimizer, delta)
