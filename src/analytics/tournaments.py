import pandas as pd
import numpy as np
import yfinance as yf
from src.analytics.black_litterman import BlackLittermanModel
from src.analytics.mvo_optimizer import MeanVarianceOptimizer
from src.analytics.hrp_optimizer import HrpOptimizer
import logging

logger = logging.getLogger("PortfolioTournament")


class PortfolioTournament:
    def __init__(self, returns: pd.DataFrame, market_caps: pd.Series):
        self.returns = returns
        self.tickers = list(returns.columns)    
        self.market_caps = market_caps
        self.S = self.returns.cov()
        self.n_assets = len(self.tickers)

        self.mvo = MeanVarianceOptimizer(returns)    
        self.delta = self.mvo.calculate_optimal_delta()
        self.sigma_daily = self.mvo._apply_ledoit_wolf_shrinkage(self.delta)
        self.sigma_ann = self.sigma_daily * 252

        self.sigma_daily_df = pd.DataFrame(self.sigma_daily, index=self.tickers, columns=self.tickers)
        self.sigma_ann_df = pd.DataFrame(self.sigma_ann, index=self.tickers, columns=self.tickers)

        self.bl_helper = BlackLittermanModel()
        self.hrp = HrpOptimizer(returns=returns)


    def run_all(self, market_caps: pd.Series):

        results = {}
        # 1 - BENCHMARK : Market Cap (align to self.tickers, fill missing with 0)
        w_mkt = (
            (market_caps / market_caps.sum())
            .reindex(self.tickers)
            .fillna(0.0)
        )
        results["Market_cap"] = w_mkt

        # 2 - MVO : Tangency Portfolio
        mvo_res = self.mvo.find_tangency_portfolio(risk_free_rate=0.04, risk_free_rate_is_annual=True)
        results["MVO_tangency"] = mvo_res["weights"]
    
        # 3 - BLACK LITTERMAN

        self.bl_helper.add_view({"AAPL": 1.0}, 0.10, confidence_score=0.3)
        self.bl_helper.add_view({"AAPL": 0.5, "MSFT": 0.5, "GOOGL": -1.0}, 0.02, confidence_score=0.5)

        E = self.bl_helper.get_posterior_returns(
            self.sigma_ann_df,
            tau = 0.025,
            market_caps=market_caps,
            benchmark_return=0.08
        )

        results["Black_Litterman"] = self.mvo.solve(
            expected_returns=E, 
            covariance=self.sigma_ann_df, 
            risk_aversion=self.bl_helper.lambda_reg
        )

        # 4 - HRP : Risk topology
        sorted_indices = self.hrp.get_quasi_diag()
        results['HRP'] = self.hrp.get_rec_bisection(covariance_matrix=self.sigma_daily_df, sorted_indices=sorted_indices)

        return results

    def calculate_metrics(self, weights_df, cov_ann):
        rows = {}
        # Ensure cov_ann has rows/cols in self.tickers order
        if isinstance(cov_ann, pd.DataFrame):
            cov = cov_ann.reindex(index=self.tickers, columns=self.tickers).values
        else:
            cov = np.asarray(cov_ann)
        for col in weights_df.columns:
            w = weights_df[col].reindex(self.tickers).fillna(0.0).to_numpy(dtype=float)
            vol = np.sqrt(float(w.T @ cov @ w))
            hhi = np.sum(w**2)
            weighted_vol = np.sum(w * np.sqrt(np.diag(cov)))
            dr = weighted_vol / vol

            rows[col] = {"Vol": vol, "HHI": hhi, "DR": dr}
        return pd.DataFrame.from_dict(rows, orient="index")





