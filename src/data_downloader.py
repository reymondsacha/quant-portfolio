# src/data_downoader.py - REFACTORED (Stateless and SRP Compliant)

import pandas as pd
import yfinance as yf
from pandas.errors import EmptyDataError
from typing import List
import logging
import datetime  # Need to import datetime for validation
from src.utils import retry, SymbolNotFoundError


logger = logging.getLogger(__name__)


class YahooDownloader:
    """
    Stateless Yahoo Finance Data Downloader.

    Responsible only for fetching historical data for a single ticker and period,
    and returning the DataFrame payload. It performs defensive data cleaning.
    """

    # --- Validation Methods (KEEP THESE, they are excellent utility methods) ---
    @staticmethod
    def _validate_tickers(tickers: List[str]) -> None:
        # Note: Now used to validate the *input* list in fetch_history, if we designed it that way,
        # but since we call yf.download with a single string, we adapt the logic.
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("Tickers must be a non-empty list of strings.")
        if not all(isinstance(t, str) and t.strip() for t in tickers):
            raise ValueError("All tickers must be non-empty strings.")

    @staticmethod
    def _validate_date_format(date_str: str) -> None:
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"Date '{date_str}' is not in 'YYYY-MM-DD' format.") from e

    @staticmethod
    def _validate_date_order(start_date: str, end_date: str) -> None:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        if start > end:
            raise ValueError("start_date must be before or equal to end_date.")

    # ------------------------------------------------------------------------
    @retry(retries=3, delay=1, backoff=2)
    def fetch_history(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical market data for a SINGLE ticker and period using yfinance.

        Args:
            ticker (str): The single ticker symbol.
            start_date (str): Start date (inclusive) in 'YYYY-MM-DD' format.
            end_date (str): End date (inclusive) in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: The fetched data (or an empty DF if not found).
        """
        # Validate the specific call arguments
        # Since we removed __init__, validation moves to the fetch method.
        # Note: You can skip date validation if you're sure main.py provides good input,
        # but for robustness, it stays.
        self._validate_date_format(start_date)
        self._validate_date_format(end_date)
        self._validate_date_order(start_date, end_date)

        try:
            # yfinance call now uses the passed arguments
            df = yf.download(
                tickers=ticker,  # <--- Single ticker passed as string
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",  # Keep the MultiIndex (Field, Ticker) structure
            )

            # Defensive check: Ensure df is not None and is a DataFrame
            if df is None or not isinstance(df, pd.DataFrame):
                raise SymbolNotFoundError(ticker, "No data returned from Yahoo Finance.")

            if df.empty:
                # LOG WARNING before raising EmptyDataError for internal checks
                logger.warning(
                    f"No data returned from Yahoo Finance for {ticker}. Returning empty DataFrame."
                )
                raise SymbolNotFoundError(ticker,"No data returned from Yahoo Finance.") 

            # --- DEFENSIVE MULTIINDEX FIX (Kept from your final working code) ---
            if isinstance(df.columns, pd.MultiIndex):
                level0 = df.columns.levels[0]

                # Heuristic
                known_fields = ["Open", "High", "Low", "Close", "Volume"]
                if set(level0) & set(known_fields):
                    df.columns = df.columns.swaplevel(0, 1)
                df = df.sort_index(axis=1)
            # ------------------------------------------------------------------

            # If it's a single ticker, we ensure it's wrapped in a MultiIndex
            # to be consistent with the multi-ticker output structure,
            # as your original code intended (though this might be simplified later).
            # We keep the structure that passed your 100% tests.
            if not isinstance(df.columns, pd.MultiIndex):
                # This handles the case where yfinance gives simple columns for a single ticker
                df.columns = pd.MultiIndex.from_product(
                    [[ticker], df.columns], names=["Ticker", "Field"]
                )

            return df  # <--- KEY CHANGE: Returns the DataFrame payload

        except SymbolNotFoundError:
            # Return an empty DataFrame on failure so the orchestrator can continue
            logger.warning(
                f"Returning empty DataFrame for invalid symbol : {ticker}."
            )
            return pd.DataFrame()

        except Exception as e:
            # LOG ERROR before raising RuntimeError for critical failures
            logger.error(
                f"Critical failure fetching data for {ticker}: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Failed to fetch data from Yahoo Finance for {ticker}: {e}"
            ) from e


# --- The __init__ and save_to_parquet methods and the __main__ block are REMOVED ---
# --- The original class attributes (self.tickers, self.data) are REMOVED ---
