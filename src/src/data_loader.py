import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional
from pandas.errors import EmptyDataError
from datetime import datetime


class YahooDownloader:
    """
    Production-grade Yahoo Finance Data Downloader.

    This class fetches historical financial data for given tickers using yfinance,
    validates input dates, and allows saving the data in Parquet format.

    Attributes:
        tickers (List[str]): List of ticker symbols to fetch.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        data (Optional[pd.DataFrame]): Fetched data, indexed by date.
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str) -> None:
        """
        Initialize the YahooDownloader.

        Args:
            tickers (List[str]): List of ticker symbols.
            start_date (str): Start date (inclusive) in 'YYYY-MM-DD' format.
            end_date (str): End date (inclusive) in 'YYYY-MM-DD' format.

        Raises:
            ValueError: If dates are invalid or if tickers is empty.
        """
        self._validate_tickers(tickers)
        self._validate_date_format(start_date)
        self._validate_date_format(end_date)
        self._validate_date_order(start_date, end_date)

        self.tickers: List[str] = tickers
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.data: Optional[pd.DataFrame] = None

    @staticmethod
    def _validate_tickers(tickers: List[str]) -> None:
        """
        Validates the tickers input.

        Args:
            tickers (List[str]): List of ticker symbols.
        Raises:
            ValueError: If the list is empty or contains non-str elements.
        """
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("Tickers must be a non-empty list of strings.")
        if not all(isinstance(t, str) and t.strip() for t in tickers):
            raise ValueError("All tickers must be non-empty strings.")

    @staticmethod
    def _validate_date_format(date_str: str) -> None:
        """
        Validates the date format.

        Args:
            date_str (str): Date string.

        Raises:
            ValueError: If the date_str is not in 'YYYY-MM-DD' format.
        """
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except Exception as e:
            raise ValueError(
                f"Date '{date_str}' is not in 'YYYY-MM-DD' format."
            ) from e

    @staticmethod
    def _validate_date_order(start_date: str, end_date: str) -> None:
        """
        Validates that start_date is before or equal to end_date.

        Args:
            start_date (str): Start date.
            end_date (str): End date.

        Raises:
            ValueError: If start_date > end_date.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        if start > end:
            raise ValueError("start_date must be before or equal to end_date.")

    def fetch(self) -> None:
        """
        Fetch historical market data for defined tickers and period using yfinance.

        Returns:
            None

        Raises:
            RuntimeError: If no data was returned for any given ticker.
        """
        try:
            df = yf.download(
                tickers=" ".join(self.tickers),
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
            if df.empty:
                raise EmptyDataError("No data returned from Yahoo Finance.")

            # Restructure DataFrame: rows -> date, columns -> MultiIndex (Ticker, Field)
            if len(self.tickers) == 1:
                # Consistent MultiIndex structure even for single ticker
                df.columns = pd.MultiIndex.from_product([self.tickers, df.columns])

            self.data = df
        except EmptyDataError as e:
            raise RuntimeError("No data found for the specified tickers and dates.") from e
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from Yahoo Finance: {e}") from e

    def save_to_parquet(self, filename: str) -> None:
        """
        Save the fetched data to Parquet format.

        Args:
            filename (str): Destination file path (must end with .parquet).

        Raises:
            ValueError: If data has not been fetched yet.
            RuntimeError: If saving fails.
        """
        if self.data is None or self.data.empty:
            raise ValueError(
                "No data to save. Call fetch() successfully before saving."
            )
        if not filename.endswith(".parquet"):
            raise ValueError("Filename must end with .parquet")
        try:
            self.data.to_parquet(filename, engine="pyarrow")
        except Exception as e:
            raise RuntimeError(f"Failed to save data to Parquet: {e}") from e


if __name__ == "__main__":
    # Test the downloader
    print("Testing YahooDownloader...")
    
    # 1. Define parameters
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start = "2020-01-01"
    end = "2023-12-31"
    
    # 2. Instantiate and fetch
    downloader = YahooDownloader(tickers, start, end)
    print(f"Fetching data for {tickers}...")
    downloader.fetch()
    
    # 3. Inspect data
    print("Data fetched successfully!")
    print(f"Shape: {downloader.data.shape}")
    print(downloader.data.head())
    
    # 4. Test Save
    downloader.save_to_parquet("market_data.parquet")
    print("Saved to market_data.parquet")
