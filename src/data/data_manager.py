# src/data_manager.py

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, cast, Optional

logger = logging.getLogger(__name__)


class DataManager:
    """
    Handles all I/O operations for ticker data storage and retrieval.
    Implements ATOMIC WRITES to ensure data integrity during crashes.
    """

    def __init__(self, base_dir: str = "data/raw") -> None:
        self.base_dir: Path = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_ticker(self, ticker: str, df: pd.DataFrame) -> Path:
        """
        Save a ticker's dataframe to Parquet format safely.

        Strategy:
        1. Flatten MultiIndex columns to single-level (storage format)
        2. Write to {ticker}.parquet.tmp
        3. Rename to {ticker}.parquet (Atomic Operation)

        Note: This method flattens MultiIndex columns before saving, as storage
        uses Hive partitioning where ticker identity is encoded in the directory
        structure. The input DataFrame may have MultiIndex columns (e.g., from
        fetch_history), but stored data always has single-level columns.

        Returns:
            Path: The absolute path to the saved file.
        """
        if df.empty:
            raise ValueError(f"Cannot save empty dataframe for ticker '{ticker}'.")

        # Flatten MultiIndex columns to single-level for storage
        # Since ticker identity is encoded in directory structure, we don't need
        # MultiIndex in the stored data. This simplifies downstream analysis.
        # Note: fetch_history always returns (Ticker, Field) structure, so we extract level 1.
        df_to_save = df.copy()
        if isinstance(df_to_save.columns, pd.MultiIndex):
            # fetch_history guarantees (Ticker, Field) structure, so extract Field level
            df_to_save.columns = df_to_save.columns.get_level_values(1)

        monthly_groups = df_to_save.groupby(pd.Grouper(freq="ME"))

        for period, chunk in monthly_groups:
            if chunk.empty:
                continue
            # 1. Extract partition keys (ensure the index is datetime-like for type checkers)
            #
            # Pyright (and pandas stubs) treat `DataFrame.index` as a generic `pd.Index`,
            # so `df.index[0].year` / `.month` is flagged unless we explicitly narrow.
            dt_index: pd.DatetimeIndex = pd.DatetimeIndex(pd.to_datetime(chunk.index))
            raw_start = dt_index.min()
            if raw_start is pd.NaT:
                raise ValueError(
                    f"Ticker '{ticker}' has an invalid datetime index (NaT encountered)."
                )
            start_ts: pd.Timestamp = cast(pd.Timestamp, raw_start)
            year: int = int(start_ts.year)
            month: str = f"{int(start_ts.month):02d}"

            # 2. Construct the partition Directory (Hive style)
            partition_dir: Path = (
                self.base_dir / ticker / f"year={year}" / f"month={month}"
            )

            file_path: Path = partition_dir / "data.parquet"
            temp_path: Path = partition_dir / "data.parquet.tmp"

            # Create all necessary parent directories
            partition_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 1. Write to temporary file
                chunk.to_parquet(
                    temp_path,
                    engine="pyarrow",
                    compression="snappy",
                    index=True,  # Explicitly preserve the Date index
                )

                # 2. Atomic Rename (The Swap)
                temp_path.replace(file_path)

            except Exception as e:
                # Cleanup garbage if write failed
                if temp_path.exists():
                    temp_path.unlink()

                raise RuntimeError(
                    f"Failed to save ticker '{ticker}' to {file_path}: {e}"
                ) from e
        return self.base_dir / ticker

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Load a ticker's data from Parquet format.

        The pyarrow engine automatically discovers and reads all files
        within the ticker's root directory, handling partition discovery.

        Returns:
            pd.DataFrame: DataFrame with single-level columns (MultiIndex is
            flattened during save_ticker, so loaded data always has simple columns).
        """

        # 1. Define the ticket root directory path
        ticker_root_path: Path = self.base_dir / ticker

        # 2. Check if the ticker root directory exists
        if not ticker_root_path.is_dir():
            raise FileNotFoundError(
                f"Ticker data directory '{ticker}' not found at {ticker_root_path}."
            )

        try:
            # 3. load the partitioned dataset
            # Pointing pandas.read_parquet to a directory tells the pyarrow engine
            # to load the entire dataset, automatically handling the partition columns.
            df = pd.read_parquet(ticker_root_path, engine="pyarrow")
            return df

        except Exception as e:
            raise RuntimeError(f"Corrupt data for '{ticker}': {e}") from e

    def list_existing_tickers(self) -> List[str]:
        """List all tickers saved as root directories in the partitioned structure."""
        return sorted(
            [
                f.name
                for f in self.base_dir.iterdir()
                if f.is_dir()
                and not f.name.startswith(".")
                and f.name != "__pycache__"
                and f.name != "0_metadata"  # Metadata cache, not a ticker
            ]
        )

    def load_returns(self, tickers: list[str], start_date: Optional[str] = None, end_date: Optional[str] = None, log_returns: bool = False) -> pd.DataFrame:
        """
        Load a ticker's returns DataFrame from Parquet format.
        """
        if not tickers:
            raise ValueError(
                "No tickers to load. Check that base_dir points to data/raw and contains "
                "ticker subdirectories (e.g. run from project root or use base_dir=<path-to-data/raw>)."
            )
        close_frames: list[pd.Series] = []
        for t in tickers:
            try:
                df_t = self.load_ticker(t)
            except (FileNotFoundError, RuntimeError) as e:
                raise FileNotFoundError(f"Make sure {t} is in the data/raw folder") from e
            df_t = df_t.copy()

            if not isinstance(df_t.index, pd.DatetimeIndex):
                df_t.index = pd.to_datetime(df_t.index)

            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = df_t.columns.get_level_values(-1)

            

            if "Close" not in df_t.columns:
                raise ValueError(f"Expected a 'Close' column for {t}, got columns={list(df_t.columns)}")
            close_frames.append(df_t["Close"].rename(t))

        prices = pd.concat(close_frames, axis=1).sort_index().ffill()
        
        if start_date is not None:
            prices = prices.loc[prices.index >= pd.to_datetime(start_date)]

        if end_date is not None:
            prices = prices.loc[prices.index <= pd.to_datetime(end_date)]

        if not log_returns:
            returns = prices.pct_change()
        else:
            returns = np.log(prices / prices.shift(1))

        return returns.dropna(how="any")

    def save_metadata(self, metadata_list: list[dict]) -> Path:
        """
        Takes a list of dictionaries, converts to DataFrame, and saves to Parquet.
        """
        df = pd.DataFrame(metadata_list).set_index("ticker")
        path: Path = self.base_dir / "0_metadata" / "latest_metadata.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow")
        return path



    def get_latest_market_cap(self, tickers: list[str]) -> pd.DataFrame:
        """
        Retrieves the most recent market cap from metadata cache.
        If cache is missing or refresh=True, it fetches from yfinance.
        """
        metadata_path: Path =self.base_dir / "0_metadata" / "latest_metadata.parquet"

        if metadata_path.exists():
            cached_df = pd.read_parquet(metadata_path, engine="pyarrow")
            shares_outstanding = cached_df["shares_outstanding"]
        else:
            raise FileNotFoundError(f"Metadata cache not found at {metadata_path}")

        close_frames = []
        for t in tickers:
            try:
                df_temp = self.load_ticker(t)
            except (FileNotFoundError, RuntimeError) as e:
                raise FileNotFoundError(f"Make sure {t} is in the data/raw folder") from e
            
            df_temp = df_temp.copy()
            if not isinstance(df_temp.index, pd.DatetimeIndex):
                df_temp.index = pd.to_datetime(df_temp.index)
            if isinstance(df_temp.columns, pd.MultiIndex):
                df_temp.columns = df_temp.columns.get_level_values(-1)
            if "Close" not in df_temp.columns:
                raise ValueError(f"Expected a 'Close' column for {t}, got columns={list(df_temp.columns)}")
            close_frames.append(df_temp["Close"].rename(t))
        
        prices = pd.concat(close_frames, axis=1).sort_index().ffill()
        latest_prices = prices.iloc[-1]
        # Align metadata to requested tickers; NaN where shares_outstanding missing
        shares_aligned = shares_outstanding.reindex(latest_prices.index)
        market_caps = latest_prices * shares_aligned
        # Drop tickers with missing shares_outstanding (yfinance failed or not in metadata)
        dropped = market_caps[market_caps.isna()].index.tolist()
        if dropped:
            logger.warning(
                "Skipped %d ticker(s) with missing shares_outstanding: %s. Re-run metadata fetch.",
                len(dropped), dropped,
            )
        return market_caps.dropna()


            

            
        

        



