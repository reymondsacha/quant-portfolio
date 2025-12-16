# src/data_manager.py

import pandas as pd
from pathlib import Path
from typing import List, cast


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
        1. Write to {ticker}.parquet.tmp
        2. Rename to {ticker}.parquet (Atomic Operation)

        Returns:
            Path: The absolute path to the saved file.
        """
        if df.empty:
            raise ValueError(f"Cannot save empty dataframe for ticker '{ticker}'.")
        

        monthly_groups = df.groupby(pd.Grouper(freq='ME'))

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
            partition_dir: Path = self.base_dir / ticker / f"year={year}" / f"month={month}"

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
                f.name  # get the directory name
                for f in self.base_dir.iterdir()
                if f.is_dir()
                and not f.name.startswith(".")
                and f.name != "__pycache__"  # Excluse non-ticker directories
            ]
        )
