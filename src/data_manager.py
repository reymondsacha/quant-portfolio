# src/data_manager.py
import pandas as pd
from pathlib import Path
from typing import List


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

        file_path: Path = self.base_dir / f"{ticker}.parquet"
        temp_path: Path = self.base_dir / f"{ticker}.parquet.tmp"

        try:
            # 1. Write to temporary file
            df.to_parquet(
                temp_path,
                engine="pyarrow",
                compression="snappy",
                index=True,  # Explicitly preserve the Date index
            )

            # 2. Atomic Rename (The Swap)
            temp_path.replace(file_path)
            return file_path

        except Exception as e:
            # Cleanup garbage if write failed
            if temp_path.exists():
                temp_path.unlink()

            raise RuntimeError(
                f"Failed to save ticker '{ticker}' to {file_path}: {e}"
            ) from e

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Load a ticker's data from Parquet format.
        """
        file_path: Path = self.base_dir / f"{ticker}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Ticker '{ticker}' not found. Available: {self.list_existing_tickers()}"
            )

        try:
            return pd.read_parquet(file_path, engine="pyarrow")
        except Exception as e:
            raise RuntimeError(f"Corrupt data for '{ticker}': {e}") from e

    def list_existing_tickers(self) -> List[str]:
        """List all tickers saved in the directory."""
        return sorted([f.stem for f in self.base_dir.glob("*.parquet")])
