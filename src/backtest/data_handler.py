from abc import ABC, abstractmethod
from typing import Iterator, cast
import queue
import pandas as pd
from pathlib import Path
from collections import deque
from src.backtest.events import MarketEvent


class DataHandler(ABC):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of events for each symbol requested.
    """

    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1):
        """
        Returns the last N bars from the latest_symbol_list,
        or fewer if less than N are available.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_list for all symbols
        in the symbol_list.
        """
        raise NotImplementedError("Should implement update_bars()")


class HistoricDataHandler(DataHandler):
    """
    HistoricDataHandler loads a Parquet file and iterates through it,
    simulating a live market feed.
    """

    def __init__(self, events_queue: queue.Queue, file_path: str | Path, symbol: str):
        self.events_queue = events_queue
        self.symbol = symbol
        self.continue_backtest = True

        # 1. Load Data
        # We enforce Path object here to sanitize input
        self.file_path = Path(file_path)
        self._data = self._load_data(self.file_path)

        # 2. Convert to Generator (The "Stream")
        self._bar_generator = cast(
            Iterator, self._data.itertuples(index=True, name="Row")
        )

        self._latest_symbol_data = deque()

    def _load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Loads parquet and ensures chronological sort.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        df = pd.read_parquet(file_path, engine="pyarrow")
        df.sort_index(inplace=True)  # CRITICAL: Prevent look-ahead bias
        return df

    def update_bars(self):
        """
        Pushes the NEXT bar to the queue.
        """
        try:
            row = next(self._bar_generator)

            # Standardize the bar data
            current_bar = {
                "timestamp": row.Index,
                "symbol": self.symbol,
                "open_price": row.Open,
                "high_price": row.High,
                "low_price": row.Low,
                "close_price": row.Close,
                "volume": row.Volume,
            }

            # Update Memory
            self._latest_symbol_data.append(current_bar)

            # Emit Event
            evt = MarketEvent(
                timestamp=row.Index,
                symbol=self.symbol,
                open_price=row.Open,
                high_price=row.High,
                low_price=row.Low,
                close_price=row.Close,
                volume=row.Volume,
            )

            self.events_queue.put(evt)

        except StopIteration:
            self.continue_backtest = False
            print(f"Backtest complete for {self.symbol}")

    def get_latest_bars(self, symbol, N=1):
        """
        Return the last N bars from internal memory
        """
        if symbol != self.symbol:
            # In a multi-symbol handler, we would look up the correct deque.
            # Here we just return empty if mismatch.
            print(f"Warning : Symbol {symbol} not found in this handler.")
            return []
        # Return the slice.
        # Note: Deque slicing is not native, so we convert to list for the slice.
        # This is O(N) copy, but acceptable for N < 1000 in Python backtesting.
        return list(self._latest_symbol_data)[-N:]
