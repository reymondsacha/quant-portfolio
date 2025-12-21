from abc import ABC, abstractmethod
from queue import Queue
import pandas as pd
from pathlib import Path
from src.backtest.events import MarketEvent

class DataHandler(ABC):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of events for each symbol requested.
    """

    @abstractmethod
    def get_latest_bars(self, symbol :str, N :int=1):
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

    def __init__(self, events_queue, file_path: str | Path, symbol: str):
        self.events_queue = events_queue
        self.symbol = symbol
        self.continue_backtest = True
        
        # 1. Load Data
        # We enforce Path object here to sanitize input
        self.file_path = Path(file_path)
        self._data = self._load_data(self.file_path)
        
        # 2. Convert to Generator (The "Stream")
        self._bar_generator = self._data.itertuples(index=True) 

    def _load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Loads parquet and ensures chronological sort.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        df.sort_index(inplace=True) # CRITICAL: Prevent look-ahead bias
        return df

    def update_bars(self):
        """
        Pushes the NEXT bar to the queue.
        """
        try:
            row = next(self._bar_generator)
            
            evt = MarketEvent(
                timestamp=row.Index, 
                symbol=self.symbol,
                open_price=row.open,
                high_price=row.high,
                low_price=row.low,
                close_price=row.close,
                volume=row.volume
            )
            
            self.events_queue.put(evt)
            
        except StopIteration:
            self.continue_backtest = False
            print(f"Backtest complete for {self.symbol}")

    def get_latest_bars(self, symbol, N=1):
        pass