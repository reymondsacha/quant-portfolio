from abc import ABC, abstractmethod
from queue import Queue
import numpy as np
import logging

from src.backtest.events import SignalEvent
from src.backtest.data_handler import DataHandler

logger = logging.getLogger("Strategy")


class Strategy(ABC):
    """
    Abstract Strategy class providing an interface for
    all subsequent (inherited) strategy handling objects.
    """

    @abstractmethod
    def calculate_signals(self, event):
        """
        Calculate signals based on the event received.
        """
        raise NotImplementedError("Should implement calculate_signals()")


class SmaCrossStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy.

    Generates a LONG signal when the short_window SMA crosses
    above the long_window SMA.
    """

    def __init__(
        self,
        data_handler: DataHandler,
        events_queue: Queue,
        short_window=20,
        long_window=50,
    ):
        self.data_handler = data_handler
        self.events_queue = events_queue
        self.short_window = short_window
        self.long_window = long_window

        # We need at least long_window + 1 bars to calculate
        # the current SMA and the previous SMA (to detect the cross).
        self.min_bars = long_window + 1

    def calculate_signals(self, event):
        """
        Reacts to a MARKET event to calculate signals.
        """
        if event.type == "MARKET":
            symbol = event.symbol

            # 1. Query the Memory (DataHandler)
            # We request enough history for the calculation
            bars = self.data_handler.get_latest_bars(symbol, N=self.min_bars)

            # If memory is insufficient, stay silent
            if len(bars) < self.min_bars:
                return

            # 2. Extract Prices (Vectorization)
            # The bars are dicts, so we extract the 'close' key
            closes = np.array([b["close_price"] for b in bars])

            # 3. Calculate SMAs (The "Thinking")

            # Current Step (t)
            # Slicing: take the last N items
            short_sma_t = np.mean(closes[-self.short_window :])
            long_sma_t = np.mean(closes[-self.long_window :])

            # Previous Step (t-1) 
            # Slicing: take from -(N+1) up to -1 (excluding the current bar)
            short_sma_prev = np.mean(closes[-self.short_window - 1 : -1])
            long_sma_prev = np.mean(closes[-self.long_window - 1 : -1])

            # 4. The Decision (Crossover Logic)
            # Check for "Golden Cross": Short crosses ABOVE Long
            if short_sma_t > long_sma_t and short_sma_prev <= long_sma_prev:
                logger.info(
                    f" SIGNAL: BUY {symbol} "
                    f"(Short: {short_sma_t:.2f} > Long: {long_sma_t:.2f})"
                )

                # Emit Signal
                signal = SignalEvent(
                    timestamp=bars[-1]["timestamp"],
                    symbol=symbol,
                    side="LONG",
                    strength=1.0,  # Fixed strength for now
                )
                self.events_queue.put(signal)

            # CASE B: Death Cross (SELL / EXIT)
            elif short_sma_t < long_sma_t and short_sma_prev >= long_sma_prev:
                logger.info(
                    f" SIGNAL: EXIT {symbol} "
                    f"(Short: {short_sma_t:.2f} < Long: {long_sma_t:.2f})"
                )

                # Emit Signal
                signal = SignalEvent(
                    timestamp=bars[-1]["timestamp"],
                    symbol=symbol,
                    side="EXIT",  # NaivePortfolio detects this and closes the position
                    strength=1.0,
                )
                self.events_queue.put(signal)
