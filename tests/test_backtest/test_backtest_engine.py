import queue
from dataclasses import dataclass
from src.backtest.events import Event, EventType
from src.backtest.engine import Backtest


@dataclass
class MockEvent(Event):
    timestamp: str = "2023-01-01"
    symbol: str = "TEST"

    def __post_init__(self):
        self.type = EventType.MARKET


class MockDataHandler:
    def __init__(self):
        self.continue_backtest = True
        self._call_count = 0
        self.symbol_list = ["TEST"]

    def update_bars(self):
        self._call_count += 1
        if self._call_count > 3:
            self.continue_backtest = False


class MockStrategy:
    def __init__(self):
        self._call_count = 0

    def calculate_signals(self, event):
        self._call_count += 1


def test_loop_processing():
    """
    Verifies that the Backtest engine processes events
    in the queue correctly until the queue is empty.
    """
    # 1. Setup (Injecting the Queue)
    test_queue = queue.Queue()
    mock_handler = MockDataHandler()
    mock_strategy = MockStrategy()

    backtest = Backtest(
        test_queue,
        mock_handler,
        mock_strategy,
        start_date="2020-01-01",
        initial_capital=100000.0,
    )

    # 2. Inject Events
    test_queue.put(MockEvent())

    # 3. Run
    backtest.run()

    # 4. Assert
    assert test_queue.empty()
    assert mock_handler._call_count > 0  # Verifies engine called for data
