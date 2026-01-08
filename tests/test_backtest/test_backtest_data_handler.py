import pytest
import queue
import pandas as pd
from src.backtest.data_handler import HistoricDataHandler


@pytest.fixture
def mock_parquet(tmp_path):
    """
    Creates a temporary parquet file.
    tmp_path is already a pathlib.Path object.
    """
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(100, 150),
            "High": range(105, 155),
            "Low": range(95, 145),
            "Close": range(100, 150),
            "Volume": [1000] * 50,
        },
        index=dates,
    )

    # We can use / operator with Path objects
    file_path = tmp_path / "test_data.parquet"
    df.to_parquet(file_path)
    return file_path


def test_stream_mechanics(mock_parquet):
    q = queue.Queue()

    # We pass the Path object directly
    handler = HistoricDataHandler(q, mock_parquet, "TEST_SYM")
    assert handler.continue_backtest is True

    # 1. First Tick
    handler.update_bars()
    assert not q.empty()
    evt1 = q.get()
    assert evt1.close_price == 100.0

    # 2. Run to the end
    for _ in range(49):
        handler.update_bars()

    assert handler.continue_backtest is True

    # 3. End of Stream
    handler.update_bars()
    assert not handler.continue_backtest


def test_buffer_window_precision(mock_parquet):
    """
    Verifies get_latest_bars(N) returns exactly N items
    and respects the current timeline cursor.
    """

    q = queue.Queue()
    symbol = "TEST_SYM"
    handler = HistoricDataHandler(q, mock_parquet, symbol)

    # Push the timeline forward by 30 steps
    # The handler should have 30 bars in memory
    # The last bar (index 29) should have a close price of 129
    for _ in range(30):
        handler.update_bars()
        q.get()

    # Case A: Request N=1 (Current Bar)
    latest_1 = handler.get_latest_bars(symbol, N=1)
    assert len(latest_1) == 1
    assert latest_1[0]["close_price"] == 129.0

    # Case B : Request N=20 (A standard window)
    window_20 = handler.get_latest_bars(symbol, N=20)
    assert len(window_20) == 20
    # Verify order : Last item is current (129), first item is 20 bars ago
    assert window_20[-1]["close_price"] == 129.0
    # Close at Inex 10 is 110
    assert window_20[0]["close_price"] == 110.0

    # Case C : Request more than available (Overflow)
    overflow = handler.get_latest_bars(symbol, N=50)
    assert len(overflow) == 30
    assert overflow[-1]["close_price"] == 129.0
    assert overflow[0]["close_price"] == 100.0

    # Case D : wrong symbol
    wrong_sym = handler.get_latest_bars("INVALID", N=1)
    assert wrong_sym == []
