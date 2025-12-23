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
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [95.0, 96.0],
            "Close": [102.0, 103.0],
            "Volume": [1000, 2000],
        },
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )

    # We can use / operator with Path objects
    file_path = tmp_path / "test_data.parquet"
    df.to_parquet(file_path)
    return file_path


def test_stream_mechanics(mock_parquet):
    q = queue.Queue()

    # We pass the Path object directly
    handler = HistoricDataHandler(q, mock_parquet, "TEST_SYM")

    # 1. First Tick
    handler.update_bars()
    assert not q.empty()
    evt1 = q.get()
    assert evt1.close_price == 102.0

    # 2. Second Tick
    handler.update_bars()
    evt2 = q.get()
    assert evt2.close_price == 103.0

    # 3. End of Stream
    handler.update_bars()
    assert not handler.continue_backtest
