from queue import Queue
from pathlib import Path

from src.backtest.engine import Backtest
from src.backtest.data_handler import HistoricDataHandler
from src.backtest.strategy import SmaCrossStrategy


def run_pipeline():
    DATA_SET_ROOT = "data/raw/AAPL"
    SYMBOL = "AAPL"
    dataset_path = Path(DATA_SET_ROOT)
    if not dataset_path.exists():
        print(f"Error : Dataset root {dataset_path} not found")
        print("Expected structure: data/raw/AAPL/year=2020/month=05/data.parquet")
    # 1. Infrastucture
    events_queue = Queue()

    # 2. The memory (datahandler)
    print(f"Loading data from {dataset_path}...")
    data_handler = HistoricDataHandler(
        events_queue=events_queue, file_path=dataset_path, symbol=SYMBOL
    )

    # 3. The brain (strategy)
    strategy = SmaCrossStrategy(
        data_handler=data_handler,
        events_queue=events_queue,
        short_window=20,
        long_window=50,
    )

    # 4. The engine (backtest)
    bt = Backtest(
        events_queue=events_queue, data_handler=data_handler, strategy=strategy
    )

    # 5. Ignite
    bt.run()


if __name__ == "__main__":
    run_pipeline()
