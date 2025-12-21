from datetime import datetime
from src.backtest.engine import Backtest
from src.backtest.events import MarketEvent, SignalEvent

def run_pipeline():
    # 1. Instantiate the Engine
    bt = Backtest()
    
    # 2. Simulate Data Feed: Push 3 dummy events into the queue
    print("Injecting dummy events...")
    
    # Event 1: Market Data arrives
    bt.events.put(MarketEvent(timestamp=datetime.now()))
    
    # Event 2: Strategy generates a Signal
    bt.events.put(SignalEvent(timestamp=datetime.now(), symbol="AAPL", side="LONG"))
    
    # Event 3: Another Market Data update
    bt.events.put(MarketEvent(timestamp=datetime.now()))

    # 3. Start the Engine
    bt.run()

if __name__ == "__main__":
    run_pipeline()