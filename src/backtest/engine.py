import queue
from src.backtest.events import EventType


class Backtest:
    def __init__(self, events_queue: queue.Queue, data_handler, strategy):
        """
        Initializes the event queue and the loop components.
        """
        self.events = events_queue
        self.data_handler = data_handler
        self.strategy = strategy
        self.continue_backtest = True

    def run(self):
        """
        The Core Event Loop.
        Executes while the queue is not empty.
        """
        print("Starting Event Loop...")

        while self.data_handler.continue_backtest:
            # Check if queue is empty to break the infinite loop
            if self.events.empty():
                self.data_handler.update_bars()

            # Fetch the next event
            try:
                event = self.events.get(block=False)
            except queue.Empty:
                continue

            # ROUTER: Dispatch event to the correct handler
            if event is not None:
                if event.type == EventType.MARKET:
                    self.strategy.calculate_signals(event)

                elif event.type == EventType.SIGNAL:
                    print(f">>> SIGNAL DETECTED: {event.symbol} {event.side}")

        print("Backtest complete")
        # We will add ORDER and FILL handlers here later
