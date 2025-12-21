import queue
import time
from src.backtest.events import Event, EventType

class Backtest:
    def __init__(self, events_queue: queue.Queue, data_handler):
        """
        Initializes the event queue and the loop components.
        """
        self.events = events_queue
        self.data_handler = data_handler
        self.continue_backtest = True

    def generate_signals(self, event):
        """
        Placeholder: This will eventually link to the Strategy class.
        """
        print(f"--- [Placeholder] Generating Signal based on {event} ---")

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
                event = self.events.get(False)
            except queue.Empty:
                break
            
            # ROUTER: Dispatch event to the correct handler
            if event is not None:
                if event.type == EventType.MARKET:
                    print(f"Processing MARKET Event: {event.timestamp}")
                    self.generate_signals(event)
                    
                elif event.type == EventType.SIGNAL:
                    print(f"Processing SIGNAL Event: {event.symbol} -> {event.side}")
                    
        print("Backtest complete")
                # We will add ORDER and FILL handlers here later
