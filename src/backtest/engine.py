import queue
from src.backtest.events import EventType
from src.backtest.portfolio import NaivePortfolio


class Backtest:
    def __init__(self, events_queue: queue.Queue, data_handler, strategy, start_date, initial_capital=1000000.0):
        """
        Initializes the event queue and the loop components.
        """
        self.events = events_queue
        self.data_handler = data_handler
        self.strategy = strategy
        self.continue_backtest = True
        self.portfolio = NaivePortfolio(
            bars = data_handler,
            events = events_queue,
            start_date = start_date,
            initial_capital = initial_capital
        )


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

                #1. MARKET DATA -- STRATEGY
                if event.type == EventType.MARKET:
                    self.strategy.calculate_signals(event)

                #2. SIGNAL -- PORTFOLIO
                elif event.type == EventType.SIGNAL:
                    print(f">>> SIGNAL DETECTED: {event.symbol} {event.side}")
                    #Portfolio decides if the trade can be afforded
                    new_order = self.portfolio.update_signal(event)
                    if new_order is not None:
                        self.events.put(new_order)

                #3. ORDER -- EXECUTION
                elif event.type == EventType.ORDER:
                    print(f">>> ORDER GENERATED: {event.print_order()}")

        print("Backtest complete")

