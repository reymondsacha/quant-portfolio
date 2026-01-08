from abc import ABC, abstractmethod
from src.backtest.events import SignalEvent, OrderEvent


class Portfolio(ABC):
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a 'bar'.
    """

    @abstractmethod
    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders
        """
        raise NotImplementedError("Should implement update_signal()")


class NaivePortfolio(Portfolio):
    """
    The NaivePortfolio sends orders to a brokerage object
    with a constant quantity size blindly, without risk
    management or position sizing.
    """

    def __init__(self, bars, events, start_date, initial_capital=100000.0):
        self.bars = bars  # DataHandler object
        self.events = events  # The Event Queue
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital

        # State Tracking

        # Positions : Quantity of shares held
        self.current_positions = {s: 0 for s in self.symbol_list}

        # Holdings: Market value of all positions
        self.current_holdings = self.construct_current_holdings()

        # History
        self.all_positions = []
        self.all_holdings = []

    def construct_current_holdings(self):
        """
        Constructs the dictionary to hold the value
        of the portfolio.
        """
        holdings = {s: 0.0 for s in self.symbol_list}
        holdings["cash"] = self.initial_capital
        holdings["commission"] = 0.0
        holdings["total"] = self.initial_capital

    def update_signal(self, event: SignalEvent):
        """
        Acts on a SignalEvent to generate an OrderEvent
        """
        if event.type == "SIGNAL":
            order_event = self._generate_naive_order(event)
            return order_event
        return None

    def _generate_naive_order(self, signal: SignalEvent):
        """
        Simply Buys/Sells a fixed quantity (100)
        regardless of cash.
        """
        order = None
        symbol = signal.symbol
        direction = signal.side

        quantity = 100
        order_type = "MKT"

        if direction == "LONG":
            order = OrderEvent(symbol, order_type, quantity, "BUY")
        elif direction == "SHORT":
            order = OrderEvent(symbol, order_type, quantity, "SELL")

        if direction == "EXIT":
            # Logic: if we are long, Sell. If Short, Buy.
            cur_qty = self.current_positions[symbol]
            if cur_qty > 0:
                order = OrderEvent(symbol, order_type, cur_qty, "SELL")
            elif cur_qty < 0:
                order = OrderEvent(symbol, order_type, quantity, "Buy")

        return order
