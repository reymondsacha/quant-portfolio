import logging
import pandas as pd
from abc import ABC, abstractmethod
from src.backtest.events import SignalEvent, OrderEvent, FillEvent

logger = logging.getLogger(__name__)

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

    @abstractmethod
    def update_fill(self, event):
        """
        Update the portfolio current positions and holdings
        from a FillEvent.
        """
        raise NotImplementedError("Should implement update_fill()")


class NaivePortfolio(Portfolio):
    """
    The NaivePortfolio sends orders to a brokerage object
    with a constant quantity size blindly, without risk
    management or position sizing.
    """

    def __init__(self, bars, events_queue, start_date, initial_capital=100000.0):
        self.bars = bars  # DataHandler object
        self.events_queue = events_queue  # The Event Queue
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
        return holdings

    def update_timeindex(self, event):
        """
        Updates the current holdings (Mark-to-Market) using the latest
        available prices and appends in history.
        """
        #1. Update Market Value of all Positions
        for s in self.symbol_list:
            try:
                bars = self.bars.get_latest_bars(s, N=1)
                if bars:
                    current_price =  bars[0]['close_price']

                    market_value = current_price * self.current_positions[s]
                    self.current_holdings[s] = market_value
                
                else:
                    pass
            except KeyError:
                pass
        
        #2. Update Total Equity
        total_equity = self.current_holdings['cash']
        for s in self.symbol_list:
            total_equity += self.current_holdings[s]
        self.current_holdings['total'] = total_equity

        #3. Record History (Snapshot)
        self.all_positions.append(self.current_positions.copy())
        self.all_holdings.append(self.current_holdings.copy())


    def update_signal(self, event: SignalEvent):
        """
        Acts on a SignalEvent to generate an OrderEvent
        """
        if event.type == "SIGNAL":
            order_event = self._generate_naive_order(event)
            if order_event is not None:
                self.events_queue.put(order_event)

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

    def update_fill(self, event: FillEvent):
        """
        Updates the portfolio current positions and holdings
        with a FillEvent.
        """

        if not isinstance(event, FillEvent):
            print(f"CRITICAL: update_fill received {type(event)} instead of FillEvent. Skipping.")
            return

        #1. Get the latest price to calculate cost
        latest_bars_list = self.bars.get_latest_bars(event.symbol, N=1)

        if not latest_bars_list:
            logger.error(f"CRITICAL: No Market Data Found for {event.symbol} to process Fill")
            return

        current_bar = latest_bars_list[0]
        fill_price = current_bar['close_price']

        #2. Calculate cost
        fill_qty = event.quantity
        direction = event.direction
        cost = fill_qty * fill_price
        commission = event.commission if event.commission else 0.0

        #3. Update Positions (the Quantity)
        if event.symbol not in self.current_positions:
            self.current_positions[event.symbol] = 0

        if direction == "BUY":
            self.current_positions[event.symbol] += fill_qty
        elif direction == "SELL":
            self.current_positions[event.symbol] -= fill_qty

        #4. Update Holdings
        if direction == "BUY":
            self.current_holdings[event.symbol] += cost
            self.current_holdings['cash'] -= (cost + commission)
        
        if direction == "SELL":
            self.current_holdings[event.symbol] -= cost
            self.current_holdings['cash'] += (cost - commission)

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings list
        of dictionaries.
        """
        curve = pd.DataFrame(self.all_holdings)
        self.equity_curve = curve
        return curve

        
               



        


