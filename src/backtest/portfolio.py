import logging
import math
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from src.backtest.events import SignalEvent, OrderEvent, FillEvent

logger = logging.getLogger("Portfolio")


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
    The NaivePortfolio sends orders to a brokerage object.
    It supports Dynamic Position Sizing based on
    percentage of equity.
    """

    def __init__(
        self,
        bars,
        events_queue,
        start_date,
        initial_capital=100000.0,
        pct_per_trade=0.10,
    ):
        self.bars = bars  # DataHandler object
        self.events_queue = events_queue  # The Event Queue
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.pct_per_trade = pct_per_trade

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
        # 1. Update Market Value of all Positions
        for s in self.symbol_list:
            try:
                bars = self.bars.get_latest_bars(s, N=1)
                if bars:
                    current_price = bars[0]["close_price"]

                    market_value = current_price * self.current_positions[s]
                    self.current_holdings[s] = market_value
            except KeyError:
                pass

        # 2. Update Total Equity
        total_equity = self.current_holdings["cash"]
        for s in self.symbol_list:
            total_equity += self.current_holdings[s]
        self.current_holdings["total"] = total_equity

        # 3. Capture Timestamp
        current_time = None
        try:
            bars = self.bars.get_latest_bars(self.symbol_list[0], N=1)
            if bars:
                current_time = bars[0]["timestamp"]
        except Exception:
            pass

        # 4. Record History (Snapshot)
        snapshot = self.current_holdings.copy()
        snapshot["timestamp"] = current_time  # type: ignore[assignment]
        self.all_positions.append(self.current_positions.copy())
        self.all_holdings.append(snapshot)

    def update_signal(self, event: SignalEvent):
        """
        Acts on a SignalEvent to generate an OrderEvent
        """
        if event.type == "SIGNAL":
            order_event = self._generate_dynamic_order(event)
            if order_event is not None:
                self.events_queue.put(order_event)

    def _generate_dynamic_order(self, signal: SignalEvent):
        """
        Calculates the quantity to trade based on a
        percentage of current total equity.
        """
        order = None
        symbol = signal.symbol
        direction = signal.side
        order_type = "MKT"

        # 1. Get current total equity
        current_equity = self.current_holdings["total"]

        # 2. Get latest price to estimate quantity
        bars = self.bars.get_latest_bars(symbol, N=1)
        if not bars:
            return None  # Cannot trade without price

        current_price = bars[0]["close_price"]

        # 3. Calculate Quantity based on pct Equity
        target_allocation = current_equity * self.pct_per_trade

        # Cash Guard
        current_cash = self.current_holdings["cash"]
        if direction == "LONG" and target_allocation > current_cash:
            target_allocation = current_cash

        # 4. Calculate Quantity to Trade
        if current_price > 0:
            quantity = int(math.floor(target_allocation / current_price))
        else:
            quantity = 0

        # Safety
        if quantity == 0 and direction != "EXIT":
            logger.warning(
                f"Signal ignored for {symbol}: Insufficient capital for 1 share"
            )

        # 5. Generate Order
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
                order = OrderEvent(symbol, order_type, abs(cur_qty), "BUY")

        return order

    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        with a FillEvent.
        """

        if not isinstance(event, FillEvent):
            logger.error(
                f"CRITICAL: update_fill received {type(event)} instead of FillEvent. Skipping."
            )
            return

        # 1. Get the latest price to calculate cost
        latest_bars_list = self.bars.get_latest_bars(event.symbol, N=1)

        if not latest_bars_list:
            logger.error(
                f"CRITICAL: No Market Data Found for {event.symbol} to process Fill"
            )
            return

        current_bar = latest_bars_list[0]
        fill_price = current_bar["close_price"]

        # 2. Calculate cost
        fill_qty = event.quantity
        direction = event.direction
        cost = fill_qty * fill_price
        commission = event.commission if event.commission else 0.0

        # 3. Update Positions (the Quantity)
        if event.symbol not in self.current_positions:
            self.current_positions[event.symbol] = 0

        if direction == "BUY":
            self.current_positions[event.symbol] += fill_qty
        elif direction == "SELL":
            self.current_positions[event.symbol] -= fill_qty

        # 4. Update Cash (holdings[symbol] is updated in update_timeindex based on market prices)
        if direction == "BUY":
            self.current_holdings["cash"] -= cost + commission
        elif direction == "SELL":
            self.current_holdings["cash"] += cost - commission

        # 5. Update Cumulative Commission
        self.current_holdings["commission"] += commission

    def create_equity_curve_dataframe(self):
        """
        Creates the equity curve and calculates basic performance metrics
        """
        curve = pd.DataFrame(self.all_holdings)

        # Set datetime index if available
        if "timestamp" in curve.columns and not bool(curve["timestamp"].isnull().all()):
            curve.set_index("timestamp", inplace=True)
            curve.index = pd.to_datetime(curve.index)

        # Calculate Returns
        curve["returns"] = (
            curve["total"].pct_change().fillna(0.0)
        )  # First period has no return (0%)
        curve["equity_curve"] = (1.0 + curve["returns"]).cumprod()

        # Calculate Drawdown
        # 1. Calculate Running Maximum (High Water Mark)
        # 2. Drawdown = (Current Value - High Water Mark) / High Water Mark
        running_max = curve["total"].cummax()
        curve["drawdown"] = (curve["total"] - running_max) / running_max

        self.equity_curve = curve
        return curve

    def output_summary_stats(self):
        """
        Returns a dictionary of performance statistics
        """
        if not hasattr(self, "equity_curve"):
            self.create_equity_curve_dataframe()

        total_return = (
            self.equity_curve["total"].iloc[-1] - self.initial_capital
        ) / self.initial_capital

        # Calculate Sharpe Ratio (Annualized)
        # Assuming daily data (252 days). If minute data, use 252*60*6.5
        returns = self.equity_curve["returns"]
        if returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0.0

        max_drawdown = self.equity_curve["drawdown"].min()

        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
        }
