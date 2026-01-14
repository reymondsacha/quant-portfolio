import pytest
from datetime import datetime

from src.backtest.portfolio import NaivePortfolio
from src.backtest.events import SignalEvent, FillEvent


class MockDataHandler:
    """
    Minimal mock of DataHandler to return a price for a given symbol.
    """

    def __init__(self):
        self.prices = {}  # We store forced prices here
        self.symbol_list = ["AAPL", "GOOG"]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns a list of dictionaries.
        """

        price = self.prices.get(symbol, 100.0)
        dummy_bar = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "open_price": price,
            "high_price": price,
            "low_price": price,
            "close_price": price,
            "volume": 1000,
        }

        return [dummy_bar] * N


@pytest.fixture
def mock_bars():
    """
    Fixture to provide a mock data handler for testing.
    """
    return MockDataHandler()


@pytest.fixture
def portfolio(mock_bars):
    """
    Sets up the portfolio with the mock bars.
    """
    initial_capital = 100000.0
    start_date = datetime(2020, 1, 1)

    p = NaivePortfolio(
        bars=mock_bars,
        events_queue=None,
        start_date=start_date,
        initial_capital=initial_capital,
        pct_per_trade=0.10,
    )
    return p


def test_generate_dynamic_sizing_order(portfolio, mock_bars):
    """
    Verify quantity calculation respects 10% allocation.
    """
    symbol = "AAPL"
    current_price = 175.50

    mock_bars.prices[symbol] = current_price

    # Creating dummy signal event
    sig = SignalEvent(timestamp=datetime.now(), symbol=symbol, side="LONG")

    # Generating order
    order = portfolio._generate_dynamic_order(sig)

    # Cash available = $100.000
    # Allocation percentage = 10% -> $10.000
    # Price = $175.50
    # Quantity = Floor(10,000 / 175.50) = 56

    expected_qty = 56

    assert order is not None
    assert order.symbol == "AAPL"
    assert order.quantity == expected_qty
    assert order.direction == "BUY"

    assert (order.quantity * current_price) <= (100000.0 * 0.10)


def test_update_fill_updates_cash_and_positions(portfolio, mock_bars):
    """
    Verify that a FillEvent correctly updates cash and position count.
    Formula: New Cash = Old Cash - (Price*Qty) - Commission
    """
    symbol = "AAPL"
    fill_price = 100.0
    quantity = 10
    commission = 1.0

    # 1. Rig the market
    mock_bars.prices[symbol] = fill_price

    # 2. Create FillEvent
    # Buying 10 shares at $100 = $1,000 cost. Commission $1. Total  hit : 1001$
    fill = FillEvent(
        timestamp=datetime.now(),
        symbol=symbol,
        exchange="ARCA",
        quantity=quantity,
        direction="BUY",
        fill_cost=None,
        commission=commission,
    )

    # 3. Action
    portfolio.update_fill(fill)

    # 4. Assert
    assert portfolio.current_positions[symbol] == 10

    # Cash check : 100,000 - 1,000 - 1.0 = 98,999.0
    expected_cash = 98999.0
    assert portfolio.current_holdings["cash"] == expected_cash


def test_create_equity_curve_calculates_drawdown(portfolio, mock_bars):
    """
    Verify the High Water Mark and Drawdown math.
    DD = (Total - Peak) / Peak
    """
    # 1. Mock the History
    # We manually inject a history of 3 days into all_holdings.
    # Day 1 : We buy $50k worth of stocks
    # Day 2 : Stock rallies +20% ($50k -> $60k)
    # Day 3 : Stock crashes -22% ($60k -> $46.8k)

    portfolio.all_holdings = [
        {"timestamp": datetime(2020, 1, 1), "cash": 50000.0, "total": 100000.0},
        {"timestamp": datetime(2020, 1, 2), "cash": 50000.0, "total": 110000.0},
        {"timestamp": datetime(2020, 1, 3), "cash": 50000.0, "total": 96800.0},
    ]

    df_curve = portfolio.create_equity_curve_dataframe()

    # ASSERTIONS

    # 1. Day 2 should be the Peak (Drawdown = 0.0)
    # 110k is the new High Water Mark
    assert df_curve.iloc[1]["drawdown"] == 0.0

    # 2. Day 3 should be in Drawdown
    # Peak at 110k, current at 96.8k
    # Loss = 110,000 - 96,800 = 13,200
    # Drawdown = -13,200 / 110,000 = -0.12

    expected_drawdown = (96800.0 - 110000.0) / 110000.0
    assert df_curve.iloc[-1]["drawdown"] == pytest.approx(expected_drawdown)
