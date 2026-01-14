import queue
import logging

from src.backtest.events import EventType
from src.backtest.portfolio import NaivePortfolio
from src.backtest.execution import SimulatedExecutionHandler

logger = logging.getLogger("Engine")


class Backtest:
    def __init__(
        self,
        events_queue: queue.Queue,
        data_handler,
        strategy,
        start_date,
        initial_capital=1000000.0,
        pct_per_trade=0.10,
    ):
        """
        Initializes the event queue and the loop components.
        """
        self.events_queue = events_queue
        self.data_handler = data_handler
        self.strategy = strategy
        self.continue_backtest = True

        self.portfolio = NaivePortfolio(
            bars=data_handler,
            events_queue=events_queue,
            start_date=start_date,
            initial_capital=initial_capital,
            pct_per_trade=pct_per_trade,
        )

        self.execution_handler = SimulatedExecutionHandler(events_queue)

    def run(self):
        """
        The Core Event Loop.
        Executes while the queue is not empty.
        """
        logger.info(
            f"Starting Backtest Event Loop with ${self.portfolio.initial_capital:,.2f}..."
        )

        while self.data_handler.continue_backtest:
            # Check if queue is empty to break the infinite loop
            if self.events_queue.empty():
                self.data_handler.update_bars()

            # Fetch the next event
            try:
                event = self.events_queue.get(block=False)
            except queue.Empty:
                continue

            # ROUTER: Dispatch event to the correct handler
            if event is not None:
                # 1. MARKET DATA -- STRATEGY
                if event.type == EventType.MARKET:
                    self.strategy.calculate_signals(event)
                    self.portfolio.update_timeindex(event)

                # 2. SIGNAL -- PORTFOLIO
                elif event.type == EventType.SIGNAL:
                    logger.info(f">>> SIGNAL DETECTED: {event.symbol} {event.side}")
                    # Portfolio decides if the trade can be afforded
                    self.portfolio.update_signal(event)

                # 3. ORDER -- EXECUTION
                elif event.type == EventType.ORDER:
                    logger.info(
                        f">>> ORDER GENERATED: {event.symbol} {event.direction} {event.quantity}"
                    )
                    self.execution_handler.execute_order(event)

                # 4. FILL -- PORTFOLIO
                elif event.type == EventType.FILL:
                    self.portfolio.update_fill(event)

        logger.info("Backtest complete")
        logger.info(f"Final Equity: ${self.portfolio.current_holdings['total']:,.2f}")

    def output_performance(self):
        """
        Outputs the portfolio performance statistics and saves the PLOT.
        """
        # Ensure the portfolio builds the curve dataframe
        if hasattr(self.portfolio, "create_equity_curve_dataframe"):
            self.portfolio.create_equity_curve_dataframe()
        else:
            logger.error("Portfolio missing 'create_equity_curve_dataframe' method.")
            return

        stats = self.portfolio.output_summary_stats()

        logger.info("-" * 40)
        logger.info(
            f"Final Portfolio Value: ${self.portfolio.current_holdings['total']:,.2f}"
        )
        logger.info(f"Total Return:      {stats['Total Return']:.2%}")
        logger.info(f"Max Drawdown:      {stats['Max Drawdown']:.2%}")
        logger.info(f"Sharpe Ratio:      {stats['Sharpe Ratio']:,.2f}")
        logger.info("-" * 40)

    def export_results(self, filename="backtest_results.csv"):
        """
        Exports the portfolio equity curve to a CSV file.
        """
        # Ensure the equity curve dataframe is generated
        df = self.portfolio.create_equity_curve_dataframe()

        # Save to CSV
        df.to_csv(filename)
        logger.info(f"[-] Results exported to {filename}")
