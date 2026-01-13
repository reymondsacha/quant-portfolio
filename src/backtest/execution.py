from abc import ABC, abstractmethod
import datetime
from queue import Queue
import logging

from src.backtest.events import OrderEvent, FillEvent

logger = logging.getLogger("Execution")

class ExecutionHandler(ABC):
    """
    The ExecutionHandler simulates the intercation
    between the Queue and the Brokerage/API.
    """

    @abstractmethod
    def execute_order(self, event: OrderEvent) -> None:
        """
        Takes an Order and executes it, producing a FillEvent.
        The FillEvent is added to the events queue.
        """
        pass


class SimulatedExecutionHandler(ExecutionHandler):
    """
    The simulated execution handler converts all order objects into
    fill objects automatically without latency, slippage, or
    fill-ratio issues.
    """

    def __init__(self, events_queue: Queue):
        """
        Initializes the handler.

        Args:
            events_queue: The Event Queue to push FillEvents to.
        """

        self.events_queue = events_queue

    def execute_order(self, event: OrderEvent) -> None:
        """
        Simply converts OrderEvents -> FillEvents.
        The FillEvent is added to the events queue.
        """
        if event.type == "ORDER":
            # 1. Define Execution Details
            # We assume immediate fill
            # "ARCA" is a placeholder for the exchange

            # Hardcoded commission per trade (Simulating a broker fee)
            commission = 1.0

            # 2. Create the Fill Event
            # Note: fill_cost (Price) is set to None because here
            # because the OrderEvent doesn't carry the current price.
            # The Portfolio will resolve the actual execution price
            # using the DataHandler.
            fill_event = FillEvent(
                timestamp=datetime.datetime.utcnow(),
                symbol=event.symbol,
                exchange="ARCA",
                quantity=event.quantity,
                direction=event.direction,
                fill_cost=None,
                commission=commission,
            )

            # 3. Push to Queue
            self.events_queue.put(fill_event)
            logger.info(
                f"--> EXECUTION:  Filled {event.direction} {event.quantity} {event.symbol} (Comm: ${commission})"
            )
