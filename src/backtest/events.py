from dataclasses import dataclass, field
from datetime import datetime


class EventType:
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


@dataclass
class Event:
    """
    Base Event class.
    init=False means 'type' is NOT required in the constructor (e.g. MarketEvent(timestamp=...)).
    """

    type: str | None = field(init=False, default=None)


@dataclass
class MarketEvent(Event):
    timestamp: datetime
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float

    def __post_init__(self):
        self.type = EventType.MARKET


@dataclass
class SignalEvent(Event):
    timestamp: datetime
    symbol: str
    side: str
    strength: float = 1.0

    def __post_init__(self):
        self.type = EventType.SIGNAL
