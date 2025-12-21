import time
import functools
import logging
from typing import Callable

# Setup a simple logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SymbolNotFoundError(Exception):
    """Custom exception raised when a ticker symbol is invalid or data is missing."""

    def __init__(self, symbol, message="Symbol data not found"):
        self.symbol = symbol
        self.message = f"{message}: {symbol}"
        super().__init__(self.message)


def retry(retries: int = 3, delay: int = 2, backoff: int = 2) -> Callable:
    """
    Decorator that retries a function call if it raises an exception.

    :param retries: Max number of retries.
    :param delay: Initial delay (seconds) between retries.
    :param backoff: Multiplier applied to delay after each failure.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)  # Preserves func.__name__ and func.__doc__
        def wrapper(*args, **kwargs):
            current_delay = delay
            attempt = 0

            while attempt < retries:
                try:
                    return func(*args, **kwargs)  # Try to execute the function
                except Exception as e:
                    attempt += 1
                    logger.warning(
                        f"Attempt {attempt}/{retries} failed for '{func.__name__}': {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    if attempt == retries:
                        # If we used up all retries, raise the error to crash the script
                        logger.error(f"All {retries} retries failed.")
                        raise e

                    time.sleep(current_delay)
                    current_delay *= backoff  # Exponential backoff

        return wrapper

    return decorator
