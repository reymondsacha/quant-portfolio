# main.py (Update your existing file in the project root)

import logging
from pathlib import Path
from typing import List
from src.data.data_downloader import YahooDownloader
from src.data.data_manager import DataManager

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Configure Logging with both console and file handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[
        logging.FileHandler("logs/data.log", mode="w"),
        logging.StreamHandler(),  # Console output
    ],
)


def run_pipeline(tickers: List[str], start_date: str, end_date: str):
    """
    Runs the full data ingestion pipeline for a list of tickers within a date range.
    """

    # Components are initialized without state/data
    downloader = YahooDownloader()
    manager = DataManager(base_dir="data/raw")

    total_tickers = len(tickers)
    success_count = 0

    logging.info(
        "--- Starting Data Pipeline for %d tickers (%s to %s) ---",
        total_tickers,
        start_date,
        end_date,
    )

    # 1. Iterate through the list of tickers
    for i, ticker in enumerate(tickers, 1):
        logging.info("[%d/%d] Processing ticker: %s", i, total_tickers, ticker)

        try:
            # 2. Fetch Data (Downloader's job) - Now requires explicit dates
            df = downloader.fetch_history(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,  # <--- Dates passed directly to the stateless method
            )

            # 3. Persist Data (Manager's job) - Only save if DataFrame is not empty
            if not df.empty:
                saved_path = manager.save_ticker(ticker, df)
                logging.info(
                    "SUCCESS: %s saved %d rows to %s", ticker, len(df), saved_path.name
                )
                success_count += 1
            else:
                logging.warning(
                    "SKIPPED %s: DataFrame was returned empty (e.g., failed to fetch or invalid ticker).",
                    ticker,
                )

        except ValueError as e:
            # Catches validation errors (e.g., bad date format)
            logging.warning("SKIPPED %s: Validation Error. %s", ticker, e)

        except RuntimeError as e:
            # Catches critical fetch/save errors (Network failure, corrupt file handling, etc.)
            logging.error("FAILED %s: Critical Error. %s", ticker, e)

    # 4. Final Report
    logging.info(
        "--- Pipeline Finished. %d/%d successful. ---", success_count, total_tickers
    )


def main():
    # Define the list of core tickers to ingest
    CORE_TICKERS = [
        "AMZN"
    ]

    # Define the primary configuration (Dates)
    START_DATE = "2020-01-01"
    END_DATE = "2023-12-31"

    run_pipeline(CORE_TICKERS, START_DATE, END_DATE)


if __name__ == "__main__":
    main()
