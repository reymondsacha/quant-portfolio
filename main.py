# main.py (Update your existing file in the project root)

import logging
from src.data_loader import (
    YahooDownloader,
)  # Note the rename from yahoo_downloader to data_loader
from src.data_manager import DataManager
from typing import List

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
        f"--- Starting Data Pipeline for {total_tickers} tickers ({start_date} to {end_date}) ---"
    )

    # 1. Iterate through the list of tickers
    for i, ticker in enumerate(tickers, 1):
        logging.info(f"[{i}/{total_tickers}] Processing ticker: {ticker}")

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
                    f"SUCCESS: {ticker} saved {len(df)} rows to {saved_path.name}"
                )
                success_count += 1
            else:
                logging.warning(
                    f"SKIPPED {ticker}: DataFrame was returned empty (e.g., failed to fetch or invalid ticker)."
                )

        except ValueError as e:
            # Catches validation errors (e.g., bad date format)
            logging.warning(f"SKIPPED {ticker}: Validation Error. {e}")

        except RuntimeError as e:
            # Catches critical fetch/save errors (Network failure, corrupt file handling, etc.)
            logging.error(f"FAILED {ticker}: Critical Error. {e}")

    # 4. Final Report
    logging.info(
        f"--- Pipeline Finished. {success_count}/{total_tickers} successful. ---"
    )


def main():
    # Define the list of core tickers to ingest
    CORE_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "^GSPC"]

    # Define the primary configuration (Dates)
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"

    run_pipeline(CORE_TICKERS, START_DATE, END_DATE)


if __name__ == "__main__":
    main()
