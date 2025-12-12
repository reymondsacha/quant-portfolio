# main.py (Located in the root of your quant-portfolio project)
import logging
from datetime import datetime, timedelta

# We import from the src package because we are outside of it
from src.data_loader import YahooDownloader
from src.data_manager import DataManager

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    """
    Orchestrates the data pipeline flow: Download -> Save -> Verify.
    """
    TICKER = "NVDA"

    # Calculate date range (1 month back from today)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # 1. Initialize our decoupled components
    downloader = YahooDownloader(
        tickers=[TICKER], start_date=start_date, end_date=end_date
    )
    manager = DataManager(base_dir="data/raw")  # Data written to data/raw/

    print(f"--- 1. FETCHING {TICKER} (Network) ---")
    try:
        # Downloader fetches data into self.data (in memory)
        downloader.fetch()
        df = downloader.data
        print(f"Downloaded {len(df)} rows.")

        print(f"--- 2. SAVING {TICKER} (Disk) ---")
        # Manager takes the DataFrame and persists it securely (Atomic Write)
        saved_path = manager.save_ticker(TICKER, df)
        print(f"Securely saved to: {saved_path}")

        print(f"--- 3. VERIFYING (Read-Back) ---")
        # Read back to ensure integrity
        df_loaded = manager.load_ticker(TICKER)
        print(f"Verification successful. Data loaded with shape: {df_loaded.shape}")

    except Exception as e:
        # Critical failure path
        logging.error(f"Pipeline Failed for {TICKER}: {e}")


if __name__ == "__main__":
    main()
