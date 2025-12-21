import pytest
import pandas as pd
from unittest.mock import patch
from src.data.data_downloader import YahooDownloader


class TestRobustness:
    """
    Tests specifically for the @retry decorator and error handling logic.
    These tests do NOT hit the real Yahoo Finance API.
    """

    @patch("src.data.data_downloader.yf.download")
    def test_retry_on_network_failure(self, mock_download):
        """
        Scenario: The API fails twice with network errors, then succeeds.
        Expectation: The code should catch errors, wait, and retry until success.
        """
        # STEP 1: Setup the mock to simulate a flaky network
        # Call 1: Raises ConnectionError
        # Call 2: Raises TimeoutError
        # Call 3: Returns a valid DataFrame
        mock_download.side_effect = [
            RuntimeError("Connection lost"),
            TimeoutError("Server busy"),
            pd.DataFrame({"Close": [100, 101]}, index=pd.Index([0, 1])),
        ]

        downloader = YahooDownloader()

        # STEP 2: Execute
        # We expect this to SUCCEED eventually, returning the dataframe
        df = downloader.fetch_history("AAPL", "2023-01-01", "2023-01-05")

        # STEP 3: Verification
        assert not df.empty
        assert len(df) == 2
        # Crucial check: Did it actually retry? Should be called 3 times.
        assert mock_download.call_count == 3
        print("\n[Pass] Retry logic successfully handled 2 network failures.")

    @patch("src.data.data_downloader.yf.download")
    def test_no_retry_on_invalid_symbol(self, mock_download):
        """
        Scenario: The API returns an empty DataFrame (valid connection, bad symbol).
        Expectation: The code should NOT retry. It should fail fast and return empty.
        """
        # STEP 1: Setup mock to return empty DataFrame immediately
        mock_download.return_value = pd.DataFrame()

        downloader = YahooDownloader()

        # STEP 2: Execute
        df = downloader.fetch_history("INVALID_SYMBOL", "2023-01-01", "2023-01-05")

        # STEP 3: Verification
        assert df.empty
        # Crucial check: Should stop after 1 attempt. Retrying a typo is useless.
        assert mock_download.call_count == 1
        print("\n[Pass] Invalid symbol handled immediately without useless retries.")

    @patch("src.data.data_downloader.yf.download")
    def test_max_retries_exceeded(self, mock_download):
        """
        Scenario: The API fails consistently (e.g., permanent outage).
        Expectation: After N retries, the error should finally be raised.
        """
        # STEP 1: Setup mock to ALWAYS fail
        mock_download.side_effect = RuntimeError("Permanent Failure")

        downloader = YahooDownloader()

        # STEP 2 & 3: Execute and expect crash
        # The @retry decorator re-raises the exception after the last attempt.
        with pytest.raises(RuntimeError):
            downloader.fetch_history("AAPL", "2023-01-01", "2023-01-05")

        # Check it tried exactly 3 times (since we set retries=3)
        assert mock_download.call_count == 3
        print("\n[Pass] System correctly gave up after max retries.")
