import pytest
import pandas as pd
from unittest.mock import patch
from src.data_loader import YahooDownloader

class TestYahooDownloader:
    
    @patch('src.data_loader.yf.download')
    def test_invalid_ticker_handling(self, mock_yf_download):
        """
        Test that the downloader raises RuntimeError when no data is returned.
        """
        
        # 1. SETUP: Mock the network call to return an empty DataFrame
        mock_yf_download.return_value = pd.DataFrame()

        # Initialize with dummy data (parameters are stored in self)
        downloader = YahooDownloader(["INVALID_TICKER"], "2020-01-01", "2020-01-05")
        
        # 2. EXECUTE & ASSERT
        # Your code catches EmptyDataError and raises RuntimeError
        # We expect "No data found" in the error message
        with pytest.raises(RuntimeError, match="No data found"):
            downloader.fetch()


    @patch('src.data_loader.yf.download')
    def test_valid_ticker_download(self, mock_yf_download):
        """
        Test that the downloader correctly stores data when the network call succeeds.
        """
        # 1. SETUP: Create fake market data (FLAT columns, not MultiIndex)
        # This simulates what yfinance returns for a single ticker before your code processes it.
        mock_data = pd.DataFrame(
            {
                "Close": [150.0, 152.0],
                "Volume": [1000, 1200]
            },
            index=pd.to_datetime(["2020-01-01", "2020-01-02"])
        )
        mock_yf_download.return_value = mock_data

        # 2. EXECUTE
        downloader = YahooDownloader(["AAPL"], "2020-01-01", "2020-01-05")
        downloader.fetch()

        # 3. ASSERT
        # Check if data was stored
        assert downloader.data is not None
        assert not downloader.data.empty
        
        # Now we check if YOUR code successfully added the 'AAPL' level
        # The resulting shape should be (2 rows, 2 columns)
        # And the columns should now be MultiIndex: ('AAPL', 'Close'), ('AAPL', 'Volume')
        assert downloader.data.columns.nlevels == 2
        assert ("AAPL", "Close") in downloader.data.columns