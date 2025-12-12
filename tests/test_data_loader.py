import pytest
import pandas as pd
from unittest.mock import patch
from src.data_loader import YahooDownloader


class TestYahooDownloader:
    
    @patch('src.data_loader.yf.download')
    def test_invalid_ticker_handling(self, mock_yf_download):
        """
        Test that the downloader returns an empty DataFrame when no data is returned.
        """
        # 1. SETUP: Mock the network call to return an empty DataFrame
        mock_yf_download.return_value = pd.DataFrame()

        # 2. EXECUTE: Stateless class - no __init__, just call fetch_history
        downloader = YahooDownloader()
        result = downloader.fetch_history("INVALID_TICKER", "2020-01-01", "2020-01-05")
        
        # 3. ASSERT: Should return empty DataFrame (not raise exception)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('src.data_loader.yf.download')
    def test_valid_ticker_download(self, mock_yf_download):
        """
        Test that the downloader correctly returns data with MultiIndex when the network call succeeds.
        """
        # 1. SETUP: Create fake market data with MultiIndex (Field, Ticker) structure
        # This simulates what yfinance returns for a single ticker
        mock_data = pd.DataFrame(
            {
                ("Close", "AAPL"): [150.0, 152.0],
                ("Volume", "AAPL"): [1000, 1200]
            },
            index=pd.to_datetime(["2020-01-01", "2020-01-02"])
        )
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        mock_yf_download.return_value = mock_data

        # 2. EXECUTE: Stateless class - call fetch_history and get DataFrame back
        downloader = YahooDownloader()
        result = downloader.fetch_history("AAPL", "2020-01-01", "2020-01-05")

        # 3. ASSERT
        # Check if data was returned
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result.shape == (2, 2)
        
        # Check if MultiIndex structure is correct: (Ticker, Field) after swap
        assert result.columns.nlevels == 2
        assert ("AAPL", "Close") in result.columns
        assert ("AAPL", "Volume") in result.columns

    @patch('src.data_loader.yf.download')
    def test_single_ticker_flat_columns(self, mock_yf_download):
        """
        Test that flat columns from yfinance are wrapped in MultiIndex structure.
        """
        # 1. SETUP: yfinance sometimes returns flat columns for single ticker
        mock_data = pd.DataFrame(
            {
                "Close": [150.0, 152.0],
                "Volume": [1000, 1200]
            },
            index=pd.to_datetime(["2020-01-01", "2020-01-02"])
        )
        mock_yf_download.return_value = mock_data

        # 2. EXECUTE
        downloader = YahooDownloader()
        result = downloader.fetch_history("AAPL", "2020-01-01", "2020-01-05")

        # 3. ASSERT: Should wrap flat columns in MultiIndex
        assert result.columns.nlevels == 2
        assert ("AAPL", "Close") in result.columns
        assert ("AAPL", "Volume") in result.columns