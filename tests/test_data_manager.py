# tests/test_data_manager.py
import pandas as pd
import pytest
from pathlib import Path
from src.data_manager import DataManager


# This fixture creates a simple DataFrame for testing
@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture to provide a consistent DataFrame for testing."""
    data = {
        "Open": [100.0, 101.0, 102.0],
        "High": [102.0, 103.0, 104.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [101.5, 102.5, 103.5],
        "Volume": [100000, 200000, 300000],
    }
    dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


# The tmp_path fixture is provided by pytest and creates a temporary
# directory unique to the test invocation.
def test_data_manager_initialization(tmp_path: Path):
    """Test that the DataManager correctly initializes and creates the directory."""
    manager = DataManager(base_dir=str(tmp_path / "test_dir"))
    assert manager.base_dir.exists()
    assert manager.base_dir.is_dir()


def test_save_and_load_ticker(tmp_path: Path, sample_dataframe: pd.DataFrame):
    """Test the full save-load cycle."""
    manager = DataManager(base_dir=str(tmp_path / "storage"))
    ticker = "TSLA"

    # 1. Test Save (Atomic Write)
    saved_path = manager.save_ticker(ticker, sample_dataframe)

    # Check return value and existence
    assert isinstance(saved_path, Path)
    assert saved_path.exists()
    assert saved_path.is_dir()
    assert saved_path.name == "TSLA"

    partition_dir = saved_path / "year=2023" / "month=01"
    # Check for temporary file (should NOT exist after successful save)
    temp_path = partition_dir / "data.parquet.tmp"
    assert not temp_path.exists(), "Temporary file should be removed or renamed."

    file_path = partition_dir / "data.parquet"
    assert file_path.exists(), "The partition file was created"
    # 2. Test Load
    df_loaded = manager.load_ticker(ticker)
    df_loaded = df_loaded[sample_dataframe.columns]
    # Check data integrity
    pd.testing.assert_frame_equal(sample_dataframe, df_loaded, check_freq=False)


def test_save_empty_dataframe_raises_error(tmp_path: Path):
    """Test that saving an empty DataFrame raises a ValueError."""
    manager = DataManager(base_dir=str(tmp_path / "storage"))
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Cannot save empty dataframe"):
        manager.save_ticker("EMPTY", empty_df)


def test_load_non_existent_ticker_raises_error(tmp_path: Path):
    """Test that attempting to load a non-existent file raises FileNotFoundError."""
    manager = DataManager(base_dir=str(tmp_path / "storage"))

    with pytest.raises(FileNotFoundError, match="not found"):
        manager.load_ticker("NONEXISTENT")


def test_list_existing_tickers(tmp_path: Path, sample_dataframe: pd.DataFrame):
    """Test that the manager correctly lists all saved tickers."""
    manager = DataManager(base_dir=str(tmp_path / "storage"))

    # Save a few tickers
    manager.save_ticker("AAPL", sample_dataframe)
    manager.save_ticker("MSFT", sample_dataframe)
    manager.save_ticker("GOOGL", sample_dataframe)

    # List and check the result (should be sorted)
    tickers = manager.list_existing_tickers()
    assert tickers == ["AAPL", "GOOGL", "MSFT"]
    assert len(tickers) == 3


# Optional: A Test for Atomic Failure Simulation (Advanced)
# This test is crucial for verifying the robustness of the atomic write,
# ensuring that the temporary file is cleaned up if an exception occurs during the write process.
def test_atomic_write_cleanup_on_failure(tmp_path: Path, mocker):
    """Simulate a failure during the to_parquet call and check for cleanup."""
    manager = DataManager(base_dir=str(tmp_path / "fail_test"))
    ticker = "FAIL"
    df = pd.DataFrame({"col": [1]}, index=pd.to_datetime(["2023-01-01"]))  # Simple DataFrame

    partition_dir = manager.base_dir / ticker / "year=2023" / "month=01"
    temp_path = partition_dir / "data.parquet.tmp"
    final_path = partition_dir / "data.parquet"

    # Mock the to_parquet function to raise an error
    # We must ensure the .tmp file is created BEFORE the mock raises the error.

    def mocked_to_parquet(*args, **kwargs):
        partition_dir.mkdir(parents=True, exist_ok=True) 
        # Manually create the temporary file before raising
        temp_path.touch()
        raise OSError("Simulated Disk Full Error")

    # Patch the pandas to_parquet method to use our mocked function
    mocker.patch.object(pd.DataFrame, "to_parquet", side_effect=mocked_to_parquet)

    with pytest.raises(RuntimeError, match="Failed to save"):
        manager.save_ticker(ticker, df)

    # Crucial check: The final file should NOT exist, and the temporary file should be cleaned up.
    assert not temp_path.exists(), "Temporary file was not cleaned up after failure."
    assert not final_path.exists(), "Final file should not exist after failure."


# tests/test_data_manager.py (Add this function to the end)


def test_load_corrupt_ticker_raises_runtime_error(tmp_path: Path):
    """
    Test the error path for corrupted data (lines 68-69 of data_manager.py).
    Simulates a file that exists but cannot be read as Parquet.
    """
    manager = DataManager(base_dir=str(tmp_path / "storage"))
    ticker = "CORRUPT"
    partition_dir = manager.base_dir / ticker / "year=2023" / "month=01"
    file_path: Path = partition_dir / "data.parquet"
    partition_dir.mkdir(parents = True, exist_ok = True)

    # 1. Create a file that exists but contains non-Parquet (corrupt) data
    # We write simple text data, which pandas will fail to parse as Parquet.
    file_path.write_text("This is not valid Parquet binary data.")

    # 2. Attempt to load the file
    # pd.read_parquet will raise an internal error (like pyarrow.lib.ArrowInvalid)
    # which our DataManager catches and re-raises as a RuntimeError.
    with pytest.raises(RuntimeError, match="Corrupt data"):
        manager.load_ticker(ticker)
