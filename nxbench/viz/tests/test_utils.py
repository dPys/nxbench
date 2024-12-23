import logging
from unittest.mock import patch

import pandas as pd
import pytest

from nxbench.viz.utils import (
    aggregate_data,
    load_and_prepare_data,
    load_data,
    preprocess_data,
)


@pytest.fixture
def raw_df():
    data = {
        "algorithm": ["bfs", "dfs", None, "bfs"],
        "execution_time": ["0.5", "1.2", "0.7", "2.1"],
        "execution_time_with_preloading": [None, "1.1", None, "2.0"],
        "memory_used": [100.0, 200.0, 150.0, None],
        "num_nodes": [10, 100, 50, 20],
        "num_edges": [50, 500, 250, 100],
        "dataset": ["ds1", "ds2", "ds1", "ds3"],
        "backend": ["networkx", "networkx", "gunrock", "graphx"],
        "is_directed": [False, True, False, True],
        "is_weighted": [False, False, True, True],
        "python_version": ["3.8", "3.9", "3.10", "3.11"],
        "backend_version": ["networkx==2.8", "networkx==2.8", "gunrock==1.0", None],
        "cpu": ["intel", "amd", "intel", "amd"],
        "os": ["linux", "linux", "windows", "mac"],
        "num_thread": ["4", "8", "4", "2"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def csv_file_path(tmp_path, raw_df):
    file_path = tmp_path / "results.csv"
    raw_df.to_csv(file_path, index=False)
    return str(file_path)


def test_load_data(csv_file_path):
    """Test that load_data reads a CSV file and returns a DataFrame."""
    df = load_data(csv_file_path)
    assert not df.empty
    # Check that data types are not coerced here, just loaded
    # e.g. 'execution_time' might still be object before preprocess_data
    assert df["execution_time"].dtype in (object, "O")
    # Check that we "hacked" pd.DataFrame.iteritems -> .items
    assert hasattr(pd.DataFrame, "iteritems")
    assert df.shape[0] == 4


def test_preprocess_data(raw_df):
    """
    Test that preprocess_data correctly cleans up DataFrame columns,
    drops rows with missing essential columns, etc.
    """
    cleaned_df = preprocess_data(raw_df)

    assert len(cleaned_df) == 2

    assert pd.api.types.is_numeric_dtype(cleaned_df["execution_time"])
    assert pd.api.types.is_numeric_dtype(cleaned_df["execution_time_with_preloading"])
    assert pd.api.types.is_numeric_dtype(cleaned_df["memory_used"])

    assert "num_nodes_bin" in cleaned_df.columns
    assert "num_edges_bin" in cleaned_df.columns

    expected_lowercase_cols = [
        "algorithm",
        "dataset",
        "backend",
        "is_directed",
        "is_weighted",
        "backend_version",
        "cpu",
        "os",
    ]
    for col in expected_lowercase_cols:
        if col not in cleaned_df.columns:
            continue
        if not cleaned_df[col].str.contains("[a-zA-Z]").any():
            continue
        assert all(cleaned_df[col].str.islower())


def test_aggregate_data():
    """Test that aggregate_data groups the data and computes the mean values."""
    test_df = pd.DataFrame(
        {
            "algorithm": ["bfs", "bfs", "dfs"],
            "dataset": ["ds1", "ds1", "ds2"],
            "backend": ["networkx", "networkx", "gunrock"],
            "backend_version": ["networkx==2.8", "networkx==2.8", "gunrock==1.0"],
            "num_nodes_bin": ["10 <= x < 50", "10 <= x < 50", "50 <= x < 100"],
            "num_edges_bin": ["50 <= x < 250", "50 <= x < 250", "250 <= x < 500"],
            "is_directed": [False, False, True],
            "is_weighted": [False, False, True],
            "python_version": ["3.8", "3.8", "3.9"],
            "cpu": ["intel", "intel", "amd"],
            "os": ["linux", "linux", "windows"],
            "num_thread": [4, 4, 8],
            "execution_time": [0.5, 0.7, 1.2],
            "execution_time_with_preloading": [0.5, 0.65, 1.1],
            "memory_used": [100.0, 150.0, 200.0],
        }
    )

    df_agg, group_columns, parcats_columns = aggregate_data(test_df)

    assert len(df_agg) == 2
    bfs_agg = df_agg.xs("bfs", level="algorithm")
    assert pytest.approx(bfs_agg["mean_execution_time"].iloc[0], 0.001) == 0.6
    assert pytest.approx(bfs_agg["mean_memory_used"].iloc[0], 0.001) == 125.0
    assert bfs_agg["sample_count"].iloc[0] == 2
    assert pytest.approx(bfs_agg["mean_preload_execution_time"].iloc[0], 0.001) == 0.575

    assert "backend_version" not in group_columns
    assert "backend_full" in group_columns

    assert "dataset" in parcats_columns
    assert "backend_full" in parcats_columns
    assert "python_version" in parcats_columns


@patch("nxbench.viz.utils.load_data")
def test_load_and_prepare_data(mock_load_data, raw_df):
    """
    Test the main orchestrating function to ensure
    it returns data in the correct format.
    """
    mock_load_data.return_value = raw_df

    logger = logging.getLogger("test_logger")
    cleaned_df, df_agg, group_columns, available_parcats_columns = (
        load_and_prepare_data("fake/path.csv", logger)
    )

    # Expect that the function used load_data internally
    mock_load_data.assert_called_once_with("fake/path.csv")

    # Check that data was cleaned
    # We already tested correctness in test_preprocess_data, so minimal checks here
    assert not cleaned_df.empty
    assert "execution_time_with_preloading" in cleaned_df.columns

    # Check aggregated data shape
    assert not df_agg.empty
    # The group columns should include "algorithm" and possibly "backend_full" etc.
    assert "algorithm" in group_columns
    assert len(available_parcats_columns) >= 1


def test_load_data_file_not_found():
    """
    Test that load_data raises FileNotFoundError or pandas.errors.EmptyDataError
    if the file doesn't exist or is invalid.
    """
    with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
        load_data("non_existent_file.csv")


@pytest.mark.parametrize("logger_level", [logging.DEBUG, logging.INFO, logging.WARNING])
def test_load_and_prepare_data_with_different_logger_levels(logger_level, raw_df):
    """
    Just a quick check to ensure that providing different logger levels
    doesn't cause issues in the pipeline.
    """
    logger = logging.getLogger("test_logger")
    logger.setLevel(logger_level)

    # We patch load_data to return our fixture
    with patch("nxbench.viz.utils.load_data", return_value=raw_df):
        cleaned_df, df_agg, group_columns, available_parcats_columns = (
            load_and_prepare_data("any_path.csv", logger)
        )
        # Basic sanity checks
        assert not cleaned_df.empty
        assert not df_agg.empty
        assert len(group_columns) > 0
        assert isinstance(available_parcats_columns, list)
