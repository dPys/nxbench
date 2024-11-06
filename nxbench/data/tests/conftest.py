import tempfile
import textwrap
from pathlib import Path

import pytest

from nxbench.data.loader import BenchmarkDataManager


@pytest.fixture
def mock_metadata():
    """Fixture to mock the _load_metadata method of BenchmarkDataManager."""
    from unittest.mock import patch

    import pandas as pd

    from nxbench.data.loader import BenchmarkDataManager

    def _mock_load_metadata(self):
        data = {
            "name": [
                "jazz",
                "08blocks",
                "patentcite",
                "imdb",
                "citeseer",
                "mixed_delimiters",
                "invalid_weights",
                "self_loops_duplicates",
                "non_sequential_ids",
                "example",
                "extra_columns",
                "twitter",
            ],
            "download_url": [
                "http://example.com/jazz.zip",
                "http://example.com/08blocks.zip",
                "http://example.com/patentcite.zip",
                "http://example.com/imdb.zip",
                "http://example.com/citeseer.zip",
                "http://example.com/mixed_delimiters.zip",
                "http://example.com/invalid_weights.zip",
                "http://example.com/self_loops_duplicates.zip",
                "http://example.com/non_sequential_ids.zip",
                "http://example.com/example.zip",
                "http://example.com/extra_columns.zip",
                "http://example.com/twitter.zip",
            ],
            "directed": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "weighted": [
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
            ],
        }
        df = pd.DataFrame(data)
        return df

    with patch.object(BenchmarkDataManager, "_load_metadata", _mock_load_metadata):
        yield


@pytest.fixture
def data_manager(mock_metadata):
    """Fixture for initializing BenchmarkDataManager with a temporary data directory and mocked metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield BenchmarkDataManager(data_dir=temp_dir)


@pytest.fixture
def create_edge_file(data_manager):
    """Fixture to create edge list files with specified content."""

    def _create_edge_file(filename: str, content: str):
        file_path = Path(data_manager.data_dir) / filename
        with file_path.open("w") as f:
            f.write(textwrap.dedent(content).lstrip())
        return file_path

    return _create_edge_file
