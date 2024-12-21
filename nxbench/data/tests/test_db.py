import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pytest

from nxbench.benchmarks.config import BenchmarkResult
from nxbench.data.db import BenchmarkDB


@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture for a temporary database path."""
    return tmp_path / "test_benchmarks.db"


@pytest.fixture
def benchmark_db(temp_db_path):
    """Fixture for initializing the BenchmarkDB."""
    return BenchmarkDB(db_path=temp_db_path)


@pytest.fixture
def sample_benchmark_result():
    """Fixture for a sample BenchmarkResult object."""
    return BenchmarkResult(
        execution_time=1.23,
        execution_time_with_preloading=1.5,
        memory_used=456.78,
        algorithm="test_algo",
        backend="test_backend",
        dataset="test_dataset",
        num_nodes=100,
        num_edges=200,
        num_thread=1,
        date=1234567,
        metadata={},
        is_directed=True,
        is_weighted=False,
        validation="passed",
        validation_message="OK",
    )


def test_init_db_creates_schema(temp_db_path):
    db = BenchmarkDB(db_path=temp_db_path)
    with sqlite3.connect(temp_db_path) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "benchmarks" in tables


def test_save_and_retrieve_results(benchmark_db, sample_benchmark_result):
    git_commit = "abc123"
    machine_info = {"cpu": "Intel i9"}
    python_version = "3.10.0"
    package_versions = {"nxbench": "1.0.0"}

    benchmark_db.save_results(
        sample_benchmark_result,
        git_commit=git_commit,
        machine_info=machine_info,
        python_version=python_version,
        package_versions=package_versions,
    )

    results = benchmark_db.get_results(as_pandas=False)
    assert len(results) == 1
    result = results[0]
    assert result["algorithm"] == "test_algo"
    assert result["backend"] == "test_backend"
    assert result["git_commit"] == git_commit
    assert result["machine_info"] == str(machine_info)
    assert result["python_version"] == python_version


def test_get_unique_values(benchmark_db, sample_benchmark_result):
    benchmark_db.save_results(sample_benchmark_result)
    unique_algorithms = benchmark_db.get_unique_values("algorithm")
    assert "test_algo" in unique_algorithms


def test_filter_results_by_date(benchmark_db, sample_benchmark_result):
    timestamp = datetime.now(timezone.utc).isoformat()
    benchmark_db.save_results(sample_benchmark_result)
    filtered_results = benchmark_db.get_results(start_date=timestamp, as_pandas=False)
    assert len(filtered_results) == 1


def test_delete_results(benchmark_db, sample_benchmark_result):
    benchmark_db.save_results(sample_benchmark_result)
    rows_deleted = benchmark_db.delete_results(algorithm="test_algo")
    assert rows_deleted == 1
    remaining_results = benchmark_db.get_results(as_pandas=False)
    assert len(remaining_results) == 0


def test_get_results_as_pandas(benchmark_db, sample_benchmark_result):
    benchmark_db.save_results(sample_benchmark_result)
    results_df = benchmark_db.get_results(as_pandas=True)
    assert isinstance(results_df, pd.DataFrame)
    assert not results_df.empty
    assert "algorithm" in results_df.columns


def test_invalid_column_unique_values(benchmark_db):
    with pytest.raises(ValueError, match="Invalid column name: nonexistent_column"):
        benchmark_db.get_unique_values("nonexistent_column")
