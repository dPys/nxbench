from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg2
import pytest

import nxbench.data.db
from nxbench.benchmarking.config import BenchmarkResult
from nxbench.data.db import BenchmarkDB

nxbench.data.db.psycopg2 = psycopg2


@pytest.fixture
def temp_db_conn_str():
    return "dbname=prefect_db user=prefect_user password=pass host=localhost"


@pytest.fixture
def benchmark_db(temp_db_conn_str):
    return BenchmarkDB(conn_str=temp_db_conn_str)


@pytest.fixture(autouse=True)
def _clear_db(benchmark_db):
    benchmark_db.truncate()
    yield
    benchmark_db.truncate()


@pytest.fixture
def sample_benchmark_result():
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


def test_init_db_creates_schema(temp_db_conn_str):
    db = BenchmarkDB(conn_str=temp_db_conn_str)
    with psycopg2.connect(temp_db_conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = "
                "'public'"
            )
            tables = [row[0] for row in cur.fetchall()]
            assert "benchmarks" in tables


def test_save_and_retrieve_results(benchmark_db, sample_benchmark_result):
    git_commit = "abc123"
    machine_info = {"cpu": "Intel i9"}
    package_versions = {"nxbench": "1.0.0"}
    benchmark_db.save_results(
        sample_benchmark_result,
        git_commit=git_commit,
        machine_info=machine_info,
        package_versions=package_versions,
    )
    results = benchmark_db.get_results(as_pandas=False)
    assert len(results) == 1
    result = results[0]
    assert result["algorithm"] == "test_algo"
    assert result["backend"] == "test_backend"
    assert result["git_commit"] == git_commit
    assert result["machine_info"] == str(machine_info)


def test_get_unique_values(benchmark_db, sample_benchmark_result):
    benchmark_db.save_results(sample_benchmark_result)
    unique_algorithms = benchmark_db.get_unique_values("algorithm")
    assert "test_algo" in unique_algorithms


def test_filter_results_by_date(benchmark_db, sample_benchmark_result):
    timestamp_before_save = datetime.now(timezone.utc).isoformat()
    benchmark_db.save_results(sample_benchmark_result)
    filtered_results = benchmark_db.get_results(
        start_date=timestamp_before_save, as_pandas=False
    )
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


def test_save_multiple_results(benchmark_db, sample_benchmark_result):
    result2 = BenchmarkResult(
        execution_time=2.34,
        execution_time_with_preloading=2.5,
        memory_used=789.01,
        algorithm="test_algo_2",
        backend="test_backend_2",
        dataset="test_dataset_2",
        num_nodes=200,
        num_edges=400,
        num_thread=2,
        date=2345678,
        metadata={},
        is_directed=False,
        is_weighted=True,
        validation="passed",
        validation_message="OK",
    )
    benchmark_db.save_results([sample_benchmark_result, result2])
    results = benchmark_db.get_results(as_pandas=False)
    assert len(results) == 2
    algo_names = {r["algorithm"] for r in results}
    assert "test_algo" in algo_names
    assert "test_algo_2" in algo_names


def test_save_results_with_minimal_data(benchmark_db):
    minimal_result = BenchmarkResult(
        execution_time=0.99,
        execution_time_with_preloading=None,
        memory_used=100.0,
        algorithm="minimal_algo",
        backend="minimal_backend",
        dataset="minimal_dataset",
        num_nodes=10,
        num_edges=20,
        num_thread=1,
        date=999999,
        metadata={},
        is_directed=False,
        is_weighted=False,
        validation=None,
        validation_message=None,
    )
    benchmark_db.save_results(minimal_result)
    results = benchmark_db.get_results(as_pandas=False)
    assert len(results) == 1
    result = results[0]
    assert result["algorithm"] == "minimal_algo"
    assert result["git_commit"] is None
    assert result["machine_info"] is None
    assert result["package_versions"] is None


def test_get_results_with_filters(benchmark_db, sample_benchmark_result):
    result2 = BenchmarkResult(
        execution_time=5.67,
        execution_time_with_preloading=6.5,
        memory_used=999.99,
        algorithm="filtered_algo",
        backend="filtered_backend",
        dataset="another_dataset",
        num_nodes=999,
        num_edges=1998,
        num_thread=4,
        date=999888,
        metadata={},
        is_directed=True,
        is_weighted=True,
        validation="passed",
        validation_message="OK",
    )
    benchmark_db.save_results([sample_benchmark_result, result2])
    filtered_results = benchmark_db.get_results(
        algorithm="filtered_algo", backend="filtered_backend", as_pandas=False
    )
    assert len(filtered_results) == 1
    assert filtered_results[0]["algorithm"] == "filtered_algo"
    assert filtered_results[0]["backend"] == "filtered_backend"


def test_delete_results_by_date(benchmark_db, sample_benchmark_result):
    benchmark_db.save_results(sample_benchmark_result)
    old_date = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    with psycopg2.connect(benchmark_db.conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE benchmarks SET timestamp=%s WHERE id=1", (old_date,))
        conn.commit()
    rows_deleted = benchmark_db.delete_results(
        before_date=datetime.now(timezone.utc).isoformat()
    )
    assert rows_deleted == 1
    remaining = benchmark_db.get_results(as_pandas=False)
    assert len(remaining) == 0
    benchmark_db.save_results(sample_benchmark_result)
    future_date = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    rows_deleted = benchmark_db.delete_results(before_date=future_date)
    assert rows_deleted == 1


def test_delete_results_no_match(benchmark_db, sample_benchmark_result):
    benchmark_db.save_results(sample_benchmark_result)
    rows_deleted = benchmark_db.delete_results(algorithm="nonexistent_algo")
    assert rows_deleted == 0
    remaining = benchmark_db.get_results(as_pandas=False)
    assert len(remaining) == 1


def test_save_results_with_error_and_parameters(benchmark_db, sample_benchmark_result):
    sample_benchmark_result.error = "Test error message"
    benchmark_db.save_results(sample_benchmark_result)
    results = benchmark_db.get_results(as_pandas=False)
    assert len(results) == 1
    result = results[0]
    assert result["error"] == "Test error message"


def test_filter_results_by_start_and_end_date(benchmark_db, sample_benchmark_result):
    benchmark_db.save_results(sample_benchmark_result)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    with psycopg2.connect(benchmark_db.conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE benchmarks SET timestamp=%s WHERE id=1", (old_timestamp,)
            )
        conn.commit()
    benchmark_db.save_results(sample_benchmark_result)
    new_timestamp = datetime.now(timezone.utc).isoformat()
    results = benchmark_db.get_results(
        start_date=old_timestamp, end_date=new_timestamp, as_pandas=False
    )
    assert len(results) == 2
    middle_timestamp = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    results = benchmark_db.get_results(
        start_date=middle_timestamp, end_date=new_timestamp, as_pandas=False
    )
    assert len(results) == 1


def test_directed_and_weighted_flags_are_integers(
    benchmark_db, sample_benchmark_result
):
    sample_benchmark_result.is_directed = True
    sample_benchmark_result.is_weighted = True
    benchmark_db.save_results(sample_benchmark_result)
    results = benchmark_db.get_results(as_pandas=False)
    assert len(results) == 1
    result = results[0]
    assert result["directed"] == 1
    assert result["weighted"] == 1
    sample_benchmark_result.is_directed = False
    sample_benchmark_result.is_weighted = False
    benchmark_db.save_results(sample_benchmark_result)
    results = benchmark_db.get_results(as_pandas=False)
    assert len(results) == 2
    new_record = results[1]
    assert new_record["directed"] == 0
    assert new_record["weighted"] == 0
