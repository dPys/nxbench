import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from nxbench.benchmarking.config import BenchmarkResult
from nxbench.benchmarking.export import ResultsExporter


@pytest.fixture(autouse=True)
def mock_benchmark_data_manager_metadata():
    """Globally patches BenchmarkDataManager._load_metadata"""
    with patch("nxbench.data.loader.BenchmarkDataManager._load_metadata") as mock_ld:
        mock_ld.return_value = pd.DataFrame({"name": ["dummy_dataset"]})
        yield mock_ld


@pytest.fixture
def mock_logger():
    with patch("nxbench.benchmarking.export.logger") as mock_log:
        yield mock_log


class TestResultsExporter:
    def test_init_results_file(self, mock_logger):
        """Ensure the exporter initializes correctly with a results file."""
        fake_path = Path("/fake/path/results.json")
        exporter = ResultsExporter(results_file=fake_path)
        assert exporter.results_file == fake_path
        mock_logger.debug.assert_not_called()

    @patch("nxbench.benchmarking.export.Path.open", new_callable=mock_open)
    def test_load_results_json_valid(self, mock_file_open, mock_logger):
        mock_json = [
            {
                "algorithm": "alg1",
                "dataset": "ds1",
                "execution_time": 1.23,
                "execution_time_with_preloading": 2.34,
                "memory_used": 100.0,
                "num_nodes": 50,
                "num_edges": 200,
                "is_directed": False,
                "is_weighted": True,
                "backend": "backend1",
                "num_thread": 4,
                "date": 20240101,
                "validation": "passed",
                "validation_message": "",
                "error": None,
                "extra_field": "metadata_value",
            }
        ]
        mock_file_open.return_value.__enter__.return_value.read.return_value = (
            json.dumps(mock_json)
        )

        exporter = ResultsExporter(results_file=Path("results.json"))
        results = exporter.load_results()
        assert len(results) == 1
        res = results[0]
        assert res.algorithm == "alg1"
        assert res.dataset == "ds1"
        assert res.metadata.get("extra_field") == "metadata_value"
        mock_logger.info.assert_any_call("Loaded 1 benchmark results from results.json")

    @patch("nxbench.benchmarking.export.Path.open", new_callable=mock_open)
    def test_load_results_json_not_list(self, mock_file_open, mock_logger):
        mock_json = {"foo": "bar"}  # Not a list
        mock_file_open.return_value.__enter__.return_value.read.return_value = (
            json.dumps(mock_json)
        )

        exporter = ResultsExporter(results_file=Path("results.json"))
        results = exporter.load_results()
        assert results == []
        mock_logger.error.assert_any_call(
            "Expected a list of results in JSON file, got <class 'dict'>"
        )

    @patch("nxbench.benchmarking.export.Path.open", new_callable=mock_open)
    def test_load_results_json_malformed(self, mock_file_open, mock_logger):
        mock_file_open.return_value.__enter__.return_value.read.return_value = (
            "invalid-json"
        )

        exporter = ResultsExporter(results_file=Path("results.json"))
        results = exporter.load_results()
        assert results == []
        mock_logger.exception.assert_any_call(
            f"Failed to load results from: {exporter.results_file}"
        )

    @patch("pandas.read_csv")
    def test_load_results_csv_valid(self, mock_read_csv, mock_logger):
        df = pd.DataFrame(
            {
                "algorithm": ["alg2", "alg3"],
                "dataset": ["ds2", "ds3"],
                "execution_time": [2.34, 3.45],
                "execution_time_with_preloading": [3.45, 4.56],
                "memory_used": [200.0, 300.0],
                "num_nodes": [100, 150],
                "num_edges": [500, 600],
                "is_directed": [True, False],
                "is_weighted": [False, True],
                "backend": ["backend2", "backend3"],
                "num_thread": [2, 4],
                "date": [20240102, 20240103],
                "validation": ["passed", "failed"],
                "validation_message": ["", "invalid result"],
                "error": [None, "Error info"],
                "extra_field": ["some metadata", "another metadata"],
            }
        )
        mock_read_csv.return_value = df

        exporter = ResultsExporter(results_file=Path("results.csv"))
        results = exporter.load_results()
        assert len(results) == 2
        first, second = results
        assert first.algorithm == "alg2"
        assert second.error == "Error info"
        mock_logger.info.assert_any_call("Loaded 2 benchmark results from results.csv")

    def test_load_results_unsupported_suffix(self, mock_logger):
        """Test loading results with an unsupported suffix."""
        exporter = ResultsExporter(results_file=Path("results.txt"))
        results = exporter.load_results()
        assert results == []
        mock_logger.exception.assert_not_called()

    def test_to_dataframe_no_results(self, mock_logger):
        """Test to_dataframe when load_results is empty."""
        exporter = ResultsExporter(results_file=Path("missing.json"))
        with patch.object(exporter, "load_results", return_value=[]):
            with pytest.raises(ValueError, match="No benchmark results found"):
                exporter.to_dataframe()

    def test_to_dataframe_success(self, mock_logger):
        """Test to_dataframe with real BenchmarkResult objects."""
        exporter = ResultsExporter(results_file=Path("dummy.csv"))
        res1 = BenchmarkResult(
            algorithm="algA",
            dataset="dsA",
            execution_time=1.0,
            execution_time_with_preloading=1.5,
            memory_used=123.0,
            num_nodes=10,
            num_edges=20,
            is_directed=False,
            is_weighted=False,
            backend="backendA",
            num_thread=2,
            date=20241231,
            metadata={"key1": "value1"},
            validation="passed",
            validation_message="OK",
        )
        res2 = BenchmarkResult(
            algorithm="algB",
            dataset="dsB",
            execution_time=2.0,
            execution_time_with_preloading=2.5,
            memory_used=456.0,
            num_nodes=30,
            num_edges=60,
            is_directed=True,
            is_weighted=True,
            backend="backendB",
            num_thread=4,
            date=20250101,
            metadata={"key2": "value2", "key3": "value3"},
            validation="failed",
            validation_message="Mismatch",
            error="SomeError",
        )

        with patch.object(exporter, "load_results", return_value=[res1, res2]):
            df = exporter.to_dataframe()
            assert len(df) == 2
            assert "algorithm" in df.columns
            assert "execution_time_with_preloading" in df.columns
            assert df.loc[0, "algorithm"] == "algA"
            assert df.loc[1, "algorithm"] == "algB"
            assert df.loc[0, "key1"] == "value1"
            assert df.loc[1, "key3"] == "value3"

    @patch("nxbench.benchmarking.export.BenchmarkDB")
    def test_export_results_csv(self, mock_db, mock_logger):
        """Test exporting results as CSV with a mock DataFrame."""
        exporter = ResultsExporter(results_file=Path("results.json"))
        df_mock = MagicMock(spec=pd.DataFrame)
        df_mock.to_csv = MagicMock()

        with patch.object(exporter, "to_dataframe", return_value=df_mock):
            exporter.export_results(Path("/tmp/out.csv"), form="csv")

            df_mock.to_csv.assert_called_once_with(Path("/tmp/out.csv"), index=False)
            mock_logger.info.assert_called_with("Exported results to CSV: /tmp/out.csv")

    @patch("nxbench.benchmarking.export.BenchmarkDB")
    def test_export_results_json(self, mock_db, mock_logger):
        """Test exporting results as JSON with a mock DataFrame."""
        exporter = ResultsExporter(results_file=Path("results.json"))
        df_mock = MagicMock(spec=pd.DataFrame)
        df_mock.to_json = MagicMock()

        with patch.object(exporter, "to_dataframe", return_value=df_mock):
            exporter.export_results(Path("/tmp/out.json"), form="json")
            df_mock.to_json.assert_called_once_with(
                Path("/tmp/out.json"), orient="records", indent=2
            )
            mock_logger.info.assert_called_with(
                "Exported results to JSON: /tmp/out.json"
            )

    @patch("nxbench.benchmarking.export.BenchmarkDB")
    def test_export_results_sql_replace(self, mock_pyver, mock_db_class, mock_logger):
        """Test exporting results into SQL with 'replace'."""
        exporter = ResultsExporter(results_file=Path("results.csv"))
        mock_db_instance = mock_db_class.return_value

        mock_results = [
            BenchmarkResult(
                algorithm="algSQL",
                dataset="dsSQL",
                execution_time=1.0,
                execution_time_with_preloading=1.2,
                memory_used=100.0,
                num_nodes=10,
                num_edges=20,
                is_directed=False,
                is_weighted=False,
                backend="backendSQL",
                num_thread=2,
                date=20240101,
                metadata={},
            )
        ]
        with patch.object(exporter, "load_results", return_value=mock_results):
            exporter.export_results(
                Path("database.db"), form="sql", if_exists="replace"
            )
            mock_db_class.assert_called_with(Path("database.db"))
            mock_db_instance.delete_results.assert_called_once()
            mock_db_instance.save_results.assert_called_once_with(
                results=mock_results,
                machine_info={},
            )
            mock_logger.info.assert_called_with(
                "Exported results to SQL database: database.db"
            )

    def test_export_results_unsupported_format(self):
        """Test that exporting fails with an unsupported format."""
        exporter = ResultsExporter(results_file=Path("results.csv"))
        # Ensure not to fail from "No benchmark results" first
        with patch.object(
            exporter, "to_dataframe", return_value=pd.DataFrame({"x": [1]})
        ):
            with pytest.raises(ValueError, match="Unsupported export format: parquet"):
                exporter.export_results(Path("output.parquet"), form="parquet")

    def test_query_results_no_filters(self):
        """Test query_results with no filters, returning the full DataFrame."""
        exporter = ResultsExporter(results_file=Path("results.csv"))
        mock_df = pd.DataFrame(
            {
                "algorithm": ["a1", "a2"],
                "dataset": ["ds1", "ds2"],
                "backend": ["b1", "b2"],
                # For now, store as integers or strings that won't break sort
                "date": [20240101, 20240102],
            }
        )
        with patch.object(exporter, "to_dataframe", return_value=mock_df):
            df_result = exporter.query_results()
            assert len(df_result) == 2
            # Sorted by ["algorithm", "dataset", "backend"]
            assert df_result.iloc[0]["algorithm"] == "a1"
            assert df_result.iloc[1]["algorithm"] == "a2"

    def test_query_results_with_filters(self):
        """Test query_results with date range as numeric ints or parseable strings."""
        exporter = ResultsExporter(results_file=Path("results.json"))
        # Store date as int so we can compare with int range or parse it in code
        mock_df = pd.DataFrame(
            {
                "algorithm": ["alg1", "alg2", "alg1", "alg2"],
                "dataset": ["dsA", "dsA", "dsB", "dsB"],
                "backend": ["b1", "b2", "b1", "b2"],
                "date": [20240101, 20240102, 20240103, 20240104],
            }
        )
        with patch.object(exporter, "to_dataframe", return_value=mock_df):
            # Filter algorithm="alg1", dataset="dsB"
            filtered = exporter.query_results(algorithm="alg1", dataset="dsB")
            assert len(filtered) == 1
            assert filtered.iloc[0]["algorithm"] == "alg1"
            assert filtered.iloc[0]["dataset"] == "dsB"

            # Filter backend="b2"
            filtered_b2 = exporter.query_results(backend="b2")
            assert len(filtered_b2) == 2
            assert all(filtered_b2["backend"] == "b2")

            with patch("nxbench.benchmarking.export.pd.to_datetime") as mock_td:
                mock_td.side_effect = lambda x: int(
                    x.replace("-", "")
                )  # "2024-01-02" -> 20240102
                filtered_date = exporter.query_results(
                    date_range=("2024-01-02", "2024-01-03")
                )
                assert len(filtered_date) > 0
