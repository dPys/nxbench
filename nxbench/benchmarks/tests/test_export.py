import json
import math
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import networkx as nx
import pandas as pd
import pytest

from nxbench.benchmarks.config import BenchmarkResult, MachineInfo
from nxbench.benchmarks.export import ResultsExporter


@pytest.fixture
def mock_benchmark_config():
    return MagicMock()


@pytest.fixture
def mock_data_manager():
    with patch("nxbench.benchmarks.export.BenchmarkDataManager") as mock_dm:
        yield mock_dm.return_value


@pytest.fixture
def mock_benchmark_db():
    with patch("nxbench.benchmarks.export.BenchmarkDB") as mock_db:
        yield mock_db.return_value


@pytest.fixture
def mock_get_benchmark_config():
    with patch("nxbench.benchmarks.export.get_benchmark_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(datasets=[])
        yield mock_cfg


@pytest.fixture
def mock_get_python_version():
    with patch("nxbench.benchmarks.export.get_python_version") as mock_pv:
        mock_pv.return_value = "3.8.10"
        yield mock_pv


@pytest.fixture
def mock_logger():
    with patch("nxbench.benchmarks.export.logger") as mock_log:
        yield mock_log


@pytest.fixture
def mock_machine_info():
    return MachineInfo(
        arch="x86_64",
        cpu="Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz",
        machine="Machine1",
        num_cpu="8",
        os="Ubuntu 20.04",
        ram="16GB",
        version=1,
    )


def create_machine_json(machine_info: MachineInfo):
    return json.dumps(
        {
            "arch": machine_info.arch,
            "cpu": machine_info.cpu,
            "machine": machine_info.machine,
            "num_cpu": machine_info.num_cpu,
            "os": machine_info.os,
            "ram": machine_info.ram,
            "version": machine_info.version,
        }
    )


class TestResultsExporter:
    def create_mock_benchmark_result(self, **kwargs):
        mock_result = MagicMock(spec=BenchmarkResult)
        mock_result.algorithm = kwargs.get("algorithm", "alg1")
        mock_result.dataset = kwargs.get("dataset", "dataset1")
        mock_result.backend = kwargs.get("backend", "backend1")
        mock_result.execution_time = kwargs.get("execution_time", 1.23)
        mock_result.memory_used = kwargs.get("memory_used", 456.78)
        mock_result.num_thread = kwargs.get("num_thread", 4)
        mock_result.num_nodes = kwargs.get("num_nodes", 100)
        mock_result.num_edges = kwargs.get("num_edges", 200)
        mock_result.is_directed = kwargs.get("is_directed", False)
        mock_result.is_weighted = kwargs.get("is_weighted", False)
        mock_result.commit_hash = kwargs.get("commit_hash", "abc123")
        mock_result.metadata = kwargs.get("metadata", {"key": "value"})
        return mock_result

    @patch("nxbench.benchmarks.export.Path.iterdir")
    @patch("nxbench.benchmarks.export.Path.is_dir")
    def test_init_with_machine_dirs(
        self,
        mock_is_dir,
        mock_iterdir,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_logger,
        mock_machine_info,
    ):
        machine_dir = Path("/fake/results/machine1")
        mock_iterdir.return_value = [machine_dir]
        mock_is_dir.return_value = True

        with patch.object(
            ResultsExporter, "_load_machine_info", return_value=mock_machine_info
        ):
            exporter = ResultsExporter(results_dir="/fake/results")

            assert exporter.results_dir == Path("/fake/results")
            assert exporter._machine_dir == machine_dir
            assert exporter._machine_info == mock_machine_info
            mock_logger.debug.assert_called_with(
                f"Machine Information: {mock_machine_info}"
            )

    @patch("nxbench.benchmarks.export.Path.iterdir")
    @patch("nxbench.benchmarks.export.Path.is_dir")
    def test_init_no_machine_dirs(
        self,
        mock_is_dir,
        mock_iterdir,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_logger,
    ):
        mock_iterdir.return_value = []
        exporter = ResultsExporter(results_dir="/fake/results")

        assert exporter.results_dir == Path("/fake/results")
        assert exporter._machine_dir is None
        assert exporter._machine_info is None
        mock_logger.warning.assert_called_with(
            "No machine directories found in /fake/results"
        )

    @patch("nxbench.benchmarks.export.Path.iterdir")
    @patch("nxbench.benchmarks.export.Path.is_dir")
    def test_init_with_invalid_machine_json(
        self,
        mock_is_dir,
        mock_iterdir,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_logger,
    ):
        machine_dir = Path("/fake/results/machine1")
        mock_iterdir.return_value = [machine_dir]
        mock_is_dir.return_value = True

        with patch.object(ResultsExporter, "_load_machine_info", return_value=None):
            exporter = ResultsExporter(results_dir="/fake/results")

            assert exporter.results_dir == Path("/fake/results")
            assert exporter._machine_dir == machine_dir
            assert exporter._machine_info is None
            mock_logger.warning.assert_called_with(
                "Machine information could not be loaded."
            )

    @patch(
        "nxbench.benchmarks.export.Path.open",
        new_callable=mock_open,
        read_data="invalid json",
    )
    @patch("nxbench.benchmarks.export.Path.is_dir")
    @patch("nxbench.benchmarks.export.Path.iterdir")
    def test_load_machine_info_json_decode_error(
        self,
        mock_iterdir,
        mock_is_dir,
        mock_open_file,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_machine_info,
    ):
        machine_dir = Path("/fake/results/machine1")
        mock_iterdir.return_value = [machine_dir]
        mock_is_dir.return_value = True

        exporter = ResultsExporter(results_dir="/fake/results")

        with patch(
            "nxbench.benchmarks.export.json.load",
            side_effect=json.JSONDecodeError("Expecting value", "", 0),
        ):
            machine_info = exporter._load_machine_info()
            assert machine_info is None
            mock_logger.warning.assert_called_with(
                f"Failed to load machine information from "
                f"{machine_dir / 'machine.json'}",
                exc_info=True,
            )

    @patch(
        "nxbench.benchmarks.export.Path.open",
        new_callable=mock_open,
        read_data='{"invalid_field": "value"}',
    )
    @patch("nxbench.benchmarks.export.Path.is_dir")
    @patch("nxbench.benchmarks.export.Path.iterdir")
    def test_load_machine_info_type_error(
        self,
        mock_iterdir,
        mock_is_dir,
        mock_open_file,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_machine_info,
    ):
        machine_dir = Path("/fake/results/machine1")
        mock_iterdir.return_value = [machine_dir]
        mock_is_dir.return_value = True

        exporter = ResultsExporter(results_dir="/fake/results")

        machine_info = exporter._load_machine_info()
        assert machine_info is None
        mock_logger.warning.assert_called_with("MachineInfo structure mismatch")

    def test_parse_measurement_none(self, mock_logger):
        exporter = ResultsExporter(results_dir=".")
        execution_time, memory_used = exporter._parse_measurement(None)
        assert math.isnan(execution_time)
        assert math.isnan(memory_used)

    def test_parse_measurement_dict_valid(self, mock_logger):
        exporter = ResultsExporter(results_dir=".")
        measurement = {"execution_time": 1.23, "memory_used": 456.78}
        execution_time, memory_used = exporter._parse_measurement(measurement)
        assert execution_time == 1.23
        assert memory_used == 456.78

    def test_parse_measurement_dict_missing_fields(self, mock_logger):
        exporter = ResultsExporter(results_dir=".")
        measurement = {"execution_time": 1.23}
        execution_time, memory_used = exporter._parse_measurement(measurement)
        assert execution_time == 1.23
        assert math.isnan(memory_used)

    def test_parse_measurement_dict_invalid_types(self, mock_logger):
        exporter = ResultsExporter(results_dir=".")
        measurement = {"execution_time": "fast", "memory_used": None}
        execution_time, memory_used = exporter._parse_measurement(measurement)
        assert math.isnan(execution_time)
        assert math.isnan(memory_used)

    def test_parse_measurement_int(self, mock_logger):
        exporter = ResultsExporter(results_dir=".")
        measurement = 2
        execution_time, memory_used = exporter._parse_measurement(measurement)
        assert execution_time == 2.0
        assert memory_used == 0.0

    def test_parse_measurement_float(self, mock_logger):
        exporter = ResultsExporter(results_dir=".")
        measurement = 3.14
        execution_time, memory_used = exporter._parse_measurement(measurement)
        assert execution_time == 3.14
        assert memory_used == 0.0

    def test_parse_measurement_invalid_type(self, mock_logger):
        exporter = ResultsExporter(results_dir=".")
        measurement = "invalid"
        execution_time, memory_used = exporter._parse_measurement(measurement)
        assert math.isnan(execution_time)
        assert math.isnan(memory_used)

    @patch("nxbench.benchmarks.export.BenchmarkDataManager")
    def test_create_benchmark_result_valid(
        self, mock_bm_data_manager, mock_logger, mock_machine_info
    ):
        mock_bm_data_manager.return_value.load_network_sync.return_value = (
            nx.Graph(),
            {"meta": "data"},
        )

        exporter = ResultsExporter(results_dir=".")
        exporter._machine_info = mock_machine_info
        mock_dataset = MagicMock(name="dataset1")
        mock_dataset.name = "dataset1"
        exporter.benchmark_config = MagicMock(datasets=[mock_dataset])

        mock_result = self.create_mock_benchmark_result()

        with patch.object(
            BenchmarkResult, "from_asv_result", return_value=mock_result
        ) as mock_from_asv_result:
            result = exporter._create_benchmark_result(
                algorithm="alg1",
                dataset="dataset1",
                backend="backend1",
                execution_time=1.23,
                memory_used=456.78,
                num_thread=None,
                commit_hash="abc123",
                date=20240101,
            )
            assert isinstance(result, MagicMock)  # Since we returned a MagicMock
            mock_bm_data_manager.return_value.load_network_sync.assert_called_once()
            mock_from_asv_result.assert_called_once()

    @patch("nxbench.benchmarks.export.BenchmarkDataManager")
    def test_create_benchmark_result_no_dataset_config(
        self, mock_bm_data_manager, mock_logger, mock_machine_info
    ):
        mock_bm_data_manager.return_value.load_network_sync.return_value = (
            nx.Graph(),
            {"meta": "data"},
        )

        exporter = ResultsExporter(results_dir=".")
        exporter._machine_info = mock_machine_info
        exporter.benchmark_config = MagicMock(datasets=[])

        mock_benchmark_result = MagicMock(spec=BenchmarkResult)

        with patch.object(
            BenchmarkResult, "from_asv_result", return_value=mock_benchmark_result
        ) as mock_from_asv_result:
            result = exporter._create_benchmark_result(
                algorithm="alg1",
                dataset="dataset1",
                backend="backend1",
                execution_time=1.23,
                memory_used=456.78,
                num_thread=4,
                commit_hash="abc123",
                date=20240101,
            )
            assert isinstance(result, MagicMock)  # Since we returned a MagicMock
            mock_bm_data_manager.return_value.load_network_sync.assert_not_called()
            mock_from_asv_result.assert_called_once()

    @patch("nxbench.benchmarks.export.BenchmarkDataManager")
    def test_create_benchmark_result_load_network_exception(
        self, mock_bm_data_manager, mock_logger, mock_machine_info
    ):
        mock_dataset = MagicMock(name="dataset1")
        mock_dataset.name = "dataset1"

        exporter = ResultsExporter(results_dir=".")
        exporter._machine_info = mock_machine_info
        exporter.benchmark_config = MagicMock(datasets=[mock_dataset])
        mock_bm_data_manager.return_value.load_network_sync.side_effect = Exception(
            "Load failed"
        )

        result = exporter._create_benchmark_result(
            algorithm="alg1",
            dataset="dataset1",
            backend="backend1",
            execution_time=1.23,
            memory_used=456.78,
            num_thread=4,
            commit_hash="abc123",
            date=20240101,
        )
        assert result is None
        mock_bm_data_manager.return_value.load_network_sync.assert_called_once()
        mock_logger.exception.assert_called_with(
            "Failed to load network for dataset 'dataset1'"
        )

    def test_parse_benchmark_name_valid(self, mock_logger):
        exporter = ResultsExporter(results_dir=".")
        bench_name = "GraphBenchmark.track_alg1_dataset1_backend1_001"
        result = exporter._parse_benchmark_name(bench_name)
        assert result == ("alg1", "dataset1", "backend1")
        mock_logger.debug.assert_called_with(
            f"Parsed benchmark name '{bench_name}': "
            f"algorithm=alg1, dataset=dataset1, backend=backend1"
        )

    def test_parse_benchmark_name_invalid(self, mock_logger):
        exporter = ResultsExporter(results_dir=".")
        bench_name = "InvalidBenchmarkName"
        result = exporter._parse_benchmark_name(bench_name)
        assert result is None
        mock_logger.warning.assert_called_with(
            f"Benchmark name '{bench_name}' does not match expected patterns."
        )

    @patch("nxbench.benchmarks.export.Path.open", new_callable=mock_open)
    def test_load_results_with_valid_files(
        self,
        mock_open_file,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_logger,
        mock_machine_info,
    ):
        mock_machine_dir = MagicMock(spec=Path)
        result_file = Path("/fake/results/machine1/result1.json")
        mock_machine_dir.glob.return_value = [result_file]

        with patch(
            "nxbench.benchmarks.export.Path.iterdir", return_value=[mock_machine_dir]
        ):
            exporter = ResultsExporter(results_dir="/fake/results")

        mock_dataset = MagicMock(name="dataset1")
        mock_dataset.name = "dataset1"
        mock_get_benchmark_config.return_value = MagicMock(datasets=[mock_dataset])

        json_content = {
            "commit_hash": "abc123",
            "date": 20240101,
            "params": {"env_vars": {"NUM_THREAD": "4"}},
            "results": {
                "GraphBenchmark.track_alg1_dataset1_backend1": [
                    [1.23],
                    {"datasets": ["dataset1"], "backends": ["backend1"]},
                ]
            },
        }
        mock_open_file.return_value.__enter__.return_value.read.return_value = (
            json.dumps(json_content)
        )

        mock_data_manager.load_network_sync.return_value = (
            nx.Graph(),
            {"meta": "data"},
        )

        mock_result = self.create_mock_benchmark_result(
            algorithm="alg1", dataset="dataset1", backend="backend1", num_thread=4
        )

        with patch.object(BenchmarkResult, "from_asv_result", return_value=mock_result):
            results = exporter.load_results()
            assert len(results) == 1
            mock_logger.debug.assert_any_call(f"BenchmarkResult created: {mock_result}")

    @patch("nxbench.benchmarks.export.Path.open", new_callable=mock_open)
    def test_load_results_with_malformed_json(
        self,
        mock_open_file,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_logger,
        mock_machine_info,
    ):
        mock_machine_dir = MagicMock(spec=Path)
        result_file = Path("/fake/results/machine1/result1.json")
        mock_machine_dir.glob.return_value = [result_file]

        with patch(
            "nxbench.benchmarks.export.Path.iterdir", return_value=[mock_machine_dir]
        ):
            exporter = ResultsExporter(results_dir="/fake/results")

        mock_open_file.return_value.__enter__.return_value.read.return_value = (
            "invalid json"
        )

        # Use patch.object to patch json.loads directly instead of json.load
        with patch.object(
            json, "loads", side_effect=json.JSONDecodeError("Expecting value", "", 0)
        ):
            results = exporter.load_results()
            assert len(results) == 0
            mock_logger.exception.assert_called_once_with(
                f"Failed to decode JSON from {result_file}"
            )

    @patch("nxbench.benchmarks.export.Path.open", new_callable=mock_open)
    def test_load_results_with_missing_env_vars(
        self,
        mock_open_file,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_logger,
        mock_machine_info,
    ):
        mock_machine_dir = MagicMock(spec=Path)
        result_file = Path("/fake/results/machine1/result1.json")
        mock_machine_dir.glob.return_value = [result_file]

        with patch(
            "nxbench.benchmarks.export.Path.iterdir", return_value=[mock_machine_dir]
        ):
            exporter = ResultsExporter(results_dir="/fake/results")

        mock_dataset = MagicMock(name="dataset1")
        mock_dataset.name = "dataset1"
        mock_get_benchmark_config.return_value = MagicMock(datasets=[mock_dataset])

        mock_open_file.return_value.__enter__.return_value.read.return_value = (
            json.dumps(
                {
                    "commit_hash": "abc123",
                    "date": 20240101,
                    "env_name": "NUM_THREAD4",
                    "results": {
                        "GraphBenchmark.track_alg1_dataset1_backend1": [
                            [1.23],
                            {"datasets": ["dataset1"], "backends": ["backend1"]},
                        ]
                    },
                }
            )
        )

        mock_data_manager.load_network_sync.return_value = (
            nx.Graph(),
            {"meta": "data"},
        )

        mock_result = self.create_mock_benchmark_result(
            algorithm="alg1", dataset="dataset1", backend="backend1", num_thread=4
        )

        with patch.object(BenchmarkResult, "from_asv_result", return_value=mock_result):
            results = exporter.load_results()
            assert len(results) == 1
            mock_logger.debug.assert_any_call(f"BenchmarkResult created: {mock_result}")

    @patch("nxbench.benchmarks.export.Path.open", new_callable=mock_open)
    def test_load_results_with_invalid_num_thread(
        self,
        mock_open_file,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_logger,
        mock_machine_info,
    ):
        mock_machine_dir = MagicMock(spec=Path)
        result_file = Path("/fake/results/machine1/result1.json")
        mock_machine_dir.glob.return_value = [result_file]

        with patch(
            "nxbench.benchmarks.export.Path.iterdir", return_value=[mock_machine_dir]
        ):
            exporter = ResultsExporter(results_dir="/fake/results")

        mock_dataset = MagicMock(name="dataset1")
        mock_dataset.name = "dataset1"
        mock_get_benchmark_config.return_value = MagicMock(datasets=[mock_dataset])

        mock_open_file.return_value.__enter__.return_value.read.return_value = (
            json.dumps(
                {
                    "commit_hash": "abc123",
                    "date": 20240101,
                    "params": {"env_vars": {"NUM_THREAD": "invalid"}},
                    "results": {
                        "GraphBenchmark.track_alg1_dataset1_backend1": [
                            [1.23],
                            {"datasets": ["dataset1"], "backends": ["backend1"]},
                        ]
                    },
                }
            )
        )

        mock_logger.reset_mock()

        mock_result = self.create_mock_benchmark_result(
            algorithm="alg1",
            dataset="dataset1",
            backend="backend1",
            num_thread=1,
        )

        with patch.object(BenchmarkResult, "from_asv_result", return_value=mock_result):
            results = exporter.load_results()
            assert len(results) == 1
            # Assert against actual message format from code
            mock_logger.warning.assert_any_call(
                f"Invalid NUM_THREAD value in {result_file}: invalid"
            )

    def test_to_dataframe_no_results(
        self,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_machine_info,
    ):
        with patch.object(ResultsExporter, "load_results", return_value=[]):
            exporter = ResultsExporter(results_dir=".")
            with pytest.raises(ValueError, match="No benchmark results found."):
                exporter.to_dataframe()
            mock_logger.error.assert_called_with("No benchmark results found.")

    @patch("nxbench.benchmarks.export.pd.DataFrame.to_csv")
    def test_to_csv(
        self,
        mock_to_csv,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_machine_info,
    ):
        df_mock = MagicMock(spec=pd.DataFrame)
        with patch.object(ResultsExporter, "to_dataframe", return_value=df_mock):
            exporter = ResultsExporter(results_dir=".")
            exporter.to_csv("output.csv")
            df_mock.to_csv.assert_called_once_with("output.csv", index=False)
            mock_logger.info.assert_called_with("Results exported to CSV: output.csv")

    @patch("nxbench.benchmarks.export.BenchmarkDB")
    def test_to_sql_replace(
        self,
        mock_benchmark_db_class,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_machine_info,
        mock_get_python_version,
    ):
        mock_db_instance = mock_benchmark_db_class.return_value
        mock_db_instance.db_path = "test.db"

        with (
            patch.object(ResultsExporter, "load_results", return_value=[]),
            patch.object(ResultsExporter, "get_machine_info", return_value={}),
        ):
            exporter = ResultsExporter(results_dir=".")
            exporter.to_sql(db_path="test.db", if_exists="replace")
            mock_benchmark_db_class.assert_called_with("test.db")
            mock_db_instance.delete_results.assert_called_once()
            mock_db_instance.save_results.assert_called_once_with(
                results=[],
                machine_info={},
                python_version="3.8.10",
                package_versions=None,
            )
            mock_logger.info.assert_called_with(
                "Results exported to SQL database: test.db"
            )

    @patch("nxbench.benchmarks.export.BenchmarkDB")
    def test_to_sql_append(
        self,
        mock_benchmark_db_class,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_machine_info,
        mock_get_python_version,
    ):
        # Setup the mock BenchmarkDB
        mock_db_instance = mock_benchmark_db_class.return_value
        mock_db_instance.db_path = "test.db"

        with (
            patch.object(ResultsExporter, "load_results", return_value=[]),
            patch.object(ResultsExporter, "get_machine_info", return_value={}),
        ):
            exporter = ResultsExporter(results_dir=".")
            exporter.to_sql(db_path="test.db", if_exists="append")
            mock_benchmark_db_class.assert_called_with("test.db")
            mock_db_instance.delete_results.assert_not_called()
            mock_db_instance.save_results.assert_called_once_with(
                results=[],
                machine_info={},
                python_version="3.8.10",
                package_versions=None,
            )
            mock_logger.info.assert_called_with(
                "Results exported to SQL database: test.db"
            )

    def test_get_machine_info_with_info(
        self,
        mock_machine_info,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
    ):
        exporter = ResultsExporter(results_dir=".")
        exporter._machine_info = mock_machine_info
        machine_info = exporter.get_machine_info()
        assert machine_info == {
            "arch": mock_machine_info.arch,
            "cpu": mock_machine_info.cpu,
            "machine": mock_machine_info.machine,
            "num_cpu": mock_machine_info.num_cpu,
            "os": mock_machine_info.os,
            "ram": mock_machine_info.ram,
            "version": mock_machine_info.version,
        }

    def test_get_machine_info_no_info(
        self,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_machine_info,
    ):
        exporter = ResultsExporter(results_dir=".")
        exporter._machine_info = None
        machine_info = exporter.get_machine_info()
        assert machine_info == {}

    @patch("nxbench.benchmarks.export.BenchmarkDB")
    def test_query_sql_as_pandas(
        self,
        mock_benchmark_db_class,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_machine_info,
    ):
        mock_db_instance = mock_benchmark_db_class.return_value
        mock_db_instance.get_results.return_value = pd.DataFrame()

        exporter = ResultsExporter(results_dir=".")
        result = exporter.query_sql(as_pandas=True)
        assert isinstance(result, pd.DataFrame)
        mock_db_instance.get_results.assert_called_once_with(
            algorithm=None,
            backend=None,
            dataset=None,
            start_date=None,
            end_date=None,
            as_pandas=True,
        )

    @patch("nxbench.benchmarks.export.BenchmarkDB")
    def test_query_sql_as_dict(
        self,
        mock_benchmark_db_class,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_machine_info,
    ):
        mock_db_instance = mock_benchmark_db_class.return_value
        mock_db_instance.get_results.return_value = [{"key": "value"}]

        exporter = ResultsExporter(results_dir=".")
        result = exporter.query_sql(as_pandas=False)
        assert isinstance(result, list)
        assert result == [{"key": "value"}]
        mock_db_instance.get_results.assert_called_once_with(
            algorithm=None,
            backend=None,
            dataset=None,
            start_date=None,
            end_date=None,
            as_pandas=False,
        )

    @patch("nxbench.benchmarks.export.BenchmarkDB")
    def test_to_sql_initialization(
        self,
        mock_benchmark_db_class,
        mock_logger,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_machine_info,
        mock_get_python_version,
    ):
        exporter = ResultsExporter(results_dir=".")
        assert exporter._db is None
        exporter.to_sql()
        mock_benchmark_db_class.assert_called_with(None)
        assert exporter._db == mock_benchmark_db_class.return_value

    @patch("nxbench.benchmarks.export.Path.open", new_callable=mock_open)
    def test_load_results_caching(
        self,
        mock_open_file,
        mock_get_benchmark_config,
        mock_data_manager,
        mock_logger,
        mock_machine_info,
    ):
        mock_machine_dir = MagicMock(spec=Path)
        result_file = Path("/fake/results/machine1/result1.json")
        mock_machine_dir.glob.return_value = [result_file]

        with patch(
            "nxbench.benchmarks.export.Path.iterdir", return_value=[mock_machine_dir]
        ):
            exporter = ResultsExporter(results_dir="/fake/results")

        mock_dataset = MagicMock(name="dataset1")
        mock_dataset.name = "dataset1"
        mock_get_benchmark_config.return_value = MagicMock(datasets=[mock_dataset])

        json_content = {
            "commit_hash": "abc123",
            "date": 20240101,
            "params": {"env_vars": {"NUM_THREAD": "4"}},
            "results": {
                "GraphBenchmark.track_alg1_dataset1_backend1": [
                    [1.23],
                    {"datasets": ["dataset1"], "backends": ["backend1"]},
                ]
            },
        }
        mock_open_file.return_value.__enter__.return_value.read.return_value = (
            json.dumps(json_content)
        )

        mock_data_manager.load_network_sync.return_value = (
            nx.Graph(),
            {"meta": "data"},
        )

        mock_result = self.create_mock_benchmark_result(
            algorithm="alg1", dataset="dataset1", backend="backend1", num_thread=4
        )

        with patch.object(BenchmarkResult, "from_asv_result", return_value=mock_result):
            results1 = exporter.load_results()
            results2 = exporter.load_results()
            assert results1 == results2
            assert len(results1) == 1
            mock_open_file.assert_called_once()
            mock_logger.debug.assert_any_call(f"BenchmarkResult created: {mock_result}")

    def test_parse_benchmark_name_standard_format(self, mock_logger):
        """Test parsing of standard benchmark names."""
        exporter = ResultsExporter(results_dir=".")
        test_cases = [
            (
                "GraphBenchmark.track_pagerank_karate_networkx",
                ("pagerank", "karate", "networkx"),
            ),
            (
                "GraphBenchmark.track_betweenness_jazz_parallel_4",
                ("betweenness", "jazz", "parallel"),
            ),
            (
                "benchmark.GraphBenchmark.track_clustering_08blocks_graphblas",
                ("clustering", "08blocks", "graphblas"),
            ),
        ]

        for bench_name, expected in test_cases:
            result = exporter._parse_benchmark_name(bench_name)
            assert result == expected, f"Failed to parse standard name: {bench_name}"
            mock_logger.debug.assert_any_call(
                f"Parsed benchmark name '{bench_name}': "
                f"algorithm={expected[0]}, dataset={expected[1]}, backend={expected[2]}"
            )
            mock_logger.reset_mock()

    def test_parse_benchmark_name_synthetic_format(self, mock_logger):
        """Test parsing of synthetic dataset benchmark names."""
        exporter = ResultsExporter(results_dir=".")
        test_cases = [
            (
                "GraphBenchmark.track_transitivity_barabasi_albert_small_networkx",
                ("transitivity", "barabasi_albert_small", "networkx"),
            ),
            (
                "GraphBenchmark.track_pagerank_erdos_renyi_medium_parallel_4",
                ("pagerank", "erdos_renyi_medium", "parallel"),
            ),
            (
                "GraphBenchmark.track_clustering_watts_strogatz_large_graphblas",
                ("clustering", "watts_strogatz_large", "graphblas"),
            ),
        ]

        for bench_name, expected in test_cases:
            result = exporter._parse_benchmark_name(bench_name)
            assert result == expected, f"Failed to parse synthetic name: {bench_name}"
            mock_logger.debug.assert_any_call(
                f"Parsed benchmark name '{bench_name}': "
                f"algorithm={expected[0]}, dataset={expected[1]}, backend={expected[2]}"
            )
            mock_logger.reset_mock()

    def test_parse_benchmark_name_edge_cases(self, mock_logger):
        """Test handling of edge cases in benchmark names."""
        exporter = ResultsExporter(results_dir=".")
        test_cases = [
            # Multiple underscores in synthetic dataset name
            (
                "GraphBenchmark.track_pagerank_power_law_cluster_graph_networkx",
                ("pagerank", "power_law_cluster_graph", "networkx"),
            ),
            # Numeric suffixes in backend
            (
                "GraphBenchmark.track_betweenness_karate_parallel_8",
                ("betweenness", "karate", "parallel"),
            ),
            # Mixed case handling
            (
                "GraphBenchmark.track_Pagerank_Karate_NetworkX",
                ("Pagerank", "Karate", "NetworkX"),
            ),
        ]

        for bench_name, expected in test_cases:
            result = exporter._parse_benchmark_name(bench_name)
            assert result == expected, f"Failed to parse edge case: {bench_name}"
            mock_logger.reset_mock()
