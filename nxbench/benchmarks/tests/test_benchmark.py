import importlib
import math
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from nxbench.benchmarks.config import AlgorithmConfig, DatasetConfig


@pytest.fixture(scope="module")
def mock_benchmark():
    """Fixture to create a mocked GraphBenchmark instance."""
    with (
        patch("nxbench.benchmarks.benchmark.get_benchmark_config") as mock_get_config,
        patch("nxbench.benchmarks.benchmark.BenchmarkDataManager") as MockDataManager,
    ):
        mock_dataset1 = DatasetConfig(name="test_dataset1", source="mock_source")
        mock_dataset2 = DatasetConfig(name="test_dataset2", source="mock_source")
        datasets = [mock_dataset1, mock_dataset2]

        mock_algo = AlgorithmConfig(name="dummy_algo", func="dummy.module.func")
        mock_algo.func_ref = MagicMock()
        mock_algo.params = {}
        algorithms = [mock_algo]

        mock_matrix = {
            "req": {
                "networkx": ["3.3"],
                "nx_parallel": ["0.2"],
                "python-graphblas": ["2024.2.0"],
            },
            "env": {
                "NUM_THREAD": ["1", "4", "8"],
                "OMP_NUM_THREADS": ["1"],
                "MKL_NUM_THREADS": ["1"],
                "OPENBLAS_NUM_THREADS": ["1"],
            },
        }

        mock_config = MagicMock()
        mock_config.datasets = datasets
        mock_config.algorithms = algorithms
        mock_config.matrix = mock_matrix

        MockDataManager.return_value.load_network_sync.return_value = (
            nx.Graph(),
            {"metadata": "test"},
        )

        import nxbench.benchmarks.benchmark

        importlib.reload(nxbench.benchmarks.benchmark)
        from nxbench.benchmarks.benchmark import GraphBenchmark

        GraphBenchmark.params = [
            [ds.name for ds in datasets],
            ["networkx", "parallel"],
            [1, 4, 8],
        ]

        benchmark_instance = GraphBenchmark()
        benchmark_instance.config = mock_config

        return benchmark_instance


@pytest.fixture(scope="module")
def mock_backends():
    with patch("nxbench.benchmarks.utils.get_available_backends") as mock:
        mock.return_value = ["networkx", "parallel"]
        yield mock


def test_backend_selection(mock_backends, mock_benchmark):
    assert "networkx" in mock_benchmark.params[1]
    assert "parallel" in mock_benchmark.params[1]
    assert "cugraph" not in mock_benchmark.params[1]


def test_graph_benchmark_initialization(mock_benchmark):
    """Test that the GraphBenchmark class initializes properly."""
    assert mock_benchmark.data_manager is not None
    assert mock_benchmark.graphs == {}


def test_setup_cache(mock_benchmark):
    """Test that setup_cache populates graphs from the configuration."""
    with patch("nxbench.benchmarks.benchmark.get_benchmark_config") as mock_get_config:
        mock_get_config.return_value = mock_benchmark.config

        mock_benchmark.data_manager.load_network_sync = MagicMock(
            return_value=(nx.Graph(), {"metadata": "test"})
        )

        mock_benchmark.setup_cache()

        assert len(mock_benchmark.graphs) == 2
        for dataset in ["test_dataset1", "test_dataset2"]:
            assert dataset in mock_benchmark.graphs
            graph, metadata = mock_benchmark.graphs[dataset]
            assert isinstance(graph, nx.Graph)
            assert metadata == {"metadata": "test"}


def test_setup_failure(mock_benchmark):
    """Test the setup method for failure when the dataset is not found."""
    mock_benchmark.graphs = {}

    result = mock_benchmark.setup("non_existent_dataset", "networkx")
    assert result is None


def test_setup_unsupported_backend(mock_benchmark):
    """Test the setup method with an unsupported backend."""
    dataset_name = "test_dataset1"
    result = mock_benchmark.setup(dataset_name, "unsupported_backend")
    assert result is None


def test_do_benchmark_setup_failure(mock_benchmark):
    """Test the do_benchmark method when setup fails."""
    mock_algo_config = mock_benchmark.config.algorithms[0]

    metrics = mock_benchmark.do_benchmark(
        mock_algo_config, "non_existent_dataset", "networkx", 1
    )
    assert math.isnan(metrics["execution_time"])
    assert math.isnan(metrics["memory_used"])


def test_get_algorithm_function_networkx(mock_benchmark):
    """Test get_algorithm_function for the networkx backend."""
    from nxbench.benchmarks.benchmark import get_algorithm_function
    from nxbench.benchmarks.config import AlgorithmConfig

    algo_config = AlgorithmConfig(name="dummy_algo", func="dummy.module.func")
    algo_config.func_ref = MagicMock(name="dummy_func_ref")

    func = get_algorithm_function(algo_config, "networkx")
    assert func == algo_config.func_ref


def test_get_algorithm_function_other_backend(mock_benchmark):
    """Test get_algorithm_function for non-networkx backends."""
    from nxbench.benchmarks.benchmark import get_algorithm_function
    from nxbench.benchmarks.config import AlgorithmConfig

    algo_config = AlgorithmConfig(name="dummy_algo", func="dummy.module.func")
    algo_config.func_ref = MagicMock(name="dummy_func_ref")

    func = get_algorithm_function(algo_config, "parallel")
    assert func.func == algo_config.func_ref
    assert func.keywords["backend"] == "parallel"


def test_process_algorithm_params(mock_benchmark):
    """Test processing algorithm parameters."""
    from nxbench.benchmarks.benchmark import process_algorithm_params

    params = {"_pos_arg": 42, "kwarg": "value"}
    pos_args, kwargs = process_algorithm_params(params)
    assert pos_args == [42]
    assert kwargs == {"kwarg": "value"}


def test_process_algorithm_params_with_function(mock_benchmark):
    """Test processing algorithm parameters with a function reference."""
    from nxbench.benchmarks.benchmark import process_algorithm_params

    params = {"_pos_arg": 42, "func_ref": {"func": "math.sqrt"}}
    pos_args, kwargs = process_algorithm_params(params)
    import math

    assert pos_args == [42]
    assert "func_ref" in kwargs
    assert callable(kwargs["func_ref"])
    assert kwargs["func_ref"] == math.sqrt


@pytest.mark.parametrize(
    ("backend_name", "expected"),
    [
        ("networkx", True),
        ("parallel", True),
        ("cugraph", False),
        ("graphblas", False),
    ],
)
def test_backend_availability(mock_benchmark, backend_name, expected):
    """Test the availability of different backends."""
    available = backend_name in mock_benchmark.params[1]
    assert available == expected
