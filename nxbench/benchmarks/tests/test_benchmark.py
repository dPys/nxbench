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
            "backend": ["networkx", "parallel"],
            "num_threads": ["1", "4", "8"],
        }

        mock_config = MagicMock()
        mock_config.datasets = datasets
        mock_config.algorithms = algorithms
        mock_config.matrix = mock_matrix

        mock_get_config.return_value = mock_config

        MockDataManager.return_value.load_network_sync.return_value = (
            nx.Graph(),
            {"metadata": "test"},
        )

        import nxbench.benchmarks.benchmark

        importlib.reload(nxbench.benchmarks.benchmark)

        @nxbench.benchmarks.benchmark.generate_benchmark_methods
        class MockGraphBenchmark(nxbench.benchmarks.benchmark.GraphBenchmark):
            config = mock_config

        benchmark_instance = MockGraphBenchmark()
        benchmark_instance.setup()

        return benchmark_instance


@pytest.fixture(scope="module")
def mock_backends():
    with patch("nxbench.benchmarks.utils.get_available_backends") as mock:
        mock.return_value = ["networkx", "parallel"]
        yield mock


def test_backend_selection(mock_backends, mock_benchmark):
    """Test that available backends are correctly identified."""
    config = mock_benchmark.config
    available_backends = ["networkx", "parallel"]
    assert all(backend in available_backends for backend in config.matrix["backend"])
    assert "cugraph" not in config.matrix["backend"]


def test_graph_benchmark_initialization(mock_benchmark):
    """Test that the GraphBenchmark class initializes properly."""
    assert mock_benchmark.data_manager is not None
    assert mock_benchmark.graphs == {}


def test_setup_cache(mock_benchmark):
    """Test that setup_cache populates graphs from the configuration."""
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
    """Test setup_cache for failure when a dataset cannot be loaded."""
    mock_benchmark.graphs = {}
    mock_benchmark.data_manager.load_network_sync.side_effect = Exception(
        "Failed to load dataset"
    )

    mock_benchmark.setup_cache()

    assert len(mock_benchmark.graphs) == 0


def test_prepare_benchmark_unsupported_backend(mock_benchmark):
    """Test the prepare_benchmark method with an unsupported backend."""
    result = mock_benchmark.prepare_benchmark("test_dataset1", "unsupported_backend")
    assert result is None


def test_prepare_benchmark_missing_dataset(mock_benchmark):
    """Test prepare_benchmark when the dataset is not found in the cache."""
    result = mock_benchmark.prepare_benchmark("non_existent_dataset", "networkx")
    assert result is None


def test_do_benchmark_setup_failure(mock_benchmark):
    """Test the do_benchmark method when setup fails (dataset not found)."""
    mock_algo_config = mock_benchmark.config.algorithms[0]

    metrics = mock_benchmark.do_benchmark(
        mock_algo_config, "non_existent_dataset", "networkx", 1
    )
    assert math.isnan(metrics["execution_time"])
    assert math.isnan(metrics["memory_used"])


def test_do_benchmark_func_ref_none(mock_benchmark):
    """Test do_benchmark when algo_config.func_ref is None, causing ImportError."""
    mock_algo_config = AlgorithmConfig(name="dummy_algo", func="dummy.module.func")
    mock_algo_config.func_ref = None  # Simulate func_ref being None

    metrics = mock_benchmark.do_benchmark(
        mock_algo_config, "test_dataset1", "networkx", 1
    )
    assert math.isnan(metrics["execution_time"])
    assert math.isnan(metrics["memory_used"])


def test_do_benchmark_algo_execution_exception(mock_benchmark):
    """Test do_benchmark when exception occurs during algorithm execution."""
    mock_algo_config = mock_benchmark.config.algorithms[0]
    mock_algo_config.func_ref.side_effect = Exception("Algorithm failed")

    # Prepare a valid graph
    mock_benchmark.prepare_benchmark = MagicMock(return_value=nx.Graph())

    metrics = mock_benchmark.do_benchmark(
        mock_algo_config, "test_dataset1", "networkx", 1
    )

    assert math.isnan(metrics["execution_time"])
    assert math.isnan(metrics["memory_used"])


def test_do_benchmark_validation_failure(mock_benchmark):
    """Test do_benchmark when validation fails."""
    mock_algo_config = mock_benchmark.config.algorithms[0]
    mock_algo_config.func_ref = MagicMock(return_value={"result": "some_value"})
    mock_algo_config.func_ref.__name__ = "dummy_algo_func"

    # Prepare a valid graph and update the graphs dictionary
    test_graph = nx.Graph()
    mock_benchmark.prepare_benchmark = MagicMock(return_value=test_graph)
    mock_benchmark.graphs["test_dataset1"] = (test_graph, {"metadata": "test"})

    # Mock the BenchmarkValidator to raise an exception during validation
    with patch("nxbench.benchmarks.benchmark.BenchmarkValidator") as MockValidator:
        mock_validator = MockValidator.return_value
        mock_validator.validate_result.side_effect = Exception("Validation failed")

        metrics = mock_benchmark.do_benchmark(
            mock_algo_config, "test_dataset1", "networkx", 1
        )

        # Even if validation fails, metrics should be returned
        assert "execution_time" in metrics
        assert "memory_used" in metrics
        assert not math.isnan(metrics["execution_time"])
        assert not math.isnan(metrics["memory_used"])


def test_get_algorithm_function_networkx(mock_benchmark):
    """Test get_algorithm_function for the networkx backend."""
    from nxbench.benchmarks.benchmark import get_algorithm_function
    from nxbench.benchmarks.config import AlgorithmConfig

    algo_config = AlgorithmConfig(name="dummy_algo", func="dummy.module.func")
    algo_config.func_ref = MagicMock(name="dummy_func_ref")

    func = get_algorithm_function(algo_config, "networkx")
    assert func == algo_config.func_ref


def test_get_algorithm_function_func_ref_none(mock_benchmark):
    """Test get_algorithm_function when func_ref is None."""
    from nxbench.benchmarks.benchmark import get_algorithm_function
    from nxbench.benchmarks.config import AlgorithmConfig

    algo_config = AlgorithmConfig(name="dummy_algo", func="dummy.module.func")
    algo_config.func_ref = None

    with pytest.raises(ImportError):
        get_algorithm_function(algo_config, "networkx")


def test_get_algorithm_function_other_backend(mock_benchmark):
    """Test get_algorithm_function for non-networkx backends."""
    from functools import partial

    from nxbench.benchmarks.benchmark import get_algorithm_function
    from nxbench.benchmarks.config import AlgorithmConfig

    algo_config = AlgorithmConfig(name="dummy_algo", func="dummy.module.func")
    algo_config.func_ref = MagicMock(name="dummy_func_ref")

    func = get_algorithm_function(algo_config, "parallel")
    assert isinstance(func, partial)
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
    import math

    from nxbench.benchmarks.benchmark import process_algorithm_params

    params = {"_pos_arg": 42, "func_ref": {"func": "math.sqrt"}}
    pos_args, kwargs = process_algorithm_params(params)
    assert pos_args == [42]
    assert "func_ref" in kwargs
    assert callable(kwargs["func_ref"])
    assert kwargs["func_ref"] == math.sqrt


def test_process_algorithm_params_func_import_error(mock_benchmark):
    """Test processing algorithm parameters when importing a function fails."""
    from nxbench.benchmarks.benchmark import process_algorithm_params

    params = {"_pos_arg": 42, "func_ref": {"func": "nonexistent.module.func"}}
    with pytest.raises(ImportError):
        process_algorithm_params(params)


def test_process_algorithm_params_func_attribute_error(mock_benchmark):
    """Test processing algorithm parameters when the function does not exist."""
    from nxbench.benchmarks.benchmark import process_algorithm_params

    params = {"_pos_arg": 42, "func_ref": {"func": "math.nonexistent_func"}}
    with pytest.raises(AttributeError):
        process_algorithm_params(params)


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
    available = backend_name in mock_benchmark.config.matrix["backend"]
    assert available == expected


def test_generated_benchmark_methods_exist(mock_benchmark):
    """Test that the generated benchmark methods exist on the GraphBenchmark
    instance.
    """
    methods = [attr for attr in dir(mock_benchmark) if attr.startswith("track_")]
    # Get the expected methods
    expected_methods = set()
    config = mock_benchmark.config
    algorithms = config.algorithms
    datasets = [ds.name for ds in config.datasets]
    backends = config.matrix["backend"]
    num_threads = [int(n) for n in config.matrix["num_threads"]]

    for algo in algorithms:
        for dataset in datasets:
            for backend in backends:
                for num_thread in num_threads:
                    method_name = f"track_{algo.name}_{dataset}_{backend}_{num_thread}"
                    expected_methods.add(method_name)

    assert set(methods) >= expected_methods
