import sys  # isort:skip
import os  # isort:skip
from unittest.mock import MagicMock, patch, AsyncMock  # isort:skip
from nxbench.benchmarks.config import (  # isort:skip
    AlgorithmConfig,
    DatasetConfig,
)

sys.modules["prefect_dask"] = MagicMock()  # isort:skip
sys.modules["prefect_dask.task_runners"] = MagicMock()  # isort:skip

import json  # isort:skip  # noqa: E402
import pytest  # isort:skip  # noqa: E402
import networkx as nx  # isort:skip  # noqa: E402

from nxbench.validation.registry import BenchmarkValidator  # isort:skip  # noqa: E402
from nxbench.data.loader import BenchmarkDataManager  # isort:skip  # noqa: E402

from nxbench.benchmarks.benchmark import (  # isort:skip  # noqa: E402
    load_config,
    setup_cache,
    configure_backend,
    run_algorithm,
    validate_results,
    collect_metrics,
    teardown_specific,
    run_single_benchmark,
    benchmark_suite,
    main_benchmark,
)


@pytest.fixture(autouse=True)
def patch_run_logger():  # noqa: PT004
    with patch("nxbench.benchmarks.benchmark.get_run_logger", return_value=MagicMock()):
        yield


@pytest.fixture
def mock_benchmark_config():
    """Fixture that returns a mock benchmark config with the required 'func'."""
    return {
        "algorithms": [
            AlgorithmConfig(
                name="alg1",
                func="networkx.algorithms.link_analysis.pagerank_alg.pagerank",
                params={"p1": 10},
            )
        ],
        "datasets": [DatasetConfig(name="ds1", source="networkrepository")],
        "env_data": {
            "pythons": ["3.10"],
            "backend": {"networkx": ["networkx==3.4.1"]},
            "num_threads": [1],
        },
    }


@pytest.fixture
def mock_benchmark_data_manager():
    """Fixture that returns a MagicMock for the BenchmarkDataManager."""
    data_manager = MagicMock(spec=BenchmarkDataManager)
    return data_manager


@pytest.fixture
def example_graph():
    """Create a small example NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([(1, 2), (2, 3)])
    return G


@pytest.fixture
def mock_algorithm_config():
    """Return a simple AlgorithmConfig with a valid 'func'."""
    return AlgorithmConfig(
        name="alg1",
        func="networkx.algorithms.link_analysis.pagerank_alg.pagerank",
        params={"p1": 10},
    )


@pytest.fixture
def mock_dataset_config():
    """Return a simple DatasetConfig."""
    return DatasetConfig(name="ds1", source="networkrepository")


@pytest.fixture
def patch_machine_info():  # noqa: PT004
    """Patch get_machine_info to return static data."""
    with patch(
        "nxbench.benchmarks.benchmark.get_machine_info",
        return_value={"machine": "test_machine", "cpu": "test_cpu"},
    ):
        yield


@pytest.fixture
def patch_python_version():  # noqa: PT004
    """Patch get_python_version to return '3.10.12' by default."""
    with patch(
        "nxbench.benchmarks.benchmark.get_python_version",
        return_value="3.10.12",
    ):
        yield


###############################################################################
#                               TEST: load_config                             #
###############################################################################


@pytest.mark.asyncio
@patch("nxbench.benchmarks.benchmark.get_benchmark_config")
async def test_load_config_success(mock_get_config, mock_benchmark_config):
    mock_get_config.return_value = MagicMock(
        algorithms=mock_benchmark_config["algorithms"],
        datasets=mock_benchmark_config["datasets"],
        env_data=mock_benchmark_config["env_data"],
    )
    result = load_config()
    assert "algorithms" in result
    assert "datasets" in result
    assert "env_data" in result
    assert len(result["algorithms"]) == 1
    assert len(result["datasets"]) == 1


###############################################################################
#                              TEST: setup_cache                              #
###############################################################################


@pytest.mark.asyncio
async def test_setup_cache_success(mock_benchmark_data_manager, example_graph):
    mock_benchmark_data_manager.load_network_sync.return_value = (
        example_graph,
        {"meta": "data"},
    )

    with patch(
        "nxbench.benchmarks.benchmark.BenchmarkDataManager",
        return_value=mock_benchmark_data_manager,
    ):
        ds_config = DatasetConfig(name="ds1", source="networkrepository")
        result = setup_cache([ds_config])
        assert "ds1" in result
        graph, metadata = result["ds1"]
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2
        assert metadata == {"meta": "data"}


@pytest.mark.asyncio
async def test_setup_cache_failure(mock_benchmark_data_manager, caplog):
    mock_benchmark_data_manager.load_network_sync.side_effect = ValueError("Load fail")

    with patch(
        "nxbench.benchmarks.benchmark.BenchmarkDataManager",
        return_value=mock_benchmark_data_manager,
    ):
        ds_config = DatasetConfig(name="ds1", source="networkrepository")
        result = setup_cache([ds_config])
        assert "ds1" not in result
        assert any(
            "Failed to load dataset 'ds1'" in rec.message for rec in caplog.records
        )


###############################################################################
#                           TEST: configure_backend                           #
###############################################################################


@pytest.mark.parametrize(
    "backend",
    [
        "networkx",
        "parallel",
        "cugraph",
        "graphblas",
    ],
)
def test_configure_backend_success(backend, example_graph):
    if backend == "networkx":
        result = configure_backend.fn(example_graph, backend, 4)
        assert result is example_graph
    elif backend in ("parallel", "cugraph"):
        mock_module = MagicMock()
        if backend == "parallel":
            mock_module.ParallelGraph.return_value = "parallel_graph"
        else:  # cugraph
            mock_module.from_networkx.return_value = "cugraph_graph"

        with patch(
            "nxbench.benchmarks.benchmark.import_module", return_value=mock_module
        ):
            if backend == "parallel":
                result_p = configure_backend.fn(example_graph, backend, 2)
                assert result_p == "parallel_graph"
            else:  # cugraph
                result_cu = configure_backend.fn(example_graph, backend, 2)
                assert result_cu == "cugraph_graph"
    else:
        # "graphblas"
        mock_module = MagicMock()
        mock_ga = MagicMock()
        mock_ga.Graph.from_networkx.return_value = "graphblas_graph"
        with patch(
            "nxbench.benchmarks.benchmark.import_module",
            side_effect=[mock_module, mock_ga],
        ):
            result_gb = configure_backend.fn(example_graph, backend, 2)
            assert result_gb == "graphblas_graph"


def test_configure_backend_unsupported(example_graph):
    with pytest.raises(ValueError, match="Unsupported backend: nonexistent"):
        configure_backend.fn(example_graph, "nonexistent", 4)


###############################################################################
#                             TEST: run_algorithm                             #
###############################################################################


@patch("nxbench.benchmarks.benchmark.memory_tracker", autospec=True)
def test_run_algorithm_success(
    mock_memory_tracker, mock_algorithm_config, example_graph
):
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = {"peak": 5000000}
    mock_cm.__exit__.return_value = None
    mock_memory_tracker.return_value = mock_cm

    mock_func = MagicMock()
    mock_func.__name__ = "mock_algo_func"
    mock_func.return_value = "algo_result"

    with patch.object(mock_algorithm_config, "get_callable", return_value=mock_func):
        result, exec_time, peak_mem, error = run_algorithm.fn(
            graph=example_graph,
            algo_config=mock_algorithm_config,
            num_thread=2,
            backend="networkx",
        )
        assert result == "algo_result"
        assert exec_time >= 0
        assert peak_mem == 5000000
        assert error is None


@patch("nxbench.benchmarks.benchmark.memory_tracker", autospec=True)
def test_run_algorithm_importerror(
    mock_memory_tracker, mock_algorithm_config, example_graph
):
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = {"peak": 5000000}
    mock_cm.__exit__.return_value = None
    mock_memory_tracker.return_value = mock_cm

    with patch.object(
        mock_algorithm_config, "get_callable", side_effect=ImportError("No module")
    ):
        res, exec_time, peak_mem, error = run_algorithm.fn(
            graph=example_graph,
            algo_config=mock_algorithm_config,
            num_thread=2,
            backend="networkx",
        )
        assert res is None
        assert exec_time == 0.0
        assert peak_mem == 0
        assert "No module" in error


@patch("nxbench.benchmarks.benchmark.memory_tracker", autospec=True)
def test_run_algorithm_exception(
    mock_memory_tracker, mock_algorithm_config, example_graph
):
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = {"peak": 5000000}
    mock_cm.__exit__.return_value = None
    mock_memory_tracker.return_value = mock_cm

    mock_func = MagicMock()
    mock_func.__name__ = "mock_algo_func"
    mock_func.side_effect = ValueError("Algorithm error")

    with patch.object(mock_algorithm_config, "get_callable", return_value=mock_func):
        res, exec_time, peak_mem, error = run_algorithm.fn(
            graph=example_graph,
            algo_config=mock_algorithm_config,
            num_thread=2,
            backend="networkx",
        )
        assert res is None
        assert exec_time > 0
        assert peak_mem == 5000000
        assert "Algorithm error" in error


###############################################################################
#                           TEST: validate_results                            #
###############################################################################


def test_validate_results_success(mock_algorithm_config, example_graph):
    with patch.object(
        BenchmarkValidator, "validate_result", return_value=None
    ) as mock_validator:
        status, msg = validate_results.fn(
            "some_result", mock_algorithm_config, example_graph
        )
        mock_validator.assert_called_once_with("some_result", "alg1", example_graph)
        assert status == "passed"
        assert msg == ""


def test_validate_results_exception(mock_algorithm_config, example_graph):
    with patch.object(
        BenchmarkValidator, "validate_result", side_effect=ValueError("validation fail")
    ):
        status, msg = validate_results.fn(
            "some_result", mock_algorithm_config, example_graph
        )
        assert status == "warning"
        assert "validation fail" in msg


###############################################################################
#                            TEST: collect_metrics                            #
###############################################################################


@pytest.mark.usefixtures("patch_machine_info")
def test_collect_metrics_no_error(example_graph, mock_algorithm_config):
    res = collect_metrics.fn(
        execution_time=1.23,
        execution_time_with_preloading=1.55,
        peak_memory=5000000,
        graph=example_graph,
        algo_config=mock_algorithm_config,
        backend="networkx",
        dataset_name="ds1",
        num_thread=4,
        validation_status="passed",
        validation_message="",
        error=None,
    )
    assert res["execution_time"] == 1.23
    assert res["execution_time_with_preloading"] == 1.55
    assert res["memory_used"] == 5000000 / (1024 * 1024)
    assert res["algorithm"] == "alg1"
    assert res["backend"] == "networkx"
    assert res["dataset"] == "ds1"
    assert res["num_thread"] == 4
    assert "machine" in res
    assert "cpu" in res
    assert "error" not in res


@pytest.mark.usefixtures("patch_machine_info")
def test_collect_metrics_with_error(example_graph, mock_algorithm_config):
    res = collect_metrics.fn(
        execution_time=99.99,
        execution_time_with_preloading=100.01,
        peak_memory=1234,
        graph=example_graph,
        algo_config=mock_algorithm_config,
        backend="networkx",
        dataset_name="ds1",
        num_thread=1,
        validation_status="failed",
        validation_message="uh oh",
        error="some error",
    )
    assert res["error"] == "some error"
    import math

    assert math.isnan(res["execution_time"])
    assert math.isnan(res["execution_time_with_preloading"])
    assert math.isnan(res["memory_used"])


###############################################################################
#                          TEST: teardown_specific                            #
###############################################################################


@pytest.mark.parametrize("backend", ["parallel", "cugraph", "networkx"])
def test_teardown_specific(backend):
    teardown_specific.fn(backend)
    if backend == "cugraph":
        # cugraph sets an env variable
        assert os.environ["NX_CUGRAPH_AUTOCONFIG"] == "False"


###############################################################################
#                       TEST: run_single_benchmark                            #
###############################################################################


@pytest.mark.asyncio
@patch("nxbench.benchmarks.benchmark.memory_tracker", autospec=True)
@patch("nxbench.benchmarks.benchmark.teardown_specific", autospec=True)
async def test_run_single_benchmark_success(
    mock_teardown,
    mock_memory_tracker,
    mock_algorithm_config,
    mock_dataset_config,
    example_graph,
    patch_machine_info,
):
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = {"peak": 5000000}
    mock_cm.__exit__.return_value = None
    mock_memory_tracker.return_value = mock_cm

    with (
        patch(
            "nxbench.benchmarks.benchmark.configure_backend",
            new=MagicMock(return_value="graph_after_config"),
        ),
        patch(
            "nxbench.benchmarks.benchmark.run_algorithm",
            new=MagicMock(return_value=("result", 1.5, 5000000, None)),
        ),
        patch(
            "nxbench.benchmarks.benchmark.validate_results",
            new=MagicMock(return_value=("passed", "")),
        ),
        patch(
            "nxbench.benchmarks.benchmark.collect_metrics",
            new=MagicMock(return_value={"metric": "dummy_value"}),
        ) as mock_collect,
    ):
        res = await run_single_benchmark(
            backend="networkx",
            num_thread=2,
            algo_config=mock_algorithm_config,
            dataset_config=mock_dataset_config,
            original_graph=example_graph,
        )
        mock_collect.assert_called_once()
        _, collected_kwargs = mock_collect.call_args
        assert "execution_time" in collected_kwargs
        assert "execution_time_with_preloading" in collected_kwargs
        assert res == {"metric": "dummy_value"}


@pytest.mark.asyncio
@patch("nxbench.benchmarks.benchmark.memory_tracker", autospec=True)
@patch("nxbench.benchmarks.benchmark.teardown_specific", autospec=True)
async def test_run_single_benchmark_exception(
    mock_teardown,
    mock_memory_tracker,
    mock_algorithm_config,
    mock_dataset_config,
    example_graph,
    patch_machine_info,
):
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = {"peak": 5000000}
    mock_cm.__exit__.return_value = None
    mock_memory_tracker.return_value = mock_cm

    with (
        patch(
            "nxbench.benchmarks.benchmark.configure_backend",
            side_effect=ValueError("backend error"),
        ),
        patch(
            "nxbench.benchmarks.benchmark.collect_metrics",
            return_value={"error": "wrapped"},
        ) as mock_collect,
    ):
        res = await run_single_benchmark(
            backend="networkx",
            num_thread=2,
            algo_config=mock_algorithm_config,
            dataset_config=mock_dataset_config,
            original_graph=example_graph,
        )
        mock_collect.assert_called_once()
        assert res == {"error": "wrapped"}


###############################################################################
#                          TEST: benchmark_suite                              #
###############################################################################


@pytest.mark.asyncio
@patch("nxbench.benchmarks.benchmark.DaskTaskRunner", new=MagicMock())
@patch("nxbench.benchmarks.benchmark.flow", new=lambda *args, **kwargs: lambda fn: fn)
@patch("nxbench.benchmarks.benchmark.run_single_benchmark", new_callable=AsyncMock)
async def test_benchmark_suite_success(
    mock_run_single,
    mock_algorithm_config,
    mock_dataset_config,
    example_graph,
):
    """
    Replace the @flow(...) decorator with a no-op so any subflow is
    just a normal async function. Also mocks out DaskTaskRunner so we
    don't trigger Prefect's telemetry code or spawn real Dask workers.
    """
    mock_run_single.return_value = {"metric": "dummy_value"}

    algorithms = [mock_algorithm_config]
    datasets = [mock_dataset_config]
    backends = ["networkx"]
    threads = [2]
    graphs = {"ds1": (example_graph, {"meta": "data"})}

    # by calling __wrapped__, we avoid the main @flow on benchmark_suite?
    results = await benchmark_suite.__wrapped__(
        algorithms=algorithms,
        datasets=datasets,
        backends=backends,
        threads=threads,
        graphs=graphs,
    )

    assert len(results) == 1
    assert results[0] == {"metric": "dummy_value"}


@pytest.mark.asyncio
async def test_benchmark_suite_missing_dataset(mock_algorithm_config):
    algorithms = [mock_algorithm_config]
    datasets = [DatasetConfig(name="not_in_graphs", source="networkrepository")]
    backends = ["networkx"]
    threads = [1]
    graphs = {}

    results = await benchmark_suite.__wrapped__(
        algorithms=algorithms,
        datasets=datasets,
        backends=backends,
        threads=threads,
        graphs=graphs,
    )

    assert results == []


###############################################################################
#                          TEST: main_benchmark                               #
###############################################################################


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_machine_info", "patch_python_version")
@patch("nxbench.benchmarks.benchmark.setup_cache", return_value={"ds1": ("graph", {})})
@patch("nxbench.benchmarks.benchmark.benchmark_suite", new_callable=AsyncMock)
@patch("nxbench.benchmarks.benchmark.load_config")
@patch("nxbench.benchmarks.benchmark.Path", autospec=True)
async def test_main_benchmark_success(
    mock_path_cls,
    mock_load_config,
    mock_benchmark_suite,
    mock_setup_cache,
    mock_benchmark_config,
    tmp_path,
    caplog,
):
    mock_path_obj = MagicMock()
    mock_path_obj.mkdir.return_value = None

    def path_side_effect(arg):
        if arg == "results":
            return tmp_path
        return tmp_path / arg

    mock_path_cls.side_effect = path_side_effect

    mock_benchmark_suite.return_value = [{"some": "result"}]
    mock_load_config.return_value = mock_benchmark_config

    with patch(
        "nxbench.benchmarks.benchmark.get_available_backends",
        return_value={"networkx": "3.4.1"},
    ):
        await main_benchmark(results_dir=tmp_path)

        files = list(tmp_path.iterdir())
        assert len(files) == 1, f"Expected 1 file in {tmp_path}, found {files}"
        with files[0].open("r") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["some"] == "result"


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_machine_info")
@patch("nxbench.benchmarks.benchmark.setup_cache", return_value={"ds1": ("graph", {})})
@patch("nxbench.benchmarks.benchmark.benchmark_suite", new_callable=AsyncMock)
@patch("nxbench.benchmarks.benchmark.load_config")
@patch("nxbench.benchmarks.benchmark.Path", autospec=True)
async def test_main_benchmark_no_backends(
    mock_path_cls,
    mock_load_config,
    mock_benchmark_suite,
    mock_setup_cache,
    mock_benchmark_config,
    tmp_path,
    caplog,
):
    mock_path_obj = MagicMock()
    mock_path_obj.mkdir.return_value = None

    def path_side_effect(arg):
        if arg == "results":
            return tmp_path
        return tmp_path / arg

    mock_path_cls.side_effect = path_side_effect

    new_config = mock_benchmark_config.copy()
    new_config["env_data"]["backend"] = {"some_nonexistent_backend": ["1.0"]}
    mock_load_config.return_value = new_config

    with patch(
        "nxbench.benchmarks.benchmark.get_available_backends",
        return_value={"networkx": "3.4.1"},
    ):
        await main_benchmark(results_dir=tmp_path)
        assert any(
            "No valid backends found or matched. Exiting." in rec.message
            for rec in caplog.records
        )

        files = list(tmp_path.iterdir())
        assert len(files) == 1, f"Expected 1 file in {tmp_path}, found {files}"
        with files[0].open("r") as f:
            data = json.load(f)
            assert len(data) == 0


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_machine_info")
@patch("nxbench.benchmarks.benchmark.setup_cache", return_value={"ds1": ("graph", {})})
@patch("nxbench.benchmarks.benchmark.benchmark_suite", new_callable=AsyncMock)
@patch("nxbench.benchmarks.benchmark.load_config")
@patch("nxbench.benchmarks.benchmark.Path", autospec=True)
async def test_main_benchmark_no_python_match(
    mock_path_cls,
    mock_load_config,
    mock_benchmark_suite,
    mock_setup_cache,
    mock_benchmark_config,
    tmp_path,
    caplog,
):
    mock_path_obj = MagicMock()
    mock_path_obj.mkdir.return_value = None

    def path_side_effect(arg):
        if arg == "results":
            return tmp_path
        return tmp_path / arg

    mock_path_cls.side_effect = path_side_effect

    new_config = mock_benchmark_config.copy()
    new_config["env_data"]["pythons"] = ["3.9"]
    mock_load_config.return_value = new_config

    with patch(
        "nxbench.benchmarks.benchmark.get_available_backends",
        return_value={"networkx": "3.4.1"},
    ):
        await main_benchmark(results_dir=tmp_path)
        assert any(
            "No requested Python version matches the actual interpreter" in rec.message
            for rec in caplog.records
        )

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        with files[0].open("r") as f:
            data = json.load(f)
            assert len(data) == 0
