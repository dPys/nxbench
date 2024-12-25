import os
import random
import tracemalloc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nxbench.benchmarking.config import BenchmarkConfig
from nxbench.benchmarking.utils import (
    MemorySnapshot,
    add_seeding,
    configure_benchmarks,
    get_available_algorithms,
    get_available_backends,
    get_benchmark_config,
    get_machine_info,
    get_python_version,
    is_graphblas_available,
    is_nx_cugraph_available,
    is_nx_parallel_available,
    load_default_config,
    memory_tracker,
    process_algorithm_params,
)


@pytest.fixture(autouse=True)
def _reset_benchmark_config():
    """Reset the global _BENCHMARK_CONFIG to None
    before and after every test to avoid side effects.
    """
    import nxbench.benchmarking.utils

    original_config = nxbench.benchmarking.utils._BENCHMARK_CONFIG
    nxbench.benchmarking.utils._BENCHMARK_CONFIG = None
    yield
    nxbench.benchmarking.utils._BENCHMARK_CONFIG = original_config


def test_configure_benchmarks_already_set():
    """Test that configure_benchmarks raises a ValueError if
    the global _BENCHMARK_CONFIG is already set.
    """
    bc = BenchmarkConfig(algorithms=[], datasets=[], env_data={}, machine_info={})
    configure_benchmarks(bc)  # sets _BENCHMARK_CONFIG
    with pytest.raises(ValueError, match="Benchmark configuration already set"):
        configure_benchmarks(bc)


def test_configure_benchmarks_invalid_type():
    """Test that configure_benchmarks raises a TypeError if a non-string,
    non-BenchmarkConfig is passed.
    """
    with pytest.raises(TypeError, match="Invalid type for configuration"):
        configure_benchmarks(123)


def test_configure_benchmarks_with_benchmark_config():
    """Test passing a BenchmarkConfig instance directly to configure_benchmarks."""
    bc = BenchmarkConfig(algorithms=[], datasets=[], env_data={}, machine_info={})
    configure_benchmarks(bc)
    # The global config should match bc
    assert get_benchmark_config() is bc


def test_configure_benchmarks_with_str():
    """Test passing a string (path) to configure_benchmarks. We patch
    BenchmarkConfig.from_yaml to ensure it's called correctly.
    """
    with patch.object(
        BenchmarkConfig, "from_yaml", return_value="mock_config"
    ) as mock_func:
        configure_benchmarks("fake_path.yaml")
        mock_func.assert_called_once_with("fake_path.yaml")
        # The global config should now be "mock_config"
        assert get_benchmark_config() == "mock_config"


def test_get_benchmark_config_no_env_no_default():
    """
    If no environment variable is set and no config is set yet,
    get_benchmark_config() calls load_default_config().
    We patch load_default_config to see if it was called.
    """
    with patch(
        "nxbench.benchmarking.utils.load_default_config", return_value="default_config"
    ) as mock_default:
        config = get_benchmark_config()
        assert config == "default_config"
        mock_default.assert_called_once()


def test_get_benchmark_config_with_env_file_not_found():
    """
    Ensure a FileNotFoundError is raised if NXBENCH_CONFIG_FILE is set
    but the file does not exist.
    """
    with patch.dict(os.environ, {"NXBENCH_CONFIG_FILE": "test_config.yaml"}):
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                get_benchmark_config()


def test_get_benchmark_config_with_env_file_found():
    """
    Ensure that if NXBENCH_CONFIG_FILE is set and found,
    BenchmarkConfig.from_yaml is called with the correct path.
    """
    with patch.dict(os.environ, {"NXBENCH_CONFIG_FILE": "test_config.yaml"}):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_absolute", return_value=True),
            patch.object(
                BenchmarkConfig, "from_yaml", return_value="loaded_from_env"
            ) as mock_func,
        ):
            config = get_benchmark_config()
            assert config == "loaded_from_env"
            mock_func.assert_called_once()


def test_load_default_config():
    """
    Directly test load_default_config to ensure it returns a valid BenchmarkConfig
    with the expected structure and defaults.
    """
    default = load_default_config()
    assert isinstance(default, BenchmarkConfig)
    assert len(default.algorithms) > 0
    assert len(default.datasets) > 0
    assert "num_threads" in default.env_data
    assert "backend" in default.env_data
    assert "pythons" in default.env_data
    # machine_info might be empty by default
    assert isinstance(default.machine_info, dict)


def test_backend_availability():
    """Test backend availability detection."""
    with patch("importlib.util.find_spec") as mock_find_spec:
        # test when backends are available
        mock_find_spec.return_value = True
        assert is_nx_cugraph_available() is True
        assert is_graphblas_available() is True
        assert is_nx_parallel_available() is True

        # test when backends are not available
        mock_find_spec.return_value = None
        assert is_nx_cugraph_available() is False
        assert is_graphblas_available() is False
        assert is_nx_parallel_available() is False


def test_get_available_backends():
    """
    Test getting the dictionary of available backends
    under two scenarios:
      1. All optional libraries (nx_cugraph, graphblas_algorithms, nx_parallel) are
      available
      2. Only networkx is available
    """
    mock_networkx_module = type("MockNetworkX", (), {"__version__": "3.4.1"})
    mock_cugraph_module = type("MockCugraph", (), {"__version__": "1.0.0"})
    mock_graphblas_module = type("MockGraphblas", (), {"__version__": "2023.10.0"})
    mock_parallel_module = type("MockParallel", (), {"__version__": "0.3rc0.dev0"})

    # First scenario: everything is installed
    with (
        patch("nxbench.benchmarking.utils.is_nx_cugraph_available", return_value=True),
        patch("nxbench.benchmarking.utils.is_graphblas_available", return_value=True),
        patch("nxbench.benchmarking.utils.is_nx_parallel_available", return_value=True),
        patch(
            "nxbench.benchmarking.utils.importlib.import_module"
        ) as mock_import_module,
        patch("nxbench.benchmarking.utils.get_version") as mock_get_version,
    ):

        def import_side_effect(name, *args, **kwargs):
            if name == "networkx":
                return mock_networkx_module
            if name == "nx_cugraph":
                return mock_cugraph_module
            if name == "graphblas_algorithms":
                return mock_graphblas_module
            if name == "nx_parallel":
                return mock_parallel_module
            raise ImportError(f"No module named {name}")

        mock_import_module.side_effect = import_side_effect

        def version_side_effect(name):
            if name == "networkx":
                return "3.4.1"
            if name == "nx_cugraph":
                return "1.0.0"
            if name == "graphblas_algorithms":
                return "2023.10.0"
            if name == "nx_parallel":
                return "0.3rc0.dev0"
            raise ImportError(f"Unknown package: {name}")

        mock_get_version.side_effect = version_side_effect

        backends = get_available_backends()
        assert "networkx" in backends
        assert backends["networkx"] == "3.4.1"
        assert "cugraph" in backends
        assert backends["cugraph"] == "1.0.0"
        assert "graphblas" in backends
        assert backends["graphblas"] == "2023.10.0"
        assert "parallel" in backends
        assert backends["parallel"] == "0.3rc0.dev0"

    # Second scenario: only networkx is installed/available
    with (
        patch("nxbench.benchmarking.utils.is_nx_cugraph_available", return_value=False),
        patch("nxbench.benchmarking.utils.is_graphblas_available", return_value=False),
        patch(
            "nxbench.benchmarking.utils.is_nx_parallel_available", return_value=False
        ),
        patch(
            "nxbench.benchmarking.utils.importlib.import_module"
        ) as mock_import_module,
        patch("nxbench.benchmarking.utils.get_version") as mock_get_version,
    ):

        def only_networkx_side_effect(name, *args, **kwargs):
            if name == "networkx":
                return mock_networkx_module
            raise ImportError(f"No module named {name}")

        mock_import_module.side_effect = only_networkx_side_effect

        def version_networkx_side_effect(pkg_name):
            if pkg_name == "networkx":
                return "3.4.1"
            raise ImportError(f"Package not found: {pkg_name}")

        mock_get_version.side_effect = version_networkx_side_effect

        backends = get_available_backends()
        # In this scenario, we expect only networkx
        assert len(backends) == 1
        assert "networkx" in backends
        assert backends["networkx"] == "3.4.1"


def test_get_python_version():
    """Test Python version string formatting."""
    version = get_python_version()
    assert len(version.split(".")) == 3
    for part in version.split("."):
        assert part.isdigit()


def test_configure_benchmarks_env_vars():
    """
    Already tested environment variable usage in detail above,
    but this test ensures FileNotFoundError is raised if path not found.
    """
    with patch.dict(os.environ, {"NXBENCH_CONFIG_FILE": "test_config.yaml"}):
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            with pytest.raises(FileNotFoundError):
                get_benchmark_config()


def test_memory_tracking():
    """Test memory usage tracking context manager."""

    def allocate_memory():
        return [0] * 1000000

    with memory_tracker() as mem:
        data = allocate_memory()
        # ensure data isn't garbage-collected prematurely
        assert len(data) == 1000000

    assert "current" in mem
    assert "peak" in mem
    assert isinstance(mem["current"], int)
    assert isinstance(mem["peak"], int)
    assert mem["peak"] >= mem["current"]


def test_memory_snapshot_compare():
    """Test direct usage of MemorySnapshot and compare_to."""
    tracemalloc.start()

    s1 = MemorySnapshot()
    s2 = MemorySnapshot()

    current, peak = s2.compare_to(s1)
    assert current == 0
    assert peak == 0

    s1.take()
    s2.take()
    current, peak = s2.compare_to(s1)
    assert isinstance(current, int)
    assert isinstance(peak, int)

    tracemalloc.stop()


def test_get_available_algorithms():
    """
    Test that get_available_algorithms looks for modules in ALGORITHM_SUBMODULES
    and filters attributes correctly.
    """
    fake_submodules = ["networkx.algorithms.approximation"]
    mock_module = MagicMock()
    mock_module.__name__ = "networkx.algorithms.approximation"

    def testfunc():
        pass

    def _privatefunc():
        pass

    mock_module.testfunc = testfunc
    mock_module._privatefunc = _privatefunc

    with (
        patch("nxbench.benchmarking.constants.ALGORITHM_SUBMODULES", fake_submodules),
        patch("nxbench.benchmarking.utils.importlib.util.find_spec", return_value=True),
        patch(
            "nxbench.benchmarking.utils.importlib.import_module",
            return_value=mock_module,
        ),
    ):
        algos = get_available_algorithms()
        assert "approximate_testfunc" in algos
        assert "_privatefunc" not in algos


def test_get_machine_info():
    info = get_machine_info()
    assert isinstance(info, dict)
    assert "arch" in info
    assert "cpu" in info
    assert "num_cpu" in info
    assert "os" in info
    assert "ram" in info


def test_process_algorithm_params():
    params = {
        "_pos1": "42",
        "kw1": "3.14",
        "kw2": "not a number",
        "nested": {"func": "os.path.join"},
    }
    pos_args, kwargs = process_algorithm_params(params)
    # pos_args should have parsed '42' -> int(42)
    assert pos_args == [42]
    # kw1 -> 3.14 (float), kw2 -> 'not a number'
    assert kwargs["kw1"] == 3.14
    assert kwargs["kw2"] == "not a number"
    import os

    assert kwargs["nested"] is os.path.join


def test_process_algorithm_params_non_numeric_string():
    # Non-numeric string should remain unchanged
    params = {"_pos1": "hello"}
    pos_args, kwargs = process_algorithm_params(params)
    assert pos_args == ["hello"]
    assert kwargs == {}


def dummy_func_no_seed():
    pass


def dummy_func_seed(seed):
    pass


def dummy_func_random_state(random_state=None):
    pass


def dummy_func_both(seed, random_state=None):
    pass


@pytest.mark.parametrize(
    ("algo_func", "kwargs_in", "expected_kwargs"),
    [
        # No seeding in function signature
        (dummy_func_no_seed, {"seed": 42}, {}),
        # Only seed in function signature
        (dummy_func_seed, {"seed": 42}, {"seed": 42}),
        # Only random_state in function signature but we do not request local RNG
        (dummy_func_random_state, {"seed": 42, "use_local_random_state": False}, {}),
        # Only random_state in function signature and we request local RNG
        (
            dummy_func_random_state,
            {"seed": 42, "use_local_random_state": True},
            {"random_state": np.random.RandomState},
        ),
        # Both in function signature, but no local RNG
        (dummy_func_both, {"seed": 42, "use_local_random_state": False}, {"seed": 42}),
        # Both in function signature with local RNG
        (
            dummy_func_both,
            {"seed": 42, "use_local_random_state": True},
            {"seed": 42, "random_state": np.random.RandomState},
        ),
    ],
)
def test_add_seeding(algo_func, kwargs_in, expected_kwargs):
    """
    Test that add_seeding modifies kwargs as expected given different
    function signatures and input parameters.
    """
    old_py_seed = random.randint(0, 100000)
    old_np_seed = np.random.randint(0, 100000)

    updated = add_seeding(kwargs_in, algo_func, algo_func.__name__)

    for k, v in expected_kwargs.items():
        if k == "random_state":
            assert k in updated
            assert isinstance(updated[k], np.random.RandomState)
        else:
            assert k in updated
            assert updated[k] == v

    if "seed" not in expected_kwargs:
        if "seed" in kwargs_in:
            pass


def test_add_seeding_non_int_seed():
    """
    Test that if the 'seed' in kwargs is non-integer, no global seeds
    are set, and none are passed to the algorithm function.
    """
    old_python_random_state = random.getstate()
    old_numpy_random_state = np.random.get_state()

    try:
        kwargs_in = {"seed": "not_an_int"}

        def dummy_func_no_seed():
            pass

        updated_kwargs = add_seeding(
            kwargs_in, dummy_func_no_seed, "dummy_func_no_seed"
        )

        assert "seed" not in updated_kwargs

        assert random.getstate() == old_python_random_state
        assert np.allclose(np.random.get_state()[1], old_numpy_random_state[1])

    finally:
        # Restore original states
        random.setstate(old_python_random_state)
        np.random.set_state(old_numpy_random_state)


@pytest.mark.parametrize("relative_path", ["some_relative_config.yaml", "./conf.yaml"])
def test_get_benchmark_config_relative_path(relative_path, tmp_path):
    """
    Ensure that if NXBENCH_CONFIG_FILE is set to a relative path,
    we correctly resolve the absolute path before calling from_yaml.
    """
    config_file = tmp_path / "some_relative_config.yaml"
    config_file.write_text("benchmarks: []")

    with (
        patch.dict(os.environ, {"NXBENCH_CONFIG_FILE": relative_path}),
        patch(
            "pathlib.Path.exists",
            return_value=True,
        ),
        patch(
            "pathlib.Path.is_absolute",
            return_value=False,
        ),
        patch.object(
            BenchmarkConfig, "from_yaml", return_value="loaded_from_relative_path"
        ) as mock_from_yaml,
        patch("pathlib.Path.resolve", return_value=config_file) as mock_resolve,
    ):
        config = get_benchmark_config()
        mock_from_yaml.assert_called_once_with(str(config_file))
        mock_resolve.assert_called_once()
        assert config == "loaded_from_relative_path"


def test_process_algorithm_params_dict_no_func():
    """
    Ensure that if a dictionary param does not have a "func" key,
    it remains unchanged and is treated as a normal dictionary.
    """
    params = {
        "_pos1": {"some_data": 123},
        "kw1": {"nested": {"no_func_here": True}},
    }
    pos_args, kwargs = process_algorithm_params(params)
    assert pos_args == [{"some_data": 123}]
    assert kwargs["kw1"] == {"nested": {"no_func_here": True}}


def test_configure_benchmarks_returns_existing_if_set():
    """
    Once a BenchmarkConfig is set, calling get_benchmark_config() again
    should return the same object rather than re-loading or re-parsing.
    """
    bc = BenchmarkConfig(algorithms=[], datasets=[], env_data={}, machine_info={})
    configure_benchmarks(bc)

    with patch("nxbench.benchmarking.utils.load_default_config") as mock_default:
        same_config = get_benchmark_config()
        mock_default.assert_not_called()
        assert same_config is bc
