import os
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

from nxbench.benchmarks.utils import (
    get_available_backends,
    get_benchmark_config,
    get_python_version,
    is_graphblas_available,
    is_nx_cugraph_available,
    is_nx_parallel_available,
)


@pytest.fixture(autouse=True)
def _reset_benchmark_config():
    import nxbench.benchmarks.utils

    original_config = nxbench.benchmarks.utils._BENCHMARK_CONFIG
    nxbench.benchmarks.utils._BENCHMARK_CONFIG = None
    yield
    nxbench.benchmarks.utils._BENCHMARK_CONFIG = original_config


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
    # Mock objects to emulate each library with an identifiable __version__
    mock_networkx_module = type("MockNetworkX", (), {"__version__": "3.4.1"})
    mock_cugraph_module = type("MockCugraph", (), {"__version__": "1.0.0"})
    mock_graphblas_module = type("MockGraphblas", (), {"__version__": "2023.10.0"})
    mock_parallel_module = type("MockParallel", (), {"__version__": "0.3rc0.dev0"})

    # First scenario: everything is installed
    with (
        patch("nxbench.benchmarks.utils.is_nx_cugraph_available", return_value=True),
        patch("nxbench.benchmarks.utils.is_graphblas_available", return_value=True),
        patch("nxbench.benchmarks.utils.is_nx_parallel_available", return_value=True),
        patch("nxbench.benchmarks.utils.importlib.import_module") as mock_import_module,
        patch("nxbench.benchmarks.utils.get_version") as mock_get_version,
    ):

        def import_side_effect(name, *args, **kwargs):
            """
            Return the correct mock module based on the import name.
            Raise ImportError for anything else.
            """
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
            """
            Return a made-up version for each package.
            Raise PackageNotFoundError if we aren't handling that package here.
            """
            if name == "networkx":
                return "3.4.1"
            if name == "nx_cugraph":
                return "1.0.0"
            if name == "graphblas_algorithms":
                return "2023.10.0"
            if name == "nx_parallel":
                return "0.3rc0.dev0"
            raise PackageNotFoundError(f"Package not found: {name}")

        mock_get_version.side_effect = version_side_effect

        backends = get_available_backends()
        assert "networkx" in backends
        assert backends["networkx"] == "3.4.1"
        # Since is_nx_cugraph_available() is True and we mocked its import,
        # cugraph should appear in the dictionary:
        assert "cugraph" in backends
        assert backends["cugraph"] == "1.0.0"
        assert "graphblas" in backends
        assert backends["graphblas"] == "2023.10.0"
        assert "parallel" in backends
        assert backends["parallel"] == "0.3rc0.dev0"

    # Second scenario: only networkx is installed/available
    with (
        patch("nxbench.benchmarks.utils.is_nx_cugraph_available", return_value=False),
        patch("nxbench.benchmarks.utils.is_graphblas_available", return_value=False),
        patch("nxbench.benchmarks.utils.is_nx_parallel_available", return_value=False),
        patch("nxbench.benchmarks.utils.importlib.import_module") as mock_import_module,
        patch("nxbench.benchmarks.utils.get_version") as mock_get_version,
    ):

        def only_networkx_side_effect(name, *args, **kwargs):
            if name == "networkx":
                return mock_networkx_module
            raise ImportError(f"No module named {name}")

        mock_import_module.side_effect = only_networkx_side_effect
        mock_get_version.side_effect = lambda pkg_name: (
            "3.4.1" if pkg_name == "networkx" else PackageNotFoundError(pkg_name)
        )

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
    """Test configuration with environment variables."""
    with patch.dict(os.environ, {"NXBENCH_CONFIG_FILE": "test_config.yaml"}):
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            with pytest.raises(FileNotFoundError):
                get_benchmark_config()


def test_memory_tracking():
    """Test memory usage tracking context manager."""
    from nxbench.benchmarks.utils import memory_tracker

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
