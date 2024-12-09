import os
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
    """Test getting list of available backends."""
    with (
        patch("nxbench.benchmarks.utils.is_nx_cugraph_available") as mock_cugraph,
        patch("nxbench.benchmarks.utils.is_graphblas_available") as mock_graphblas,
        patch("nxbench.benchmarks.utils.is_nx_parallel_available") as mock_parallel,
    ):

        mock_cugraph.return_value = True
        mock_graphblas.return_value = True
        mock_parallel.return_value = True

        backends = get_available_backends()
        assert "networkx" in backends
        assert "cugraph" in backends
        assert "graphblas" in backends
        assert "parallel" in backends

        mock_cugraph.return_value = False
        mock_graphblas.return_value = False
        mock_parallel.return_value = False

        backends = get_available_backends()
        assert backends == ["networkx"]


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
        # ensure data exists to prevent premature garbage collection
        assert len(data) == 1000000

    assert "current" in mem
    assert "peak" in mem
    assert isinstance(mem["current"], int)
    assert isinstance(mem["peak"], int)
    assert mem["peak"] >= mem["current"]
