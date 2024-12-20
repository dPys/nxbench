import gc
import importlib
import inspect
import logging
import os
import platform
import sys
import tracemalloc
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from nxbench.benchmarks.config import AlgorithmConfig, BenchmarkConfig, DatasetConfig
from nxbench.benchmarks.constants import ALGORITHM_SUBMODULES

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("nxbench")

_BENCHMARK_CONFIG: BenchmarkConfig | None = None


def configure_benchmarks(config: BenchmarkConfig | str):
    global _BENCHMARK_CONFIG  # noqa: PLW0603
    if _BENCHMARK_CONFIG is not None:
        raise ValueError("Benchmark configuration already set")
    if isinstance(config, BenchmarkConfig):
        _BENCHMARK_CONFIG = config
    elif isinstance(config, str):
        _BENCHMARK_CONFIG = BenchmarkConfig.from_yaml(config)
    else:
        raise TypeError("Invalid type for configuration")


def get_benchmark_config() -> BenchmarkConfig:
    global _BENCHMARK_CONFIG  # noqa: PLW0603
    if _BENCHMARK_CONFIG is not None:
        return _BENCHMARK_CONFIG

    config_file = os.getenv("NXBENCH_CONFIG_FILE")
    if config_file:
        config_path = Path(config_file)

        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.debug(f"Resolved config file path: {config_path}")

        _BENCHMARK_CONFIG = BenchmarkConfig.from_yaml(str(config_path))
    else:
        _BENCHMARK_CONFIG = load_default_config()

    return _BENCHMARK_CONFIG


def load_default_config() -> BenchmarkConfig:
    default_algorithms = [
        AlgorithmConfig(
            name="pagerank",
            func="networkx.algorithms.link_analysis.pagerank_alg.pagerank",
            params={"alpha": 0.85},
        ),
    ]
    default_datasets = [
        DatasetConfig(name="08blocks", source="networkrepository"),
        DatasetConfig(name="jazz", source="networkrepository"),
        DatasetConfig(name="karate", source="networkrepository"),
        DatasetConfig(name="enron", source="networkrepository"),
    ]

    env_data = {
        "num_threads": ["1", "4"],
        "backend": {
            "networkx": ["networkx==3.4.2"],
            "graphblas": ["graphblas_algorithms==2023.10.0"],
        },
        "pythons": ["3.10", "3.11"],
    }

    return BenchmarkConfig(
        algorithms=default_algorithms,
        datasets=default_datasets,
        env_data=env_data,
        machine_info={},
    )


def is_nx_cugraph_available():
    try:
        import importlib.util
    except ImportError:
        return False
    else:
        return importlib.util.find_spec("nx_cugraph") is not None


def is_graphblas_available():
    try:
        import importlib.util
    except ImportError:
        return False
    else:
        return importlib.util.find_spec("graphblas_algorithms") is not None


def is_nx_parallel_available():
    try:
        import importlib.util
    except ImportError:
        return False
    else:
        return importlib.util.find_spec("nx_parallel") is not None


def get_python_version() -> str:
    """Get formatted Python version string."""
    version_info = sys.version_info
    return f"{version_info.major}.{version_info.minor}.{version_info.micro}"


def get_available_backends() -> dict[str, str]:
    """Return a dict of available backends and their versions."""
    available = {}

    try:
        networkx = importlib.import_module("networkx")
        nx_version = getattr(networkx, "__version__", None)
        if nx_version is None:
            nx_version = get_version("networkx")
        available["networkx"] = nx_version
    except (ImportError, PackageNotFoundError):
        pass

    if is_nx_cugraph_available():
        try:
            nx_cugraph = importlib.import_module("nx_cugraph")
            cugraph_version = getattr(nx_cugraph, "__version__", None)
            if cugraph_version is None:
                cugraph_version = get_version("nx_cugraph")
            available["cugraph"] = cugraph_version
        except (ImportError, PackageNotFoundError):
            pass

    if is_graphblas_available():
        try:
            graphblas_algorithms = importlib.import_module("graphblas_algorithms")
            gb_version = getattr(graphblas_algorithms, "__version__", None)
            if gb_version is None:
                gb_version = get_version("graphblas_algorithms")
            available["graphblas"] = gb_version
        except (ImportError, PackageNotFoundError):
            pass

    if is_nx_parallel_available():
        try:
            nx_parallel = importlib.import_module("nx_parallel")
            parallel_version = getattr(nx_parallel, "__version__", None)
            if parallel_version is None:
                parallel_version = get_version("nx_parallel")
            available["parallel"] = parallel_version
        except (ImportError, PackageNotFoundError):
            pass

    logger.debug(f"Available backends and versions: {available}")
    return available


class MemorySnapshot:
    """Class to store and diff memory snapshots."""

    def __init__(self, snapshot=None):
        """Initialize with optional tracemalloc snapshot."""
        self.snapshot = snapshot

    def take(self):
        """Take a new snapshot."""
        self.snapshot = tracemalloc.take_snapshot()

    def compare_to(self, other: "MemorySnapshot") -> tuple[int, int]:
        """Compare this snapshot to another and return (current, peak) memory diff in
        bytes.
        """
        if not self.snapshot or not other.snapshot:
            return 0, 0

        stats = self.snapshot.compare_to(other.snapshot, "lineno")
        current = sum(stat.size_diff for stat in stats)
        peak = sum(stat.size for stat in stats)
        return current, peak


@contextmanager
def memory_tracker():
    """Track memory usage of code block.

    Returns dict with 'current' and 'peak' memory usage in bytes.
    Memory usage is measured as the difference between before and after execution.
    """
    gc.collect()
    tracemalloc.start()

    baseline = MemorySnapshot()
    baseline.take()

    mem = {}
    try:
        yield mem
    finally:
        gc.collect()
        end = MemorySnapshot()
        end.take()
        current, peak = end.compare_to(baseline)

        mem["current"] = current
        mem["peak"] = peak

        tracemalloc.stop()


def get_available_algorithms():
    """Get algorithms from specified NetworkX submodules and custom
    algorithms.

    Returns
    -------
    Dict[str, Callable]
        Dictionary of available algorithms.
    """
    nx_algorithm_dict: dict[str, Callable] = {}

    for submodule in ALGORITHM_SUBMODULES:
        spec = importlib.util.find_spec(submodule)
        if spec is None:
            continue
        module = importlib.import_module(submodule)

        for attr_name in dir(module):
            if not attr_name.startswith("_") and not any(
                attr_name.startswith(prefix)
                for prefix in [
                    "is_",
                    "has_",
                    "get_",
                    "set_",
                    "contains_",
                    "write_",
                    "read_",
                    "to_",
                    "from_",
                    "generate_",
                    "make_",
                    "create_",
                    "build_",
                    "delete_",
                    "remove_",
                    "not_implemented",
                    "np_random_state",
                ]
            ):
                try:
                    attr = getattr(module, attr_name)
                except AttributeError:
                    continue
                if inspect.isfunction(attr):
                    if "approximation" in module.__name__:
                        nx_algorithm_dict[f"approximate_{attr_name}"] = attr
                    else:
                        nx_algorithm_dict[attr_name] = attr

    return nx_algorithm_dict


def get_machine_info():
    info = {
        "arch": platform.machine(),
        "cpu": platform.processor(),
        "num_cpu": str(psutil.cpu_count(logical=True)),
        "os": f"{platform.system()} {platform.release()}",
        "ram": str(psutil.virtual_memory().total),
    }
    # info["docker"] = os.path.exists("/.dockerenv")
    return info
