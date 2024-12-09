import gc
import logging
import os
import sys
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

from nxbench.benchmarks.config import AlgorithmConfig, BenchmarkConfig, DatasetConfig

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

    default_matrix = {
        "req": {
            "networkx": ["3.4.2"],
            "graphblas_algorithms": ["2023.10.0"],
        },
        "env_nobuild": {
            "NUM_THREAD": ["1", "4"],
        },
    }
    return BenchmarkConfig(
        algorithms=default_algorithms,
        datasets=default_datasets,
        matrix=default_matrix,
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


def get_available_backends() -> list[str]:
    backends = ["networkx"]

    if is_nx_cugraph_available():
        backends.append("cugraph")

    if is_graphblas_available():
        backends.append("graphblas")

    if is_nx_parallel_available():
        backends.append("parallel")

    logger.debug(f"Available backends: {backends}")
    return backends


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
        gc.collect()

        end = MemorySnapshot()
        end.take()
        current, peak = end.compare_to(baseline)

        mem["current"] = current
        mem["peak"] = peak

    finally:
        tracemalloc.stop()
