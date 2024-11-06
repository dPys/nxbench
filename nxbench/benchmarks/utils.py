import logging
import os
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
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        _BENCHMARK_CONFIG = BenchmarkConfig.from_yaml(config_file)
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
        AlgorithmConfig(
            name="louvain_communities",
            func="networkx.algorithms.community.louvain.louvain_communities",
            requires_undirected=True,
        ),
    ]
    default_datasets = [
        DatasetConfig(name="08blocks", source="networkrepository"),
        DatasetConfig(name="jazz", source="networkrepository"),
        DatasetConfig(name="karate", source="networkrepository"),
        DatasetConfig(name="patentcite", source="networkrepository"),
        DatasetConfig(name="IMDB", source="networkrepository"),
        DatasetConfig(name="citeseer", source="networkrepository"),
        DatasetConfig(name="enron", source="networkrepository"),
        DatasetConfig(name="twitter", source="networkrepository"),
    ]
    return BenchmarkConfig(
        algorithms=default_algorithms,
        datasets=default_datasets,
        machine_info={},
        output_dir="results",
    )


def is_cugraph_available():
    try:
        import importlib.util
    except ImportError:
        return False
    else:
        return importlib.util.find_spec("cugraph") is not None


def is_graphblas_available():
    try:
        import importlib.util
    except ImportError:
        return False
    else:
        return importlib.util.find_spec("graphblas") is not None


def is_nx_parallel_available():
    try:
        import importlib.util
    except ImportError:
        return False
    else:
        return importlib.util.find_spec("nx_parallel") is not None
