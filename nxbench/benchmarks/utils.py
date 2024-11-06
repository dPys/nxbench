import os
from pathlib import Path
from typing import Optional

from nxbench.benchmarks.config import AlgorithmConfig, BenchmarkConfig, DatasetConfig

_BENCHMARK_CONFIG: Optional["BenchmarkConfig"] = None


def is_cugraph_available():
    try:
        import cugraph

        return True
    except ImportError:
        return False


def is_graphblas_available():
    try:
        import graphblas

        return True
    except ImportError:
        return False


def is_nx_parallel_available():
    try:
        import nx_parallel

        return True
    except ImportError:
        return False


def configure_benchmarks(config: BenchmarkConfig | Path | str) -> None:
    """Configure the benchmark suite.

    Parameters
    ----------
    config : BenchmarkConfig or Path or str
        Either a BenchmarkConfig instance or path to a YAML config file

    Raises
    ------
    ValueError
        If configuration is invalid or already set
    """
    global _BENCHMARK_CONFIG

    if _BENCHMARK_CONFIG is not None:
        raise ValueError("Benchmark configuration already set")

    if isinstance(config, (str, Path)):
        config = BenchmarkConfig.from_yaml(config)
    elif not isinstance(config, BenchmarkConfig):
        raise ValueError(f"Invalid config type: {type(config)}")

    _BENCHMARK_CONFIG = config

    config.output_dir.mkdir(parents=True, exist_ok=True)


def get_benchmark_config() -> BenchmarkConfig:
    """Get the current benchmark configuration.

    Returns
    -------
    BenchmarkConfig
        Current configuration
    """
    global _BENCHMARK_CONFIG
    if _BENCHMARK_CONFIG is not None:
        return _BENCHMARK_CONFIG

    config_path = os.environ.get("NXBENCH_CONFIG_FILE")
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            _BENCHMARK_CONFIG = BenchmarkConfig.from_yaml(config_file)
            return _BENCHMARK_CONFIG
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    _BENCHMARK_CONFIG = load_default_config()
    return _BENCHMARK_CONFIG


def load_default_config() -> BenchmarkConfig:
    """Load the default benchmark configuration."""
    return BenchmarkConfig(
        algorithms=[
            AlgorithmConfig(
                name="pagerank",
                func="networkx.algorithms.link_analysis.pagerank_alg.pagerank",
                params={"alpha": 0.85},
                groups=["centrality"],
            ),
            AlgorithmConfig(
                name="louvain_communities",
                func="networkx.algorithms.community.louvain.louvain_communities",
                requires_undirected=True,
                groups=["community"],
            ),
        ],
        datasets=[
            DatasetConfig(name="08blocks", source="networkrepository"),
            DatasetConfig(name="jazz", source="networkrepository"),
            DatasetConfig(name="karate", source="networkrepository"),
            DatasetConfig(name="patentcite", source="networkrepository"),
            DatasetConfig(name="IMDB", source="networkrepository"),
            DatasetConfig(name="citeseer", source="networkrepository"),
            DatasetConfig(name="enron", source="networkrepository"),
            DatasetConfig(name="twitter", source="networkrepository"),
        ],
    )
