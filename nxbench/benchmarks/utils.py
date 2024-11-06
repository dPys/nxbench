import os
from pathlib import Path
from typing import Optional, cast

from nxbench.benchmarks.config import AlgorithmConfig, BenchmarkConfig, DatasetConfig


class BenchmarkManager:
    """Singleton class to manage benchmark configuration."""

    _instance: Optional["BenchmarkManager"] = None
    _config: BenchmarkConfig | None = None

    def __new__(cls) -> "BenchmarkManager":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast(BenchmarkManager, cls._instance)

    @property
    def config(self) -> BenchmarkConfig:
        """Get current configuration or load default."""
        if self._config is not None:
            return self._config

        config_path = os.environ.get("NXBENCH_CONFIG_FILE")
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                self._config = BenchmarkConfig.from_yaml(config_file)
                return self._config
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        self._config = self._load_default_config()
        return self._config

    def configure(self, config: BenchmarkConfig | Path | str) -> None:
        """Configure the benchmark suite.

        Parameters
        ----------
        config : BenchmarkConfig or Path or str
            Either a BenchmarkConfig instance or path to a YAML config file

        Raises
        ------
        ValueError
            If configuration is already set
        TypeError
            If configuration type is invalid
        """
        if self._config is not None:
            raise ValueError("Benchmark configuration already set")

        if isinstance(config, (str, Path)):
            self._config = BenchmarkConfig.from_yaml(config)
        elif isinstance(config, BenchmarkConfig):
            self._config = config
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

        self._config.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_default_config() -> BenchmarkConfig:
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


def is_cugraph_available() -> bool:
    """Check if cugraph package is available."""
    try:
        import importlib.util

        return importlib.util.find_spec("cugraph") is not None
    except ImportError:
        return False


def is_graphblas_available() -> bool:
    """Check if graphblas package is available."""
    try:
        import importlib.util

        return importlib.util.find_spec("graphblas") is not None
    except ImportError:
        return False


def is_nx_parallel_available() -> bool:
    """Check if nx_parallel package is available."""
    try:
        import importlib.util

        return importlib.util.find_spec("nx_parallel") is not None
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
    BenchmarkManager().configure(config)


def get_benchmark_config() -> BenchmarkConfig:
    """Get the current benchmark configuration.

    Returns
    -------
    BenchmarkConfig
        Current configuration
    """
    return BenchmarkManager().config


def load_default_config() -> BenchmarkConfig:
    """Load the default benchmark configuration."""
    return BenchmarkManager._load_default_config()
