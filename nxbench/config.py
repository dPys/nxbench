"""Benchmark configuration handling."""

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")

_BENCHMARK_CONFIG: Optional["BenchmarkConfig"] = None

__all__ = [
    "AlgorithmConfig",
    "DatasetConfig",
    "BenchmarkConfig",
    "configure_benchmarks",
    "get_benchmark_config",
    "load_default_config",
]


@dataclass
class AlgorithmConfig:
    """Configuration for a graph algorithm to benchmark."""

    name: str
    func: str
    params: Dict[str, Any] = field(default_factory=dict)
    requires_directed: bool = False
    requires_undirected: bool = False
    requires_weighted: bool = False
    validate_result: Optional[str] = None
    groups: List[str] = field(default_factory=lambda: ["default"])
    min_rounds: int = 3
    warmup: bool = True
    warmup_iterations: int = 1

    def __post_init__(self):
        """Validate and resolve the function reference."""
        module_path, func_name = self.func.rsplit(".", 1)
        try:
            module = __import__(module_path, fromlist=[func_name])
            self.func_ref = getattr(module, func_name)
        except (ImportError, AttributeError) as e:
            logger.error(
                f"Failed to import function '{self.func}' for algorithm '{self.name}': {e}"
            )
            self.func_ref = None

        if self.validate_result:
            mod_path, val_func = self.validate_result.rsplit(".", 1)
            try:
                module = __import__(mod_path, fromlist=[val_func])
                self.validate_ref = getattr(module, val_func)
            except (ImportError, AttributeError) as e:
                logger.error(
                    f"Failed to import validation function '{self.validate_result}' for algorithm '{self.name}': {e}"
                )
                self.validate_ref = None
        else:
            self.validate_ref = None


@dataclass
class DatasetConfig:
    name: str
    source: str
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class BenchmarkConfig:
    """Complete benchmark suite configuration."""

    algorithms: List[AlgorithmConfig]
    datasets: List[DatasetConfig]
    machine_info: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "BenchmarkConfig":
        """Load configuration from YAML file.

        Parameters
        ----------
        path : str or Path
            Path to YAML configuration file

        Returns
        -------
        BenchmarkConfig
            Loaded and validated configuration
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open() as f:
            data = yaml.safe_load(f)

        algorithms_data = data.get("algorithms") or []
        datasets_data = data.get("datasets") or []

        if not isinstance(algorithms_data, list):
            logger.error(f"'algorithms' should be a list in the config file: {path}")
            algorithms_data = []

        if not isinstance(datasets_data, list):
            logger.error(f"'datasets' should be a list in the config file: {path}")
            datasets_data = []

        algorithms = [AlgorithmConfig(**algo_data) for algo_data in algorithms_data]

        datasets = [DatasetConfig(**ds_data) for ds_data in datasets_data]

        return cls(
            algorithms=algorithms,
            datasets=datasets,
            machine_info=data.get("machine_info", {}),
        )

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Parameters
        ----------
        path : str or Path
            Output path for YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dictionaries
        data = {
            "algorithms": [
                {k: v for k, v in algo.__dict__.items() if not k.endswith("_ref")}
                for algo in self.algorithms
            ],
            "datasets": [
                {k: v for k, v in ds.__dict__.items()} for ds in self.datasets
            ],
            "machine_info": self.machine_info,
        }

        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False)


def configure_benchmarks(config: Union[BenchmarkConfig, Path, str]) -> None:
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

    # Set up output directory
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
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
    else:
        _BENCHMARK_CONFIG = load_default_config()
        return _BENCHMARK_CONFIG


def load_default_config() -> BenchmarkConfig:
    """Load the default benchmark configuration.

    Returns
    -------
    BenchmarkConfig
        Default configuration with common algorithms and datasets
    """
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
            DatasetConfig(
                name="08blocks",
                source="networkrepository",
            ),
            DatasetConfig(
                name="jazz",
                source="networkrepository",
            ),
        ],
    )
