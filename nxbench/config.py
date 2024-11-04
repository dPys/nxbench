"""Benchmark configuration handling."""

import logging
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
    func: str  # fully qualified function name (e.g. "networkx.pagerank")
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
        module = __import__(module_path, fromlist=[func_name])
        self.func_ref = getattr(module, func_name)

        if self.validate_result:
            mod_path, val_func = self.validate_result.rsplit(".", 1)
            module = __import__(mod_path, fromlist=[val_func])
            self.validate_ref = getattr(module, val_func)
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
    output_dir: Path = Path("benchmark_results")
    histogram: bool = True
    json_report: bool = True
    compare: bool = True
    machine_info: Dict[str, Any] = field(default_factory=dict)
    pytest_benchmark_options: Dict[str, Any] = field(default_factory=dict)

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

        algorithms = [
            AlgorithmConfig(**algo_data) for algo_data in data.get("algorithms", [])
        ]

        datasets = [DatasetConfig(**ds_data) for ds_data in data.get("datasets", [])]

        return cls(
            algorithms=algorithms,
            datasets=datasets,
            output_dir=Path(data.get("output_dir", "benchmark_results")),
            histogram=data.get("histogram", True),
            json_report=data.get("json_report", True),
            compare=data.get("compare", True),
            machine_info=data.get("machine_info", {}),
            pytest_benchmark_options=data.get("pytest_benchmark_options", {}),
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
            "output_dir": str(self.output_dir),
            "histogram": self.histogram,
            "json_report": self.json_report,
            "compare": self.compare,
            "machine_info": self.machine_info,
            "pytest_benchmark_options": self.pytest_benchmark_options,
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

    Raises
    ------
    RuntimeError
        If configuration hasn't been set
    """
    global _BENCHMARK_CONFIG
    if _BENCHMARK_CONFIG is None:
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
                func="networkx.pagerank",
                params={"alpha": 0.85},
                groups=["centrality"],
            ),
            # AlgorithmConfig(
            #     name="betweenness_centrality",
            #     func="networkx.betweenness_centrality",
            #     params={"k": 100, "normalized": True},
            #     requires_directed=False,
            #     groups=["centrality", "path_based"],
            # ),
            AlgorithmConfig(
                name="louvain_communities",
                func="networkx.algorithms.community.louvain_communities",
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
