"""Benchmark configuration handling."""

import logging
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import yaml

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")

__all__ = [
    "AlgorithmConfig",
    "DatasetConfig",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkMetrics",
]


@dataclass
class AlgorithmConfig:
    """Configuration for a graph algorithm to benchmark."""

    name: str
    func: str
    params: dict[str, Any] = field(default_factory=dict)
    requires_directed: bool = False
    requires_undirected: bool = False
    requires_weighted: bool = False
    validate_result: str | None = None
    groups: list[str] = field(default_factory=lambda: ["default"])

    def get_func_ref(self):
        module_path, func_name = self.func.rsplit(".", 1)
        try:
            module = __import__(module_path, fromlist=[func_name])
            return getattr(module, func_name)
        except (ImportError, AttributeError):
            logger.exception(
                f"Failed to import function '{self.func}' for algorithm '{self.name}'"
            )
            return None

    def get_validate_ref(self):
        if self.validate_result:
            mod_path, val_func = self.validate_result.rsplit(".", 1)
            try:
                module = __import__(mod_path, fromlist=[val_func])
                return getattr(module, val_func)
            except (ImportError, AttributeError):
                logger.exception(
                    f"Failed to import validation function '{self.validate_result}' "
                    f"for algorithm '{self.name}'"
                )
                return None
        else:
            return None

    def get_callable(self, backend_name: str) -> Any:
        """Retrieve a callable suitable for the given backend."""
        func = self.get_func_ref()
        if func is None:
            raise ImportError(
                f"Function '{self.func}' could not be imported for algorithm "
                f"'{self.name}'"
            )
        if backend_name != "networkx":
            return partial(func, backend=backend_name)
        return func


@dataclass
class DatasetConfig:
    name: str
    source: str
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] | None = field(default=None)


@dataclass
class BenchmarkConfig:
    """Complete benchmark suite configuration."""

    algorithms: list[AlgorithmConfig]
    datasets: list[DatasetConfig]
    machine_info: dict[str, Any] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("~/results"))
    env_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
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

        env_data = data.get("environ") or {}

        algorithms = [AlgorithmConfig(**algo_data) for algo_data in algorithms_data]
        datasets = [DatasetConfig(**ds_data) for ds_data in datasets_data]

        return cls(
            algorithms=algorithms,
            datasets=datasets,
            machine_info=data.get("machine_info", {}),
            output_dir=Path(data.get("output_dir", "~/results")),
            env_data=env_data,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "algorithms": [
                {k: v for k, v in algo.__dict__.items() if not k.endswith("_ref")}
                for algo in self.algorithms
            ],
            "datasets": [dict(ds.__dict__.items()) for ds in self.datasets],
            "machine_info": self.machine_info,
            "output_dir": str(self.output_dir),
        }

        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False)


@dataclass
class BenchmarkResult:
    """Container for benchmark execution results."""

    algorithm: str
    dataset: str
    execution_time: float
    execution_time_with_preloading: float
    memory_used: float
    num_nodes: int
    num_edges: int
    is_directed: bool
    is_weighted: bool
    backend: str
    num_thread: int
    date: int
    metadata: dict[str, Any]

    validation: str = "unknown"
    validation_message: str = ""
    error: str | None = None


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""

    execution_time: float
    memory_used: float
