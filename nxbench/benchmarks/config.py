"""Benchmark configuration handling."""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
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
    min_rounds: int = 3
    warmup: bool = True
    warmup_iterations: int = 1

    def __post_init__(self):
        """Validate and resolve the function reference."""
        module_path, func_name = self.func.rsplit(".", 1)
        try:
            module = __import__(module_path, fromlist=[func_name])
            self.func_ref = getattr(module, func_name)
        except (ImportError, AttributeError):
            logger.exception(
                f"Failed to import function '{self.func}' for algorithm '{self.name}'"
            )
            self.func_ref = None

        if self.validate_result:
            mod_path, val_func = self.validate_result.rsplit(".", 1)
            try:
                module = __import__(mod_path, fromlist=[val_func])
                self.validate_ref = getattr(module, val_func)
            except (ImportError, AttributeError):
                logger.exception(
                    f"Failed to import validation function '{self.validate_result}' "
                    f"for algorithm '{self.name}'"
                )
                self.validate_ref = None
        else:
            self.validate_ref = None


@dataclass
class DatasetConfig:
    name: str
    source: str
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] | None = field(default=None)


@dataclass
class MachineInfo:
    """Container for machine information."""

    arch: str
    cpu: str
    machine: str
    num_cpu: str
    os: str
    ram: str
    version: int


@dataclass
class BenchmarkConfig:
    """Complete benchmark suite configuration."""

    algorithms: list[AlgorithmConfig]
    datasets: list[DatasetConfig]
    matrix: dict[str, Any]
    machine_info: dict[str, Any] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("~/results"))
    env_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
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
        matrix_data = data.get("matrix") or {}

        if not isinstance(algorithms_data, list):
            logger.error(f"'algorithms' should be a list in the config file: {path}")
            algorithms_data = []

        if not isinstance(datasets_data, list):
            logger.error(f"'datasets' should be a list in the config file: {path}")
            datasets_data = []

        if not isinstance(matrix_data, dict):
            logger.error(f"'matrix' should be a dict in the config file: {path}")
            matrix_data = {}

        env_data = data.get("env_config") or {}

        algorithms = [AlgorithmConfig(**algo_data) for algo_data in algorithms_data]

        datasets = [DatasetConfig(**ds_data) for ds_data in datasets_data]

        return cls(
            algorithms=algorithms,
            datasets=datasets,
            matrix=matrix_data,
            machine_info=data.get("machine_info", {}),
            output_dir=Path(data.get("output_dir", "~/results")),
            env_data=env_data,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Parameters
        ----------
        path : str or Path
            Output path for YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "algorithms": [
                {k: v for k, v in algo.__dict__.items() if not k.endswith("_ref")}
                for algo in self.algorithms
            ],
            "datasets": [dict(ds.__dict__.items()) for ds in self.datasets],
            "matrix": self.matrix,
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
    memory_used: float
    num_nodes: int
    num_edges: int
    is_directed: bool
    is_weighted: bool
    backend: str
    num_thread: int
    commit_hash: str
    date: int
    metadata: dict[str, Any]

    @classmethod
    def from_asv_result(
        cls,
        asv_result: dict[str, Any],
        graph: nx.Graph | nx.DiGraph | None = None,
        num_thread: int = 1,
        commit_hash: str = "unknown",
        date: int = 0,
    ):
        """Create BenchmarkResult from ASV benchmark output."""
        execution_time = asv_result.get("execution_time", 0.0)
        memory_used = asv_result.get("memory_used", 0.0)
        dataset = asv_result.get("dataset", "Unknown")
        backend = asv_result.get("backend", "Unknown")
        algorithm = asv_result.get("algorithm", "Unknown")

        logger.debug(f"execution_time: {execution_time}, type: {type(execution_time)}")
        logger.debug(f"memory_used: {memory_used}, type: {type(memory_used)}")

        if not isinstance(execution_time, (int, float)):
            logger.error(f"Non-numeric execution_time: {execution_time}")
            execution_time = float("nan")
        if not isinstance(memory_used, (int, float)):
            logger.error(f"Non-numeric memory_used: {memory_used}")
            memory_used = float("nan")

        if graph is None:
            graph = nx.Graph()
            graph.graph["name"] = dataset

        return cls(
            algorithm=algorithm,
            dataset=dataset,
            execution_time=execution_time,
            memory_used=memory_used,
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            is_directed=graph.is_directed(),
            is_weighted=nx.is_weighted(graph),
            backend=backend,
            num_thread=num_thread,
            commit_hash=commit_hash,
            date=date,
            metadata={},
        )


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""

    execution_time: float
    memory_used: float
