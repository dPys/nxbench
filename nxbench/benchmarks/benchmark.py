"""Core benchmark functionality and result handling."""

import logging
import warnings
import traceback
import time
import tracemalloc
from functools import partial
from dataclasses import dataclass
from typing import Any, Dict, Union, List, Any, Tuple

import networkx as nx

from _nxbench.config import _config as nxbench_config
from nxbench.data.loader import BenchmarkDataManager
from nxbench.config import get_benchmark_config, AlgorithmConfig
from nxbench.validation.registry import BenchmarkValidator

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


__all__ = [
    "BenchmarkResult",
    "GraphBenchmark",
    "BenchmarkMetrics",
    "get_algorithm_function",
    "process_algorithm_params",
]


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
    metadata: Dict[str, Any]

    @classmethod
    def from_asv_result(
        cls, asv_result: Dict[str, Any], graph: Union[nx.Graph, nx.DiGraph, None] = None
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
            metadata={},
        )


def generate_benchmark_methods(cls):
    config = get_benchmark_config()
    algorithms = config.algorithms

    def make_benchmark_method(algo_config):
        algo_name = algo_config.name

        def track_method(self, dataset_name: str, backend: str):
            """Run benchmark and return metrics."""
            metrics = self.do_benchmark(algo_config, dataset_name, backend)
            logger.debug(f"Track {algo_name} results: {metrics}")
            return metrics

        track_method.__name__ = f"track_{algo_name}"
        track_method.unit = "seconds+MB"

        return track_method

    for algo_config in algorithms:
        track_method = make_benchmark_method(algo_config)
        setattr(cls, track_method.__name__, track_method)

    return cls


config = get_benchmark_config()
datasets = [ds.name for ds in config.datasets]


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


backends = ["networkx"]

if is_cugraph_available():
    backends.append("cugraph")

if is_graphblas_available():
    backends.append("graphblas")

if is_nx_parallel_available():
    backends.append("parallel")


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""

    execution_time: float
    memory_used: float


@generate_benchmark_methods
class GraphBenchmark:
    """Base class for all graph algorithm benchmarks."""

    param_names = ["dataset_name", "backend"]
    params = [datasets, backends]

    def __init__(self):
        self.data_manager = BenchmarkDataManager()
        self.graphs = {}
        self.current_graph = None
        self.current_backend = None

    def setup_cache(self):
        """Cache graph data for benchmarks."""
        self.graphs = {}
        for dataset_name in self.params[0]:
            dataset_config = next(
                (
                    ds
                    for ds in get_benchmark_config().datasets
                    if ds.name == dataset_name
                ),
                None,
            )
            if dataset_config is None:
                logger.warning(f"Dataset configuration for '{dataset_name}' not found.")
                continue
            try:
                graph, metadata = self.data_manager.load_network_sync(dataset_config)
                self.graphs[dataset_name] = (graph, metadata)
                logger.debug(
                    f"Cached dataset '{dataset_name}' with {graph.number_of_nodes()} nodes"
                )
            except Exception as e:
                logger.error(f"Failed to load dataset '{dataset_name}': {e}")

    def setup(self, dataset_name: str, backend: str) -> bool:
        """Setup for each benchmark iteration."""
        if not self.graphs:
            self.setup_cache()

        dataset_name = dataset_name.strip("'")

        graph_data = self.graphs.get(dataset_name)
        if graph_data is None:
            logger.error(f"Graph for dataset '{dataset_name}' not found in cache.")
            return False

        self.current_graph, metadata = graph_data
        self.current_backend = backend

        try:
            if backend == "parallel":
                import nx_parallel

                nx_parallel.set_config(n_jobs=nxbench_config.num_thread)
            elif backend == "cugraph":
                import cugraph

                edge_attr = "weight" if nx.is_weighted(self.current_graph) else None
                self.current_graph = cugraph.from_networkx(
                    self.current_graph, edge_attrs=edge_attr
                )
            elif backend == "graphblas":
                import graphblas_algorithms as ga

                self.current_graph = ga.Graph.from_networkx(self.current_graph)
            else:
                logger.error(f"Unsupported backend: {backend}")
                return False
            return True
        except ImportError as e:
            logger.error(f"Backend '{backend}' import failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting up backend '{backend}': {e}")
            logger.debug(traceback.format_exc())
            return False

    def do_benchmark(
        self, algo_config: AlgorithmConfig, dataset_name: str, backend: str
    ) -> dict:
        logger.debug(
            f"Running benchmark for {algo_config.name} on {dataset_name} with {backend}"
        )

        if not self.setup(dataset_name, backend):
            return {"execution_time": float("nan"), "memory_used": float("nan")}

        try:
            algo_func = get_algorithm_function(algo_config, backend)
            logger.debug(
                f"Got algorithm function: {algo_func.func.__name__ if hasattr(algo_func, 'func') else algo_func.__name__}"
            )
        except (ImportError, AttributeError) as e:
            logger.error(f"Function not available for backend {backend}: {e}")
            logger.debug(traceback.format_exc())
            return {"execution_time": float("nan"), "memory_used": float("nan")}

        try:
            pos_args, kwargs = process_algorithm_params(algo_config.params)
            tracemalloc.start()
            start_time = time.perf_counter()
            result = algo_func(self.current_graph, *pos_args, **kwargs)
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            validator = BenchmarkValidator()
            try:
                validator.validate_result(result, algo_config.name, self.current_graph)
                logger.debug(
                    f"Validation passed for algorithm '{algo_config.name}' on dataset '{dataset_name}'"
                )
            except Exception as e:
                logger.warning(f"Validation warning for '{algo_config.name}': {e}")

            metrics = {
                "execution_time": end_time - start_time,
                "memory_used": peak / (1024 * 1024),  # MB
            }
            logger.debug(f"Benchmark results for {algo_config.name}: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error running algorithm '{algo_config.name}': {str(e)}")
            logger.debug(traceback.format_exc())
            return {"execution_time": float("nan"), "memory_used": float("nan")}


def get_algorithm_function(algo_config: AlgorithmConfig, backend_name: str) -> Any:
    """Retrieve the algorithm function for the specified backend."""
    if algo_config.func_ref is None:
        raise ImportError(
            f"Function '{algo_config.func}' could not be imported for algorithm "
            f"'{algo_config.name}'"
        )

    return partial(algo_config.func_ref, backend=backend_name)


def process_algorithm_params(
    params: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    pos_args = []
    kwargs = {}
    for key, value in params.items():
        if isinstance(value, dict) and "func" in value:
            module_path, func_name = value["func"].rsplit(".", 1)
            module = __import__(module_path, fromlist=[func_name])
            value = getattr(module, func_name)
        if key.startswith("_"):
            pos_args.append(value)
        else:
            kwargs[key] = value
    logger.debug(
        f"Processed algorithm parameters: pos_args={pos_args}, kwargs={kwargs}"
    )
    return pos_args, kwargs
