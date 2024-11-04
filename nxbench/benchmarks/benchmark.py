"""Core benchmark functionality and result handling."""

import logging
import warnings
import traceback
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, Union, List, Any, Tuple

import networkx as nx

from nxbench.data.loader import BenchmarkDataManager
from nxbench.config import get_benchmark_config, AlgorithmConfig
from nxbench.validation.registry import BenchmarkValidator

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


__all__ = ["BenchmarkResult", "GraphBenchmark"]


@dataclass
class BenchmarkResult:
    """Container for benchmark execution results."""

    algorithm: str
    dataset: str
    execution_time: float
    memory_used: float
    result: Any
    result_type: str
    num_nodes: int
    num_edges: int
    is_directed: bool
    is_weighted: bool
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

    @classmethod
    def from_asv_result(
        cls, asv_result: Dict[str, Any], graph: Union[nx.Graph, nx.DiGraph, None] = None
    ):
        """Create BenchmarkResult from ASV benchmark output."""
        execution_time = asv_result.get("execution_time", 0.0)
        memory_used = asv_result.get("memory_used", 0.0)

        result = asv_result.get("result")
        if memory_used == 0.0 and isinstance(result, dict):
            memory_used = result.get("memory_used", 0.0)

        if graph is None:
            graph = nx.Graph()

        return cls(
            algorithm=asv_result.get("name", ""),
            dataset=graph.graph.get("name", "unknown"),
            execution_time=execution_time,
            memory_used=memory_used,
            result=result,
            result_type=type(result).__name__,
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            is_directed=graph.is_directed(),
            is_weighted=nx.is_weighted(graph),
            parameters=asv_result.get("params", {}),
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


def is_nx_parallel_available():
    try:
        import nx_parallel

        return True
    except ImportError:
        return False


backends = ["networkx"]

if is_cugraph_available():
    backends.append("cugraph")

if is_nx_parallel_available():
    backends.append("nx_parallel")


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
            if backend == "nx_parallel":
                import nx_parallel
            elif backend == "cugraph":
                if not is_cugraph_available():
                    logger.error("cugraph not available")
                    return False
                import cugraph

                if nx.is_weighted(self.current_graph):
                    edge_attr = "weight"
                else:
                    edge_attr = None
                self.current_graph = cugraph.from_networkx(
                    self.current_graph, edge_attrs=edge_attr
                )
            return True
        except ImportError as e:
            logger.error(f"Backend {backend} not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting up backend {backend}: {e}")
            return False

    def do_benchmark(
        self, algo_config: AlgorithmConfig, dataset_name: str, backend: str
    ) -> dict:
        logger.debug(
            f"Running benchmark for {algo_config.name} on {dataset_name} with {backend}"
        )

        # Setup environment and graph
        if not self.setup(dataset_name, backend):
            return {"execution_time": float("nan"), "memory_used": float("nan")}

        try:
            algo_func = get_algorithm_function(algo_config, backend)
            logger.debug(f"Got algorithm function: {algo_func.__name__}")
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
    if backend_name == "networkx":
        if algo_config.func_ref is None:
            raise ImportError(
                f"Function '{algo_config.func}' could not be imported for algorithm '{algo_config.name}'"
            )
        return algo_config.func_ref
    elif backend_name == "cugraph":
        try:
            import cugraph
        except ImportError:
            raise ImportError("cugraph is not installed.")
        func_name = algo_config.func.split(".")[-1]
        if hasattr(cugraph, func_name):
            return getattr(cugraph, func_name)
        else:
            mapping = {
                "pagerank": cugraph.pagerank,
            }
            if func_name in mapping:
                return mapping[func_name]
            else:
                raise AttributeError(f"Function '{func_name}' not found in cugraph.")
    elif backend_name == "nx_parallel":
        try:
            import nx_parallel
        except ImportError:
            raise ImportError("nx_parallel is not installed.")
        nx_func_name = algo_config.func
        nxp_func_name = nx_func_name.replace("networkx", "nx_parallel")
        module_path, func_name = nxp_func_name.rsplit(".", 1)
        try:
            module = __import__(module_path, fromlist=[func_name])
            func = getattr(module, func_name)
            return func
        except ImportError as e:
            raise ImportError(f"Could not import function '{nxp_func_name}': {e}")
        except AttributeError as e:
            raise AttributeError(
                f"Function '{func_name}' not found in nx_parallel: {e}"
            )
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")


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
