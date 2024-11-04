"""Core benchmark functionality and result handling."""

import logging
import warnings
import time
from dataclasses import dataclass
from typing import Any, Dict, Union, List, Any, Tuple

import networkx as nx

from nxbench.data.loader import BenchmarkDataManager
from nxbench.config import get_benchmark_config, AlgorithmConfig
from nxbench.validation.registry import BenchmarkValidator

logger = logging.getLogger(__name__)

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
        cls, asv_result: Dict[str, Any], graph: Union[nx.Graph, nx.DiGraph]
    ):
        """Create BenchmarkResult from ASV benchmark output."""
        return cls(
            algorithm=asv_result["name"],
            dataset=graph.graph.get("name", "unknown"),
            execution_time=asv_result["stats"]["mean"],
            memory_used=asv_result.get("memory", 0.0),
            result=asv_result.get("result"),
            result_type=type(asv_result.get("result")).__name__,
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            is_directed=graph.is_directed(),
            is_weighted=nx.is_weighted(graph),
            parameters=asv_result.get("params", {}),
            metadata=asv_result.get("meta", {}),
        )


class GraphBenchmark:
    """Base class for all graph algorithm benchmarks."""

    param_names = ["dataset_name", "backend"]

    @classmethod
    def get_params(cls):
        config = get_benchmark_config()
        datasets = [ds.name for ds in config.datasets]
        backends = ["networkx", "cugraph"]
        return [datasets, backends]

    def __init__(self):
        self.data_manager = BenchmarkDataManager()
        self.graphs = {}
        self.current_graph = None
        self.current_backend = None

    def setup_cache(self):
        """Cache graph data for benchmarks."""
        self.graphs = {}
        for dataset_name in self.get_params()[0]:
            dataset_config = next(
                (
                    ds
                    for ds in get_benchmark_config().datasets
                    if ds.name == dataset_name
                ),
                None,
            )
            if dataset_config is None:
                logger.error(f"Dataset configuration for '{dataset_name}' not found.")
                continue
            try:
                graph, metadata = self.data_manager.load_network_sync(dataset_config)
                self.graphs[dataset_name] = graph
                logger.info(f"Cached dataset '{dataset_name}'.")
            except Exception as e:
                logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        return graphs

    def setup(self, dataset_name: str, backend: str):
        """Setup for each benchmark iteration."""
        if not self.graphs:
            self.graphs = self.setup_cache()
        self.current_graph = self.graphs.get(dataset_name)
        if self.current_graph is None:
            raise ValueError(f"Graph for dataset '{dataset_name}' not found in cache.")
        self.current_backend = backend

        if backend == "cugraph":
            try:
                import cugraph

                self.current_graph = cugraph.from_networkx(self.current_graph)
            except ImportError:
                raise NotImplementedError("cugraph not available")


def run_benchmarks(
    backend: str, collection: str, profile: bool
) -> List[BenchmarkResult]:
    """
    Run benchmarks as per the configurations and return the results.

    Parameters
    ----------
    backend : str
        The backend to benchmark (e.g., 'networkx', 'cugraph'). If 'all', benchmarks all backends.
    collection : str
        The dataset collection to use. If 'all', uses all datasets in the configuration.
    profile : bool
        If True, enables profiling during benchmarking.

    Returns
    -------
    List[BenchmarkResult]
        A list of BenchmarkResult objects containing the benchmark data.
    """
    config = get_benchmark_config()
    results = []

    data_manager = BenchmarkDataManager()
    validator = BenchmarkValidator()

    algorithms = config.algorithms
    datasets = config.datasets

    if collection != "all":
        datasets = [ds for ds in datasets if ds.name == collection]

    for dataset_config in datasets:
        try:
            graph, metadata = data_manager.load_network_sync(dataset_config)
            logger.info(f"Loaded dataset '{dataset_config.name}'")
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_config.name}': {e}")
            continue

        for algo_config in algorithms:
            if algo_config.requires_directed and not graph.is_directed():
                logger.warning(
                    f"Skipping algorithm '{algo_config.name}' for dataset '{dataset_config.name}' (requires directed graph)"
                )
                continue
            if algo_config.requires_undirected and graph.is_directed():
                logger.warning(
                    f"Skipping algorithm '{algo_config.name}' for dataset '{dataset_config.name}' (requires undirected graph)"
                )
                continue
            if algo_config.requires_weighted and not nx.is_weighted(graph):
                logger.warning(
                    f"Skipping algorithm '{algo_config.name}' for dataset '{dataset_config.name}' (requires weighted graph)"
                )
                continue

            backends = [backend] if backend != "all" else ["networkx", "cugraph"]

            for backend_name in backends:
                logger.info(
                    f"Benchmarking algorithm '{algo_config.name}' on dataset '{dataset_config.name}' using backend '{backend_name}'"
                )

                try:
                    backend_graph = prepare_graph_for_backend(graph, backend_name)
                except Exception as e:
                    logger.error(
                        f"Failed to prepare graph for backend '{backend_name}': {e}"
                    )
                    continue

                try:
                    algo_func = get_algorithm_function(algo_config, backend_name)
                except Exception as e:
                    logger.error(
                        f"Failed to retrieve algorithm function for '{algo_config.name}' using backend '{backend_name}': {e}"
                    )
                    continue

                try:
                    exec_time, memory_used, result = run_single_benchmark(
                        algo_func, backend_graph, algo_config.params, profile=profile
                    )
                except Exception as e:
                    logger.error(
                        f"Benchmark failed for algorithm '{algo_config.name}' on dataset '{dataset_config.name}': {e}"
                    )
                    continue

                try:
                    validator.validate_result(
                        result, algo_config.name, graph, raise_errors=True
                    )
                    logger.info(
                        f"Validation passed for algorithm '{algo_config.name}' on dataset '{dataset_config.name}'"
                    )
                except Exception as e:
                    logger.error(
                        f"Validation failed for algorithm '{algo_config.name}' on dataset '{dataset_config.name}': {e}"
                    )

                benchmark_result = BenchmarkResult(
                    algorithm=algo_config.name,
                    dataset=dataset_config.name,
                    execution_time=exec_time,
                    memory_used=memory_used,
                    result=result,
                    result_type=type(result).__name__,
                    num_nodes=graph.number_of_nodes(),
                    num_edges=graph.number_of_edges(),
                    is_directed=graph.is_directed(),
                    is_weighted=nx.is_weighted(graph),
                    parameters=algo_config.params,
                    metadata=metadata,
                )

                results.append(benchmark_result)

    return results


def prepare_graph_for_backend(graph: nx.Graph, backend_name: str) -> Any:
    """
    Prepare the graph for the specified backend.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph to prepare.
    backend_name : str
        The backend for which to prepare the graph.

    Returns
    -------
    Any
        The graph in the format required by the backend.
    """
    if backend_name == "networkx":
        return graph
    elif backend_name == "cugraph":
        try:
            import cugraph
            import cudf

            if nx.is_weighted(graph):
                edge_attr = "weight"
            else:
                edge_attr = None
            cu_graph = cugraph.from_networkx(graph, edge_attrs=edge_attr)
            return cu_graph
        except ImportError:
            raise ImportError("cugraph is not installed or cannot be imported.")
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")


def get_algorithm_function(algo_config: AlgorithmConfig, backend_name: str) -> Any:
    """
    Retrieve the algorithm function for the specified backend.

    Parameters
    ----------
    algo_config : AlgorithmConfig
        The algorithm configuration.
    backend_name : str
        The backend for which to retrieve the function.

    Returns
    -------
    Any
        The algorithm function.

    Raises
    ------
    ImportError
        If the backend module cannot be imported.
    AttributeError
        If the algorithm function is not found in the backend module.
    """
    if backend_name == "networkx":
        return algo_config.func_ref
    elif backend_name == "cugraph":
        import cugraph

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
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")


def run_single_benchmark(
    algo_func: Any, graph: Any, params: Dict[str, Any], profile: bool = False
) -> Tuple[float, float, Any]:
    """
    Run a single benchmark for a given algorithm function and graph.

    Parameters
    ----------
    algo_func : callable
        The algorithm function to benchmark.
    graph : Any
        The graph on which to run the algorithm.
    params : Dict[str, Any]
        Parameters to pass to the algorithm function.
    profile : bool
        If True, enables profiling during execution.

    Returns
    -------
    Tuple[float, float, Any]
        A tuple containing execution time, memory used, and the result.

    Raises
    ------
    Exception
        If the algorithm execution fails.
    """
    import tracemalloc

    if profile:
        tracemalloc.start()

    start_time = time.perf_counter()
    try:
        result = algo_func(graph, **params)
    except Exception as e:
        raise e
    end_time = time.perf_counter()

    if profile:
        current, peak = tracemalloc.get_traced_memory()
        memory_used = peak / (1024 * 1024)
        tracemalloc.stop()
    else:
        memory_used = 0.0

    exec_time = end_time - start_time

    return exec_time, memory_used, result
