"""Core benchmark functionality and result handling."""

import logging
import warnings
import time
import os
import tracemalloc
import json
from pathlib import Path
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

        execution_time = asv_result.get("stats", {}).get("mean", 0.0)

        memory_used = 0.0
        result = asv_result.get("result")
        if isinstance(result, str):
            try:
                result_dict = json.loads(result)
                memory_used = result_dict.get("memory_used_mb", 0.0)
            except json.JSONDecodeError:
                logger.error("Failed to decode memory usage JSON string.")

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

        def benchmark_method(self, dataset_name, backend):
            self.run_benchmark(algo_config, dataset_name, backend)

        benchmark_method.__name__ = f"time_{algo_name}"
        return benchmark_method

    for algo_config in algorithms:
        method = make_benchmark_method(algo_config)
        setattr(cls, method.__name__, method)

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
                logger.error(f"Dataset configuration for '{dataset_name}' not found.")
                continue
            try:
                graph, metadata = self.data_manager.load_network_sync(dataset_config)
                self.graphs[dataset_name] = graph
                logger.info(f"Cached dataset '{dataset_name}'.")
            except Exception as e:
                logger.error(f"Failed to load dataset '{dataset_name}': {e}")

    def setup(self, dataset_name: str, backend: str):
        """Setup for each benchmark iteration."""
        if not self.graphs:
            self.setup_cache()
        self.current_graph = self.graphs.get(dataset_name)
        if self.current_graph is None:
            logger.error(f"Graph for dataset '{dataset_name}' not found in cache.")
            raise NotImplementedError(f"Dataset '{dataset_name}' not available.")
        self.current_backend = backend

        if backend == "cugraph":
            if not is_cugraph_available():
                logger.error("cugraph not available")
                raise NotImplementedError("cugraph not available")
            import cugraph

            if nx.is_weighted(self.current_graph):
                edge_attr = "weight"
            else:
                edge_attr = None
            self.current_graph = cugraph.from_networkx(
                self.current_graph, edge_attrs=edge_attr
            )

    def run_benchmark(
        self, algo_config: AlgorithmConfig, dataset_name: str, backend: str
    ) -> float:
        """Run the benchmark for a given algorithm and return execution time."""
        try:
            self.setup(dataset_name, backend)
        except NotImplementedError:
            logger.warning(
                f"Skipping benchmark for algorithm '{algo_config.name}' on dataset '{dataset_name}' with backend '{backend}'."
            )
            return 0.0

        try:
            algo_func = get_algorithm_function(algo_config, backend)
        except (ImportError, AttributeError) as e:
            logger.error(f"Skipping algorithm '{algo_config.name}' due to error: {e}")
            return 0.0

        try:
            pos_args, kwargs = process_algorithm_params(algo_config.params)
            tracemalloc.start()
            start_time = time.perf_counter()
            result = algo_func(self.current_graph, *pos_args, **kwargs)
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            exec_time = end_time - start_time
            memory_used = peak / (1024 * 1024)  # convert to MB

        except Exception as e:
            logger.error(f"Error running algorithm '{algo_config.name}': {e}")
            return 0.0

        validator = BenchmarkValidator()
        try:
            validator.validate_result(result, algo_config.name, self.current_graph)
            logger.info(
                f"Validation passed for algorithm '{algo_config.name}' on dataset '{dataset_name}' with backend '{backend}'."
            )
        except Exception as e:
            logger.error(f"Validation failed for '{algo_config.name}': {e}")

        return exec_time


def get_algorithm_function(algo_config: AlgorithmConfig, backend_name: str) -> Any:
    """
    Retrieve the algorithm function for the specified backend.
    """
    if backend_name == "networkx":
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
                # Add more mappings as needed
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
    """Process algorithm parameters to separate positional arguments and keyword arguments,
    and resolve any function references.

    Parameters
    ----------
    params : Dict[str, Any]
        The parameters dict from AlgorithmConfig.

    Returns
    -------
    Tuple[List[Any], Dict[str, Any]]
        A tuple containing a list of positional arguments and a dictionary of keyword arguments.
    """
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
    return pos_args, kwargs
