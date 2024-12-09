"""Core benchmark functionality and result handling."""

import logging
import os
import time
import traceback
import warnings
from functools import partial
from importlib import import_module
from typing import Any

import networkx as nx

from nxbench.benchmarks.config import AlgorithmConfig
from nxbench.benchmarks.utils import (
    get_available_backends,
    get_benchmark_config,
    is_graphblas_available,
    is_nx_cugraph_available,
    is_nx_parallel_available,
    memory_tracker,
)
from nxbench.data.loader import BenchmarkDataManager
from nxbench.validation.registry import BenchmarkValidator

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


__all__ = [
    "generate_benchmark_methods",
    "GraphBenchmark",
    "get_algorithm_function",
    "process_algorithm_params",
]


def generate_benchmark_methods(cls):
    """Generate benchmark methods dynamically for each combination of algorithm,
    backend, and number of threads without redundant executions.
    """
    config = cls.config
    algorithms = config.algorithms
    datasets = [ds.name for ds in config.datasets]
    available_backends = get_available_backends()
    matrix_config = config.matrix
    if matrix_config:
        backends = [
            backend
            for backend in matrix_config.get("backend", ["networkx"])
            if backend in available_backends
        ]
        num_thread_values = [int(v) for v in matrix_config.get("num_threads", ["1"])]
    else:
        backends = ["networkx"]
        num_thread_values = ["1"]

    def make_benchmark_method(algo_config, dataset_name, backend, num_thread):
        """Create a unique benchmark method for the given algorithm, dataset, backend,
        and number of threads combination.
        """
        algo_name = algo_config.name
        orig_dataset_name = dataset_name
        method_name = f"track_{algo_name}_{orig_dataset_name}_{backend}_{num_thread}"

        def track_method(self):
            """Run benchmark and return metrics for the unique combination."""
            logger.debug(
                f"Starting track_method for {method_name} with backend={backend}, "
                f"threads={num_thread}, dataset={orig_dataset_name}"
            )
            metrics = self.do_benchmark(
                algo_config, orig_dataset_name, backend, num_thread
            )
            logger.debug(f"Track {method_name} results: {metrics}")
            return metrics

        track_method.__name__ = method_name
        track_method.unit = "seconds+MB"

        return track_method

    generated_methods = set()
    for algo_config in algorithms:
        for dataset_name in datasets:
            for backend in backends:
                for num_thread in num_thread_values:
                    method_signature = (
                        algo_config.name,
                        dataset_name,
                        backend,
                        num_thread,
                    )
                    if method_signature not in generated_methods:
                        track_method = make_benchmark_method(
                            algo_config, dataset_name, backend, num_thread
                        )
                        setattr(cls, track_method.__name__, track_method)
                        generated_methods.add(method_signature)

    return cls


@generate_benchmark_methods
class GraphBenchmark:
    """Base class for all graph algorithm benchmarks."""

    config = get_benchmark_config()

    def __init__(self):
        self.data_manager = BenchmarkDataManager()
        self.graphs = {}

    def setup_cache(self):
        """Cache graph data for benchmarks."""
        self.graphs = {}

        datasets = [ds.name for ds in self.config.datasets]

        for dataset_name in datasets:
            dataset_config = next(
                (ds for ds in self.config.datasets if ds.name == dataset_name),
                None,
            )
            if dataset_config is None:
                logger.warning(f"Dataset configuration for '{dataset_name}' not found.")
                continue
            try:
                graph, metadata = self.data_manager.load_network_sync(dataset_config)
                self.graphs[dataset_name] = (graph, metadata)
                logger.debug(
                    f"Cached dataset '{dataset_name}' with "
                    f"{graph.number_of_nodes()} nodes"
                )
            except Exception:
                logger.exception(f"Failed to load dataset '{dataset_name}'")

    def setup(self):
        """ASV setup method. Called before any benchmarks are run."""
        logger.debug("ASV setup: Initializing benchmark cache.")
        self.setup_cache()

    def prepare_benchmark(
        self, dataset_name: str, backend: str, num_thread: int = 1
    ) -> Any:
        """Initialize the dataset and backend, returning the converted graph."""
        logger.debug(
            f"Preparing benchmark with dataset={dataset_name}, "
            f"backend={backend}, threads={num_thread}"
        )

        dataset_name = dataset_name.strip("'")
        logger.debug(f"Looking for dataset '{dataset_name}' in cache")

        graph_data = self.graphs.get(dataset_name)
        if graph_data is None:
            logger.error(f"Graph for dataset '{dataset_name}' not found in cache.")
            logger.debug(f"Available datasets: {list(self.graphs.keys())}")
            return None

        original_graph, metadata = graph_data
        logger.debug(
            f"Found graph with {original_graph.number_of_nodes()} nodes, "
            f"{original_graph.number_of_edges()} edges"
        )

        for var_name in [
            "NUM_THREAD",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ]:
            os.environ[var_name] = str(num_thread)

        if backend == "networkx":
            return original_graph

        if "parallel" in backend and is_nx_parallel_available():
            try:
                nxp = import_module("nx_parallel")
            except ImportError:
                logger.exception("nx-parallel backend not available")
                return None
            return nxp.ParallelGraph(original_graph)

        if "cugraph" in backend and is_nx_cugraph_available():
            try:
                cugraph = import_module("nx_cugraph")
            except ImportError:
                logger.exception("cugraph backend not available")
                return None
            try:
                edge_attr = "weight" if nx.is_weighted(original_graph) else None
                return cugraph.from_networkx(original_graph, edge_attrs=edge_attr)
            except Exception:
                logger.exception("Error converting graph to cugraph format")
                return None

        if "graphblas" in backend and is_graphblas_available():
            try:
                ga = import_module("graphblas_algorithms")
            except ImportError:
                logger.exception("graphblas_algorithms backend not available")
                return None
            try:
                return ga.Graph.from_networkx(original_graph)
            except Exception:
                logger.exception("Error converting graph to graphblas format")
                return None
        else:
            logger.error(f"Unsupported backend: {backend}")
            return None

    def do_benchmark(
        self,
        algo_config: AlgorithmConfig,
        dataset_name: str,
        backend: str,
        num_thread: int,
    ) -> dict:
        logger.debug(
            f"Running benchmark for {algo_config.name} on {dataset_name} with "
            f"{backend} using {num_thread} threads"
        )

        converted_graph = self.prepare_benchmark(dataset_name, backend, num_thread)
        if converted_graph is None:
            return {"execution_time": float("nan"), "memory_used": float("nan")}

        try:
            algo_func = get_algorithm_function(algo_config, backend)
            if isinstance(algo_func, partial):
                alg_func_name = algo_func.func.__name__
            else:
                alg_func_name = algo_func.__name__
            logger.debug(f"Got algorithm function: {alg_func_name}")
        except (ImportError, AttributeError):
            logger.exception(f"Function not available for backend {backend}")
            logger.debug(traceback.format_exc())
            self.teardown_specific(backend, num_thread)
            return {"execution_time": float("nan"), "memory_used": float("nan")}

        try:
            pos_args, kwargs = process_algorithm_params(algo_config.params)

            with memory_tracker() as mem:
                start_time = time.perf_counter()
                result = algo_func(converted_graph, *pos_args, **kwargs)
                end_time = time.perf_counter()

            execution_time = end_time - start_time
            current, peak = mem["current"], mem["peak"]

            if not isinstance(result, (float, int)):
                result = dict(result)

            original_graph, _ = self.graphs[dataset_name]
            validator = BenchmarkValidator()
            try:
                validator.validate_result(result, algo_config.name, original_graph)
                logger.debug(
                    f"Validation passed for algorithm '{algo_config.name}' on "
                    f"dataset '{dataset_name}'"
                )
            except Exception:
                logger.warning(f"Validation warning for '{algo_config.name}'")

            metrics = {
                "execution_time": execution_time,
                "memory_used": peak / (1024 * 1024),  # bytes to MB
            }
            logger.debug(f"Benchmark results for {algo_config.name}: {metrics}")
        except Exception:
            logger.exception(f"Error running algorithm '{algo_config.name}'")
            logger.debug(traceback.format_exc())
            metrics = {"execution_time": float("nan"), "memory_used": float("nan")}
        finally:
            self.teardown_specific(backend, num_thread)

        return metrics

    def teardown_specific(self, backend: str, num_thread: int = 1):
        """Reset any backend-specific configurations to avoid state leakage."""
        if "parallel" in backend:
            logger.debug("Tearing down parallel backend configurations.")
            nx.config.backends.parallel.active = False
            nx.config.backends.parallel.n_jobs = 1

            os.environ["NUM_THREAD"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"

    def teardown(self):
        """ASV teardown method. Called after all benchmarks are run."""
        logger.debug("ASV teardown: Cleaning up if necessary.")


def get_algorithm_function(algo_config: AlgorithmConfig, backend_name: str) -> Any:
    """Retrieve the algorithm function for the specified backend.

    Parameters
    ----------
    algo_config : AlgorithmConfig
        Configuration object containing details about the algorithm, including its
        function reference.
    backend_name : str
        The name of the backend for which the algorithm function is being retrieved.

    Returns
    -------
    Any
        The algorithm function or a partially applied function for the specified
        backend.

    Raises
    ------
    ImportError
        If the function reference for the algorithm is not found.
    """
    if algo_config.func_ref is None:
        raise ImportError(
            f"Function '{algo_config.func}' could not be imported for algorithm "
            f"'{algo_config.name}'"
        )
    if backend_name != "networkx":
        return partial(algo_config.func_ref, backend=backend_name)
    return algo_config.func_ref


def process_algorithm_params(
    params: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Process and separate algorithm parameters into positional and keyword arguments.

    Parameters
    ----------
    params : dict[str, Any]
        A dictionary of algorithm parameters, where keys can indicate either positional
        or keyword arguments.

    Returns
    -------
    tuple[list[Any], dict[str, Any]]
        A tuple containing a list of positional arguments and a dictionary of keyword
        arguments.

    Notes
    -----
    Parameters prefixed with an underscore ("_") are treated as positional arguments.
    If a parameter value is a
    dictionary containing a "func" key, the function is imported dynamically.
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
    logger.debug(
        f"Processed algorithm parameters: pos_args={pos_args}, kwargs={kwargs}"
    )
    return pos_args, kwargs
