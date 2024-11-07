"""Core benchmark functionality and result handling."""

import gc
import logging
import os
import time
import traceback
import tracemalloc
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, ClassVar

import networkx as nx

from nxbench.benchmarks.config import AlgorithmConfig
from nxbench.benchmarks.utils import get_available_backends, get_benchmark_config
from nxbench.data.loader import BenchmarkDataManager
from nxbench.validation.registry import BenchmarkValidator

warnings.filterwarnings("ignore")

logger = logging.getLogger("nxbench")


__all__ = [
    "generate_benchmark_methods",
    "memory_tracker",
    "GraphBenchmark",
    "get_algorithm_function",
    "process_algorithm_params",
]


config = get_benchmark_config()
datasets = [ds.name for ds in config.datasets]
available_backends = get_available_backends()
backends = [
    backend
    for backend, version_list in config.matrix.get(
        "req", {"networkx": ["3.4.2"]}
    ).items()
    if backend in available_backends
]
num_thread_values = [
    int(v) for v in config.matrix.get("env_nobuild", {}).get("NUM_THREAD", ["1"])
]


@contextmanager
def memory_tracker():
    tracemalloc.start()
    mem = {}
    try:
        yield mem
    finally:
        mem["current"], mem["peak"] = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.reset_peak()
        gc.collect()


def generate_benchmark_methods(cls):
    config = get_benchmark_config()
    algorithms = config.algorithms

    def make_benchmark_method(algo_config):
        algo_name = algo_config.name

        def track_method(self, dataset_name: str, backend: str, num_thread: int = 1):
            """Run benchmark and return metrics."""
            metrics = self.do_benchmark(algo_config, dataset_name, backend, num_thread)
            logger.debug(f"Track {algo_name} results: {metrics}")
            return metrics

        track_method.__name__ = f"track_{algo_name}"
        track_method.unit = "seconds+MB"

        return track_method

    for algo_config in algorithms:
        track_method = make_benchmark_method(algo_config)
        setattr(cls, track_method.__name__, track_method)

    return cls


@generate_benchmark_methods
class GraphBenchmark:
    """Base class for all graph algorithm benchmarks."""

    param_names: ClassVar[list[str]] = ["dataset_name", "backend", "num_thread"]
    params: ClassVar[list[Any]] = [datasets, backends, num_thread_values]

    def __init__(self):
        self.data_manager = BenchmarkDataManager()
        self.graphs = {}

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
                    f"Cached dataset '{dataset_name}' with "
                    f"{graph.number_of_nodes()} nodes"
                )
            except Exception:
                logger.exception(f"Failed to load dataset '{dataset_name}'")

    def setup(self, dataset_name: str, backend: str, num_thread: int = 1) -> Any:
        """Initialize the dataset and backend, returning the converted graph."""
        if not self.graphs:
            self.setup_cache()

        dataset_name = dataset_name.strip("'")

        graph_data = self.graphs.get(dataset_name)
        if graph_data is None:
            logger.error(f"Graph for dataset '{dataset_name}' not found in cache.")
            return None

        original_graph, metadata = graph_data

        try:
            if backend == "networkx":
                converted_graph = original_graph
            elif "parallel" in backend:
                os.environ["NUM_THREAD"] = str(num_thread)
                os.environ["OMP_NUM_THREADS"] = str(num_thread)
                os.environ["MKL_NUM_THREADS"] = str(num_thread)
                os.environ["OPENBLAS_NUM_THREADS"] = str(num_thread)

                nx.config.backends.parallel.active = True
                nx.config.backends.parallel.n_jobs = num_thread
                converted_graph = original_graph
            elif "cugraph" in backend:
                import cugraph

                edge_attr = "weight" if nx.is_weighted(original_graph) else None
                converted_graph = cugraph.from_networkx(
                    original_graph, edge_attrs=edge_attr
                )
            elif "graphblas" in backend:
                import graphblas_algorithms as ga

                converted_graph = ga.Graph.from_networkx(original_graph)
            else:
                logger.error(f"Unsupported backend: {backend}")
                return None
        except ImportError:
            logger.exception(f"Backend '{backend}' import failed")
            return None
        except Exception:
            logger.exception(f"Error setting up backend '{backend}'")
            logger.debug(traceback.format_exc())
            return None
        else:
            return converted_graph

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

        converted_graph = self.setup(dataset_name, backend, num_thread)
        if converted_graph is None:
            return {"execution_time": float("nan"), "memory_used": float("nan")}

        try:
            algo_func = get_algorithm_function(algo_config, backend)
            alg_func_name = (
                algo_func.func.__name__
                if hasattr(algo_func, "func")
                else algo_func.__name__
            )
            logger.debug(f"Got algorithm function: {alg_func_name}")
        except (ImportError, AttributeError):
            logger.exception(f"Function not available for backend {backend}")
            logger.debug(traceback.format_exc())
            self.teardown(dataset_name, backend, num_thread)
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
            self.teardown(dataset_name, backend, num_thread)

        return metrics

    def teardown(self, dataset_name: str, backend: str, num_thread: int = 1):
        """Reset any backend-specific configurations to avoid state leakage."""
        if backend == "parallel":
            nx.config.backends.parallel.active = False
            nx.config.backends.parallel.n_jobs = 1

            # reset env vars
            os.environ["NUM_THREAD"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"


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
