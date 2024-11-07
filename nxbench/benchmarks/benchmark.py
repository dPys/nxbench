"""Core benchmark functionality and result handling."""

import gc
import logging
import time
import traceback
import tracemalloc
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, ClassVar

import networkx as nx

from _nxbench.config import _config as package_config
from nxbench.benchmarks.config import AlgorithmConfig
from nxbench.benchmarks.utils import (
    get_benchmark_config,
    is_cugraph_available,
    is_graphblas_available,
    is_nx_parallel_available,
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


config = get_benchmark_config()
datasets = [ds.name for ds in config.datasets]


backends = ["networkx"]

if is_cugraph_available():
    backends.append("cugraph")

if is_graphblas_available():
    backends.append("graphblas")

if is_nx_parallel_available():
    backends.append("parallel")


@contextmanager
def memory_tracker():
    tracemalloc.start()
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.reset_peak()


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


@generate_benchmark_methods
class GraphBenchmark:
    """Base class for all graph algorithm benchmarks."""

    param_names: ClassVar[list[str]] = ["dataset_name", "backend"]
    params: ClassVar[list[Any]] = [datasets, backends]

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
                    f"Cached dataset '{dataset_name}' with {graph.number_of_nodes()} "
                    f"nodes"
                )
            except Exception:
                logger.exception(f"Failed to load dataset '{dataset_name}'")

    def setup(self, dataset_name: str, backend: str) -> bool:
        """Initialize the dataset and backend."""
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
            if backend == "networkx":
                pass
            elif backend == "parallel":
                nx.config.backends.parallel.active = True
                nx.config.backends.parallel.n_jobs = package_config.num_thread
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
        except ImportError:
            logger.exception(f"Backend '{backend}' import failed")
            return False
        except Exception:
            logger.exception(f"Error setting up backend '{backend}'")
            logger.debug(traceback.format_exc())
            return False
        else:
            return True

    def do_benchmark(
        self, algo_config: AlgorithmConfig, dataset_name: str, backend: str
    ) -> dict:
        logger.debug(
            f"Running benchmark for {algo_config.name} on {dataset_name} with {backend}"
        )

        converted_graph = self.setup(dataset_name, backend)
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
            self.teardown(backend)
            return {"execution_time": float("nan"), "memory_used": float("nan")}

        try:
            pos_args, kwargs = process_algorithm_params(algo_config.params)

            with memory_tracker() as mem:
                start_time = time.perf_counter()
                result = algo_func(converted_graph, *pos_args, **kwargs)
                end_time = time.perf_counter()

            execution_time = end_time - start_time
            current, peak = mem

            gc.collect()

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
            self.teardown(backend)

        return metrics

    def teardown(self, backend: str):
        """Reset any backend-specific configurations to avoid state carryover."""
        if backend == "parallel":
            nx.config.backends.parallel.active = False
            nx.config.backends.parallel.n_jobs = 1


def get_algorithm_function(algo_config: AlgorithmConfig, backend_name: str) -> Any:
    """Retrieve the algorithm function for the specified backend."""
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
