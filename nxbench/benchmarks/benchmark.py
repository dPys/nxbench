import asyncio
import json
import logging
import os
import time
import uuid
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Any

import networkx as nx
from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner
from prefect_dask.task_runners import DaskTaskRunner

from nxbench.benchmarks.config import AlgorithmConfig, DatasetConfig
from nxbench.benchmarks.utils import (
    get_available_backends,
    get_benchmark_config,
    get_machine_info,
    memory_tracker,
)
from nxbench.data.loader import BenchmarkDataManager
from nxbench.validation.registry import BenchmarkValidator

logger = logging.getLogger("nxbench")

os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = (
    "postgresql+asyncpg://prefect_user:pass@localhost:5432/prefect_db"
)
os.environ["PREFECT_ORION_DATABASE_CONNECTION_POOL_SIZE"] = "5"
os.environ["PREFECT_ORION_DATABASE_CONNECTION_MAX_OVERFLOW"] = "10"
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"
os.environ["PREFECT_ORION_API_ENABLE_TASK_RUN_DATA_PERSISTENCE"] = "false"

MAX_WORKERS = 4
run_uuid = uuid.uuid4().hex


def load_config() -> dict[str, Any]:
    """Load benchmark configuration dynamically."""
    config = get_benchmark_config()
    return {
        "algorithms": config.algorithms,
        "datasets": config.datasets,
        "env_data": config.env_data,
    }


def setup_cache(
    datasets: list[DatasetConfig],
) -> dict[str, tuple[nx.Graph, dict[str, Any]]]:
    """Load and cache datasets to avoid redundant loading."""
    data_manager = BenchmarkDataManager()
    graphs = {}
    for dataset_config in datasets:
        dataset_name = dataset_config.name
        try:
            graph, metadata = data_manager.load_network_sync(dataset_config)
            graphs[dataset_name] = (graph, metadata)
            logger.debug(
                f"Cached dataset '{dataset_name}' with {graph.number_of_nodes()} "
                f"nodes and {graph.number_of_edges()} edges."
            )
        except Exception:
            logger.exception(f"Failed to load dataset '{dataset_name}'")
    return graphs


@task(name="configure_backend", cache_key_fn=None, persist_result=False)
def configure_backend(original_graph: nx.Graph, backend: str, num_thread: int) -> Any:
    """Prepare the backend for execution."""
    if backend == "networkx":
        graph = original_graph
    elif "parallel" in backend:
        try:
            nxp = import_module("nx_parallel")
            graph = nxp.ParallelGraph(original_graph)
            logger.debug("Configured nx_parallel backend.")
        except ImportError:
            logger.exception("nx-parallel backend not available.")
            raise
    elif "cugraph" in backend:
        try:
            cugraph = import_module("nx_cugraph")
            edge_attr = "weight" if nx.is_weighted(original_graph) else None
            graph = cugraph.from_networkx(original_graph, edge_attrs=edge_attr)
            logger.debug("Configured cugraph backend.")
        except ImportError:
            logger.exception("cugraph backend not available.")
            raise
        except Exception:
            logger.exception("Error converting graph to cugraph format")
            raise
    elif "graphblas" in backend:
        try:
            gb = import_module("graphblas")
            ga = import_module("graphblas_algorithms")
            gb.ss.config["nthreads"] = num_thread
            graph = ga.Graph.from_networkx(original_graph)
            logger.debug("Configured graphblas backend.")
        except ImportError:
            logger.exception("graphblas_algorithms backend not available.")
            raise
        except Exception:
            logger.exception("Error converting graph to graphblas format")
            raise
    else:
        logger.error(f"Unsupported backend: {backend}")
        raise ValueError(f"Unsupported backend: {backend}")

    return graph


@task(name="run_algorithm", cache_key_fn=None, persist_result=False)
def run_algorithm(
    graph: Any, algo_config: AlgorithmConfig, num_thread: int
) -> tuple[Any, float, int, str | None]:
    """Run the algorithm on the configured backend."""
    logger = get_run_logger()
    algo_func = algo_config.get_func_ref()
    if algo_func is None:
        logger.error(f"Function '{algo_config.func}' could not be imported.")
        return None, 0.0, 0, f"Function '{algo_config.func}' could not be imported."

    error = None
    try:
        original_env = {}
        vars_to_set = [
            "NUM_THREAD",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ]
        for var_name in vars_to_set:
            original_env[var_name] = os.environ.get(var_name)
            os.environ[var_name] = str(num_thread)

        with memory_tracker() as mem:
            start_time = time.perf_counter()
            result = algo_func(graph)
            end_time = time.perf_counter()
        execution_time = end_time - start_time
        peak_memory = mem["peak"]
        logger.debug(f"Algorithm '{algo_config.name}' executed successfully.")
    except Exception as e:
        logger.exception("Algorithm run failed")
        execution_time = time.perf_counter() - start_time
        peak_memory = mem.get("peak", 0)
        result = None
        error = str(e)
    finally:
        for var_name in vars_to_set:
            if original_env[var_name] is None:
                del os.environ[var_name]
            else:
                os.environ[var_name] = original_env[var_name]

    return result, execution_time, peak_memory, error


@task(name="validate_results", cache_key_fn=None, persist_result=False)
def validate_results(
    result: Any, algo_config: AlgorithmConfig, graph: Any
) -> tuple[str, str]:
    logger = get_run_logger()
    validator = BenchmarkValidator()
    try:
        validator.validate_result(result, algo_config.name, graph)
        logger.debug(f"Validation passed for algorithm '{algo_config.name}'.")
    except Exception as e:
        logger.warning(f"Validation warning for '{algo_config.name}'")
        return "warning", str(e)
    return "passed", ""


@task(name="collect_metrics", cache_key_fn=None, persist_result=False)
def collect_metrics(
    execution_time: float,
    peak_memory: int,
    graph: Any,
    algo_config: AlgorithmConfig,
    backend: str,
    dataset_name: str,
    num_thread: int,
    validation_status: str,
    validation_message: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Collect and format metrics for the benchmark run."""
    logger = get_run_logger()

    if not isinstance(graph, nx.Graph) and hasattr(graph, "to_networkx"):
        graph = graph.to_networkx()

    if error:
        metrics = {
            "execution_time": float("nan"),
            "memory_used": float("nan"),
            "num_nodes": (graph.number_of_nodes()),
            "num_edges": (graph.number_of_edges()),
            "algorithm": algo_config.name,
            "backend": backend,
            "dataset": dataset_name,
            "num_thread": num_thread,
            "error": error,
            "validation": validation_status,
            "validation_message": validation_message,
        }
        metrics.update(get_machine_info())
        logger.error(
            f"Benchmark failed for algorithm '{algo_config.name}' on dataset '"
            f"{dataset_name}': {error}"
        )
    else:
        metrics = {
            "execution_time": execution_time,
            "memory_used": peak_memory / (1024 * 1024),  # convert to MB
            "num_nodes": (graph.number_of_nodes()),
            "num_edges": (graph.number_of_edges()),
            "algorithm": algo_config.name,
            "backend": backend,
            "dataset": dataset_name,
            "num_thread": num_thread,
            "validation": validation_status,
            "validation_message": validation_message,
        }
        metrics.update(get_machine_info())
        logger.info(
            f"Benchmark completed for algorithm '{algo_config.name}' on dataset '"
            f"{dataset_name}'."
        )

    return metrics


@task(name="teardown_specific", cache_key_fn=None, persist_result=False)
def teardown_specific(backend: str):
    logger = get_run_logger()
    if "parallel" in backend:
        logger.debug("Tearing down parallel backend configurations.")
        nx.config.backends.parallel.active = False
        nx.config.backends.parallel.n_jobs = 1

    if "cugraph" in backend:
        os.environ["NX_CUGRAPH_AUTOCONFIG"] = "False"
    logger.debug(f"Reset environment variables for backend '{backend}'.")


async def run_single_benchmark(
    backend: str,
    num_thread: int,
    algo_config: AlgorithmConfig,
    dataset_config: DatasetConfig,
    original_graph: nx.Graph,
) -> dict[str, Any] | None:
    """Benchmark flow for a single combination of parameters."""
    logger = get_run_logger()
    logger.info(
        f"Running benchmark for dataset '{dataset_config.name}' with backend "
        f"'{backend}' and {num_thread} threads."
    )

    try:
        graph = configure_backend(original_graph, backend, num_thread)
        result, execution_time, peak_memory, error = run_algorithm(
            graph, algo_config, num_thread
        )
        validation_status, validation_message = validate_results(
            result, algo_config, graph
        )
        metrics = collect_metrics(
            execution_time=execution_time,
            peak_memory=peak_memory,
            graph=graph,
            algo_config=algo_config,
            backend=backend,
            dataset_name=dataset_config.name,
            num_thread=num_thread,
            validation_status=validation_status,
            validation_message=validation_message,
            error=error,
        )
    except Exception as e:
        metrics = collect_metrics(
            execution_time=float("nan"),
            peak_memory=0,
            graph=original_graph,
            algo_config=algo_config,
            backend=backend,
            dataset_name=dataset_config.name,
            num_thread=num_thread,
            validation_status="failed",
            validation_message=str(e),
            error=str(e),
        )
    finally:
        teardown_specific(backend)
        logger.info("Teared down resources after benchmarking.")

    return metrics


@flow(
    name="multiverse_benchmark",
    flow_run_name=f"run_{run_uuid}",
    task_runner=ThreadPoolTaskRunner(max_workers=MAX_WORKERS),
)
async def benchmark_suite(
    algorithms: list[AlgorithmConfig],
    datasets: list[DatasetConfig],
    backends: list[str],
    threads: list[int],
    graphs: dict[str, tuple[nx.Graph, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Run the full suite of benchmarks in parallel using asyncio."""
    logger = get_run_logger()
    logger.info("Starting benchmark suite.")

    def create_benchmark_subflow(name_suffix: str, resource_type: str, num_thread: int):
        @flow(
            name="benchmark_subflow",
            flow_run_name=name_suffix,
            task_runner=DaskTaskRunner(
                cluster_kwargs={
                    "n_workers": 1,
                    "resources": {resource_type: 1},
                    "threads_per_worker": num_thread,
                    "processes": False,
                }
            ),
        )
        async def benchmark_subflow(
            backend: str,
            num_thread: int,
            algo_config: AlgorithmConfig,
            dataset_config: DatasetConfig,
            original_graph: nx.Graph,
        ) -> dict[str, Any] | None:
            return await run_single_benchmark(
                backend,
                num_thread,
                algo_config,
                dataset_config,
                original_graph,
            )

        return benchmark_subflow

    tasks = []
    for backend, num_thread, algo_config, dataset_config in product(
        backends, threads, algorithms, datasets
    ):
        dataset_name = dataset_config.name
        if dataset_name not in graphs:
            logger.warning(f"Dataset '{dataset_name}' not cached. Skipping.")
            continue
        original_graph, _ = graphs[dataset_name]
        resource_type = "GPU" if backend == "cugraph" else "process"
        name_suffix = f"{algo_config.name}_{dataset_name}_{backend}_{num_thread}"

        unique_subflow = create_benchmark_subflow(
            name_suffix, resource_type, num_thread
        )

        tasks.append(
            unique_subflow(
                backend=backend,
                num_thread=num_thread,
                algo_config=algo_config,
                dataset_config=dataset_config,
                original_graph=original_graph,
            )
        )

    return await asyncio.gather(*tasks)


def main_benchmark(
    results_dir: Path = Path("results"),
):
    """Execute benchmarks using Prefect."""
    final_results = []
    timestamp = str(time.strftime("%Y%m%d%H%M%S"))

    try:
        config = load_config()

        algorithms = config["algorithms"]
        datasets = config["datasets"]
        env_data = config["env_data"]

        available_backends = get_available_backends()

        chosen_backends = [
            backend
            for backend in env_data.get("backend", ["networkx"])
            if backend in available_backends
        ]
        if not chosen_backends:
            logger.error("No valid backends selected. Exiting.")
            return

        num_threads = env_data.get("num_threads", [1])
        if not isinstance(num_threads, list):
            num_threads = [num_threads]
        num_threads = [int(x) for x in num_threads]

        graphs = setup_cache(datasets)

        final_results = asyncio.run(
            benchmark_suite(
                algorithms=algorithms,
                datasets=datasets,
                backends=chosen_backends,
                threads=num_threads,
                graphs=graphs,
            )
        )

    finally:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        out_file = results_dir / f"{run_uuid}_{timestamp}.json"

        with out_file.open("w") as f:
            json.dump(final_results, f, indent=4)

        logger.info(f"Benchmark suite results saved to {out_file}")
